import os
import torch
import trimesh
import threading
import numpy as np
import nvdiffrast.torch as dr
from typing import Union, Optional, List, Tuple
from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera

_NVDR_DEBUG_SYNC = os.environ.get('NVDR_DEBUG_SYNC', '0') == '1'
_NVDR_NO_GLCTX_CACHE = os.environ.get('NVDR_NO_GLCTX_CACHE', '0') == '1'
_NVDR_LOG_MESH = os.environ.get('NVDR_LOG_MESH', '0') == '1'
# 仅在显存预估占用超过该阈值（字节）时才走 chunked 路径。
# 默认 40 GiB —— 绝大多数 mesh 都能直接整体渲染。
_NVDR_CHUNK_MEM_THRESHOLD_BYTES = int(
    os.environ.get('NVDR_CHUNK_MEM_THRESHOLD_BYTES', str(40 * 1024 * 1024 * 1024))
)
# 0 表示按显存预算自动推导 chunk_size；设为正整数则强制覆盖。
_NVDR_CHUNK_SIZE = int(os.environ.get('NVDR_CHUNK_SIZE', '0'))
_NVDR_OVERFLOW_MAX_SPLIT_DEPTH = int(os.environ.get('NVDR_OVERFLOW_MAX_SPLIT_DEPTH', '8'))

# 经验系数：约 0.006 字节 / (face·pixel)。
# 拟合自实测：~5M faces @ 1024×1024 ≈ 30 GiB，~7M faces @ 1024×1024 ≈ 40 GiB。
# 同时用于显存预估和 chunk 大小反推，保持两者一致。
_NVDR_BYTES_PER_FACE_PIXEL = 0.006


def _estimateRasterizeMemoryBytes(num_faces: int, num_vertices: int, height: int, width: int) -> int:
    """粗略估算 nvdiffrast 整体光栅化的显存峰值占用（字节）。

    主要构成（经验估算）：
      - vertices_clip:   V * 4 floats * 4 bytes = 16 * V
      - rast_out + rast_out_db: 2 * H * W * 4 * 4 bytes = 32 * H * W
      - 内部 subtriangle / bin / tile 缓冲：约 F * pixels * 系数
        （nvdiffrast 在大 F + 大分辨率下会产生 O(F * 像素密度) 的中间数据）

    这里只是用于决策是否切换到 chunk 渲染，不要求精确。
    """
    pixels = max(1, int(height) * int(width))
    verts_bytes = 16 * max(0, int(num_vertices))
    out_bytes = 32 * pixels
    intermediate_bytes = int(_NVDR_BYTES_PER_FACE_PIXEL * float(num_faces) * float(pixels))
    return verts_bytes + out_bytes + intermediate_bytes


def _autoChunkSize(
    num_faces: int,
    num_vertices: int,
    height: int,
    width: int,
    budget_bytes: int,
) -> int:
    """根据显存预算反推单个 chunk 最多能塞下多少 faces。

    与 `_estimateRasterizeMemoryBytes` 同构：
        budget >= 16·V + 32·H·W + α·F_chunk·H·W
      ⇒ F_chunk_max = (budget - verts - out) / (α · pixels)

    返回值至少为 1，避免极端情况下出现 chunk_size=0 死循环。
    """
    pixels = max(1, int(height) * int(width))
    verts_bytes = 16 * max(0, int(num_vertices))
    out_bytes = 32 * pixels
    # 留 25% 余量给 z_buffer/face_buffer 等 chunk 路径自身的额外张量
    usable = float(budget_bytes) * 0.75 - float(verts_bytes + out_bytes)
    if usable <= 0:
        return 1

    per_face_bytes = _NVDR_BYTES_PER_FACE_PIXEL * float(pixels)
    if per_face_bytes <= 0:
        return max(1, int(num_faces))

    estimated = int(usable / per_face_bytes)
    return max(1, min(estimated, int(num_faces)))


def _debug_sync(device, tag: str = ''):
    if _NVDR_DEBUG_SYNC:
        torch.cuda.synchronize(device)
        if tag:
            print(f'[DEBUG][NVDiffRastRenderer] sync OK after {tag}')


class NVDiffRastRenderer(object):
    _glctx_cache: dict = {}
    _glctx_lock = threading.Lock()

    def __init__(self) -> None:
        return

    @staticmethod
    def getGlctx(device: str) -> dr.RasterizeCudaContext:
        if _NVDR_NO_GLCTX_CACHE:
            return dr.RasterizeCudaContext(device=device)
        with NVDiffRastRenderer._glctx_lock:
            if device not in NVDiffRastRenderer._glctx_cache:
                NVDiffRastRenderer._glctx_cache[device] = dr.RasterizeCudaContext(device=device)
            return NVDiffRastRenderer._glctx_cache[device]

    @staticmethod
    def resetGlctx(device: Optional[str] = None) -> None:
        """Evict cached RasterizeCudaContext after a CUDA error.

        If *device* is given only that device's context is removed; otherwise
        the entire cache is cleared.  This is a best-effort measure — once the
        CUDA context is truly poisoned even a fresh glctx will not help, but
        it prevents a definitely-bad object from being reused.
        """
        with NVDiffRastRenderer._glctx_lock:
            if device is not None:
                NVDiffRastRenderer._glctx_cache.pop(device, None)
            else:
                NVDiffRastRenderer._glctx_cache.clear()

    @staticmethod
    def isTextureExist(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> bool:
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
            return False

        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            if hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture is not None:
                return True
            if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
                return True

        return False

    @staticmethod
    def _hasVisiblePixels(rast_out: torch.Tensor) -> bool:
        return bool((rast_out[:, :, :, 3] > 0).any().item())

    @staticmethod
    def _sanitizeTextureInputs(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        vertex_count: int,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Validate and extract UV + texture image from *mesh*.

        Returns (use_texture, uvs, tex_img).  When *use_texture* is False the
        caller should fall back to vertex-color rendering.
        """
        if not NVDiffRastRenderer.isTextureExist(mesh):
            return False, None, None

        uvs = getattr(mesh.visual, 'uv', None)
        if uvs is None:
            return False, None, None

        uvs = np.asarray(uvs)
        if uvs.ndim != 2 or uvs.shape[1] < 2:
            print(
                f'[WARN][NVDiffRastRenderer] Invalid UV shape {uvs.shape}, '
                'falling back to vertex colors'
            )
            return False, None, None

        uvs = uvs[:, :2].copy()

        if uvs.shape[0] != vertex_count:
            print(
                f'[WARN][NVDiffRastRenderer] UV count ({uvs.shape[0]}) != '
                f'vertex count ({vertex_count}), falling back to vertex colors'
            )
            return False, None, None

        if not np.isfinite(uvs).all():
            print(
                '[WARN][NVDiffRastRenderer] Non-finite UV values, '
                'falling back to vertex colors'
            )
            return False, None, None

        mat = getattr(mesh.visual, 'material', None)
        tex_img = None
        if mat is not None:
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                tex_img = np.asarray(mat.baseColorTexture)
            elif hasattr(mat, 'image') and mat.image is not None:
                tex_img = np.asarray(mat.image)

        if tex_img is None or tex_img.size == 0:
            return False, None, None

        if tex_img.ndim < 2 or not np.isfinite(tex_img).all():
            print(
                '[WARN][NVDiffRastRenderer] Invalid or non-finite texture, '
                'falling back to vertex colors'
            )
            return False, None, None

        return True, uvs, tex_img

    @staticmethod
    def _computeVertexNormals(
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vertices: [V, 3]
            faces: [F, 3]
        Returns:
            vertex_normals: [V, 3] 已归一化
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)

        faces_flat = faces.reshape(-1)
        face_normals_rep = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        vertex_normals = torch.zeros_like(vertices)
        vertex_normals.index_add_(0, faces_flat, face_normals_rep)

        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1, eps=1e-8)
        return vertex_normals

    @staticmethod
    def sampleRandomLighting(
        num_lights: int = 2,
        ambient_range=(0.2, 0.4),
        diffuse_total_range=(0.7, 1.0),
        space: str = 'world',
        upper_bias: float = 0.35,
        seed: Optional[int] = None,
    ) -> dict:
        """采样一套随机（灰度·多光源）光照参数。

        返回 dict: {'lights': [{'direction': [x, y, z], 'intensity': float}, ...],
                    'ambient': float, 'space': 'world'|'camera'}

        - 灰度：每个方向光只贡献亮度 intensity * max(0, n·l)，不带颜色，保持 albedo 色调。
        - space='world'：方向定义在世界系。配合「每物体采样一次、所有视图共享同一 dict」
          即可获得多视图一致的光照（同一表面点在不同视角亮度一致）。
        - 多个光源的 diffuse 强度之和归一化到 diffuse_total_range，避免过曝；第 0 个为主光
          (key)，分得更大权重，其余为补光 (fill)。
        - upper_bias>0 时把光源方向往「上方」(world +Z，与默认 up_direction=[0,0,1] 一致) 偏，
          避免纯底光的非自然观感。
        - 传 seed 可复现（每物体一套时建议用物体索引/名做 seed）。
        """
        rng = np.random.default_rng(seed)
        n = max(1, int(num_lights))
        ambient = float(rng.uniform(*ambient_range))
        total = float(rng.uniform(*diffuse_total_range))

        raw = rng.uniform(0.0, 1.0, size=n)
        raw[0] += 1.0  # 主光权重更大
        weights = raw / raw.sum() * total

        lights = []
        for i in range(n):
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-8)
            if upper_bias > 0.0:
                d[2] = abs(d[2]) + upper_bias  # 推向上半球
                d = d / (np.linalg.norm(d) + 1e-8)
            lights.append({
                'direction': d.astype(float).tolist(),
                'intensity': float(weights[i]),
            })
        return {'lights': lights, 'ambient': ambient, 'space': space}

    @staticmethod
    def _computeShading(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: 'Camera',
        rast_out: torch.Tensor,
        faces: torch.Tensor,
        vertices: torch.Tensor,
        vertices_tensor: Optional[torch.Tensor]=None,
        light_direction: Union[torch.Tensor, np.ndarray, list]=[1, 1, 1],
        ambient_weight: float = 0.3,
        diffuse_weight: float = 0.7,
        lighting: Optional[dict] = None,
    ) -> torch.Tensor:
        """计算 Lambertian shading 系数（lighting 非空时走多光源灰度，支持 world/camera 空间）。

        Args:
            mesh: 原始网格，用于读取预计算的 vertex_normals。
            camera: 当前相机。
            rast_out: nvdiffrast 光栅化输出 [1, H, W, 4]。
            faces: [F, 3] 面索引。
            vertices: [V, 3] 顶点坐标。
            vertices_tensor: 若非 None，表示顶点来自外部张量，需重新计算法线。
            light_direction: 相机空间中的光照方向。
            ambient_weight: 环境光强度 (0~1)。
            diffuse_weight: 漫反射光强度 (0~1)。

        Returns:
            shading: [1, H, W] 每像素亮度系数。
        """
        if vertices_tensor is not None:
            vertex_normals = NVDiffRastRenderer._computeVertexNormals(vertices, faces)
        else:
            mesh_normals = np.asarray(mesh.vertex_normals)
            if mesh_normals.shape[0] == vertices.shape[0]:
                vertex_normals = toTensor(mesh_normals, torch.float32, camera.device)
            else:
                vertex_normals = NVDiffRastRenderer._computeVertexNormals(vertices, faces)

        normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)
        _debug_sync(camera.device, 'dr.interpolate[normals]')
        normals_world = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

        # --- 多光源·灰度 path（random_lighting / 显式 lighting spec 时启用）---
        if lighting is not None:
            space = lighting.get('space', 'world')
            ambient = float(lighting.get('ambient', 0.3))
            lights = lighting.get('lights', [])
            # world 空间：法线本就在世界系，直接用 -> 多视图一致；
            # camera 空间：转到相机系，光随相机走（与 legacy 一致）。
            normals_used = (torch.matmul(normals_world, camera.R.T)
                            if space == 'camera' else normals_world)
            shading = torch.full(
                normals_used.shape[:-1], ambient,
                dtype=normals_used.dtype, device=camera.device,
            )
            for one in lights:
                d = toTensor(one['direction'], torch.float32, camera.device)
                d = torch.nn.functional.normalize(d, dim=0, eps=1e-8)
                ndotl = torch.clamp(torch.sum(normals_used * d, dim=-1), min=0.0)
                shading = shading + float(one.get('intensity', 1.0)) * ndotl
            return torch.clamp(shading, 0.0, 1.0)

        # --- legacy：单方向光（相机空间）---
        light_dir = toTensor(light_direction, torch.float32, camera.device)
        light_dir = torch.nn.functional.normalize(light_dir, dim=0, eps=1e-8)

        normals_cam = torch.matmul(normals_world, camera.R.T)
        diffuse = torch.clamp(
            torch.sum(normals_cam * light_dir, dim=-1), min=0.0, max=1.0
        )
        shading = ambient_weight + diffuse_weight * diffuse
        return torch.clamp(shading, 0.0, 1.0)

    @staticmethod
    def _applyBackground(
        image: torch.Tensor,
        rast_out: torch.Tensor,
        bg_color: list,
        device: str,
    ) -> torch.Tensor:
        """将背景色应用到 image [1, H, W, C] 的非物体区域。"""
        mask = rast_out[:, :, :, 3:4] > 0  # [1, H, W, 1]
        bg = toTensor(bg_color[:3], torch.float32, device) / 255.0
        # reshape 到 [1, 1, 1, C] 并利用广播
        bg = bg.view(1, 1, 1, -1)
        return torch.where(mask, image, bg.expand_as(image))

    @staticmethod
    def _is_rasterize_overflow_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(kw in msg for kw in (
            'subtriangle count overflow',
            'cuda error: 700',
            'illegal memory access',
        ))

    @staticmethod
    def _iter_rasterize_subchunks(
        glctx: dr.RasterizeCudaContext,
        vertices_clip: torch.Tensor,
        faces: torch.Tensor,
        global_face_ids: torch.Tensor,
        resolution: List[int],
        face_start: int,
        face_end: int,
        split_depth: int = 0,
    ):
        if faces.shape[0] == 0:
            return

        try:
            rast_out, rast_out_db = dr.rasterize(
                glctx, vertices_clip, faces, resolution
            )
            yield faces, global_face_ids, rast_out, rast_out_db
            return
        except RuntimeError as exc:
            if not NVDiffRastRenderer._is_rasterize_overflow_error(exc):
                raise

            max_split_depth = _NVDR_OVERFLOW_MAX_SPLIT_DEPTH
            if faces.shape[0] <= 1 or split_depth >= max_split_depth:
                raise RuntimeError(
                    f'NVDiffRastRenderer chunked rasterize unable to split further '
                    f'(faces=[{face_start},{face_end}), F={faces.shape[0]}, '
                    f'split_depth={split_depth}, max_split_depth={max_split_depth}, '
                    f'resolution={resolution}): {exc}'
                ) from exc

            mid = max(1, faces.shape[0] // 2)
            yield from NVDiffRastRenderer._iter_rasterize_subchunks(
                glctx,
                vertices_clip,
                faces[:mid].contiguous(),
                global_face_ids[:mid].contiguous(),
                resolution,
                face_start,
                face_start + mid,
                split_depth + 1,
            )
            yield from NVDiffRastRenderer._iter_rasterize_subchunks(
                glctx,
                vertices_clip,
                faces[mid:].contiguous(),
                global_face_ids[mid:].contiguous(),
                resolution,
                face_start + mid,
                face_end,
                split_depth + 1,
            )

    @staticmethod
    def _rasterize_faces_chunked(
        glctx: dr.RasterizeCudaContext,
        vertices_clip: torch.Tensor,
        faces: torch.Tensor,
        resolution: List[int],
        chunk_size: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 nvdiffrast 自身的 z/w 深度做 chunk z-buffer 合并。

        第一性原理：nvdiffrast 完整 `dr.rasterize` 内部用整数化的 `z/w`
        作为深度键（FineRaster.inl 中的 `atomicMin(pDepth, depth)`）。
        chunk 路径要保持等价，只能用同一个 `sub_rast[..., 2:3]`
        （即输出的 `z/w`）作合并键，否则换成 camera-space Z、view-space
        depth 等都会因为符号/缩放不同而让“更远的面”覆盖“更近的面”，
        从而出现大面积可见三角丢失。
        """
        height, width = resolution
        rast_out = torch.zeros(
            (1, height, width, 4), dtype=torch.float32, device=device
        )
        rast_out_db = torch.zeros(
            (1, height, width, 4), dtype=torch.float32, device=device
        )
        z_buffer = torch.full(
            (1, height, width, 1), float('inf'), dtype=torch.float32, device=device
        )
        face_buffer = torch.full(
            (1, height, width, 1),
            torch.iinfo(torch.int64).max,
            dtype=torch.int64,
            device=device,
        )
        depth_merge_eps = 1e-7
        chunk_size = max(1, min(int(chunk_size), int(faces.shape[0])))
        successful_chunks = 0

        for start in range(0, faces.shape[0], chunk_size):
            end = min(start + chunk_size, faces.shape[0])
            block_faces = faces[start:end].contiguous()
            block_global_ids = torch.arange(
                start, end, device=device, dtype=torch.int64
            )

            for sub_faces, sub_global_ids, sub_rast, sub_rast_db in (
                NVDiffRastRenderer._iter_rasterize_subchunks(
                    glctx,
                    vertices_clip,
                    block_faces,
                    block_global_ids,
                    resolution,
                    start,
                    end,
                )
            ):
                hit_mask = sub_rast[..., 3:4] > 0
                if not bool(hit_mask.any().item()):
                    continue

                z = sub_rast[..., 2:3]
                z = torch.where(
                    hit_mask,
                    z,
                    torch.full_like(z, float('inf')),
                )
                z = torch.nan_to_num(
                    z,
                    nan=float('inf'),
                    posinf=float('inf'),
                    neginf=float('inf'),
                )

                local_tri_id = sub_rast[..., 3:4].long() - 1
                safe_local_tri_id = local_tri_id.clamp(min=0)
                pix_face_id = sub_global_ids[safe_local_tri_id.squeeze(-1)].unsqueeze(-1)
                pix_face_id = pix_face_id.to(dtype=torch.int64)

                better_depth = z < (z_buffer - depth_merge_eps)
                same_depth = (z - z_buffer).abs() <= depth_merge_eps
                better_id = pix_face_id < face_buffer
                update = hit_mask & (better_depth | (same_depth & better_id))
                if not bool(update.any().item()):
                    continue

                sub_rast_remapped = sub_rast.clone()
                sub_rast_remapped[..., 3:4] = pix_face_id.to(torch.float32) + 1.0

                successful_chunks += 1
                rast_out = torch.where(update.expand_as(rast_out), sub_rast_remapped, rast_out)
                rast_out_db = torch.where(update.expand_as(rast_out_db), sub_rast_db, rast_out_db)
                z_buffer = torch.where(update, z, z_buffer)
                face_buffer = torch.where(update, pix_face_id, face_buffer)

        if successful_chunks == 0:
            raise RuntimeError(
                f'chunked rasterize produced no valid chunks '
                f'(F={faces.shape[0]}, resolution={resolution})'
            )

        return rast_out, rast_out_db

    @staticmethod
    def rasterize(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        vertices_tensor: Optional[torch.Tensor] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> dict:
        """
        执行基础光栅化，返回可复用的光栅化结果字典。

        near / far: 可选裁剪面。None 时用 camera.getWorld2NVDiffRast 的默认值；
        显式传入可避免相机距离/尺度超出默认 [near, far] 时把网格裁掉。

        Returns:
            dict: {vertices, faces, rast_out, rast_out_db, vertices_clip}
        """
        if vertices_tensor is not None:
            vertices = vertices_tensor
        else:
            vertices = toTensor(mesh.vertices, torch.float32, camera.device)

        faces = toTensor(mesh.faces, torch.int32, camera.device)
        vertices = vertices.contiguous()
        faces = faces.contiguous()

        n_verts = vertices.shape[0]
        n_faces = faces.shape[0]
        if n_verts == 0 or n_faces == 0:
            raise ValueError(
                f'rasterize received empty mesh (V={n_verts}, F={n_faces})'
            )

        if faces.max() >= n_verts or faces.min() < 0:
            bad_faces = ((faces < 0) | (faces >= n_verts)).any(dim=1)
            n_bad = bad_faces.sum().item()
            faces = faces[~bad_faces].contiguous()
            print(
                f'[WARN][NVDiffRastRenderer::rasterize] '
                f'Removed {n_bad} faces with out-of-bound indices'
            )
            n_faces = faces.shape[0]
            if n_faces == 0:
                raise ValueError(
                    f'rasterize has no valid faces after OOB removal (V={n_verts})'
                )

        if not torch.isfinite(vertices).all():
            vertices = torch.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)

        vertices_homo = torch.cat([
            vertices,
            torch.ones((n_verts, 1), dtype=torch.float32, device=camera.device)
        ], dim=1)

        mvp_kwargs = {}
        if near is not None:
            mvp_kwargs['near'] = float(near)
        if far is not None:
            mvp_kwargs['far'] = float(far)
        mvp = camera.getWorld2NVDiffRast(**mvp_kwargs)
        vertices_clip = torch.matmul(vertices_homo, mvp.T).unsqueeze(0).contiguous()

        if not torch.isfinite(vertices_clip).all():
            vertices_clip = torch.nan_to_num(
                vertices_clip, nan=0.0, posinf=1e4, neginf=-1e4
            )

        resolution = [camera.height, camera.width]
        if _NVDR_LOG_MESH:
            print(
                f'[INFO][NVDiffRastRenderer::rasterize] '
                f'V={n_verts} F={n_faces} resolution={resolution} device={camera.device}'
            )

        glctx = NVDiffRastRenderer.getGlctx(camera.device)
        estimated_mem_bytes = _estimateRasterizeMemoryBytes(
            n_faces, n_verts, resolution[0], resolution[1]
        )
        use_chunked = estimated_mem_bytes >= _NVDR_CHUNK_MEM_THRESHOLD_BYTES
        chunk_size = _NVDR_CHUNK_SIZE if _NVDR_CHUNK_SIZE > 0 else _autoChunkSize(
            n_faces, n_verts, resolution[0], resolution[1],
            _NVDR_CHUNK_MEM_THRESHOLD_BYTES,
        )

        try:
            if use_chunked:
                rast_out, rast_out_db = NVDiffRastRenderer._rasterize_faces_chunked(
                    glctx=glctx,
                    vertices_clip=vertices_clip,
                    faces=faces,
                    resolution=resolution,
                    chunk_size=chunk_size,
                    device=camera.device,
                )
            else:
                rast_out, rast_out_db = dr.rasterize(
                    glctx, vertices_clip, faces, resolution
                )
        except RuntimeError as exc:
            NVDiffRastRenderer.resetGlctx(camera.device)
            if not NVDiffRastRenderer._is_rasterize_overflow_error(exc):
                raise RuntimeError(
                    f'NVDiffRastRenderer.rasterize failed '
                    f'(V={n_verts}, F={n_faces}, resolution={resolution}, '
                    f'device={camera.device}): {exc}'
                ) from exc

            if use_chunked:
                raise RuntimeError(
                    f'NVDiffRastRenderer chunked rasterize failed '
                    f'(V={n_verts}, F={n_faces}, resolution={resolution}, '
                    f'device={camera.device}): {exc}'
                ) from exc

            glctx = NVDiffRastRenderer.getGlctx(camera.device)
            rast_out, rast_out_db = NVDiffRastRenderer._rasterize_faces_chunked(
                glctx=glctx,
                vertices_clip=vertices_clip,
                faces=faces,
                resolution=resolution,
                chunk_size=chunk_size,
                device=camera.device,
            )

        _debug_sync(camera.device, 'dr.rasterize')

        return {
            'vertices': vertices,
            'faces': faces,
            'rast_out': rast_out,
            'rast_out_db': rast_out_db,
            'vertices_clip': vertices_clip,
        }

    @staticmethod
    def renderMask(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """
        渲染二值mask图

        Returns:
            dict: mask [H,W], rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

        # vertices = rasterize_dict['vertices']
        faces = rasterize_dict['faces']
        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        vertices_clip = rasterize_dict['vertices_clip']

        mask_1ch = (rast_out[:, :, :, 3:4] > 0).float()  # [1, H, W, 1]

        if enable_antialias and NVDiffRastRenderer._hasVisiblePixels(rast_out):
            mask_1ch = dr.antialias(mask_1ch.contiguous(), rast_out, vertices_clip, faces)
            _debug_sync(camera.device, 'dr.antialias[mask]')

        mask_hw = mask_1ch[0, :, :, 0]  # [H, W]

        return {
            'mask': mask_hw,
            'rgb': mask_hw.unsqueeze(-1).expand(-1, -1, 3),
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderVertexColor(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
        paint_color: Optional[list] = None,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
        enable_lighting: bool = True,
        ambient_weight: float = 0.3,
        diffuse_weight: float = 0.7,
        lighting: Optional[dict] = None,
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """
        渲染基于法向的 shading 图

        Returns:
            dict: rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

        vertices = rasterize_dict['vertices']
        faces = rasterize_dict['faces']
        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        vertices_clip = rasterize_dict['vertices_clip']

        if paint_color is not None:
            paint_color_tensor = toTensor(paint_color, torch.float32, camera.device)[:3]
            if paint_color_tensor.max() > 1.0:
                paint_color_tensor = paint_color_tensor / 255.0
            vertex_colors = paint_color_tensor.unsqueeze(0).expand(vertices.shape[0], -1)
        elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vc = np.asarray(mesh.visual.vertex_colors)
            if vc.shape[0] == vertices.shape[0]:
                vertex_colors = toTensor(vc[:, :3], torch.float32, camera.device) / 255.0
            else:
                vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)
        else:
            vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)

        colors_interp, _ = dr.interpolate(vertex_colors.unsqueeze(0).contiguous(), rast_out, faces)
        _debug_sync(camera.device, 'dr.interpolate[colors]')
        image = torch.clamp(colors_interp, 0.0, 1.0)

        if enable_lighting:
            shading = NVDiffRastRenderer._computeShading(
                mesh, camera, rast_out, faces, vertices,
                vertices_tensor, light_direction,
                ambient_weight, diffuse_weight, lighting,
            )
            image = image * shading.unsqueeze(-1)

        image = NVDiffRastRenderer._applyBackground(image, rast_out, bg_color, camera.device)

        if enable_antialias and NVDiffRastRenderer._hasVisiblePixels(rast_out):
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)
            _debug_sync(camera.device, 'dr.antialias[vertexColor]')

        return {
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderMatcap(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        matcap: Union[torch.Tensor, np.ndarray],
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
        flip_x: bool = False,
        flip_y: bool = True,
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """Matcap (material-capture) shading: sample a matcap image by the camera-space
        surface normal. Reproduces the Blender solid-view matcap look (e.g. plaster/clay):
        the whole "studio lighting + matte material" is baked into the matcap sphere image,
        sampled as matcap[(1 - n.y)/2, (n.x + 1)/2].

        Args:
            matcap: [Hm, Wm, 3] image (np.ndarray or tensor), values in 0-1 or 0-255.
            flip_x / flip_y: flip the sampling axes to match the matcap's convention.
        """
        import torch.nn.functional as F

        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)
        vertices = rasterize_dict['vertices']
        faces = rasterize_dict['faces']
        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        vertices_clip = rasterize_dict['vertices_clip']

        if vertices_tensor is not None:
            vn = NVDiffRastRenderer._computeVertexNormals(vertices, faces)
        else:
            mn = np.asarray(mesh.vertex_normals)
            if mn.shape[0] == vertices.shape[0]:
                vn = toTensor(mn, torch.float32, camera.device)
            else:
                vn = NVDiffRastRenderer._computeVertexNormals(vertices, faces)

        normals_interp, _ = dr.interpolate(vn.unsqueeze(0), rast_out, faces)
        normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)
        n_cam = torch.matmul(normals_interp, camera.R.T)            # [1, H, W, 3] camera space

        u = n_cam[..., 0]
        v = n_cam[..., 1]
        if flip_x:
            u = -u
        if flip_y:
            v = -v
        grid = torch.stack([u, v], dim=-1).clamp(-1.0, 1.0)        # [1, H, W, 2] in [-1, 1]

        mat = toTensor(matcap, torch.float32, camera.device)
        if mat.max() > 1.5:
            mat = mat / 255.0
        mat = mat[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, 3, Hm, Wm]
        samp = F.grid_sample(mat, grid, mode='bilinear',
                             align_corners=False, padding_mode='border')  # [1, 3, H, W]
        image = samp.permute(0, 2, 3, 1).contiguous()              # [1, H, W, 3]

        image = NVDiffRastRenderer._applyBackground(image, rast_out, bg_color, camera.device)
        if enable_antialias and NVDiffRastRenderer._hasVisiblePixels(rast_out):
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

        return {
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def _wrapUV(uv: torch.Tensor, wrap_s: str, wrap_t: str) -> torch.Tensor:
        """Apply GL wrap modes to interpolated UV coordinates.

        Args:
            uv: [B, H, W, 2] interpolated UV from dr.interpolate.
            wrap_s: 'repeat' | 'mirrored_repeat' | 'clamp' for U axis.
            wrap_t: 'repeat' | 'mirrored_repeat' | 'clamp' for V axis.

        Returns:
            UV tensor with all values in [0, 1].
        """
        out = uv.clone()
        for ch, mode in enumerate([wrap_s, wrap_t]):
            if mode == 'repeat':
                out[..., ch] = out[..., ch] % 1.0
            elif mode == 'mirrored_repeat':
                t = out[..., ch] % 2.0
                out[..., ch] = torch.where(t > 1.0, 2.0 - t, t)
            else:
                out[..., ch] = out[..., ch].clamp(0.0, 1.0)
        return out

    @staticmethod
    def renderTexture(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
        paint_color: Optional[list] = None,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
        enable_lighting: bool = True,
        ambient_weight: float = 0.3,
        diffuse_weight: float = 0.7,
        lighting: Optional[dict] = None,
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """
        渲染网格纹理颜色，可选光照。无纹理时 fallback 到 renderVertexColor。

        Args:
            enable_lighting: 是否对纹理施加 Lambertian 光照，避免单色纹理看起来像 mask。
            ambient_weight: 环境光强度 (0~1)。
            diffuse_weight: 漫反射光强度 (0~1)。

        Returns:
            dict: rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertex_count = vertices_tensor.shape[0] if vertices_tensor is not None else len(mesh.vertices)
        use_texture, uvs_np, tex_img = NVDiffRastRenderer._sanitizeTextureInputs(mesh, vertex_count)

        if not use_texture:
            return NVDiffRastRenderer.renderVertexColor(
                mesh=mesh,
                camera=camera,
                light_direction=light_direction,
                paint_color=paint_color,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                enable_antialias=enable_antialias,
                enable_lighting=enable_lighting,
                ambient_weight=ambient_weight,
                diffuse_weight=diffuse_weight,
                lighting=lighting,
                rasterize_dict=rasterize_dict,
            )

        if len(tex_img.shape) == 2:
            tex_img = np.stack([tex_img] * 3, axis=-1)
        elif tex_img.shape[-1] == 4:
            tex_img = tex_img[:, :, :3]

        texture = toTensor(tex_img, torch.float32, 'cpu') / 255.0
        texture = texture.flip(0).contiguous()

        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

        vertices = rasterize_dict['vertices']
        faces = rasterize_dict['faces']
        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        vertices_clip = rasterize_dict['vertices_clip']

        uvs_tensor = toTensor(uvs_np, torch.float32, camera.device)
        texture = texture.unsqueeze(0).to(camera.device)

        uv_interp, _ = dr.interpolate(uvs_tensor.unsqueeze(0), rast_out, faces)
        _debug_sync(camera.device, 'dr.interpolate[uv]')

        wrap_s, wrap_t = 'clamp', 'clamp'
        if hasattr(mesh, 'metadata') and 'uv_wrap_mode' in mesh.metadata:
            wrap_s, wrap_t = mesh.metadata['uv_wrap_mode']
        uv_interp = NVDiffRastRenderer._wrapUV(uv_interp, wrap_s, wrap_t)

        image = dr.texture(texture, uv_interp, filter_mode='linear')
        _debug_sync(camera.device, 'dr.texture')

        c = image.shape[-1]
        if c < 3:
            image = torch.cat([image, torch.zeros(*image.shape[:-1], 3 - c, device=image.device)], dim=-1)
        elif c > 3:
            image = image[..., :3]

        if enable_lighting:
            shading = NVDiffRastRenderer._computeShading(
                mesh, camera, rast_out, faces, vertices,
                vertices_tensor, light_direction,
                ambient_weight, diffuse_weight, lighting,
            )
            image = image * shading.unsqueeze(-1)

        image = NVDiffRastRenderer._applyBackground(image, rast_out, bg_color, camera.device)

        if enable_antialias and NVDiffRastRenderer._hasVisiblePixels(rast_out):
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)
            _debug_sync(camera.device, 'dr.antialias[texture]')

        return {
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderDepth(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = False,
        rasterize_dict: Optional[dict] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> dict:
        """
        渲染深度图

        near / far: 可选裁剪面，透传给 rasterize（None 时用相机默认）。

        Returns:
            dict: depth [H,W], rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(
                mesh, camera, vertices_tensor, near=near, far=far,
            )

        vertices = rasterize_dict['vertices']
        faces = rasterize_dict['faces']
        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        vertices_clip = rasterize_dict['vertices_clip']

        vertices_interp, _ = dr.interpolate(vertices.unsqueeze(0), rast_out, faces)
        _debug_sync(camera.device, 'dr.interpolate[depth_verts]')

        R_row2 = camera.R[2, :]
        depth = -(torch.matmul(vertices_interp[0], R_row2) + camera.t[2])  # [H, W]

        mask = rast_out[0, :, :, 3] > 0
        depth = torch.where(mask, depth, torch.zeros_like(depth))

        valid_depth = depth[mask]
        if valid_depth.numel() > 0:
            depth_min = valid_depth.min()
            depth_range = valid_depth.max() - depth_min
            depth_normalized = (depth - depth_min) / (depth_range + 1e-8)
        else:
            depth_normalized = torch.zeros_like(depth)

        depth_normalized = torch.where(mask, depth_normalized, torch.ones_like(depth_normalized))

        # 单通道 antialias 后再扩展到 3 通道
        depth_1ch = depth_normalized.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]

        if enable_antialias and NVDiffRastRenderer._hasVisiblePixels(rast_out):
            depth_1ch = dr.antialias(depth_1ch.contiguous(), rast_out, vertices_clip, faces)
            _debug_sync(camera.device, 'dr.antialias[depth]')

        depth_vis = depth_1ch.expand(-1, -1, -1, 3)  # [1, H, W, 3]
        image = NVDiffRastRenderer._applyBackground(depth_vis, rast_out, bg_color, camera.device)

        return {
            'depth': depth,
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderNormal(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = False,
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """
        渲染面法向图（flat 着色），每个三角面内法线恒定。
        使用 rast_out 中的 triangle_id 直接查表面法线，无需重新光栅化。

        Returns:
            dict: world [H,W,3], camera [H,W,3], rgb_world [H,W,3], rgb_camera [H,W,3],
                  rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

        vertices = rasterize_dict['vertices']
        faces = rasterize_dict['faces']
        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        vertices_clip = rasterize_dict['vertices_clip']

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1, eps=1e-8)

        # rast_out[..., 3] 是 1-based triangle_id，0 表示背景
        tri_id = rast_out[0, :, :, 3].long()  # [H, W]

        # 添加一行零向量作为背景占位，使 0-index 映射到零法线
        face_normals_padded = torch.cat([
            torch.zeros(1, 3, device=face_normals.device, dtype=face_normals.dtype),
            face_normals,
        ], dim=0)

        tri_id = tri_id.clamp(0, face_normals_padded.shape[0] - 1)
        normals_world = face_normals_padded[tri_id]  # [H, W, 3]
        normals_camera = torch.matmul(normals_world, camera.R.T)

        normals_world_mapped = (normals_world * 0.5 + 0.5).unsqueeze(0)  # [1, H, W, 3]
        normals_camera_mapped = (normals_camera * 0.5 + 0.5).unsqueeze(0)

        normals_world_vis = NVDiffRastRenderer._applyBackground(normals_world_mapped, rast_out, bg_color, camera.device)
        normals_camera_vis = NVDiffRastRenderer._applyBackground(normals_camera_mapped, rast_out, bg_color, camera.device)

        if enable_antialias and NVDiffRastRenderer._hasVisiblePixels(rast_out):
            combined = torch.cat([normals_world_vis, normals_camera_vis], dim=-1)
            combined = dr.antialias(combined.contiguous(), rast_out, vertices_clip, faces)
            _debug_sync(camera.device, 'dr.antialias[normal]')
            normals_world_vis = combined[:, :, :, :3]
            normals_camera_vis = combined[:, :, :, 3:]

        return {
            'world': normals_world,
            'camera': normals_camera,
            'rgb_world': normals_world_vis[0],
            'rgb_camera': normals_camera_vis[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderGray(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        ibl: Optional[dict] = None,
        material: Optional[dict] = None,
        ibl_hdr_path: Optional[str] = None,
        pbr_seed: Optional[int] = None,
        exposure: float = 1.2,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """PBR + IBL 灰模渲染：Cook-Torrance GGX under an HDR env with a randomly
        sampled gray material — TEXTURE-INDEPENDENT (the mesh's own texture/UV is
        ignored). Thin wrapper over the atomic shader
        ``camera_control.Method.pbr_ibl.render_pbr_ibl``.

        IBL env + material are prepared once per object; they follow the same
        convention as ``lighting`` in :meth:`render`:
          - 显式传入 ``ibl`` (``prepare_ibl`` 的返回) + ``material`` (``sample_material``
            的返回) 时直接使用 —— 多视图一致请在循环外各准备一次再传进来。
          - 未传则按 ``ibl_hdr_path`` (.hdr/.exr 文件或目录) / ``pbr_seed`` 在本次调用内现建；
            ``ibl_hdr_path`` 亦可省略 —— 此时自动 fallback 到 camera-control 自带的内置 HDR
            资产（无需外部设置）。注意逐相机调用会各采一套材质，多视图将不一致。

        Returns:
            dict: rgb [H,W,3] in [0,1], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        from camera_control.Method.pbr_ibl import (
            prepare_ibl, sample_material, render_pbr_ibl, resolveIblHdrPath,
        )

        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

        if ibl is None:
            # 自包含：未显式给 ibl/ibl_hdr_path 时，自动 fallback 到 camera-control
            # 自带的内置 HDR 资产（resolveIblHdrPath 内部按 __file__ 定位绝对路径）。
            hdr_path = resolveIblHdrPath(ibl_hdr_path)
            ibl = prepare_ibl(hdr_path, camera.device)
        if material is None:
            material = sample_material(pbr_seed, camera.device)

        rgb = render_pbr_ibl(
            mesh, camera, ibl, material,
            exposure=exposure, rasterize_dict=rasterize_dict,
            bg_color=bg_color, vertices_tensor=vertices_tensor,
        )

        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        return {
            'rgb': rgb,
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderTexturedPbrIbl(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        ibl: Optional[dict] = None,
        material: Optional[dict] = None,
        ibl_hdr_path: Optional[str] = None,
        pbr_seed: Optional[int] = None,
        exposure: float = 1.2,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """PBR + IBL，但漫反射 albedo 取自 mesh 自带纹理（texture-DEPENDENT 版 renderGray）。

        光照模型与 :meth:`renderGray` 完全一致（同一 IBL env + 同一 roughness/metallic），
        只把灰材质 base_color 换成逐像素纹理采样值，因此"有纹理 / 无纹理"两路光照风格
        一致、仅 albedo 不同。无纹理时 ``albedo=None`` 自动退回灰材质 base_color（=灰模
        PBR+IBL），保证 'auto' 路径总有合理输出。纹理按 sRGB->linear 解码后作为 linear
        albedo 喂 PBR 着色（与灰材质 base 同处于 linear 空间）。
        """
        from camera_control.Method.pbr_ibl import (
            prepare_ibl, sample_material, render_pbr_ibl, resolveIblHdrPath,
        )

        if ibl is None:
            ibl = prepare_ibl(resolveIblHdrPath(ibl_hdr_path), camera.device)
        if material is None:
            material = sample_material(pbr_seed, camera.device)
        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

        vertex_count = (vertices_tensor.shape[0] if vertices_tensor is not None
                        else len(mesh.vertices))
        use_texture, uvs_np, tex_img = NVDiffRastRenderer._sanitizeTextureInputs(
            mesh, vertex_count)

        albedo = None
        if use_texture:
            if len(tex_img.shape) == 2:
                tex_img = np.stack([tex_img] * 3, axis=-1)
            elif tex_img.shape[-1] == 4:
                tex_img = tex_img[:, :, :3]
            texture = toTensor(tex_img, torch.float32, 'cpu') / 255.0
            texture = texture.flip(0).contiguous().unsqueeze(0).to(camera.device)
            uvs_tensor = toTensor(uvs_np, torch.float32, camera.device)
            faces = rasterize_dict['faces']
            rast_out = rasterize_dict['rast_out']
            uv_interp, _ = dr.interpolate(uvs_tensor.unsqueeze(0), rast_out, faces)
            wrap_s, wrap_t = 'clamp', 'clamp'
            if hasattr(mesh, 'metadata') and 'uv_wrap_mode' in mesh.metadata:
                wrap_s, wrap_t = mesh.metadata['uv_wrap_mode']
            uv_interp = NVDiffRastRenderer._wrapUV(uv_interp, wrap_s, wrap_t)
            tex = dr.texture(texture, uv_interp, filter_mode='linear')[0]    # [H,W,c] in [0,1]
            c = tex.shape[-1]
            if c < 3:
                tex = torch.cat(
                    [tex, torch.zeros(*tex.shape[:-1], 3 - c, device=tex.device)], dim=-1)
            elif c > 3:
                tex = tex[..., :3]
            albedo = tex.clamp(0.0, 1.0) ** 2.2     # sRGB -> linear

        rgb = render_pbr_ibl(
            mesh, camera, ibl, material,
            exposure=exposure, rasterize_dict=rasterize_dict,
            bg_color=bg_color, vertices_tensor=vertices_tensor, albedo=albedo,
        )

        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        return {
            'rgb': rgb,
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def render(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        render_types: List[str]=['mask', 'rgb', 'depth', 'normal'],
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
        paint_color: Optional[list] = None,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
        enable_lighting: bool = True,
        ambient_weight: float = 0.3,
        diffuse_weight: float = 0.7,
        lighting: Optional[dict] = None,
        random_lighting: bool = False,
        light_seed: Optional[int] = None,
        light_kwargs: Optional[dict] = None,
        ibl: Optional[dict] = None,
        material: Optional[dict] = None,
        ibl_hdr_path: Optional[str] = None,
        pbr_seed: Optional[int] = None,
        pbr_exposure: float = 1.2,
        rasterize_dict: Optional[dict] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> dict:
        """
        组合渲染接口，根据 render_types 一次性输出多种结果。

        near / far: 可选裁剪面，透传给底层 rasterize（None 时用相机默认）。
        相机距离/场景尺度超出默认 [near, far] 时，显式传入可避免网格被裁掉。

        随机光照（灰度·多光源）：
          - lighting: 显式光照 spec（见 sampleRandomLighting 返回格式）。给定时直接使用。
          - random_lighting=True 且 lighting=None：本次调用内采样一套。注意——若在多视图循环里
            逐相机调用 render()，每个视图会采到不同的光；要「每物体一套、所有视图共享」，
            请在循环外采样一次（如 sampleRenderData 所做）并把同一 lighting 传进来。
          - light_seed / light_kwargs: 透传给 sampleRandomLighting。

        PBR 灰模（render_types 含 'gray'）：
          - ibl / material: 显式传入预备好的 IBL env (prepare_ibl) 与材质
            (sample_material)。多视图一致请在外部各准备一次再传入（与 lighting 同理）。
          - ibl_hdr_path / pbr_seed: 未传 ibl/material 时据此在本次调用内现建；ibl_hdr_path
            省略时自动 fallback 到 camera-control 自带的内置 HDR 资产（无需外部设置）。
          - pbr_exposure: PBR 曝光，透传给 renderGray。

        render_types 可包含：
          - 'mask'  : 输出 'mask'（以及内部用于合成的光栅结果）
          - 'rgb'   : 使用 renderTexture，输出 'rgb'
          - 'gray'  : 使用 renderGray (PBR+IBL 灰模, 纹理无关)，输出 'gray'
          - 'rgb_pbr': 使用 renderTexturedPbrIbl (PBR+IBL，albedo 取自纹理，无纹理
                      fallback 灰模)，输出 'rgb_pbr'
          - 'depth' : 使用 renderDepth，输出 'depth' 与 'rgb_depth'
          - 'normal': 使用 renderNormal，输出
                      'normal_world', 'normal_camera',
                      'rgb_normal_world', 'rgb_normal_camera'

        无论选择何种渲染类型，最终都额外返回：
          - 'rasterize_output'
          - 'bary_derivs'
        """
        # 先确保有统一的光栅化结果，供所有子渲染函数复用
        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(
                mesh, camera, vertices_tensor, near=near, far=far,
            )

        # 本次调用内采样随机光照（多视图一致请在外部采样并通过 lighting 传入）
        if lighting is None and random_lighting:
            lighting = NVDiffRastRenderer.sampleRandomLighting(
                seed=light_seed, **(light_kwargs or {}))

        results: dict = {}

        # mask 渲染（主要用于几何占据）
        if 'mask' in render_types:
            mask_out = NVDiffRastRenderer.renderMask(
                mesh=mesh,
                camera=camera,
                vertices_tensor=vertices_tensor,
                enable_antialias=enable_antialias,
                rasterize_dict=rasterize_dict,
            )
            results['mask'] = mask_out['mask']

        # 纹理 / 顶点颜色渲染，提供主 rgb 图
        if 'rgb' in render_types:
            tex_out = NVDiffRastRenderer.renderTexture(
                mesh=mesh,
                camera=camera,
                light_direction=light_direction,
                paint_color=paint_color,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                enable_antialias=enable_antialias,
                enable_lighting=enable_lighting,
                ambient_weight=ambient_weight,
                diffuse_weight=diffuse_weight,
                lighting=lighting,
                rasterize_dict=rasterize_dict,
            )
            results['rgb'] = tex_out['rgb']

        # 灰模渲染：PBR + IBL（灰材质，纹理无关），主 rgb 写到 'gray'
        if 'gray' in render_types:
            gray_out = NVDiffRastRenderer.renderGray(
                mesh=mesh,
                camera=camera,
                ibl=ibl,
                material=material,
                ibl_hdr_path=ibl_hdr_path,
                pbr_seed=pbr_seed,
                exposure=pbr_exposure,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                rasterize_dict=rasterize_dict,
            )
            results['gray'] = gray_out['rgb']

        # 纹理 PBR+IBL：与 'gray' 同一套 IBL 光照，diffuse albedo 取自纹理（无纹理时
        # 自动 fallback 到灰材质 = 灰模），主 rgb 写到 'rgb_pbr'。
        if 'rgb_pbr' in render_types:
            pbr_out = NVDiffRastRenderer.renderTexturedPbrIbl(
                mesh=mesh,
                camera=camera,
                ibl=ibl,
                material=material,
                ibl_hdr_path=ibl_hdr_path,
                pbr_seed=pbr_seed,
                exposure=pbr_exposure,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                rasterize_dict=rasterize_dict,
            )
            results['rgb_pbr'] = pbr_out['rgb']

        # 深度渲染，depth + depth 可视化 rgb
        #FIXME: depth should not antialias
        if 'depth' in render_types:
            depth_out = NVDiffRastRenderer.renderDepth(
                mesh=mesh,
                camera=camera,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                enable_antialias=enable_antialias,
                rasterize_dict=rasterize_dict,
            )
            results['depth'] = depth_out['depth']
            results['rgb_depth'] = depth_out['rgb']

        # 法向渲染（世界坐标系 + 相机坐标系）
        #FIXME: normal should not antialias
        if 'normal' in render_types:
            normal_out = NVDiffRastRenderer.renderNormal(
                mesh=mesh,
                camera=camera,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                enable_antialias=enable_antialias,
                rasterize_dict=rasterize_dict,
            )
            results['normal_world'] = normal_out['world']
            results['normal_camera'] = normal_out['camera']
            results['rgb_normal_world'] = normal_out['rgb_world']
            results['rgb_normal_camera'] = normal_out['rgb_camera']

        # 公用光栅化结果
        rast_out = rasterize_dict['rast_out']
        rast_out_db = rasterize_dict['rast_out_db']
        results['rasterize_output'] = rast_out[0]
        results['bary_derivs'] = rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0])

        return results
