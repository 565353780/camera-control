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
    ) -> torch.Tensor:
        """计算 Lambertian shading 系数。

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
        normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

        light_dir = toTensor(light_direction, torch.float32, camera.device)
        light_dir = torch.nn.functional.normalize(light_dir, dim=0, eps=1e-8)

        normals_cam = torch.matmul(normals_interp, camera.R.T)
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
    ) -> dict:
        """
        执行基础光栅化，返回可复用的光栅化结果字典。

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

        mvp = camera.getWorld2NVDiffRast()
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
                ambient_weight, diffuse_weight,
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
                ambient_weight, diffuse_weight,
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
    ) -> dict:
        """
        渲染深度图

        Returns:
            dict: depth [H,W], rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        if rasterize_dict is None:
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

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
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """
        组合渲染接口，根据 render_types 一次性输出多种结果。

        render_types 可包含：
          - 'mask'  : 输出 'mask'（以及内部用于合成的光栅结果）
          - 'rgb'   : 使用 renderTexture，输出 'rgb'
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
            rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)

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
                rasterize_dict=rasterize_dict,
            )
            results['rgb'] = tex_out['rgb']

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
