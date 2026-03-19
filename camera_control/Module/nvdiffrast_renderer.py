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
            bad = (faces < 0) | (faces >= n_verts)
            n_bad = bad.sum().item()
            faces[bad] = 0
            print(
                f'[WARN][NVDiffRastRenderer::rasterize] '
                f'Clamped {n_bad} out-of-bound face indices'
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

        glctx = NVDiffRastRenderer.getGlctx(camera.device)
        rast_out, rast_out_db = dr.rasterize(
            glctx, vertices_clip, faces,
            resolution=[camera.height, camera.width]
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

        colors_interp, _ = dr.interpolate(vertex_colors.unsqueeze(0).contiguous(), rast_out, faces)
        _debug_sync(camera.device, 'dr.interpolate[colors]')
        colors_interp = torch.clamp(colors_interp, 0.0, 1.0)

        light_dir_cam = toTensor(light_direction, torch.float32, camera.device)
        light_dir_cam = torch.nn.functional.normalize(light_dir_cam, dim=0, eps=1e-8)

        normals_cam = torch.matmul(normals_interp, camera.R.T)

        diffuse = torch.clamp(
            torch.sum(normals_cam * light_dir_cam, dim=-1), min=0.0, max=1.0
        )
        shading = 0.3 + 0.7 * diffuse

        image = colors_interp * shading.unsqueeze(-1)

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
        rasterize_dict: Optional[dict] = None,
    ) -> dict:
        """
        渲染网格纹理颜色（无光照）。无纹理时 fallback 到 renderVertexColor。

        仅在无纹理时支持light_direction

        Returns:
            dict: rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertex_count = vertices_tensor.shape[0] if vertices_tensor is not None else len(mesh.vertices)
        use_texture, uvs_np, tex_img = NVDiffRastRenderer._sanitizeTextureInputs(mesh, vertex_count)

        if not use_texture:
            return NVDiffRastRenderer.renderVertexColor(
                mesh=mesh, camera=camera, light_direction=light_direction,
                paint_color=paint_color, bg_color=bg_color,
                vertices_tensor=vertices_tensor, enable_antialias=enable_antialias,
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

        # vertices = rasterize_dict['vertices']
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
        enable_antialias: bool = True,
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
        enable_antialias: bool = True,
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
                rasterize_dict=rasterize_dict,
            )
            results['rgb'] = tex_out['rgb']

        # 深度渲染，depth + depth 可视化 rgb
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
