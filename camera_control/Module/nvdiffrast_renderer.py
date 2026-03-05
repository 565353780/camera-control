import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Union, Optional

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera


class NVDiffRastRenderer(object):
    _glctx_cache: dict = {}

    def __init__(self) -> None:
        return

    @staticmethod
    def getGlctx(device: str) -> dr.RasterizeCudaContext:
        if device not in NVDiffRastRenderer._glctx_cache:
            NVDiffRastRenderer._glctx_cache[device] = dr.RasterizeCudaContext(device=device)
        return NVDiffRastRenderer._glctx_cache[device]

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
    def _computeFaceNormalAttrs(
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> tuple:
        """
        构造面法线的 flat 属性数组，使 nvdiffrast 插值后每个三角面内法线恒定。

        Args:
            vertices: [V, 3]
            faces: [F, 3] int32
        Returns:
            flat_normals: [F*3, 3] 每个面的法线重复 3 次
            flat_pos_idx: [F, 3] int32 重新编号的面索引 (0..F*3-1)
            flat_vertices: [F*3, 3] 按面展开的顶点坐标
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1, eps=1e-8)

        flat_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        flat_pos_idx = torch.arange(
            faces.shape[0] * 3, device=faces.device, dtype=torch.int32
        ).reshape(-1, 3)
        flat_vertices = torch.cat([v0, v1, v2], dim=1).reshape(-1, 3)

        return flat_normals, flat_pos_idx, flat_vertices

    @staticmethod
    def _rasterize(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        vertices_tensor: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        执行基础光栅化。

        Returns:
            tuple: (vertices, faces, rast_out, rast_out_db, vertices_clip)
        """
        if vertices_tensor is not None:
            vertices = vertices_tensor
        else:
            vertices = toTensor(mesh.vertices, torch.float32, camera.device)

        faces = toTensor(mesh.faces, torch.int32, camera.device)

        vertices_homo = torch.cat([
            vertices,
            torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=camera.device)
        ], dim=1)

        mvp = camera.getWorld2NVDiffRast()
        vertices_clip = torch.matmul(vertices_homo, mvp.T).unsqueeze(0).contiguous()

        glctx = NVDiffRastRenderer.getGlctx(camera.device)
        rast_out, rast_out_db = dr.rasterize(
            glctx, vertices_clip, faces,
            resolution=[camera.height, camera.width]
        )

        return vertices, faces, rast_out, rast_out_db, vertices_clip

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
    def renderMask(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染二值mask图

        Returns:
            dict: mask [H,W], rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertices, faces, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        mask_1ch = (rast_out[:, :, :, 3:4] > 0).float()  # [1, H, W, 1]

        if enable_antialias:
            mask_1ch = dr.antialias(mask_1ch.contiguous(), rast_out, vertices_clip, faces)

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
    ) -> dict:
        """
        渲染基于法向的 shading 图

        Returns:
            dict: rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertices, faces, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        if paint_color is not None:
            paint_color_tensor = toTensor(paint_color, torch.float32, camera.device)[:3]
            if paint_color_tensor.max() > 1.0:
                paint_color_tensor = paint_color_tensor / 255.0
            vertex_colors = paint_color_tensor.unsqueeze(0).expand(vertices.shape[0], -1)
        elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = toTensor(mesh.visual.vertex_colors[:, :3], torch.float32, camera.device) / 255.0
        else:
            vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)

        if vertices_tensor is not None:
            vertex_normals = NVDiffRastRenderer._computeVertexNormals(vertices, faces)
        else:
            vertex_normals = toTensor(mesh.vertex_normals, torch.float32, camera.device)

        normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)
        normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

        colors_interp, _ = dr.interpolate(vertex_colors.unsqueeze(0).contiguous(), rast_out, faces)
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

        if enable_antialias:
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

        return {
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderTexture(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        paint_color: Optional[list] = None,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染网格纹理颜色（无光照）。无纹理时 fallback 到 renderVertexColor。

        Returns:
            dict: rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        # 先检查纹理再决定是否需要光栅化，避免 fallback 时重复光栅化
        use_texture = False
        texture = None
        uvs = None

        if NVDiffRastRenderer.isTextureExist(mesh):
            uvs_np = mesh.visual.uv
            vertices_count = len(mesh.vertices) if vertices_tensor is None else vertices_tensor.shape[0]
            if uvs_np.shape[0] != vertices_count:
                print(f'[WARN][NVDiffRastRenderer::renderTexture] UV count ({uvs_np.shape[0]}) != vertex count ({vertices_count}), falling back to vertex colors')
            else:
                tex_img = None
                mat = mesh.visual.material
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    tex_img = np.array(mat.baseColorTexture)
                elif hasattr(mat, 'image') and mat.image is not None:
                    tex_img = np.array(mat.image)

                if tex_img is not None:
                    if len(tex_img.shape) == 2:
                        tex_img = np.stack([tex_img] * 3, axis=-1)
                    elif tex_img.shape[-1] == 4:
                        tex_img = tex_img[:, :, :3]

                    texture = toTensor(tex_img, torch.float32, 'cpu') / 255.0
                    texture = texture.flip(0)
                    uvs = uvs_np
                    use_texture = True

        if not use_texture:
            return NVDiffRastRenderer.renderVertexColor(
                mesh=mesh, camera=camera, light_direction=[1, 1, 1],
                paint_color=paint_color, bg_color=bg_color,
                vertices_tensor=vertices_tensor, enable_antialias=enable_antialias,
            )

        vertices, faces, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        uvs_tensor = toTensor(uvs, torch.float32, camera.device)
        texture = texture.unsqueeze(0).to(camera.device)

        uv_interp, _ = dr.interpolate(uvs_tensor.unsqueeze(0), rast_out, faces)
        image = dr.texture(texture, uv_interp, filter_mode='linear')

        c = image.shape[-1]
        if c < 3:
            image = torch.cat([image, torch.zeros(*image.shape[:-1], 3 - c, device=image.device)], dim=-1)
        elif c > 3:
            image = image[..., :3]

        image = NVDiffRastRenderer._applyBackground(image, rast_out, bg_color, camera.device)

        if enable_antialias:
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

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
    ) -> dict:
        """
        渲染深度图

        Returns:
            dict: depth [H,W], rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertices, faces, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        vertices_interp, _ = dr.interpolate(vertices.unsqueeze(0), rast_out, faces)

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

        if enable_antialias:
            depth_1ch = dr.antialias(depth_1ch.contiguous(), rast_out, vertices_clip, faces)

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
    ) -> dict:
        """
        渲染面法向图（flat 着色），每个三角面内法线恒定。

        Returns:
            dict: world [H,W,3], camera [H,W,3], rgb_world [H,W,3], rgb_camera [H,W,3],
                  rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        if vertices_tensor is not None:
            vertices = vertices_tensor
        else:
            vertices = toTensor(mesh.vertices, torch.float32, camera.device)

        faces = toTensor(mesh.faces, torch.int32, camera.device)

        flat_normals, flat_pos_idx, flat_vertices = NVDiffRastRenderer._computeFaceNormalAttrs(vertices, faces)

        vertices_homo = torch.cat([
            flat_vertices,
            torch.ones((flat_vertices.shape[0], 1), dtype=torch.float32, device=camera.device),
        ], dim=1)

        mvp = camera.getWorld2NVDiffRast()
        vertices_clip = torch.matmul(vertices_homo, mvp.T).unsqueeze(0).contiguous()

        glctx = NVDiffRastRenderer.getGlctx(camera.device)
        rast_out, rast_out_db = dr.rasterize(
            glctx, vertices_clip, flat_pos_idx,
            resolution=[camera.height, camera.width],
        )

        normals_interp, _ = dr.interpolate(flat_normals.unsqueeze(0), rast_out, flat_pos_idx)
        normals_world = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

        normals_camera = torch.matmul(normals_world, camera.R.T)

        normals_world_mapped = normals_world * 0.5 + 0.5
        normals_camera_mapped = normals_camera * 0.5 + 0.5

        normals_world_vis = NVDiffRastRenderer._applyBackground(normals_world_mapped, rast_out, bg_color, camera.device)
        normals_camera_vis = NVDiffRastRenderer._applyBackground(normals_camera_mapped, rast_out, bg_color, camera.device)

        if enable_antialias:
            combined = torch.cat([normals_world_vis, normals_camera_vis], dim=-1)
            combined = dr.antialias(combined.contiguous(), rast_out, vertices_clip, flat_pos_idx)
            normals_world_vis = combined[:, :, :, :3]
            normals_camera_vis = combined[:, :, :, 3:]

        return {
            'world': normals_world[0],
            'camera': normals_camera[0],
            'rgb_world': normals_world_vis[0],
            'rgb_camera': normals_camera_vis[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }
