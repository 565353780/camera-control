import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Union, Optional

from camera_control.Method.data import toNumpy, toTensor
from camera_control.Module.camera import Camera


class NVDiffRastRenderer(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def renderImage(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
        is_gray: bool = False,
        paint_color: Optional[list] = None,
    ) -> dict:
        """
        使用nvdiffrast渲染三角网格

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            light_direction: 光照方向（世界坐标系），默认为[1, 1, 1]
            is_gray: 如果为True，使用normals渲染；否则尝试使用texture
            paint_color: 可选的颜色列表（长度为3，0-1或0-255范围）

        Returns:
            dict包含:
                - image: [H, W, 3] 渲染的图像 (BGR格式, uint8, 适配OpenCV)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        # 1. 提取基础几何信息
        vertices = toTensor(mesh.vertices, torch.float32, camera.device)  # [V, 3]
        faces = toTensor(mesh.faces, torch.int32, camera.device)  # [F, 3]
        vertex_normals = toTensor(mesh.vertex_normals, torch.float32, camera.device)  # [V, 3]

        # 2. 提取顶点颜色
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = toTensor(mesh.visual.vertex_colors[:, :3], torch.float32, camera.device) / 255.0
        else:
            vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)

        # 3. 处理paint_color（如果指定）
        if paint_color is not None:
            paint_color_tensor = toTensor(paint_color, torch.float32, camera.device)[:3]
            if torch.max(paint_color_tensor) > 1.0:
                paint_color_tensor = paint_color_tensor / 255.0
            vertex_colors = paint_color_tensor.unsqueeze(0).expand(vertices.shape[0], -1)

        # 4. 提取UV和texture（仅在非灰度且未指定paint_color时）
        uvs = None
        texture = None
        use_texture = False

        if not is_gray and paint_color is None:
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                uvs = toTensor(mesh.visual.uv, torch.float32, camera.device)

            # 提取texture
            if uvs is not None:
                tex_img = None
                # 尝试多种纹理源
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    if hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture is not None:
                        tex_img = np.array(mesh.visual.material.baseColorTexture)
                    elif hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
                        tex_img = np.array(mesh.visual.material.image)

                if tex_img is not None:
                    # 转换为RGB格式
                    if len(tex_img.shape) == 2:  # 灰度图
                        tex_img = np.stack([tex_img] * 3, axis=-1)
                    elif tex_img.shape[-1] == 4:  # RGBA
                        tex_img = tex_img[:, :, :3]

                    # 转换为tensor并翻转Y轴（nvdiffrast纹理坐标习惯）
                    texture = toTensor(tex_img, torch.float32, camera.device) / 255.0
                    texture = texture.flip(0).unsqueeze(0)  # [1, H, W, 3]
                    use_texture = True

        # 5. MVP变换和光栅化
        vertices_homo = torch.cat([
            vertices,
            torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=camera.device)
        ], dim=1)  # [V, 4]

        bbox_length = torch.max(vertices, dim=0)[0] - torch.min(vertices, dim=0)[0]
        mvp = camera.getWorld2NVDiffRast(bbox_length)
        vertices_clip = torch.matmul(vertices_homo, mvp.T).unsqueeze(0).contiguous()  # [1, V, 4]

        glctx = dr.RasterizeCudaContext(device=camera.device)
        rast_out, rast_out_db = dr.rasterize(
            glctx,
            vertices_clip,
            faces,
            resolution=[camera.height, camera.width]
        )  # [1, H, W, 4]

        # 6. 计算最终颜色
        if use_texture:
            # 使用纹理渲染（直接渲染纹理颜色，不需要光照）
            uv_interp, _ = dr.interpolate(uvs.unsqueeze(0), rast_out, faces)  # [1, H, W, 2]
            texture_color = dr.texture(texture, uv_interp, filter_mode='linear')  # [1, H, W, 3]
            image = texture_color[0]  # [H, W, 3]
        else:
            # 使用normals和顶点颜色渲染（需要光照）
            # 插值法线
            normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)  # [1, H, W, 3]
            normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)

            # 插值顶点颜色
            colors_interp, _ = dr.interpolate(vertex_colors.unsqueeze(0), rast_out, faces)  # [1, H, W, 3]
            colors_interp = torch.clamp(colors_interp, 0.0, 1.0)

            # 计算光照
            light_dir = toTensor(light_direction, torch.float32, camera.device)
            light_dir = light_dir / (torch.norm(light_dir) + 1e-8)

            R_view = camera.R  # [3, 3]
            normals_cam = torch.matmul(normals_interp, R_view.T)  # [1, H, W, 3]
            light_dir_cam = torch.matmul(light_dir, R_view.T)
            light_dir_cam = light_dir_cam / (torch.norm(light_dir_cam) + 1e-8)

            diffuse = torch.sum(normals_cam * light_dir_cam[None, None, None, :], dim=-1)  # [1, H, W]
            diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
            shading = 0.3 + 0.7 * diffuse  # 环境光 + 漫反射

            # 应用光照
            image = colors_interp[0] * shading.unsqueeze(-1)  # [H, W, 3]

        # 7. 处理背景为白色
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        background = torch.ones_like(image)
        image = torch.where(mask.unsqueeze(-1), image, background)
 
        # 8. 转换为numpy uint8并转换颜色通道（RGB -> BGR，适配OpenCV）
        render_image_np = np.clip(np.rint(toNumpy(image) * 255), 0, 255).astype(np.uint8)
        render_image_np = render_image_np[:, :, ::-1]  # RGB -> BGR

        return {
            'image': render_image_np,  # [H, W, 3] BGR格式
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }
