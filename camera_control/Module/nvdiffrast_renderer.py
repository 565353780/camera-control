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
    def isTextureExist(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> bool:
        """
        检查mesh是否包含UV和纹理信息

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象

        Returns:
            bool: 如果mesh包含有效的UV和纹理则返回True，否则返回False
        """
        # 检查是否有UV坐标
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
            return False

        # 检查是否有纹理材质
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            if hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture is not None:
                return True
            elif hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
                return True

        return False

    @staticmethod
    def _rasterize(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera
    ) -> tuple:
        """
        执行基础的光栅化操作（内部辅助方法）

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象

        Returns:
            tuple: (vertices, faces, vertex_normals, rast_out, rast_out_db, glctx)
        """
        # 提取基础几何信息
        vertices = toTensor(mesh.vertices, torch.float32, camera.device)  # [V, 3]
        faces = toTensor(mesh.faces, torch.int32, camera.device)  # [F, 3]
        vertex_normals = toTensor(mesh.vertex_normals, torch.float32, camera.device)  # [V, 3]

        # MVP变换和光栅化
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

        return vertices, faces, vertex_normals, rast_out, rast_out_db, glctx

    @staticmethod
    def renderVertexColor(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
        paint_color: Optional[list] = None,
    ) -> dict:
        """
        渲染基于法向计算的shading图

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            light_direction: 光照方向（世界坐标系），默认为[1, 1, 1]
            paint_color: 可选的颜色列表（长度为3，0-1或0-255范围）

        Returns:
            dict包含:
                - image: [H, W, 3] 渲染的shading图像 (BGR格式, uint8, 适配OpenCV)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx = NVDiffRastRenderer._rasterize(mesh, camera)

        # 提取或设置顶点颜色
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = toTensor(mesh.visual.vertex_colors[:, :3], torch.float32, camera.device) / 255.0
        else:
            vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)

        # 处理paint_color（如果指定）
        if paint_color is not None:
            paint_color_tensor = toTensor(paint_color, torch.float32, camera.device)[:3]
            if torch.max(paint_color_tensor) > 1.0:
                paint_color_tensor = paint_color_tensor / 255.0
            vertex_colors = paint_color_tensor.unsqueeze(0).expand(vertices.shape[0], -1)

        # 插值法线
        normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)  # [1, H, W, 3]
        normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)

        # 插值顶点颜色
        colors_interp, _ = dr.interpolate(vertex_colors.unsqueeze(0).contiguous(), rast_out, faces)  # [1, H, W, 3]
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

        # 处理背景为白色
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        background = torch.ones_like(image)
        image = torch.where(mask.unsqueeze(-1), image, background)

        # 转换为numpy uint8并转换颜色通道（RGB -> BGR，适配OpenCV）
        render_image_np = np.clip(np.rint(toNumpy(image) * 255), 0, 255).astype(np.uint8)
        render_image_np = render_image_np[..., ::-1]  # RGB -> BGR

        return {
            'image': render_image_np[0],  # [H, W, 3] BGR格式
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderTexture(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        paint_color: Optional[list] = None,
    ) -> dict:
        """
        渲染网格本身颜色（包括纹理或顶点颜色，不带光照）

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            paint_color: 可选的颜色列表（长度为3，0-1或0-255范围）

        Returns:
            dict包含:
                - image: [H, W, 3] 渲染的图像 (BGR格式, uint8, 适配OpenCV)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx = NVDiffRastRenderer._rasterize(mesh, camera)

        # 检查是否使用纹理
        use_texture = False
        if NVDiffRastRenderer.isTextureExist(mesh):
            # 提取UV和texture
            uvs = toTensor(mesh.visual.uv, torch.float32, camera.device)

            tex_img = None
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

        if use_texture:
            # 使用纹理渲染
            uv_interp, _ = dr.interpolate(uvs.unsqueeze(0), rast_out, faces)  # [1, H, W, 2]
            image = dr.texture(texture, uv_interp, filter_mode='linear')[0]  # [H, W, 3]
        else:
            # 使用顶点颜色渲染
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vertex_colors = toTensor(mesh.visual.vertex_colors[:, :3], torch.float32, camera.device) / 255.0
            else:
                vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)

            # 处理paint_color（如果指定）
            if paint_color is not None:
                paint_color_tensor = toTensor(paint_color, torch.float32, camera.device)[:3]
                if torch.max(paint_color_tensor) > 1.0:
                    paint_color_tensor = paint_color_tensor / 255.0
                vertex_colors = paint_color_tensor.unsqueeze(0).expand(vertices.shape[0], -1)

            # 插值顶点颜色
            colors_interp, _ = dr.interpolate(vertex_colors.unsqueeze(0).contiguous(), rast_out, faces)  # [1, H, W, 3]
            image = torch.clamp(colors_interp[0], 0.0, 1.0)  # [H, W, 3]

        # 处理背景为白色
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        background = torch.ones_like(image)
        image = torch.where(mask.unsqueeze(-1), image, background)

        # 转换为numpy uint8并转换颜色通道（RGB -> BGR，适配OpenCV）
        render_image_np = np.clip(np.rint(toNumpy(image) * 255), 0, 255).astype(np.uint8)
        render_image_np = render_image_np[..., ::-1]  # RGB -> BGR

        return {
            'image': render_image_np,  # [H, W, 3] BGR格式
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderDepth(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
    ) -> dict:
        """
        渲染深度图

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象

        Returns:
            dict包含:
                - depth: [H, W] 深度图 (float32, 单位：米)
                - depth_normalized: [H, W, 3] 归一化深度图用于可视化 (BGR格式, uint8)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx = NVDiffRastRenderer._rasterize(mesh, camera)

        # 计算世界坐标系下的深度（Z坐标）
        vertices_interp, _ = dr.interpolate(vertices.unsqueeze(0), rast_out, faces)  # [1, H, W, 3]
        
        # 将顶点转换到相机坐标系
        vertices_cam = torch.matmul(vertices_interp[0] - camera.t, camera.R.T)  # [H, W, 3]
        depth = vertices_cam[:, :, 2]  # [H, W] - 相机坐标系下的Z值

        # 处理背景
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        depth = torch.where(mask, depth, torch.zeros_like(depth))

        # 创建归一化的深度图用于可视化
        valid_depth = depth[mask]
        if valid_depth.numel() > 0:
            depth_min = valid_depth.min()
            depth_max = valid_depth.max()
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            depth_normalized = torch.zeros_like(depth)

        depth_normalized = torch.where(mask, depth_normalized, torch.ones_like(depth_normalized))

        # 转换为BGR格式的uint8图像（灰度图）
        depth_vis = torch.stack([depth_normalized] * 3, dim=-1)  # [H, W, 3]
        depth_vis_np = np.clip(np.rint(toNumpy(depth_vis) * 255), 0, 255).astype(np.uint8)
        depth_vis_np = depth_vis_np[..., ::-1]  # RGB -> BGR

        return {
            'depth': toNumpy(depth),  # [H, W] float32
            'image': depth_vis_np,  # [H, W, 3] BGR格式
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderNormal(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
    ) -> dict:
        """
        渲染法向图

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象

        Returns:
            dict包含:
                - normal_world: [H, W, 3] 世界坐标系下的法向图 (BGR格式, uint8)
                - normal_camera: [H, W, 3] 相机坐标系下的法向图 (BGR格式, uint8)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx = NVDiffRastRenderer._rasterize(mesh, camera)

        # 插值法线（世界坐标系）
        normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)  # [1, H, W, 3]
        normals_world = normals_interp[0] / (torch.norm(normals_interp[0], dim=-1, keepdim=True) + 1e-8)

        # 转换到相机坐标系
        normals_camera = torch.matmul(normals_world, camera.R.T)  # [H, W, 3]

        # 处理背景
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        background = torch.ones_like(normals_world)
        normals_world_vis = torch.where(mask.unsqueeze(-1), normals_world, background)
        normals_camera_vis = torch.where(mask.unsqueeze(-1), normals_camera, background)

        # 将法向从[-1, 1]映射到[0, 1]用于可视化
        normals_world_vis = (normals_world_vis + 1.0) * 0.5
        normals_camera_vis = (normals_camera_vis + 1.0) * 0.5

        # 转换为numpy uint8并转换颜色通道（RGB -> BGR，适配OpenCV）
        normal_world_np = np.clip(np.rint(toNumpy(normals_world_vis) * 255), 0, 255).astype(np.uint8)
        normal_world_np = normal_world_np[..., ::-1]  # RGB -> BGR

        normal_camera_np = np.clip(np.rint(toNumpy(normals_camera_vis) * 255), 0, 255).astype(np.uint8)
        normal_camera_np = normal_camera_np[..., ::-1]  # RGB -> BGR

        return {
            'normal_world': normal_world_np,  # [H, W, 3] BGR格式
            'normal_camera': normal_camera_np,  # [H, W, 3] BGR格式
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }
