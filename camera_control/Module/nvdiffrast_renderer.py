import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Union, Optional

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera


class NVDiffRastRenderer(object):
    # 类级别的 glctx 缓存，按 device 存储
    _glctx_cache: dict = {}

    def __init__(self) -> None:
        return

    @staticmethod
    def getGlctx(device: str) -> dr.RasterizeCudaContext:
        """
        获取或创建指定设备上的 RasterizeCudaContext

        Args:
            device: CUDA 设备字符串

        Returns:
            dr.RasterizeCudaContext: 光栅化上下文
        """
        if device not in NVDiffRastRenderer._glctx_cache:
            NVDiffRastRenderer._glctx_cache[device] = dr.RasterizeCudaContext(device=device)
        return NVDiffRastRenderer._glctx_cache[device]

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
    def _computeVertexNormals(
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        从顶点和面片计算可微的顶点法线

        Args:
            vertices: [V, 3] 顶点坐标
            faces: [F, 3] 面片索引

        Returns:
            vertex_normals: [V, 3] 顶点法线（已归一化）
        """
        # 获取三角形的三个顶点
        v0 = vertices[faces[:, 0]]  # [F, 3]
        v1 = vertices[faces[:, 1]]  # [F, 3]
        v2 = vertices[faces[:, 2]]  # [F, 3]

        # 计算面法线（未归一化，长度与面积成正比）
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = torch.cross(e1, e2, dim=1)  # [F, 3]

        # 累加到顶点法线
        vertex_normals = torch.zeros_like(vertices)
        vertex_normals = vertex_normals.index_add(0, faces[:, 0], face_normals)
        vertex_normals = vertex_normals.index_add(0, faces[:, 1], face_normals)
        vertex_normals = vertex_normals.index_add(0, faces[:, 2], face_normals)

        # 归一化
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)

        return vertex_normals

    @staticmethod
    def _rasterize(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        vertices_tensor: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        执行基础的光栅化操作（内部辅助方法）

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            vertices_tensor: 若提供，则使用该tensor替换mesh.vertices以支持可微渲染

        Returns:
            tuple: (vertices, faces, vertex_normals, rast_out, rast_out_db, glctx, vertices_clip)
        """
        # 提取基础几何信息
        if vertices_tensor is not None:
            vertices = vertices_tensor
        else:
            vertices = toTensor(mesh.vertices, torch.float32, camera.device)  # [V, 3]

        faces = toTensor(mesh.faces, torch.int32, camera.device)  # [F, 3]

        # 如果使用自定义顶点，重新计算法线以保持梯度流
        if vertices_tensor is not None:
            vertex_normals = NVDiffRastRenderer._computeVertexNormals(vertices, faces)
        else:
            vertex_normals = toTensor(mesh.vertex_normals, torch.float32, camera.device)  # [V, 3]

        # MVP变换和光栅化
        vertices_homo = torch.cat([
            vertices,
            torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=camera.device)
        ], dim=1)  # [V, 4]

        # bbox_length = torch.max(vertices, dim=0)[0] - torch.min(vertices, dim=0)[0]
        mvp = camera.getWorld2NVDiffRast()
        vertices_clip = torch.matmul(vertices_homo, mvp.T).unsqueeze(0).contiguous()  # [1, V, 4]

        glctx = NVDiffRastRenderer.getGlctx(camera.device)
        rast_out, rast_out_db = dr.rasterize(
            glctx,
            vertices_clip,
            faces,
            resolution=[camera.height, camera.width]
        )  # [1, H, W, 4]

        return vertices, faces, vertex_normals, rast_out, rast_out_db, glctx, vertices_clip

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
        渲染基于法向计算的shading图

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            light_direction: 光照方向（世界坐标系），默认为[1, 1, 1]
            paint_color: 可选的颜色列表（长度为3，0-1或0-255范围）
            bg_color: 背景颜色列表（长度为3，0-255范围），默认为[255, 255, 255]（白色）
            vertices_tensor: 可选的顶点tensor；若提供则作为渲染用顶点参与梯度传播
            enable_antialias: 是否启用抗锯齿（对梯度流至关重要），默认True

        Returns:
            dict包含:
                - image: [H, W, 3] 渲染的shading图像 (RGB格式, float32 tensor)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

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
        light_dir_cam = toTensor(light_direction, torch.float32, camera.device)
        # make light start from camera local coord
        # light_dir_cam = torch.matmul(light_dir, camera.R.T)
        light_dir_cam = light_dir_cam / (torch.norm(light_dir_cam) + 1e-8)

        normals_cam = torch.matmul(normals_interp, camera.R.T)  # [1, H, W, 3]

        diffuse = torch.sum(normals_cam * light_dir_cam[None, None, None, :], dim=-1)  # [1, H, W]
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        shading = 0.3 + 0.7 * diffuse  # 环境光 + 漫反射

        # 应用光照
        image = colors_interp * shading.unsqueeze(-1)  # [1, H, W, 3]

        # 处理背景色
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        bg_color_tensor = toTensor(bg_color, torch.float32, camera.device)[:3] / 255.0
        background = bg_color_tensor.view(1, 1, 1, 3).expand_as(image)  # [1, H, W, 3]
        image = torch.where(mask.unsqueeze(0).unsqueeze(-1), image, background)

        # 应用抗锯齿以确保边缘处的正确梯度流
        if enable_antialias:
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

        return {
            'rgb': image[0],  # [H, W, 3] RGB tensor
            'rasterize_output': rast_out[0],  # [H, W, 4]
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
        渲染网格本身颜色（包括纹理或顶点颜色，不带光照）

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            paint_color: 可选的颜色列表（长度为3，0-1或0-255范围）
            bg_color: 背景颜色列表（长度为3，0-255范围），默认为[255, 255, 255]（白色）
            vertices_tensor: 可选的顶点tensor；若提供则作为渲染用顶点参与梯度传播
            enable_antialias: 是否启用抗锯齿（对梯度流至关重要），默认True

        Returns:
            dict包含:
                - image: [H, W, 3] 渲染的图像 (RGB格式, float32 tensor)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        # 检查是否使用纹理
        use_texture = False
        if NVDiffRastRenderer.isTextureExist(mesh):
            # 提取UV和texture
            uvs = toTensor(mesh.visual.uv, torch.float32, camera.device)

            # 验证 UV 数量与顶点数量是否匹配
            if uvs.shape[0] != vertices.shape[0]:
                print(f'[WARN][NVDiffRastRenderer::renderTexture] UV count ({uvs.shape[0]}) != vertex count ({vertices.shape[0]}), falling back to vertex colors')
            else:
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

        if not use_texture:
            return NVDiffRastRenderer.renderVertexColor(
                mesh=mesh,
                camera=camera,
                light_direction=[1, 1, 1],
                paint_color=paint_color,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                enable_antialias=enable_antialias,
            )

        # 使用纹理渲染
        uv_interp, _ = dr.interpolate(uvs.unsqueeze(0), rast_out, faces)  # [1, H, W, 2]
        image = dr.texture(texture, uv_interp, filter_mode='linear')  # [1, H, W, 3]

        # 处理背景色
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        bg_color_tensor = toTensor(bg_color, torch.float32, camera.device)[:3] / 255.0
        background = bg_color_tensor.view(1, 1, 1, 3).expand_as(image)  # [1, H, W, 3]
        image = torch.where(mask.unsqueeze(0).unsqueeze(-1), image, background)

        # 应用抗锯齿以确保边缘处的正确梯度流
        if enable_antialias:
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

        return {
            'rgb': image[0],  # [H, W, 3] RGB tensor
            'rasterize_output': rast_out[0],  # [H, W, 4]
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

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            bg_color: 背景颜色列表（长度为3，0-255范围），默认为[255, 255, 255]（白色）
            vertices_tensor: 可选的顶点tensor；若提供则作为渲染用顶点参与梯度传播
            enable_antialias: 是否启用抗锯齿（对梯度流至关重要），默认True

        Returns:
            dict包含:
                - depth: [H, W] 深度图 (float32, 单位：米)
                - image: [H, W, 3] 归一化深度图用于可视化 (RGB格式, float32 tensor)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        # 计算世界坐标系下的深度（Z坐标）
        vertices_interp, _ = dr.interpolate(vertices.unsqueeze(0), rast_out, faces)  # [1, H, W, 3]

        # 将顶点转换到相机坐标系
        # 正确的变换: p_camera = R @ p_world + t
        vertices_cam = torch.matmul(vertices_interp[0], camera.R.T) + camera.t  # [H, W, 3]
        # 由于相机坐标系Z向后（看向-Z方向），物体在前方时Z为负值，所以深度取-Z
        depth = -vertices_cam[:, :, 2]  # [H, W] - 深度（距离相机的正值）

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

        # 转换为RGB格式（灰度图）
        depth_vis = torch.stack([depth_normalized] * 3, dim=-1).unsqueeze(0)  # [1, H, W, 3]

        # 应用背景色（在RGB格式下）
        bg_color_tensor = toTensor(bg_color, torch.float32, camera.device)[:3] / 255.0
        background = bg_color_tensor.view(1, 1, 1, 3).expand_as(depth_vis)  # [1, H, W, 3]
        image = torch.where(mask.unsqueeze(0).unsqueeze(-1), depth_vis, background)

        # 应用抗锯齿以确保边缘处的正确梯度流
        if enable_antialias:
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

        return {
            'depth': depth,  # [H, W] float32
            'rgb': image[0],  # [H, W, 3] RGB tensor
            'rasterize_output': rast_out[0],  # [H, W, 4]
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
        渲染法向图

        Args:
            mesh: trimesh.Trimesh或trimesh.Scene对象
            camera: Camera对象
            bg_color: 背景颜色列表（长度为3，0-255范围），默认为[255, 255, 255]（白色）
            vertices_tensor: 可选的顶点tensor；若提供则作为渲染用顶点参与梯度传播
            enable_antialias: 是否启用抗锯齿（对梯度流至关重要），默认True

        Returns:
            dict包含:
                - normal_world: [H, W, 3] 世界坐标系下的法向图 (RGB格式, float32 tensor)
                - normal_camera: [H, W, 3] 相机坐标系下的法向图 (RGB格式, float32 tensor)
                - rasterize_output: [H, W, 4] rasterize输出
                - bary_derivs: [H, W, 4] 重心坐标导数
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, glctx, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        # 插值法线（世界坐标系）
        normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)  # [1, H, W, 3]
        normals_world = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)  # [1, H, W, 3]

        # 转换到相机坐标系
        normals_camera = torch.matmul(normals_world, camera.R.T)  # [1, H, W, 3]

        # 将法向从[-1, 1]映射到[0, 1]用于可视化
        normals_world_mapped = (normals_world + 1.0) * 0.5
        normals_camera_mapped = (normals_camera + 1.0) * 0.5

        # 处理背景
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        bg_color_tensor = toTensor(bg_color, torch.float32, camera.device)[:3] / 255.0
        background = bg_color_tensor.view(1, 1, 1, 3).expand_as(normals_world_mapped)  # [1, H, W, 3]
        normals_world_vis = torch.where(mask.unsqueeze(0).unsqueeze(-1), normals_world_mapped, background)
        normals_camera_vis = torch.where(mask.unsqueeze(0).unsqueeze(-1), normals_camera_mapped, background)

        # 应用抗锯齿以确保边缘处的正确梯度流
        if enable_antialias:
            normals_world_vis = dr.antialias(normals_world_vis.contiguous(), rast_out, vertices_clip, faces)
            normals_camera_vis = dr.antialias(normals_camera_vis.contiguous(), rast_out, vertices_clip, faces)

        return {
            'world': normals_world[0],  # [H, W, 3] XYZ tensor
            'camera': normals_camera[0],  # [H, W, 3] XYZ tensor
            'rgb_world': normals_world_vis[0],  # [H, W, 3] RGB tensor
            'rgb_camera': normals_camera_vis[0],  # [H, W, 3] RGB tensor
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }
