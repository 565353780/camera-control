import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Union

from camera_control.Method.data import toNumpy, toTensor
from camera_control.Module.camera import Camera


class NVDiffRastRenderer(object):
    def __init__(self) -> None:
        return

    def renderImage(
        self,
        mesh: trimesh.Trimesh,
        camera: Camera,
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
    ) -> dict:
        """
        使用nvdiffrast渲染三角网格，并获取渲染图中每个像素对应的
        三角网格表面的顶点插值信息

        Args:
            camera: Camera对象，包含相机的所有参数（位置、旋转、内参等）
            light_direction: 光照方向（世界坐标系），默认为[1, 1, 1]

        Returns:
            dict包含:
                - image: [H, W, 3] 渲染的图像 (RGB)
                - rasterize_output: [H, W, 4] rasterize主输出 (u, v, z/w, triangle_id)
                - bary_derivs: [H, W, 4] 重心坐标的图像空间导数 (du/dX, du/dY, dv/dX, dv/dY)
        """
        # 获取三角网格信息
        vertices = toTensor(mesh.vertices, torch.float32, camera.device)  # [V, 3]
        faces = toTensor(mesh.faces, torch.int32, camera.device)  # [F, 3]
        vertex_normals = toTensor(mesh.vertex_normals, torch.float32, camera.device)  # [V, 3]
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = toTensor(mesh.visual.vertex_colors[:, :3] / 255.0, torch.float32, camera.device)  # [V, 3]
        else:
            vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)

        # 设置光照方向（世界坐标系）
        light_direction = toTensor(light_direction, device=camera.device)
        light_direction = light_direction / (torch.norm(light_direction) + 1e-8)

        # 顶点转为齐次裁剪空间
        vertices_homo = torch.cat([
            vertices,
            torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=camera.device)
        ], dim=1)  # [V, 4]

        # 得到 nvdiffrast 所需 MVP 变换
        bbox_length = torch.max(vertices, dim=0)[0] - torch.min(vertices, dim=0)[0]
        mvp = camera.getWorld2NVDiffRast(bbox_length)

        # 顶点裁剪
        vertices_clip_batch = torch.matmul(vertices_homo, mvp.T).unsqueeze(0).contiguous()  # [1, V, 4]

        glctx = dr.RasterizeCudaContext(device=camera.device)

        rast_out, rast_out_db = dr.rasterize(
            glctx,
            vertices_clip_batch,  # [1, V, 4]
            faces,
            resolution=[camera.height, camera.width]
        )  # rast_out: [1, H, W, 4]

        # 执行属性插值：normals 和 colors
        vert_normals_batch = vertex_normals.unsqueeze(0)    # [1, V, 3]
        vert_colors_batch = vertex_colors.unsqueeze(0)      # [1, V, 3]

        normals_interp, _ = dr.interpolate(vert_normals_batch, rast_out, faces)  # [1, H, W, 3]
        colors_interp, _ = dr.interpolate(vert_colors_batch, rast_out, faces)    # [1, H, W, 3]

        # 归一化插值后的法向量
        normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)  # [1, H, W, 3]
        colors_interp = torch.clamp(colors_interp, 0.0, 1.0)  # [1, H, W, 3]

        # 获取世界到相机坐标的旋转部分，将法线和光照从世界变换到相机
        # R_view: camera.world2camera 的左上3x3部分
        R_view = camera.R   # [3,3]
        normals_cam = torch.matmul(normals_interp, R_view.T)    # [1, H, W, 3]
        light_dir_cam = torch.matmul(light_direction, R_view.T) # [3]
        light_dir_cam = light_dir_cam / (torch.norm(light_dir_cam) + 1e-8)

        # Lambert 着色
        diffuse = torch.sum(normals_cam * light_dir_cam[None, None, None, :], dim=-1)  # [1, H, W]
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        ambient = 0.3
        shaded = ambient + (1.0 - ambient) * diffuse     # [1, H, W]

        # 应用插值颜色并拼出RGB图像
        image = colors_interp * shaded.unsqueeze(-1)     # [1, H, W, 3]
        image = image[0]                                 # [H, W, 3]

        # 处理背景为白色
        mask = rast_out[0, :, :, 3] > 0                  # [H, W]
        background = torch.ones_like(image)
        image = torch.where(mask.unsqueeze(-1), image, background)
        render_image_np = np.clip(np.rint(toNumpy(image) * 255), 0, 255).astype(np.uint8)

        result = {
            'image': render_image_np,  # [H, W, 3]
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),  # [H, W, 4]
        }

        return result
