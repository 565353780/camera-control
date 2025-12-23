import os
import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Optional, Union

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera


class NVDiffRastRenderer(object):
    def __init__(
        self,
        mesh_file_path: Optional[str]=None,
        device: str = 'cuda:0',
        color: list=[178, 178, 178],
    ) -> None:
        self.mesh: trimesh.Trimesh = None
        self.device = torch.device(device)

        if mesh_file_path is not None:
            self.loadMeshFile(mesh_file_path, color)
        return

    def isValid(self) -> bool:
        if self.mesh is None:
            return False

        return True

    def loadMeshFile(
        self,
        mesh_file_path: str,
        color: list=[178, 178, 178],
    ) -> bool:
        if not os.path.exists(mesh_file_path):
            print('[ERROR][NVDiffRastRenderer::loadMeshFile]')
            print('\t mesh file not exist!')
            print('\t mesh_file_path:', mesh_file_path)
            return False

        self.mesh = trimesh.load(mesh_file_path)

        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(
                [g for g in self.mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )

        if not isinstance(self.mesh, trimesh.Trimesh):
            print('[ERROR][NVDiffRastRenderer::loadMeshFile]')
            print('\t load mesh failed!')
            print('\t mesh_file_path:', mesh_file_path)
            return False

        if not hasattr(self.mesh.visual, 'vertex_colors') or self.mesh.visual.vertex_colors is None:
            num_verts = len(self.mesh.vertices)
            vertex_colors = np.tile(np.array(color), (num_verts, 1))
            self.mesh.visual.vertex_colors = vertex_colors

        if not hasattr(self.mesh, 'vertex_normals') or self.mesh.vertex_normals is None:
            self.mesh.compute_vertex_normals()

        return True

    def renderImage(
        self,
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
        width = camera.width
        height = camera.height
        fx = camera.fx
        fy = camera.fy
        cx = camera.cx
        cy = camera.cy

        pos = camera.pos.float().to(self.device)  # [3]
        rot = camera.rot.float().to(self.device)  # [3, 3]

        # 坐标系转换矩阵定义：
        # 1. Camera类坐标系 -> OpenGL坐标系
        #    Camera: X=right, Y=down, Z=forward
        #    OpenGL: X=right, Y=up, Z=-forward
        camera_to_opengl = torch.tensor([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=torch.float32, device=self.device)
        
        # 2. OpenGL坐标系 -> 图像坐标系（用于投影后的Y轴翻转）
        #    OpenGL: 原点左下角，Y向上
        #    Image:  原点左上角，Y向下
        opengl_to_image = torch.tensor([
            [1.0,  0.0,  0.0,  0.0],
            [0.0, -1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0,  0.0],
            [0.0,  0.0,  0.0,  1.0]
        ], dtype=torch.float32, device=self.device)

        vertices = torch.from_numpy(self.mesh.vertices).float().to(self.device)  # [V, 3]
        faces = torch.from_numpy(self.mesh.faces).int().to(self.device)  # [F, 3]
        vertex_normals = torch.from_numpy(self.mesh.vertex_normals).float().to(self.device)  # [V, 3]

        # 设置光照方向（世界坐标系）
        light_direction = toTensor(light_direction, device=self.device)
        light_direction = light_direction / (torch.norm(light_direction) + 1e-8)

        # 4. 构建投影矩阵（根据相机内参）
        # 计算 near 和 far 裁剪面
        bbox_size = np.linalg.norm(np.max(self.mesh.vertices, axis=0) - np.min(self.mesh.vertices, axis=0))
        near = bbox_size * 0.1
        far = bbox_size * 10.0

        # 从相机内参构建标准 OpenGL 投影矩阵
        # 参考：https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        proj_mtx = torch.zeros((4, 4), dtype=torch.float32, device=self.device)

        # 使用完整的相机内参：fx, fy, cx, cy（标准OpenGL投影，不含坐标系转换）
        proj_mtx[0, 0] = 2.0 * fx / width
        proj_mtx[1, 1] = 2.0 * fy / height
        proj_mtx[0, 2] = (width - 2.0 * cx) / width
        proj_mtx[1, 2] = (2.0 * cy - height) / height
        proj_mtx[2, 2] = (far + near) / (near - far)
        proj_mtx[2, 3] = (2.0 * far * near) / (near - far)
        proj_mtx[3, 2] = -1.0

        # 5. 构建视图矩阵
        # Camera类的rot是从Camera坐标系到世界坐标系的变换
        # rot.T 是从世界坐标系到Camera坐标系的变换
        # 然后通过camera_to_opengl转换到OpenGL坐标系
        R_world_to_camera = rot.T  # [3, 3] 世界坐标系 -> Camera类坐标系
        R_view = camera_to_opengl @ R_world_to_camera  # [3, 3] 世界坐标系 -> OpenGL坐标系

        view_mtx = torch.eye(4, dtype=torch.float32, device=self.device)
        view_mtx[:3, :3] = R_view
        view_mtx[:3, 3] = -R_view @ pos  # 平移部分

        # 6. 组合变换矩阵：图像坐标系 <- OpenGL <- 世界坐标系
        # MVP = (OpenGL->Image) @ Projection @ View
        mvp = opengl_to_image @ proj_mtx @ view_mtx  # [4, 4]

        # 顶点变换到裁剪空间
        vertices_homo = torch.cat([
            vertices,
            torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=self.device)
        ], dim=1)  # [V, 4]

        vertices_clip = torch.matmul(vertices_homo, mvp.t())  # [V, 4]
        vertices_clip_batch = vertices_clip.unsqueeze(0).contiguous()

        # 光栅化
        glctx = dr.RasterizeCudaContext(device=self.device)

        rast_out, rast_out_db = dr.rasterize(
            glctx,
            vertices_clip_batch,  # [1, V, 4]
            faces,
            resolution=[height, width]
        )

        # 法向着色：插值顶点法向量
        normals_interp, _ = dr.interpolate(
            vertex_normals.unsqueeze(0),  # [1, V, 3]
            rast_out,
            faces
        )

        # 归一化插值后的法向量
        normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)

        # 将法向量从世界坐标系转换到OpenGL相机坐标系
        # 注意：法向量的变换使用R_view.T（而不是view矩阵的逆）
        normals_cam = torch.matmul(normals_interp[0], R_view.T)  # [H, W, 3]

        # 将光照方向也转换到OpenGL相机坐标系
        light_dir_cam = torch.matmul(light_direction, R_view.T)  # [3]
        light_dir_cam = light_dir_cam / (torch.norm(light_dir_cam) + 1e-8)

        # 计算Lambert着色
        diffuse = torch.sum(normals_cam * light_dir_cam, dim=-1, keepdim=False)  # [H, W]
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)

        # 添加环境光
        ambient = 0.3
        image = ambient + (1.0 - ambient) * diffuse  # [H, W]

        # 处理背景
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        background = torch.ones_like(image)  # 白色背景
        image = torch.where(mask, image, background)
        render_image = image.detach().cpu().numpy()
        render_image_np = np.clip(np.rint(render_image * 255), 0, 255).astype(np.uint8)

        result = {
            'image': render_image_np,  # [H, W]
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),  # [H, W, 4]
        }

        return result
