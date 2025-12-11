import torch
import numpy as np
import open3d as o3d
from typing import Union

from uv_cube_gen.Data.camera import CameraData
from uv_cube_gen.Method.data import toTensor

class Camera(CameraData):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 240.0,
        pos: Union[torch.Tensor, np.ndarray, list] = [0, 0, 0],
        look_at: Union[torch.Tensor, np.ndarray, list] = [1, 0, 0],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 1, 0],
        rot: Union[torch.Tensor, np.ndarray, list, None] = None,
    ) -> None:
        CameraData.__init__(
            self,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            pos,
            look_at,
            up,
            rot,
        )
        return

    @classmethod
    def fromUVPoints(
        cls,
        points: Union[torch.Tensor, np.ndarray, list],
        uv: Union[torch.Tensor, np.ndarray, list],
    ):
        if points.shape[:-1] != uv.shape[:-1]:
            print('[ERROR][Camera::fromUVPoints]')
            print('\t points and uv shape not matched!')
            return None

        camera = cls()

        return camera

    def project_points_to_uv(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        points = toTensor(points)

        if points.ndim == 1:
            points = points.unsqueeze(0)

        points_camera = torch.matmul(points - self.pos, self.rot.T)

        x, y, z = points_camera[..., 0], points_camera[..., 1], points_camera[..., 2]

        z_safe = torch.where(z > 1e-8, z, torch.ones_like(z) * 1e-8)
        u_pixel = self.fx * x / z_safe + self.cx
        v_pixel = self.fy * y / z_safe + self.cy

        u = (u_pixel - self.width / 2.0) / self.width
        v = (v_pixel - self.height / 2.0) / self.height

        invalid_mask = z <= 1e-8
        u = torch.where(invalid_mask, torch.full_like(u, float('nan')), u)
        v = torch.where(invalid_mask, torch.full_like(v, float('nan')), v)

        uv = torch.stack([u, v], dim=-1)

        return uv

    def toO3DMesh(
        self,
        far: float=0.1,
        color: list=[0, 1, 0],
    ) -> o3d.geometry.TriangleMesh:
        half_width = (self.width / self.fx) * far
        half_height = (self.height / self.fy) * far

        far_corners = np.array([
            [-half_width, half_height, far],
            [half_width, half_height, far],
            [half_width, -half_height, far],
            [-half_width, -half_height, far],
        ])

        pos = self.pos.numpy()
        far_corners_world = (self.rot.numpy() @ far_corners.T).T + pos

        vertices = np.vstack([far_corners_world, pos.reshape(1, 3)])

        triangles = []

        triangles.append([0, 2, 1])
        triangles.append([0, 3, 2])

        triangles.append([4, 0, 1])
        triangles.append([4, 1, 2])
        triangles.append([4, 2, 3])
        triangles.append([4, 3, 0])

        frustum = o3d.geometry.TriangleMesh()
        frustum.vertices = o3d.utility.Vector3dVector(vertices)
        frustum.triangles = o3d.utility.Vector3iVector(triangles)
        frustum.paint_uniform_color(color)
        frustum.compute_vertex_normals()

        return frustum
