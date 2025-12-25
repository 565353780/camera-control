import torch
import numpy as np
from typing import Union, Optional

from camera_control.Data.camera import CameraData
from camera_control.Method.data import toNumpy, toTensor
from camera_control.Module.camera_refiner import solve_and_refine

class Camera(CameraData):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        pos: Union[torch.Tensor, np.ndarray, list] = [0, 0, 0],
        look_at: Union[torch.Tensor, np.ndarray, list] = [1, 0, 0],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 0, 1],
        world2camera: Union[torch.Tensor, np.ndarray, list, None] = None,
        dtype=torch.float32,
        device: str = 'cpu',
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
            world2camera,
            dtype,
            device,
        )
        return

    @classmethod
    def fromUVPoints(
        cls,
        points: Union[torch.Tensor, np.ndarray, list],
        uv: Union[torch.Tensor, np.ndarray, list],
        width: int = 640,
        height: int = 480,
        dtype=torch.float32,
        device: str = 'cpu',
    ):
        fx = 500
        fy = 500

        points = toNumpy(points)

        if points.shape[-1] != 3:
            points = points[..., :3]

        points = points.reshape(-1, 3)
        uv = toNumpy(uv).reshape(-1, 2)

        if points.shape[0] != uv.shape[0]:
            print('[ERROR][Camera::fromUVPoints]')
            print('\t points and uv num not matched!')
            return None

        valid_mask = ~(np.isnan(uv[:, 0]) | np.isnan(uv[:, 1]))
        if valid_mask.sum() < 6:
            print('[ERROR][Camera::fromUVPoints]')
            print('\t Not enough valid points (need at least 6)!')
            return None

        points = points[valid_mask]
        uv = uv[valid_mask]

        uv[:, 1] = 1.0 - uv[:, 1]

        pixels = uv * np.array([width, height], dtype=np.float64)

        R, t, K = solve_and_refine(
            points,
            pixels,
            width=width,
            height=height,
            fx_init=fx,
            fy_init=fy,
            fix_principal_point=True,
        )

        camera = cls(
            width,
            height,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            dtype=dtype,
            device=device,
        )

        C = np.diag([1, -1, -1])

        R = C @ R
        t = C @ t

        camera.setWorld2CameraByRt(R, t)

        return camera


    def project_points_to_uv(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        将世界坐标系中的3D点投影到UV坐标

        坐标系定义：
        - 相机坐标系：X右，Y上，Z朝向相机背后（相机看向-Z）
        - UV坐标系：原点在左下角(0,0)，u向右，v向上
        - 投影公式：u_pixel = fx * X / (-Z) + cx，v_pixel = fy * Y / (-Z) + cy
        - 可见点条件：Z < 0（点在相机前方）

        Returns:
            uv: shape (..., 2)，范围 [0, 1]
            对于不可见的点（Z >= 0或太近），返回NaN
        """
        points = toTensor(points, self.dtype, self.device)

        if points.ndim == 1:
            points = points.unsqueeze(0)

        if points.shape[-1] == 3:
            points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        else:
            points_homo = points
        points_camera = torch.matmul(points_homo, self.world2camera.T)[..., :3]

        x, y, z = points_camera[..., 0], points_camera[..., 1], points_camera[..., 2]

        # 相机看向-Z方向，可见点应该在Z < 0的区域
        # 投影公式：u_pixel = fx * X / (-Z) + cx
        # 即除以 (-Z)，对于Z<0的点，-Z>0是正数

        # 标记无效点（在相机后方 Z>=0 或太近 Z>=-1e-6的点）
        invalid_mask = z >= -1e-6

        # 对于有效点，计算投影
        # 为避免除零，将z限制在安全范围内
        z_safe = torch.clamp(z, max=-1e-6)
        u_pixel = self.fx * x / (-z_safe) + self.cx
        v_pixel = self.fy * y / (-z_safe) + self.cy

        # UV坐标归一化：原点在(0, 0)左下角
        u = u_pixel / self.width
        v = v_pixel / self.height

        # 将无效点标记为NaN
        u = torch.where(invalid_mask, torch.full_like(u, float('nan')), u)
        v = torch.where(invalid_mask, torch.full_like(v, float('nan')), v)

        uv = torch.stack([u, v], dim=-1)

        return uv
