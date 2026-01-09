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

    def toImageUV(self) -> np.ndarray:
        u = np.arange(self.width).astype(np.float32) / (self.width - 1.0)
        v = np.arange(self.height).astype(np.float32) / (self.height - 1.0)

        # 构造网格
        uu, vv = np.meshgrid(u, v, indexing='xy')  # uu/vv shape: [height, width]

        # 按照"图片左下角为0,0，u向右，v向上"来构造v
        # 原始opencv: (0,0)在左上，v向下递增
        vv_new = 1.0 - vv  # [height, width]

        uv = np.zeros((self.height, self.width, 2), dtype=np.float32)

        uv[:, :, 0] = uu          # u
        uv[:, :, 1] = vv_new      # v (左下角)
        return uv

    def projectUV2Points(
        self,
        uv: Union[torch.Tensor, np.ndarray, list],
        depth: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        将UV坐标和深度值反投影到世界坐标系中的3D点

        坐标系定义：
        - 相机坐标系：X右，Y上，Z朝向相机背后（相机看向-Z）
        - UV坐标系：原点在左下角(0,0)，u向右，v向上
        - depth：相机前方的距离（正值），即 depth = -Z

        Args:
            uv: UV坐标，shape (..., 2)，范围 [0, 1]
            depth: 深度值，shape (...,) 或 (..., 1)，相机前方的距离（正值）

        Returns:
            points: 世界坐标系中的3D点，shape (..., 3)
        """
        uv = toTensor(uv, self.dtype, self.device)
        depth = toTensor(depth, self.dtype, self.device)

        # 处理输入形状
        if uv.ndim == 1:
            uv = uv.unsqueeze(0)
        if depth.ndim == 0:
            depth = depth.unsqueeze(0)
        if depth.ndim > 0 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)

        # 确保uv至少有2维，最后一维是2
        if uv.shape[-1] != 2:
            print('[ERROR][Camera::projectUV2Points]')
            print(f'\t uv last dimension must be 2, got {uv.shape}')
            return torch.empty(0, 3, dtype=self.dtype, device=self.device)

        # 获取uv的前N-1维形状
        uv_shape = uv.shape[:-1]  # 去掉最后一维（2）

        # 处理depth的形状，使其与uv_shape兼容
        # 如果depth是标量，直接广播
        if depth.ndim == 0:
            depth = depth.expand(uv_shape)
        # 如果depth的维度少于uv_shape，在左侧添加维度
        elif depth.ndim < len(uv_shape):
            # 在depth前面添加维度
            for _ in range(len(uv_shape) - depth.ndim):
                depth = depth.unsqueeze(0)
            # 尝试expand到uv_shape
            try:
                depth = depth.expand(uv_shape)
            except RuntimeError:
                print('[ERROR][Camera::projectUV2Points]')
                print(f'\t uv shape: {uv.shape}, depth shape: {depth.shape}')
                print('\t Cannot broadcast depth to match uv shape!')
                return torch.empty(0, 3, dtype=self.dtype, device=self.device)
        # 如果depth的维度等于uv_shape，检查是否可以广播
        elif depth.ndim == len(uv_shape):
            if depth.shape != uv_shape:
                try:
                    # 尝试广播
                    depth = depth.expand(uv_shape)
                except RuntimeError:
                    print('[ERROR][Camera::projectUV2Points]')
                    print(f'\t uv shape: {uv.shape}, depth shape: {depth.shape}')
                    print('\t Cannot broadcast depth to match uv shape!')
                    return torch.empty(0, 3, dtype=self.dtype, device=self.device)
        # 如果depth的维度大于uv_shape，报错
        else:
            print('[ERROR][Camera::projectUV2Points]')
            print(f'\t uv shape: {uv.shape}, depth shape: {depth.shape}')
            print('\t depth has more dimensions than uv (excluding last dim)!')
            return torch.empty(0, 3, dtype=self.dtype, device=self.device)

        # 将UV坐标（范围[0,1]）转换为像素坐标
        u_pixel = uv[..., 0] * self.width
        v_pixel = uv[..., 1] * self.height

        # 反投影到相机坐标系
        # 投影公式：u_pixel = fx * X / (-Z) + cx
        # 因此：X = (u_pixel - cx) * (-Z) / fx
        # 由于 depth = -Z（depth是相机前方的距离），所以：
        x_camera = (u_pixel - self.cx) * depth / self.fx
        y_camera = (v_pixel - self.cy) * depth / self.fy
        z_camera = -depth  # Z轴指向相机背后，所以是负值

        # 组合成相机坐标系中的3D点
        points_camera = torch.stack([x_camera, y_camera, z_camera], dim=-1)

        # 转换为齐次坐标
        points_camera_homo = torch.cat([
            points_camera,
            torch.ones_like(points_camera[..., :1])
        ], dim=-1)

        # 从相机坐标系转换到世界坐标系
        points_world_homo = torch.matmul(points_camera_homo, self.camera2world.T)
        points_world = points_world_homo[..., :3]

        return points_world
