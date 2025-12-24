import cv2
import torch
import numpy as np
from typing import Union, Optional

from camera_control.Data.camera import CameraData
from camera_control.Method.data import toNumpy, toTensor

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
    def fromUVPointsKernel(
        cls,
        points: Union[torch.Tensor, np.ndarray, list],
        uv: Union[torch.Tensor, np.ndarray, list],
        width: int = 640,
        height: int = 480,
        dtype=torch.float32,
        device: str = 'cpu',
    ):
        points = toTensor(points, dtype, device)
        if points.shape[-1] == 3:
            points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        else:
            points_homo = points

        points_homo = points_homo.reshape(-1, 4)

        points = points_homo[..., :3]
        uv = toTensor(uv, dtype, device).reshape(-1, 2)

        if points.shape[0] != uv.shape[0]:
            print('[ERROR][Camera::fromUVPointsKernel]')
            print('\t points and uv num not matched!')
            return None

        valid_mask = ~(torch.isnan(uv[:, 0]) | torch.isnan(uv[:, 1]))
        if valid_mask.sum() < 6:
            print('[ERROR][Camera::fromUVPoints]')
            print('\t Not enough valid points (need at least 6)!')
            return None

        points = points[valid_mask]
        uv = uv[valid_mask]
        n_points = points.shape[0]

        # Hartley归一化：提高数值稳定性
        # 归一化3D点
        points_mean = points.mean(dim=0)
        points_centered = points - points_mean
        points_scale = (points_centered ** 2).sum(dim=1).mean() / 3.0
        points_scale = torch.clamp(torch.sqrt(points_scale), min=1e-8)
        points_norm = points_centered / points_scale

        # 归一化2D点（UV坐标转换为像素坐标）
        scale_2d = torch.tensor([width, height], device=device, dtype=dtype)
        image_points = uv * scale_2d
        image_mean = image_points.mean(dim=0)
        image_centered = image_points - image_mean
        image_scale = (image_centered ** 2).sum(dim=1).mean() / 2.0
        image_scale = torch.clamp(torch.sqrt(image_scale), min=1e-8)
        image_points_norm = image_centered / image_scale

        # 向量化构建DLT矩阵A
        X, Y, Z = points_norm.T
        u, v = image_points_norm.T
        ones = torch.ones(n_points, device=device, dtype=dtype)

        A = torch.zeros(2 * n_points, 12, device=device, dtype=dtype)
        A[0::2, 0:4] = torch.stack([X, Y, Z, ones], dim=1)
        A[0::2, 8:12] = torch.stack([-u * X, -u * Y, -u * Z, -u], dim=1)
        A[1::2, 4:8] = torch.stack([X, Y, Z, ones], dim=1)
        A[1::2, 8:12] = torch.stack([-v * X, -v * Y, -v * Z, -v], dim=1)

        # SVD求解投影矩阵
        _, _, Vt = torch.linalg.svd(A, full_matrices=False)
        P_norm = Vt[-1].reshape(3, 4)

        # 反归一化
        inv_points_scale = 1.0 / points_scale
        T_3d = torch.eye(4, device=device, dtype=dtype)
        T_3d[:3, :3] *= inv_points_scale
        T_3d[:3, 3] = -points_mean * inv_points_scale

        T_2d_inv = torch.eye(3, device=device, dtype=dtype)
        T_2d_inv[:2, :2] *= image_scale
        T_2d_inv[:2, 2] = image_mean

        P = T_2d_inv @ P_norm @ T_3d

        # RQ分解：P = K[R|t]，提取K和[R|t]
        M = P[:, :3]

        # RQ分解：M = K @ R（使用翻转技巧）
        J = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=device, dtype=dtype)
        M_flip = J @ M @ J
        Q_flip, R_qr_flip = torch.linalg.qr(M_flip.T)
        K = J @ R_qr_flip.T @ J
        R = J @ Q_flip.T @ J

        # 确保K对角元素为正
        diag_signs = torch.sign(torch.diag(K))
        K = K * diag_signs.unsqueeze(1)
        R = R * diag_signs.unsqueeze(0)

        # 提取t并归一化K
        K_inv = torch.linalg.inv(K)
        Rt = K_inv @ P
        t = Rt[:, 3]
        k_scale = K[2, 2]
        if abs(k_scale.item()) > 1e-8:
            K = K / k_scale

        # 提取内参
        fx_est = K[0, 0].item()
        fy_est = K[1, 1].item()
        cx_est = (K[0, 2] + width / 2.0).item()
        cy_est = (K[1, 2] + height / 2.0).item()

        # SVD正交化R
        U, _, Vt_svd = torch.linalg.svd(Rt[:, :3], full_matrices=False)
        R = U @ Vt_svd

        pos_tensor = -(R.T @ t)

        if torch.linalg.det(R) < 0:
            R = -R
        rot_tensor = R.T

        camera = cls(
            width=width,
            height=height,
            fx=fx_est,
            fy=fy_est,
            cx=cx_est,
            cy=cy_est,
            dtype=dtype,
            device=device,
        )

        camera.setWorld2CameraByRt(rot_tensor, pos_tensor)
        return camera

    @classmethod
    def fromUVPointsV1(
        cls,
        points: Union[torch.Tensor, np.ndarray, list],
        uv: Union[torch.Tensor, np.ndarray, list],
        width: int = 640,
        height: int = 480,
        dtype=torch.float32,
        device: str = 'cpu',
    ):
        points = toTensor(points, dtype, device)

        if points.shape[-1] == 3:
            points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        else:
            points_homo = points

        points_homo = points_homo.reshape(-1, 4)
        uv = toTensor(uv, dtype, device).reshape(-1, 2)

        if points_homo.shape[0] != uv.shape[0]:
            print('[ERROR][Camera::fromUVPointsV1]')
            print('\t points and uv num not matched!')
            return None

        valid_mask = ~(torch.isnan(uv[:, 0]) | torch.isnan(uv[:, 1]))
        if valid_mask.sum() < 6:
            print('[ERROR][Camera::fromUVPointsV1]')
            print('\t Not enough valid points (need at least 6)!')
            return None

        points_homo = points_homo[valid_mask]
        uv = uv[valid_mask]

        C3 = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=dtype))
        C4 = torch.diag(torch.tensor([1, -1, -1, 1], device=device, dtype=dtype))

        flip_points_homo = points_homo @ C4

        center_uv = uv - 0.5
        flip_uv = center_uv
        flip_uv[:, 1] *= -1.0

        camera = cls.fromUVPointsKernel(
            flip_points_homo,
            flip_uv,
            width,
            height,
            dtype,
            device,
        )

        R = C3 @ camera.R.T @ C3
        t = -R @ C3 @ camera.t

        camera.setWorld2CameraByRt(R, t)
        return camera

    @classmethod
    def fromUVPointsV2(
        cls,
        points: Union[torch.Tensor, np.ndarray, list],
        uv: Union[torch.Tensor, np.ndarray, list],
        width: int = 640,
        height: int = 480,
        dtype=torch.float64,
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
            print('[ERROR][Camera::fromUVPointsV2]')
            print('\t points and uv num not matched!')
            return None

        valid_mask = ~(np.isnan(uv[:, 0]) | np.isnan(uv[:, 1]))
        if valid_mask.sum() < 6:
            print('[ERROR][Camera::fromUVPointsV2]')
            print('\t Not enough valid points (need at least 6)!')
            return None

        points = points[valid_mask]
        uv = uv[valid_mask]

        uv[:, 1] = 1.0 - uv[:, 1]

        pixels = uv * np.array([width, height], dtype=np.float64)

        K = np.array([
            [fx, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        ret, rvec_est, tvec_est = cv2.solvePnP(
            points,
            pixels,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

        ret, rvec_final, tvec_final = cv2.solvePnP(
            points,
            pixels,
            K,
            dist_coeffs,
            rvec=rvec_est,
            tvec=tvec_est,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ret:
            print('[ERROR][Camera::fromUVPointsV2]')
            print('\t solvePnP via opencv failed!')
            exit()

        R_mat, _ = cv2.Rodrigues(rvec_final)

        camera = cls(
            width,
            height,
            fx,
            fy,
            dtype=dtype,
            device=device,
        )

        C = np.diag([-1, 1, -1])

        R = C @ R_mat
        t = C @ tvec_final.flatten()

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
