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
        width: int = 640,
        height: int = 480,
    ):
        points = toTensor(points).reshape(-1, 3)
        uv = toTensor(uv).reshape(-1, 2)

        if points.shape[0] != uv.shape[0]:
            print('[ERROR][Camera::fromUVPoints]')
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
        device = points.device
        dtype = points.dtype

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
        if torch.linalg.det(R) < 0:
            R = -R

        # 转换为相机参数（左手坐标系）
        rot_tensor = -R
        pos_tensor = -(R.T @ t)

        return cls(
            width=width,
            height=height,
            fx=fx_est,
            fy=fy_est,
            cx=cx_est,
            cy=cy_est,
            pos=pos_tensor,
            rot=rot_tensor,
        )

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
