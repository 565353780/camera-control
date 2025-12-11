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
        cx: float = 320.0,
        cy: float = 240.0,
        fx: float = 500.0,
        fy: float = 500.0,
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

        # UV坐标定义：ui = (ui, vi) ∈ [0,1]^2，相机中心为(0,0)
        # 投影方程：ui = π(P, xi)，其中P = K[R|t]是投影矩阵
        # 最小化重投影误差：P* = arg min_P sum_i ||π(P, xi) - ui||^2
        
        # 转换为numpy用于求解
        points_np = points.detach().cpu().numpy()
        uv_np = uv.detach().cpu().numpy()
        
        # Hartley归一化：提高数值稳定性
        # 1. 归一化3D点
        points_mean = points_np.mean(axis=0)
        points_centered = points_np - points_mean
        points_scale = np.sqrt((points_centered ** 2).sum(axis=1).mean()) / np.sqrt(3.0)
        if points_scale < 1e-8:
            points_scale = 1.0
        T_3d = np.eye(4)
        T_3d[:3, :3] = np.eye(3) / points_scale
        T_3d[:3, 3] = -points_mean / points_scale
        points_norm = (points_np - points_mean) / points_scale
        
        # 2. 归一化2D点（UV坐标，范围[0,1]^2，相机中心为0）
        # UV坐标已经是归一化的，但我们需要转换为像素坐标进行归一化
        # u_pixel = u * width, v_pixel = v * height（相对于相机中心的像素坐标）
        u_pixel = uv_np[:, 0] * width
        v_pixel = uv_np[:, 1] * height
        image_points = np.stack([u_pixel, v_pixel], axis=1)
        image_mean = image_points.mean(axis=0)
        image_centered = image_points - image_mean
        image_scale = np.sqrt((image_centered ** 2).sum(axis=1).mean()) / np.sqrt(2.0)
        if image_scale < 1e-8:
            image_scale = 1.0
        T_2d = np.eye(3)
        T_2d[:2, :2] = np.eye(2) / image_scale
        T_2d[:2, 2] = -image_mean / image_scale
        image_points_norm = (image_points - image_mean) / image_scale
        
        # 使用DLT算法（最小二乘）求解投影矩阵P
        # 投影方程: s * [u_norm, v_norm, 1]^T = P * [X, Y, Z, 1]^T
        # 其中(u_norm, v_norm)是归一化后的图像坐标
        A = np.zeros((2 * n_points, 12))
        for i in range(n_points):
            X, Y, Z = points_norm[i]
            u, v = image_points_norm[i]
            
            # 第一个方程: u * (p31*X + p32*Y + p33*Z + p34) = p11*X + p12*Y + p13*Z + p14
            A[2*i, 0] = X
            A[2*i, 1] = Y
            A[2*i, 2] = Z
            A[2*i, 3] = 1
            A[2*i, 8] = -u * X
            A[2*i, 9] = -u * Y
            A[2*i, 10] = -u * Z
            A[2*i, 11] = -u
            
            # 第二个方程: v * (p31*X + p32*Y + p33*Z + p34) = p21*X + p22*Y + p23*Z + p24
            A[2*i+1, 4] = X
            A[2*i+1, 5] = Y
            A[2*i+1, 6] = Z
            A[2*i+1, 7] = 1
            A[2*i+1, 8] = -v * X
            A[2*i+1, 9] = -v * Y
            A[2*i+1, 10] = -v * Z
            A[2*i+1, 11] = -v
        
        # 使用SVD求解最小二乘问题：P* = arg min ||A * vec(P)||^2
        _, _, Vt = np.linalg.svd(A)
        P_norm = Vt[-1].reshape(3, 4)
        
        # 反归一化：P = T_2d^{-1} * P_norm * T_3d
        T_2d_inv = np.linalg.inv(T_2d)
        P = T_2d_inv @ P_norm @ T_3d
        
        # RQ分解：将P分解为K和[R|t]
        # P = K[R|t]，其中K是内参矩阵（上三角），R是旋转矩阵，t是平移向量
        # 提取P的前3列：M = P[:, :3] = K @ R
        M = P[:, :3]
        
        # RQ分解：M = K @ R，其中K是上三角矩阵，R是正交矩阵
        # 使用翻转技巧实现RQ分解
        # 创建翻转矩阵J（上下翻转然后左右翻转）
        J = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        M_flip = J @ M @ J
        # 对M_flip^T做QR分解
        Q_flip, R_qr_flip = np.linalg.qr(M_flip.T)
        # M_flip = R_qr_flip^T @ Q_flip^T = K_flip @ R_flip
        K_flip = R_qr_flip.T  # 下三角
        R_flip = Q_flip.T     # 正交
        # 翻转回来得到上三角K和正交R
        K = J @ K_flip @ J
        R = J @ R_flip @ J
        
        # 确保K的对角元素为正（内参矩阵的焦距应该为正）
        # 同时调整R以保持M = K @ R的关系
        for i in range(3):
            if K[i, i] < 0:
                K[:, i] = -K[:, i]
                R[i, :] = -R[i, :]
        
        # 归一化K，使K[2,2] = 1
        if abs(K[2, 2]) > 1e-8:
            K = K / K[2, 2]
        
        # 从K提取内参
        # K = [[fx, s, cx_offset], [0, fy, cy_offset], [0, 0, 1]]
        # 其中cx_offset和cy_offset是相对于相机中心的偏移（像素单位）
        fx_est = K[0, 0]
        fy_est = K[1, 1]
        cx_offset = K[0, 2]  # 相对于相机中心的x偏移（像素）
        cy_offset = K[1, 2]  # 相对于相机中心的y偏移（像素）
        # 转换为相对于图像左上角的像素坐标
        cx_est = cx_offset + width / 2.0
        cy_est = cy_offset + height / 2.0
        
        # 从P提取平移向量t
        # P = K[R|t]，所以 [R|t] = K^{-1} @ P
        K_inv = np.linalg.inv(K)
        Rt = K_inv @ P
        R_extracted = Rt[:, :3]
        t = Rt[:, 3]
        
        # 使用从Rt提取的R（更准确），并进行SVD正交化以确保是有效的旋转矩阵
        U, _, Vt_svd = np.linalg.svd(R_extracted)
        R = U @ Vt_svd
        
        # 确保行列式为1（右手坐标系）
        if np.linalg.det(R) < 0:
            R = -R
        
        # 转换到相机参数表示
        # P = K[R|t]，其中R是从世界坐标系到相机坐标系的旋转，t是平移向量
        # 代码中的投影过程（行向量形式）: points_camera = (points_world - pos) @ rot.T
        # 标准PnP（列向量形式）: points_camera = R @ points_world + t
        # 转换为行向量形式: points_camera = points_world @ R.T + t.T
        # 对比: points_world @ R.T + t.T = points_world @ rot.T - pos @ rot.T
        # 所以: R = rot（旋转矩阵）
        # 并且: t = -pos @ R，即 pos = -R.T @ t（相机位置）
        # 相机旋转矩阵: rot的列向量是[right, up, forward]在世界坐标系中的表示
        rot_cam = R

        # 相机位置: pos = -R.T @ t（从平移向量t计算相机在世界坐标系中的位置）
        pos_cam = -R.T @ t

        # 转换为torch tensor
        # FIXME: 取负号才能得到正确结果
        rot_tensor = -toTensor(rot_cam)
        pos_tensor = -toTensor(pos_cam)

        # 创建相机对象（使用从K提取的内参）
        camera = cls(
            width=width,
            height=height,
            fx=fx_est,
            fy=fy_est,
            cx=cx_est,
            cy=cy_est,
            pos=pos_tensor,
            rot=rot_tensor,
        )

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
