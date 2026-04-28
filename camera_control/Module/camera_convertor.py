import os
import cv2
import torch
import trimesh
import numpy as np
import open3d as o3d

from tqdm import tqdm
from copy import deepcopy
from shutil import rmtree
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor

from camera_control.Method.pcd import toPcd
from camera_control.Method.data import toNumpy, toTensor
from camera_control.Method.rotate import decompose_similarity_from_T, invert_similarity
from camera_control.Module.camera import Camera


class CameraConvertor(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def transformCameras(
        camera_list: List[Camera],
        world_transform: Union[torch.Tensor, np.ndarray, list],
    ) -> List[Camera]:
        """
        用给定的 4x4 世界变换矩阵对所有相机的世界位姿进行变换。

        约定：传入的 world_transform 是 Open3D ICP 矩阵的转置（行向量右乘约定）。
        结合 SVD 分解，滤除所有仿射/错切噪声，确保对相机的变换是绝对严格的刚体+均匀缩放。
        """
        transformed_list = deepcopy(camera_list)
        if not transformed_list:
            return transformed_list

        device = transformed_list[0].device
        dtype = transformed_list[0].dtype

        # =========================================================
        # 1. 矩阵桥接与 SVD 极度提纯
        # =========================================================
        # T_right 是行向量右乘矩阵 (p_new = p_old @ T_right)
        T_right = toTensor(world_transform, dtype, device).reshape(4, 4)

        # 使用 SVD 提取出绝对纯净的左乘参数 R_left, s, t_left
        R_left, s, t_left = decompose_similarity_from_T(T_right.T, enforce_positive_scale=True)

        scale_safe = s.clamp(min=1e-8)

        T_clean_left_inv = invert_similarity(R_left, scale_safe, t_left)

        for cam in transformed_list:
            # =========================================================
            # Step A: 乘以带有 1/s 缩放的纯净逆矩阵
            # =========================================================
            # p_cam = W2C_old @ (T_clean_left^-1) @ p_world_new_col
            cam.world2camera = cam.world2camera @ T_clean_left_inv

            # =========================================================
            # Step B: 【核心刚体补偿】
            # 将前三行乘以 scale_safe，消除齐次逆矩阵带来的 1/s 缩放污染。
            # 这一步将 W2C 完美复原为正交的 SE(3) 刚体位姿，
            # 并在物理空间中将相机的物理位置等比例地精确推远 s 倍！
            # =========================================================
            cam.world2camera[:3, :] = cam.world2camera[:3, :] * scale_safe

            # =========================================================
            # Step C: 更新深度与点云重投影
            # =========================================================
            if getattr(cam, "depth", None) is not None:
                # 场景放大 s 倍，相机观察同视角的物理深度同步放大 s 倍
                cam.depth = cam.depth * scale_safe

            # =========================================================
            # Step D: 法线旋转更新
            # =========================================================
            if getattr(cam, "normal_world", None) is not None:
                cam.normal_world = (cam.normal_world @ R_left.T).clone()
                cam.normal_world = cam.normal_world / (torch.linalg.norm(cam.normal_world, dim=-1, keepdim=True) + 1e-8)

        return transformed_list

    @staticmethod
    def getZAxisNormalizeTransform(
        camera_list: List[Camera],
        x_direction: list=[1, 0, 0],
        y_direction: list=[0, 1, 0],
        z_direction: list=[0, 0, 1],
    ) -> torch.Tensor:
        """
        计算用于归一化相机阵列世界坐标系的 4x4 变换矩阵：
        1. 旋转使所有相机 Y 轴（图像朝上）的平均方向对齐 z_direction
        2. 绕 z_direction 轴旋转使第一个相机的视线方向对齐 -x_direction
        3. 平移使所有视线射线的最近公共交点位于世界原点

        相机坐标系约定：X 右、Y 上、Z 后，相机看向 -Z。

        Args:
            camera_list: 相机列表（只读）
            x_direction: 目标坐标系 X 轴方向（cam0 视线将对齐 -x_direction）
            y_direction: 目标坐标系 Y 轴方向（保留参数，实际由 cross(z_dir, x_dir) 确定）
            z_direction: 目标坐标系 Z 轴方向（平均相机朝上方向将对齐此方向）

        Returns:
            world_transform: 4x4 张量，符合 `transformCameras` 的行向量右乘约定
                             （即 Open3D ICP 矩阵的转置），可直接传入 transformCameras。
        """
        device = camera_list[0].device
        dtype = camera_list[0].dtype

        z_cam = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
        y_cam = torch.tensor([0., 1., 0.], dtype=dtype, device=device)

        # Gram-Schmidt 正交归一化目标坐标轴
        z_dir = torch.tensor(z_direction, dtype=dtype, device=device)
        z_dir = z_dir / (torch.linalg.norm(z_dir) + 1e-8)

        x_dir = torch.tensor(x_direction, dtype=dtype, device=device)
        x_dir = x_dir - torch.dot(x_dir, z_dir) * z_dir
        x_dir = x_dir / (torch.linalg.norm(x_dir) + 1e-8)

        def rot_from_a_to_b(a, b):
            """Rodrigues 旋转：将单位向量 a 旋转到 b，含反平行处理。"""
            v = torch.cross(a, b)
            c = torch.dot(a, b)
            s = torch.linalg.norm(v)
            if s < 1e-8:
                if c > 0:
                    return torch.eye(3, dtype=dtype, device=device)
                perp = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
                if torch.abs(torch.dot(a, perp)) > 0.9:
                    perp = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
                n = torch.cross(a, perp)
                n = n / torch.linalg.norm(n)
                return 2.0 * n[:, None] @ n[None, :] - torch.eye(3, dtype=dtype, device=device)
            vx = torch.tensor([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=dtype, device=device)
            return torch.eye(3, dtype=dtype, device=device) + vx + vx @ vx * ((1 - c) / (s * s))

        # =====================================================
        # Step 1: 求旋转 Q_up — 平均 image-Y 对齐 z_direction
        # =====================================================
        y_avg = torch.stack([cam.R.T @ y_cam for cam in camera_list]).mean(dim=0)
        y_avg = y_avg / (torch.linalg.norm(y_avg) + 1e-8)

        Q_up = rot_from_a_to_b(y_avg, z_dir)

        # =====================================================
        # Step 2: 求旋转 Q_yaw — cam0 视线对齐 -x_direction（绕 z_direction 轴）
        # 注：Q_up 作用后 cam0 的视线变为 Q_up @ look0_orig，
        #     这里直接用该等价形式，避免先真的旋转再读出
        # =====================================================
        look0_orig = -(camera_list[0].R.T @ z_cam)
        look0 = Q_up @ look0_orig

        look0_proj = look0 - torch.dot(look0, z_dir) * z_dir
        look0_proj = look0_proj / (torch.linalg.norm(look0_proj) + 1e-8)

        Q_yaw = rot_from_a_to_b(look0_proj, -x_dir)

        # 合并旋转右乘因子：R_new = R_old @ M
        M = Q_up.T @ Q_yaw.T

        # =====================================================
        # Step 3: 求平移 focus — 原始世界坐标系下视线的最近公共交点
        # 几何不变性：旋转后坐标系下的 focus = M^T @ focus_world
        # 代回 t_new = t_old + R_new @ focus_rotated 即得 t_new = t_old + R_old @ focus_world
        # =====================================================
        origins, dirs = [], []
        for cam in camera_list:
            C = -cam.R.T @ cam.t
            d = -(cam.R.T @ z_cam)
            d = d / (torch.linalg.norm(d) + 1e-8)
            origins.append(C)
            dirs.append(d)

        origins = torch.stack(origins)
        dirs = torch.stack(dirs)

        I = torch.eye(3, dtype=dtype, device=device)
        A = torch.zeros((3, 3), dtype=dtype, device=device)
        b = torch.zeros((3,), dtype=dtype, device=device)

        for o, d in zip(origins, dirs):
            P = I - d[:, None] @ d[None, :]
            A += P
            b += P @ o

        focus_world = torch.linalg.solve(A, b)

        # =====================================================
        # 直接构造 transformCameras 所需的行向量右乘矩阵 world_transform
        # （= Open3D ICP 矩阵的转置），避免任何数值求逆：
        #
        # transformCameras 内部对 w2c 最终的右乘量为
        #     T_clean_left_inv = [[R_l^T, -R_l^T @ t_l], [0, 1]]
        # 其中 (R_l, t_l) 由 world_transform.T 分解得到。
        #
        # 我们希望该最终量等于刚体 T_rigid = [[M, focus_world], [0, 1]]，
        # 反推得 R_l = M^T，t_l = -M^T @ focus_world，于是
        #     world_transform = [[M,             0],
        #                        [-focus_world^T @ M, 1]]
        # =====================================================
        world_transform = torch.eye(4, dtype=dtype, device=device)
        world_transform[:3, :3] = M
        world_transform[3, :3] = -(focus_world @ M)

        return world_transform

    @staticmethod
    def getPolarNormalizeTransform(
        camera_list: List[Camera],
        x_direction: list=[1, 0, 0],
        y_direction: list=[0, 1, 0],
        z_direction: list=[0, 0, 1],
        up_cluster_angle_deg: float=15.0,
        min_cluster_ratio: float=0.3,
    ) -> torch.Tensor:
        """
        与 `getZAxisNormalizeTransform` 语义一致，但对「主 up 方向」的估计改为
        基于球面方向聚类的最密集模态，而不是简单的算术平均：
        1. 收集所有相机在世界系下的 image-Y（up）方向；
        2. 在单位球面上寻找最密集的方向簇（最多邻居的候选），
           用该簇内方向的平均作为主 up 方向；
        3. 旋转使主 up 方向对齐 z_direction；
        4. 绕 z_direction 轴旋转使第一个相机视线对齐 -x_direction；
        5. 平移使所有视线射线的最近公共交点位于世界原点。

        这样可以抗离群视角，更符合「所有相机最集中的 up 方向」的直觉。

        Args:
            camera_list: 相机列表（只读）
            x_direction: 目标坐标系 X 轴方向（cam0 视线将对齐 -x_direction）
            y_direction: 目标坐标系 Y 轴方向（保留参数，实际由 cross(z_dir, x_dir) 确定）
            z_direction: 目标坐标系 Z 轴方向（主 up 方向将对齐此方向）
            up_cluster_angle_deg: 主簇的角度阈值（度），与候选中心夹角不大于此阈值视为同簇
            min_cluster_ratio: 主簇大小占相机总数的最低比例，低于此比例时回退到简单平均

        Returns:
            world_transform: 4x4 张量，符合 `transformCameras` 的行向量右乘约定
                             （即 Open3D ICP 矩阵的转置），可直接传入 transformCameras。
        """
        device = camera_list[0].device
        dtype = camera_list[0].dtype

        z_cam = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
        y_cam = torch.tensor([0., 1., 0.], dtype=dtype, device=device)

        z_dir = torch.tensor(z_direction, dtype=dtype, device=device)
        z_dir = z_dir / (torch.linalg.norm(z_dir) + 1e-8)

        x_dir = torch.tensor(x_direction, dtype=dtype, device=device)
        x_dir = x_dir - torch.dot(x_dir, z_dir) * z_dir
        x_dir = x_dir / (torch.linalg.norm(x_dir) + 1e-8)

        def rot_from_a_to_b(a, b):
            """Rodrigues 旋转：将单位向量 a 旋转到 b，含反平行处理。"""
            v = torch.cross(a, b)
            c = torch.dot(a, b)
            s = torch.linalg.norm(v)
            if s < 1e-8:
                if c > 0:
                    return torch.eye(3, dtype=dtype, device=device)
                perp = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
                if torch.abs(torch.dot(a, perp)) > 0.9:
                    perp = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
                n = torch.cross(a, perp)
                n = n / torch.linalg.norm(n)
                return 2.0 * n[:, None] @ n[None, :] - torch.eye(3, dtype=dtype, device=device)
            vx = torch.tensor([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=dtype, device=device)
            return torch.eye(3, dtype=dtype, device=device) + vx + vx @ vx * ((1 - c) / (s * s))

        # =====================================================
        # Step 1: 球面聚类估计主 up 方向
        # 每个相机在世界系下的 image-Y 方向即 cam.R.T @ y_cam，
        # 其球面表示 (phi, theta) 与 Camera.polar 同约定；这里
        # 直接在单位向量空间用余弦相似度做邻居计数，避免 atan2 在
        # 极点 / theta 卷绕带来的数值问题。
        # =====================================================
        ups_world = torch.stack([cam.R.T @ y_cam for cam in camera_list])
        ups_world = ups_world / (torch.linalg.norm(ups_world, dim=1, keepdim=True) + 1e-8)

        n_cams = ups_world.shape[0]

        cos_thresh = float(np.cos(np.deg2rad(up_cluster_angle_deg)))
        cos_matrix = ups_world @ ups_world.T  # (N, N)
        neighbor_mask = cos_matrix >= cos_thresh
        neighbor_count = neighbor_mask.sum(dim=1)

        best_idx = int(torch.argmax(neighbor_count).item())
        best_count = int(neighbor_count[best_idx].item())
        min_cluster_size = max(1, int(np.ceil(min_cluster_ratio * n_cams)))

        if best_count >= min_cluster_size:
            cluster_members = ups_world[neighbor_mask[best_idx]]
            u_main = cluster_members.mean(dim=0)
            if torch.linalg.norm(u_main) < 1e-6:
                # 理论上同簇方向夹角不超过阈值，平均不应退化；仍给回退。
                u_main = ups_world.mean(dim=0)
        else:
            # 视角过于分散，无显著主簇，回退到简单平均，
            # 与 getZAxisNormalizeTransform 保持一致。
            print('[WARN][CameraConvertor::getPolarNormalizeTransform]')
            print('\t no dominant up cluster found, falling back to mean.')
            print('\t best_count / n_cams:', best_count, '/', n_cams)
            u_main = ups_world.mean(dim=0)

        u_main = u_main / (torch.linalg.norm(u_main) + 1e-8)

        Q_up = rot_from_a_to_b(u_main, z_dir)

        # =====================================================
        # Step 2: 求 Q_yaw — cam0 视线对齐 -x_direction
        # =====================================================
        look0_orig = -(camera_list[0].R.T @ z_cam)
        look0 = Q_up @ look0_orig

        look0_proj = look0 - torch.dot(look0, z_dir) * z_dir
        proj_norm = torch.linalg.norm(look0_proj)
        if proj_norm < 1e-6:
            # cam0 视线与 z_dir 近乎共线，yaw 退化：此时绕 z_dir
            # 的任意旋转都不会让 cam0 的视线更接近 -x_dir，
            # 直接跳过。
            print('[WARN][CameraConvertor::getPolarNormalizeTransform]')
            print('\t cam0 forward is nearly parallel to z_direction, skip yaw alignment.')
            Q_yaw = torch.eye(3, dtype=dtype, device=device)
        else:
            look0_proj = look0_proj / (proj_norm + 1e-8)
            Q_yaw = rot_from_a_to_b(look0_proj, -x_dir)

        M = Q_up.T @ Q_yaw.T

        # =====================================================
        # Step 3: 求平移 focus — 原始世界坐标系下视线的最近公共交点
        # =====================================================
        origins, dirs = [], []
        for cam in camera_list:
            C = -cam.R.T @ cam.t
            d = -(cam.R.T @ z_cam)
            d = d / (torch.linalg.norm(d) + 1e-8)
            origins.append(C)
            dirs.append(d)

        origins = torch.stack(origins)
        dirs = torch.stack(dirs)

        I = torch.eye(3, dtype=dtype, device=device)
        A = torch.zeros((3, 3), dtype=dtype, device=device)
        b = torch.zeros((3,), dtype=dtype, device=device)

        for o, d in zip(origins, dirs):
            P = I - d[:, None] @ d[None, :]
            A += P
            b += P @ o

        focus_world = torch.linalg.solve(A, b)

        # =====================================================
        # 构造 transformCameras 所需的 world_transform
        # （与 getZAxisNormalizeTransform 完全一致的约定）
        # =====================================================
        world_transform = torch.eye(4, dtype=dtype, device=device)
        world_transform[:3, :3] = M
        world_transform[3, :3] = -(focus_world @ M)

        return world_transform

    @staticmethod
    def getXZPlaneNormalizeTransform(
        camera_list: List[Camera],
        x_direction: list=[1, 0, 0],
        y_direction: list=[0, 1, 0],
        z_direction: list=[0, 0, 1],
        up_cluster_angle_deg: float=15.0,
        min_cluster_ratio: float=0.3,
    ) -> torch.Tensor:
        """
        基于「世界 Z 轴大致位于每个相机的 Y(上)-Z(后) 平面内」的观察估计主 up 轴：
          若 z 位于相机 YZ 平面内，则 z 应与相机 X 轴（该平面的法线）在世界系下近似垂直。
          因此以 z_candidate = argmin_{||z||=1} sum_i (r_i · z)^2 估计主 up 轴，
          其中 r_i 为相机 i 在世界系下的右方向；该解为矩阵
              S = sum_i r_i r_i^T
          的最小特征值对应的特征向量。

        对「歪相机」的离群点处理：
          1. 先用所有相机的 up 方向在球面上做聚类，取最密集的主簇作为内点相机；
          2. 仅在内点相机的 right 向量上做最小特征向量拟合，避免倾斜相机把 z 拉偏。
          若主簇占比不足 min_cluster_ratio，则回退使用全部相机。

        其余对齐步骤与 getPolarNormalizeTransform 保持一致：
          - 旋转使 z_candidate 对齐 z_direction；
          - 绕 z_direction 旋转使 cam0 视线对齐 -x_direction；
          - 平移使所有视线射线的最近公共交点位于世界原点。

        Args:
            camera_list: 相机列表（只读）
            x_direction: 目标坐标系 X 轴方向（cam0 视线将对齐 -x_direction）
            y_direction: 目标坐标系 Y 轴方向（保留参数，实际由 cross(z_dir, x_dir) 确定）
            z_direction: 目标坐标系 Z 轴方向（主 up 方向将对齐此方向）
            up_cluster_angle_deg: up 方向聚类角度阈值（度），用于筛选非歪相机
            min_cluster_ratio: 主簇大小最低占比，低于该比例时回退为全量内点

        Returns:
            world_transform: 4x4 张量，符合 `transformCameras` 的行向量右乘约定
        """
        device = camera_list[0].device
        dtype = camera_list[0].dtype

        x_cam = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
        y_cam = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
        z_cam = torch.tensor([0., 0., 1.], dtype=dtype, device=device)

        z_dir = torch.tensor(z_direction, dtype=dtype, device=device)
        z_dir = z_dir / (torch.linalg.norm(z_dir) + 1e-8)

        x_dir = torch.tensor(x_direction, dtype=dtype, device=device)
        x_dir = x_dir - torch.dot(x_dir, z_dir) * z_dir
        x_dir = x_dir / (torch.linalg.norm(x_dir) + 1e-8)

        def rot_from_a_to_b(a, b):
            """Rodrigues 旋转：将单位向量 a 旋转到 b，含反平行处理。"""
            v = torch.cross(a, b)
            c = torch.dot(a, b)
            s = torch.linalg.norm(v)
            if s < 1e-8:
                if c > 0:
                    return torch.eye(3, dtype=dtype, device=device)
                perp = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
                if torch.abs(torch.dot(a, perp)) > 0.9:
                    perp = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
                n = torch.cross(a, perp)
                n = n / torch.linalg.norm(n)
                return 2.0 * n[:, None] @ n[None, :] - torch.eye(3, dtype=dtype, device=device)
            vx = torch.tensor([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=dtype, device=device)
            return torch.eye(3, dtype=dtype, device=device) + vx + vx @ vx * ((1 - c) / (s * s))

        # =====================================================
        # Step 1a: 用 up 方向球面聚类筛选内点相机（剔除"歪"相机）
        # =====================================================
        ups_world = torch.stack([cam.R.T @ y_cam for cam in camera_list])
        ups_world = ups_world / (torch.linalg.norm(ups_world, dim=1, keepdim=True) + 1e-8)

        n_cams = ups_world.shape[0]
        cos_thresh = float(np.cos(np.deg2rad(up_cluster_angle_deg)))
        cos_matrix = ups_world @ ups_world.T
        neighbor_mask = cos_matrix >= cos_thresh
        neighbor_count = neighbor_mask.sum(dim=1)

        best_idx = int(torch.argmax(neighbor_count).item())
        best_count = int(neighbor_count[best_idx].item())
        min_cluster_size = max(1, int(np.ceil(min_cluster_ratio * n_cams)))

        if best_count >= min_cluster_size:
            inlier_mask = neighbor_mask[best_idx]
        else:
            # 视角过于分散，没有明显"非歪"主簇，退化为全量相机。
            print('[WARN][CameraConvertor::getXZPlaneNormalizeTransform]')
            print('\t no dominant up cluster found, using all cameras as inliers.')
            print('\t best_count / n_cams:', best_count, '/', n_cams)
            inlier_mask = torch.ones(n_cams, dtype=torch.bool, device=device)

        # =====================================================
        # Step 1b: 内点相机上拟合 z_candidate —— sum_i r_i r_i^T 的最小特征向量
        # 几何含义：z_candidate 同时"最贴近"所有相机的 YZ 平面。
        # =====================================================
        rights_world = torch.stack([cam.R.T @ x_cam for cam in camera_list])
        rights_world = rights_world / (torch.linalg.norm(rights_world, dim=1, keepdim=True) + 1e-8)

        r_inliers = rights_world[inlier_mask]
        ups_inliers = ups_world[inlier_mask]

        if r_inliers.shape[0] < 2:
            # 内点过少，约束矩阵秩不足，退化为 up 均值。
            print('[WARN][CameraConvertor::getXZPlaneNormalizeTransform]')
            print('\t too few inliers for XZ-plane fitting, falling back to mean up.')
            u_main = ups_inliers.mean(dim=0) if ups_inliers.shape[0] > 0 else ups_world.mean(dim=0)
        else:
            S = r_inliers.T @ r_inliers
            S = 0.5 * (S + S.T)
            eigvals, eigvecs = torch.linalg.eigh(S)
            # eigvals 升序排列，最小特征值对应的特征向量即为 z_candidate。
            u_main = eigvecs[:, 0]

            # 特征向量仅在符号上有二义性；用内点 up 均值定号，
            # 保证 u_main 指向"多数相机头顶"方向。
            up_ref = ups_inliers.mean(dim=0)
            if torch.linalg.norm(up_ref) < 1e-6:
                up_ref = ups_world.mean(dim=0)
            if torch.dot(u_main, up_ref) < 0:
                u_main = -u_main

            # 记录最小残差，便于线上定位数据退化情况
            # （例如所有相机位姿共面会让最小特征值接近次小）。
            min_eig = float(eigvals[0].item())
            mid_eig = float(eigvals[1].item())
            if mid_eig > 1e-8 and min_eig / mid_eig > 0.3:
                print('[WARN][CameraConvertor::getXZPlaneNormalizeTransform]')
                print('\t smallest / middle eigenvalue ratio is high:', min_eig / mid_eig)
                print('\t z_candidate may be weakly constrained (cameras near-coplanar right-axes).')

        u_main = u_main / (torch.linalg.norm(u_main) + 1e-8)

        Q_up = rot_from_a_to_b(u_main, z_dir)

        # =====================================================
        # Step 2: 求 Q_yaw — cam0 视线对齐 -x_direction
        # =====================================================
        look0_orig = -(camera_list[0].R.T @ z_cam)
        look0 = Q_up @ look0_orig

        look0_proj = look0 - torch.dot(look0, z_dir) * z_dir
        proj_norm = torch.linalg.norm(look0_proj)
        if proj_norm < 1e-6:
            print('[WARN][CameraConvertor::getXZPlaneNormalizeTransform]')
            print('\t cam0 forward is nearly parallel to z_direction, skip yaw alignment.')
            Q_yaw = torch.eye(3, dtype=dtype, device=device)
        else:
            look0_proj = look0_proj / (proj_norm + 1e-8)
            Q_yaw = rot_from_a_to_b(look0_proj, -x_dir)

        M = Q_up.T @ Q_yaw.T

        # =====================================================
        # Step 3: 求平移 focus — 原始世界系下视线最近公共交点
        # =====================================================
        origins, dirs = [], []
        for cam in camera_list:
            C = -cam.R.T @ cam.t
            d = -(cam.R.T @ z_cam)
            d = d / (torch.linalg.norm(d) + 1e-8)
            origins.append(C)
            dirs.append(d)

        origins = torch.stack(origins)
        dirs = torch.stack(dirs)

        I = torch.eye(3, dtype=dtype, device=device)
        A = torch.zeros((3, 3), dtype=dtype, device=device)
        b = torch.zeros((3,), dtype=dtype, device=device)

        for o, d in zip(origins, dirs):
            P = I - d[:, None] @ d[None, :]
            A += P
            b += P @ o

        focus_world = torch.linalg.solve(A, b)

        world_transform = torch.eye(4, dtype=dtype, device=device)
        world_transform[:3, :3] = M
        world_transform[3, :3] = -(focus_world @ M)

        return world_transform

    @staticmethod
    def _process_camera_for_pcd(
        camera: Camera,
        conf_thresh: float,
        use_mask: bool,
        mask_smaller_pixel_num: int,
    ):
        mask = camera.toDepthMask(conf_thresh, use_mask, mask_smaller_pixel_num)
        points, _ = camera.toDepthPoints(conf_thresh, use_mask, mask_smaller_pixel_num)

        depth_uv = None

        colors = None
        if getattr(camera, "image", None) is not None:
            depth_uv = camera.toDepthUV()
            image_colors = camera.sampleRGBAtUV(depth_uv)
            colors = image_colors[mask]

        normals = None
        if getattr(camera, "normal_world", None) is not None:
            if depth_uv is None:
                depth_uv = camera.toDepthUV()
            sampled_normals = camera.sampleNormalWorldAtUV(depth_uv)
            normals = sampled_normals[mask]

        return points, normals, colors

    @staticmethod
    def createDepthPcd(
        camera_list: List[Camera],
        conf_thresh: float=0.8,
        use_mask: bool=True,
        mask_smaller_pixel_num: int=0,
    ) -> o3d.geometry.PointCloud:
        print('[INFO][CameraConvertor::createDepthPcd]')
        print('\t start create depth pcd...')
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    CameraConvertor._process_camera_for_pcd,
                    cam,
                    conf_thresh,
                    use_mask,
                    mask_smaller_pixel_num,
                )
                for cam in camera_list
            ]
            results = [
                f.result()
                for f in tqdm(futures)
            ]
        points_list = [r[0] for r in results]
        normals_list = [r[1] for r in results]
        colors_list = [r[2] for r in results]

        points = torch.cat(points_list, dim=0)
        # 仅当所有相机都具备对应模态时才聚合，避免点云与颜色/法线长度不一致。
        colors = (
            torch.cat(colors_list, dim=0)
            if all(c is not None for c in colors_list)
            else None
        )
        normals = (
            torch.cat(normals_list, dim=0)
            if all(n is not None for n in normals_list)
            else None
        )

        pcd = toPcd(points, colors, normals)
        return pcd

    @staticmethod
    def createColmapDataFolder(
        cameras: List[Camera],
        save_data_folder_path: str,
        pcd: Union[trimesh.Trimesh, o3d.geometry.PointCloud, torch.Tensor, np.ndarray, list, str, None]=None,
        point_num_max: Optional[int]=None,
    ) -> bool:
        """
        创建用于训练3DGS的COLMAP格式数据文件夹

        生成的数据结构：
        save_data_folder_path/
        ├── images/           # 渲染的图像
        │   ├── 00000.png
        │   ├── 00001.png
        │   └── ...
        └── sparse/
            └── 0/
                ├── cameras.txt   # 相机内参 (PINHOLE模型)
                ├── images.txt    # 图像外参 (四元数 + 平移)
                └── points3D.ply  # 3D点云

        坐标系转换说明：
        - 原始相机坐标系 (camera.py): X右，Y上，Z后（相机看向 -Z 方向）
        - COLMAP相机坐标系: X右，Y下，Z前（相机看向 +Z 方向）
        - 世界坐标系保持不变（与mesh坐标系一致）
        - 只转换相机坐标系，不转换世界坐标系
        """
        if pcd is not None:
            if isinstance(pcd, str):
                if not os.path.exists(pcd):
                    print('[ERROR][CameraConvertor::createColmapDataFolder]')
                    print('\t pcd file not exist!')
                    print('\t pcd:', pcd)
                    return False

                pcd = o3d.io.read_point_cloud(pcd)
            elif isinstance(pcd, trimesh.Trimesh):
                pcd = toPcd(pcd.vertices)
            elif isinstance(pcd, torch.Tensor) or isinstance(pcd, np.ndarray) or isinstance(pcd, list):
                points = toNumpy(pcd).reshape(-1, 3)
                pcd = toPcd(points)

            if not isinstance(pcd, o3d.geometry.PointCloud):
                print('[ERROR][CameraConvertor::createColmapDataFolder]')
                print('\t pcd is not o3d.geometry.PointCloud!')
                return False

            if point_num_max is not None:
                point_num = len(pcd.points)
                if point_num > point_num_max:
                    sample_ratio = point_num_max / point_num
                    pcd = pcd.random_down_sample(sample_ratio)

            # 检查pcd，如果没有颜色则全部赋值为[128,128,128]
            if not pcd.has_colors():
                colors = np.tile(np.array([[128, 128, 128]], dtype=np.float64) / 255.0, (np.asarray(pcd.points).shape[0], 1))
                pcd.colors = o3d.utility.Vector3dVector(colors)
            # 如果没有法向，估计法向
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        camera_num = len(cameras)
        height, width = cameras[0].image.shape[:2]
        fx = cameras[0].fx
        fy = cameras[0].fy
        cx = cameras[0].cx
        cy = cameras[0].cy

        # 创建文件夹结构
        if save_data_folder_path[-1] != '/':
            save_data_folder_path += '/'

        if os.path.exists(save_data_folder_path):
            rmtree(save_data_folder_path)

        image_folder_path = save_data_folder_path + 'images/'

        mask_folder_path = save_data_folder_path + 'masks/'
        masked_image_folder_path = save_data_folder_path + 'masked_images/'

        depth_folder_path = save_data_folder_path + 'depths/'
        depth_vis_folder_path = save_data_folder_path + 'depths_vis/'
        masked_depth_vis_folder_path = save_data_folder_path + 'masked_depths_vis/'

        normal_world_folder_path = save_data_folder_path + 'normal_worlds/'
        normal_world_vis_folder_path = save_data_folder_path + 'normal_worlds_vis/'
        masked_normal_world_vis_folder_path = save_data_folder_path + 'masked_normal_worlds_vis/'

        normal_camera_folder_path = save_data_folder_path + 'normal_cameras/'
        normal_camera_vis_folder_path = save_data_folder_path + 'normal_cameras_vis/'
        masked_normal_camera_vis_folder_path = save_data_folder_path + 'masked_normal_cameras_vis/'

        sparse_folder_path = save_data_folder_path + 'sparse/0/'

        os.makedirs(image_folder_path, exist_ok=True)

        os.makedirs(mask_folder_path, exist_ok=True)
        os.makedirs(masked_image_folder_path, exist_ok=True)

        os.makedirs(depth_folder_path, exist_ok=True)
        os.makedirs(depth_vis_folder_path, exist_ok=True)
        os.makedirs(masked_depth_vis_folder_path, exist_ok=True)

        os.makedirs(normal_world_folder_path, exist_ok=True)
        os.makedirs(normal_world_vis_folder_path, exist_ok=True)
        os.makedirs(masked_normal_world_vis_folder_path, exist_ok=True)

        os.makedirs(normal_camera_folder_path, exist_ok=True)
        os.makedirs(normal_camera_vis_folder_path, exist_ok=True)
        os.makedirs(masked_normal_camera_vis_folder_path, exist_ok=True)

        os.makedirs(sparse_folder_path, exist_ok=True)

        # 准备cameras.txt内容
        # COLMAP格式: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # PINHOLE模型参数: fx, fy, cx, cy
        cameras_txt_lines = [
            "# Camera list with one line of data per camera:\n",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
            f"# Number of cameras: 1\n",
            f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n"
        ]

        # 准备images.txt内容
        # COLMAP格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # 然后是一行2D点（可以为空）
        images_txt_lines = [
            "# Image list with two lines of data per image:\n",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
            f"# Number of images: {camera_num}\n",
        ]

        def _process_one_camera(camera_idx):
            camera = cameras[camera_idx]
            image_filename = camera.image_id
            format = '.' + image_filename.split('.')[-1]
            image_basename = image_filename.split(format)[0]

            colmap_pose = camera.toColmapPose().cpu().numpy()

            cv2.imwrite(image_folder_path + image_filename, camera.toImageCV(use_mask=False))

            if camera.mask is not None:
                cv2.imwrite(mask_folder_path + image_filename, camera.toMaskCV())
                cv2.imwrite(masked_image_folder_path + image_filename, camera.toImageCV(use_mask=True))

            if camera.depth is not None:
                np.save(depth_folder_path + image_basename + '.npy', camera.depth_with_conf.cpu().numpy())
                cv2.imwrite(depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=False))
                cv2.imwrite(masked_depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=True))

            if camera.normal_world is not None:
                np.save(normal_world_folder_path + image_basename + '.npy', camera.normal_world.cpu().numpy())
                cv2.imwrite(normal_world_vis_folder_path + image_filename, camera.toNormalWorldVisCV(use_mask=False))
                cv2.imwrite(masked_normal_world_vis_folder_path + image_filename, camera.toNormalWorldVisCV(use_mask=True))

            if camera.normal_camera is not None:
                np.save(normal_camera_folder_path + image_basename + '.npy', camera.normal_camera.cpu().numpy())
                cv2.imwrite(normal_camera_vis_folder_path + image_filename, camera.toNormalCameraVisCV(use_mask=False))
                cv2.imwrite(masked_normal_camera_vis_folder_path + image_filename, camera.toNormalCameraVisCV(use_mask=True))

            return (camera_idx, image_filename, colmap_pose)

        print('[INFO][MeshRenderer::createColmapDataFolder]')
        print('\t start create colmap data folder...')
        # 预热 torch.linalg，避免多线程并发触发懒加载导致
        # "lazy wrapper should be called at most once" 错误
        torch.linalg.eigh(torch.zeros((1, 1), dtype=cameras[0].dtype, device=cameras[0].device))
        max_workers = min(32, (os.cpu_count() or 4) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(_process_one_camera, range(camera_num)), total=camera_num, desc='colmap data'))

        for (camera_idx, image_filename, colmap_pose) in results:
            images_txt_lines.append(
                f"{camera_idx + 1} {colmap_pose[0]:.10f} {colmap_pose[1]:.10f} {colmap_pose[2]:.10f} {colmap_pose[3]:.10f} "
                f"{colmap_pose[4]:.10f} {colmap_pose[5]:.10f} {colmap_pose[6]:.10f} 1 {image_filename}\n"
            )
            images_txt_lines.append("\n")

        # 写入cameras.txt
        with open(sparse_folder_path + 'cameras.txt', 'w') as f:
            f.writelines(cameras_txt_lines)

        # 写入images.txt
        with open(sparse_folder_path + 'images.txt', 'w') as f:
            f.writelines(images_txt_lines)

        # 生成points3D.ply点云文件
        if pcd is not None:
            print('\t generating points3D.ply from points...')
            ply_path = sparse_folder_path + 'points3D.ply'
            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)

        print(f'\t saved to: {save_data_folder_path}')
        print(f'\t total images: {camera_num}')
        if pcd is not None:
            print(f'\t total points: {len(pcd.points)}')
        return True

    @staticmethod
    def loadColmapDataFolder(
        colmap_data_folder_path: str,
        image_folder_name: str='images',
        mask_folder_name: str='masks',
        depth_folder_name: str='depths',
        normal_folder_name: str='normals',
    ) -> List[Camera]:
        """
        从 COLMAP 数据目录加载相机列表。
        cameras.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] (PINHOLE: fx, fy, cx, cy)
        images.txt: 每张图两行，第一行 IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        """
        if not colmap_data_folder_path.endswith('/'):
            colmap_data_folder_path += '/'
        image_folder_path = colmap_data_folder_path + image_folder_name + '/'
        mask_folder_path = colmap_data_folder_path + mask_folder_name + '/'
        depth_folder_path = colmap_data_folder_path + depth_folder_name + '/'
        normal_folder_path = colmap_data_folder_path + normal_folder_name + '/'

        camera_intrinsic_file_path = colmap_data_folder_path + 'sparse/0/cameras.txt'
        camera_extrinsic_file_path = colmap_data_folder_path + 'sparse/0/images.txt'

        if not os.path.exists(image_folder_path):
            print('[ERROR][CameraConvertor::loadColmapDataFolder] image folder not exist:', image_folder_path)
            return []
        if not os.path.exists(camera_intrinsic_file_path):
            print('[ERROR][CameraConvertor::loadColmapDataFolder] cameras.txt not exist:', camera_intrinsic_file_path)
            return []
        if not os.path.exists(camera_extrinsic_file_path):
            print('[ERROR][CameraConvertor::loadColmapDataFolder] images.txt not exist:', camera_extrinsic_file_path)
            return []

        # 解析 cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy (PINHOLE)
        cameras_dict = {}
        with open(camera_intrinsic_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                if model == 'PINHOLE' and len(parts) >= 8:
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    cameras_dict[camera_id] = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

        # 解析 images.txt: 每张图两行，第一行 IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        image_records = []
        with open(camera_extrinsic_file_path, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            image_records.append({
                'image_id': image_id,
                'pose': [qw, qx, qy, qz, tx, ty, tz],
                'camera_id': camera_id,
                'name': name,
            })
            if i < len(lines):
                i += 1  # 跳过 POINTS2D 行

        def _process_one_record(rec):
            """处理单条图像记录，返回 Camera 或 None（失败时）。"""
            camera_id = rec['camera_id']
            image_filename = rec['name']
            format = '.' + image_filename.split('.')[-1]
            image_basename = image_filename.split(format)[0]

            if camera_id not in cameras_dict:
                print('[WARN][CameraConvertor::loadColmapDataFolder] unknown camera_id:', camera_id, 'skip image', image_filename)
                return None
            cam_info = cameras_dict[camera_id]
            intrinsic = np.array([
                [cam_info['fx'], 0, cam_info['cx']],
                [0, cam_info['fy'], cam_info['cy']],
                [0, 0, 1],
            ], dtype=np.float64)
            pose = rec['pose']
            camera = Camera.fromColmapPose(pose, intrinsic)

            image_file_path = image_folder_path + image_filename
            if not camera.loadImageFile(image_file_path):
                print('[WARN][CameraConvertor::loadColmapDataFolder]')
                print('\t loadImageFile failed!')
                print('\t image file path:', image_file_path)
                return None

            mask_file_path = mask_folder_path + image_filename
            if os.path.exists(mask_file_path):
                if not camera.loadMaskFile(mask_file_path):
                    print('[WARN][CameraConvertor::loadColmapDataFolder]')
                    print('\t loadMaskFile failed!')
                    print('\t mask file path:', mask_file_path)

            depth_file_path = depth_folder_path + image_basename + '.npy'
            if os.path.exists(depth_file_path):
                if not camera.loadDepthFile(depth_file_path):
                    print('[WARN][CameraConvertor::loadColmapDataFolder]')
                    print('\t loadDepthFile failed!')
                    print('\t depth file path:', depth_file_path)

            normal_file_path = normal_folder_path + image_filename
            if os.path.exists(normal_file_path):
                if not camera.loadNormalWorldFile(normal_file_path):
                    print('[WARN][CameraConvertor::loadColmapDataFolder]')
                    print('\t loadNormalWorldFile failed!')
                    print('\t normal file path:', normal_file_path)

            camera.image_id = image_filename
            return camera

        print('[INFO][CameraConvertor::loadColmapDataFolder]')
        print('\t start load colmap data...')
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(_process_one_record, image_records),
                total=len(image_records),
                desc='load colmap',
            ))
        camera_list = [c for c in results if isinstance(c, Camera)]
        camera_list.sort(key=lambda c: c.image_id)
        return camera_list
