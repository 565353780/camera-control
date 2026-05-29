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

from camera_control.Method.pcd import toPcd, toTrimeshPcd
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
    def getCamerasSphere(
        camera_list: List[Camera],
    ) -> torch.Tensor:
        """
        估计能够最优匹配所有相机布局的球的球心与半径。

        几何模型：
            理想的 orbit 阵列满足 p_i = c + r * b_i，其中
              - p_i 为相机 i 在世界系下的位置；
              - b_i 为相机 i 的"后方向"（即视线方向的反方向）的单位向量，
                等价于相机的极坐标方向；
              - c 为球心，r 为球半径。
            这使得 (c, r) 共 4 个未知数对每个相机贡献 3 个等式，
            可用线性最小二乘闭式求解。

        策略（鲁棒）：
            1. 主路径——线性最小二乘：相机数 ≥ 2 且后方向有足够的角散度时，
               解法方程 A x = rhs，x = [c; r]。检查 r 是否为正以及残差是否合理，
               并在显著外点存在时做一次 MAD 阈值 inlier refit。
            2. 退化路径——相机仅位于一侧或几乎同向时，半径不可由位置 + 方向
               的线性约束观测：
               - 取后方向均值 n_dir 为"垂直相机朝向的平均平面"的法线；
               - 平均平面过相机位置均值，相机在平面上的 2D 投影点构造
                 凸包并取多边形质心，作为球心在该平面上的投影；
               - 用相机到该投影中心的稳健距离（中位数）作为原始半径；
               - 球心 = 投影中心 - raw_radius * n_dir，使球心位于相机
                 看向的一侧、距平均平面 raw_radius 的位置（这恰好等价于
                 用户所述："球心距平均平面为 sphere_radius，平均平面投影
                 凸包中心为球心投影"，差一个全局尺度因子，由后续 normalize
                 完成）。

        Args:
            camera_list: 相机列表（至少 1 个）。

        Returns:
            长度为 4 的张量 [cx, cy, cz, r]，dtype/device 与首相机一致；
            r > 0。
        """
        if len(camera_list) == 0:
            raise ValueError(
                '[CameraConvertor::getCamerasSphere] camera_list is empty.'
            )

        device = camera_list[0].device
        dtype = camera_list[0].dtype

        z_cam = torch.tensor([0., 0., 1.], dtype=dtype, device=device)

        # =====================================================
        # 收集相机位置 p_i 和单位后方向 b_i = R_i^T @ z_cam
        # （与 Camera.forwardDirection = -R[2] 保持一致：b_i = -forward）
        # =====================================================
        centers = torch.stack([-cam.R.T @ cam.t for cam in camera_list])  # (N, 3)
        back_dirs = torch.stack([cam.R.T @ z_cam for cam in camera_list])  # (N, 3)
        back_dirs = back_dirs / (
            torch.linalg.norm(back_dirs, dim=1, keepdim=True) + 1e-8
        )

        n_cams = centers.shape[0]

        # =====================================================
        # 评估后方向的"散度"，用于判断主路径是否可用：
        #   M_back = sum_i (b_i - mean) (b_i - mean)^T
        # 其最大特征值 / N ≈ 单位向量到均值的 RMS 偏差平方。
        # 阈值取约 sin^2(5°) ≈ 0.0076，更小则视为相机视线近乎共线，
        # 此时半径列不可观测。
        # =====================================================
        mean_back = back_dirs.mean(dim=0)
        centered_back = back_dirs - mean_back[None, :]
        M_back = centered_back.T @ centered_back
        M_back = 0.5 * (M_back + M_back.T)
        eig_back = torch.linalg.eigvalsh(M_back)
        spread = float(eig_back[-1].item()) / max(1.0, float(n_cams))
        spread_threshold = 0.0076

        sphere_center: Optional[torch.Tensor] = None
        sphere_radius_est: Optional[float] = None

        if n_cams >= 2 and spread > spread_threshold:
            # =====================================================
            # 主路径：线性最小二乘闭式解
            #   每相机贡献等式 c + r * b_i = p_i（3 个标量）。
            #   将 [c; r] 记为 4 维向量 x，A_i = [I_3 | b_i]。
            #   法方程 (sum A_i^T A_i) x = sum A_i^T p_i 展开为：
            #     [N I_3       sum b_i ] [c] = [sum p_i        ]
            #     [sum b_i^T   sum |b|^2] [r]   [sum b_i . p_i  ]
            #   其中 sum |b|^2 = N（单位向量）。
            # =====================================================
            I3 = torch.eye(3, dtype=dtype, device=device)
            sum_b = back_dirs.sum(dim=0)
            N_t = torch.tensor(float(n_cams), dtype=dtype, device=device)

            A_mat = torch.zeros((4, 4), dtype=dtype, device=device)
            A_mat[:3, :3] = N_t * I3
            A_mat[:3, 3] = sum_b
            A_mat[3, :3] = sum_b
            A_mat[3, 3] = N_t

            rhs = torch.zeros((4,), dtype=dtype, device=device)
            rhs[:3] = centers.sum(dim=0)
            rhs[3] = (centers * back_dirs).sum()

            try:
                x = torch.linalg.solve(A_mat, rhs)
                c_lin = x[:3]
                r_lin = float(x[3].item())
            except Exception:
                c_lin = centers.mean(dim=0)
                r_lin = -1.0

            if r_lin > 0:
                # MAD inlier refit，抑制少量外点。
                residuals = torch.linalg.norm(
                    centers - (c_lin[None, :] + r_lin * back_dirs), dim=1
                )
                med = torch.median(residuals)
                mad = torch.median(torch.abs(residuals - med)).clamp(min=1e-8)
                inlier_mask = residuals <= med + 3.0 * 1.4826 * mad
                n_in = int(inlier_mask.sum().item())

                if n_in >= max(2, int(0.5 * n_cams)) and n_in < n_cams:
                    cb = centers[inlier_mask]
                    bb = back_dirs[inlier_mask]
                    Nn_t = torch.tensor(float(n_in), dtype=dtype, device=device)
                    sum_b2 = bb.sum(dim=0)

                    A2 = torch.zeros((4, 4), dtype=dtype, device=device)
                    A2[:3, :3] = Nn_t * I3
                    A2[:3, 3] = sum_b2
                    A2[3, :3] = sum_b2
                    A2[3, 3] = Nn_t

                    rhs2 = torch.zeros((4,), dtype=dtype, device=device)
                    rhs2[:3] = cb.sum(dim=0)
                    rhs2[3] = (cb * bb).sum()

                    try:
                        x2 = torch.linalg.solve(A2, rhs2)
                        if float(x2[3].item()) > 0:
                            c_lin = x2[:3]
                            r_lin = float(x2[3].item())
                    except Exception:
                        pass

                sphere_center = c_lin
                sphere_radius_est = r_lin
            else:
                # 主路径解出非正半径，说明几何与 orbit 模型不符，转退化路径。
                print('[WARN][CameraConvertor::getCamerasSphere]')
                print('\t linear sphere fit yields non-positive radius:', r_lin)
                print('\t falling back to projected-plane estimation.')

        if sphere_center is None or sphere_radius_est is None:
            # =====================================================
            # 退化路径：相机视线集中或相机数过少
            # =====================================================
            mean_back_norm = torch.linalg.norm(mean_back)

            if mean_back_norm > 1e-6:
                n_dir = mean_back / mean_back_norm
            else:
                # 极少见：方向均值近 0 但又没通过散度阈值；用相机位置 PCA
                # 最小特征向量当法线，并按多数后方向定号。
                pos_mean = centers.mean(dim=0)
                centers_zm = centers - pos_mean[None, :]
                cov_p = centers_zm.T @ centers_zm
                cov_p = 0.5 * (cov_p + cov_p.T)
                _, eigvecs_p = torch.linalg.eigh(cov_p)
                n_dir = eigvecs_p[:, 0]
                # 任意定号：让 n_dir 与多数 back_dir 同侧
                if (back_dirs @ n_dir).sum() < 0:
                    n_dir = -n_dir

            n_dir = n_dir / (torch.linalg.norm(n_dir) + 1e-8)

            plane_point = centers.mean(dim=0)

            # 投影到过 plane_point、法向 n_dir 的平面
            offsets = centers - plane_point[None, :]
            depths = offsets @ n_dir  # (N,)
            centers_planar = centers - depths[:, None] * n_dir[None, :]

            # 平面 2D 正交基
            e_axis = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
            if torch.abs(torch.dot(e_axis, n_dir)) > 0.9:
                e_axis = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
            u_axis = e_axis - torch.dot(e_axis, n_dir) * n_dir
            u_axis = u_axis / (torch.linalg.norm(u_axis) + 1e-8)
            v_axis = torch.linalg.cross(n_dir, u_axis)
            v_axis = v_axis / (torch.linalg.norm(v_axis) + 1e-8)

            uv = torch.stack([
                (centers_planar - plane_point[None, :]) @ u_axis,
                (centers_planar - plane_point[None, :]) @ v_axis,
            ], dim=1)  # (N, 2)

            centroid_uv = CameraConvertor._convexHullCentroid2D(uv)

            centroid_3d = (
                plane_point
                + centroid_uv[0] * u_axis
                + centroid_uv[1] * v_axis
            )

            # 原始半径：相机到 centroid_3d 的 3D 距离中位数。
            # 这同时利用了平面上的散度和（若有）沿 n_dir 的微小深度变化。
            dists = torch.linalg.norm(centers - centroid_3d[None, :], dim=1)
            raw_radius = float(torch.median(dists).item())

            if raw_radius < 1e-6:
                print('[WARN][CameraConvertor::getCamerasSphere]')
                print('\t cameras nearly coincide, using unit fallback radius.')
                raw_radius = 1.0

            # 球心位于相机看向的一侧（即 -n_dir 方向），距平均平面 raw_radius。
            sphere_center = centroid_3d - raw_radius * n_dir
            sphere_radius_est = raw_radius

        radius_tensor = torch.tensor(
            [float(sphere_radius_est)], dtype=dtype, device=device
        )
        return torch.cat([sphere_center, radius_tensor])

    @staticmethod
    def _convexHullCentroid2D(uv: torch.Tensor) -> torch.Tensor:
        """
        计算 2D 点集凸包多边形的（按面积加权的）质心。

        - n = 0：抛错；
        - n = 1：返回该点；
        - n = 2 或所有点共线：返回点集均值（线段中点等价）；
        - n ≥ 3：用 Andrew's monotone chain 求凸包，再用 shoelace 公式
          求多边形质心（不是顶点平均）。

        Args:
            uv: (N, 2) 张量

        Returns:
            (2,) 张量，与 uv 同 dtype/device
        """
        n = uv.shape[0]
        if n == 0:
            raise ValueError(
                '[CameraConvertor::_convexHullCentroid2D] empty point set.'
            )
        if n == 1:
            return uv[0].clone()
        if n == 2:
            return uv.mean(dim=0)

        pts_np = toNumpy(uv, np.float64)

        # 字典序排序（先 x 后 y）
        order = np.lexsort((pts_np[:, 1], pts_np[:, 0]))
        sorted_pts = pts_np[order]

        # 去重，避免 monotone chain 退化
        unique_pts = [sorted_pts[0]]
        for p in sorted_pts[1:]:
            if not np.allclose(p, unique_pts[-1], atol=1e-12):
                unique_pts.append(p)
        sorted_pts = np.array(unique_pts)

        if sorted_pts.shape[0] < 3:
            return uv.mean(dim=0)

        def cross_2d(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[np.ndarray] = []
        for p in sorted_pts:
            while (
                len(lower) >= 2
                and cross_2d(lower[-2], lower[-1], p) <= 0
            ):
                lower.pop()
            lower.append(p)

        upper: List[np.ndarray] = []
        for p in sorted_pts[::-1]:
            while (
                len(upper) >= 2
                and cross_2d(upper[-2], upper[-1], p) <= 0
            ):
                upper.pop()
            upper.append(p)

        hull = np.array(lower[:-1] + upper[:-1])

        if hull.shape[0] < 3:
            # 共线退化：用沿主轴投影的中点更稳，等价于 sorted_pts 的 bbox 中心。
            mid = 0.5 * (sorted_pts.min(axis=0) + sorted_pts.max(axis=0))
            return torch.tensor(mid, dtype=uv.dtype, device=uv.device)

        # Shoelace 多边形质心
        x = hull[:, 0]
        y = hull[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross_terms = x * y_next - x_next * y
        area2 = float(cross_terms.sum())  # = 2 * 有符号面积

        if abs(area2) < 1e-12:
            mid = 0.5 * (hull.min(axis=0) + hull.max(axis=0))
            return torch.tensor(mid, dtype=uv.dtype, device=uv.device)

        cx = float(((x + x_next) * cross_terms).sum() / (3.0 * area2))
        cy = float(((y + y_next) * cross_terms).sum() / (3.0 * area2))
        return torch.tensor([cx, cy], dtype=uv.dtype, device=uv.device)

    @staticmethod
    def getPolarScaleNormalizeTransform(
        camera_list: List[Camera],
        sphere_radius: float=4.0,
    ) -> torch.Tensor:
        """
        基于相机阵列的最优拟合球，构造平移 + 均匀缩放的归一化变换：
          - 将估计球心移到世界原点；
          - 将估计球半径缩放为 `sphere_radius`。

        球心和半径由 `getCamerasSphere` 估计：理想 orbit 输入会得到
        紧致的闭式解；当相机仅分布于一侧或几乎同向时，球心定义为
        "距相机平均平面为 sphere_radius、且其在平面上的投影为相机
        投影凸包质心"的点，由该函数已经自动处理。

        返回的 4x4 张量与 `transformCameras` 的行向量右乘约定一致
        （即 Open3D ICP 矩阵的转置），可直接传入 `transformCameras`。

        构造原理（与 `getZAxisNormalizeTransform` 同套约定）：
          列形式左乘矩阵 T_left = [[s I, -s c], [0, 1]]，
          其中 c 为原始世界系下的球心，s = sphere_radius / r 为均匀缩放。
          故 world_transform = T_left^T 满足：
              world_transform[:3, :3] = s I,
              world_transform[3, :3]  = -s c,
              world_transform[3, 3]   = 1.

        Args:
            camera_list: 相机列表（只读）
            sphere_radius: 归一化后的目标球半径，必须为正

        Returns:
            world_transform: 4x4 张量
        """
        if len(camera_list) == 0:
            raise ValueError(
                '[CameraConvertor::getPolarScaleNormalizeTransform]'
                ' camera_list is empty.'
            )
        if sphere_radius <= 0:
            raise ValueError(
                '[CameraConvertor::getPolarScaleNormalizeTransform]'
                ' sphere_radius must be positive, got {}.'.format(sphere_radius)
            )

        device = camera_list[0].device
        dtype = camera_list[0].dtype

        sphere = CameraConvertor.getCamerasSphere(camera_list)
        sphere_center = sphere[:3]
        raw_radius = float(sphere[3].item())

        if raw_radius <= 1e-8:
            # getCamerasSphere 已做退化保护，这里再加一道防御。
            print('[WARN][CameraConvertor::getPolarScaleNormalizeTransform]')
            print('\t estimated raw radius is degenerate:', raw_radius)
            raw_radius = 1.0

        scale = float(sphere_radius) / raw_radius
        scale_t = torch.tensor(scale, dtype=dtype, device=device)

        world_transform = torch.eye(4, dtype=dtype, device=device)
        world_transform[:3, :3] = scale_t * torch.eye(3, dtype=dtype, device=device)
        world_transform[3, :3] = -scale_t * sphere_center

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
    def getBestCameraPose(
        camera: Camera,
        pts: np.ndarray,
        safe_pixel_num: int = 10,
    ) -> np.ndarray:
        """求固定朝向 + 固定内参下，仅平移相机得到的最佳视角。

        几何构造：
          相机的有效图像区域被 safe_pixel_num 收缩到
              [u_min, u_max] x [v_min, v_max]
          对应一个新的视锥（四棱锥），它的顶点在相机位置 C，四个侧面分别由
          四条像素边界对应的射线扫成，所以四个侧面都过 C，只需法向即可写出
          平面方程。
          以相机系下深度 D = -Z（朝向 -Z）为单位，四条边界的射线斜率：
              k_left   = (u_min - cx) / fx       (典型 < 0，左侧)
              k_right  = (u_max - cx) / fx       (典型 > 0，右侧)
              k_bottom = (v_min - cy) / fy       (典型 < 0，下方)
              k_top    = (v_max - cy) / fy       (典型 > 0，上方)
          相机系下，指向视锥内部的四个侧面法向：
              n_L_cam = ( 1, 0,  k_left )    左面：把点推向右
              n_R_cam = (-1, 0, -k_right)    右面：把点推向左
              n_B_cam = ( 0, 1,  k_bottom)   下面：把点推向上
              n_T_cam = ( 0,-1, -k_top   )   上面：把点推向下
          在世界坐标系下（用 right = R[0], up = R[1], forward = -R[2]
          作为基），等价于：
              N_L = right - k_left * forward
              N_R = -right + k_right * forward
              N_B = up - k_bottom * forward
              N_T = -up + k_top * forward
          点 P 在该侧面对应的半空间内的条件是 N · (P - C) >= 0。
          把 C 写成 C = cr * right + cu * up + cf * forward，并定义每个点的
          投影坐标 x_i = P_i · right, y_i = P_i · up, s_i = P_i · forward，
          则四个侧面的半空间约束等价于：
              k_left   * (s_i - cf) <= x_i - cr <= k_right * (s_i - cf)
              k_bottom * (s_i - cf) <= y_i - cu <= k_top   * (s_i - cf)
              s_i - cf > 0  (点位于相机前方)

        求解策略（"可行的几何最优"）：
          仅平移有 3 自由度，一般无法让点云同时与四个侧面相切。
          1) 沿 forward 方向把相机推到最近：cf 取尽可能大，直到水平/竖直
             两组对面中至少一组同时相切（"binding axis"）。
          2) 另一方向（"non-binding axis"）保留余量，再在余量内部移动相机
             使点云投影 bbox 在 safe 区域内尽量居中。
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            raise ValueError('pts must be a non-empty Nx3 array.')

        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.shape[0] == 0:
            raise ValueError('pts contains no finite points.')

        safe = float(safe_pixel_num)
        if safe < 0:
            raise ValueError('safe_pixel_num must be non-negative.')

        width = float(camera.width)
        height = float(camera.height)
        u_min = safe
        u_max = width - 1.0 - safe
        v_min = safe
        v_max = height - 1.0 - safe
        if u_min > u_max or v_min > v_max:
            raise ValueError('safe_pixel_num leaves no valid image area.')

        def _normalize(vec: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                raise ValueError('Cannot normalize near-zero camera direction.')
            return vec / norm

        # =============================================================
        # Step 1: 取相机朝向的世界基 (right, up, forward)
        # =============================================================
        forward = _normalize(
            camera.forwardDirection.detach().cpu().numpy().astype(np.float64)
        )
        raw_up = camera.upDirection.detach().cpu().numpy().astype(np.float64)

        up = raw_up - np.dot(raw_up, forward) * forward
        if np.linalg.norm(up) < 1e-8:
            axes = np.eye(3, dtype=np.float64)
            fallback = axes[int(np.argmin(np.abs(axes @ forward)))]
            up = fallback - np.dot(fallback, forward) * forward
        up = _normalize(up)

        right = _normalize(np.cross(forward, up))
        up = _normalize(np.cross(right, forward))

        fx = float(camera.fx)
        fy = float(camera.fy)
        cx = float(camera.cx)
        cy = float(camera.cy)

        # =============================================================
        # Step 2: 由 safe 区域反推真实视锥四个侧面的"内法向"
        #         (相机系/世界系都给出，便于阅读)
        # =============================================================
        k_left = (u_min - cx) / fx
        k_right = (u_max - cx) / fx
        k_bottom = (v_min - cy) / fy
        k_top = (v_max - cy) / fy
        if not (k_left < k_right and k_bottom < k_top):
            raise ValueError('Invalid effective frustum after applying safe pixels.')

        # 世界系内法向（未单位化，但符号正确）：
        #   N · (P - C) >= 0  <=>  P 在该侧面对应的内半空间。
        # 这里只用它们对应的标量约束求解，故无需显式归一化。
        # N_L = right - k_left * forward
        # N_R = -right + k_right * forward
        # N_B = up - k_bottom * forward
        # N_T = -up + k_top * forward

        # =============================================================
        # Step 3: 把四个半空间约束写成 (cr, cu, cf) 上的线性不等式
        # =============================================================
        x = pts @ right
        y = pts @ up
        s = pts @ forward

        # 等价形式（已展开 N_* · (P - C) >= 0）：
        #   k_left   * (s_i - cf) <= x_i - cr <= k_right * (s_i - cf)
        #   k_bottom * (s_i - cf) <= y_i - cu <= k_top   * (s_i - cf)
        #   s_i - cf > 0
        # 对水平方向：固定 cf，cr 必须落入区间
        #   max_i [x_i - k_right * (s_i - cf)] <= cr
        #                                     <= min_i [x_i - k_left * (s_i - cf)]
        # 区间非空 <=> (k_right - k_left) * cf
        #              <= min_i (x_i - k_left * s_i) - max_i (x_i - k_right * s_i)
        # 给出水平方向允许的最大 cf：
        cf_x = (
            np.min(x - k_left * s) - np.max(x - k_right * s)
        ) / (k_right - k_left)
        cf_y = (
            np.min(y - k_bottom * s) - np.max(y - k_top * s)
        ) / (k_top - k_bottom)

        # 深度限制：所有点必须在相机前方，即 cf < min_i s_i。
        scene_scale = max(float(np.max(np.ptp(pts, axis=0))), 1.0)
        eps = 1e-9 * scene_scale
        cf_depth = float(np.min(s)) - eps

        # 取最小者：水平/竖直/深度三者中最先收紧的那一项决定 cf。
        # 处于该项时，对应轴的两侧侧面分别与点云相切（"两侧夹紧"）。
        cf = min(float(cf_x), float(cf_y), cf_depth)

        # =============================================================
        # Step 4: 在该 cf 下求 cr / cu 的可行区间
        # =============================================================
        cr_low = float(np.max(x - k_right * (s - cf)))
        cr_high = float(np.min(x - k_left * (s - cf)))
        cu_low = float(np.max(y - k_top * (s - cf)))
        cu_high = float(np.min(y - k_bottom * (s - cf)))

        tol = 1e-6 * scene_scale
        if cr_low > cr_high + tol or cu_low > cu_high + tol:
            raise RuntimeError('Failed to find a feasible camera position.')

        if cr_low > cr_high:
            cr_mid = 0.5 * (cr_low + cr_high)
            cr_low = cr_mid
            cr_high = cr_mid
        if cu_low > cu_high:
            cu_mid = 0.5 * (cu_low + cu_high)
            cu_low = cu_mid
            cu_high = cu_mid

        depth = s - cf
        if np.min(depth) <= 0:
            raise RuntimeError('Failed to keep all points in front of the camera.')

        # =============================================================
        # Step 5: 在 non-binding 轴上选 cr / cu 让投影 bbox 居中
        #         投影像素：u = fx * (x_i - cr) / D_i + cx
        #                   v = fy * (y_i - cu) / D_i + cy
        #         depth 已固定，故 u_i 关于 cr 严格递减，bbox 中心也严格递减。
        #         binding 轴上 (low ≈ high) 自然退化到唯一解。
        # =============================================================
        def _projected_bbox_center(
            coord: np.ndarray,
            offset: float,
            focal: float,
            principal: float,
        ) -> float:
            pixels = focal * (coord - offset) / depth + principal
            return 0.5 * (float(np.min(pixels)) + float(np.max(pixels)))

        def _solve_centered_offset(
            coord: np.ndarray,
            low: float,
            high: float,
            focal: float,
            principal: float,
            target_center: float,
        ) -> float:
            if high - low <= tol:
                return 0.5 * (low + high)

            low_center = _projected_bbox_center(coord, low, focal, principal)
            high_center = _projected_bbox_center(coord, high, focal, principal)

            # 单调递减：target 落在区间外则取相应端点。
            if target_center >= low_center:
                return low
            if target_center <= high_center:
                return high

            left = low
            right_bound = high
            for _ in range(80):
                mid = 0.5 * (left + right_bound)
                center = _projected_bbox_center(coord, mid, focal, principal)
                if center > target_center:
                    left = mid
                else:
                    right_bound = mid
            return 0.5 * (left + right_bound)

        cr = _solve_centered_offset(
            x,
            cr_low,
            cr_high,
            fx,
            cx,
            0.5 * (u_min + u_max),
        )
        cu = _solve_centered_offset(
            y,
            cu_low,
            cu_high,
            fy,
            cy,
            0.5 * (v_min + v_max),
        )

        new_pos = cr * right + cu * up + cf * forward
        return new_pos.astype(np.float32)

    @staticmethod
    def getBestCameraPoses(
        camera_list: List[Camera],
        pts: np.ndarray,
        safe_pixel_num: int = 10,
    ) -> np.ndarray:
        if len(camera_list) == 0:
            return np.empty((0, 3), dtype=np.float32)

        with ThreadPoolExecutor() as executor:
            best_poses = list(
                executor.map(
                    lambda camera: CameraConvertor.getBestCameraPose(
                        camera=camera,
                        pts=pts,
                        safe_pixel_num=safe_pixel_num,
                    ),
                    camera_list,
                )
            )

        return np.stack(best_poses, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def getBestPoseCameras(
        camera_list: List[Camera],
        pts: np.ndarray,
        safe_pixel_num: int = 10,
    ) -> List[Camera]:
        best_pose_camera_list = deepcopy(camera_list)

        best_poses = CameraConvertor.getBestCameraPoses(
            camera_list=camera_list,
            pts=pts,
            safe_pixel_num=safe_pixel_num,
        )

        for i in range(len(camera_list)):
            best_pose_camera_list[i].setWorldPose(pos=best_poses[i])
        return best_pose_camera_list

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
        visualization_camera_scale: float=0.1,
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
        ccm_vis_folder_path = save_data_folder_path + 'ccms_vis/'
        masked_ccm_vis_folder_path = save_data_folder_path + 'masked_ccms_vis/'

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
        os.makedirs(ccm_vis_folder_path, exist_ok=True)
        os.makedirs(masked_ccm_vis_folder_path, exist_ok=True)

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

        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]

        def _process_one_camera(camera_idx):
            camera = cameras[camera_idx]
            image_basename = os.path.splitext(camera.image_id)[0]
            image_filename = image_basename + '.png'

            colmap_pose = camera.toColmapPose().cpu().numpy()

            cv2.imwrite(image_folder_path + image_filename, camera.toImageCV(use_mask=False), png_params)

            if camera.mask is not None:
                cv2.imwrite(mask_folder_path + image_filename, camera.toMaskCV(), png_params)
                cv2.imwrite(masked_image_folder_path + image_filename, camera.toImageCV(use_mask=True), png_params)

            if camera.depth is not None:
                np.save(depth_folder_path + image_basename + '.npy', camera.depth_with_conf.cpu().numpy())
                cv2.imwrite(depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=False), png_params)
                cv2.imwrite(masked_depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=True), png_params)
                cv2.imwrite(ccm_vis_folder_path + image_filename, camera.toCCMVisCV(use_mask=False), png_params)
                cv2.imwrite(masked_ccm_vis_folder_path + image_filename, camera.toCCMVisCV(use_mask=True), png_params)

            if camera.normal_world is not None:
                np.save(normal_world_folder_path + image_basename + '.npy', camera.normal_world.cpu().numpy())
                cv2.imwrite(normal_world_vis_folder_path + image_filename, camera.toNormalWorldVisCV(use_mask=False), png_params)
                cv2.imwrite(masked_normal_world_vis_folder_path + image_filename, camera.toNormalWorldVisCV(use_mask=True), png_params)

            if camera.normal_camera is not None:
                np.save(normal_camera_folder_path + image_basename + '.npy', camera.normal_camera.cpu().numpy())
                cv2.imwrite(normal_camera_vis_folder_path + image_filename, camera.toNormalCameraVisCV(use_mask=False), png_params)
                cv2.imwrite(masked_normal_camera_vis_folder_path + image_filename, camera.toNormalCameraVisCV(use_mask=True), png_params)

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

        def _o3dMeshToTrimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> Optional[trimesh.Trimesh]:
            vertices = np.asarray(o3d_mesh.vertices)
            triangles = np.asarray(o3d_mesh.triangles)
            if vertices.shape[0] == 0 or triangles.shape[0] == 0:
                return None
            tm = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            if o3d_mesh.has_vertex_colors():
                vertex_colors = np.asarray(o3d_mesh.vertex_colors)
                vertex_colors_uint8 = np.clip(vertex_colors * 255.0, 0, 255).astype(np.uint8)
                rgba = np.concatenate(
                    [vertex_colors_uint8, np.full((vertex_colors_uint8.shape[0], 1), 255, dtype=np.uint8)],
                    axis=1,
                )
                tm.visual.vertex_colors = rgba
            return tm

        scene = trimesh.Scene()
        cameras_mesh = CameraConvertor.toCamerasMesh(
            camera_list=cameras,
            scale=visualization_camera_scale,
        )
        cameras_tm = _o3dMeshToTrimesh(cameras_mesh)
        if cameras_tm is not None:
            scene.add_geometry(cameras_tm, node_name='cameras')

        pcd_tm = toTrimeshPcd(pcd)
        if pcd_tm is not None:
            scene.add_geometry(pcd_tm, node_name='points3D')

        visualization_file_path = save_data_folder_path + 'visualization.glb'
        scene.export(visualization_file_path)
        print(f'\t saved visualization to: {visualization_file_path}')

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
                if len(parts) < 7:
                    continue
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                fx = float(parts[4])
                if model == 'PINHOLE' and len(parts) >= 8:
                    fy, cx, cy = float(parts[5]), float(parts[6]), float(parts[7])
                    cameras_dict[camera_id] = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
                elif model == 'SIMPLE_PINHOLE' and len(parts) >= 7:
                    fy = fx
                    cx = float(parts[5])
                    cy = float(parts[6])
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

    @staticmethod
    def toCamerasMesh(
        camera_list: List[Camera],
        scale: float=0.1,
    ) -> o3d.geometry.TriangleMesh:
        cameras_mesh = o3d.geometry.TriangleMesh()

        for camera in camera_list:
            cameras_mesh += camera.toO3DMesh(far=scale)
            cameras_mesh += camera.toO3DAxisMesh(length=scale)

        return cameras_mesh
