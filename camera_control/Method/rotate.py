import torch
import numpy as np
from typing import Optional, Tuple, Union


def _to_float_tensor(
    data: Union[torch.Tensor, np.ndarray, list],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """把 tensor / ndarray / list 统一成 float tensor, 保留原 device (无则 CPU)。"""
    if torch.is_tensor(data):
        if data.is_floating_point():
            return data
        return data.to(dtype=dtype)
    return torch.as_tensor(np.asarray(data, dtype=np.float64), dtype=dtype)


def rotmat2qvec(R: torch.Tensor) -> torch.Tensor:
    """
    将3x3旋转矩阵转换为四元数 (w, x, y, z)
    使用与 COLMAP 一致的 rotmat2qvec 算法

    Args:
        R: 3x3旋转矩阵 (torch.Tensor)

    Returns:
        四元数 [w, x, y, z] (torch.Tensor)
    """
    dtype = R.dtype
    device = R.device
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    three = torch.tensor(3.0, dtype=dtype, device=device)

    # 平铺R
    R_flat = R.flatten()
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R_flat[0], R_flat[1], R_flat[2], \
                                                   R_flat[3], R_flat[4], R_flat[5], \
                                                   R_flat[6], R_flat[7], R_flat[8]
    K = torch.stack([
        torch.stack([Rxx - Ryy - Rzz,        zero,        zero,        zero], dim=0),
        torch.stack([Ryx + Rxy, Ryy - Rxx - Rzz,        zero,        zero], dim=0),
        torch.stack([Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy,        zero], dim=0),
        torch.stack([Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz], dim=0)
    ], dim=0) / three

    # 用torch计算特征值和特征向量
    eigvals, eigvecs = torch.linalg.eigh(K)
    max_idx = torch.argmax(eigvals)
    qvec = eigvecs[:, max_idx][[3, 0, 1, 2]]  # [w, x, y, z]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec


def qvec2rotmat(qvec: torch.Tensor) -> torch.Tensor:
    """
    将四元数 (w, x, y, z) 转换为 3x3 旋转矩阵
    与 COLMAP 的 rotmat2qvec 互为逆变换

    Args:
        qvec: 四元数 [w, x, y, z] (torch.Tensor)

    Returns:
        3x3 旋转矩阵 (torch.Tensor)
    """
    qw, qx, qy, qz = qvec[0], qvec[1], qvec[2], qvec[3]
    R = torch.stack([
        torch.stack([1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy], dim=0),
        torch.stack([2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx], dim=0),
        torch.stack([2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy], dim=0),
    ], dim=0)
    return R

def decompose_similarity_from_T(
    T: Union[torch.Tensor, list],
    enforce_positive_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从任意 4x4 ICP 变换矩阵 T 中
    求最接近的 similarity 变换参数 (R, s, t)

    使得：
        T ≈ [ sR  t ]
            [  0   1 ]

    Returns:
        R: (3,3) 正交旋转矩阵
        s: 标量 scale
        t: (3,) 平移向量
    """

    if not torch.is_tensor(T):
        T = torch.tensor(T, dtype=torch.float32)

    T = T.reshape(4, 4)

    M = T[:3, :3]
    t = T[:3, 3].clone()

    # --- SVD 分解 ---
    U, S, Vt = torch.linalg.svd(M)

    # 最优旋转
    R = U @ Vt

    # 修正 reflection
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # 最优 scale（Frobenius 最小二乘）
    s = (S.sum()) / 3.0

    if enforce_positive_scale:
        s = torch.abs(s)

    return R, s, t

def estimate_similarity_transform_from_points(
    source_points: Union[torch.Tensor, np.ndarray, list],
    target_points: Union[torch.Tensor, np.ndarray, list],
    allow_rotation: bool = True,
    allow_scale: bool = True,
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """从两组同长度、一一对应的有序点集求严格 Sim(3) 相似变换。

    用 Umeyama (least-squares similarity) 求解,使得在列向量左乘约定下:
        target_col ≈ s * R @ source_col + t
    其中 R 为正交旋转 (det=+1, 无剪切/反射), s 为各向同性缩放, t 为平移。

    返回的是 **行向量右乘** 约定的 4x4 ``world_transform`` (即列向量左乘矩阵的转置),
    满足:
        target_row ≈ source_row @ world_transform[:3, :3] + world_transform[3, :3]
    可直接传给 ``CameraConvertor.transformCameras`` (其约定与 Open3D ICP 矩阵的转置一致),
    也可用 ``decompose_similarity_from_T(world_transform.T)`` 反求 (R, s, t)。

    退化保护:
        - 点数不足 (rotation 需 >=3, 仅 scale/translation 需 >=2) 返回 None;
        - 点集方差过小 (轨迹塌缩) 时旋转/缩放不可靠, 返回 None;
        - ``allow_rotation=False`` 时 R 固定为单位阵 (仅估缩放+平移);
        - ``allow_scale=False`` 时 s 固定为 1 (仅估旋转+平移)。

    Args:
        source_points: (N, 3) 源点集。
        target_points: (N, 3) 目标点集, 与 source 一一对应、顺序一致。
        allow_rotation: 是否估计旋转, 否则 R = I。
        allow_scale: 是否估计各向同性缩放, 否则 s = 1。
        eps: 数值下限, 防止除零。

    Returns:
        world_transform: (4, 4) torch.Tensor, 行向量右乘约定; 退化时返回 None。
    """
    source = _to_float_tensor(source_points).reshape(-1, 3)
    target = _to_float_tensor(target_points).reshape(-1, 3)

    if source.shape[0] != target.shape[0]:
        print('[ERROR][rotate::estimate_similarity_transform_from_points]')
        print('\t source / target point count mismatch:',
              source.shape[0], target.shape[0])
        return None

    num_points = source.shape[0]
    min_points = 3 if allow_rotation else 2
    if num_points < min_points:
        print('[WARN][rotate::estimate_similarity_transform_from_points]')
        print('\t not enough points to estimate transform:', num_points,
              '(need >=', min_points, ')')
        return None

    dtype = source.dtype
    device = source.device

    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    source_var = (source_centered ** 2).sum() / num_points
    if source_var < eps:
        print('[WARN][rotate::estimate_similarity_transform_from_points]')
        print('\t source points are degenerate (near-zero variance).')
        return None

    singular_values = None
    reflection_fix = torch.ones(3, dtype=dtype, device=device)
    if allow_rotation:
        # 协方差矩阵 (列向量约定): cov = (1/N) * target_c^T @ source_c
        cov = (target_centered.T @ source_centered) / num_points
        U, singular_values, Vt = torch.linalg.svd(cov)

        # 处理反射: 保证 det(R) = +1
        if torch.det(U @ Vt) < 0:
            reflection_fix[-1] = -1.0
        R = U @ torch.diag(reflection_fix) @ Vt
    else:
        R = torch.eye(3, dtype=dtype, device=device)

    if allow_scale:
        if allow_rotation and singular_values is not None:
            # Umeyama 闭式 scale: s = trace(D @ Σ) / var(source)
            scale = (singular_values * reflection_fix).sum() / source_var
        else:
            # 无旋转时, scale 退化为 sqrt(var(target)/var(source))。
            target_var = (target_centered ** 2).sum() / num_points
            scale = torch.sqrt(target_var / source_var.clamp(min=eps))
        scale = scale.clamp(min=eps)
    else:
        scale = torch.tensor(1.0, dtype=dtype, device=device)

    # 列向量左乘平移: t = target_mean - s * R @ source_mean
    t = target_mean - scale * (R @ source_mean)

    # 列向量左乘 4x4: T_left = [sR | t; 0 | 1]
    T_left = build_similarity_matrix(R, scale, t)

    # transformCameras 约定行向量右乘 world_transform = 列向量左乘矩阵的转置。
    world_transform = T_left.T.contiguous()
    return world_transform


def build_similarity_matrix(R, s, t):
    T_clean = torch.eye(4, dtype=R.dtype, device=R.device)
    T_clean[:3, :3] = s * R
    T_clean[:3, 3] = t
    return T_clean

def invert_similarity(R, s, t):
    """
    输入:
        R: (3,3)
        s: scalar
        t: (3,)
    输出:
        T_inv: (4,4)
    """

    device = R.device
    dtype = R.dtype

    R_inv = R.T
    s_inv = 1.0 / s

    A = s_inv * R_inv
    b = -s_inv * (R_inv @ t)

    T_inv = torch.eye(4, dtype=dtype, device=device)
    T_inv[:3, :3] = A
    T_inv[:3, 3] = b

    return T_inv
