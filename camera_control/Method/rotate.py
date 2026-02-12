import torch
from typing import Tuple, Union


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
