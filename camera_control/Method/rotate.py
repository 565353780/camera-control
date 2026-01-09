import torch


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
