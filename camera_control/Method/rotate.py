import numpy as np


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """
    将3x3旋转矩阵转换为四元数 (w, x, y, z)
    使用与 COLMAP 一致的 rotmat2qvec 算法

    Args:
        R: 3x3旋转矩阵

    Returns:
        四元数 [w, x, y, z]
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec.astype(np.float64)
