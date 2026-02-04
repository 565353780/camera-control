import torch
import trimesh
import numpy as np
from typing import List

from camera_control.Module.camera import Camera


def sampleFibonacciPolars(num_polars: int) -> np.ndarray:
    """
    使用Fibonacci球面采样生成均匀分布的极角 (phi, theta)。

    Args:
        num_polars: 极角对的数量

    Returns:
        polars: shape (num_polars, 2) 的数组，每行为 (phi, theta)；
               phi 为极角/天顶角 [0, pi]，theta 为方位角。
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(num_polars, dtype=np.float64)
    if num_polars > 1:
        y = 1.0 - (i / (num_polars - 1)) * 2.0
    else:
        y = np.array([0.0])
    phi = np.arccos(np.clip(y, -1.0, 1.0))
    theta = golden_angle * i
    return np.stack([phi, theta], axis=1)

def sampleFibonacciRotations(num_rotations: int) -> np.ndarray:
    """
    基于 sampleFibonacciPolars 并行生成 num_rotations 个 3x3 旋转矩阵。
    R = Rz(phi) @ Ry(theta)，phi 为方位角 (xy 平面)，theta 为极角 (与 z 轴夹角)。

    Args:
        num_rotations: 旋转数量

    Returns:
        R: shape (num_rotations, 3, 3) 的旋转矩阵数组
    """
    polars = sampleFibonacciPolars(num_rotations)  # (n, 2), (phi, theta)
    phi = polars[:, 0]
    theta = polars[:, 1]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    n = num_rotations
    Rz = np.zeros((n, 3, 3), dtype=np.float64)
    Rz[:, 0, 0] = cphi
    Rz[:, 0, 1] = -sphi
    Rz[:, 1, 0] = sphi
    Rz[:, 1, 1] = cphi
    Rz[:, 2, 2] = 1.0
    Ry = np.zeros((n, 3, 3), dtype=np.float64)
    Ry[:, 0, 0] = cth
    Ry[:, 0, 2] = sth
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sth
    Ry[:, 2, 2] = cth
    return np.matmul(Rz, Ry)

def sampleFibonacciSpherePoints(
    num_points: int,
    radius: float,
    center: np.ndarray,
) -> np.ndarray:
    """
    使用Fibonacci球面采样生成均匀分布的点

    Args:
        num_points: 点的数量
        radius: 球的半径
        center: 球心位置

    Returns:
        points: shape (num_points, 3) 的点坐标数组
    """
    polars = sampleFibonacciPolars(num_points)  # (num_points, 2), (phi, theta)
    phi = polars[:, 0]
    theta = polars[:, 1]
    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta)
    y = np.cos(phi)
    z = sin_phi * np.sin(theta)
    points = np.stack([x, y, z], axis=1) * radius + center
    return points

def sampleCamera(
    mesh: trimesh.Trimesh,
    camera_num: int = 12,
    camera_dist: float = 2.5,
    width: int = 518,
    height: int = 518,
    fx: float = 500.0,
    fy: float = 500.0,
    dtype = torch.float32,
    device: str = 'cuda:0',
) -> List[Camera]:
    """
    创建围绕mesh均匀分布的相机和深度数据

    Args:
        mesh: 输入的三角网格
        camera_num: 相机数量
        camera_dist: 相机距离mesh中心的距离
        width: 图像宽度
        height: 图像高度
        fx: 焦距x
        fy: 焦距y

    Returns:
        camera_list: 相机列表
    """
    # 计算mesh的bbox center
    bbox = mesh.bounds  # shape: (2, 3), [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    bbox_center = (bbox[0] + bbox[1]) / 2.0

    # 使用Fibonacci球面采样生成均匀分布的相机位置
    camera_positions = sampleFibonacciSpherePoints(
        camera_num, camera_dist, bbox_center
    )[..., [2, 0, 1]]

    # 创建相机列表
    camera_list = []
    for i in range(camera_num):
        camera = Camera(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            pos=camera_positions[i],
            look_at=bbox_center,
            up=[0, 1, 0],
            dtype=dtype,
            device=device,
        )
        camera_list.append(camera)

    return camera_list
