import torch
import trimesh
import numpy as np
from typing import List

from camera_control.Module.camera import Camera


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
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(num_points):
        # 纵向位置 [-1, 1]
        y = 1 - (i / float(num_points - 1)) * 2
        # 该高度处的半径
        radius_at_y = np.sqrt(1 - y * y)
        # 横向角度
        theta = phi * i

        # 计算球面坐标
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        # 缩放到指定半径并平移到中心
        point = np.array([x, y, z]) * radius + center
        points.append(point)

    return np.array(points)

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
