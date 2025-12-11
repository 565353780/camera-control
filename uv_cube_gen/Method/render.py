import numpy as np
import open3d as o3d


def create_coordinate_frame(origin=np.array([0, 0, 0]), size=0.2):
    """
    创建坐标轴

    Args:
        origin: 原点位置
        size: 轴的长度

    Returns:
        coord_frame: 坐标轴LineSet
    """
    points = np.array([
        origin,  # 0: 原点
        origin + [size, 0, 0],  # 1: X轴
        origin + [0, size, 0],  # 2: Y轴
        origin + [0, 0, size],  # 3: Z轴
    ])

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector([
        [0, 1],  # X轴（红色）
        [0, 2],  # Y轴（绿色）
        [0, 3],  # Z轴（蓝色）
    ])
    lines.colors = o3d.utility.Vector3dVector([
        [1, 0, 0],  # X轴红色
        [0, 1, 0],  # Y轴绿色
        [0, 0, 1],  # Z轴蓝色
    ])

    return lines
