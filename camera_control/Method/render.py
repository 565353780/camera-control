import torch
import numpy as np
import open3d as o3d
from typing import Union


def toPcd(
    points: Union[torch.Tensor, np.ndarray, list],
    color: list=[1, 0, 0],
) -> o3d.geometry.PointCloud:
    if isinstance(points, list):
        points = np.array(points)
    elif isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if points.shape[-1] == 4:
        points = points[..., :3]

    points = points.reshape(-1, 3).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.paint_uniform_color(color)
    return pcd

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

def create_line_set(
    start_pos: Union[torch.Tensor, np.ndarray, list],
    end_pos: Union[torch.Tensor, np.ndarray, list],
    color=[1, 0, 0],
) -> o3d.geometry.LineSet:
    """
    创建线段集合
    
    Args:
        start_pos: 起点位置，可以是单个点(3或1x3)或多个点(Nx3)
        end_pos: 终点位置，可以是单个点(3或1x3)或多个点(Nx3)
        color: 线段颜色，默认红色
    
    支持的4种情况：
        1. 单点到单点：一条线段
        2. 单点到多点：从单个起点到多个终点，多条线段
        3. 多点到单点：从多个起点到单个终点，多条线段
        4. 相同数量的起点和终点：逐点连线，N条线段
    
    Returns:
        line_set: LineSet对象
    """
    # 转换为numpy数组
    def to_numpy(x):
        if isinstance(x, list):
            x = np.asarray(x)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float64)

    start_pos = to_numpy(start_pos)
    end_pos = to_numpy(end_pos)

    if start_pos.shape[-1] != 3:
        start_pos = start_pos[..., :3]

    if end_pos.shape[-1] != 3:
        end_pos = end_pos[..., :3]

    start_pos = start_pos.reshape(-1, 3)
    end_pos = end_pos.reshape(-1, 3)

    num_start = start_pos.shape[0]
    num_end = end_pos.shape[0]

    # 判断并处理4种情况
    if num_start == 1 and num_end == 1:
        # 情况1：单点到单点
        points = np.vstack([start_pos, end_pos])
        lines = np.array([[0, 1]])
    elif num_start == 1 and num_end > 1:
        # 情况2：单点到多点
        points = np.vstack([start_pos, end_pos])
        lines = np.array([[0, i+1] for i in range(num_end)])
    elif num_start > 1 and num_end == 1:
        # 情况3：多点到单点
        points = np.vstack([start_pos, end_pos])
        end_idx = num_start  # 终点的索引
        lines = np.array([[i, end_idx] for i in range(num_start)])
    elif num_start == num_end:
        # 情况4：相同数量的起点和终点，逐点连线
        points = np.vstack([start_pos, end_pos])
        lines = np.array([[i, i + num_start] for i in range(num_start)])
    else:
        raise ValueError(
            f"不支持的输入形状组合：start_pos有{num_start}个点，end_pos有{num_end}个点。"
            f"仅支持：1-1, 1-N, N-1, 或 N-N（N相同）"
        )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set
