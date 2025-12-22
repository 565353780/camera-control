import torch
import numpy as np

from camera_control.Method.data import toTensor


def sample_points_in_front_of_camera(
    camera,
    n_points=100,
    distance_range=(1.0, 10.0),
    fov_range=(-0.4, 0.4),
) -> torch.Tensor:
    """在相机前方根据朝向直接生成随机点"""
    # 获取相机坐标系基向量
    forward = camera.rot[:, 2].numpy()  # 前向方向（Z轴）
    right = camera.rot[:, 0].numpy()    # 右方向（X轴）
    down = camera.rot[:, 1].numpy()       # 下方向（Y轴）
    pos = camera.pos.numpy()

    # 在相机坐标系中直接生成点（向量化操作）
    # 随机距离
    distances = np.random.uniform(distance_range[0], distance_range[1], n_points)
    # 随机水平角度（相对于前向）
    angles_x = np.random.uniform(fov_range[0], fov_range[1], n_points)
    # 随机垂直角度（相对于前向）
    angles_y = np.random.uniform(fov_range[0], fov_range[1], n_points)

    # 计算点在相机坐标系中的偏移
    # 使用tan计算横向和纵向偏移
    offsets_x = np.tan(angles_x) * distances  # 右方向偏移
    offsets_y = np.tan(angles_y) * distances   # 上方向偏移

    # 批量转换到世界坐标系（向量化）
    # 每个点 = pos + offset_x * right + offset_y * down + distance * forward
    points_world = (
        pos + 
        offsets_x[:, np.newaxis] * right +
        offsets_y[:, np.newaxis] * down +
        distances[:, np.newaxis] * forward
    )

    return toTensor(points_world)
