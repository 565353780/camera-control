import torch
import numpy as np

from camera_control.Method.data import toTensor


def sample_points_in_front_of_camera(
    camera,
    n_points=100,
    distance_range=(1.0, 10.0),
    fov_range=(-0.4, 0.4),
) -> torch.Tensor:
    """
    在相机前方根据朝向直接生成随机点
    
    相机坐标系定义：
    - X轴：右方向
    - Y轴：上方向
    - Z轴：后方向（相机看向 -Z 方向）
    
    Args:
        camera: Camera对象
        n_points: 采样点数
        distance_range: 距离范围 (min, max)
        fov_range: 视场角范围，单位弧度
    
    Returns:
        世界坐标系中的3D点，shape (n_points, 3)
    """
    # 获取 camera2world 变换矩阵
    camera2world = camera.camera2world
    
    # 在相机坐标系中生成点（使用torch）
    # 相机看向 -Z 方向，所以前方的点 Z < 0
    
    # 随机距离
    distances = torch.from_numpy(np.random.uniform(distance_range[0], distance_range[1], n_points)).float()
    
    # 随机水平和垂直角度
    angles_x = torch.from_numpy(np.random.uniform(fov_range[0], fov_range[1], n_points)).float()
    angles_y = torch.from_numpy(np.random.uniform(fov_range[0], fov_range[1], n_points)).float()
    
    # 在相机坐标系中生成点
    # 主方向是 -Z（前方），然后加上 X 和 Y 的偏移
    # X = tan(angle_x) * distance
    # Y = tan(angle_y) * distance
    # Z = -distance（负数，因为看向 -Z）
    points_camera = torch.zeros((n_points, 3), dtype=torch.float32)
    points_camera[:, 0] = torch.tan(angles_x) * distances  # X: 右方向偏移
    points_camera[:, 1] = torch.tan(angles_y) * distances  # Y: 上方向偏移
    points_camera[:, 2] = -distances                        # Z: 负数，在相机前方
    
    # 转换到齐次坐标
    points_camera_homo = torch.cat([
        points_camera, 
        torch.ones((n_points, 1), dtype=torch.float32)
    ], dim=1)
    
    # 转换到世界坐标系（使用torch）
    points_world_homo = (camera2world @ points_camera_homo.T).T
    points_world = points_world_homo[:, :3]
    
    return points_world
