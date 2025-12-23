import torch
import numpy as np
import open3d as o3d

from camera_control.Method.random_sample import sample_points_in_front_of_camera
from camera_control.Method.render import create_line_set
from camera_control.Module.camera import Camera


def demo():
    """
    测试新的相机坐标系定义和估计算法
    
    坐标系定义：
    - 相机坐标系：X右，Y上，Z后（看向 -Z 方向）
    - UV坐标系：原点在 (0,0) 左下角，u沿X轴（向右），v沿Y轴（向上）
    """
    n_points = 100000

    pos = np.random.randn(3) * 5.0
    # 随机看向的方向（在相机前方）
    look_at = pos + np.random.randn(3) * 10.0
    # 随机上方向
    up = np.random.randn(3)
    up = up / (np.linalg.norm(up) + 1e-8)

    camera = Camera(pos=pos, look_at=look_at, up=up)

    print("=" * 60)
    print("原始相机参数:")
    camera.outputInfo()
    print("=" * 60)

    points = sample_points_in_front_of_camera(camera, n_points, distance_range=(2, 4))
    uv = camera.project_points_to_uv(points)

    # 添加噪声
    noise = 0.3 * (torch.rand_like(uv) - 0.5) * 2
    uv_noisy = uv + noise

    # 从带噪声的UV坐标估计相机
    estimated_camera = Camera.fromUVPoints(
        points,
        uv_noisy,
        width=camera.width,
        height=camera.height,
    )
    assert estimated_camera is not None

    print("\n估计的相机参数:")
    estimated_camera.outputInfo()
    print("=" * 60)

    # 计算误差
    pos_error = torch.linalg.norm(camera.pos - estimated_camera.pos)
    rot_error = torch.linalg.norm(camera.rot - estimated_camera.rot, ord='fro')
    
    print(f"\n位置误差: {pos_error.item():.6f}")
    print(f"旋转矩阵误差 (Frobenius范数): {rot_error.item():.6f}")
    
    # 计算重投影误差
    uv_reprojected = estimated_camera.project_points_to_uv(points)
    valid_mask = ~(torch.isnan(uv[:, 0]) | torch.isnan(uv_reprojected[:, 0]))
    if valid_mask.sum() > 0:
        reproj_error = torch.linalg.norm(
            uv[valid_mask] - uv_reprojected[valid_mask], 
            dim=1
        ).mean()
        print(f"平均重投影误差: {reproj_error.item():.6f}")
    
    print("=" * 60)

    # 可视化
    geometry_list = []

    valid_mask = ~torch.isnan(uv[:, 0])
    valid_points = points[valid_mask]

    # 添加有效点云（红色）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points.numpy())
    pcd.paint_uniform_color([1, 0, 0])
    geometry_list.append(pcd)

    # 连接线（青色）
    lineset = create_line_set(camera.pos.numpy(), valid_points.numpy(), color=[0, 1, 1])
    geometry_list.append(lineset)

    # 相机视锥
    # 原始相机（绿色）
    geometry_list.append(camera.toO3DMesh(far=1.0, color=[0, 1, 0]))
    # 估计相机（蓝色）
    geometry_list.append(estimated_camera.toO3DMesh(far=1.0, color=[0, 0, 1]))

    # 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometry_list.append(coord_frame)

    o3d.visualization.draw_geometries(geometry_list)

    return True
