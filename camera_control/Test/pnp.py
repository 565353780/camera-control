import torch
import numpy as np
import open3d as o3d

from camera_control.Method.random_sample import sample_points_in_front_of_camera
from camera_control.Method.render import create_line_set, toPcd
from camera_control.Module.camera import Camera


def create_random_camera(seed=None):
    """
    创建随机相机

    相机坐标系定义：
    - X轴：向右
    - Y轴：向上
    - Z轴：向后（相机看向 -Z 方向）
    """
    if seed is not None:
        np.random.seed(seed)

    # 随机相机位置（在单位球附近）
    pos = np.random.randn(3) * 5.0
    # 随机看向的方向（在相机前方）
    look_at = pos + np.random.randn(3) * 10.0
    # 随机上方向
    up = np.random.randn(3)

    return Camera(pos=pos, look_at=look_at, up=up)


def test():
    """
    测试PnP求解算法

    通过随机生成相机和3D点，投影到UV坐标，然后使用fromUVPoints恢复相机参数，
    计算恢复的相机与原始相机之间的误差。

    相机坐标系定义：
    - X轴：向右
    - Y轴：向上
    - Z轴：向后（相机看向 -Z 方向）

    UV坐标系定义：
    - 原点在 (0, 0) 左下角
    - u沿X轴（向右）
    - v沿Y轴（向上）
    """
    N = 5  # 相机数量
    n_points = 100  # 每个相机采样的点数

    print(f"创建 {N} 个随机相机...")
    original_cameras = []
    estimated_cameras = []  # (estimated_camera, original_index) 的列表
    camera_data = []  # 存储每个相机的点云数据: (all_points, valid_points, camera_index)

    for i in range(N):
        print(f"\n处理相机 {i+1}/{N}...")
        # 创建随机相机
        camera = create_random_camera(seed=i)
        original_cameras.append(camera)

        # 在相机前方采样点
        points = sample_points_in_front_of_camera(camera, n_points)

        # 计算UV坐标
        uv = camera.project_points_to_uv(points)

        # 过滤有效点
        valid_mask = ~torch.isnan(uv[:, 0])
        valid_points = points[valid_mask]
        valid_uv = uv[valid_mask]

        print(f"  有效点数: {len(valid_points)}/{n_points}")

        if len(valid_points) < 6:
            print(f"  警告: 相机 {i+1} 有效点不足，跳过求解")

            camera_mesh = camera.toO3DMesh()
            pcd = toPcd(points)
            o3d.visualization.draw_geometries([camera_mesh, pcd])
            continue

        # 使用fromUVPoints求解相机参数
        estimated_camera = Camera.fromUVPointsV2(
            valid_points,
            valid_uv,
            width=camera.width,
            height=camera.height,
            device=camera.device,
        )
        if estimated_camera is None:
            print(f"  警告: 相机 {i+1} 求解失败!")
            continue

        estimated_cameras.append((estimated_camera, i))

        # 保存点云数据用于可视化
        camera_data.append((points, valid_points.numpy(), i))

        # 计算误差
        R_error = torch.norm(estimated_camera.R - camera.R).item()
        t_error = torch.norm(estimated_camera.t - camera.t).item()

        # 计算重投影误差
        uv_reprojected = estimated_camera.project_points_to_uv(valid_points)
        reproj_error = torch.linalg.norm(valid_uv - uv_reprojected, dim=1).mean().item()

        print(f"  位置误差: {t_error:.6f}")
        print(f"  旋转矩阵误差 (Frobenius范数): {R_error:.6f}")
        print(f"  重投影误差: {reproj_error:.6f}")

    print(f"\n成功求解 {len(estimated_cameras)}/{N} 个相机")

    # 使用Open3D可视化
    print("\n创建可视化...")
    geometries = []
    
    # 为每个相机添加点云和连线
    for all_points, valid_points, cam_idx in camera_data:
        camera = original_cameras[cam_idx]
        cam_pos = camera.pos.numpy()
        
        # 添加当前相机（绿色）
        mesh = camera.toO3DMesh(far=1.0, color=[0, 1, 0])
        geometries.append(mesh)
        
        # 添加所有采样点云（浅灰色）
        pcd_all = toPcd(all_points, [0.7, 0.7, 0.7])
        geometries.append(pcd_all)
        
        # 添加有效点云（黄色）
        pcd_valid = toPcd(valid_points, [1, 1, 0])
        geometries.append(pcd_valid)
        
        # 添加相机到有效点的连线（青色）
        # 为了性能，只连接部分点（每5个点连接一次）
        step = max(1, len(valid_points) // 20)  # 最多连接20条线
        for j in range(0, len(valid_points), step):
            line = create_line_set(cam_pos, valid_points[j], color=[0, 1, 1])
            geometries.append(line)
    
    # 添加估计的相机（蓝色）和连接线（红色）
    for estimated_camera, orig_idx in estimated_cameras:
        # 添加估计的相机
        est_mesh = estimated_camera.toO3DMesh(far=1.0, color=[0, 0, 1])
        geometries.append(est_mesh)
        
        # 添加连接线（红色），显示位置误差
        orig_pos = original_cameras[orig_idx].pos.numpy()
        est_pos = estimated_camera.pos.numpy()
        line = create_line_set(orig_pos, est_pos, color=[1, 0, 0])
        geometries.append(line)
    
    # 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)
    
    # 可视化
    print("打开可视化窗口...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="PnP求解误差可视化",
        width=1200,
        height=800
    )
    
    return True
