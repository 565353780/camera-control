import torch
import numpy as np
import open3d as o3d

from uv_cube_gen.Module.camera import Camera


def create_random_camera(seed=None):
    """创建随机相机"""
    if seed is not None:
        np.random.seed(seed)

    # 随机相机位置（在单位球附近）
    pos = np.random.randn(3) * 5.0
    # 随机看向的方向（在相机前方）
    look_at = pos + np.random.randn(3) * 10.0
    # 随机上方向
    up = np.random.randn(3)
    up = up / (np.linalg.norm(up) + 1e-8)

    return Camera(pos=pos, look_at=look_at, up=up)


def sample_points_in_front_of_camera(camera, n_points=100, distance_range=(1.0, 10.0), fov_range=(-0.4, 0.4)):
    """在相机前方根据朝向直接生成随机点"""
    # 获取相机坐标系基向量
    forward = camera.rot[:, 2].numpy()  # 前向方向（Z轴）
    right = camera.rot[:, 0].numpy()    # 右方向（X轴）
    up = camera.rot[:, 1].numpy()       # 上方向（Y轴）
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
    # 每个点 = pos + offset_x * right + offset_y * up + distance * forward
    points_world = (
        pos + 
        offsets_x[:, np.newaxis] * right + 
        offsets_y[:, np.newaxis] * up + 
        distances[:, np.newaxis] * forward
    )
    
    return points_world


def create_line_set(start_pos, end_pos, color=[1, 0, 0]):
    """创建连接两个点的线段"""
    points = np.array([start_pos, end_pos])
    lines = np.array([[0, 1]])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set


def test():
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
        points_tensor = torch.from_numpy(points).float()

        # 计算UV坐标
        uv = camera.project_points_to_uv(points_tensor)
        
        # 过滤有效点
        valid_mask = ~torch.isnan(uv[:, 0])
        valid_points = points_tensor[valid_mask]
        valid_uv = uv[valid_mask]
        
        print(f"  有效点数: {len(valid_points)}/{n_points}")
        
        if len(valid_points) < 6:
            print(f"  警告: 相机 {i+1} 有效点不足，跳过求解")
            continue
        
        # 使用fromUVPoints求解相机参数
        estimated_camera = Camera.fromUVPoints(valid_points, valid_uv, width=camera.width, height=camera.height)
        if estimated_camera is None:
            print(f"  警告: 相机 {i+1} 求解失败!")
            continue
        
        estimated_cameras.append((estimated_camera, i))
        
        # 保存点云数据用于可视化
        camera_data.append((points, valid_points.numpy(), i))
        
        # 计算误差
        pos_error = torch.norm(estimated_camera.pos - camera.pos).item()
        rot_error = torch.norm(estimated_camera.rot - camera.rot).item()
        print(f"  位置误差: {pos_error:.6f}")
        print(f"  旋转误差: {rot_error:.6f}")
    
    print(f"\n成功求解 {len(estimated_cameras)}/{N} 个相机")
    
    # 使用Open3D可视化
    print("\n创建可视化...")
    geometries = []
    
    # 为每个相机添加点云和连线
    for all_points, valid_points, cam_idx in camera_data:
        camera = original_cameras[cam_idx]
        cam_pos = camera.pos.numpy()
        
        # 添加当前相机（绿色）
        mesh = camera.toO3DMesh(far=2.0, color=[0, 1, 0])
        geometries.append(mesh)
        
        # 添加所有采样点云（浅灰色）
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(all_points)
        pcd_all.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_all)
        
        # 添加有效点云（黄色）
        pcd_valid = o3d.geometry.PointCloud()
        pcd_valid.points = o3d.utility.Vector3dVector(valid_points)
        pcd_valid.paint_uniform_color([1, 1, 0])
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
        est_mesh = estimated_camera.toO3DMesh(far=2.0, color=[0, 0, 1])
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
