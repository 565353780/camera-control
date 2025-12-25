import time
import torch
import numpy as np
import open3d as o3d

from camera_control.Method.mesh import normalize_mesh, sample_points_from_mesh
from camera_control.Method.data import toTensor
from camera_control.Method.render import create_coordinate_frame
from camera_control.Module.camera import Camera


def visualize_with_open3d_animated(mesh, points_xyz, camera_base, n_frames=360):
    """
    使用open3d可视化网格、采样点、相机位置、图像平面和连线，并创建旋转动画
    
    相机坐标系定义：
    - X轴：向右
    - Y轴：向上
    - Z轴：向后（相机看向 -Z 方向）
    
    动画说明：
    - 相机位置固定不变，相机朝向会旋转
    - 相机会绕初始朝向方向旋转（方位角变化）
    - 同时相机会上下摆动（仰角变化）
    
    Args:
        mesh: open3d TriangleMesh对象
        points_xyz: 采样点，形状为 (N, 3)，torch tensor
        camera_base: 基础相机对象
        n_frames: 动画帧数
    """

    # 转换为numpy用于open3d可视化，转换为tensor用于计算
    if isinstance(points_xyz, torch.Tensor):
        points_xyz_np = points_xyz.detach().cpu().numpy()
        points_xyz_tensor = points_xyz
    else:
        points_xyz_np = np.asarray(points_xyz)
        points_xyz_tensor = toTensor(points_xyz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz_np)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色点云

    # 创建坐标轴
    coord_frame = create_coordinate_frame(origin=np.array([0, 0, 0]), size=0.3)

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Camera Rotation Animation", width=1200, height=800)

    # 添加固定几何体
    # vis.add_geometry(mesh)
    # vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)

    vis_camera = camera_base.clone()
    
    print(f"相机初始位置: {vis_camera.pos.numpy()}")

    # 投影并计算有效点
    uv = vis_camera.project_points_to_uv(points_xyz_tensor)
    valid_mask = ~torch.isnan(uv[:, 0])
    valid_points_np = points_xyz_np[valid_mask.cpu().numpy()]

    # 创建相机视锥体
    camera_frustum = vis_camera.toO3DMesh()
    vis.add_geometry(camera_frustum)

    # 创建图像平面
    # 图像平面位于相机前方（-Z方向）
    image_distance = 0.5
    half_width_world = (vis_camera.width / vis_camera.fx) * image_distance
    half_height_world = (vis_camera.height / vis_camera.fy) * image_distance
    
    # 在相机坐标系中定义图像平面的四个角点
    # 相机坐标系：X右，Y上，Z后（相机看向-Z）
    # 图像平面在 Z = -image_distance 处
    corners_camera = torch.tensor([
        [-half_width_world, -half_height_world, -image_distance],  # 左下
        [half_width_world, -half_height_world, -image_distance],   # 右下
        [half_width_world, half_height_world, -image_distance],    # 右上
        [-half_width_world, half_height_world, -image_distance],   # 左上
    ], dtype=torch.float32)
    
    # 转换到世界坐标系（使用torch）
    camera2world = vis_camera.camera2world
    corners_camera_homo = torch.cat([
        corners_camera, 
        torch.ones((4, 1), dtype=torch.float32)
    ], dim=1)
    corners_world_homo = (camera2world @ corners_camera_homo.T).T
    corners_world = corners_world_homo[:, :3].numpy()

    image_plane = o3d.geometry.TriangleMesh()
    image_plane.vertices = o3d.utility.Vector3dVector(corners_world)
    image_plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    image_plane.paint_uniform_color([1, 1, 0])
    image_plane.compute_vertex_normals()
    vis.add_geometry(image_plane)

    # 创建连线
    if len(valid_points_np) > 0:
        line_points = []
        line_indices = []
        for i, point in enumerate(valid_points_np):
            line_points.append(vis_camera.pos.numpy())
            line_points.append(point)
            line_indices.append([i * 2, i * 2 + 1])

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(line_points)
        lines.lines = o3d.utility.Vector2iVector(line_indices)
        lines.colors = o3d.utility.Vector3dVector([[0.0, 0.8, 0.0] for _ in range(len(line_indices))])  # 绿色表示可见
        vis.add_geometry(lines)
    else:
        lines = None

    # 创建反投影点的蓝色点云
    reprojected_pcd = o3d.geometry.PointCloud()
    if len(valid_points_np) > 0:
        # 计算初始反投影点
        valid_indices = torch.where(valid_mask)[0]
        valid_points_world = points_xyz_tensor[valid_indices]
        valid_uv = uv[valid_indices]
        
        # 将可见点通过 world2camera 变换到相机坐标系
        valid_points_world_homo = torch.cat([
            valid_points_world,
            torch.ones((len(valid_points_world), 1), dtype=vis_camera.dtype, device=vis_camera.device)
        ], dim=1)
        valid_points_camera_homo = torch.matmul(valid_points_world_homo, vis_camera.world2camera.T)
        valid_points_camera = valid_points_camera_homo[:, :3]
        
        # 提取深度值：使用 -z 作为深度
        depth = -valid_points_camera[:, 2]
        
        # 使用 projectUV2Points 反投影
        reprojected_points_world = vis_camera.projectUV2Points(valid_uv, depth)
        reprojected_points_np = reprojected_points_world.detach().cpu().numpy()
        
        reprojected_pcd.points = o3d.utility.Vector3dVector(reprojected_points_np)
    else:
        # 如果没有有效点，创建空点云
        reprojected_pcd.points = o3d.utility.Vector3dVector([])
    
    reprojected_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    vis.add_geometry(reprojected_pcd)

    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.7)

    print("开始动画，按Q退出...")

    # 动画循环
    frame = 0
    last_time = time.time()
    frame_duration = 1.0 / 30.0  # 30 FPS
    
    # 保存初始相机位置（固定不变）
    initial_camera_pos = vis_camera.pos.numpy().copy()
    
    # 初始朝向向量（从相机位置指向原点）
    initial_look_at = np.array([0.0, 0.0, 0.0])
    initial_forward = initial_look_at - initial_camera_pos
    initial_distance = np.linalg.norm(initial_forward)
    initial_forward_normalized = initial_forward / (initial_distance + 1e-8)
    
    # 建立局部坐标系：以初始forward为参考方向
    # 使用世界坐标系的Z轴作为参考，计算右向量和上向量
    world_up_ref = np.array([0.0, 0.0, 1.0])
    initial_right = np.cross(initial_forward_normalized, world_up_ref)
    if np.linalg.norm(initial_right) < 1e-6:
        # 如果初始forward与world_up平行，使用Y轴
        initial_right = np.cross(initial_forward_normalized, np.array([0.0, 1.0, 0.0]))
    initial_right = initial_right / (np.linalg.norm(initial_right) + 1e-8)
    initial_up = np.cross(initial_right, initial_forward_normalized)
    initial_up = initial_up / (np.linalg.norm(initial_up) + 1e-8)

    while True:
        if not vis.poll_events():
            break

        current_time = time.time()
        if current_time - last_time < frame_duration:
            continue
        last_time = current_time

        # 计算旋转角度
        # phi: 方位角，完整旋转一周（0 到 2π）
        phi = 2 * np.pi * frame / n_frames
        
        # theta: 仰角偏移，上下摆动（-30度到30度）
        theta_min = -np.pi / 6  # -30度
        theta_max = np.pi / 6   # 30度
        theta = theta_min + (theta_max - theta_min) * (0.5 + 0.5 * np.sin(2 * np.pi * frame / n_frames))
        
        # 使用Rodrigues旋转公式，让相机绕世界坐标系的Z轴旋转phi角度
        # 这样可以确保完整旋转一周
        
        # 第一步：绕世界坐标系的Z轴旋转phi（方位角旋转）
        rotation_axis = world_up_ref  # 绕Z轴旋转
        if np.abs(phi) > 1e-6:
            # Rodrigues旋转公式：R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
            k = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
            k_cross = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            R_phi = np.eye(3) + np.sin(phi) * k_cross + (1 - np.cos(phi)) * (k_cross @ k_cross)
        else:
            R_phi = np.eye(3)
        
        # 旋转初始forward、right和up向量
        rotated_forward = R_phi @ initial_forward_normalized
        rotated_right = R_phi @ initial_right
        rotated_up = R_phi @ initial_up
        
        # 第二步：绕旋转后的right方向旋转theta（仰角摆动）
        if np.abs(theta) > 1e-6:
            k = rotated_right
            k_cross = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            R_theta = np.eye(3) + np.sin(theta) * k_cross + (1 - np.cos(theta)) * (k_cross @ k_cross)
        else:
            R_theta = np.eye(3)
        
        # 计算最终的forward和up方向
        new_forward = R_theta @ rotated_forward
        new_forward = new_forward / (np.linalg.norm(new_forward) + 1e-8)
        
        # 计算新的up向量
        new_up = R_theta @ rotated_up
        new_up = new_up / (np.linalg.norm(new_up) + 1e-8)
        
        # 确保new_up与new_forward垂直
        new_up = new_up - np.dot(new_up, new_forward) * new_forward
        if np.linalg.norm(new_up) > 1e-6:
            new_up = new_up / np.linalg.norm(new_up)
        else:
            # 如果new_up为零，使用world_up_ref
            new_up = world_up_ref - np.dot(world_up_ref, new_forward) * new_forward
            new_up = new_up / (np.linalg.norm(new_up) + 1e-8)
        
        # 计算新的look_at点（保持距离不变）
        new_look_at = initial_camera_pos + new_forward * initial_distance

        # 更新相机参数（位置保持不变）
        vis_camera.setWorld2Camera(
            look_at=new_look_at,
            up=new_up,
            pos=initial_camera_pos,
        )

        # 重新投影
        uv = vis_camera.project_points_to_uv(points_xyz_tensor)
        valid_mask = ~torch.isnan(uv[:, 0])
        valid_points_np = points_xyz_np[valid_mask.cpu().numpy()]
        
        # 更新反投影点云
        if len(valid_points_np) > 0:
            valid_indices = torch.where(valid_mask)[0]
            valid_points_world = points_xyz_tensor[valid_indices]
            valid_uv = uv[valid_indices]
            
            # 将可见点通过 world2camera 变换到相机坐标系
            valid_points_world_homo = torch.cat([
                valid_points_world,
                torch.ones((len(valid_points_world), 1), dtype=vis_camera.dtype, device=vis_camera.device)
            ], dim=1)
            valid_points_camera_homo = torch.matmul(valid_points_world_homo, vis_camera.world2camera.T)
            valid_points_camera = valid_points_camera_homo[:, :3]
            
            # 提取深度值：使用 -z 作为深度
            depth = -valid_points_camera[:, 2]
            
            # 使用 projectUV2Points 反投影
            reprojected_points_world = vis_camera.projectUV2Points(valid_uv, depth)
            reprojected_points_np = reprojected_points_world.detach().cpu().numpy()
            
            reprojected_pcd.points = o3d.utility.Vector3dVector(reprojected_points_np)
            vis.update_geometry(reprojected_pcd)
        else:
            # 如果没有有效点，清空点云
            reprojected_pcd.points = o3d.utility.Vector3dVector([])
            vis.update_geometry(reprojected_pcd)
        
        # 每30帧打印一次可见点数量，用于验证背面不可见
        if frame % 30 == 0:
            valid_count = valid_mask.sum().item()
            print(f"帧 {frame:3d} | 相机位置: [{initial_camera_pos[0]:.2f}, {initial_camera_pos[1]:.2f}, {initial_camera_pos[2]:.2f}] | "
                  f"朝向角度: φ={np.degrees(phi):.1f}° θ={np.degrees(theta):.1f}° | "
                  f"可见点: {valid_count}/{len(points_xyz_tensor)} ({100*valid_count/len(points_xyz_tensor):.1f}%)")

        # 更新相机视锥体
        new_frustum = vis_camera.toO3DMesh()
        camera_frustum.points = new_frustum.points
        camera_frustum.lines = new_frustum.lines
        vis.update_geometry(camera_frustum)

        # 更新图像平面（使用torch）
        camera2world = vis_camera.camera2world
        corners_camera_homo = torch.cat([
            corners_camera, 
            torch.ones((4, 1), dtype=torch.float32)
        ], dim=1)
        corners_world_homo = (camera2world @ corners_camera_homo.T).T
        corners_world = corners_world_homo[:, :3].numpy()
        image_plane.vertices = o3d.utility.Vector3dVector(corners_world)
        image_plane.compute_vertex_normals()
        vis.update_geometry(image_plane)

        # 更新连线
        if len(valid_points_np) > 0:
            line_points = []
            line_indices = []
            for i, point in enumerate(valid_points_np):
                line_points.append(vis_camera.pos.numpy())
                line_points.append(point)
                line_indices.append([i * 2, i * 2 + 1])

            if lines is None:
                lines = o3d.geometry.LineSet()
                vis.add_geometry(lines)

            lines.points = o3d.utility.Vector3dVector(line_points)
            lines.lines = o3d.utility.Vector2iVector(line_indices)
            lines.colors = o3d.utility.Vector3dVector([[0.0, 0.8, 0.0] for _ in range(len(line_indices))])  # 绿色表示可见
            vis.update_geometry(lines)
        elif lines is not None:
            # 如果没有有效点，清空连线
            lines.points = o3d.utility.Vector3dVector([])
            lines.lines = o3d.utility.Vector2iVector([])
            vis.update_geometry(lines)

        vis.update_renderer()

        frame += 1
        if frame >= n_frames:
            frame = 0  # 循环播放

    vis.destroy_window()


def test():
    """
    测试UV投影和可视化

    从网格表面采样点，投影到相机UV坐标，并可视化投影过程。

    相机坐标系定义：
    - X轴：向右
    - Y轴：向上
    - Z轴：向后（相机看向 -Z 方向）

    UV坐标系定义：
    - 原点在 (0, 0) 左下角
    - u沿X轴（向右）
    - v沿Y轴（向上）
    """
    mesh_file_path = "/Users/chli/chLi/Dataset/Bunny/bunny/reconstruction/bun_zipper.ply"
    n_points = 10000

    # 创建相机：位于斜上方，看向原点，Z轴向上
    # 相机会在动画中绕物体旋转，测试各个角度的可见性
    camera = Camera(pos=[0.7, 0.7, 0.5], look_at=[0, 0, 0], up=[0, 0, 1])

    print("=" * 60)
    print("UV投影测试 - 相机旋转验证")
    print("=" * 60)
    print("\n相机初始参数:")
    camera.outputInfo()
    print("\n动画说明:")
    print("- 相机位置固定不变，相机朝向会旋转")
    print("- 相机会绕初始朝向方向旋转（方位角变化）")
    print("- 同时相机会上下摆动（仰角变化）")
    print("- 绿色连线表示可见点（相机前方，Z<0）")
    print("- 蓝色点云表示反投影的点（通过 projectUV2Points 计算）")
    print("- 相机背后的点不会有连线（返回NaN）")
    print("- 终端会每30帧输出一次可见点统计信息")
    print("=" * 60)

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    print(f"\n原始网格顶点数: {len(mesh.vertices)}")

    mesh = normalize_mesh(mesh)

    print(f"正在从网格表面采样 {n_points} 个点...")
    points_xyz = sample_points_from_mesh(mesh, n_points)

    # 投影到UV坐标
    print("正在投影点到UV坐标...")
    uv = camera.project_points_to_uv(points_xyz)

    # 打印统计信息
    valid_count = (~torch.isnan(uv[:, 0])).sum().item()
    invalid_count = torch.isnan(uv[:, 0]).sum().item()
    print(f"\n投影统计:")
    print(f"总点数: {n_points}")
    print(f"有效点（相机前面）: {valid_count}")
    print(f"无效点（相机背面，返回nan）: {invalid_count}")
    
    if valid_count > 0:
        valid_uv = uv[~torch.isnan(uv[:, 0])]
        print(f"\nUV坐标范围:")
        print(f"u: [{valid_uv[:, 0].min():.3f}, {valid_uv[:, 0].max():.3f}]")
        print(f"v: [{valid_uv[:, 1].min():.3f}, {valid_uv[:, 1].max():.3f}]")

    print(f"\n前5个点的UV坐标:")
    print(uv[:5])

    # 测试 projectUV2Points 函数的正确性
    print("\n" + "=" * 60)
    print("测试 projectUV2Points 函数正确性")
    print("=" * 60)
    
    # 获取可见点（非NaN的点）
    valid_mask = ~torch.isnan(uv[:, 0])
    valid_indices = torch.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("[WARNING] 没有可见点，无法测试 projectUV2Points")
    else:
        # 获取可见点的世界坐标
        valid_points_world = torch.from_numpy(points_xyz[valid_indices]).to(dtype=camera.dtype, device=camera.device)
        valid_uv = uv[valid_indices]
        
        print(f"\n可见点数量: {len(valid_indices)}")
        
        # 将可见点通过 world2camera 变换到相机坐标系
        valid_points_world_homo = torch.cat([
            valid_points_world,
            torch.ones((len(valid_points_world), 1), dtype=camera.dtype, device=camera.device)
        ], dim=1)
        valid_points_camera_homo = torch.matmul(valid_points_world_homo, camera.world2camera.T)
        valid_points_camera = valid_points_camera_homo[:, :3]
        
        # 提取深度值：使用 -z 作为深度（因为相机看向 -Z 方向，Z 轴向后）
        # 在相机坐标系中，可见点的 Z < 0，所以深度 = -Z > 0
        depth = -valid_points_camera[:, 2]
        
        print(f"\n相机坐标系中的点统计:")
        print(f"X范围: [{valid_points_camera[:, 0].min():.4f}, {valid_points_camera[:, 0].max():.4f}]")
        print(f"Y范围: [{valid_points_camera[:, 1].min():.4f}, {valid_points_camera[:, 1].max():.4f}]")
        print(f"Z范围: [{valid_points_camera[:, 2].min():.4f}, {valid_points_camera[:, 2].max():.4f}]")
        print(f"深度范围: [{depth.min():.4f}, {depth.max():.4f}]")
        
        # 使用 projectUV2Points 将 UV 和深度反投影回世界坐标系
        print("\n正在使用 projectUV2Points 反投影...")
        reprojected_points_world = camera.projectUV2Points(valid_uv, depth)
        
        # 计算误差
        point_errors = torch.linalg.norm(reprojected_points_world - valid_points_world, dim=1)
        mean_error = point_errors.mean().item()
        max_error = point_errors.max().item()
        min_error = point_errors.min().item()
        median_error = torch.median(point_errors).item()
        
        print(f"\n反投影误差统计:")
        print(f"平均误差: {mean_error:.6f}")
        print(f"中位数误差: {median_error:.6f}")
        print(f"最大误差: {max_error:.6f}")
        print(f"最小误差: {min_error:.6f}")
        
        # 计算相对误差（相对于点的距离）
        point_distances = torch.linalg.norm(valid_points_world, dim=1)
        relative_errors = point_errors / (point_distances + 1e-8)
        mean_relative_error = relative_errors.mean().item() * 100
        max_relative_error = relative_errors.max().item() * 100
        
        print(f"\n相对误差统计:")
        print(f"平均相对误差: {mean_relative_error:.4f}%")
        print(f"最大相对误差: {max_relative_error:.4f}%")
        
        # 检查误差是否在合理范围内（考虑到浮点精度）
        tolerance = 1e-5
        if mean_error < tolerance:
            print(f"\n✓ 测试通过！平均误差 ({mean_error:.6f}) < 容差 ({tolerance})")
        else:
            print(f"\n✗ 测试警告！平均误差 ({mean_error:.6f}) >= 容差 ({tolerance})")
        
        # 显示前几个点的详细对比
        print(f"\n前5个点的详细对比:")
        print(f"{'索引':<6} {'原始点 (世界)':<40} {'反投影点 (世界)':<40} {'误差':<12}")
        print("-" * 100)
        for i in range(min(5, len(valid_indices))):
            orig = valid_points_world[i].cpu().numpy()
            reproj = reprojected_points_world[i].cpu().numpy()
            err = point_errors[i].item()
            print(f"{i:<6} [{orig[0]:7.4f}, {orig[1]:7.4f}, {orig[2]:7.4f}]  "
                  f"[{reproj[0]:7.4f}, {reproj[1]:7.4f}, {reproj[2]:7.4f}]  {err:.6e}")
        
        print("=" * 60)

    # 使用open3d可视化动画
    print("\n正在打开可视化窗口...")
    visualize_with_open3d_animated(mesh, points_xyz, camera, n_frames=360)
    return True
