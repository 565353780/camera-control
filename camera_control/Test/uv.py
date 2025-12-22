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
    
    Args:
        mesh: open3d TriangleMesh对象
        points_xyz: 采样点，形状为 (N, 3)，torch tensor
        camera_base: 基础相机参数字典（用于提取内参）
        base_camera_pos: 基础相机位置，形状为 (3,)
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
    vis.add_geometry(mesh)
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)

    vis_camera = camera_base.clone()

    # 投影并计算有效点
    uv = vis_camera.project_points_to_uv(points_xyz_tensor)
    valid_mask = ~torch.isnan(uv[:, 0])
    valid_points_np = points_xyz_np[valid_mask.cpu().numpy()]

    # 创建相机视锥体
    camera_frustum = vis_camera.toO3DMesh()
    vis.add_geometry(camera_frustum)

    # 创建图像平面
    image_distance = 0.5
    half_width_world = (vis_camera.width / vis_camera.fx) * image_distance
    half_height_world = (vis_camera.height / vis_camera.fy) * image_distance
    corners_camera = np.array([
        [-half_width_world, half_height_world, image_distance],
        [half_width_world, half_height_world, image_distance],
        [half_width_world, -half_height_world, image_distance],
        [-half_width_world, -half_height_world, image_distance],
    ])
    corners_world = (vis_camera.rot.numpy() @ corners_camera.T).T + vis_camera.pos.numpy()

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
        lines.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3] for _ in range(len(line_indices))])
        vis.add_geometry(lines)
    else:
        lines = None

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

    while True:
        if not vis.poll_events():
            break
        
        current_time = time.time()
        if current_time - last_time < frame_duration:
            continue
        last_time = current_time
        
        # 相机位置固定，围绕Y轴旋转
        angle = 2 * np.pi * frame / n_frames
        
        # 计算围绕Y轴旋转的look_at点
        # 初始方向是从相机到原点，即(1, 0, 0)方向
        # 围绕Y轴旋转这个方向
        initial_forward = np.array([1.0, 0.0, 0.0])  # 从(-1,0,0)指向原点的方向
        
        # 围绕Y轴旋转
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Y轴旋转矩阵
        R_y = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        R_z = np.array([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        R = R_z @ R_y
        R = R_z

        # 旋转后的forward方向
        rotated_forward = R @ initial_forward

        # 计算look_at点（相机位置 + 旋转后的方向 * 距离）
        look_at_distance = 1.0  # 可以调整这个距离
        look_at = vis_camera.pos.numpy() + rotated_forward * look_at_distance

        # 更新相机参数（位置固定，朝向围绕Y轴旋转）
        vis_camera.setPoseByLookAt(look_at)

        # 重新投影
        uv = vis_camera.project_points_to_uv(points_xyz_tensor)
        valid_mask = ~torch.isnan(uv[:, 0])
        valid_points_np = points_xyz_np[valid_mask.cpu().numpy()]

        # 更新相机视锥体
        new_frustum = vis_camera.toO3DMesh()
        camera_frustum.points = new_frustum.points
        camera_frustum.lines = new_frustum.lines
        vis.update_geometry(camera_frustum)

        # 更新图像平面
        corners_world = (vis_camera.rot.numpy() @ corners_camera.T).T + vis_camera.pos.numpy()
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
            lines.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3] for _ in range(len(line_indices))])
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
    mesh_file_path = "/Users/chli/chLi/Dataset/Bunny/bunny/reconstruction/bun_zipper.ply"
    n_points = 10000

    camera = Camera(pos=[-1, 0, 0], look_at=[0, 0, 0], up=[0, 0, 1])

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    print(f"原始网格顶点数: {len(mesh.vertices)}")

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
    print(f"\n前5个点的UV坐标:")
    print(uv[:5])

    # 使用open3d可视化动画
    print("\n正在打开可视化窗口...")
    visualize_with_open3d_animated(mesh, points_xyz, camera, n_frames=360)
    return True
