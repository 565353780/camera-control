import torch
import numpy as np
import open3d as o3d
import time


def project_points_to_uv(points_xyz, camera):
    """
    使用标准相机参数将3D点投影到UV坐标
    
    Args:
        points_xyz: 世界坐标系中的点，形状为 (N, 3)，torch tensor
        camera: 相机参数字典，包含：
            - width: 图像宽度
            - height: 图像高度
            - fx: x方向焦距
            - fy: y方向焦距
            - cx: x方向主点
            - cy: y方向主点
            - R: 3x3旋转矩阵（从世界坐标系到相机坐标系）
            - t: 3x1平移向量（从世界坐标系到相机坐标系）
    
    Returns:
        uv: 图像坐标 [u, v]，形状为 (N, 2)，torch tensor
            uv的范围可以是无穷大，但相机图像范围是-0.5到0.5
            在相机背面的点（z<=0）返回nan
    """
    # 确保points_xyz是torch tensor
    if not isinstance(points_xyz, torch.Tensor):
        points_xyz = torch.tensor(points_xyz, dtype=torch.float32)
    
    # 提取相机参数并转换为torch tensor
    R = camera["R"]
    t = camera["t"]
    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]
    width = camera["width"]
    height = camera["height"]
    
    # 转换为torch tensor
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float32)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    if not isinstance(fx, torch.Tensor):
        fx = torch.tensor(fx, dtype=torch.float32)
    if not isinstance(fy, torch.Tensor):
        fy = torch.tensor(fy, dtype=torch.float32)
    if not isinstance(cx, torch.Tensor):
        cx = torch.tensor(cx, dtype=torch.float32)
    if not isinstance(cy, torch.Tensor):
        cy = torch.tensor(cy, dtype=torch.float32)
    if not isinstance(width, torch.Tensor):
        width = torch.tensor(width, dtype=torch.float32)
    if not isinstance(height, torch.Tensor):
        height = torch.tensor(height, dtype=torch.float32)
    
    # 确保t是1D向量 (3,)
    if t.ndim == 2:
        t = t.squeeze(-1)
    
    # 确保points_xyz是2D的 (N, 3)
    if points_xyz.ndim == 1:
        points_xyz = points_xyz.unsqueeze(0)
    
    # 1. 将世界坐标转换到相机坐标系
    # P_cam = R.T @ P_world + t = R.T @ (P_world - camera_pos)
    # points_xyz: (N, 3), R: (3, 3), t: (3,)
    points_camera = torch.matmul(points_xyz, R.T) + t  # (N, 3) @ (3, 3) + (3,) = (N, 3)
    
    # 2. 提取x, y, z坐标
    x, y, z = points_camera[..., 0], points_camera[..., 1], points_camera[..., 2]
    
    # 3. 透视投影到像素坐标
    # u_pixel = fx * x / z + cx
    # v_pixel = fy * y / z + cy
    z_safe = torch.where(z > 1e-8, z, torch.ones_like(z) * 1e-8)
    u_pixel = fx * x / z_safe + cx
    v_pixel = fy * y / z_safe + cy
    
    # 4. 归一化到-0.5到0.5范围
    # u_norm = (u_pixel - width/2) / width
    # v_norm = (v_pixel - height/2) / height
    u = (u_pixel - width / 2.0) / width
    v = (v_pixel - height / 2.0) / height
    
    # 5. 对于z<=0的点（在相机背面），返回nan
    invalid_mask = z <= 1e-8
    u = torch.where(invalid_mask, torch.full_like(u, float('nan')), u)
    v = torch.where(invalid_mask, torch.full_like(v, float('nan')), v)
    
    uv = torch.stack([u, v], dim=-1)  # (N, 2)
    
    return uv


def normalize_mesh(mesh):
    """
    将网格归一化到-0.5到0.5的单位空间中
    
    Args:
        mesh: open3d TriangleMesh对象
    
    Returns:
        normalized_mesh: 归一化后的网格
    """
    vertices = np.asarray(mesh.vertices)
    
    # 计算边界框
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    scale = (max_bound - min_bound).max()
    
    # 归一化：先平移到中心，再缩放到-0.5到0.5
    if scale > 1e-8:
        # 除以scale使最大维度变为1，再乘以0.5使范围变为[-0.5, 0.5]
        normalized_vertices = (vertices - center) / scale * 0.5
    else:
        normalized_vertices = vertices - center
    
    # 创建新的网格
    normalized_mesh = o3d.geometry.TriangleMesh()
    normalized_mesh.vertices = o3d.utility.Vector3dVector(normalized_vertices)
    normalized_mesh.triangles = mesh.triangles
    normalized_mesh.vertex_normals = mesh.vertex_normals
    normalized_mesh.triangle_normals = mesh.triangle_normals
    
    return normalized_mesh


def sample_points_from_mesh(mesh, n_points):
    """
    从网格表面采样点
    
    Args:
        mesh: open3d TriangleMesh对象
        n_points: 采样点数
    
    Returns:
        points: 采样点，形状为 (n_points, 3)，numpy数组
    """
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    points = np.asarray(pcd.points)
    return points


def create_camera_at_position(camera_pos, look_at=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    """
    创建位于指定位置的相机参数
    
    Args:
        camera_pos: 相机位置，形状为 (3,)，numpy数组或torch tensor
        look_at: 相机看向的点，形状为 (3,)，默认[0,0,0]
        up: 相机的上方向，形状为 (3,)，默认[0,1,0]
    
    Returns:
        camera: 相机参数字典
    """
    # 转换为numpy
    if isinstance(camera_pos, torch.Tensor):
        camera_pos = camera_pos.detach().cpu().numpy()
    if isinstance(look_at, torch.Tensor):
        look_at = look_at.detach().cpu().numpy()
    if isinstance(up, torch.Tensor):
        up = up.detach().cpu().numpy()
    
    camera_pos = np.array(camera_pos).flatten()
    look_at = np.array(look_at).flatten()
    up = np.array(up).flatten()
    
    # 计算相机坐标系
    # z轴：从相机指向look_at（相机坐标系中z轴指向相机前方）
    forward = look_at - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    # x轴：right = forward × up
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    # y轴：up = right × forward（重新计算以确保正交）
    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-8)
    
    # 构建旋转矩阵（从世界坐标系到相机坐标系）
    # R的列向量表示相机坐标系的x, y, z轴在世界坐标系中的表示
    # 相机坐标系：x向右，y向上，z向前（指向相机前方）
    R = np.column_stack([right, up_corrected, forward])
    
    # 平移向量：正确的转换应该是 P_cam = R.T @ (P_world - camera_pos)
    # 展开：P_cam = R.T @ P_world - R.T @ camera_pos
    # 所以 t = -R.T @ camera_pos
    t = -R.T @ camera_pos
    
    return R, t


def create_camera_frustum(camera_pos, R_np, width, height, fx, fy, far=0.1):
    """
    创建相机视锥体（四棱锥）
    
    Args:
        camera_pos: 相机位置，形状为 (3,)
        R_np: 旋转矩阵，形状为 (3, 3)
        width: 图像宽度
        height: 图像高度
        fx: x方向焦距
        fy: y方向焦距
        far: 远平面距离（图像平面距离）
    
    Returns:
        frustum: 视锥体网格
    """
    # 计算图像平面的四个角点（在相机坐标系中）
    half_width = (width / fx) * far
    half_height = (height / fy) * far
    
    # 近平面和远平面的角点（相机坐标系）
    far_corners = np.array([
        [-half_width, half_height, far],
        [half_width, half_height, far],
        [half_width, -half_height, far],
        [-half_width, -half_height, far],
    ])
    
    # 转换到世界坐标系
    far_corners_world = (R_np @ far_corners.T).T + camera_pos
    
    # 创建视锥体（四棱锥）
    vertices = np.vstack([far_corners_world, camera_pos.reshape(1, 3)])
    # 顶点索引：0-3远平面，4相机位置
    
    # 创建三角形面
    triangles = []
    # 远平面
    triangles.append([0, 2, 1])
    triangles.append([0, 3, 2])
    # 四个侧面（从相机位置到远平面的四个角）
    triangles.append([4, 0, 1])  # 左-上
    triangles.append([4, 1, 2])  # 上-右
    triangles.append([4, 2, 3])  # 右-下
    triangles.append([4, 3, 0])  # 下-左
   
    frustum = o3d.geometry.TriangleMesh()
    frustum.vertices = o3d.utility.Vector3dVector(vertices)
    frustum.triangles = o3d.utility.Vector3iVector(triangles)
    frustum.paint_uniform_color([0, 0, 1])  # 蓝色
    frustum.compute_vertex_normals()
    
    return frustum


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


def visualize_with_open3d_animated(mesh, points_xyz, camera_base, base_camera_pos, n_frames=360):
    """
    使用open3d可视化网格、采样点、相机位置、图像平面和连线，并创建旋转动画
    
    Args:
        mesh: open3d TriangleMesh对象
        points_xyz: 采样点，形状为 (N, 3)，torch tensor
        camera_base: 基础相机参数字典（用于提取内参）
        base_camera_pos: 基础相机位置，形状为 (3,)
        n_frames: 动画帧数
    """
    # 转换为numpy
    if isinstance(points_xyz, torch.Tensor):
        points_xyz_np = points_xyz.detach().cpu().numpy()
    else:
        points_xyz_np = points_xyz
    
    # 提取相机内参
    width = camera_base["width"]
    height = camera_base["height"]
    fx = camera_base["fx"]
    fy = camera_base["fy"]
    
    if isinstance(width, torch.Tensor):
        width = width.item()
    if isinstance(height, torch.Tensor):
        height = height.item()
    if isinstance(fx, torch.Tensor):
        fx = fx.item()
    if isinstance(fy, torch.Tensor):
        fy = fy.item()
    
    # 创建点云（固定，不随相机旋转变化）
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
    
    # 相机位置固定在(-1, 0, 0)，相机围绕Y轴旋转
    camera_pos = base_camera_pos.copy()  # 固定位置(-1, 0, 0)
    
    # 初始角度（相机初始朝向原点）
    initial_angle = 0.0
    
    # 创建初始相机（初始朝向原点）
    look_at = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])  # Y轴向上
    R_np, t_np = create_camera_at_position(camera_pos, look_at=look_at, up=up)
    R = torch.from_numpy(R_np).float()
    t = torch.from_numpy(t_np).float()
    
    camera = {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "R": R,
        "t": t,
    }
    
    # 投影并计算有效点
    uv = project_points_to_uv(points_xyz, camera)
    valid_mask = ~torch.isnan(uv[:, 0])
    valid_points = points_xyz_np[valid_mask]
    
    # 创建相机视锥体
    camera_frustum = create_camera_frustum(camera_pos, R_np, width, height, fx, fy)
    vis.add_geometry(camera_frustum)
    
    # 创建图像平面
    image_distance = 0.5
    half_width_world = (width / fx) * image_distance
    half_height_world = (height / fy) * image_distance
    corners_camera = np.array([
        [-half_width_world, half_height_world, image_distance],
        [half_width_world, half_height_world, image_distance],
        [half_width_world, -half_height_world, image_distance],
        [-half_width_world, -half_height_world, image_distance],
    ])
    corners_world = (R_np @ corners_camera.T).T + camera_pos
    
    image_plane = o3d.geometry.TriangleMesh()
    image_plane.vertices = o3d.utility.Vector3dVector(corners_world)
    image_plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    image_plane.paint_uniform_color([1, 1, 0])
    image_plane.compute_vertex_normals()
    vis.add_geometry(image_plane)
    
    # 创建连线
    if len(valid_points) > 0:
        line_points = []
        line_indices = []
        for i, point in enumerate(valid_points):
            line_points.append(camera_pos)
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
        
        # 旋转后的forward方向
        rotated_forward = R_y @ initial_forward
        
        # 计算look_at点（相机位置 + 旋转后的方向 * 距离）
        look_at_distance = 1.0  # 可以调整这个距离
        look_at = camera_pos + rotated_forward * look_at_distance
        
        # up向量保持为Y轴方向
        up = np.array([0.0, 1.0, 0.0])
        
        # 更新相机参数（位置固定，朝向围绕Y轴旋转）
        R_np, t_np = create_camera_at_position(camera_pos, look_at=look_at, up=up)
        R = torch.from_numpy(R_np).float()
        t = torch.from_numpy(t_np).float()
        
        camera = {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "R": R,
            "t": t,
        }
        
        # 重新投影
        uv = project_points_to_uv(points_xyz, camera)
        valid_mask = ~torch.isnan(uv[:, 0])
        valid_points = points_xyz_np[valid_mask]
        
        # 更新相机视锥体
        new_frustum = create_camera_frustum(camera_pos, R_np, width, height, fx, fy)
        camera_frustum.vertices = new_frustum.vertices
        camera_frustum.triangles = new_frustum.triangles
        camera_frustum.compute_vertex_normals()
        vis.update_geometry(camera_frustum)
        
        # 更新图像平面
        corners_world = (R_np @ corners_camera.T).T + camera_pos
        image_plane.vertices = o3d.utility.Vector3dVector(corners_world)
        image_plane.compute_vertex_normals()
        vis.update_geometry(image_plane)
        
        # 更新连线
        if len(valid_points) > 0:
            line_points = []
            line_indices = []
            for i, point in enumerate(valid_points):
                line_points.append(camera_pos)
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


if __name__ == "__main__":
    mesh_file_path = "/Users/chli/chLi/Dataset/Bunny/bunny/reconstruction/bun_zipper.ply"
    
    # 读取网格
    print(f"正在读取网格: {mesh_file_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    if len(mesh.vertices) == 0:
        raise ValueError(f"无法读取网格文件: {mesh_file_path}")
    print(f"原始网格顶点数: {len(mesh.vertices)}")
    
    # 归一化网格到-0.5到0.5的单位空间
    print("正在归一化网格...")
    mesh = normalize_mesh(mesh)
    print("网格已归一化到-0.5到0.5的单位空间")
    
    # 从mesh表面采样点
    n_points = 10000
    print(f"正在从网格表面采样 {n_points} 个点...")
    points_xyz_np = sample_points_from_mesh(mesh, n_points)
    points_xyz = torch.from_numpy(points_xyz_np).float()
    print(f"采样完成，共 {len(points_xyz)} 个点")
    
    # 设置相机位置在(-1, 0, 0)
    camera_pos = np.array([-1.0, 0.0, 0.0])
    print(f"相机位置: {camera_pos}")
    
    # 创建相机参数
    W, H = 640, 480
    fx, fy = 500.0, 500.0
    cx, cy = W / 2.0, H / 2.0
    
    # 计算相机旋转和平移
    R_np, t_np = create_camera_at_position(camera_pos)
    R = torch.from_numpy(R_np).float()
    t = torch.from_numpy(t_np).float()
    
    camera = {
        "width": W,
        "height": H,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "R": R,
        "t": t,
    }
    
    # 投影到UV坐标
    print("正在投影点到UV坐标...")
    uv = project_points_to_uv(points_xyz, camera)
    
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
    visualize_with_open3d_animated(mesh, points_xyz, camera, camera_pos, n_frames=360)
