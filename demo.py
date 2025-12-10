import torch
import numpy as np
import open3d as o3d


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


def visualize_with_open3d(mesh, points_xyz, uv, camera, camera_pos):
    """
    使用open3d可视化网格、采样点、相机位置、图像平面和连线
    
    Args:
        mesh: open3d TriangleMesh对象
        points_xyz: 采样点，形状为 (N, 3)，numpy数组或torch tensor
        uv: UV坐标，形状为 (N, 2)，torch tensor
        camera: 相机参数字典
        camera_pos: 相机位置，形状为 (3,)
    """
    # 转换为numpy
    if isinstance(points_xyz, torch.Tensor):
        points_xyz_np = points_xyz.detach().cpu().numpy()
    else:
        points_xyz_np = points_xyz
    
    if isinstance(uv, torch.Tensor):
        uv_np = uv.detach().cpu().numpy()
    else:
        uv_np = uv
    
    # 提取相机参数
    width = camera["width"]
    height = camera["height"]
    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]
    R = camera["R"]
    
    if isinstance(width, torch.Tensor):
        width = width.item()
    if isinstance(height, torch.Tensor):
        height = height.item()
    if isinstance(fx, torch.Tensor):
        fx = fx.item()
    if isinstance(fy, torch.Tensor):
        fy = fy.item()
    if isinstance(cx, torch.Tensor):
        cx = cx.item()
    if isinstance(cy, torch.Tensor):
        cy = cy.item()
    if isinstance(R, torch.Tensor):
        R_np = R.detach().cpu().numpy()
    else:
        R_np = R
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz_np)
    
    # 根据UV有效性设置颜色
    valid_mask = ~np.isnan(uv_np[:, 0])
    colors = np.zeros((len(points_xyz_np), 3))
    colors[valid_mask] = [0, 1, 0]  # 绿色：有效点
    colors[~valid_mask] = [1, 0, 0]  # 红色：相机背面点
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建相机位置可视化（蓝色球体）
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    camera_sphere.translate(camera_pos)
    camera_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    
    # 创建图像平面
    # 图像平面在相机坐标系中位于z=1的位置（假设焦距为1的归一化单位）
    # 图像平面的大小由width和height以及fx, fy决定
    # 在相机坐标系中，图像平面的四个角点：
    # 左上: (-width/2/fx, height/2/fy, 1)
    # 右上: (width/2/fx, height/2/fy, 1)
    # 右下: (width/2/fx, -height/2/fy, 1)
    # 左下: (-width/2/fx, -height/2/fy, 1)
    
    # 使用一个合理的距离（例如0.5）来显示图像平面
    image_distance = 0.5
    half_width_world = (width / fx) * image_distance
    half_height_world = (height / fy) * image_distance
    
    # 在相机坐标系中的四个角点
    corners_camera = np.array([
        [-half_width_world, half_height_world, image_distance],   # 左上
        [half_width_world, half_height_world, image_distance],    # 右上
        [half_width_world, -half_height_world, image_distance],   # 右下
        [-half_width_world, -half_height_world, image_distance],  # 左下
    ])
    
    # 转换到世界坐标系
    # P_world = R @ P_cam + camera_pos
    corners_world = (R_np @ corners_camera.T).T + camera_pos
    
    # 创建图像平面（使用LineSet绘制边框）
    image_lines = o3d.geometry.LineSet()
    image_lines.points = o3d.utility.Vector3dVector(corners_world)
    image_lines.lines = o3d.utility.Vector2iVector([
        [0, 1], [1, 2], [2, 3], [3, 0]  # 四个边
    ])
    image_lines.colors = o3d.utility.Vector3dVector([[1, 1, 0] for _ in range(4)])  # 黄色边框
    
    # 创建图像平面（半透明平面）
    image_plane = o3d.geometry.TriangleMesh()
    image_plane.vertices = o3d.utility.Vector3dVector(corners_world)
    image_plane.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2], [0, 2, 3]  # 两个三角形组成平面
    ])
    image_plane.paint_uniform_color([1, 1, 0])  # 黄色
    image_plane.compute_vertex_normals()
    
    # 创建点到相机的连线（只显示有效点）
    line_points = []
    line_indices = []
    valid_points = points_xyz_np[valid_mask]
    
    for i, point in enumerate(valid_points):
        line_points.append(camera_pos)
        line_points.append(point)
        line_indices.append([i * 2, i * 2 + 1])
    
    if len(line_points) > 0:
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(line_points)
        lines.lines = o3d.utility.Vector2iVector(line_indices)
        lines.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(len(line_indices))])  # 灰色连线
    else:
        lines = None
    
    # 可视化所有对象
    geometries = [mesh, pcd, camera_sphere, image_lines, image_plane]
    if lines is not None:
        geometries.append(lines)
    
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Mesh, Points, Camera and Image Plane",
                                      width=1200, height=800)


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
    
    # 使用open3d可视化
    print("\n正在打开可视化窗口...")
    visualize_with_open3d(mesh, points_xyz, uv, camera, camera_pos)
