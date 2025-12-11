import torch
import open3d as o3d

from uv_cube_gen.Method.mesh import normalize_mesh, sample_points_from_mesh
from uv_cube_gen.Module.camera import Camera


def test():
    mesh_file_path = "/Users/chli/chLi/Dataset/Bunny/bunny/reconstruction/bun_zipper.ply"
    n_points = 10000

    camera = Camera(pos=[-1, 0, 0], look_at=[0, 0, 0], up=[0, 1, 0])

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

    camera.outputInfo()

    new_camera = Camera.fromUVPoints(points_xyz, uv)
    assert new_camera is not None

    new_camera.outputInfo()

    return True
