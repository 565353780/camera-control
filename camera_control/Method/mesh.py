import trimesh
import numpy as np
import open3d as o3d


def normalizeMesh(mesh: trimesh.Trimesh, target_length: float = 0.99) -> trimesh.Trimesh:
    """
    将mesh通过平移和缩放，归一化到原点为bbox中心，bbox最长边为target_length

    Args:
        mesh: 输入的trimesh.Trimesh对象
        target_length: 归一化后bbox的最长边

    Returns:
        归一化后的trimesh.Trimesh对象
    """
    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_size = bbox_max - bbox_min
    max_length = bbox_size.max()

    # 平移
    vertices = mesh.vertices - bbox_center

    # 缩放
    scale = target_length / (max_length if max_length > 0 else 1.0)
    vertices = vertices * scale

    normed_mesh = mesh.copy()
    normed_mesh.vertices = vertices

    return normed_mesh

def normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
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
