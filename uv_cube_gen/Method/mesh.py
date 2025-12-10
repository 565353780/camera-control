import numpy as np
import open3d as o3d


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
