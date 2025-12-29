import os
import trimesh
from typing import Optional


def loadMeshFile(
    mesh_file_path: str,
) -> Optional[trimesh.Trimesh]:
    """
    加载三角网格文件，支持多种格式（包括glb、obj等）

    Args:
        mesh_file_path: 网格文件路径
        force_merge: 当加载的是Scene时，是否强制合并为单个Trimesh（默认True）

    Returns:
        trimesh.Trimesh对象，如果失败则返回None
    """
    if not os.path.exists(mesh_file_path):
        print('[ERROR][io::loadMeshFile]')
        print('\t mesh file not exist!')
        print('\t mesh_file_path:', mesh_file_path)
        return None

    mesh = trimesh.load(mesh_file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    # 确保是Trimesh类型
    if not isinstance(mesh, trimesh.Trimesh):
        print('[ERROR][io::loadMeshFile]')
        print(f'\t Loaded object is not a Trimesh, got type: {type(mesh)}')
        return None

    # 计算顶点法线（如果不存在）
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.vertex_normals = mesh.vertex_normals  # 这会触发自动计算

    return mesh
