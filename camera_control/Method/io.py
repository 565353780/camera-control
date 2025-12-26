import os
import trimesh
import numpy as np
from typing import Optional


def loadMeshFile(
    mesh_file_path: str,
    color: list=[178, 178, 178],
) -> Optional[trimesh.Trimesh]:
    if not os.path.exists(mesh_file_path):
        print('[ERROR][io::loadMeshFile]')
        print('\t mesh file not exist!')
        print('\t mesh_file_path:', mesh_file_path)
        return None

    mesh = trimesh.load(mesh_file_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        print('[ERROR][NVDiffRastRenderer::loadMeshFile]')
        print('\t load mesh failed!')
        print('\t mesh_file_path:', mesh_file_path)
        return False

    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        num_verts = len(mesh.vertices)
        vertex_colors = np.tile(np.array(color), (num_verts, 1))
        mesh.visual.vertex_colors = vertex_colors

    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.compute_vertex_normals()

    return mesh
