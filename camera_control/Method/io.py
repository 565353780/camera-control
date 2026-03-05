import io
import os
import cv2
import trimesh
import numpy as np

from typing import Optional, Union


def loadImage(
    image_file_path: str,
    is_gray: bool=False,
) -> Optional[np.ndarray]:
    imread_flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR

    image_data = cv2.imread(image_file_path, imread_flag)

    if image_data is None:
        print('[ERROR][io::loadImage]')
        print('\t imread failed!')
        print('\t image_file_path:', image_file_path)
        return None

    return image_data

def postProcessMesh(mesh: Union[trimesh.Trimesh, trimesh.Scene],
) -> Optional[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    if not isinstance(mesh, trimesh.Trimesh):
        print('[ERROR][io::postProcessMesh]')
        print(f'\t Loaded object is not a Trimesh, got type: {type(mesh)}')
        return None

    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.vertex_normals = mesh.vertex_normals

    return mesh

def loadMeshStream(
    mesh_stream: io.BytesIO,
    file_type: str,
) -> Optional[trimesh.Trimesh]:
    try:
        mesh_stream.seek(0)
        mesh = trimesh.load(mesh_stream, file_type=file_type, process=False)
    except Exception as e:
        print('[ERROR][io::loadMeshStream]')
        print('\t Failed to load mesh from stream:', e)
        return None

    mesh = postProcessMesh(mesh)
    if mesh is None:
        print('[ERROR][io::loadMeshStream]')
        print('\t postProcessMesh failed!')

    return mesh

def loadMeshFile(
    mesh_file_path: str,
) -> Optional[trimesh.Trimesh]:
    if not os.path.exists(mesh_file_path):
        print('[ERROR][io::loadMeshFile]')
        print('\t mesh file not exist!')
        print('\t mesh_file_path:', mesh_file_path)
        return None

    mesh = trimesh.load(mesh_file_path, process=False)

    mesh = postProcessMesh(mesh)
    if mesh is None:
        print('[ERROR][io::loadMeshFile]')
        print('\t postProcessMesh failed!')

    return mesh
