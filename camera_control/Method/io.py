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

def _sanitize_mesh(
    mesh: trimesh.Trimesh,
    print_progress: bool=False,
) -> trimesh.Trimesh:
    """Remove degenerate faces, non-finite vertices, and unreferenced vertices
    so that downstream CUDA renderers (nvdiffrast) never receive bad geometry.

    Operates on a copy; the original mesh is not mutated.  Returns the
    original mesh unchanged when no problems are found (fast path).
    """
    verts = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces).copy()
    n_verts_orig, n_faces_orig = verts.shape[0], faces.shape[0]

    if n_verts_orig == 0 or n_faces_orig == 0:
        return mesh

    # --- 1. Mark non-finite vertices and remove faces that reference them ---
    vert_valid = np.isfinite(verts).all(axis=1)
    if not vert_valid.all():
        if print_progress:
            print(f'[WARN][io::_sanitize_mesh] {(~vert_valid).sum()} non-finite vertices')
        face_ok = vert_valid[faces].all(axis=1)
        faces = faces[face_ok]

    # --- 2. Remove faces with out-of-bound indices ---
    oob = (faces < 0) | (faces >= verts.shape[0])
    if oob.any():
        bad = oob.any(axis=1)
        if print_progress:
            print(f'[WARN][io::_sanitize_mesh] {bad.sum()} faces with out-of-bound indices')
        faces = faces[~bad]

    # --- 3. Remove degenerate faces (any two vertex indices equal) ---
    degen = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    if degen.any():
        if print_progress:
            print(f'[WARN][io::_sanitize_mesh] {degen.sum()} degenerate faces')
        faces = faces[~degen]

    # --- fast path: nothing was removed ---
    if faces.shape[0] == n_faces_orig and vert_valid.all():
        return mesh

    if faces.shape[0] == 0:
        if print_progress:
            print('[ERROR][io::_sanitize_mesh] 0 faces remaining after cleanup')
        return mesh

    # --- 4. Compact: remove unreferenced vertices & remap face indices ---
    referenced = np.zeros(verts.shape[0], dtype=bool)
    referenced[faces.ravel()] = True
    old_to_new = np.full(verts.shape[0], -1, dtype=np.int64)
    new_ids = np.where(referenced)[0]
    old_to_new[new_ids] = np.arange(new_ids.shape[0])
    verts = verts[new_ids]
    faces = old_to_new[faces]

    # --- 5. Rebuild mesh, preserving per-vertex attributes when possible ---
    new_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # --- 5a. Migrate vertex_normals ---
    try:
        normals = mesh._cache.get('vertex_normals')
        if normals is not None:
            normals = np.asarray(normals)
            if normals.shape[0] == n_verts_orig:
                new_mesh.vertex_normals = normals[new_ids]
    except Exception:
        pass

    # --- 5b. Migrate vertex_colors ---
    try:
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = np.asarray(mesh.visual.vertex_colors)
            if colors.shape[0] == n_verts_orig:
                new_mesh.visual.vertex_colors = colors[new_ids]
    except Exception:
        pass

    # --- 5c. Migrate UV + material (texture) ---
    try:
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv = np.asarray(mesh.visual.uv)
            if uv.shape[0] == n_verts_orig:
                new_visual_kwargs = {'uv': uv[new_ids]}
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    new_visual_kwargs['material'] = mesh.visual.material
                new_mesh.visual = trimesh.visual.TextureVisuals(**new_visual_kwargs)
            else:
                if print_progress:
                    print(
                        f'[WARN][io::_sanitize_mesh] UV count ({uv.shape[0]}) != '
                        f'vertex count ({n_verts_orig}), cannot remap UV. '
                        f'Texture will be lost.'
                    )
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    try:
                        new_mesh.visual = trimesh.visual.TextureVisuals(
                            material=mesh.visual.material)
                    except Exception:
                        pass
    except Exception:
        pass

    n_removed = n_faces_orig - faces.shape[0]
    if n_removed > 0:
        if print_progress:
            print(
                f'[INFO][io::_sanitize_mesh] Cleaned: '
                f'{n_verts_orig}v/{n_faces_orig}f -> {verts.shape[0]}v/{faces.shape[0]}f '
                f'({n_removed} faces removed)'
            )

    return new_mesh


def postProcessMesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    print_progress: bool=False,
) -> Optional[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    if not isinstance(mesh, trimesh.Trimesh):
        print('[ERROR][io::postProcessMesh]')
        print(f'\t Loaded object is not a Trimesh, got type: {type(mesh)}')
        return None

    mesh = _sanitize_mesh(mesh, print_progress)

    # Access vertex_normals to trigger lazy computation if not already cached.
    _ = mesh.vertex_normals

    return mesh

def loadMeshStream(
    mesh_stream: io.BytesIO,
    file_type: str,
    print_progress: bool=False,
) -> Optional[trimesh.Trimesh]:
    try:
        mesh_stream.seek(0)
        mesh = trimesh.load(mesh_stream, file_type=file_type, process=False)
    except Exception as e:
        print('[ERROR][io::loadMeshStream]')
        print('\t Failed to load mesh from stream:', e)
        return None

    mesh = postProcessMesh(mesh, print_progress)
    if mesh is None:
        print('[ERROR][io::loadMeshStream]')
        print('\t postProcessMesh failed!')

    return mesh

def loadMeshFile(
    mesh_file_path: str,
    print_progress: bool=False,
) -> Optional[trimesh.Trimesh]:
    if not os.path.exists(mesh_file_path):
        print('[ERROR][io::loadMeshFile]')
        print('\t mesh file not exist!')
        print('\t mesh_file_path:', mesh_file_path)
        return None

    mesh = trimesh.load(mesh_file_path, process=False)

    mesh = postProcessMesh(mesh, print_progress)
    if mesh is None:
        print('[ERROR][io::loadMeshFile]')
        print('\t postProcessMesh failed!')

    return mesh
