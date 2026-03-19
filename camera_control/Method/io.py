import io
import os
import cv2
import json
import struct
import trimesh
import numpy as np
from PIL import Image

from typing import Optional, Tuple, Union


_GL_CLAMP_TO_EDGE = 33071
_GL_MIRRORED_REPEAT = 33648
_GL_REPEAT = 10497

_WRAP_MODE_MAP = {
    _GL_CLAMP_TO_EDGE: 'clamp',
    _GL_MIRRORED_REPEAT: 'mirrored_repeat',
    _GL_REPEAT: 'repeat',
}


def _parseGlbJson(data: bytes) -> Optional[dict]:
    """Extract the JSON chunk from a GLB (glTF 2.0 binary) payload."""
    if len(data) < 12:
        return None
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67:  # 'glTF'
        return None
    if len(data) < 20:
        return None
    chunk_length, chunk_type = struct.unpack_from('<II', data, 12)
    if chunk_type != 0x4E4F534A:  # 'JSON'
        return None
    json_bytes = data[20:20 + chunk_length]
    try:
        return json.loads(json_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _extractGlbUVWrapMode(
    data: bytes,
    print_progress: bool = False,
) -> Tuple[str, str]:
    """Return (wrap_s, wrap_t) as string mode names from GLB sampler metadata.

    Falls back to 'clamp' when the GLB cannot be parsed or contains no sampler.
    """
    gltf = _parseGlbJson(data)
    if gltf is None:
        return 'clamp', 'clamp'

    samplers = gltf.get('samplers', [])
    if not samplers:
        return 'clamp', 'clamp'

    sampler = samplers[0]
    wrap_s_val = sampler.get('wrapS', _GL_REPEAT)
    wrap_t_val = sampler.get('wrapT', _GL_REPEAT)

    wrap_s = _WRAP_MODE_MAP.get(wrap_s_val, 'clamp')
    wrap_t = _WRAP_MODE_MAP.get(wrap_t_val, 'clamp')

    if print_progress:
        print(f'[INFO][io::_extractGlbUVWrapMode] wrapS={wrap_s_val}({wrap_s}), wrapT={wrap_t_val}({wrap_t})')

    return wrap_s, wrap_t


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

_TEX_MAX_SIZE = 65536  # nvdiffrast TEX_MAX_MIP_LEVEL=16 → 2^16

def _clamp_texture(mesh: trimesh.Trimesh, max_size: int, print_progress: bool) -> trimesh.Trimesh:
    """Resize texture images that exceed nvdiffrast's maximum dimension."""
    try:
        mat = getattr(mesh.visual, 'material', None)
        if mat is None:
            return mesh

        for attr in ('baseColorTexture', 'image'):
            img = getattr(mat, attr, None)
            if img is None:
                continue
            arr = np.asarray(img)
            h, w = arr.shape[:2]
            if h <= max_size and w <= max_size:
                continue
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            if print_progress:
                print(
                    f'[WARN][io::_clamp_texture] Resizing texture '
                    f'{w}x{h} -> {new_w}x{new_h} (max {max_size})'
                )
            resized = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
            setattr(mat, attr, resized)
    except Exception:
        pass
    return mesh

def _sanitize_mesh(
    mesh: trimesh.Trimesh,
    print_progress: bool=False,
) -> Optional[trimesh.Trimesh]:
    """Remove degenerate faces, non-finite vertices, and unreferenced vertices
    so that downstream CUDA renderers (nvdiffrast) never receive bad geometry.

    Operates on a copy; the original mesh is not mutated.  Returns the
    original mesh unchanged when no problems are found (fast path).
    """
    verts = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces).copy()
    n_verts_orig, n_faces_orig = verts.shape[0], faces.shape[0]

    if n_verts_orig == 0 or n_faces_orig == 0:
        print('[ERROR][io::_sanitize_mesh] Input mesh has 0 vertices or 0 faces')
        return None

    # --- 1. Mark non-finite vertices and remove faces that reference them ---
    vert_valid = np.isfinite(verts).all(axis=1)
    if not vert_valid.all():
        if print_progress:
            print(f'[WARN][io::_sanitize_mesh] {(~vert_valid).sum()} non-finite vertices')
        face_ok = vert_valid[faces].all(axis=1)
        faces = faces[face_ok]

    # --- 3. Remove faces with out-of-bound indices ---
    oob = (faces < 0) | (faces >= verts.shape[0])
    if oob.any():
        bad = oob.any(axis=1)
        if print_progress:
            print(f'[WARN][io::_sanitize_mesh] {bad.sum()} faces with out-of-bound indices')
        faces = faces[~bad]

    # --- 4. Remove degenerate faces (any two vertex indices equal) ---
    degen = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    if degen.any():
        if print_progress:
            print(f'[WARN][io::_sanitize_mesh] {degen.sum()} degenerate faces')
        faces = faces[~degen]

    # --- 5. Remove zero-area faces ---
    if faces.shape[0] > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area_sq = (cross * cross).sum(axis=1)
        zero_area = area_sq < 1e-20
        if zero_area.any():
            if print_progress:
                print(f'[WARN][io::_sanitize_mesh] {zero_area.sum()} zero-area faces')
            faces = faces[~zero_area]

    # --- fast path: nothing was removed and no clamp happened ---
    no_changes = (
        faces.shape[0] == n_faces_orig
        and vert_valid.all()
    )
    if no_changes:
        return mesh

    if faces.shape[0] == 0:
        print('[ERROR][io::_sanitize_mesh] 0 faces remaining after cleanup')
        return None

    # --- 6. Compact: remove unreferenced vertices & remap face indices ---
    referenced = np.zeros(verts.shape[0], dtype=bool)
    referenced[faces.ravel()] = True
    old_to_new = np.full(verts.shape[0], -1, dtype=np.int64)
    new_ids = np.where(referenced)[0]
    old_to_new[new_ids] = np.arange(new_ids.shape[0])
    verts = verts[new_ids]
    faces = old_to_new[faces]

    # --- 7. Rebuild mesh, preserving per-vertex attributes when possible ---
    new_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # --- 7a. Migrate vertex_normals ---
    try:
        normals = mesh._cache.get('vertex_normals')
        if normals is not None:
            normals = np.asarray(normals)
            if normals.shape[0] == n_verts_orig:
                new_mesh.vertex_normals = normals[new_ids]
    except Exception:
        pass

    # --- 7b. Migrate vertex_colors ---
    try:
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = np.asarray(mesh.visual.vertex_colors)
            if colors.shape[0] == n_verts_orig:
                new_mesh.visual.vertex_colors = colors[new_ids]
    except Exception:
        pass

    # --- 7c. Migrate UV + material (texture) ---
    try:
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv = np.asarray(mesh.visual.uv)
            if uv.shape[0] == n_verts_orig:
                new_uv = uv[new_ids]
                if not np.isfinite(new_uv).all():
                    new_uv = np.nan_to_num(new_uv, nan=0.0, posinf=1.0, neginf=0.0)
                new_visual_kwargs = {'uv': new_uv}
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
    max_texture_size: int=_TEX_MAX_SIZE,
    uv_wrap_mode: Optional[Tuple[str, str]]=None,
) -> Optional[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    if not isinstance(mesh, trimesh.Trimesh):
        print('[ERROR][io::postProcessMesh]')
        print(f'\t Loaded object is not a Trimesh, got type: {type(mesh)}')
        return None

    mesh = _clamp_texture(mesh, max_texture_size, print_progress)
    mesh = _sanitize_mesh(mesh, print_progress)
    if mesh is None:
        return None

    if uv_wrap_mode is not None:
        mesh.metadata['uv_wrap_mode'] = uv_wrap_mode

    normals = np.array(mesh.vertex_normals, copy=True)
    bad_mask = ~np.isfinite(normals).all(axis=1)
    norm_len = np.linalg.norm(normals, axis=1)
    bad_mask |= norm_len < 1e-12
    if bad_mask.any():
        n_bad = bad_mask.sum()
        if print_progress:
            print(f'[WARN][io::postProcessMesh] {n_bad} invalid vertex normals, replacing with random unit vectors')
        rng = np.random.default_rng(0)
        rand_normals = rng.standard_normal((n_bad, 3))
        rand_normals /= np.linalg.norm(rand_normals, axis=1, keepdims=True)
        normals[bad_mask] = rand_normals
        mesh.vertex_normals = normals

    return mesh

def loadMeshStream(
    mesh_stream: io.BytesIO,
    file_type: str,
    print_progress: bool=False,
) -> Optional[trimesh.Trimesh]:
    uv_wrap_mode = None
    if file_type.lower() in ('glb', 'gltf'):
        mesh_stream.seek(0)
        raw = mesh_stream.read()
        uv_wrap_mode = _extractGlbUVWrapMode(raw, print_progress)
        mesh_stream.seek(0)

    try:
        mesh_stream.seek(0)
        mesh = trimesh.load(mesh_stream, file_type=file_type, process=False)
    except Exception as e:
        print('[ERROR][io::loadMeshStream]')
        print('\t Failed to load mesh from stream:', e)
        return None

    mesh = postProcessMesh(mesh, print_progress, uv_wrap_mode=uv_wrap_mode)
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

    uv_wrap_mode = None
    ext = os.path.splitext(mesh_file_path)[1].lower()
    if ext in ('.glb', '.gltf'):
        try:
            with open(mesh_file_path, 'rb') as f:
                raw = f.read()
            uv_wrap_mode = _extractGlbUVWrapMode(raw, print_progress)
        except Exception:
            pass

    mesh = trimesh.load(mesh_file_path, process=False)

    mesh = postProcessMesh(mesh, print_progress, uv_wrap_mode=uv_wrap_mode)
    if mesh is None:
        print('[ERROR][io::loadMeshFile]')
        print('\t postProcessMesh failed!')

    return mesh
