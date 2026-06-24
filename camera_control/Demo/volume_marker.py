import os

import trimesh
import numpy as np
import open3d as o3d

from camera_control.Method.io import loadMeshFile
from camera_control.Method.sample import sampleCameras
from camera_control.Method.render import toVisibleVolumeMesh
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.mesh_renderer import MeshRenderer
from camera_control.Module.volume_marker import VolumeMarker


def _o3d_mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices).astype(np.float64)
    faces = np.asarray(mesh.triangles).astype(np.int64)

    vertex_colors = None
    colors = np.asarray(mesh.vertex_colors)
    if colors.shape == vertices.shape:
        vertex_colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )


def _add_camera_meshes(
    mesh: o3d.geometry.TriangleMesh,
    camera_list,
) -> o3d.geometry.TriangleMesh:
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh += mesh

    for camera in camera_list:
        combined_mesh += camera.toO3DMesh(far=0.2)

    combined_mesh.compute_vertex_normals()
    return combined_mesh


def _save_volume_glb_with_cameras(
    mesh: o3d.geometry.TriangleMesh,
    camera_list,
    glb_file_path: str,
) -> None:
    glb_folder_path = os.path.dirname(glb_file_path)
    if glb_folder_path:
        os.makedirs(glb_folder_path, exist_ok=True)

    mesh_with_cameras = _add_camera_meshes(mesh, camera_list)
    _o3d_mesh_to_trimesh(mesh_with_cameras).export(glb_file_path)


def demo_mark_glb():
    data_folder_path = '/home/lichanghao/chLi/Dataset/test_glb/'
    output_folder_path = data_folder_path + 'volume_marker_debug/'
    glb_file_path = data_folder_path + 'gt_mesh.glb'

    mesh = loadMeshFile(glb_file_path)

    sample_points, _ = trimesh.sample.sample_surface(mesh, 4096)
    sample_points = np.asarray(sample_points)

    camera_list = sampleCameras(
        mesh=mesh,
        candidate_camera_num=8,
        camera_num=8,
        camera_dist_range=[2.5, 2.5],
        width=1024,
        height=1024,
        fovx_degree_range=[30, 30],
        up_direction=[0, 1, 0],
    )

    camera_list = CameraConvertor.getBestPoseCameras(
        camera_list=camera_list,
        pts=sample_points,
    )

    for camera in camera_list:
        camera.to(device='cuda:0')

    camera_list = MeshRenderer.renderCameraData(
        mesh=mesh,
        camera_list=camera_list,
        device='cuda:0',
        enable_antialias=False,
    )

    visible_volume = VolumeMarker.markVisible(
        camera_list=camera_list,
        volume_resolution=16,
        geometry=sample_points,
        debug=True,
        debug_folder_path=output_folder_path,
        debug_prefix='glb',
    )

    n_valid = int((visible_volume == VolumeMarker.VALID).sum().item())
    n_free = int((visible_volume < 0).sum().item())
    n_unknown = int((visible_volume == VolumeMarker.UNKNOWN).sum().item())
    print(f'[demo_mark_glb] VALID={n_valid}, FREE={n_free}, UNKNOWN={n_unknown}')

    # 隐藏 UNKNOWN 灰色四面体、显示全部 FREE 层级（max_free_k=None），
    # 便于直接核对 VALID/FREE 的几何与 FREE_KN 距离分层。
    mesh = toVisibleVolumeMesh(
        labels=visible_volume,
        max_free_k=1,
        show_unknown=True,
    )

    _save_volume_glb_with_cameras(
        mesh=mesh,
        camera_list=camera_list,
        glb_file_path=output_folder_path + 'vis_volume_label_pcd.glb',
    )
    return True

def demo_mark_real_data():
    data_folder_path = '/home/lichanghao/chLi/MMVideoReconV1/JJ/20260427_164113_431091/'
    colmap_data_folder_path = data_folder_path + '08_colmap_gs/'
    gs_file_path = data_folder_path + 'gs_normalized.ply'

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    for camera in camera_list:
        camera.to(device='cuda:0')

    gs_points = np.asarray(o3d.io.read_point_cloud(gs_file_path).points)

    visible_volume = VolumeMarker.markVisible(
        camera_list=camera_list,
        volume_resolution=16,
        geometry=gs_points,
    )

    mesh = toVisibleVolumeMesh(
        labels=visible_volume,
        max_free_k=1,
    )

    _save_volume_glb_with_cameras(
        mesh=mesh,
        camera_list=camera_list,
        glb_file_path=colmap_data_folder_path + 'vis_volume_label_pcd.glb',
    )

    for i in range(len(camera_list)):
        visible_volume = VolumeMarker.markVisible(
            camera_list=[camera_list[i]],
            volume_resolution=16,
            geometry=gs_points,
        )

        mesh = toVisibleVolumeMesh(
            labels=visible_volume,
            max_free_k=1,
        )

        _save_volume_glb_with_cameras(
            mesh=mesh,
            camera_list=[camera_list[i]],
            glb_file_path=colmap_data_folder_path + f'vis_volume_label_{i}.glb',
        )
    return True
