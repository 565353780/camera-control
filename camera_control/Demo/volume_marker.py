import trimesh
import numpy as np
import open3d as o3d

from camera_control.Method.io import loadMeshFile
from camera_control.Method.sample import sampleCameras
from camera_control.Method.render import toVisibleVolumeMesh
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.volume_marker import VolumeMarker


def demo_mark_glb():
    data_folder_path = '/home/lichanghao/chLi/Dataset/test_glb/'
    glb_file_path = data_folder_path + 'gt_mesh.glb'

    mesh = loadMeshFile(glb_file_path)

    sample_points, _ = trimesh.sample.sample_surface(mesh, 4096)
    sample_points = np.asarray(sample_points)

    trimesh.PointCloud(sample_points).export(data_folder_path + 'sample_points.ply')

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

    visible_volume = VolumeMarker.markVisible(
        camera_list=camera_list,
        volume_resolution=16,
        geometry=sample_points,
    )

    mesh = toVisibleVolumeMesh(
        labels=visible_volume,
        max_free_k=1,
    )

    o3d.io.write_triangle_mesh(data_folder_path + 'vis_volume_label_pcd.ply', mesh)
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

    o3d.io.write_triangle_mesh(colmap_data_folder_path + 'vis_volume_label_pcd.ply', mesh)

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

        o3d.io.write_triangle_mesh(colmap_data_folder_path + f'vis_volume_label_{i}.ply', mesh)
    return True
