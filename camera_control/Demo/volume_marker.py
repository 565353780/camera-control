import numpy as np
import open3d as o3d

from camera_control.Method.render import toVisibleVolumeMesh
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.volume_marker import VolumeMarker


def demo_mark_volume():
    colmap_data_folder_path = '/home/lichanghao/chLi/MMVideoReconV1/JJ/20260427_164113_431091/08_colmap_gs_1024/'
    gs_file_path = '/home/lichanghao/chLi/MMVideoReconV1/JJ/20260427_164113_431091/gs_normalized.ply'

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    for camera in camera_list:
        camera.to(device='cuda:0')

    gs_points = np.asarray(o3d.io.read_point_cloud(gs_file_path).points)

    visible_volume = VolumeMarker.markVisible(
        camera_list=camera_list,
        volume_resolution=64,
        points=gs_points,
    )

    mesh = toVisibleVolumeMesh(visible_volume)

    o3d.io.write_triangle_mesh(colmap_data_folder_path + 'vis_volume_label.ply', mesh)

    for i in range(len(camera_list)):
        visible_volume = VolumeMarker.markVisible(
            camera_list=[camera_list[i]],
            volume_resolution=16,
            points=gs_points,
        )

        mesh = toVisibleVolumeMesh(visible_volume)

        o3d.io.write_triangle_mesh(colmap_data_folder_path + f'vis_volume_label_{i}.ply', mesh)
    return True
