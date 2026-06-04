import open3d as o3d

from camera_control.Method.render import toVisibleVolumeMesh
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.volume_marker import VolumeMarker


if __name__ == '__main__':
    colmap_data_folder_path = '/home/lichanghao/chLi/MMVideoReconV1/JJ/20260427_164113_431091/08_colmap_gs_1024/'

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)
    camera_list = [camera_list[0]]

    ccm = camera_list[0].toCCM()
    ccm_points = ccm[camera_list[0].toDepthMask()]
    ccm_pcd = o3d.geometry.PointCloud()
    ccm_pcd.points = o3d.utility.Vector3dVector(
        ccm_points.detach().cpu().numpy().astype("float64")
    )
    o3d.io.write_point_cloud(colmap_data_folder_path + 'camera_0_ccm.ply', ccm_pcd, write_ascii=True)

    visible_volume = VolumeMarker.markVisible(
        camera_list=camera_list,
        volume_resolution=16,
    )

    mesh = toVisibleVolumeMesh(visible_volume)

    o3d.io.write_triangle_mesh(colmap_data_folder_path + 'vis_volume_label.ply', mesh)
