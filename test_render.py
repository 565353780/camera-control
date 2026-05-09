import os
import cv2
import trimesh

from tqdm import trange

from camera_control.Method.io import loadMeshFile
from camera_control.Method.mesh import normalizeMesh
from camera_control.Module.mesh_renderer import MeshRenderer

if __name__ == '__main__':
    mesh_file_path = './1.glb'
    save_debug_folder_path = './debug/'
    os.makedirs(save_debug_folder_path, exist_ok=True)

    #mesh = loadMeshFile(mesh_file_path)
    mesh = trimesh.load(mesh_file_path, process=False, force='mesh')

    normalized_mesh = normalizeMesh(mesh)

    mesh.export(save_debug_folder_path + '1_output.glb')
    normalized_mesh.export(save_debug_folder_path + '1_norm.glb')

    camera_list = MeshRenderer.sampleRenderData(
        normalized_mesh,
        candidate_camera_num=4,
        camera_num=4,
        camera_dist_range=[1.3, 1.3],
        width=1024,
        height=1024,
        fovx_degree_range=[60, 60],
        up_direction=[0, 1, 0],
        enable_antialias=False,
        safe_pixel_num=10,
    )

    for i in trange(len(camera_list)):
        cv2.imwrite(
            f'{save_debug_folder_path}rgb_{i}.png',
            camera_list[i].toImageCV(),
        )

        cv2.imwrite(
            f'{save_debug_folder_path}normal_{i}.png',
            camera_list[i].toNormalCameraVisCV(),
        )

        cv2.imwrite(
            f'{save_debug_folder_path}depth_{i}.png',
            camera_list[i].toDepthVisCV(),
        )
