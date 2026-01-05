import os

from camera_control.Method.io import loadMeshFile

from camera_control.Module.mesh_renderer import MeshRenderer


def demo() -> bool:
    home = os.environ['HOME']
    mesh_file_path = home + "/chLi/Dataset/MM/Match/nezha/nezha.glb"
    save_data_folder_path = home + "/chLi/Dataset/MM/Match/nezha/"
    camera_num = 20
    camera_dist = 2.5
    device = 'cuda:0'

    mesh = loadMeshFile(mesh_file_path)

    MeshRenderer.createOmniVGGTDataFolder(
        mesh,
        save_data_folder_path + 'omnivggt/',
        camera_num=camera_num,
        camera_dist=camera_dist,
        device=device,
    )

    MeshRenderer.createDA3DataFile(
        mesh,
        save_data_folder_path + 'da3/render_data.npy',
        camera_num=camera_num,
        camera_dist=camera_dist,
        device=device,
    )
    return True
