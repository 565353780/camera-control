import os

from camera_control.Method.io import loadMeshFile

from camera_control.Module.mesh_renderer import MeshRenderer


def demo() -> bool:
    shape_id = "gold_dragon"

    home = os.environ['HOME']
    mesh_file_path = home + "/chLi/Dataset/MM/Match/" + shape_id + "/" + shape_id + ".glb"
    save_data_folder_path = home + "/chLi/Dataset/MM/Match/" + shape_id + "/"
    camera_num = 20
    camera_dist = 2.5
    bg_color = [255, 255, 255]
    device = 'cuda:0'

    mesh = loadMeshFile(mesh_file_path)

    if True:
        MeshRenderer.createColmapDataFolder(
            mesh,
            save_data_folder_path + 'colmap/',
            camera_num=camera_num,
            camera_dist=0.5,
            width=518,
            height=518,
            bg_color=[0, 0, 0],
            device=device,
        )
 
    if False:
        MeshRenderer.createOmniVGGTDataFolder(
            mesh,
            save_data_folder_path + 'omnivggt/',
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=518,
            height=518,
            bg_color=bg_color,
            device=device,
        )

    if False:
        MeshRenderer.createDA3DataFile(
            mesh,
            save_data_folder_path + 'da3/render_data.npy',
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=504,
            height=504,
            bg_color=bg_color,
            device=device,
        )
    return True
