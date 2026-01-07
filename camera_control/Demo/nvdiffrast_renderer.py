import os
import cv2

from camera_control.Method.io import loadMeshFile
from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer


def demo():
    home = os.environ['HOME']
    mesh_file_path = home + '/chLi/Dataset/MM/Match/1024result/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb'
    #mesh_file_path = home + '/chLi/Dataset/MM/Match/GTstageone/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af_decoded.ply'
    mesh_file_path = home + '/chLi/Dataset/MM/Match/nezha/nezha.glb'
    paint_color = [178, 178, 178]
    bg_color = [255, 255, 255]
    view_ratio = 0.95 
    device = "cuda:0"

    mesh = loadMeshFile(mesh_file_path)
    assert mesh

    camera = Camera(
        width=2560,
        height=1440,
        pos=[0, 0, 1],
        look_at=[0, 0, 0],
        up=[0, 1, 0],
        device=device,
    )
    camera.focusOnPoints(mesh.vertices, view_ratio)

    light_direction = [1, 1, 1]

    render_texture_dict = NVDiffRastRenderer.renderTexture(
        mesh=mesh,
        camera=camera,
        bg_color=bg_color,
    )

    render_vertex_color_dict = NVDiffRastRenderer.renderVertexColor(
        mesh=mesh,
        camera=camera,
        light_direction=light_direction,
        paint_color=paint_color,
        bg_color=bg_color,
    )

    render_depth_dict = NVDiffRastRenderer.renderDepth(
        mesh=mesh,
        camera=camera,
        bg_color=bg_color,
    )

    render_normal_dict = NVDiffRastRenderer.renderNormal(
        mesh=mesh,
        camera=camera,
        bg_color=bg_color,
    )

    os.makedirs('./output/', exist_ok=True)

    cv2.imwrite('./output/test_render_texture.png', render_texture_dict['image'])
    cv2.imwrite('./output/test_render_vertex_color.png', render_vertex_color_dict['image'])
    cv2.imwrite('./output/test_render_depth.png', render_depth_dict['image'])
    cv2.imwrite('./output/test_render_normal_camera.png', render_normal_dict['normal_camera'])
    cv2.imwrite('./output/test_render_normal_world.png', render_normal_dict['normal_world'])
    return True
