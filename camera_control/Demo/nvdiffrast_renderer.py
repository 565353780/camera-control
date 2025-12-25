import os
import cv2

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer


def demo():
    home = os.environ['HOME']
    mesh_file_path = home + '/chLi/Dataset/MM/Match/1024result/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb'
    mesh_file_path = home + '/chLi/Dataset/MM/Match/GTstageone/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af_decoded.ply'
    color = [178, 178, 178]
    device = "cuda:0"

    nvdiffrast_renderer = NVDiffRastRenderer(
        mesh_file_path,
        color,
    )

    view_ratio = 0.95 

    camera = Camera(
        width=2560,
        height=1440,
        pos=[0, 0, 1],
        look_at=[0, 0, 0],
        up=[0, 1, 0],
        device=device,
    )
    camera.outputInfo()
    camera.focusOnPoints(nvdiffrast_renderer.mesh.vertices, view_ratio)
    camera.outputInfo()

    light_direction = [1, 1, 1]

    render_dict = nvdiffrast_renderer.renderImage(
        camera,
        light_direction,
    )

    if render_dict is None:
        print('render failed!')
        return False

    for key, value in render_dict.items():
        try:
            print(key, value.shape)
        except:
            pass

    cv2.imwrite('./output/test_render.png', render_dict['image'])
    return True
