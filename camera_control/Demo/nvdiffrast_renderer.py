import os
import cv2
import numpy as np

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer


def demo():
    home = os.environ['HOME']
    mesh_file_path = home + '/chLi/Dataset/MM/Match/1024result/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb'
    device = "cuda:0"
    color = [178, 178, 178]

    nvdiffrast_renderer = NVDiffRastRenderer(
        mesh_file_path,
        device,
        color,
    )

    min_bound = np.min(nvdiffrast_renderer.mesh.vertices, axis=0)
    max_bound = np.max(nvdiffrast_renderer.mesh.vertices, axis=0)
    center = (min_bound + max_bound) / 2.0

    camera = Camera(
        width=2560,
        height=1440,
        pos=center + [0, 0, 1],
        look_at=center,
        up=[0, 1, 0],
        device=device,
    )
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
