import os
import torch
import imageio
import numpy as np
import nvdiffrast.torch as dr

from camera_control.Method.data import toNumpy, toTensor
from camera_control.Module.camera import Camera


def test():
    device = 'cuda:0'

    glctx = dr.RasterizeCudaContext(device=device)

    pos = toTensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], torch.float32, device)
    col = toTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], torch.float32, device)
    tri = toTensor([[0, 1, 2]], torch.int32, device)

    camera = Camera(
        width=640,
        height=480,
        pos=[0, 0, 2],
        look_at=[0, 0, 0],
        up=[0, 1, 0],
        device=device,
    )

    mvp = camera.getWorld2NVDiffRast(bbox_length=[2, 2, 2])

    pos_clip = torch.matmul(pos, mvp.T).contiguous()  # [1, V, 4]

    rast, _ = dr.rasterize(
        glctx,
        pos_clip,
        tri,
        resolution=[camera.height, camera.width]
    )

    out, _ = dr.interpolate(col, rast, tri)

    img = np.clip(np.rint(toNumpy(out)[0] * 255), 0, 255).astype(np.uint8)

    print("Saving to './output/tri.png'.")
    os.makedirs('./output/', exist_ok=True)
    imageio.imsave('./output/tri.png', img)
    return True
