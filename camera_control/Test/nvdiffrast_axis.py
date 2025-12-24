import os
import torch
from camera_control.Method.data import toNumpy
import imageio
import numpy as np
import nvdiffrast.torch as dr

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def test():
    device = 'cuda:0'

    glctx = dr.RasterizeCudaContext(device=device)

    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    col = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)

    camera = Camera(
        width=640,
        height=480,
        pos=[0, 0, 2],
        look_at=[0, 0, -1],
        up=[0, 1, 0],
        device=device,
    )

    mvp = NVDiffRastRenderer.getWorld2NVDiffRast(camera, [2, 2, 2])

    vertices_clip_batch = torch.matmul(pos, mvp.T).contiguous()  # [1, V, 4]

    rast, _ = dr.rasterize(
        glctx,
        vertices_clip_batch,  # [1, V, 4]
        tri,
        resolution=[camera.height, camera.width]
    )  # rast_out: [1, H, W, 4]

    out, _ = dr.interpolate(col, rast, tri)  # [1, H, W, 3]

    img = np.clip(np.rint(toNumpy(out)[0] * 255), 0, 255).astype(np.uint8)

    print("Saving to './output/tri.png'.")
    os.makedirs('./output/', exist_ok=True)
    imageio.imsave('./output/tri.png', img)
    return True
