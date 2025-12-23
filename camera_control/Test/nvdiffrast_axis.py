import os
import torch
import imageio
import numpy as np
import nvdiffrast.torch as dr

from camera_control.Module.camera import Camera

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def test():
    glctx = dr.RasterizeCudaContext()

    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    col = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)

    camera = Camera(
        width=640,
        height=480,
        fx=500,
        fy=500,
        cx=320,
        cy=240,
        pos=[0, 0, 0],
        look_at=[0, 0, 1],
        up=[0, 1, 0],
    )

    rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
    out, _ = dr.interpolate(col, rast, tri)

    img = out.cpu().numpy()[0, ::-1, :, :] # Flip vertically.
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

    print("Saving to './output/tri.png'.")
    os.makedirs('./output/', exist_ok=True)
    imageio.imsave('./output/tri.png', img)
    return True
