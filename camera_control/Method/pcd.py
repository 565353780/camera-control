import torch
import numpy as np
import open3d as o3d

from camera_control.Method.data import toNumpy


def toPcd(points, colors=None) -> o3d.geometry.PointCloud:
    points = toNumpy(points, np.float64).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        if isinstance(colors, torch.Tensor):
            if colors.dtype ==  torch.uint8:
                colors = colors.to(torch.float64) / 255.0
        elif isinstance(colors, np.ndarray):
            if colors.dtype ==  np.uint8:
                colors = colors.astype(np.float64) / 255.0

        colors = toNumpy(colors, np.float64).reshape(-1, 3)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
