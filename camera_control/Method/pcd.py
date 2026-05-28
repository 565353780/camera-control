import torch
import trimesh
import numpy as np
import open3d as o3d

from typing import Optional

from camera_control.Method.data import toNumpy


def toPcd(points, colors=None, normals=None) -> o3d.geometry.PointCloud:
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

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(toNumpy(normals, np.float64).reshape(-1, 3))
    return pcd


def toTrimeshPcd(pcd: o3d.geometry.PointCloud) -> Optional[trimesh.PointCloud]:
    if pcd is None or len(pcd.points) == 0:
        return None

    pcd_points = np.asarray(pcd.points)
    if pcd.has_colors():
        pcd_colors = np.asarray(pcd.colors)
        pcd_colors_uint8 = np.clip(pcd_colors * 255.0, 0, 255).astype(np.uint8)
        pcd_rgba = np.concatenate(
            [pcd_colors_uint8, np.full((pcd_colors_uint8.shape[0], 1), 255, dtype=np.uint8)],
            axis=1,
        )
        return trimesh.PointCloud(vertices=pcd_points, colors=pcd_rgba)

    return trimesh.PointCloud(vertices=pcd_points)
