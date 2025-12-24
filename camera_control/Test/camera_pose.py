import torch
import numpy as np
import open3d as o3d

from camera_control.Method.data import toTensor
from camera_control.Method.render import toPcd
from camera_control.Module.camera import Camera

def test():
    pos = [0, 1, 2]
    look_at = [3, 4, 5]
    up = [0, 0, 1]
    render = False

    camera = Camera(
        width=640,
        height=480,
        fx=500,
        fy=500,
        cx=320,
        cy=240,
        pos=[0, 1, 2],
        look_at=[3, 4, 5],
        up=[0, 0, 1],
    )

    print(camera.world2camera)
    print(camera.R)
    print(camera.t)

    points = [
        [1, 2, 3, 1],
        [3, 4, 5, 1],
        [1, 1, 1, 1],
        [5, 5, 5, 1],
    ]

    points = toTensor(points)

    points_camera = points @ camera.world2camera.T

    if render:
        pcd = toPcd(points, [0, 1, 0])
        pcd_camera = toPcd(points_camera, [1, 0, 0])
        camera_mesh = camera.toO3DMesh(0.5, [0, 0, 1])

        # 添加open3d的坐标轴可视化

        # 使用LineSet画三条不同颜色的轴线
        points_axis = np.array([
            [0, 0, 0],    # origin
            [0.5, 0, 0],  # X
            [0, 0.5, 0],  # Y
            [0, 0, 0.5],  # Z
        ])
        lines_axis = [
            [0, 1],  # X
            [0, 2],  # Y
            [0, 3],  # Z
        ]
        colors_axis = [
            [1, 0, 0],  # X - red
            [0, 1, 0],  # Y - green
            [0, 0, 1],  # Z - blue
        ]
        axis_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_axis),
            lines=o3d.utility.Vector2iVector(lines_axis),
        )
        axis_lines.colors = o3d.utility.Vector3dVector(colors_axis)

        o3d.visualization.draw_geometries([pcd, pcd_camera, camera_mesh, axis_lines])

    points_world = points_camera @ camera.camera2world.T

    points_diff = points_world - points

    max_error = torch.max(points_diff)

    assert max_error < 1e-5, print('max_error:', max_error)

    exit()
    return True
