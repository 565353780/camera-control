import torch
import numpy as np
import open3d as o3d

from uv_cube_gen.Method.random_sample import sample_points_in_front_of_camera
from uv_cube_gen.Method.render import create_line_set
from uv_cube_gen.Module.camera import Camera


def demo():
    n_points = 100

    pos = np.random.randn(3) * 5.0
    # 随机看向的方向（在相机前方）
    look_at = pos + np.random.randn(3) * 10.0
    # 随机上方向
    up = np.random.randn(3)
    up = up / (np.linalg.norm(up) + 1e-8)

    camera = Camera(pos=pos, look_at=look_at, up=up)

    points = sample_points_in_front_of_camera(camera, n_points)
    uv = camera.project_points_to_uv(points)

    estimated_camera = Camera.fromUVPoints(
        points,
        uv,
        width=camera.width,
        height=camera.height,
    )
    assert estimated_camera is not None

    geometry_list = []

    valid_mask = ~torch.isnan(uv[:, 0])
    valid_points = points[valid_mask]

    # 添加有效点云（黄色）
    pcd= o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.paint_uniform_color([1, 0, 0])
    geometry_list.append(pcd)

    lineset = create_line_set(camera.pos, valid_points, color=[0, 1, 1])
    geometry_list.append(lineset)
 

    geometry_list += [
        camera.toO3DMesh(far=1.0, color=[0, 1, 0]),
        estimated_camera.toO3DMesh(far=1.0, color=[0, 0, 1]),
    ]

    o3d.visualization.draw_geometries(geometry_list)

    return True
