import os
import numpy as np
import open3d as o3d

from camera_control.Method.pcd import toPcd
from camera_control.Method.filter import searchMainClusterPointMask


def test():
    pcd_file_path = '/Users/chli/Documents/debug_source_pcd.ply'
    save_pcd_file_path = '/Users/chli/Documents/debug_filtered_pcd.ply'

    pcd = o3d.io.read_point_cloud(pcd_file_path)

    points = np.asarray(pcd.points)

    point_idxs = searchMainClusterPointMask(
        points=points,
    )

    main_cluster_points = points[point_idxs]

    main_cluster_pcd = toPcd(main_cluster_points)

    o3d.io.write_point_cloud(save_pcd_file_path, main_cluster_pcd)
    return True
