import os
import cv2
import torch
import numpy as np
import open3d as o3d
from shutil import rmtree
from typing import Union, List

from camera_control.Method.data import toNumpy
from camera_control.Module.camera import Camera


class CameraConvertor(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def createColmapDataFolder(
        cameras: List[Camera],
        images: Union[torch.Tensor, np.ndarray, list],
        pcd: o3d.geometry.PointCloud,
        save_data_folder_path: str,
    ) -> bool:
        """
        创建用于训练3DGS的COLMAP格式数据文件夹

        生成的数据结构：
        save_data_folder_path/
        ├── images/           # 渲染的图像
        │   ├── 00000.png
        │   ├── 00001.png
        │   └── ...
        └── sparse/
            └── 0/
                ├── cameras.txt   # 相机内参 (PINHOLE模型)
                ├── images.txt    # 图像外参 (四元数 + 平移)
                └── points3D.ply  # 3D点云

        坐标系转换说明：
        - 原始相机坐标系 (camera.py): X右，Y上，Z后（相机看向 -Z 方向）
        - COLMAP相机坐标系: X右，Y下，Z前（相机看向 +Z 方向）
        - 世界坐标系保持不变（与mesh坐标系一致）
        - 只转换相机坐标系，不转换世界坐标系
        """
        images = toNumpy(images, np.uint8)

        image_num, height, width = images.shape[:3]

        fx = cameras[0].fx
        fy = cameras[0].fy
        cx = cameras[0].cx
        cy = cameras[0].cy

        if os.path.exists(save_data_folder_path):
            rmtree(save_data_folder_path)

        # 创建文件夹结构
        if not save_data_folder_path.endswith('/'):
            save_data_folder_path += '/'

        images_folder_path = save_data_folder_path + 'images/'
        sparse_folder_path = save_data_folder_path + 'sparse/0/'
        os.makedirs(images_folder_path, exist_ok=True)
        os.makedirs(sparse_folder_path, exist_ok=True)

        # 准备cameras.txt内容
        # COLMAP格式: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # PINHOLE模型参数: fx, fy, cx, cy
        cameras_txt_lines = [
            "# Camera list with one line of data per camera:\n",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
            f"# Number of cameras: 1\n",
            f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n"
        ]

        # 准备images.txt内容
        # COLMAP格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # 然后是一行2D点（可以为空）
        images_txt_lines = [
            "# Image list with two lines of data per image:\n",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
            f"# Number of images: {image_num}\n",
        ]

        print('[INFO][MeshRenderer::createColmapDataFolder]')
        print('\t start create colmap data folder...')
        for i in range(image_num):
            camera = cameras[i]
            rgb = images[i]

            colmap_pose = camera.toColmapPose()

            # 图像文件名
            image_name = f"{i:06d}.png"
            image_id = i + 1  # COLMAP的ID从1开始

            # 保存图像
            cv2.imwrite(images_folder_path + image_name, rgb)

            # 添加到images.txt
            # 格式: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            images_txt_lines.append(
                f"{image_id} {colmap_pose[0]:.10f} {colmap_pose[1]:.10f} {colmap_pose[2]:.10f} {colmap_pose[3]:.10f} "
                f"{colmap_pose[4]:.10f} {colmap_pose[5]:.10f} {colmap_pose[6]:.10f} 1 {image_name}\n"
            )
            # 空行表示没有2D特征点
            images_txt_lines.append("\n")

        # 写入cameras.txt
        with open(sparse_folder_path + 'cameras.txt', 'w') as f:
            f.writelines(cameras_txt_lines)

        # 写入images.txt
        with open(sparse_folder_path + 'images.txt', 'w') as f:
            f.writelines(images_txt_lines)

        # 生成points3D.ply点云文件
        print('\t generating points3D.ply from points...')
        ply_path = sparse_folder_path + 'points3D.ply'
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)

        print(f'\t saved to: {save_data_folder_path}')
        print(f'\t total images: {image_num}')
        print(f'\t total points: {len(pcd.points)}')
        return True
