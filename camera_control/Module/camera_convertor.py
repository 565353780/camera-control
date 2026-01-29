import os
import cv2
import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm
from shutil import rmtree
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor

from camera_control.Method.pcd import toPcd
from camera_control.Module.camera import Camera


class CameraConvertor(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def createDepthPcd(
        camera_list: List[Camera],
        conf_thresh: float=0.8,
    ) -> o3d.geometry.PointCloud:
        uv = camera_list[0].toImageUV()

        points_list = []
        colors_list = []
        for i in range(len(camera_list)):
            camera = camera_list[i]

            image = camera.image  # (H, W, 3), float32 or similar, tensor
            depth = camera.depth  # (H, W), tensor
            conf = camera.conf    # (H, W), tensor

            conf_1d = conf.reshape(-1)
            conf_thresh_tensor = torch.quantile(conf_1d, conf_thresh)
            conf_mask = conf >= conf_thresh_tensor  # (H, W), bool tensor

            # Find valid indices (flattened and 2d)
            H, W = image.shape[:2]
            conf_mask_flat = conf_mask.flatten()
            valid_indices = torch.nonzero(conf_mask_flat, as_tuple=False).squeeze(1)

            # Prepare UV coordinates
            # uv shape: (H, W, 2), so flatten to (H*W, 2), gather with valid_indices
            uv_flat = uv.reshape(-1, 2)
            valid_uv = uv_flat[valid_indices]  # (M, 2)

            # Depth flatten and gather
            depth_flat = depth.reshape(-1)
            valid_depth = depth_flat[valid_indices]  # (M,)

            points = camera.projectUV2Points(valid_uv, valid_depth)

            # Get color for valid_uv
            u = valid_uv[:, 0]  # (M,)
            v = valid_uv[:, 1]  # (M,)

            # Compute image indices for color lookup (tensor ops, not np)
            col_idx = torch.clamp((u * (W - 1)).round().long(), 0, W - 1)  # (M,)
            row_idx = torch.clamp(((1.0 - v) * (H - 1)).round().long(), 0, H - 1)  # (M,)

            # Gather color with advanced indices, use torch.flip for RGB->BGR if needed
            valid_colors = image[row_idx, col_idx]  # (M, 3), still tensor
            valid_colors = valid_colors[..., [2, 1, 0]]  # flip channels for BGR if needed

            points_list.append(points)
            colors_list.append(valid_colors)

        points = torch.cat(points_list, dim=0)
        colors = torch.cat(colors_list, dim=0)

        pcd = toPcd(points, colors)

        return pcd

    @staticmethod
    def createColmapDataFolder(
        cameras: List[Camera],
        pcd: Union[o3d.geometry.PointCloud, str],
        save_data_folder_path: str,
        point_num_max: Optional[int]=None,
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
        if isinstance(pcd, str):
            if not os.path.exists(pcd):
                print('[ERROR][CameraConvertor::createColmapDataFolder]')
                print('\t pcd file not exist!')
                print('\t pcd:', pcd)
                return False

            pcd = o3d.io.read_point_cloud(pcd)

        if not isinstance(pcd, o3d.geometry.PointCloud):
            print('[ERROR][CameraConvertor::createColmapDataFolder]')
            print('\t pcd is not o3d.geometry.PointCloud!')
            return False

        if point_num_max is not None:
            point_num = len(pcd.points)
            if point_num > point_num_max:
                sample_ratio = point_num_max / point_num
                pcd = pcd.random_down_sample(sample_ratio)

        # 检查pcd，如果没有颜色则全部赋值为[128,128,128]
        if not pcd.has_colors():
            colors = np.tile(np.array([[128, 128, 128]], dtype=np.float64) / 255.0, (np.asarray(pcd.points).shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        # 如果没有法向，估计法向
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        camera_num = len(cameras)
        height, width = cameras[0].image.shape[:2]
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
            f"# Number of images: {camera_num}\n",
        ]

        print('[INFO][MeshRenderer::createColmapDataFolder]')
        print('\t start create colmap data folder...')
        for i in range(camera_num):
            camera = cameras[i]

            colmap_pose = camera.toColmapPose().cpu().numpy()

            # 图像文件名
            image_name = f"{i:06d}.png"
            image_id = i + 1  # COLMAP的ID从1开始

            # 保存图像
            cv2.imwrite(images_folder_path + image_name, camera.image_cv)

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
        print(f'\t total images: {camera_num}')
        print(f'\t total points: {len(pcd.points)}')
        return True

    @staticmethod
    def loadColmapDataFolder(
        colmap_data_folder_path: str,
        image_folder_name: str='images',
    ) -> List[Camera]:
        """
        从 COLMAP 数据目录加载相机列表。
        cameras.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] (PINHOLE: fx, fy, cx, cy)
        images.txt: 每张图两行，第一行 IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        """
        if not colmap_data_folder_path.endswith('/'):
            colmap_data_folder_path += '/'
        image_folder_path = colmap_data_folder_path + image_folder_name + '/'
        camera_intrinsic_file_path = colmap_data_folder_path + 'sparse/0/cameras.txt'
        camera_extrinsic_file_path = colmap_data_folder_path + 'sparse/0/images.txt'

        if not os.path.exists(image_folder_path):
            print('[ERROR][CameraConvertor::loadColmapDataFolder] image folder not exist:', image_folder_path)
            return []
        if not os.path.exists(camera_intrinsic_file_path):
            print('[ERROR][CameraConvertor::loadColmapDataFolder] cameras.txt not exist:', camera_intrinsic_file_path)
            return []
        if not os.path.exists(camera_extrinsic_file_path):
            print('[ERROR][CameraConvertor::loadColmapDataFolder] images.txt not exist:', camera_extrinsic_file_path)
            return []

        # 解析 cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy (PINHOLE)
        cameras_dict = {}
        with open(camera_intrinsic_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                if model == 'PINHOLE' and len(parts) >= 8:
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    cameras_dict[camera_id] = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

        # 解析 images.txt: 每张图两行，第一行 IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        image_records = []
        with open(camera_extrinsic_file_path, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            image_records.append({
                'image_id': image_id,
                'pose': [qw, qx, qy, qz, tx, ty, tz],
                'camera_id': camera_id,
                'name': name,
            })
            if i < len(lines):
                i += 1  # 跳过 POINTS2D 行

        def _process_one_record(rec):
            """处理单条图像记录，返回 Camera 或 None（失败时）。"""
            camera_id = rec['camera_id']
            if camera_id not in cameras_dict:
                print('[WARN][CameraConvertor::loadColmapDataFolder] unknown camera_id:', camera_id, 'skip image', rec['name'])
                return None
            cam_info = cameras_dict[camera_id]
            intrinsic = np.array([
                [cam_info['fx'], 0, cam_info['cx']],
                [0, cam_info['fy'], cam_info['cy']],
                [0, 0, 1],
            ], dtype=np.float64)
            pose = rec['pose']
            camera = Camera.fromColmapPose(pose, intrinsic)
            image_path = os.path.join(image_folder_path, rec['name'])
            if not camera.loadImageFile(image_path):
                print('[WARN][CameraConvertor::loadColmapDataFolder] load image failed:', image_path)
                return None
            return camera

        print('[INFO][CameraConvertor::loadColmapDataFolder]')
        print('\t start load colmap data...')
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(_process_one_record, image_records),
                total=len(image_records),
                desc='load colmap',
            ))
        camera_list = [c for c in results if isinstance(c, Camera)]
        return camera_list
