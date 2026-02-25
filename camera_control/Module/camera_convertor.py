import os
import cv2
import torch
import trimesh
import numpy as np
import open3d as o3d

from tqdm import tqdm
from copy import deepcopy
from shutil import rmtree
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor

from camera_control.Method.pcd import toPcd
from camera_control.Method.data import toNumpy, toTensor
from camera_control.Method.rotate import decompose_similarity_from_T, invert_similarity
from camera_control.Module.camera import Camera


class CameraConvertor(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def transformCameras(
        camera_list: List[Camera],
        world_transform: Union[torch.Tensor, np.ndarray, list],
    ) -> List[Camera]:
        """
        用给定的 4x4 世界变换矩阵对所有相机的世界位姿进行变换。

        约定：传入的 world_transform 是 Open3D ICP 矩阵的转置（行向量右乘约定）。
        结合 SVD 分解，滤除所有仿射/错切噪声，确保对相机的变换是绝对严格的刚体+均匀缩放。
        """
        transformed_list = deepcopy(camera_list)
        if not transformed_list:
            return transformed_list

        device = transformed_list[0].device
        dtype = transformed_list[0].dtype

        # =========================================================
        # 1. 矩阵桥接与 SVD 极度提纯
        # =========================================================
        # T_right 是行向量右乘矩阵 (p_new = p_old @ T_right)
        T_right = toTensor(world_transform, dtype, device).reshape(4, 4)

        # 使用 SVD 提取出绝对纯净的左乘参数 R_left, s, t_left
        R_left, s, t_left = decompose_similarity_from_T(T_right.T, enforce_positive_scale=True)

        scale_safe = s.clamp(min=1e-8)

        T_clean_left_inv = invert_similarity(R_left, scale_safe, t_left)

        for cam in transformed_list:
            # =========================================================
            # Step A: 乘以带有 1/s 缩放的纯净逆矩阵
            # =========================================================
            # p_cam = W2C_old @ (T_clean_left^-1) @ p_world_new_col
            cam.world2camera = cam.world2camera @ T_clean_left_inv

            # =========================================================
            # Step B: 【核心刚体补偿】
            # 将前三行乘以 scale_safe，消除齐次逆矩阵带来的 1/s 缩放污染。
            # 这一步将 W2C 完美复原为正交的 SE(3) 刚体位姿，
            # 并在物理空间中将相机的物理位置等比例地精确推远 s 倍！
            # =========================================================
            cam.world2camera[:3, :] = cam.world2camera[:3, :] * scale_safe

            # =========================================================
            # Step C: 更新深度与点云重投影
            # =========================================================
            if getattr(cam, "depth", None) is not None:
                # 场景放大 s 倍，相机观察同视角的物理深度同步放大 s 倍
                cam.depth = cam.depth * scale_safe
                cam.updateCCM()

            # =========================================================
            # Step D: 法线旋转更新
            # =========================================================
            if getattr(cam, "normal", None) is not None:
                # 你的 normal 约定为行向量右乘；
                # SVD 提取出的 R_left 是左乘旋转矩阵，其等价的右乘旋转矩阵正是它的转置 R_left.T
                cam.normal = (cam.normal @ R_left.T).clone()
                cam.normal = cam.normal / (torch.linalg.norm(cam.normal, dim=-1, keepdim=True) + 1e-8)

        return transformed_list

    @staticmethod
    def normalizeCameras(
        camera_list: List[Camera],
    ) -> List[Camera]:
        normalized_camera_list = deepcopy(camera_list)

        device = normalized_camera_list[0].device
        dtype = normalized_camera_list[0].dtype

        # =====================================================
        # Step 1: 主光轴最近公共点 -> 世界原点
        # =====================================================
        origins, dirs = [], []
        z_cam = torch.tensor([0., 0., 1.], dtype=dtype, device=device)

        for cam in camera_list:
            R, t = cam.R, cam.t
            C = -R.T @ t
            d = -R.T @ z_cam
            d = d / (torch.linalg.norm(d) + 1e-8)
            origins.append(C)
            dirs.append(d)

        origins = torch.stack(origins)
        dirs = torch.stack(dirs)

        I = torch.eye(3, dtype=dtype, device=device)
        A = torch.zeros((3, 3), dtype=dtype, device=device)
        b = torch.zeros((3,), dtype=dtype, device=device)

        for o, d in zip(origins, dirs):
            P = I - d[:, None] @ d[None, :]
            A += P
            b += P @ o

        focus = torch.linalg.solve(A, b)

        for cam in normalized_camera_list:
            cam.world2camera[:3, 3] += cam.R @ focus

        # =====================================================
        # Step 2.1: 平均 image-Y -> 世界 +Z
        # =====================================================
        y_cam = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
        z_world = torch.tensor([0., 0., 1.], dtype=dtype, device=device)

        y_world = torch.stack([cam.R.T @ y_cam for cam in normalized_camera_list]).mean(dim=0)
        y_world = y_world / (torch.linalg.norm(y_world) + 1e-8)

        def rot_from_a_to_b(a, b):
            v = torch.cross(a, b)
            c = torch.dot(a, b)
            s = torch.linalg.norm(v)
            if s < 1e-8:
                return torch.eye(3, dtype=dtype, device=device)
            vx = torch.tensor([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=dtype, device=device)
            return torch.eye(3, dtype=dtype, device=device) + vx + vx @ vx * ((1 - c) / (s * s))

        Q_y = rot_from_a_to_b(y_world, z_world)

        for cam in normalized_camera_list:
            cam.world2camera[:3, :3] = cam.R @ Q_y.T

        # =====================================================
        # Step 2.2: 仅绕世界 Z 轴，使 cam0.image-Z → +X
        # =====================================================
        cam0 = normalized_camera_list[0]
        z0_world = cam0.R.T @ z_cam

        # 投影到 XY 平面
        z0_xy = z0_world.clone()
        z0_xy[2] = 0.0
        z0_xy = z0_xy / (torch.linalg.norm(z0_xy) + 1e-8)

        # yaw 角
        yaw = torch.atan2(z0_xy[1], z0_xy[0])

        cos_y, sin_y = torch.cos(-yaw), torch.sin(-yaw)
        Q_z = torch.tensor([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=dtype, device=device)

        for cam in normalized_camera_list:
            cam.world2camera[:3, :3] = cam.R @ Q_z.T

        # =====================================================
        # Step 2.3: 方向一致性修正（避免整体翻转）
        # =====================================================
        x_cam = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
        x0_world = normalized_camera_list[0].R.T @ x_cam

        if x0_world[1] < 0:  # image-X 不应指向 -Y
            Q_flip = torch.tensor([
                [-1.,  0., 0.],
                [ 0., -1., 0.],
                [ 0.,  0., 1.]
            ], dtype=dtype, device=device)

            for cam in normalized_camera_list:
                cam.world2camera[:3, :3] = cam.R @ Q_flip.T

        # ===============================
        # Step 3: 世界坐标系轴置换 (ZXY -> XYZ)
        # ===============================
        P = torch.tensor([
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
        ], dtype=dtype, device=device)

        for cam in normalized_camera_list:
            cam.world2camera[:3, :3] = cam.world2camera[:3, :3] @ P.T
        return normalized_camera_list

    @staticmethod
    def _process_camera_for_pcd(camera: Camera, conf_thresh: float, use_mask: bool):
        mask = camera.toDepthMask(conf_thresh, use_mask)
        image_colors = camera.sampleRGBAtUV(camera.toDepthUV())
        colors = image_colors[mask]
        points, _ = camera.toDepthPoints(conf_thresh, use_mask)
        return points, colors

    @staticmethod
    def createDepthPcd(
        camera_list: List[Camera],
        conf_thresh: float=0.8,
        use_mask: bool=True,
    ) -> o3d.geometry.PointCloud:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda cam: CameraConvertor._process_camera_for_pcd(cam, conf_thresh, use_mask),
                camera_list,
            ))
        points_list = [r[0] for r in results]
        colors_list = [r[1] for r in results]

        points = torch.cat(points_list, dim=0)
        colors = torch.cat(colors_list, dim=0)

        pcd = toPcd(points, colors)

        return pcd

    @staticmethod
    def createColmapDataFolder(
        cameras: List[Camera],
        pcd: Union[trimesh.Trimesh, o3d.geometry.PointCloud, torch.Tensor, np.ndarray, list, str],
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
        elif isinstance(pcd, trimesh.Trimesh):
            pcd = toPcd(pcd.vertices)
        elif isinstance(pcd, torch.Tensor) or isinstance(pcd, np.ndarray) or isinstance(pcd, list):
            points = toNumpy(pcd).reshape(-1, 3)
            pcd = toPcd(points)

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

        # 创建文件夹结构
        if save_data_folder_path[-1] != '/':
            save_data_folder_path += '/'

        if os.path.exists(save_data_folder_path):
            rmtree(save_data_folder_path)

        image_folder_path = save_data_folder_path + 'images/'

        mask_folder_path = save_data_folder_path + 'masks/'
        masked_image_folder_path = save_data_folder_path + 'masked_images/'

        depth_folder_path = save_data_folder_path + 'depths/'
        depth_vis_folder_path = save_data_folder_path + 'depths_vis/'
        masked_depth_vis_folder_path = save_data_folder_path + 'masked_depths_vis/'

        sparse_folder_path = save_data_folder_path + 'sparse/0/'

        os.makedirs(image_folder_path, exist_ok=True)

        os.makedirs(mask_folder_path, exist_ok=True)
        os.makedirs(masked_image_folder_path, exist_ok=True)

        os.makedirs(depth_folder_path, exist_ok=True)
        os.makedirs(depth_vis_folder_path, exist_ok=True)
        os.makedirs(masked_depth_vis_folder_path, exist_ok=True)

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

        def _process_one_camera(camera_idx):
            camera = cameras[camera_idx]
            image_filename = camera.image_id
            format = '.' + image_filename.split('.')[-1]
            image_basename = image_filename.split(format)[0]

            colmap_pose = camera.toColmapPose().cpu().numpy()

            cv2.imwrite(image_folder_path + image_filename, camera.image_cv)

            cv2.imwrite(mask_folder_path + image_filename, camera.mask_cv)
            cv2.imwrite(masked_image_folder_path + image_filename, camera.toMaskedImageCV())

            np.save(depth_folder_path + image_basename + '.npy', camera.depth_with_conf)
            cv2.imwrite(depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=False))
            cv2.imwrite(masked_depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=True))
            return (camera_idx, image_filename, colmap_pose)

        print('[INFO][MeshRenderer::createColmapDataFolder]')
        print('\t start create colmap data folder...')
        max_workers = min(32, (os.cpu_count() or 4) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(_process_one_camera, range(camera_num)), total=camera_num, desc='colmap data'))

        for (camera_idx, image_filename, colmap_pose) in results:
            images_txt_lines.append(
                f"{camera_idx + 1} {colmap_pose[0]:.10f} {colmap_pose[1]:.10f} {colmap_pose[2]:.10f} {colmap_pose[3]:.10f} "
                f"{colmap_pose[4]:.10f} {colmap_pose[5]:.10f} {colmap_pose[6]:.10f} 1 {image_filename}\n"
            )
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
        mask_folder_name: str='masks',
        depth_folder_name: str='depths',
    ) -> List[Camera]:
        """
        从 COLMAP 数据目录加载相机列表。
        cameras.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] (PINHOLE: fx, fy, cx, cy)
        images.txt: 每张图两行，第一行 IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        """
        if not colmap_data_folder_path.endswith('/'):
            colmap_data_folder_path += '/'
        image_folder_path = colmap_data_folder_path + image_folder_name + '/'
        mask_folder_path = colmap_data_folder_path + mask_folder_name + '/'
        depth_folder_path = colmap_data_folder_path + depth_folder_name + '/'
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
            image_filename = rec['name']
            format = '.' + image_filename.split('.')[-1]
            image_basename = image_filename.split(format)[0]

            if camera_id not in cameras_dict:
                print('[WARN][CameraConvertor::loadColmapDataFolder] unknown camera_id:', camera_id, 'skip image', image_filename)
                return None
            cam_info = cameras_dict[camera_id]
            intrinsic = np.array([
                [cam_info['fx'], 0, cam_info['cx']],
                [0, cam_info['fy'], cam_info['cy']],
                [0, 0, 1],
            ], dtype=np.float64)
            pose = rec['pose']
            camera = Camera.fromColmapPose(pose, intrinsic)

            image_file_path = image_folder_path + image_filename
            if not camera.loadImageFile(image_file_path):
                print('[WARN][CameraConvertor::loadColmapDataFolder]')
                print('\t loadImageFile failed!')
                print('\t image file path:', image_file_path)
                return None

            mask_file_path = mask_folder_path + image_filename
            if os.path.exists(mask_file_path):
                if not camera.loadMaskFile(mask_file_path):
                    print('[WARN][CameraConvertor::loadColmapDataFolder]')
                    print('\t loadMaskFile failed!')
                    print('\t mask file path:', mask_file_path)

            depth_file_path = depth_folder_path + image_basename + '.npy'
            if os.path.exists(depth_file_path):
                if not camera.loadDepthFile(depth_file_path):
                    print('[WARN][CameraConvertor::loadColmapDataFolder]')
                    print('\t loadDepthFile failed!')
                    print('\t depth file path:', depth_file_path)

            camera.image_id = image_filename
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
        camera_list.sort(key=lambda c: c.image_id)
        return camera_list
