import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import trange
from shutil import rmtree

from camera_control.Method.data import toNumpy
from camera_control.Method.sample import sampleCamera
from camera_control.Method.path import createFileFolder
from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

class MeshRenderer(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def sampleRenderData(
        mesh: trimesh.Trimesh,
        camera_num: int = 20,
        camera_dist: float = 2.5,
        width: int = 518,
        height: int = 518,
        fx: float = 500.0,
        fy: float = 500.0,
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> dict:
        camera_list = sampleCamera(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            dtype=dtype,
            device=device,
        )

        render_data_dict = {}

        print('[INFO][MeshRenderer::sampleRenderData]')
        print('\t start sample render data...')
        for i in trange(len(camera_list)):
            camera = camera_list[i]

            camera_data_dict = camera.toDict()

            render_image_dict = NVDiffRastRenderer.renderTexture(
                mesh=mesh,
                camera=camera,
            )

            render_depth_dict = NVDiffRastRenderer.renderDepth(
                mesh=mesh,
                camera=camera,
            )

            image = render_image_dict['image']
            depth = toNumpy(render_depth_dict['depth'], np.float32)
            depth_image = render_depth_dict['image']

            render_data_dict[i] = {
                'camera': camera_data_dict,
                'rgb': image,
                'depth': depth,
                'depth_vis': depth_image,
            }
        return render_data_dict

    @staticmethod
    def rotationMatrixToQuaternion(R: np.ndarray) -> np.ndarray:
        """
        将3x3旋转矩阵转换为四元数 (w, x, y, z)
        使用与 COLMAP 一致的 rotmat2qvec 算法

        Args:
            R: 3x3旋转矩阵

        Returns:
            四元数 [w, x, y, z]
        """
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec.astype(np.float64)

    @staticmethod
    def createColmapDataFolder(
        mesh: trimesh.Trimesh,
        save_data_folder_path: str,
        camera_num: int = 20,
        camera_dist: float = 2.5,
        width: int = 518,
        height: int = 518,
        fx: float = 500.0,
        fy: float = 500.0,
        dtype = torch.float32,
        device: str = 'cuda:0',
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
                └── points3D.ply  # 3D点云 (从mesh的vertices生成)

        坐标系转换说明：
        - 原始相机坐标系 (camera.py): X右，Y上，Z后（相机看向 -Z 方向）
        - COLMAP相机坐标系: X右，Y下，Z前（相机看向 +Z 方向）
        - 世界坐标系保持不变（与mesh坐标系一致）
        - 只转换相机坐标系，不转换世界坐标系

        Args:
            mesh: 要渲染的网格
            save_data_folder_path: 保存数据的文件夹路径
            camera_num: 采样的相机数量
            camera_dist: 相机到物体中心的距离
            width: 图像宽度
            height: 图像高度
            fx: 焦距x
            fy: 焦距y
            dtype: 数据类型
            device: 计算设备

        Returns:
            是否成功
        """
        # 内参主点（COLMAP坐标系，原点在左上角）
        cx = width / 2.0
        cy = height / 2.0

        if os.path.exists(save_data_folder_path):
            rmtree(save_data_folder_path)

        # 创建文件夹结构
        if not save_data_folder_path.endswith('/'):
            save_data_folder_path += '/'

        images_folder_path = save_data_folder_path + 'images/'
        sparse_folder_path = save_data_folder_path + 'sparse/0/'
        os.makedirs(images_folder_path, exist_ok=True)
        os.makedirs(sparse_folder_path, exist_ok=True)

        # 渲染数据
        render_data_dict = MeshRenderer.sampleRenderData(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            dtype=dtype,
            device=device,
        )

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
            f"# Number of images: {len(render_data_dict)}\n",
        ]

        # 坐标系转换矩阵：只转换相机坐标系（Y和Z轴翻转），保持世界坐标系不变
        # 原始坐标系: X右，Y上，Z后
        # COLMAP坐标系: X右，Y下，Z前
        C = np.diag([1.0, -1.0, -1.0, 1.0])

        print('[INFO][MeshRenderer::createColmapDataFolder]')
        print('\t start create colmap data folder...')
        for key, single_render_data_dict in render_data_dict.items():
            camera_data_dict = single_render_data_dict['camera']
            rgb = single_render_data_dict['rgb']

            camera = Camera.fromDict(camera_data_dict)

            # 获取原始坐标系下的world2camera矩阵
            world2camera = toNumpy(camera.world2camera, np.float64)

            # 只转换相机坐标系，不转换世界坐标系
            # world2camera_colmap = C @ world2camera
            # 这样点云（世界坐标系）保持不变，只有相机坐标系从原始坐标系转换到COLMAP坐标系
            world2camera_colmap = C @ world2camera

            # 提取旋转矩阵和平移向量
            R = world2camera_colmap[:3, :3]
            t = world2camera_colmap[:3, 3]

            # 将旋转矩阵转换为四元数 (w, x, y, z)，使用COLMAP的rotmat2qvec算法
            quat = MeshRenderer.rotationMatrixToQuaternion(R)
            qw, qx, qy, qz = quat

            # 图像文件名
            image_name = f"{key:05d}.png"
            image_id = key + 1  # COLMAP的ID从1开始

            # 保存图像
            cv2.imwrite(images_folder_path + image_name, rgb)

            # 添加到images.txt
            # 格式: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            images_txt_lines.append(
                f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {image_name}\n"
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
        print('\t generating points3D.ply from mesh vertices...')
        vertices = toNumpy(mesh.vertices, np.float32)

        # 获取顶点颜色（如果存在）
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = mesh.visual.vertex_colors
            # 确保颜色是uint8格式
            if vertex_colors.dtype != np.uint8:
                if vertex_colors.max() <= 1.0:
                    vertex_colors = (vertex_colors * 255.0).astype(np.uint8)
                else:
                    vertex_colors = vertex_colors.astype(np.uint8)
            # 确保是RGB格式（3通道）
            if vertex_colors.shape[1] > 3:
                vertex_colors = vertex_colors[:, :3]
        else:
            # 默认使用白色
            vertex_colors = np.ones((vertices.shape[0], 3), dtype=np.uint8) * 255

        # 获取顶点法线（如果存在）
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            vertex_normals = toNumpy(mesh.vertex_normals, np.float32)
        else:
            # 如果没有法线，计算法线或使用零向量
            try:
                mesh.vertex_normals = mesh.vertex_normals  # 触发自动计算
                vertex_normals = toNumpy(mesh.vertex_normals, np.float32)
            except:
                # 如果计算失败，使用零向量
                vertex_normals = np.zeros((vertices.shape[0], 3), dtype=np.float32)

        # 创建点云mesh（只包含顶点，不包含面）
        point_cloud_mesh = trimesh.Trimesh(
            vertices=vertices,
            vertex_colors=vertex_colors,
            vertex_normals=vertex_normals,
            process=False
        )

        # 导出为PLY格式
        ply_path = sparse_folder_path + 'points3D.ply'
        point_cloud_mesh.export(ply_path, file_type='ply')

        print(f'\t saved to: {save_data_folder_path}')
        print(f'\t total images: {len(render_data_dict)}')
        print(f'\t total points: {vertices.shape[0]}')
        return True

    @staticmethod
    def createOmniVGGTDataFolder(
        mesh: trimesh.Trimesh,
        save_data_folder_path: str,
        camera_num: int = 20,
        camera_dist: float = 2.5,
        width: int = 518,
        height: int = 518,
        fx: float = 500.0,
        fy: float = 500.0,
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> bool:
        if os.path.exists(save_data_folder_path):
            rmtree(save_data_folder_path)

        camera_folder_path = save_data_folder_path + 'cameras/'
        image_folder_path = save_data_folder_path + 'images/'
        depth_folder_path = save_data_folder_path + 'depths/'
        os.makedirs(camera_folder_path, exist_ok=True)
        os.makedirs(image_folder_path, exist_ok=True)
        os.makedirs(depth_folder_path, exist_ok=True)

        depth_vis_folder_path = save_data_folder_path + 'depths_vis/'
        os.makedirs(depth_vis_folder_path, exist_ok=True)

        render_data_dict = MeshRenderer.sampleRenderData(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            dtype=dtype,
            device=device,
        )

        first_camera = Camera.fromDict(render_data_dict[0]['camera'])

        world2cameraCV_global = toNumpy(first_camera.world2cameraCV, np.float32)
        camera2worldCV_global = toNumpy(first_camera.camera2worldCV, np.float32)

        with open(camera_folder_path + 'c2wCV.txt', 'w') as f:
            for j in range(3):
                f.write(str(camera2worldCV_global[j][0]) + '\t')
                f.write(str(camera2worldCV_global[j][1]) + '\t')
                f.write(str(camera2worldCV_global[j][2]) + '\t')
                f.write(str(camera2worldCV_global[j][3]) + '\n')

        print('[INFO][MeshRenderer::createOmniVGGTDataFolder]')
        print('\t start create omnivggt data folder...')
        for key, single_render_data_dict in render_data_dict.items():
            camera_data_dict = single_render_data_dict['camera']
            rgb = single_render_data_dict['rgb']
            depth = single_render_data_dict['depth']
            depth_vis = single_render_data_dict['depth_vis']

            camera = Camera.fromDict(camera_data_dict)

            camera2world = toNumpy(camera.camera2worldCV, np.float32) @ world2cameraCV_global
            intrinsic = toNumpy(camera.intrinsic, np.float32)
            with open(camera_folder_path + str(key) + '.txt', 'w') as f:
                for j in range(3):
                    f.write(str(camera2world[j][0]) + '\t')
                    f.write(str(camera2world[j][1]) + '\t')
                    f.write(str(camera2world[j][2]) + '\t')
                    f.write(str(camera2world[j][3]) + '\n')
                for j in range(3):
                    f.write(str(intrinsic[j][0]) + '\t')
                    f.write(str(intrinsic[j][1]) + '\t')
                    f.write(str(intrinsic[j][2]) + '\n')

            depth = np.where(depth == 0, 1e11, depth)

            cv2.imwrite(image_folder_path + str(key) + '.png', rgb)
            np.save(depth_folder_path + str(key) + '.npy', depth)
            cv2.imwrite(depth_vis_folder_path + str(key) + '.png', depth_vis)
        return True

    @staticmethod
    def createDA3DataFile(
        mesh: trimesh.Trimesh,
        save_data_file_path: str,
        camera_num: int = 20,
        camera_dist: float = 2.5,
        width: int = 518,
        height: int = 518,
        fx: float = 500.0,
        fy: float = 500.0,
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> bool:
        render_data_dict = MeshRenderer.sampleRenderData(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            dtype=dtype,
            device=device,
        )

        images = []
        extrinsics = []
        intrinsics = []

        print('[INFO][ImageMeshMapper::createDA3DataFile]')
        print('\t start create da3 data file...')
        for key, single_render_data_dict in render_data_dict.items():
            camera_data_dict = single_render_data_dict['camera']
            rgb = single_render_data_dict['rgb'][..., ::-1]
            # depth = single_render_data_dict['depth']
            # depth_vis = single_render_data_dict['depth_vis']

            camera = Camera.fromDict(camera_data_dict)

            extrinsic = toNumpy(camera.world2cameraCV, np.float32)
            intrinsic = toNumpy(camera.intrinsic, np.float32)

            images.append(rgb)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

        images_np = np.stack(images, axis=0)
        extrinsics_np = np.stack(extrinsics, axis=0)
        intrinsics_np = np.stack(intrinsics, axis=0)

        da3_data_dict = {
            'images': images_np,
            'extrinsics': extrinsics_np,
            'intrinsics': intrinsics_np,
        }

        createFileFolder(save_data_file_path)
        np.save(save_data_file_path, da3_data_dict, allow_pickle=True)
        return True
