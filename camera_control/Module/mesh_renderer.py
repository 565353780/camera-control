import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import trange
from shutil import rmtree
from typing import Optional, List

from camera_control.Method.pcd import toPcd
from camera_control.Method.data import toNumpy
from camera_control.Method.sample import sampleCameras
from camera_control.Method.path import createFileFolder
from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer
from camera_control.Module.camera_convertor import CameraConvertor


class MeshRenderer(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def _writeMatrixRows(f, matrix: np.ndarray, row_num: int, col_num: int) -> None:
        """将 ``matrix`` 的前 ``row_num`` 行、前 ``col_num`` 列以 tab 分隔、逐行换行写入文件句柄 ``f``。"""
        for j in range(row_num):
            f.write('\t'.join(str(matrix[j][k]) for k in range(col_num)) + '\n')
        return

    @staticmethod
    def renderCameraData(
        mesh: trimesh.Trimesh,
        camera_list: List[Camera],
        bg_color: list=[255, 255, 255],
        device: str='cuda:0',
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool=True,
        random_lighting: bool=False,
        lighting: Optional[dict]=None,
        light_seed: Optional[int]=None,
        light_kwargs: Optional[dict]=None,
        set_image_id: bool=False,
    ) -> List[Camera]:
        """在调用方给定的相机位姿 / 内参上渲染 mask/rgb/depth/normal 并原地 load 回每个相机。

        这是 "已知视角 -> 渲染条件" 的纯原子: 不重新采样相机、不调整位姿。
        :meth:`sampleRenderData` 在采样 + 调整位姿后复用本函数完成渲染。
        返回的就是传入的 ``camera_list``。

        Args:
            set_image_id: 是否按索引为每个相机写入 ``image_id`` (``{i:06d}.png``)。
        """
        # 每物体采样一套世界系随机光照，所有视图共享 -> 多视图光照一致
        if lighting is None and random_lighting:
            lighting = NVDiffRastRenderer.sampleRandomLighting(
                space='world', seed=light_seed, **(light_kwargs or {}))

        print('[INFO][MeshRenderer::renderCameraData]')
        print('\t start render camera data...')
        for i in trange(len(camera_list)):
            camera = camera_list[i]
            camera.to(device=device)

            render_dict = NVDiffRastRenderer.render(
                mesh=mesh,
                camera=camera,
                render_types=['mask', 'rgb', 'depth', 'normal'],
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
                enable_antialias=enable_antialias,
                lighting=lighting,
            )

            if set_image_id:
                camera.image_id = f'{i:06d}.png'

            camera.loadMask(render_dict['mask'])

            camera.loadImage(render_dict['rgb'])

            camera.loadDepth(render_dict['depth'])

            # renderer 已同时算好 normal_world 与 normal_camera (满足
            #   normal_camera = normal_world @ R^T, 与 Camera 同步公式一致),
            # 直接各自 load 并关闭同步, 避免重复一次 [H,W,3] 的法线变换矩阵乘。
            camera.loadNormalWorld(render_dict['normal_world'], is_update_normal_camera=False)
            camera.loadNormalCamera(render_dict['normal_camera'], is_update_normal_world=False)

        return camera_list

    @staticmethod
    def sampleRenderData(
        mesh: trimesh.Trimesh,
        candidate_camera_num: int = 20,
        camera_num: int = 20,
        camera_dist_range: List[float] = [2.5, 2.5],
        width: int = 512,
        height: int = 512,
        fovx_degree_range: List[float] = [60.0, 60.0],
        bg_color: list=[255, 255, 255],
        dtype = torch.float32,
        device: str = 'cuda:0',
        focus_center_ratio: float=1.0,
        up_direction: Optional[List[float]] = [0, 0, 1],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool=True,
        safe_pixel_num: Optional[int]=None,
        random_lighting: bool=False,
        lighting: Optional[dict]=None,
        light_seed: Optional[int]=None,
        light_kwargs: Optional[dict]=None,
    ) -> List[Camera]:
        """采样相机位姿 (可选最优位姿筛选), 再复用 :meth:`renderCameraData` 渲染。"""
        camera_list = sampleCameras(
            mesh=mesh,
            candidate_camera_num=candidate_camera_num,
            camera_num=camera_num,
            camera_dist_range=camera_dist_range,
            width=width,
            height=height,
            fovx_degree_range=fovx_degree_range,
            dtype=dtype,
            device=device,
            focus_center_ratio=focus_center_ratio,
            up_direction=up_direction,
        )

        if safe_pixel_num is not None:
            camera_list = CameraConvertor.getBestPoseCameras(
                camera_list=camera_list,
                pts=mesh.vertices,
                safe_pixel_num=safe_pixel_num,
            )

        return MeshRenderer.renderCameraData(
            mesh=mesh,
            camera_list=camera_list,
            bg_color=bg_color,
            device=device,
            vertices_tensor=vertices_tensor,
            enable_antialias=enable_antialias,
            random_lighting=random_lighting,
            lighting=lighting,
            light_seed=light_seed,
            light_kwargs=light_kwargs,
            set_image_id=True,
        )

    @staticmethod
    def createColmapDataFolder(
        mesh: trimesh.Trimesh,
        save_data_folder_path: str,
        candidate_camera_num: int = 20,
        camera_num: int = 20,
        camera_dist_range: List[float] = [2.5, 2.5],
        width: int = 512,
        height: int = 512,
        fovx_degree_range: List[float] = [60.0, 60.0],
        bg_color: list=[255, 255, 255],
        dtype = torch.float32,
        device: str = 'cuda:0',
        focus_center_ratio: float=1.0,
        up_direction: Optional[List[float]] = [0, 0, 1],
        enable_antialias: bool=True,
        random_lighting: bool=False,
        lighting: Optional[dict]=None,
        light_seed: Optional[int]=None,
        light_kwargs: Optional[dict]=None,
    ) -> bool:
        camera_list = MeshRenderer.sampleRenderData(
            mesh=mesh,
            candidate_camera_num=candidate_camera_num,
            camera_num=camera_num,
            camera_dist_range=camera_dist_range,
            width=width,
            height=height,
            fovx_degree_range=fovx_degree_range,
            bg_color=bg_color,
            dtype=dtype,
            device=device,
            focus_center_ratio=focus_center_ratio,
            up_direction=up_direction,
            enable_antialias=enable_antialias,
            random_lighting=random_lighting,
            lighting=lighting,
            light_seed=light_seed,
            light_kwargs=light_kwargs,
        )

        # 处理有/无顶点颜色、纹理等多种情况
        visual = mesh.visual
        vertex_colors = None

        # 1. 直接存在 vertex_colors 的情况（ColorVisuals）
        if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
            vertex_colors = visual.vertex_colors[:, :3]
        else:
            # 2. 纹理可视化（TextureVisuals），尝试烘焙为顶点颜色
            try:
                visual_color = visual.to_color()
                if hasattr(visual_color, 'vertex_colors') and visual_color.vertex_colors is not None:
                    vertex_colors = visual_color.vertex_colors[:, :3]
            except Exception:
                vertex_colors = None

        # 3. 仍然拿不到颜色时，使用白色占位
        if vertex_colors is None:
            vertex_colors = np.ones((len(mesh.vertices), 3), dtype=np.uint8) * 255

        pcd = toPcd(mesh.vertices, vertex_colors)

        return CameraConvertor.createColmapDataFolder(
            cameras=camera_list,
            save_data_folder_path=save_data_folder_path,
            pcd=pcd,
        )

    @staticmethod
    def createOmniVGGTDataFolder(
        mesh: trimesh.Trimesh,
        save_data_folder_path: str,
        candidate_camera_num: int = 20,
        camera_num: int = 20,
        camera_dist_range: List[float] = [2.5, 2.5],
        width: int = 512,
        height: int = 512,
        fovx_degree_range: List[float] = [60.0, 60.0],
        bg_color: list=[255, 255, 255],
        dtype = torch.float32,
        device: str = 'cuda:0',
        focus_center_ratio: float=1.0,
        up_direction: Optional[List[float]] = [0, 0, 1],
        enable_antialias: bool=True,
        random_lighting: bool=False,
        lighting: Optional[dict]=None,
        light_seed: Optional[int]=None,
        light_kwargs: Optional[dict]=None,
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

        camera_list = MeshRenderer.sampleRenderData(
            mesh=mesh,
            candidate_camera_num=candidate_camera_num,
            camera_num=camera_num,
            camera_dist_range=camera_dist_range,
            width=width,
            height=height,
            fovx_degree_range=fovx_degree_range,
            bg_color=bg_color,
            dtype=dtype,
            device=device,
            focus_center_ratio=focus_center_ratio,
            up_direction=up_direction,
            enable_antialias=enable_antialias,
            random_lighting=random_lighting,
            lighting=lighting,
            light_seed=light_seed,
            light_kwargs=light_kwargs,
        )

        first_camera = camera_list[0]

        world2cameraCV_global = toNumpy(first_camera.world2cameraCV, np.float32)
        camera2worldCV_global = toNumpy(first_camera.camera2worldCV, np.float32)

        with open(camera_folder_path + 'c2wCV.txt', 'w') as f:
            MeshRenderer._writeMatrixRows(f, camera2worldCV_global, 3, 4)

        print('[INFO][MeshRenderer::createOmniVGGTDataFolder]')
        print('\t start create omnivggt data folder...')
        for i in range(len(camera_list)):
            camera = camera_list[i]
            rgb = camera.toImageCV(use_mask=False)
            depth = camera.depth
            depth_vis = camera.toDepthVisCV()

            camera2world = toNumpy(camera.camera2worldCV, np.float32) @ world2cameraCV_global
            intrinsic = toNumpy(camera.intrinsic, np.float32)
            with open(camera_folder_path + str(i) + '.txt', 'w') as f:
                MeshRenderer._writeMatrixRows(f, camera2world, 3, 4)
                MeshRenderer._writeMatrixRows(f, intrinsic, 3, 3)

            depth = np.where(depth == 0, 1e11, depth)

            cv2.imwrite(image_folder_path + str(i) + '.png', rgb)
            np.save(depth_folder_path + str(i) + '.npy', depth)
            cv2.imwrite(depth_vis_folder_path + str(i) + '.png', depth_vis)
        return True

    @staticmethod
    def createDA3DataFile(
        mesh: trimesh.Trimesh,
        save_data_file_path: str,
        candidate_camera_num: int = 20,
        camera_num: int = 20,
        camera_dist_range: List[float] = [2.5, 2.5],
        width: int = 512,
        height: int = 512,
        fovx_degree_range: List[float] = [60.0, 60.0],
        bg_color: list=[255, 255, 255],
        dtype = torch.float32,
        device: str = 'cuda:0',
        focus_center_ratio: float=1.0,
        up_direction: Optional[List[float]] = [0, 0, 1],
        enable_antialias: bool=True,
        random_lighting: bool=False,
        lighting: Optional[dict]=None,
        light_seed: Optional[int]=None,
        light_kwargs: Optional[dict]=None,
    ) -> bool:
        camera_list = MeshRenderer.sampleRenderData(
            mesh=mesh,
            candidate_camera_num=candidate_camera_num,
            camera_num=camera_num,
            camera_dist_range=camera_dist_range,
            width=width,
            height=height,
            fovx_degree_range=fovx_degree_range,
            bg_color=bg_color,
            dtype=dtype,
            device=device,
            focus_center_ratio=focus_center_ratio,
            up_direction=up_direction,
            enable_antialias=enable_antialias,
            random_lighting=random_lighting,
            lighting=lighting,
            light_seed=light_seed,
            light_kwargs=light_kwargs,
        )

        images = []
        extrinsics = []
        intrinsics = []

        print('[INFO][ImageMeshMapper::createDA3DataFile]')
        print('\t start create da3 data file...')
        for i in range(len(camera_list)):
            camera = camera_list[i]
            rgb = camera.toImageCV(use_mask=False)

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
