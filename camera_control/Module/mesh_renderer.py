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
from camera_control.Method.sample import sampleCamera
from camera_control.Method.path import createFileFolder
from camera_control.Module.rgbd_camera import RGBDCamera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer
from camera_control.Module.camera_convertor import CameraConvertor


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
        bg_color: list=[255, 255, 255],
        dtype = torch.float32,
        device: str = 'cuda:0',
        vertices_tensor: Optional[torch.Tensor] = None,
    ) -> List[RGBDCamera]:
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

        print('[INFO][MeshRenderer::sampleRenderData]')
        print('\t start sample render data...')
        for i in trange(len(camera_list)):
            camera = camera_list[i]

            render_image_dict = NVDiffRastRenderer.renderTexture(
                mesh=mesh,
                camera=camera,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
            )

            render_depth_dict = NVDiffRastRenderer.renderDepth(
                mesh=mesh,
                camera=camera,
                bg_color=bg_color,
                vertices_tensor=vertices_tensor,
            )

            image = (render_image_dict['image'] * 255.0)[..., ::-1]
            camera.loadImage(image)

            camera.loadDepth(render_depth_dict['depth'])

        return camera_list

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
        bg_color: list=[255, 255, 255],
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> bool:
        camera_list = MeshRenderer.sampleRenderData(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            bg_color=bg_color,
            dtype=dtype,
            device=device,
        )

        pcd = toPcd(mesh.vertiecs, mesh.visual.vertex_colors)

        return CameraConvertor.createColmapDataFolder(
            cameras=camera_list,
            pcd=pcd,
            save_data_folder_path=save_data_folder_path,
        )

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
        bg_color: list=[255, 255, 255],
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

        camera_list = MeshRenderer.sampleRenderData(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            bg_color=bg_color,
            dtype=dtype,
            device=device,
        )

        first_camera = camera_list[0]

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
        for i in range(len(camera_list)):
            camera = camera_list[i]
            rgb = camera.image_cv
            depth = camera.depth
            depth_vis = camera.depth_vis_cv

            camera2world = toNumpy(camera.camera2worldCV, np.float32) @ world2cameraCV_global
            intrinsic = toNumpy(camera.intrinsic, np.float32)
            with open(camera_folder_path + str(i) + '.txt', 'w') as f:
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

            cv2.imwrite(image_folder_path + str(i) + '.png', rgb)
            np.save(depth_folder_path + str(i) + '.npy', depth)
            cv2.imwrite(depth_vis_folder_path + str(i) + '.png', depth_vis)
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
        bg_color: list=[255, 255, 255],
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> bool:
        camera_list = MeshRenderer.sampleRenderData(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            bg_color=bg_color,
            dtype=dtype,
            device=device,
        )

        images = []
        extrinsics = []
        intrinsics = []

        print('[INFO][ImageMeshMapper::createDA3DataFile]')
        print('\t start create da3 data file...')
        for i in range(len(camera_list)):
            camera = camera_list[i]
            rgb = camera.image_cv

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
