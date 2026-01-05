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
            rgb = single_render_data_dict['rgb']
            # depth = single_render_data_dict['depth']
            # depth_vis = single_render_data_dict['depth_vis']

            camera = Camera.fromDict(camera_data_dict)

            camera2world = toNumpy(camera.camera2world, np.float32)
            intrinsic = toNumpy(camera.intrinsic, np.float32)

            images.append(rgb)
            extrinsics.append(camera2world)
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
