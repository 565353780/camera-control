import torch
import numpy as np

from typing import List

from camera_control.Module.camera import Camera


class CameraFilter(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def sampleFarCameraIdxs(
        camera_list: List[Camera],
        sample_camera_num: int,
    ) -> List[int]:
        if sample_camera_num >= len(camera_list):
            return list(range(len(camera_list)))

        selected_indices = [0]  # 第一个视角默认会选中
        selected = [0]

        if sample_camera_num == 1:
            return selected_indices

        poses = [camera.pos for camera in camera_list]  # list of torch.Tensor([x, y, z])
        poses_np = torch.stack(poses).cpu().numpy()  # shape (N,3)

        available = list(range(1, len(poses)))
        for _ in range(sample_camera_num-1):
            dists = []
            for i in available:
                min_dist = min(np.linalg.norm(poses_np[i] - poses_np[j]) for j in selected)
                dists.append(min_dist)
            max_idx = available[np.argmax(dists)]
            selected.append(max_idx)
            available.remove(max_idx)
        selected_indices = selected

        return selected_indices

    @staticmethod
    def sampleFarCameras(
        camera_list: List[Camera],
        sample_camera_num: int,
    ) -> List[Camera]:
        far_camera_idxs = CameraFilter.sampleFarCameraIdxs(camera_list, sample_camera_num)
        return [camera_list[idx] for idx in far_camera_idxs]
