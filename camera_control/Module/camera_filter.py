import torch

from tqdm import trange
from typing import List

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera


class CameraFilter(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def selectFPSCameras(
        camera_list: List[Camera],
        camera_num: int,
    ) -> List[int]:
        """
        基于相机世界坐标的最远点采样 (FPS) 选择相机。
        第一次选择的相机默认为 camera_list[0]，后续每次选择与已选集合距离最远的相机。

        Args:
            camera_list: 相机列表
            camera_num: 需要选择的相机数量

        Returns:
            按 FPS 顺序选中的相机索引
        """
        if not camera_list:
            return []
        if camera_num <= 0:
            return []
        n = len(camera_list)
        if camera_num >= n:
            return list(range(n))

        # 使用 camera.pos（世界坐标，来自 world2camera / camera2world）收集所有相机位置
        positions = []
        for cam in camera_list:
            pos = cam.camera2world[:3, 3]  # (3,) 世界坐标
            pos = toTensor(pos, torch.float32, "cpu")
            positions.append(pos)
        positions = torch.stack(positions, dim=0)  # (n, 3)

        print('[INFO][CameraFilter::selectFPSCameras]')
        print('\t start select fps cameras...')
        selected = [0]  # 第一次默认为 camera[0]
        for _ in trange(camera_num - 1):
            unselected = [i for i in range(n) if i not in selected]
            if not unselected:
                break
            pos_selected = positions[selected]  # (S, 3)
            pos_unselected = positions[unselected]  # (U, 3)
            # 每个未选点到已选集合的最小距离
            diff = pos_unselected.unsqueeze(1) - pos_selected.unsqueeze(0)  # (U, S, 3)
            dist_to_set = torch.linalg.norm(diff, dim=-1).min(dim=1).values  # (U,)
            best_local_idx = dist_to_set.argmax().item()
            best_global_idx = unselected[best_local_idx]
            selected.append(best_global_idx)

        return selected
