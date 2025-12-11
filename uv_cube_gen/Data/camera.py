import torch
import numpy as np
from typing import Union
from copy import deepcopy

from uv_cube_gen.Method.data import toTensor


class CameraData(object):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 240.0,
        pos: Union[torch.Tensor, np.ndarray, list] = [0, 0, 0],
        look_at: Union[torch.Tensor, np.ndarray, list] = [1, 0, 0],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 1, 0],
        rot: Union[torch.Tensor, np.ndarray, list, None] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.pos = toTensor(pos)

        if rot is not None:
            self.rot = toTensor(rot)
        else:
            self.setPoseByLookAt(look_at, up)
        return

    def clone(self):
        return deepcopy(self)

    def setPoseByLookAt(
        self,
        look_at: Union[torch.Tensor, np.ndarray, list],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 1, 0],
        pos: Union[torch.Tensor, np.ndarray, list, None] = None,
    ) -> bool:
        look_at = toTensor(look_at)
        up = toTensor(up)

        if pos is not None:
            pos = toTensor(pos)
        else:
            pos = self.pos

        # 确保所有向量都是1D tensor
        if look_at.ndim > 1:
            look_at = look_at.flatten()
        if up.ndim > 1:
            up = up.flatten()
        if pos.ndim > 1:
            pos = pos.flatten()

        forward = look_at - pos
        forward = forward / (torch.linalg.norm(forward) + 1e-8)

        right = torch.linalg.cross(forward, up)
        right = right / (torch.linalg.norm(right) + 1e-8)

        up_corrected = torch.linalg.cross(right, forward)
        up_corrected = up_corrected / (torch.linalg.norm(up_corrected) + 1e-8)

        rot = torch.stack([right, up_corrected, forward], dim=1)

        self.pos = pos
        self.rot = rot
        return True

    def outputInfo(
        self,
        info_level: int = 0,
    ) -> bool:
        line_start = '\t' * info_level

        right = self.rot[:, 0]
        up = self.rot[:, 1]
        forward = self.rot[:, 2]

        print(line_start + '[INFO][CameraData]')
        print(line_start + '\t image_size: [', self.width, ',', self.height, ']')
        print(line_start + '\t focal: [', self.fx, ',', self.fy, ',', self.cx, ',', self.cy, ']')
        print(line_start + '\t pos:', self.pos.numpy().tolist())
        print(line_start + '\t forward:', forward.numpy().tolist())
        print(line_start + '\t up:', up.numpy().tolist())
        print(line_start + '\t right:', right.numpy().tolist())
        return True
