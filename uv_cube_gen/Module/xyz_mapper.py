import torch
import numpy as np
from typing import Union

from uv_cube_gen.Method.data import toTensor

class XYZMapper(object):
    def __init__(
        self,
        x_dir: Union[torch.Tensor, np.ndarray, list] = [1, 0, 0],
        y_dir: Union[torch.Tensor, np.ndarray, list] = [0, 1, 0],
        z_dir: Union[torch.Tensor, np.ndarray, list] = [0, 0, 1],
    ) -> None:
        x_dir = toTensor(x_dir)
        y_dir = toTensor(y_dir)
        z_dir = toTensor(z_dir)

        self.rot = torch.stack([x_dir, y_dir, z_dir], dim=1)
        return

    def toWorld(
        self,
        local_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        local_points = toTensor(local_points)

        world_points = torch.matmul(local_points, self.rot.T)

        return world_points

    def toLocal(
        self,
        world_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        world_points = toTensor(world_points)

        local_points = torch.matmul(world_points, self.rot)

        return local_points
