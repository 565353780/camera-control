import torch
import numpy as np
from typing import Union

from camera_control.Method.data import toNumpy, toTensor


class NormalChannel(object):
    def __init__(self) -> None:
        self.normal: torch.Tensor = None
        return

    def update(self) -> bool:
        if self.normal is not None:
            self.normal = self.normal.to(dtype=self.dtype, device=self.device)
        return True

    @property
    def normal_cv(self) -> np.ndarray:
        return toNumpy(self.normal * 255.0, np.uint8)[..., ::-1]

    def loadNormal(
        self,
        normal: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        self.normal = toTensor(normal, self.dtype, self.device).reshape(self.height, self.width, 3)
        return True
