import torch
import numpy as np
from typing import Union

from camera_control.Method.data import toNumpy, toTensor


class RGBChannel(object):
    def __init__(self) -> None:
        self.image: torch.Tensor = None
        return

    def update(self) -> bool:
        if self.image is not None:
            self.image = self.image.to(dtype=self.dtype, device=self.device)
        return True

    @property
    def image_cv(self) -> np.ndarray:
        return toNumpy(self.image * 255.0, np.uint8)[..., ::-1]

    def loadImage(
        self,
        image: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        self.image = toTensor(image, self.dtype, self.device).reshape(self.height, self.width, 3)
        return True
