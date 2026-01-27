import os
import cv2
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

    def loadImageFile(
        self,
        image_file_path: str,
    ) -> bool:
        if not os.path.exists(image_file_path):
            print('[ERROR][RGBChannel::loadImageFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return False

        image = cv2.imread(image_file_path)[..., ::-1].astype(np.float32) / 255.0

        self.height, self.width = image.shape[:2]

        return self.loadImage(image)
