import os
import cv2
import torch
import numpy as np
from typing import List, Union, Optional

from camera_control.Method.data import toNumpy, toTensor


class RGBChannel(object):
    def __init__(self) -> None:
        self.image: torch.Tensor = None
        self.image_id: Optional[str]=None
        return

    def update(self) -> bool:
        if self.image is not None:
            self.image = self.image.to(dtype=self.dtype, device=self.device)
        return True

    @property
    def image_cv(self) -> np.ndarray:
        return toNumpy(self.image * 255.0, np.uint8)[..., ::-1]

    def setImageSize(
        self,
        width: int,
        height: int,
    ) -> bool:
        self.width = int(width)
        self.height = int(height)
        return True

    def loadImage(
        self,
        image: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        image = toTensor(image, self.dtype, self.device)

        self.setImageSize(image.shape[1], image.shape[0])

        self.image = image.reshape(self.height, self.width, 3)
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

        return self.loadImage(image)

    def toImageUV(self) -> torch.Tensor:
        """
        生成每个像素的归一化UV坐标，原点在左下角，u向右，v向上。
        返回: torch.Tensor, shape (height, width, 2), dtype为self.dtype, device为self.device
        """
        u = torch.arange(self.width, dtype=self.dtype, device=self.device) / (self.width - 1.0)
        v = torch.arange(self.height, dtype=self.dtype, device=self.device) / (self.height - 1.0)

        uu, vv = torch.meshgrid(u, v, indexing='xy')  # uu/vv shape: [height, width]
        vv_new = 1.0 - vv  # [height, width], 左下角为0,0，v向上增大

        uv = torch.stack([uu, vv_new], dim=-1)  # shape: (height, width, 2)
        return uv

    def toMaskedImage(
        self,
        background_color: List[float] = [255, 255, 255],
    ) -> torch.Tensor:
        """
        self.mask 不存在时等价于返回 self.image；存在时按 image 每个像素的 UV 在 mask 上
        最近邻采样得到遮罩，True 处保留 self.image，False 处填 background_color。
        background_color: [R, G, B]，默认 0–255，内部会除以 255；输出 tensor 为 [0, 1]。
        返回: (H, W, 3) tensor
        """
        assert self.image is not None
        if getattr(self, "mask", None) is None:
            return self.image
        uv = self.toImageUV()  # (H, W, 2)
        mask_t = self.sampleMaskAtUV(uv).unsqueeze(-1)  # (H, W, 1)
        bg = toTensor(background_color, self.dtype, self.device) / 255.0
        bg = bg.view(1, 1, 3) if bg.numel() == 3 else bg
        out = torch.where(mask_t, self.image, bg.to(self.image.dtype))
        return out

    def toMaskedImageCV(
        self,
        background_color: List[float] = [255, 255, 255],
    ) -> np.ndarray:
        """toMaskedImage 的 OpenCV BGR uint8 版本。无 mask 时等价于 self.image_cv。"""
        return toNumpy(self.toMaskedImage(background_color) * 255.0, np.uint8)[..., ::-1]
