import torch
import numpy as np
from typing import List, Union, Optional

from camera_control.Method.io import loadImage
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

    def setImageSize(
        self,
        width: int,
        height: int,
    ) -> bool:
        self.width = int(width)
        self.height = int(height)

        if self.image is None:
            return True
        if self.image.shape[0] == self.height and self.image.shape[1] == self.width:
            return True

        self.image = torch.nn.functional.interpolate(
            self.image.permute(2, 0, 1).unsqueeze(0),
            size=(self.height, self.width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        return True

    def loadImage(
        self,
        image: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        image = toTensor(image, self.dtype, self.device)

        self.setImageSize(image.shape[1], image.shape[0])

        self.image = image.reshape(self.height, self.width, 3)
        return True

    def loadImageCV(
        self,
        image_cv: np.ndarray,
    ) -> bool:
        image = image_cv[..., ::-1].astype(np.float32) / 255.0
        return self.loadImage(image)

    def loadImageFile(
        self,
        image_file_path: str,
    ) -> bool:
        image = loadImage(image_file_path)

        if image is None:
            print('[ERROR][RGBChannel::loadImageFile]')
            print('\t loadImage failed!')
            return False

        return self.loadImageCV(image)

    def toImageUV(self) -> torch.Tensor:
        """
        生成每个像素中心的归一化UV坐标，原点在左下角，u向右，v向上。
        返回: torch.Tensor, shape (height, width, 2), dtype为self.dtype, device为self.device
        """
        w = max(self.width, 1)
        h = max(self.height, 1)
        u = (torch.arange(w, dtype=self.dtype, device=self.device) + 0.5) / w
        v = (torch.arange(h, dtype=self.dtype, device=self.device) + 0.5) / h

        uu, vv = torch.meshgrid(u, v, indexing='xy')  # uu/vv shape: [height, width]
        vv_new = 1.0 - vv  # [height, width], 左下角为0,0，v向上增大

        uv = torch.stack([uu, vv_new], dim=-1)  # shape: (height, width, 2)
        return uv

    def toImage(
        self,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        """
        use_mask=False 或 mask 不存在时返回原始 self.image；
        存在时按 image 每个像素的 UV 在 mask 上最近邻采样得到遮罩，
        True 处保留 self.image，False 处填 background_color。
        background_color: [R, G, B]，默认 0–255，内部会除以 255。
        返回: (H, W, 3) tensor
        """
        assert self.image is not None
        if not use_mask or getattr(self, "mask", None) is None:
            return self.image
        uv = self.toImageUV()
        mask_t = self.sampleMaskAtUV(uv, mask_smaller_pixel_num).unsqueeze(-1)
        bg = toTensor(background_color, self.dtype, self.device) / 255.0
        bg = bg.view(1, 1, 3) if bg.numel() == 3 else bg
        return torch.where(mask_t, self.image, bg.to(self.image.dtype))

    def toImageCV(
        self,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> np.ndarray:
        return toNumpy(self.toImage(
            background_color=background_color,
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        ) * 255.0, np.uint8)[..., ::-1]

    def sampleRGBAtUV(self, uv_grid: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        按归一化 UV 在 self.image 上做最近邻采样，供各模态按自身 UV 对齐 image，节省重复计算。
        uv_grid: (..., 2) 归一化 UV [0,1]，u 向右、v 向上（v=0 为左下）。
        image 行 0 对应 v=1（上），行 Mh-1 对应 v=0（下）。
        返回: 与 uv_grid 前若干维同形状的 (..., 3) float tensor。
        """
        assert self.image is not None
        uv_grid = toTensor(uv_grid, torch.float32, self.device)
        Mh, Mw = self.image.shape[0], self.image.shape[1]
        u = uv_grid[..., 0]
        v = uv_grid[..., 1]
        idx_w = (u * Mw - 0.5).round().long().clamp(0, Mw - 1)
        idx_h = ((1.0 - v) * Mh - 0.5).round().long().clamp(0, Mh - 1)
        return self.image[idx_h, idx_w]
