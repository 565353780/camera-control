import torch
import numpy as np
from typing import Union

from camera_control.Method.data import toNumpy, toTensor


class MaskChannel(object):
    def __init__(self) -> None:
        self.mask: torch.Tensor = None
        return

    def update(self) -> bool:
        if self.mask is not None:
            self.mask = self.mask.to(dtype=torch.bool, device=self.device)
        return True

    @property
    def mask_cv(self) -> np.ndarray:
        """
        Converts the boolean mask tensor to an RGB (uint8) numpy image for OpenCV.
        Mask True/1 -> [255, 255, 255], False/0 -> [0, 0, 0]
        """
        mask_np = toNumpy(self.mask, np.bool_)
        mask_gray = (mask_np.astype(np.uint8)) * 255  # shape: H x W, values 0 or 255
        mask_rgb = np.stack([mask_gray]*3, axis=-1)   # shape: H x W x 3
        return mask_rgb

    def loadMask(
        self,
        mask: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        self.mask = toTensor(mask, torch.bool, self.device)
        return True

    def sampleMaskAtUV(self, uv_grid: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        按归一化 UV 在 self.mask 上做最近邻采样，供各模态按自身 UV 对齐 mask，节省重复计算。
        uv_grid: (..., 2) 归一化 UV [0,1]，u 向右、v 向上（v=0 为左下）。
        mask 行 0 对应 v=1（上），行 Mh-1 对应 v=0（下）。
        返回: 与 uv_grid 前若干维同形状的 bool tensor。
        """
        assert self.mask is not None
        uv_grid = toTensor(uv_grid, torch.float32, self.device)
        Mh, Mw = self.mask.shape[0], self.mask.shape[1]
        u = uv_grid[..., 0]
        v = uv_grid[..., 1]
        idx_w = (u * (Mw - 1)).round().long().clamp(0, Mw - 1)
        idx_h = ((1.0 - v) * (Mh - 1)).round().long().clamp(0, Mh - 1)
        return self.mask[idx_h, idx_w]
