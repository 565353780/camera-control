import os
import cv2
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

    def loadMask(
        self,
        mask: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        self.mask = toTensor(mask, torch.bool, self.device)
        return True

    def loadMaskFile(
        self,
        mask_file_path: str,
    ) -> bool:
        if not os.path.exists(mask_file_path):
            print('[ERROR][MaskChannel::loadMaskFile]')
            print('\t mask file not exist!')
            print('\t mask_file_path:', mask_file_path)
            return False

        img = cv2.imread(mask_file_path)
        if img is None:
            print('[ERROR][MaskChannel::loadMaskFile]')
            print('\t cv2.imread failed!')
            print('\t mask_file_path:', mask_file_path)
            return False

        if img.ndim == 2:
            mask_np = (img > 0)
        else:
            # 3 通道 (BGR)：任一通道 > 0 视为前景
            mask_np = (img.max(axis=-1) > 0)
        self.mask = toTensor(mask_np, torch.bool, self.device)  # (H, W)
        return True

    def toMask(self, mask_smaller_pixel_num: int=0) -> torch.Tensor:
        """
        返回 self.mask 的 bool tensor；mask_smaller_pixel_num > 0 时通过形态学腐蚀收缩物体区域。
        """
        assert self.mask is not None
        if mask_smaller_pixel_num <= 0:
            return self.mask

        k = 2 * mask_smaller_pixel_num + 1
        kernel = torch.ones(1, 1, k, k, device=self.mask.device, dtype=torch.float32)
        mask_float = self.mask.float().unsqueeze(0).unsqueeze(0)
        eroded = torch.nn.functional.conv2d(mask_float, kernel, padding=mask_smaller_pixel_num)
        return (eroded.squeeze(0).squeeze(0) >= k * k).bool()

    def toMaskCV(
        self,
        mask_smaller_pixel_num: int=0,
    ) -> np.ndarray:
        """
        Converts the boolean mask tensor to an RGB (uint8) numpy image for OpenCV.
        Mask True/1 -> [255, 255, 255], False/0 -> [0, 0, 0]
        mask_smaller_pixel_num: >0 时对 mask 做形态学腐蚀（False 区域向外扩散），得到更保守的物体 mask。
        """
        mask_np = toNumpy(self.toMask(mask_smaller_pixel_num), np.bool_)
        mask_gray = (mask_np.astype(np.uint8)) * 255
        mask_rgb = np.stack([mask_gray]*3, axis=-1)
        return mask_rgb

    def sampleMaskAtUV(
        self,
        uv_grid: Union[torch.Tensor, np.ndarray],
        mask_smaller_pixel_num: int=0,
    ) -> torch.Tensor:
        """
        按归一化 UV 在 self.mask 上做最近邻采样，供各模态按自身 UV 对齐 mask，节省重复计算。
        uv_grid: (..., 2) 归一化 UV [0,1]，u 向右、v 向上（v=0 为左下）。
        mask 行 0 对应 v=1（上），行 Mh-1 对应 v=0（下）。
        mask_smaller_pixel_num: >0 时对 mask 做形态学腐蚀（False 区域向外扩散），得到更保守的物体 mask。
        返回: 与 uv_grid 前若干维同形状的 bool tensor。
        """
        uv_grid = toTensor(uv_grid, torch.float32, self.device)

        mask = self.toMask(mask_smaller_pixel_num)
        Mh, Mw = mask.shape[0], mask.shape[1]
        u = uv_grid[..., 0]
        v = uv_grid[..., 1]
        idx_w = (u * Mw - 0.5).round().long().clamp(0, Mw - 1)
        idx_h = ((1.0 - v) * Mh - 0.5).round().long().clamp(0, Mh - 1)
        return mask[idx_h, idx_w]

    def sampleMaskWithSize(self, width: int, height: int, mask_smaller_pixel_num: int = 0) -> torch.Tensor:
        """
        按给定宽高生成像素中心 UV 网格，再通过 sampleMaskAtUV 采样 mask。
        返回: (height, width) bool tensor。
        """
        w = max(width, 1)
        h = max(height, 1)
        u = (torch.arange(w, dtype=torch.float32, device=self.device) + 0.5) / w
        v = (torch.arange(h, dtype=torch.float32, device=self.device) + 0.5) / h
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        vv = 1.0 - vv
        uv_grid = torch.stack([uu, vv], dim=-1)  # (height, width, 2)
        return self.sampleMaskAtUV(uv_grid, mask_smaller_pixel_num)
