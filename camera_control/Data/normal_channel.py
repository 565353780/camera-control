import torch
import numpy as np
from typing import List, Union

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
        self.normal = toTensor(normal, self.dtype, self.device)
        return True

    def toNormalUV(self) -> torch.Tensor:
        """
        生成 self.normal 每个像素的归一化 UV [0,1]，与 camera 的 UV 约定一致。
        返回: (H, W, 2) tensor，H/W 来自 self.normal.shape[:2]。
        """
        assert self.normal is not None
        h, w = self.normal.shape[0], self.normal.shape[1]
        u = torch.arange(w, dtype=self.dtype, device=self.device) / max(w - 1, 1)
        v = torch.arange(h, dtype=self.dtype, device=self.device) / max(h - 1, 1)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        vv_new = 1.0 - vv  # 左下角为 (0,0)，v 向上
        uv = torch.stack([uu, vv_new], dim=-1)
        return uv

    def toMaskedNormal(
        self,
        background_color: List[float] = [255, 255, 255],
    ) -> torch.Tensor:
        """
        self.mask 不存在时等价于返回 self.normal；存在时按 normal 每个像素的 UV 在 mask 上
        最近邻采样得到遮罩，True 处保留 self.normal，False 处填 background_color。
        background_color: [R, G, B]，默认 0–255，内部会除以 255；输出 tensor 为 [0, 1]。
        返回: (H, W, 3) tensor
        """
        assert self.normal is not None
        if getattr(self, "mask", None) is None:
            return self.normal
        uv = self.toNormalUV()  # (H, W, 2)
        mask_t = self.sampleMaskAtUV(uv).unsqueeze(-1)  # (H, W, 1)
        bg = toTensor(background_color, self.dtype, self.device) / 255.0
        bg = bg.view(1, 1, 3) if bg.numel() == 3 else bg
        out = torch.where(mask_t, self.normal, bg.to(self.normal.dtype))
        return out

    def toMaskedNormalCV(
        self,
        background_color: List[float] = [255, 255, 255],
    ) -> np.ndarray:
        """toMaskedNormal 的 OpenCV BGR uint8 版本。无 mask 时等价于 self.normal_cv。"""
        return toNumpy(self.toMaskedNormal(background_color) * 255.0, np.uint8)[..., ::-1]
