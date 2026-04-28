import torch
import numpy as np
from typing import List, Union

from camera_control.Method.io import loadImage
from camera_control.Method.data import toNumpy, toTensor


class BaseNormalChannel(object):
    """
    法线通道基类。子类通过 attr_name 指定存储属性名（如 'normal_world' 或 'normal_camera'），
    所有读写操作均通过 attr_name 来定位目标属性，避免多重继承下的名称冲突。
    """

    def __init__(self, attr_name: str) -> None:
        setattr(self, attr_name, None)
        return

    @staticmethod
    def _update_normal(obj, attr_name: str) -> bool:
        val = getattr(obj, attr_name)
        if val is not None:
            setattr(obj, attr_name, val.to(dtype=obj.dtype, device=obj.device))
        return True

    @staticmethod
    def _load_normal(
        obj,
        attr_name: str,
        normal: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        setattr(obj, attr_name, toTensor(normal, obj.dtype, obj.device))
        return True

    @staticmethod
    def _load_normal_file(
        obj,
        attr_name: str,
        normal_file_path: str,
    ) -> bool:
        normal = loadImage(normal_file_path)

        if normal is None:
            print(f'[ERROR][{type(obj).__name__}::loadNormalFile]')
            print('\t loadImage failed!')
            return False

        normal = normal[..., ::-1].astype(np.float32) / 255.0

        return BaseNormalChannel._load_normal(obj, attr_name, normal)

    @staticmethod
    def _to_normal_uv(obj, attr_name: str) -> torch.Tensor:
        """
        生成 normal 每个像素中心的归一化 UV [0,1]，与 camera 的 UV 约定一致。
        返回: (H, W, 2) tensor，H/W 来自 normal.shape[:2]。
        """
        val = getattr(obj, attr_name)
        assert val is not None
        h, w = val.shape[0], val.shape[1]
        u = (torch.arange(w, dtype=obj.dtype, device=obj.device) + 0.5) / max(w, 1)
        v = (torch.arange(h, dtype=obj.dtype, device=obj.device) + 0.5) / max(h, 1)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        vv_new = 1.0 - vv
        uv = torch.stack([uu, vv_new], dim=-1)
        return uv

    @staticmethod
    def _sample_normal_at_uv(
        obj,
        attr_name: str,
        uv_grid: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        按归一化 UV 在 normal map 上做最近邻采样，与 RGBChannel.sampleRGBAtUV 同约定，
        便于 depth / mask 等其它模态按各自 UV 对齐 normal，节省重复网格构造。

        uv_grid: (..., 2) 归一化 UV [0,1]，u 向右、v 向上（v=0 为左下）。
        normal map 行 0 对应 v=1（上），行 Mh-1 对应 v=0（下）。
        返回: 与 uv_grid 前若干维同形状的 (..., 3) tensor。
        """
        val = getattr(obj, attr_name)
        assert val is not None
        uv_grid = toTensor(uv_grid, torch.float32, obj.device)
        Mh, Mw = val.shape[0], val.shape[1]
        u = uv_grid[..., 0]
        v = uv_grid[..., 1]
        idx_w = (u * Mw - 0.5).round().long().clamp(0, Mw - 1)
        idx_h = ((1.0 - v) * Mh - 0.5).round().long().clamp(0, Mh - 1)
        return val[idx_h, idx_w]

    @staticmethod
    def _to_normal(
        obj,
        attr_name: str,
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        """
        mask 不存在或 use_mask=False 时返回原始 normal；
        存在时按 normal 每个像素的 UV 在 mask 上最近邻采样得到遮罩，
        True 处保留 normal，False 处填零向量。
        返回: (H, W, 3) tensor
        """
        val = getattr(obj, attr_name)
        assert val is not None
        if not use_mask or getattr(obj, "mask", None) is None:
            return val
        uv = BaseNormalChannel._to_normal_uv(obj, attr_name)
        mask_t = obj.sampleMaskAtUV(uv, mask_smaller_pixel_num).unsqueeze(-1)
        out = torch.where(mask_t, val, torch.zeros_like(val))
        return out

    @staticmethod
    def _to_normal_vis(
        obj,
        attr_name: str,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        """
        可视化 normal：先 * 0.5 + 0.5 映射到 [0, 1]，再用 background_color 填充 mask 外区域。
        background_color: [R, G, B]，默认 0–255，内部会除以 255。
        返回: (H, W, 3) tensor
        """
        val = getattr(obj, attr_name)
        assert val is not None
        vis = val * 0.5 + 0.5
        if not use_mask or getattr(obj, "mask", None) is None:
            return vis
        uv = BaseNormalChannel._to_normal_uv(obj, attr_name)
        mask_t = obj.sampleMaskAtUV(uv, mask_smaller_pixel_num).unsqueeze(-1)
        bg = toTensor(background_color, obj.dtype, obj.device) / 255.0
        bg = bg.view(1, 1, 3) if bg.numel() == 3 else bg
        return torch.where(mask_t, vis, bg.to(vis.dtype))

    @staticmethod
    def _to_normal_vis_cv(
        obj,
        attr_name: str,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> np.ndarray:
        return toNumpy(
            BaseNormalChannel._to_normal_vis(obj, attr_name, background_color, use_mask, mask_smaller_pixel_num) * 255.0,
            np.uint8,
        )[..., ::-1]
