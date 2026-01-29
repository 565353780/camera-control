import torch
import numpy as np
from typing import List, Optional, Union, Tuple

from camera_control.Method.data import toNumpy, toTensor


class DepthChannel(object):
    """
    depth 与 camera.image 的宽高可能不一致，统一用归一化 UV [0,1] 作为位置信息，
    与 camera 的 UV 一致，便于与 image 等其它通道对齐。
    """
    def __init__(self) -> None:
        self.depth: torch.Tensor = None
        self.conf: torch.Tensor = None
        self.valid_depth_mask: torch.Tensor = None
        self.ccm: torch.Tensor = None
        # depth map 的宽高（可能与 camera 的 width/height 即 image 尺寸不同）
        self.depth_width: int = 0
        self.depth_height: int = 0
        return

    def update(self) -> bool:
        if self.depth is not None:
            self.depth = self.depth.to(dtype=self.dtype, device=self.device)
        if self.conf is not None:
            self.conf = self.conf.to(dtype=self.dtype, device=self.device)
        if self.valid_depth_mask is not None:
            self.valid_depth_mask = self.valid_depth_mask.to(dtype=self.dtype, device=self.device)
        if self.ccm is not None:
            self.ccm = self.ccm.to(dtype=self.dtype, device=self.device)
        return True

    def toDepthUV(self) -> torch.Tensor:
        """
        生成 self.depth 每个像素的归一化 UV 坐标 [0,1]，与 camera 的 UV 约定一致。
        不依赖 camera 的 width/height，仅用 depth 的宽高生成网格。
        返回: (depth_height, depth_width, 2)
        """
        w = max(self.depth_width, 1)
        h = max(self.depth_height, 1)
        u = torch.arange(w, dtype=self.dtype, device=self.device) / max(w - 1, 1)
        v = torch.arange(h, dtype=self.dtype, device=self.device) / max(h - 1, 1)
        uu, vv = torch.meshgrid(u, v, indexing='xy')  # [depth_height, depth_width]
        vv_new = 1.0 - vv  # 左下角为 (0,0)，v 向上增大，与 camera UV 一致
        uv = torch.stack([uu, vv_new], dim=-1)  # (depth_height, depth_width, 2)
        return uv

    def loadDepth(
        self,
        depth: Union[torch.Tensor, np.ndarray, list],
        conf: Union[torch.Tensor, np.ndarray, list, None]=None,
    ) -> bool:
        depth = toTensor(depth, self.dtype, self.device)
        if depth.ndim == 1:
            raise ValueError("loadDepth: depth 至少需要 2 维 (H, W)")
        h, w = int(depth.shape[0]), int(depth.shape[1])
        depth = depth.reshape(h, w)
        self.depth_height = h
        self.depth_width = w

        if conf is None:
            conf = torch.ones_like(depth)
        else:
            conf = toTensor(conf, self.dtype, self.device).reshape(h, w)

        # 存储 depth map
        self.depth = depth
        self.conf = conf

        # 记录有效像素位置
        self.valid_depth_mask = (depth > 1e-5) & (depth < 1e5)

        # 用 depth 每个像素的 UV 作为统一位置信息反投影，与 camera.image 的 UV 一致
        uv = self.toDepthUV()  # (depth_height, depth_width, 2)
        self.ccm = self.projectUV2Points(uv, depth)
        return True

    def queryPixelPoints(
        self,
        query_pixel: Union[torch.Tensor, np.ndarray, list],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        给定 depth 像素坐标，查询 self.ccm 得到空间点坐标和有效 mask。
        query_pixel: [..., 2] 最后一维为 (u, v)，即 depth map 上的像素坐标。
        返回: (points: [..., 3], confs: [...], valid_mask: [...])
        """
        query_pixel_tensor = toTensor(query_pixel, torch.int64, self.device)
        orig_shape = query_pixel_tensor.shape[:-1]
        query_pixel_flat = query_pixel_tensor.reshape(-1, 2)  # (N, 2)

        # 在 depth 像素范围内裁剪
        u = query_pixel_flat[:, 0].clamp(0, self.depth_width - 1)
        v = query_pixel_flat[:, 1].clamp(0, self.depth_height - 1)

        points = self.ccm[v, u]  # (N, 3)
        confs = self.conf[v, u]
        valid_mask = self.valid_depth_mask[v, u]  # (N,)

        points = points.reshape(*orig_shape, 3)
        confs = confs.reshape(orig_shape)
        valid_mask = valid_mask.reshape(orig_shape)

        return points, confs, valid_mask

    def queryUVPoints(
        self,
        query_uv: Union[torch.Tensor, np.ndarray, list],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        给定归一化 UV [0,1]（与 camera 的 UV 一致），在 depth 上找最近像素取 depth，
        再用 query_uv 与取到的 depth 反投影得到 3D 点。
        query_uv: [..., 2] 最后一维为归一化 [u, v]，范围 [0, 1]。
        返回: (points: [..., 3], confs: [...], valid_mask: [...])
        """
        # 转换为张量，保持原 shape
        query_uv_tensor = toTensor(query_uv, self.dtype, self.device)
        orig_shape = query_uv_tensor.shape[:-1]
        query_uv_flat = query_uv_tensor.reshape(-1, 2)  # (N, 2)

        # 在 depth 网格上由归一化 UV 得到最近像素（UV 与 camera 一致，v=0 左下）
        u_pixel = query_uv_flat[:, 0] * (self.depth_width - 1) if self.depth_width > 1 else torch.zeros_like(query_uv_flat[:, 0])
        v_pixel = (1.0 - query_uv_flat[:, 1]) * (self.depth_height - 1) if self.depth_height > 1 else torch.zeros_like(query_uv_flat[:, 1])  # UV v 向上，grid 行向下

        u_nearest = u_pixel.round().long().clamp(0, self.depth_width - 1)
        v_nearest = v_pixel.round().long().clamp(0, self.depth_height - 1)

        # 从depth map获取最近像素的depth值
        depth_values = self.depth[v_nearest, u_nearest]  # (N,)
        confs = self.conf[v_nearest, u_nearest]  # (N,)
        valid_mask = self.valid_depth_mask[v_nearest, u_nearest]  # (N,)

        # 使用准确的query_uv值和获取的depth值反投影得到3D点
        # query_uv_flat 是归一化的UV坐标，shape (N, 2)
        # depth_values 是对应的depth值，shape (N,)
        points = self.projectUV2Points(query_uv_flat, depth_values)  # (N, 3)

        # 恢复原 shape
        points = points.reshape(*orig_shape, 3)
        confs = confs.reshape(orig_shape)
        valid_mask = valid_mask.reshape(orig_shape)

        return points, confs, valid_mask

    def toDepthVis(
        self,
        depth_min: Optional[float]=None,
        depth_max: Optional[float]=None,
    ) -> torch.Tensor:
        """
        将self.depth转换为可视化的RGB格式tensor图像

        Returns:
            depth_vis: [H, W, 3] RGB格式的深度可视化图像，值在0-1范围内
        """
        assert self.depth is not None

        # 获取有效深度值
        mask = self.valid_depth_mask
        valid_depth = self.depth[mask]

        # 归一化深度值
        if valid_depth.numel() > 0:
            if depth_min is None:
                depth_min = valid_depth.min()
            if depth_max is None:
                depth_max = valid_depth.max()

            depth_normalized = (self.depth - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            depth_normalized = torch.zeros_like(self.depth)

        # 将无效像素的归一化深度设置为0
        depth_normalized = torch.where(mask, depth_normalized, torch.zeros_like(depth_normalized))

        # 转换为RGB格式（灰度图，三个通道相同）
        depth_vis = torch.stack([depth_normalized] * 3, dim=-1)  # [H, W, 3]

        return depth_vis

    def toDepthVisCV(
        self,
        depth_min: Optional[float]=None,
        depth_max: Optional[float]=None,
    ) -> np.ndarray:
        return toNumpy(self.toDepthVis(depth_min, depth_max) * 255.0, np.uint8)[..., ::-1]

    def toMaskedDepth(
        self,
    ) -> torch.Tensor:
        """
        self.mask 不存在时等价于 toDepthVis；存在时按 depth 每个像素的 UV 在 mask 上
        最近邻采样得到遮罩，True 处保留深度可视化，False 处填 background_color。
        同时与 valid_depth_mask 取与：无效深度处仍为背景。
        background_color: [R, G, B]，默认 0–255，内部会除以 255；输出 tensor 为 [0, 1]。
        返回: (depth_height, depth_width, 3) tensor
        """
        if getattr(self, "mask", None) is None:
            mask_t = self.valid_depth_mask
        else:
            uv = self.toDepthUV()  # (H, W, 2)
            mask_t = self.sampleMaskAtUV(uv) & self.valid_depth_mask  # (H, W)
        mask_t = mask_t.unsqueeze(-1)  # (H, W, 1)
        bg = torch.zeros([1, 1, 3], dtype=self.dtype, device=self.device)
        out = torch.where(mask_t, self.depth, bg)
        return out

    def toMaskedDepthVis(
        self,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        self.mask 不存在时等价于 toDepthVis；存在时按 depth 每个像素的 UV 在 mask 上
        最近邻采样得到遮罩，True 处保留深度可视化，False 处填 background_color。
        同时与 valid_depth_mask 取与：无效深度处仍为背景。
        background_color: [R, G, B]，默认 0–255，内部会除以 255；输出 tensor 为 [0, 1]。
        返回: (depth_height, depth_width, 3) tensor
        """
        depth_vis = self.toDepthVis(depth_min, depth_max)
        if getattr(self, "mask", None) is None:
            mask_t = self.valid_depth_mask
        else:
            uv = self.toDepthUV()  # (H, W, 2)
            mask_t = self.sampleMaskAtUV(uv) & self.valid_depth_mask  # (H, W)
        mask_t = mask_t.unsqueeze(-1)  # (H, W, 1)
        bg = torch.zeros([1, 1, 3], dtype=self.dtype, device=self.device)
        out = torch.where(mask_t, depth_vis, bg)
        return out

    def toMaskedDepthVisCV(
        self,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
    ) -> np.ndarray:
        """toMaskedDepth 的 OpenCV BGR uint8 版本。无 mask 时等价于 toDepthVisCV。"""
        return toNumpy(
            self.toMaskedDepthVis(depth_min, depth_max) * 255.0,
            np.uint8,
        )[..., ::-1]

    def toMaskedPoints(
        self,
        conf_thresh: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        先从 valid_depth 中按 conf 分位数 conf_thresh 得到置信度阈值并筛选；
        再判断 mask 是否存在，存在则按 depth 的 UV 在 mask 上最近邻采样，进一步筛掉不在 mask 内的点。
        返回: (points: (N, 3) tensor, confs: (N,) tensor)
        conf_thresh=None 表示不做置信度筛选；conf_thresh=0.8 表示保留 conf 处于 80% 分位及以上的点。
        """
        assert self.ccm is not None
        mask_t = self.valid_depth_mask
        if conf_thresh is not None and self.conf is not None:
            conf_flat = self.conf[mask_t]
            if conf_flat.numel() == 0:
                return (
                    torch.empty(0, 3, dtype=self.ccm.dtype, device=self.device),
                    torch.empty(0, dtype=self.conf.dtype, device=self.device),
                )
            thresh_value = torch.quantile(conf_flat.float(), conf_thresh, interpolation="lower")
            mask_t = mask_t & (self.conf >= thresh_value)
        if getattr(self, "mask", None) is not None:
            uv = self.toDepthUV()
            mask_t = mask_t & self.sampleMaskAtUV(uv)
        points = self.ccm[mask_t]  # (N, 3)
        confs = (
            self.conf[mask_t]
            if self.conf is not None
            else torch.ones(points.shape[0], dtype=self.ccm.dtype, device=self.device)
        )  # (N,)
        return points, confs
