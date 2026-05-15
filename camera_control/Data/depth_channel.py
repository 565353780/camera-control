import os
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
            self.valid_depth_mask = self.valid_depth_mask.to(dtype=torch.bool, device=self.device)
        return True

    def toDepthUV(self) -> torch.Tensor:
        """
        生成 self.depth 每个像素中心的归一化 UV 坐标 [0,1]，与 camera 的 UV 约定一致。
        不依赖 camera 的 width/height，仅用 depth 的宽高生成网格。
        返回: (depth_height, depth_width, 2)
        """
        w = max(self.depth_width, 1)
        h = max(self.depth_height, 1)
        u = (torch.arange(w, dtype=self.dtype, device=self.device) + 0.5) / w
        v = (torch.arange(h, dtype=self.dtype, device=self.device) + 0.5) / h
        uu, vv = torch.meshgrid(u, v, indexing='xy')  # [depth_height, depth_width]
        vv_new = 1.0 - vv  # 左下角为 (0,0)，v 向上增大，与 camera UV 一致
        uv = torch.stack([uu, vv_new], dim=-1)  # (depth_height, depth_width, 2)
        return uv

    def _computeCCM(self) -> torch.Tensor:
        """从当前 depth 实时计算 camera coordinate map。"""
        uv = self.toDepthUV()
        return self.projectUV2Points(uv, self.depth)

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

        return True

    def loadDepthFile(
        self,
        depth_file_path: str,
    ) -> bool:
        if not os.path.exists(depth_file_path):
            print('[ERROR][DepthChannel::loadDepthFile]')
            print('\t depth file not exist!')
            print('\t depth_file_path:', depth_file_path)
            return False

        data = np.load(depth_file_path)

        if data.ndim > 2:
            if data.shape[2] > 2:
                print('[ERROR][DepthChannel::loadDepthFile]')
                print('\t depth file shape valid!')
                print('\t data.shape:', data.shape)
                return False

            if data.shape[2] == 2:
                depth = data[..., 0]
                conf = data[..., 1]

                return self.loadDepth(depth, conf)

            data = data.unsqueeze(-1)

        return self.loadDepth(data)

    def queryPixelPoints(
        self,
        query_pixel: Union[torch.Tensor, np.ndarray, list],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        给定 depth 像素坐标，实时计算 ccm 并查询空间点坐标和有效 mask。
        query_pixel: [..., 2] 最后一维为 (u, v)，即 depth map 上的像素坐标。
        返回: (points: [..., 3], confs: [...], valid_mask: [...])
        """
        query_pixel_tensor = toTensor(query_pixel, torch.int64, self.device)
        orig_shape = query_pixel_tensor.shape[:-1]
        query_pixel_flat = query_pixel_tensor.reshape(-1, 2)  # (N, 2)

        # 在 depth 像素范围内裁剪
        u = query_pixel_flat[:, 0].clamp(0, self.depth_width - 1)
        v = query_pixel_flat[:, 1].clamp(0, self.depth_height - 1)

        ccm = self._computeCCM()
        points = ccm[v, u]  # (N, 3)
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

        # 像素 i 占据 UV 区间 [i/N, (i+1)/N)，反向: pixel = floor(uv * N)
        u_nearest = (query_uv_flat[:, 0] * self.depth_width).long().clamp(0, self.depth_width - 1)
        v_nearest = ((1.0 - query_uv_flat[:, 1]) * self.depth_height).long().clamp(0, self.depth_height - 1)

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

    @property
    def depth_with_conf(self) -> torch.Tensor:
        return torch.stack([self.depth, self.conf], dim=-1)  # HxW -> HxWx2

    def toDepthMask(
        self,
        conf_thresh: Optional[float] = None,
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        if not use_mask or getattr(self, "mask", None) is None:
            mask_t = self.valid_depth_mask
        else:
            uv = self.toDepthUV()  # (H, W, 2)
            mask_t = self.sampleMaskAtUV(uv, mask_smaller_pixel_num) & self.valid_depth_mask  # (H, W)

        if conf_thresh is not None and self.conf is not None:
            if self.conf.numel() > 0:
                thresh_value = torch.quantile(self.conf.float(), conf_thresh, interpolation="lower")
                mask_t = mask_t & (self.conf >= thresh_value)
        return mask_t

    def toDepth(
        self,
        conf_thresh: Optional[float] = None,
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        depth_mask = self.toDepthMask(
            conf_thresh=conf_thresh,
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        )
        # 仅在 depth_mask 为 True 的位置保留原始 depth，其他位置设为 0
        valid_depth = torch.where(depth_mask, self.depth, torch.zeros_like(self.depth))
        return valid_depth

    def toCCM(
        self,
        conf_thresh: Optional[float] = None,
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        """
        从当前 depth 实时计算 ccm，返回 mask 区域内的 ccm，其他地方置零。

        Returns:
            valid_ccm: [H, W, 3]，mask 为 True 的地方保留 ccm，其余为 0
        """
        ccm = self._computeCCM()
        depth_mask = self.toDepthMask(
            conf_thresh=conf_thresh,
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        )  # (H, W)

        mask_expanded = depth_mask.unsqueeze(-1).expand_as(ccm)  # (H, W, 3)
        valid_ccm = torch.where(mask_expanded, ccm, torch.zeros_like(ccm))
        return valid_ccm

    def toDepthVis(
        self,
        depth_min: Optional[float]=None,
        depth_max: Optional[float]=None,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        """
        将self.depth转换为可视化的RGB格式tensor图像。
        mask 外像素填 background_color，默认白色。
        background_color: [R, G, B]，0–255，内部会除以 255。

        Returns:
            depth_vis: [H, W, 3] RGB格式的深度可视化图像，值在0-1范围内
        """
        assert self.depth is not None

        mask = self.toDepthMask(use_mask=use_mask, mask_smaller_pixel_num=mask_smaller_pixel_num)
        valid_depth = self.depth[mask]

        if valid_depth.numel() > 0:
            if depth_min is None:
                depth_min = valid_depth.min()
            if depth_max is None:
                depth_max = valid_depth.max()

            depth_normalized = (self.depth - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            depth_normalized = torch.zeros_like(self.depth)

        depth_normalized = depth_normalized.clamp(0.0, 1.0)

        depth_rgb = torch.stack([depth_normalized] * 3, dim=-1)  # [H, W, 3]

        bg = toTensor(background_color, self.dtype, self.device) / 255.0
        bg = bg.view(1, 1, 3) if bg.numel() == 3 else bg
        mask_expanded = mask.unsqueeze(-1).expand_as(depth_rgb)
        depth_vis = torch.where(mask_expanded, depth_rgb, bg.to(depth_rgb.dtype).expand_as(depth_rgb))
        return depth_vis

    def toCCMVis(
        self,
        conf_thresh: Optional[float] = None,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        """
        将 CCM 转换为可视化的 RGB 格式 tensor 图像。
        按公式 ccm_color = ccm + 0.5 将 CCM 坐标映射到颜色空间，
        并 clamp 到 [0, 1]，mask 外像素填 background_color，默认白色。
        background_color: [R, G, B]，0–255，内部会除以 255。

        Returns:
            ccm_vis: [H, W, 3] RGB 格式的 CCM 可视化图像，值在 0-1 范围内
        """
        ccm = self._computeCCM()

        depth_mask = self.toDepthMask(
            conf_thresh=conf_thresh,
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        )  # (H, W)

        ccm_color = (ccm + 0.5).clamp(0.0, 1.0)

        bg = toTensor(background_color, self.dtype, self.device) / 255.0
        bg = bg.view(1, 1, 3) if bg.numel() == 3 else bg
        mask_expanded = depth_mask.unsqueeze(-1).expand_as(ccm_color)
        ccm_vis = torch.where(mask_expanded, ccm_color, bg.to(ccm_color.dtype).expand_as(ccm_color))
        return ccm_vis

    def toDepthVisCV(
        self,
        depth_min: Optional[float]=None,
        depth_max: Optional[float]=None,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> np.ndarray:
        return toNumpy(self.toDepthVis(
            depth_min=depth_min,
            depth_max=depth_max,
            background_color=background_color,
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        ) * 255.0, np.uint8)[..., ::-1]

    def toCCMVisCV(
        self,
        conf_thresh: Optional[float] = None,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> np.ndarray:
        return toNumpy(self.toCCMVis(
            conf_thresh=conf_thresh,
            background_color=background_color,
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        ) * 255.0, np.uint8)[..., ::-1]

    def toDepthPoints(
        self,
        conf_thresh: Optional[float] = None,
        use_mask: bool=True,
        mask_smaller_pixel_num: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        先从 valid_depth 中按 conf 分位数 conf_thresh 得到置信度阈值并筛选；
        再判断 mask 是否存在，存在则按 depth 的 UV 在 mask 上最近邻采样，进一步筛掉不在 mask 内的点。
        返回: (points: (N, 3) tensor, confs: (N,) tensor)
        conf_thresh=None 表示不做置信度筛选；conf_thresh=0.8 表示保留 conf 处于 80% 分位及以上的点。
        """
        ccm = self._computeCCM()
        mask_t = self.toDepthMask(conf_thresh, use_mask, mask_smaller_pixel_num)
        points = ccm[mask_t]  # (N, 3)
        confs = (
            self.conf[mask_t]
            if self.conf is not None
            else torch.ones(points.shape[0], dtype=self.dtype, device=self.device)
        )  # (N,)
        return points, confs
