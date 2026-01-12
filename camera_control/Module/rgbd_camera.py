import torch
import numpy as np
from typing import Optional, Union, Tuple

from camera_control.Method.data import toNumpy, toTensor
from camera_control.Module.camera import Camera


class RGBDCamera(Camera):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 500,
        fy: float = 500,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        pos: Union[torch.Tensor, np.ndarray, list] = [0, 0, 0],
        look_at: Union[torch.Tensor, np.ndarray, list] = [1, 0, 0],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 0, 1],
        world2camera: Union[torch.Tensor, np.ndarray, list, None] = None,
        dtype=torch.float32,
        device: str = 'cpu',
    ) -> None:
        super().__init__(
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            pos,
            look_at,
            up,
            world2camera,
            dtype,
            device,
        )
        return

    @property
    def depth_vis(self) -> torch.Tensor:
        """
        将self.depth转换为可视化的RGB格式tensor图像

        Returns:
            depth_vis: [H, W, 3] RGB格式的深度可视化图像，值在0-1范围内
        """
        # 获取有效深度值
        mask = self.valid_depth_mask
        valid_depth = self.depth[mask]

        # 归一化深度值
        if valid_depth.numel() > 0:
            depth_min = valid_depth.min()
            depth_max = valid_depth.max()
            depth_normalized = (self.depth - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            depth_normalized = torch.zeros_like(self.depth)

        # 将无效像素的归一化深度设置为0
        depth_normalized = torch.where(mask, depth_normalized, torch.zeros_like(depth_normalized))

        # 转换为RGB格式（灰度图，三个通道相同）
        depth_vis = torch.stack([depth_normalized] * 3, dim=-1)  # [H, W, 3]

        return depth_vis

    @property
    def image_cv(self) -> np.ndarray:
        return toNumpy(self.image * 255.0, np.uint8)[..., ::-1]

    @property
    def depth_vis_cv(self) -> np.ndarray:
        return toNumpy(self.depth_vis * 255.0, np.uint8)[..., ::-1]

    def loadImage(
        self,
        image: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        self.image = toTensor(image, torch.uint8)
        return True

    def loadDepth(
        self,
        depth: Union[torch.Tensor, np.ndarray, list],
        conf: Union[torch.Tensor, np.ndarray, list, None]=None,
    ) -> bool:
        uv = self.toImageUV()
        depth = toTensor(depth, self.dtype, self.device).reshape(self.height, self.width)

        if conf is None:
            conf = torch.ones_like(depth)
        else:
            conf = toTensor(conf, self.dtype, self.device).reshape(self.height, self.width)

        # 存储depth map
        self.depth = depth
        self.conf = conf

        # 记录有效像素位置
        self.valid_depth_mask = (depth > 1e-5) & (depth < 1e5)

        # self.ccm 为 HxWx3 的空间点坐标
        self.ccm = self.projectUV2Points(uv, depth)
        return True

    def queryPixelPoints(
        self,
        query_pixel: Union[torch.Tensor, np.ndarray, list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定像素坐标，查询 self.ccm 得到空间点坐标和有效mask。
        query_pixel: [..., 2] (最后一维是[u, v])
        返回: (points: [..., 3], valid_mask: [...])
        """
        # 转换为张量，保持原 shape
        query_pixel_tensor = toTensor(query_pixel, torch.int64, self.device)
        orig_shape = query_pixel_tensor.shape[:-1]
        query_pixel_flat = query_pixel_tensor.reshape(-1, 2)  # (N, 2)

        # 获取 u, v 坐标（像素坐标），裁剪到有效范围
        u = query_pixel_flat[:, 0].clamp(0, self.width - 1)
        v = query_pixel_flat[:, 1].clamp(0, self.height - 1)

        # 取对应点
        points = self.ccm[v, u]  # (N, 3)
        valid_mask = self.valid_depth_mask[v, u]  # (N,)

        # 恢复原 shape
        points = points.reshape(*orig_shape, 3)
        valid_mask = valid_mask.reshape(orig_shape)

        return points, valid_mask

    def queryUVPoints(
        self,
        query_uv: Union[torch.Tensor, np.ndarray, list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定归一化UV坐标，找到最近的像素，使用该像素的depth和准确的query_uv值反投影得到3D点。
        query_uv: [..., 2] (最后一维是归一化的[u, v]，范围[0, 1])
        返回: (points: [..., 3], valid_mask: [...])
        """
        # 转换为张量，保持原 shape
        query_uv_tensor = toTensor(query_uv, self.dtype, self.device)
        orig_shape = query_uv_tensor.shape[:-1]
        query_uv_flat = query_uv_tensor.reshape(-1, 2)  # (N, 2)

        # 将归一化UV坐标转换为像素坐标（浮点数）
        u_pixel = query_uv_flat[:, 0] * self.width
        v_pixel = query_uv_flat[:, 1] * self.height

        # 找到最近的整数像素坐标
        u_nearest = u_pixel.round().long().clamp(0, self.width - 1)
        v_nearest = v_pixel.round().long().clamp(0, self.height - 1)

        # 从depth map获取最近像素的depth值
        depth_values = self.depth[v_nearest, u_nearest]  # (N,)
        valid_mask = self.valid_depth_mask[v_nearest, u_nearest]  # (N,)

        # 使用准确的query_uv值和获取的depth值反投影得到3D点
        # query_uv_flat 是归一化的UV坐标，shape (N, 2)
        # depth_values 是对应的depth值，shape (N,)
        points = self.projectUV2Points(query_uv_flat, depth_values)  # (N, 3)

        # 恢复原 shape
        points = points.reshape(*orig_shape, 3)
        valid_mask = valid_mask.reshape(orig_shape)

        return points, valid_mask

