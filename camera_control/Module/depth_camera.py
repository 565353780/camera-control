import torch
import numpy as np
from typing import Optional, Union, Tuple

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera


class DepthCamera(Camera):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 500,
        fy: float = 500,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        pos: Union[torch.Tensor, np.ndarray, list] = ...,
        look_at: Union[torch.Tensor, np.ndarray, list] = ...,
        up: Union[torch.Tensor, np.ndarray, list] = ...,
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

    def loadDepth(
        self,
        depth: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        uv = self.toImageUV()
        depth = toTensor(depth, self.dtype, self.device).reshape(self.height, self.width)

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
