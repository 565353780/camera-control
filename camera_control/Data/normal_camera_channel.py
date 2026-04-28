import numpy as np
import torch
from typing import List, Union

from camera_control.Data.base_normal_channel import BaseNormalChannel

_ATTR = 'normal_camera'


class NormalCameraChannel(BaseNormalChannel):
    def __init__(self) -> None:
        BaseNormalChannel.__init__(self, attr_name=_ATTR)
        return

    def update(self) -> bool:
        return BaseNormalChannel._update_normal(self, _ATTR)

    def loadNormalCamera(
        self,
        normal: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        return BaseNormalChannel._load_normal(self, _ATTR, normal)

    def loadNormalCameraFile(
        self,
        normal_file_path: str,
    ) -> bool:
        return BaseNormalChannel._load_normal_file(self, _ATTR, normal_file_path)

    def toNormalCameraUV(self) -> torch.Tensor:
        return BaseNormalChannel._to_normal_uv(self, _ATTR)

    def sampleNormalCameraAtUV(
        self,
        uv_grid: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        return BaseNormalChannel._sample_normal_at_uv(self, _ATTR, uv_grid)

    def toNormalCamera(
        self,
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        return BaseNormalChannel._to_normal(self, _ATTR, use_mask, mask_smaller_pixel_num)

    def toNormalCameraVis(
        self,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> torch.Tensor:
        return BaseNormalChannel._to_normal_vis(self, _ATTR, background_color, use_mask, mask_smaller_pixel_num)

    def toNormalCameraVisCV(
        self,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> np.ndarray:
        return BaseNormalChannel._to_normal_vis_cv(self, _ATTR, background_color, use_mask, mask_smaller_pixel_num)
