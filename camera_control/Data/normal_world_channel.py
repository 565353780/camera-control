import numpy as np
import torch
from typing import List, Union

from camera_control.Data.base_normal_channel import BaseNormalChannel

_ATTR = 'normal_world'


class NormalWorldChannel(BaseNormalChannel):
    def __init__(self) -> None:
        BaseNormalChannel.__init__(self, attr_name=_ATTR)
        return

    def update(self) -> bool:
        return BaseNormalChannel._update_normal(self, _ATTR)

    def loadNormalWorld(
        self,
        normal: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        return BaseNormalChannel._load_normal(self, _ATTR, normal)

    def loadNormalWorldFile(
        self,
        normal_file_path: str,
    ) -> bool:
        return BaseNormalChannel._load_normal_file(self, _ATTR, normal_file_path)

    def toNormalWorldUV(self) -> torch.Tensor:
        return BaseNormalChannel._to_normal_uv(self, _ATTR)

    def toNormalWorld(
        self,
        use_mask: bool = True,
    ) -> torch.Tensor:
        return BaseNormalChannel._to_normal(self, _ATTR, use_mask)

    def toNormalWorldVis(
        self,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
    ) -> torch.Tensor:
        return BaseNormalChannel._to_normal_vis(self, _ATTR, background_color, use_mask)

    def toNormalWorldVisCV(
        self,
        background_color: List[float] = [255, 255, 255],
        use_mask: bool = True,
    ) -> np.ndarray:
        return BaseNormalChannel._to_normal_vis_cv(self, _ATTR, background_color, use_mask)
