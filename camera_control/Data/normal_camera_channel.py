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

    @property
    def normal_camera_cv(self) -> np.ndarray:
        return BaseNormalChannel._get_normal_cv(self, _ATTR)

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

    def toMaskedNormalCamera(
        self,
        background_color: List[float] = [255, 255, 255],
    ) -> torch.Tensor:
        return BaseNormalChannel._to_masked_normal(self, _ATTR, background_color)

    def toMaskedNormalCameraCV(
        self,
        background_color: List[float] = [255, 255, 255],
    ) -> np.ndarray:
        return BaseNormalChannel._to_masked_normal_cv(self, _ATTR, background_color)
