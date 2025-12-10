import torch
import numpy as np
from typing import Union

def toTensor(
    data: Union[torch.Tensor, np.ndarray, list],
    dtype=torch.float32,
    device: str = 'cpu',
) -> torch.Tensor:
    if isinstance(data, list):
        data = np.asarray(data)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.to(device, dtype=dtype)
    return data
