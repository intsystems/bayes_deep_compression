from typing import Dict

import numpy as np
import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented")

    def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Not implemented")

    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError("Not implemented")
