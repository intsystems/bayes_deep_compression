from typing import Generic, TypeVar

import torch

from src.methods.bayes.base.output import BaseOutputModel


class BaseLoss(torch.nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor: ...
