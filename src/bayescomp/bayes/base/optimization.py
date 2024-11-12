from typing import Generic, TypeVar

import torch

from bayescomp.bayes.base.output import BaseOutputModel


class BaseLoss(torch.nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor: ...
