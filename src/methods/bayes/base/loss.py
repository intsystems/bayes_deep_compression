from typing import Generic, TypeVar

import torch

from src.methods.bayes import BaseOutputModel

T = TypeVar("T", bound=BaseOutputModel)


class BaseLoss(torch.nn.Module, Generic[T]):
    def forward(self, model_output: T) -> torch.Tensor: ...
