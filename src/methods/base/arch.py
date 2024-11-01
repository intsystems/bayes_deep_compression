from typing import Generic, TypeVar

import torch.nn

from src.methods.base.output import BaseOutputModel

T = TypeVar("T", bound=BaseOutputModel)


class BaseBayesModel(torch.nn.Module, Generic[T]):
    def forward(self, x: torch.Tensor[torch.float32]) -> T: ...
