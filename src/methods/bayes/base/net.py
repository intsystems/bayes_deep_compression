from typing import Generic, TypeVar

import torch.nn

from src.methods.bayes.base import BaseOutputModel

T = TypeVar("T", bound=BaseOutputModel)


class 

class BaseBayesModel(torch.nn.Module, Generic[T]):
    def forward(self, x: torch.Tensor[torch.float32]) -> T: ...
