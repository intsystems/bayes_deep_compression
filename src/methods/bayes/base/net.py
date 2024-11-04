from abc import abstractmethod
from typing import Generic, TypeVar

import torch.nn
import torch.nn as nn

from src.methods.bayes.base.output import BaseOutputModel

OutputT = TypeVar("OutputT", bound=BaseOutputModel)


class MLPBayesModel(torch.nn.Module, Generic[OutputT]):
    def __init__(self, layer_list: list[nn.Linear]):
        self.layer_list = layer_list

    @abstractmethod
    def forward(self, x: torch.Tensor[torch.float32]) -> list[OutputT]: ...
