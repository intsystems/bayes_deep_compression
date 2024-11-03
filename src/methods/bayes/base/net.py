from typing import Generic, TypeVar
import torch.nn as nn
from abc import abstractmethod
import torch.nn
from src.methods.bayes.base.output import BaseOutputModel

OutputT = TypeVar("OutputT", bound=BaseOutputModel)


class BayesLinearLayer(Generic[OutputT], nn.Linear):
    def __init__(self, in_features, out_features, device=None):
        super().__init__(in_features, out_features, False, device, torch.float32)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> OutputT: ...


LinearT = TypeVar("LinearT", bound=BayesLinearLayer)

class 


class MLPBayesModel(torch.nn.Module, Generic[LinearT]):
    def __init__(self, bayesian_layer_list: list[LinearT]):
        self.bayesian_layer_list = bayesian_layer_list

    @abstractmethod
    def forward(self, x: torch.Tensor[torch.float32]): ...
   
