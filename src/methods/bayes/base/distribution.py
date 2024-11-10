from typing import Generic, TypeVar
import torch
import torch.nn as nn

from src.methods.bayes.base.net import ModelT


class MLP(nn.Module):
    def __init__(self, layer_list: list[nn.Linear], activation: list[nn.Module]):
        self.net = nn.ModuleList(
            nn.Sequential([layer, activation]) for layer in layer_list
        )

    def forward(self,x) -> torch.Tensor:
        return self.net(x)
    
class BaseLayerDistribution(nn.Module):
    ...


class BaseNetDistribution(Generic[ModelT]):
    def __init__(self): ...

    @property
    def stats(self): ...

    def sample(self, *args) -> nn.Module: ...


class BaseNetEnsemble(Generic[ModelT]):
    def __init__(self, distribution: BaseNetDistribution) -> None:
        self.distribution = distribution

    def predict(self): ...
