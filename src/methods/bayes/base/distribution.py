from typing import Generic, TypeVar
import torch
import torch.nn as nn

from src.methods.bayes.base.trainer import ModelT


class MLP(nn.Module):
    def __init__(self, layer_list: list[nn.Linear], activation: list[nn.Module]):
        self.net = nn.ModuleList(
            nn.Sequential([layer, activation]) for layer in layer_list
        )

    def forward(self,x) -> torch.Tensor:
        return self.net(x)
    
class BaseLayerDistribution(Generic[ModelT]):
    ...


class BaseNetDistribution(Generic[ModelT]):
    def __init__(self): ...

    @property
    def stats(self): ...

    def sample(self, *args) -> MLP: ...


DistributionT = TypeVar("DistributionT", bound=BaseNetDistribution)


class BaseNetEnsemble(Generic[DistributionT]):
    def __init__(self, distribution: BaseNetDistribution):
        self.distribution = distribution

    def predict(self): ...
