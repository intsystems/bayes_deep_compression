from typing import Generic, TypeVar

import torch.nn as nn

from src.methods.bayes.base.trainer import ModelT


class BaseNetDistribution(Generic[ModelT]):
    def __init__(self): ...

    def sample(self, *args) -> nn.Module: ...


DistributionT = TypeVar("DistributionT", bound=BaseNetDistribution)


class BaseNetEnsemble(Generic[DistributionT]):
    def __init__(self, distribution: BaseNetDistribution):
        self.distribution = distribution

    def predict(self): ...
