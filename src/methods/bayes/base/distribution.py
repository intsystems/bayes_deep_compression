import torch.nn as nn
from typing import Generic, TypeVar
from src.methods.bayes.base.trainer import ModelT


class BaseNetDistribution(Generic[ModelT]):
    def __init__(self): ...

    def sample(self, *args) -> nn.Module: ..
    
class BaseNetEnsemble:
    def __init__(self, distribution: BaseNetDistribution):
        self.distribution = distribution
