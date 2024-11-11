import torch
import torch.nn as nn 
from src.methods.bayes.base.net_distribution import BaseNetDistribution
from src.methods.bayes.base.distribution import ParamDist
import torch.distributions as td
from typing import TypeVar, Generic
from torch.distributions.utils import _standard_normal, broadcast_all
from numbers import Number, Real
from torch.types import _size

class VarBayesModuleNetDistribution(BaseNetDistribution ):
    def __init__(self, base_module: nn.Module, weight_distribution: dict[str, ParamDist]) -> None:
        super().__init__(base_module=base_module, weight_distribution=weight_distribution)
        
