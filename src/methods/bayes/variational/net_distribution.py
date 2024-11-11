import torch
import torch.nn as nn 
from src.methods.bayes.base.distribution import BaseNetDistribution
from src.methods.bayes.variational.net import VarBayesModuleNet
import torch.distributions as td
from torch.distributions.utils import _standard_normal, broadcast_all
from numbers import Number, Real
from torch.types import _size

class VarBayesModuleNetDistribution(BaseNetDistribution[VarBayesModuleNet]):
    ...