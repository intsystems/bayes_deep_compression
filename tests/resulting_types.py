"""
This file defines the types that should be returned by Bayesian models.
They are used in initializing models and in determining losses.
These classes are necessary so that you can train models with different
 losses and not sacrifice the separability of the model from the loss.
"""

# from typing import TypeVar, Generic
from dataclasses import dataclass, fields

import torch

@dataclass
class BaseModelOutput:
    """
    Base class for bayessian model output

    Args:   
        :output: result of model applying to inputs
    """
    outputs: torch.Tensor

    def __post_init__(self):
        _fields = fields(self)
        self.names = [f.name for f in _fields]

    def get_names(self):
        """
        For fast get list of fields names
        """
        return self.names
    
    def __getitem__(self, name):
        return self.__getattribute__(name)

@dataclass
class LogNormalLossModelOutput(BaseModelOutput):
    """
    Class of return of model for training with LogNormal Divergence

    Args:
        :prior_likelihood: model parameters likelihoods for prior distribution
        :posterior_likelihood: model parameters likelihoods for posterior distribution
    """

    prior_likelihood: torch.Tensor
    posterior_likelihood: torch.Tensor

@dataclass
class KLLossModelOutput(BaseModelOutput):
    """
    Class of return of model for training with KL Divergence
    Args:
        :kullback_leubler: KulbackLeibler Divergence of model parameters
    """
    kullback_leubler: torch.Tensor