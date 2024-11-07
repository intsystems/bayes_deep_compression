from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, List
from abc import abstractmethod
from pydantic.types import PositiveInt
from collections import defaultdict

import torch
import torch.nn as nn

from bayess.nn.reparametrizations import BasePriorPosteriorReparam

# @dataclass
# class ExtraCalculations:
#     kl_divergence: bool = False
#     prior_likelihood: bool = False
#     posterior_likelihood: bool = False

#     data: Dict = {}

#     def __call__(self, parameter: torch.Tensor, reparam: BasePriorPosteriorReparam):
#         if self.kl_divergence:
#             self.data['kl_divergence'] = reparam.get_kl
#             this.prior_likelihood = None
#         if self.posterior_likelihood:
#             this.posterior_likelihood = None
#     def get_handler(self ):
#         class ExtraHandler:
#             def __init__(this) -> None:
#                 if self.kl_divergence:
#                     this.kl_divergence = None
#                 if self.prior_likelihood:
#                     this.prior_likelihood = None
#                 if self.posterior_likelihood:
#                     this.posterior_likelihood = None
#             def __call__(this, parameter: torch.Tensor, reparam: BasePriorPosteriorReparam):

                
                

# OPTIMIZE_KL = ExtraCalculations(kl_divergence=True)
# OPTIMIZE_LIKELIHOODS = ExtraCalculations(prior_likelihood=True, \
#                                          posterior_likelihoods=True)

@dataclass
class likelihood_handler:
    prior_likelihood: float
    posterior_likelihood: float
    
    def __add__(self, other):
        self.prior_likelihood += other.prior_likelihood
        self.posterior_likelihood += other.posterior_likelihood
        return self

type ReparamFabric = Callable[[str, nn.Module, tuple], BasePriorPosteriorReparam]


class BaseModel(nn.Module):
    """
    base class for bayessian models
    """
    def __init__(self, reparam_fabric: ReparamFabric,) -> None:
        super(BaseModel, self).__init__()
        self.reparam_fabric = reparam_fabric
    
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward method, should get as input input tensor and number of samples
        """
        raise NotImplementedError("implement me")


class BayessLinear(BaseModel):
    def __init__(self, n_in, n_out, reparam_fabric: ReparamFabric):
        
        super(BayessLinear, self).__init__(reparam_fabric)

        self.W = reparam_fabric((n_in, n_out))
        self.b = reparam_fabric((1, n_out))

    def forward(self, x):
        W = self.W.rsample()
        b = self.b.rsample()

        rez = x @ W + b

        likelihoods = likelihood_handler(
                prior_likelihood=self.W.prior_likelihood(W).sum() + self.b.prior_likelihood(b).sum(), \
                posterior_likelihood=self.W.posterior_likelihood(W).sum() + self.b.posterior_likelihood(b).sum()\
                )

        return rez, likelihoods
    
class MLP(nn.Module):
    def __init__(self, dimentions: List[PositiveInt],) -> None:
        super().__init__()

        submodules = []
        self.dimentions = dimentions
        for i in range(len(dimentions) - 2):
            submodules.append(nn.Linear(dimentions[i],dimentions[i+1]))
            submodules.append(nn.ReLU(inplace = True))
        submodules.append(nn.Linear(dimentions[-2],dimentions[-1]))
        self.net = nn.ParameterList(submodules)
    def forward(self, x):
        for elem in self.net:
            x = elem(x)
        return x

     
class BayessMLP(BaseModel):
    def __init__(self,  dimentions: List[PositiveInt], reparam_fabric: ReparamFabric) -> None:
        super().__init__(reparam_fabric)

        submodules = []
        self.dimentions = dimentions
        for i in range(len(dimentions) - 2):
            submodules.append(BayessLinear(dimentions[i], dimentions[i+1], reparam_fabric))
            submodules.append(nn.ReLU(inplace=True))
        submodules.append(BayessLinear(dimentions[-2], dimentions[-1], reparam_fabric))

        self.net = nn.ParameterList(submodules)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, likelihood_handler]:            
        likelihoods = likelihood_handler(0., 0.)

        for elem in self.net:
            output = elem(input)
            if isinstance(output, torch.Tensor):
                input = output
                continue
            else:
                input, tmp_likelihoods = output
                likelihoods += tmp_likelihoods
        
        return input, likelihoods    
