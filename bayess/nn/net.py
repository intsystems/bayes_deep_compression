"""
ds
"""

from pydantic.types import PositiveInt
from typing import Generic, TypeVar, List
from collections import defaultdict
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bayess.priors.priors import isotropic_gauss_prior, isotropic_gauss

from bayess.nn.functional import get_normal_kl
from bayess.resulting_types import KLLossModelOutput, RenuiLossModelOutput, BaseModelOutput


T = TypeVar("T")

class BaseModel(nn.Module, Generic[T]):
    """
    base class for bayessian models
    """
    def __init__(self, output_type: type[T]) -> None:
        super().__init__()
        self.output_type = output_type
    
    @abstractmethod
    def get_mle_model(self,):
        """
        should return nn.Module with maximum likelihood posterior approximating parameters
        """
        raise NotImplementedError("implement me")
    
    @abstractmethod
    def forward(self, input: Tensor) -> T:
        """
        forward method, should get as input input tensor and number of samples
        """
        raise NotImplementedError("implement me")
    @abstractmethod
    def predict(self, input: Tensor) -> BaseModelOutput:
        """
        Predict target values using MLE weights
        """
        raise NotImplementedError("implement me")

class BayessLinear(BaseModel):
    """
    Simple bayessian Linear layer
    """
    def __init__(self,n_in, n_out, prior = isotropic_gauss_prior(0, 1.), output_type: BaseModelOutput = BaseModelOutput):
        super(BayessLinear, self).__init__(output_type)

        if self.output_type is KLLossModelOutput:
            setattr(self, "forward", self.forward_kl)
        elif self.output_type is RenuiLossModelOutput:
            setattr(self, "forward", self.forward_likelihoods)
        elif self.output_type is BaseModelOutput:
            pass
        else:
            raise NotImplementedError(f"Specify forward method for {self.output_type} return type.")


        self.n_in = n_in
        self.n_out = n_out

        if prior is None:
            prior = isotropic_gauss_prior(0, 1.)
        self.prior = prior

        # since reparametrisation is hardcoded, then posterior is so
        self.posterior = isotropic_gauss()

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(n_in, n_out).uniform_(-0.1, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(n_out).uniform_(-0.1, 0.1))

        self.weight_sigma = nn.Parameter(torch.Tensor(n_in, n_out).uniform_(-3, -1))
        self.bias_sigma = nn.Parameter(torch.Tensor(n_out).uniform_(-3, -1))
    
    def get_mle_model(self):
        layer = nn.Linear(self.n_in, self.n_out)
        layer.weight.copy_(self.weight_mu)
        layer.bias.copy_(self.bias_mu)
        return layer
    
    def _std_precompute(self):
        std_weight = 1e-8 + F.softplus(self.weight_sigma, beta = 1, threshold = 20)
        std_b = 1e-8 + F.softplus(self.bias_sigma, beta = 1, threshold = 20)
        return std_weight, std_b

    def _sample_params(self, std_weight: Tensor, std_b: Tensor) -> tuple[Tensor, Tensor]:    
        eps_weight = self.weight_mu.new(self.weight_mu.size()).normal_()
        eps_b = self.bias_mu.new(self.bias_mu.size()).normal_()

        weight = self.weight_mu + std_weight * eps_weight
        b = self.bias_mu + std_b * eps_b

        return weight, b
    
    def forward_kl(self, input: Tensor) -> T:
        """
        Forward function for output of type 'KLLossModelOutput' 
        """
        std_weight, std_b = self._std_precompute()
        weight, b = self._sample_params(std_weight, std_b)
        output = torch.mm(input, weight) + b.expand(input.size()[0], self.n_out)

        kl_loss = get_normal_kl(self.weight_mu, std_weight).sum() + \
                    get_normal_kl(self.bias_mu, std_b)

        rez = KLLossModelOutput(outputs = output, kullback_leubler=kl_loss)
        return rez

    def forward_likelihoods(self, input: Tensor) -> T:
        """
        Forward function for output of type 'RenuiLossModelOutput' 
        """
        std_weight, std_b = self._std_precompute()
        weight, b = self._sample_params(std_weight, std_b)

        output = torch.mm(input, weight) + b.expand(input.size()[0], self.n_out)

        prior_likelihood = self.prior.loglike(weight) + self.prior.loglike(b)

        posterior_likelihood = self.posterior.loglike(weight, self.weight_mu, std_weight) + \
                            self.posterior.loglike(b, self.bias_mu, std_b)

        rez = RenuiLossModelOutput(outputs= output,
                                prior_likelihood= prior_likelihood,
                                posterior_likelihood= posterior_likelihood)
        return rez
   
    def forward(self, input: Tensor) -> T:
        std_weight, std_b = self._std_precompute()
        weight, b = self._sample_params(std_weight, std_b)
        output = torch.mm(input, weight) + b.expand(input.size()[0], self.n_out)
        rez = BaseModelOutput(outputs=output)
        return rez


    def predict(self, input: Tensor):
        output = torch.mm(input, self.weight_mu) + self.bias_mu.expand(input.size()[0], self.n_out)
        rez = BaseModelOutput(outputs=output)
        return rez


    
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
    def __init__(self, output_type: T, dimentions: List[PositiveInt], prior = None) -> None:
        assert output_type in [KLLossModelOutput, RenuiLossModelOutput, BaseModelOutput]
        super().__init__(output_type)

        submodules = []
        self.dimentions = dimentions
        for i in range(len(dimentions) - 2):
            submodules.append(BayessLinear(dimentions[i],dimentions[i+1],prior=prior, output_type= output_type))
            submodules.append(nn.ReLU(inplace = True))
        submodules.append(BayessLinear(dimentions[-2],dimentions[-1], prior=prior, output_type= output_type))

        self.net = nn.ParameterList(submodules)
    
    def get_mle_model(self):
        submodules = []
        for elem in self.net:
            if hasattr(elem, "get_mle_model"):
                elem = elem.get_mle_model()
            submodules.append(elem)
        net = nn.Sequential(submodules)
        return net
    
    def forward(self, input: Tensor) -> BaseModelOutput:            
        result = defaultdict(float)

        for elem in self.net:
            output = elem(input)
            if isinstance(output, Tensor):
                input = output
                continue
            else:
                # if there is a Bayessian layer
                input = output.outputs
                for field_name in output.get_names():
                    if field_name == "outputs":
                        continue
                    result[field_name] += output[field_name]
        
        result["outputs"] = input
        result = self.output_type(**result)
        return result        