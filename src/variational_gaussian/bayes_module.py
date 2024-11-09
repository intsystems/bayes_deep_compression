from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesModule(nn.Module):
    prior_distribution_cls: type[torch.distributions.distribution.Distribution]
    posterior_distribution_cls: type[torch.distributions.distribution.Distribution]
    
    def __init__(self, module: nn.Module):
        self.base_module = module
        self.posteriror_params = nn.ParameterDict()

    def sample(self):
        for i in range(len(range(len(self.list_scale_mus)))):
            # sample gaussian for each weight and scale
            param_epsilons = Variable(torch.randn_like(self.list_param_mus[i]))
            scale_epsilons = Variable(torch.randn_like(self.list_param_mus[i]))
            # calculate sample using reparametrization
            scale_sample = self.list_scale_mus[i] + scale_epsilons * (self.list_scale_mus[i]) * torch.sqrt(torch.exp(self.list_scale_alphas_log[i].data))
            param_sample = scale_sample * (self.list_param_mus[i] + param_epsilons * self.list_param_std[i])
            del_attr(self.base_module, self.index_to_name[i].split("."))
            set_attr(self.base_module, self.index_to_name[i].split("."), param_sample)

    def forward(self, *args, **kwargs):
        self.sample()

        return self.base_module(*args, **kwargs)
    def eval(self):
        self.base_module.eval()

    def train(self):
        self.base_module.train()

    @property
    def posterior(self):
        #self.params = {mus: , sigmas: }
        self.dist = self.posterior_distribution_cls(self.posteriror_params)
        return self.dist.sample()
    
    def sample_n(self, x, n: int):
        pass