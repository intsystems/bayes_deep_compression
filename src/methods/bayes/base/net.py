import copy
from typing import Optional

import torch.nn as nn
from src.methods.bayes.base.distribution import ParamDist
from src.methods.bayes.base.net_distribution import BaseNetDistribution


class BayesModule(nn.Module):
    prior_distribution_cls: Optional[ParamDist]
    posterior_distribution_cls: type[ParamDist]
    is_posterior_trainable: bool
    is_prior_trainable: bool

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        posterior: dict[str, ParamDist] = {}
        self.prior: dict[str, Optional[ParamDist]] = {}

        i = 0
        # Itereate to create posterior and prior dist for each parameter
        for name, p in list(module.named_parameters()):
            p.requires_grad = False
            posterior[name] = self.posterior_distribution_cls.from_parameter(p)
            self.prior[name] = None
            if self.prior_distribution_cls is not None:
                self.prior[name] = self.prior_distribution_cls.from_parameter(p)
            i += 1
        # BaseNetDistribution - это не Module
        self.net_distribution = BaseNetDistribution(module, weight_distribution=posterior)

        # key - weight_name, value - distribution_args: nn.ParameterDict
        self.posterior_params = nn.ParameterList()
        for dist in self.posterior.values():
            param_dict = nn.ParameterDict(dist.get_params())
            for param in param_dict.values():
                param.requires_grad = self.is_posterior_trainable
            self.posterior_params.append(param_dict)

        self.prior_params = nn.ParameterList()
        for dist in self.prior.values():
            if isinstance(dist, ParamDist):
                param_dict = nn.ParameterDict(dist.get_params())
                for param in param_dict.values():
                    param.requires_grad = self.is_prior_trainable
                self.prior_params.append(param_dict)

    @property
    def posterior(self) -> dict[str, ParamDist]:
        return self.net_distribution.weight_distribution

    @property
    def base_module(self) -> nn.Module:
        return self.net_distribution.base_module

    def sample(self) -> dict[str, nn.Parameter]:
        return self.net_distribution.sample_params()

    @property
    def weights(self) -> dict[str, nn.Parameter]:
        return dict(self.base_module.named_parameters())

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)

    def eval(self):
        self.base_module.eval()

    def train(self):
        self.base_module.train()


class BaseBayesModuleNet(nn.Module):
    def __init__(self, base_module: nn.Module, module_dict: nn.ModuleDict):
        super().__init__()
        #self.__dict__["base_module"] = base_module
        self.base_module = base_module
        self.module_dict = module_dict

    def sample(self) -> dict[str, nn.Parameter]:
        param_sample_dict: dict[str, nn.Parameter] = {}
        for module_name, module in self.module_dict.items():
            if isinstance(module, BayesModule):
                parameter_dict = module.sample()
                for parameter_name, p in parameter_dict.items():
                    param_sample_dict[f'{module_name}.{parameter_name}'] = p
        return param_sample_dict

    @property
    def weights(self) -> dict[str, nn.Parameter]:
        weights: dict[str, nn.Parameter] = {}
        for module_name, module in self.module_dict.items():
            if isinstance(module, BayesModule):
                for parameter_name, p in module.weights.items():
                    weights[f'{module_name}.{parameter_name}'] = p
        return weights

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)

    def flush_weights(self) -> None:
        for module in self.module_dict.values():
            if isinstance(module, BayesModule):
                module.flush_weights()

    def sample_model(self) -> nn.Module:
        self.sample()
        model = copy.deepcopy(self.base_module)
        self.flush_weights()
        return model

    def eval(self):
        return self.base_module.eval()

    def train(self):
        return self.base_module.train()

    @property
    def posterior(self) -> dict[str, ParamDist]:
        # self.params = {mus: , sigmas: }
        posteriors: dict[str, ParamDist] = {}
        for module_name, module in self.module_dict.items():
            if isinstance(module, BayesModule):
                for parameter_name, parameter_posterior in module.posterior.items():
                    posteriors[f'{module_name}.{parameter_name}'] = parameter_posterior
        return posteriors

    @property
    def prior(self) -> dict[str, Optional[ParamDist]]:
        # self.params = {mus: , sigmas: }
        priors: dict[str, Optional[ParamDist]] = {}
        for module_name, module in self.module_dict.items():
            module_prior = None
            if isinstance(module, BayesModule):
                module_prior = module.prior
                priors.update(module_prior)
                for parameter_name, parameter_prior in module.prior.items():
                    priors[f'{module_name}.{parameter_name}'] = parameter_prior 
        return priors
