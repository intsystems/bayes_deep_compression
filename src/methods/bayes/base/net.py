from typing import Optional
import torch.nn as nn
import copy
from src.methods.bayes.base.distribution import ParamDist
from src.methods.bayes.base.net_distribution import BaseNetDistribution


class BayesModule(nn.Module):
    prior_distribution_cls: Optional[ParamDist]
    posterior_distribution_cls: type[ParamDist]

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        posterior: dict[str, ParamDist] = {}
        self.prior: dict[str, ParamDist] = {}
        i = 0
        # Itereate to create posterior and prior dist for each parameter
        for name, p in list(module.named_parameters()):
            p.requires_grad = False
            posterior[name] = self.posterior_distribution_cls.from_parameter(p)
            self.prior[name] = None
            if self.prior_distribution_cls is not None:
                self.prior[name] = self.prior_distribution_cls.from_parameter(p)
            i += 1
        self.net_distribution = BaseNetDistribution(
            module, weight_distribution=posterior
        )

        # key - weight_name, value - distribution_args: nn.ParameterDict
        self.posterior_params = nn.ParameterList()
        for dist in self.posterior.values():
            self.posterior_params.append(nn.ParameterDict(dist.get_params()))

        self.prior_params = nn.ParameterList()
        if self.prior_distribution_cls is not None:
            for dist in self.prior.values():
                self.prior_params.append(dist.get_params())

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

    def eval(self) -> None:
        self.base_module.eval()

    def train(self) -> None:
        self.base_module.train()


class BaseBayesModuleNet(nn.Module):
    def __init__(self, base_module: nn.Module, module_list: nn.ModuleList):
        super().__init__()
        self.__dict__["base_module"] = base_module
        self.module_list = module_list

    def sample(self) -> dict[str, nn.Parameter]:
        param_sample_dict: dict[str, nn.Parameter] = {}
        for module in self.module_list:
            if isinstance(module, BayesModule):
                parameter_dict = module.sample()
                param_sample_dict.update(parameter_dict)
        return param_sample_dict

    def get_distr_params(
        self, param_type_name: str
    ) -> dict[str, dict[str, nn.Parameter]]:
        params_dict = {}
        for module in self.module_list:
            if isinstance(module, BayesModule):
                for key in getattr(module, param_type_name):
                    parameter_dict = getattr(module, param_type_name)[key]
                    params_dict.setdefault(key, {})
                    params_dict[key].update(parameter_dict.get_distr_params())
        return params_dict

    @property
    def weights(self) -> dict[nn.Parameter]:
        weights = {}
        for module in self.module_list:
            module_posterior = None
            if isinstance(module, BayesModule):
                module_posterior = module.weights
            weights.update(module_posterior)
        return weights

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)

    def flush_weights(self) -> None: ...
    def sample_model(self) -> nn.Module:
        self.sample()
        model = copy.deepcopy(self.base_module)
        self.flush_weights()
        return model

    def eval(self) -> None:
        self.base_module.eval()

    def train(self) -> None:
        self.base_module.train()

    @property
    def posterior(self) -> dict[str, ParamDist]:
        # self.params = {mus: , sigmas: }
        posteriors = {}
        for module in self.module_list:
            module_posterior = None
            if isinstance(module, BayesModule):
                module_posterior = module.posterior
            posteriors.update(module_posterior)
        return posteriors

    @property
    def prior(self):
        # self.params = {mus: , sigmas: }
        priors = {}
        for module in self.module_list:
            module_prior = None
            if isinstance(module, BayesModule):
                module_prior = module.prior
            priors.update(module_prior)
        return priors
