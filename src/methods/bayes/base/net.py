import copy
from typing import Optional

import torch.nn as nn
from src.methods.bayes.base.distribution import ParamDist
from src.methods.bayes.base.net_distribution import BaseNetDistribution


class BayesModule(nn.Module):
    """ Abstract envelope around arbitrary nn.Module to substitute all its nn.Parameters
         with ParamDist. It transform it into bayessian Module. New distribution is a
         variational distribution which mimics the true posterior distribution.

        To specify bayes Module with custom posterior, please inherit this class and
         specify fields under. 
    """
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
        # this step is needed to register nn.Parameters of the ParamDists inside this class
        self.posterior_params = nn.ParameterList()
        for dist in self.posterior.values():
            param_dict = nn.ParameterDict(dist.get_params())
            for param in param_dict.values():
                param.requires_grad = self.is_posterior_trainable
            self.posterior_params.append(param_dict)
        # equal steps for prior distribution
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
        # TODO: когда мы сэмплим, в base_module у нас все nn.Paramters заменятся на 
        # тензора. Данные метод будет некорректен.
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
    """ General envelope around arbitary nn.Module which is going to include nn.Modules and BayesModules
         as submodules. 
    """

    def __init__(self, base_module: nn.Module, module_list: nn.ModuleList):
        """_summary_

        Args:
            base_module (nn.Module): custom Module which is going to have some BayesModule as submodules
            module_list (nn.ModuleList): all submodules of the base_module supposed to be trained. This 
                may be nn.Module or BayesModule. Such division is required because base_module is not
                registred as Module in this class.
        """
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

    def get_distr_params(self, param_type_name: str) -> dict[str, dict[str, nn.Parameter]]:
        params_dict: dict[str, dict[str, nn.Parameter]] = {}
        for module in self.module_list:
            if isinstance(module, BayesModule):
                for key in getattr(module, param_type_name):
                    parameter_dict = getattr(module, param_type_name)[key]
                    params_dict.setdefault(key, {})
                    params_dict[key].update(parameter_dict.get_distr_params())
        return params_dict

    @property
    def weights(self) -> dict[str, nn.Parameter]:
        weights: dict[str, nn.Parameter] = {}
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

    def flush_weights(self) -> None:
        for module in self.module_list:
            if isinstance(module, BayesModule):
                # TODO: надо указать flush_weights в BayesModule
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
        for module in self.module_list:
            module_posterior = None
            if isinstance(module, BayesModule):
                module_posterior = module.posterior
                posteriors.update(module_posterior)
        return posteriors

    @property
    def prior(self) -> dict[str, Optional[ParamDist]]:
        # self.params = {mus: , sigmas: }
        priors: dict[str, Optional[ParamDist]] = {}
        for module in self.module_list:
            module_prior = None
            if isinstance(module, BayesModule):
                module_prior = module.prior
                priors.update(module_prior)
        return priors
