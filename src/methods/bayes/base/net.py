import copy
from typing import Optional
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from src.methods.bayes.base.distribution import ParamDist
from src.methods.bayes.base.net_distribution import BaseNetDistribution


class BayesLayer(nn.Module, ABC):
    """Abstract envelope around arbitrary nn.Module to substitute all its nn.Parameters
     with ParamDist. It transform it into bayessian Module. New distribution is a
     variational distribution which mimics the true posterior distribution.

    To specify bayes Module with custom posterior, please inherit this class and
    specify fields under.
    Attributes:
        prior_distribution_cls: Type of prior distribution that is used in this layer
        posterior_distribution_cls: Type of posterior distribution that is used in this layer
        is_posterior_trainable: Is posterior trainable
        is_posterior_trainable: Is prior trainable
    """

    prior_distribution_cls: Optional[ParamDist]
    posterior_distribution_cls: type[ParamDist]
    is_posterior_trainable: bool
    is_prior_trainable: bool

    def __init__(self, module: nn.Module) -> None:
        """_summary_

        Args:
            module (nn.Module): custom Module layer which is going to be converted to BayesLayer
        """
        super().__init__()
        posterior: dict[str, ParamDist] = {}
        self.prior: dict[str, Optional[ParamDist]] = {}
        """Prior disctribution for each weight"""
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
        self.net_distribution = BaseNetDistribution(
            module, weight_distribution=posterior
        )
        """Posterior net disctribution that is trained using data to fit to evaluate 
        probapility of each net consider the data"""
        self.posterior_params = nn.ParameterList()
        """
        key - weight_name, value - distribution_args: nn.ParameterDict
        this step is needed to register nn.Parameters of the ParamDists inside this class
        """
        for dist in self.posterior.values():
            param_dict = nn.ParameterDict(dist.get_params())
            for param in param_dict.values():
                param.requires_grad = self.is_posterior_trainable
            self.posterior_params.append(param_dict)
        # equal steps for prior distribution
        self.prior_params = nn.ParameterList()
        """
        key - weight_name, value - distribution_args: nn.ParameterDict
        this step is needed to register nn.Parameters of the ParamDists inside this class
        """
        for dist in self.prior.values():
            if isinstance(dist, ParamDist):
                param_dict = nn.ParameterDict(dist.get_params())
                for param in param_dict.values():
                    param.requires_grad = self.is_prior_trainable
                self.prior_params.append(param_dict)

    @property
    def posterior(self) -> dict[str, ParamDist]:
        """
        Returns posterior distribution for each weight

        Returns:
            dict[str, ParamDist]: dictionary where key - name of weight,
                value - postrior distribution of weight
        """
        return self.net_distribution.weight_distribution

    @property
    def base_module(self) -> nn.Module:
        """Return base_module that stores last sample module
        Return:
            nn.Module: strored module
        """
        return self.net_distribution.base_module

    def sample(self) -> dict[str, nn.Parameter]:
        """Sample new parameters from net distribution
        Return:
            dict[str, nn.Parameter]: new sampled parameters"""
        return self.net_distribution.sample_params()

    @property
    def weights(self) -> dict[str, nn.Parameter]:
        """
        Returns all weights in base_module

        Returns:
            dict[str, ParamDist]: dictionary where key - name of weight,
                value - weight value
        """
        weights: dict[str, nn.Parameter] = {}
        for param_name in self.net_distribution.weight_distribution:
            cur_param: torch.Tensor = getattr(self.base_module, param_name)
            weights[param_name] = cur_param
        return weights

    @property
    def device(self):
        """
        Return device of module
        """
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        """
        Make forward of last sampled module
        """
        return self.base_module(*args, **kwargs)

    def eval(self):
        """
        Alias of self.base_module.eval()
        """
        self.base_module.eval()

    def train(self):
        """
        Alias of self.base_module.train()
        """
        self.base_module.train()

    @abstractmethod
    def flush_weights(self) -> None:
        """
        This method simply set as tensors all weights that will be calculated by this layer and,
        so it will work properly when layer is initialized.
        """
        ...


class BaseBayesNet(nn.Module):
    """General envelope around arbitary nn.Module which is going to include nn.Modules and BayesModules
    as submodules.
    """

    def __init__(self, base_module: nn.Module, module_dict: nn.ModuleDict):
        """_summary_

        Args:
            base_module (nn.Module): custom Module which is going to have some BayesModule as submodules
            module_dict (nn.ModuleDict): all submodules of the base_module supposed to be trained. This
                may be nn.Module or BayesModule. Such division is required because base_module is not
                registred as Module in this class.
        """
        super().__init__()
        # self.__dict__["base_module"] = base_module
        self.base_module = base_module
        self.module_dict = module_dict

    def sample(self) -> dict[str, nn.Parameter]:
        """Sample new parameters of base_module from current posterior distribution

        Returns:
            dict[str, nn.Parameter]: new sampled paramters in form of dictionary
        """
        param_sample_dict: dict[str, nn.Parameter] = {}
        for module_name, module in self.module_dict.items():
            if isinstance(module, BayesLayer):
                parameter_dict = module.sample()
                for parameter_name, p in parameter_dict.items():
                    param_sample_dict[f"{module_name}.{parameter_name}"] = p
        return param_sample_dict

    @property
    def weights(self) -> dict[str, nn.Parameter]:
        """
        Returns all weights in base_module

        Returns:
            dict[str, ParamDist]: dictionary where key - name of weight,
                value - weight value
        """
        weights: dict[str, nn.Parameter] = {}
        for module_name, module in self.module_dict.items():
            for parameter_name, p in module.weights.items():
                weights[f"{module_name}.{parameter_name}"] = p
        return weights

    @property
    def device(self):
        """
        Return device of net
        """
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)

    def flush_weights(self) -> None:
        """
        This method simply set as tensors all weights that will be calculated by this layer and,
        so it will work properly when layer is initialized.
        """
        for module in self.module_dict.values():
            if isinstance(module, BayesLayer):
                module.flush_weights()

    def sample_model(self) -> nn.Module:
        """Sample base model and return deepcopy of it.

        Returns:

        """
        self.sample()

        for module in self.module_dict.values():
            if isinstance(module, BayesLayer):
                for param_name in module.net_distribution.weight_distribution:
                    cur_param: torch.Tensor = getattr(module.base_module, param_name)
                    setattr(module.base_module, param_name, nn.Parameter(cur_param))
                    
        model = copy.deepcopy(self.base_module)
        self.flush_weights()
        return model

    def eval(self):
        """
        Alias of self.base_module.eval()
        """
        return self.base_module.eval()

    def train(self):
        """
        Alias of self.base_module.train()
        """
        return self.base_module.train()

    @property
    def posterior(self) -> dict[str, ParamDist]:
        """
        Returns posterior distribution for each weight

        Returns:
            dict[str, ParamDist]: dictionary where key - name of weight,
                value - postrior distribution of weight
        """
        # self.params = {mus: , sigmas: }
        posteriors: dict[str, ParamDist] = {}
        for module_name, module in self.module_dict.items():
            if isinstance(module, BayesLayer):
                for parameter_name, parameter_posterior in module.posterior.items():
                    posteriors[f"{module_name}.{parameter_name}"] = parameter_posterior
        return posteriors

    @property
    def prior(self) -> dict[str, Optional[ParamDist]]:
        """
        Returns prior distribution for each weight

        Returns:
            dict[str, ParamDist]: dictionary where key - name of weight,
                value - prior distribution of weight
        """
        # self.params = {mus: , sigmas: }
        priors: dict[str, Optional[ParamDist]] = {}
        for module_name, module in self.module_dict.items():
            module_prior = None
            if isinstance(module, BayesLayer):
                module_prior = module.prior
                priors.update(module_prior)
                for parameter_name, parameter_prior in module.prior.items():
                    priors[f"{module_name}.{parameter_name}"] = parameter_prior
        return priors
