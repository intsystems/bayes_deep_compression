import copy

import torch
import torch.nn as nn
from src.methods.bayes.base.distribution import ParamDist
from src.utils.attribute import del_attr, get_attr, set_attr


class BaseNetDistribution:
    """ 
    Base class for distribution of nets. This class sees nets as elements of distribution.
    It helps sample nets from this distribution or estimate statistics of distribution.
    For this purpose it have base module architecture and distribution of parameters for 
    each of it weights.
    """
    def __init__(self, base_module: nn.Module, weight_distribution: dict[str, ParamDist]) -> None:
        super().__init__()
        self.base_module: nn.Module = base_module
        """Show default architecture of module for which should evalute parameters"""
        self.weight_distribution: dict[str, ParamDist] = weight_distribution
        """Distribution of parameter for each named parameter of base_module"""

    def detach(self):
        """
        Detach(Made deepcopy) base module from original module
        """
        self.base_module = copy.deepcopy(self.base_module)

    def sample_params(self) -> dict[str, nn.Parameter]:
        """
        Sample only model parameter from distribution and 
        return it.
        
        Returns:
            dict[str, nn.Parameter]: Return dict of sampled parameters, where 
                key is name of parameter, value is valeu of parameter
        """
        param_sample_dict: dict[str, nn.Parameter] = {}
        for param_name, param_posterior in self.weight_distribution.items():
            param_sample = param_posterior.rsample()
            param_sample_dict[param_name] = nn.Parameter(param_sample)
            del_attr(self.base_module, param_name.split("."))
            set_attr(self.base_module, param_name.split("."), param_sample)
        return param_sample_dict

    def sample_model(self) -> nn.Module:
        """
        Sample only model from distribution and 
        return it. Note that model is the same that in base_module.
        
        Returns:
            nn.Module: sampled base_module with sampled parameters
        """
        self.sample_params()
        return self.base_module

    def set_map_params(self) -> None:
        for param_name, dist in self.weight_distribution.items():
            pt = dist.map()
            # pt = torch.nn.Parameter(pt.to_sparse())
            set_attr(self.base_module, param_name.split("."), pt)

    def set_mean_params(self) -> None:
        for param_name, dist in self.weight_distribution.items():
            pt = dist.mean()
            # pt = torch.nn.Parameter(pt.to_sparse())
            set_attr(self.base_module, param_name.split("."), pt)

    def set_params(self) -> None:
        for param_name, dist in self.weight_distribution.items():
            pt = dist.mean()
            # pt = torch.nn.Parameter(pt.to_sparse())
            set_attr(self.base_module, param_name.split("."), pt)

    def __replace_with_parameters(self):
        for param_name in self.weight_distribution.keys():
            pt = get_attr(self.base_module, param_name.split("."))
            pt = nn.Parameter(pt)
            set_attr(self.base_module, param_name.split("."), pt)

    def get_model(self) -> nn.Module:
        self.__replace_with_parameters()
        return self.base_module

    def get_model_snapshot(self) -> nn.Module:
        return copy.deepcopy(self.get_model())


class BaseNetDistributionPruner:
    def __init__(self, net_distribution: BaseNetDistribution):
        self.net_distribution = net_distribution
        self.dropout_mask_dict: dict[str, nn.Parameter] = {}
        for name_dist, dist in self.net_distribution.weight_distribution.items():
            self.dropout_mask_dict[name_dist] = nn.Parameter(torch.ones_like(dist.sample()))

    def prune(self, threshold: float | dict[str, float]):
        for weight_name in self.net_distribution.weight_distribution:
            weight_threshold = threshold
            if isinstance(weight_threshold, dict):
                weight_threshold = weight_threshold[weight_name]
            self.prune_weight(weight_name, weight_threshold)

    def prune_weight(self, weight_name: str, threshold: float) -> None:
        self.set_weight_dropout_mask(weight_name, threshold)
        pt = get_attr(self.net_distribution.base_module, weight_name.split("."))
        pt = pt * self.dropout_mask_dict[weight_name]
        pt = nn.Parameter(pt)
        set_attr(self.net_distribution.base_module, weight_name.split("."), pt)

    def set_weight_dropout_mask(self, weight_name: str, threshold: float) -> None:
        dist = self.net_distribution.weight_distribution[weight_name]
        self.dropout_mask_dict[weight_name].data = 1.0 * (dist.log_z_test() >= threshold)

    def prune_stats(self) -> int:
        prune_cnt = 0
        for dropout in self.dropout_mask_dict.values():
            prune_cnt += (1 - dropout).sum()
        return prune_cnt

    def total_params(self) -> int:
        out = sum(p.numel() for p in self.dropout_mask_dict.values())
        return out


class BaseNetEnsemble:
    def __init__(self, net_distribution: BaseNetDistribution) -> None:
        self.net_distribution = net_distribution

    def predict(self):
        ...