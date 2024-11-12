import torch
import torch.nn as nn
import copy

from src.methods.bayes.base.distribution import ParamDist
from src.utils.attribute import del_attr, set_attr, get_attr


class BaseNetDistribution:
    def __init__(
        self, base_module: nn.Module, weight_distribution: dict[str, ParamDist]
    ) -> None:
        super().__init__()
        self.base_module: nn.Module = base_module
        self.weight_distribution: dict[str, ParamDist] = weight_distribution

    def detach(self):
        self.base_module = copy.deepcopy(self.base_module)

    def sample_params(self) -> dict[str, nn.Parameter]:
        param_sample_dict: dict[str, nn.Parameter] = {}
        for param_name, param_posterior in self.weight_distribution.items():
            param_sample = param_posterior.rsample()
            param_sample_dict[param_name] = nn.Parameter(param_sample)
            del_attr(self.base_module, param_name.split("."))
            set_attr(self.base_module, param_name.split("."), param_sample)
        return param_sample_dict

    def sample_model(self) -> nn.Module:
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
        for param_name, dist in self.weight_distribution.items():
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
            self.dropout_mask_dict[name_dist] = nn.Parameter(
                torch.ones_like(dist.sample())
            )

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
        self.dropout_mask_dict[weight_name].data = 1.0 * (
            dist.log_z_test() >= threshold
        )

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

    def predict(self): ...
