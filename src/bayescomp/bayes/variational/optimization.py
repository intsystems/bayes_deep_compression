from bayescomp.bayes.base.optimization import BaseLoss
from bayescomp.bayes.variational.distribution import ParamDist, LogUniformVarDist
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from abc import abstractmethod

class VarKLLoss(BaseLoss):
    @dataclass
    class AggregationResult:
        total_loss: torch.Tensor
        fit_loss: torch.Tensor
        dist_loss: torch.Tensor

    def __init__(self):
        super().__init__()
    @abstractmethod
    def forward(
        self,
        param_sample_list: dict[str, nn.Parameter],
        posterior: dict[str, LogUniformVarDist],
        prior: dict[str, Optional[ParamDist]],
    ) -> torch.Tensor: ...
    @abstractmethod
    def aggregate(
        self, fit_losses: list, dist_losses: list, beta: float
    ) -> AggregationResult: ...
class LogUniformVarKLLoss(VarKLLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        param_sample_list,
        posterior: dict[str, LogUniformVarDist],
        prior: dict[str, None],
    ) -> torch.Tensor:
        k1 = torch.tensor(0.63576)
        k2 = torch.tensor(1.87320)
        k3 = torch.tensor(1.48695)
        KL_z = torch.tensor(0)
        for dist in posterior.values():
            KL_z_element = (
                k1 * F.sigmoid(k2 + k3 * dist.scale_alphas_log)
                - 0.5 * F.softplus(-dist.scale_alphas_log)
                - k1
            )
            KL_z = KL_z + KL_z_element.sum()
        KL_w = torch.tensor(0)
        for dist in posterior.values():
            KL_w_element = 0.5 * (
                torch.log(1 / torch.exp(dist.param_std_log) ** 2)
                + torch.exp(dist.param_std_log) ** 2
                + dist.param_mus**2
                - 1
            )
            KL_w = KL_w + KL_w_element.sum()

        return -KL_z + KL_w

    def aggregate(
        self, fit_losses: list, dist_losses: list, beta: float
    ) -> VarKLLoss.AggregationResult:
        fit_loss = torch.mean(torch.stack(fit_losses))
        dist_loss = torch.stack(dist_losses)[0]
        total_loss = fit_loss + beta * dist_loss
        return VarKLLoss.AggregationResult(total_loss, fit_loss, dist_loss)
