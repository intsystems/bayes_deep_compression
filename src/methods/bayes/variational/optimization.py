from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.bayes.base.optimization import BaseLoss
from src.methods.bayes.variational.distribution import LogUniformVarDist, ParamDist


class VarDistLoss(BaseLoss):
    """
    Abstract class for Distribution losses. Your distribution loss should
    be computed using prior and posterior classes and parameters, sampled from posterior.

    In forward method loss should realize logic of loss for one sampled weights.

    In aggregate method loss aggregate the data losses and distribution losses for samples.

    Aggregation returns `VarDistLoss.AggregationResult`

    """

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
        *,
        param_sample_dict: Dict[str, nn.Parameter],
        posterior: Dict[str, ParamDist],
        prior: Dict[str, ParamDist],
    ) -> torch.Tensor:
        """
        This method computes loss for one sampled parameters.

        Args:
            param_sample_dict: {param_name: nn.Parameter}
                    sampled parameters on network.
            posterior:  {param_name: ParamDist}
                    posterior distribution of net parameters.
            prior:  {param_name: ParamDist}
                    prior distribution of net parameters.
        """
        ...

    @abstractmethod
    def aggregate(self, fit_losses: list, dist_losses: list, beta: float) -> AggregationResult:
        ...


class LogUniformVarKLLoss(VarDistLoss):
    def __init__(self):
        super().__init__()

    def forward(self, posterior: dict[str, LogUniformVarDist], **kwargs) -> torch.Tensor:
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

    def aggregate(self, fit_losses: list, dist_losses: list, beta: float) -> VarDistLoss.AggregationResult:
        fit_loss = torch.mean(torch.stack(fit_losses))
        dist_loss = torch.stack(dist_losses)[0]
        total_loss = fit_loss + beta * dist_loss
        return VarDistLoss.AggregationResult(total_loss, fit_loss, dist_loss)


# TODO: пофиксить
class NormVarKLLoss(VarDistLoss):
    def forward(
        self,
        posterior_params: nn.ParameterDict,
        prior_parmeter: Optional[nn.ParameterDict] = None,
    ) -> torch.Tensor:
        r"""
        Computes KL loss between factorized normals

        Args:
            posterior_params (nn.ParameterList): factorized normal variational distribution
            prior_parmeter (Optional[nn.ParameterList]): assumed fixed N($\mu$, $\sigma$) for all paramteres.
                As it is possible to analitically find optimal ($\mu$, $\sigma$), this parameter is ignored here.
        """
        # optimal \alpha parameters
        # sigma_opt here is the squared sigma
        mu_opt = 0
        sigma_opt = 0
        # number of posterior random variables = num of parameters
        n_params = 0

        # compute mu
        for param in posterior_params["loc"]:
            mu_opt += param.sum()
            n_params += param.numel()
        mu_opt /= n_params

        # compute sigma in two steps
        for param in posterior_params["loc"]:
            sigma_opt += torch.sum((param - mu_opt) ** 2) / n_params
        for param in posterior_params["log_scale"]:
            sigma_opt += torch.exp(param).sum() / n_params

        # compute kl-loss between posterior and prior in two steps
        kl_loss = 0
        for param in posterior_params["loc"]:
            kl_loss += torch.sum(0.5 * (1 / sigma_opt) * (param - mu_opt) ** 2)
        for param in posterior_params["log_scale"]:
            kl_loss += -0.5 * param.sum() + param.numel() * 0.5 * torch.log(sigma_opt)
            kl_loss += torch.sum(-0.5 + 0.5 * torch.exp(param) / sigma_opt)

        return kl_loss


class VarRenuiLoss(VarDistLoss):
    """
    Loss with Renui divergence.
    https://arxiv.org/pdf/1602.02311v1
    """

    def __init__(self, alpha: float = -1, aggregation: str = "weighting"):
        assert aggregation in ["weighting", "sample"]
        self.alpha = alpha
        self.aggregation = aggregation
        super().__init__()

    def forward(
        self,
        param_sample_dict: dict[str, nn.Parameter],
        posterior: dict[str, ParamDist],
        prior: dict[str, ParamDist],
    ) -> torch.Tensor:
        prior_likelihood: torch.tensor = 0.0
        posterior_likelihood: torch.tensor = 0.0

        for parameter, posterior_distr, prior_distr in zip(
            param_sample_dict.values(), posterior.values(), prior.values()
        ):
            prior_likelihood += prior_distr.log_prob(parameter).sum()
            posterior_likelihood += posterior_distr.log_prob(parameter).sum()
            # if key == "fc2.bias":
            # print(posterior_distr.loc)
            # print(prior_likelihood)

        return prior_likelihood - posterior_likelihood

    def aggregate(self, fit_losses, dist_losses, beta_param) -> torch.Tensor:
        fit_losses = torch.stack(fit_losses)
        dist_losses = torch.stack(dist_losses)

        stat_tensor = -fit_losses + beta_param * dist_losses

        if self.aggregation == "sample":
            positions = F.gumbel_softmax(((1 - self.alpha) * stat_tensor.detach()), hard=True, dim=0)
        elif self.aggregation == "weighting":
            positions = F.softmax(((1 - self.alpha) * stat_tensor.detach()), dim=0).detach()
        else:
            raise ValueError()

        loss = -(positions * (stat_tensor)).sum()

        # print(loss)
        return VarDistLoss.AggregationResult(
            total_loss=loss, fit_loss=fit_losses.mean(), dist_loss=-dist_losses.mean()
        )
