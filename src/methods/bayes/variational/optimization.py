from src.methods.bayes.base.optimization import BaseLoss
from src.methods.bayes.variational.distribution import ParamDist, LogUniformVarDist
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from abc import abstractmethod


class VarDistLoss(BaseLoss):
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


class LogUniformVarKLLoss(VarDistLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        param_sample_list: dict[str, nn.Parameter],
        posterior: dict[str, LogUniformVarDist],
        prior: dict[str, Optional[ParamDist]],
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
    ) -> VarDistLoss.AggregationResult:
        fit_loss = torch.mean(torch.stack(fit_losses))
        dist_loss = torch.stack(dist_losses)[0]
        total_loss = fit_loss + beta * dist_loss
        return VarDistLoss.AggregationResult(total_loss, fit_loss, dist_loss)

class NormVarKLLoss(VarDistLoss):
    def forward(
        self,
        posterior_params: nn.ParameterDict,
        prior_parmeter: Optional[nn.ParameterDict] = None,
    ) -> torch.Tensor:
        """Computes KL loss between factorized normals

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
    def __init__(self, alpha: float = -1,  aggregation: str = "weighting"):
        assert aggregation in ["weighting", "sample"]
        self.alpha = alpha
        self.aggregation = aggregation
        super().__init__()

    def forward(self,
                param_sample_dict: dict[str, nn.Parameter],
                posterior: dict[str, ParamDist],
                prior: dict[str, ParamDist]
            ) -> torch.Tensor:
        """ Computes KL loss between factorized normals

        Args:
            posterior_params (nn.ParameterList): factorized normal variational distribution
            prior_parmeter (Optional[nn.ParameterList]): assumed fixed N($\mu$, $\sigma$) for all paramteres. 
                As it is possible to analitically find optimal ($\mu$, $\sigma$), this parameter is ignored here.
        """

        prior_likelihood: torch.tensor = 0.
        posterior_likelihood: torch.tensor = 0.

        for (key,parameter), posterior_distr, prior_distr in zip(param_sample_dict.items(),
                                                       posterior.values(),
                                                       prior.values()):
            prior_likelihood += prior_distr.log_prob(parameter).sum()
            posterior_likelihood += posterior_distr.log_prob(parameter).sum()
            # if key == "fc2.bias":
                # print(posterior_distr.loc)
                # print(prior_likelihood)

        return prior_likelihood - posterior_likelihood

    def aggregate(self, fit_losses, dist_losses, beta_param) -> torch.Tensor:
        fit_losses = torch.stack(fit_losses)
        dist_losses = torch.stack(dist_losses)

        stat_tensor = - fit_losses + beta_param * dist_losses

        if self.aggregation == "sample":
            positions = F.gumbel_softmax(((1 - self.alpha) * stat_tensor.detach()), hard=True, dim=0)
        elif self.aggregation == "weighting":
            positions = F.softmax(((1 - self.alpha) * stat_tensor.detach()), dim=0).detach()
        else:
            raise ValueError()

        loss = -(positions * (stat_tensor)).sum()

        # print(loss)
        return VarDistLoss.AggregationResult(total_loss=loss,
                                            fit_loss=fit_losses.mean(),
                                            dist_loss=-dist_losses.mean()
                                            )
    
    
