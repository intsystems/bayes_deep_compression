from typing import Optional

import torch
import torch.nn as nn

from .optimization import VarKLLoss


class NormVarKLLoss(VarKLLoss):
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
