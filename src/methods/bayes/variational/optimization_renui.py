
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.bayes.variational.distribution import ParamDist, LogUniformVarDist

from .optimization import VarKLLoss


class VarRenuiLoss(VarKLLoss):
    def __init__(self, alpha: float = -1,  aggregation: str = "weighting"):
        assert aggregation in ["weighting", "sample"]
        self.alpha = alpha
        self.aggregation = aggregation
        super().__init__()

    def forward(self,
                param_sample_dict: dict[str, LogUniformVarDist],
                posterior: dict[str, LogUniformVarDist],
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
        return VarKLLoss.AggregationResult(total_loss=loss,
                                            fit_loss=fit_losses.mean(),
                                            dist_loss=-dist_losses.mean()
                                            )
    
    
