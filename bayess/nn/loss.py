"""
Base loss class
"""
from typing import Generic, TypeVar, Dict, List
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from bayess.resulting_types import BaseModelOutput, RenuiLossModelOutput, KLLossModelOutput

type Loss = nn.modules.loss._Loss

T = TypeVar("T")

class BaseLoss(nn.Module, Generic[T]):
    def __init__(self, data_loss: Loss) -> None:
        super().__init__()
        self.data_loss = data_loss

    def forward(self, input: List[BaseModelOutput], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        There input is a list of model exits of length self.n_samples
        """
        n_samples = len(input)

        loss = 0.
        for i in range(n_samples):
            model_output_i = input[i]
            preds = model_output_i.outputs
            loss += self.data_loss(preds, target)

        loss = loss/self.n_samples

        return {'total_loss': loss}
        

class RenuiLoss(BaseLoss):
    def __init__(self, data_loss: Loss, batches_count:float, alpha: float = -1, aggregation = "weighting") -> None:        
        super().__init__(data_loss)
        assert aggregation in ["weighting", "sample"]
        self.alpha = alpha
        self.batches_count = batches_count
        self.aggregation = aggregation

    def forward(self, input: List[RenuiLossModelOutput], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        n_samples = len(input)

        report_metrics = defaultdict(float)

        stat_tensor = torch.zeros((n_samples, 1)) #rows are likelihoods of items
        for i in range(n_samples):
            model_output_i = input[i]

            labels = model_output_i.outputs
            data_likelihood = -self.data_loss(labels, target)

            posterior_likelihood = model_output_i.posterior_likelihood
            prior_likelihood = model_output_i.prior_likelihood
            
            tmp_val = self.batches_count * data_likelihood + prior_likelihood - posterior_likelihood
            stat_tensor[i] = tmp_val

            report_metrics['prior_likelihood'] += prior_likelihood.detach()
            report_metrics['posterior_likelihood'] += posterior_likelihood.detach()
            report_metrics['data_likelihood'] += data_likelihood.detach()
        
        for key, val in report_metrics.items():
            report_metrics[key] = val/n_samples

        if self.aggregation == "sample":
            positions = F.gumbel_softmax(((1 - self.alpha)* stat_tensor.detach()),hard=True, dim=0)
        elif self.aggregation == "weighting":
            positions = F.softmax(((1 - self.alpha)* stat_tensor.detach()), dim=0).detach()
        else:
            raise ValueError()
        
        loss = - positions * (stat_tensor)
        total_loss = loss.sum()
        report_metrics["total_loss"] = total_loss
        return  report_metrics


class KLLoss(BaseLoss):
    def __init__(self, data_loss: Loss, beta:float,) -> None:        
        """
        
        :beta: scaling factor for data loss. 
        Following theory, the best value to set is count of batches in dataset

        """
        super().__init__(data_loss)
        self.beta = beta


    def forward(self, input: List[KLLossModelOutput], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # compute kl loss
        n_samples = len(input)

        report_metrics = defaultdict(float)
        loss = 0.
        for i in range(n_samples):
            model_output_i = input[i]
            labels = model_output_i.outputs

            data_likelihood = -self.data_loss(labels, target)
            kl_loss = model_output_i.kullback_leubler
            
            loss += -self.beta * data_likelihood + kl_loss

            report_metrics['kl_loss'] += kl_loss.detach()
            report_metrics['data_likelihood'] += data_likelihood.detach()
        
        for key, val in report_metrics.items():
            report_metrics[key] = val/n_samples

        loss_final = loss / n_samples

        report_metrics["total_loss"] = loss_final
        return  report_metrics
        