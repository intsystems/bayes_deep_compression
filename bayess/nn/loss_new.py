from typing import Generic, TypeVar, Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from bayess.nn.net_new import likelihood_handler

type Loss = nn.modules.loss._Loss
class RenuiLoss(nn.Module):
    def __init__(self, data_loss: Loss, batches_count:float, alpha: float = -1, aggregation = "weighting") -> None:        
        super().__init__()
        self.data_loss = data_loss
        assert aggregation in ["weighting", "sample"]
        self.alpha = alpha
        self.batches_count = batches_count
        self.aggregation = aggregation

    def forward(self,  input: List[Tuple[torch.Tensor, likelihood_handler]], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        n_samples = len(input)

        report_metrics = defaultdict(float)

        stat_tensor = torch.zeros((n_samples, 1)) #rows are likelihoods of items
        for i in range(n_samples):
            labels_i, likelihoods_i = input[i]

            data_likelihood = -self.data_loss(labels_i, target)

            posterior_likelihood = likelihoods_i.posterior_likelihood
            prior_likelihood = likelihoods_i.prior_likelihood
            
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