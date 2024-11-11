from src.methods.bayes.base.optimization import BaseLoss
from src.methods.bayes.variational.distribution import ParamDist, LogNormVarDist
import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional

class VarKLLoss(BaseLoss):
    def __init__(self):
        super().__init__()
    def forward(self, param_sample_list, posterior: dict[str, LogNormVarDist], prior: Optional[dict[str, ParamDist]]) -> torch.Tensor:
        ...
class NormLogVarKLLoss(VarKLLoss):
    def __init__(self):
        super().__init__()
    def forward(self, param_sample_list, posterior: dict[str, LogNormVarDist], prior: Optional[dict[str, ParamDist]]) -> torch.Tensor: 
        
        k1 = torch.tensor(0.63576)
        k2 = torch.tensor(1.87320)
        k3 = torch.tensor(1.48695)
        KL_z = 0
        for dist in posterior.values():
            KL_z_element = k1 * F.sigmoid(k2 + k3 * dist.scale_alphas_log) \
                - 0.5 * F.softplus(-dist.scale_alphas_log) - k1
            KL_z = KL_z + KL_z_element.sum()
        KL_w = 0
        for dist in posterior.values():
            KL_w_element = 0.5 * (torch.log(1/torch.exp(dist.param_std_log)**2) + \
                                  torch.exp(dist.param_std_log)**2 + dist.param_mus ** 2 - 1)
            KL_w = KL_w  + KL_w_element.sum()
        
        return -KL_z + KL_w 
    def aggregate(self, losses: list) -> torch.Tensor: 
        return losses[0]