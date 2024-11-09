from src.methods.bayes.base.optimization import BaseLoss
import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional

class VarKLLoss(BaseLoss):
    def __init__(self):
        super().__init__()
    def forward(self, posterior_params: nn.ParameterDict, prior_parmeter: Optional[nn.ParameterDict]) -> torch.Tensor: ...
class NormLogVarKLLoss(VarKLLoss):
    def __init__(self):
        super().__init__()
    def forward(self, posterior_params: nn.ParameterDict, prior_parmeter: Optional[nn.ParameterDict] = None) -> torch.Tensor: 
        
        k1 = torch.tensor(0.63576)
        k2 = torch.tensor(1.87320)
        k3 = torch.tensor(1.48695)
        KL_z = 0
        for i in range(len(posterior_params["param_mus"])):
            KL_z_element = k1 * F.sigmoid(k2 + k3 * posterior_params["scale_alphas_log"][i]) \
                - 0.5 * F.softplus(-posterior_params['scale_alphas_log'][i]) - k1
            KL_z = KL_z + KL_z_element.sum()
        KL_w = 0
        for i in range(len(posterior_params["param_mus"])):
            KL_w_element = 0.5 * (torch.log(1/torch.exp(posterior_params["param_std_log"][i])**2) + \
                                  torch.exp(posterior_params["param_std_log"][i])**2 + posterior_params["param_mus"][i] ** 2 - 1)
            KL_w = KL_w  + KL_w_element.sum()
        
        return -KL_z + KL_w 