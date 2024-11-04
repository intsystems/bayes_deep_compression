from dataclasses import dataclass

import torch


@dataclass
class KFLinearOutput:
    activation: torch.Tensor
    pre_activation: torch.Tensor
