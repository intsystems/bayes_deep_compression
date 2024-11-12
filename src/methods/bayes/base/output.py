from dataclasses import dataclass

import torch


@dataclass
class BaseOutputModel:
    output: torch.Tensor
