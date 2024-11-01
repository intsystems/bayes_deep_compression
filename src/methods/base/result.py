import torch
from dataclasses import dataclass

@dataclass
class BaseOutputModel:
    output: torch.Tensor[torch.float32]
