import torch
from abc import abstractmethod, ABC
class BaseLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor: ...
