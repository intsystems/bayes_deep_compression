from abc import ABC, abstractmethod

import torch


class BaseLoss(torch.nn.Module, ABC):
    """
    Abstract class for Distribution losses. Your distribution loss should
    be computed using prior and posterior classes and parameters, sampled from posterior.

    In forward method loss should realize logic of loss for one sampled weights.
    """
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        This method computes loss for one sampled parameters.
        """
        ...
