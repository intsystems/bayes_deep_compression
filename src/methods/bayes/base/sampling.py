import torch.nn as nn


class BaseSampler:
    def __call__(self, *args) -> nn.Module: ...
