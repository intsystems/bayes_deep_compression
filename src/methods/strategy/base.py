from torch.utils.data import DataLoader

from src.methods.bayes.base.distribution import BaseNetDistribution


class BaseStrategy:
    def __init__(self, model: BaseNetDistribution, dataset: DataLoader): ...
