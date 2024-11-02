from torch.utils.data import DataLoader

from src.methods.base.arch import BaseBayesModel


class BaseStrategy:
    def __init__(self, model: BaseBayesModel, dataset: DataLoader): ...
