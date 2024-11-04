from torch.utils.data.dataloader import DataLoader
import torch
from dataclasses import dataclass


@dataclass
class DatasetLoaderOutput:
    x_train: torch.Tensor
    y_train: torch.Tensor


class DatasetLoader(DataLoader):
    def __next__(self) -> DatasetLoaderOutput: ...
