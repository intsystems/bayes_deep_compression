from dataclasses import dataclass

import torch
from torch.utils.data.dataloader import DataLoader


@dataclass
class DatasetLoaderOutput:
    x_train: torch.Tensor
    y_train: torch.Tensor


class DatasetLoader(DataLoader):
    loss: torch.nn.MSELoss

    def __iter__(self):
        return self

    def __next__(self) -> DatasetLoaderOutput: ...
