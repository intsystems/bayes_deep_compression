from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.resource.dataset import DatasetLoader


class CIFARDatasetLoader(DatasetLoader):
    loss = CrossEntropyLoss

    def __init__(self, batch_size: int, num_workers: int, train: bool):
        super().__init__(
            CIFAR10(
                train=train,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                    ]
                ),
            ).train(),
            batch_size,
            num_workers,
        )
