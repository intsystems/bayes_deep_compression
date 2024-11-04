from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from src.resource.dataset import DatasetLoader


class MNISTDatasetLoader(DatasetLoader):
    loss = CrossEntropyLoss

    def __init__(self, batch_size: int, num_workers: int, train: bool):
        super().__init__(MNIST(train=train).train(), batch_size, num_workers)
