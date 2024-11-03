from torchvision.datasets import MNIST
from src.resource.dataset import DatasetLoader


class MNISTDatasetLoader(DatasetLoader):
    def __init__(self, batch_size: int, num_workers: int):
        super().__init__(MNIST().train(), batch_size, num_workers)
