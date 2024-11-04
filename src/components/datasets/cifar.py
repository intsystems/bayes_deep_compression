from src.resource.dataset import DatasetLoader
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss

class CIFARDatasetLoader(DatasetLoader):
    loss = CrossEntropyLoss
    
    def __init__(self, batch_size: int, num_workers: int, train: bool): 
        super().__init__(CIFAR10(train=train).train(), batch_size, num_workers)