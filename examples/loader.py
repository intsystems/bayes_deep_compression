from torch.utils.data import DataLoader

from torchvision import transforms, datasets


class TrainLoader(DataLoader):
    def __init__(self, batch_size=3):
        super().__init__(
            datasets.MNIST(
                root="../data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=3,
        )
