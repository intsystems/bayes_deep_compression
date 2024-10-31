import torch
import torch.nn as nn


class ClassificationNet(nn.Module):
    """ classical NN architecture for multiclass classification
    """
    def __init__(self, img_size: tuple[int], num_classes: int, num_layers: int, hidden_size: int):
        super().__init__()

        layers = [nn.BatchNorm2d(1)]
        layers.append(nn.Flatten())
        layers.extend([nn.Linear(img_size[0] * img_size[1] * img_size[2], hidden_size)])

        for i in range(num_layers - 2):
            if i % 2 == 0:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        layers.append(nn.Linear(hidden_size, num_classes))
        self.transform = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: logits for classes
        """
        return self.transform(x)




