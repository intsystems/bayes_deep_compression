""" Common fixtures for training tests
"""
import pytest
import shutil
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms


class Classifier(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def mnist_classifier() -> nn.Module:
    return Classifier()


@pytest.fixture
def mnist_dataset() -> Dataset:
    data_dir = Path("tmp/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=False, transform=transforms.ToTensor()
    )
    yield train_dataset

    # clean dataset dir
    # shutil.rmtree(data_dir)
    # data_dir.parent.rmdir()
