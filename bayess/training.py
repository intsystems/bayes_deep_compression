from collections import defaultdict
from IPython.display import clear_output
from typing import Dict, List, Optional

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tqdm.auto import tqdm

from examples.visualize import show_samples, visualize_2d_samples, plot_training_curves
from bayess.nn.net import BaseModel


def train_epoch(
    epoch: int,
    model: BaseModel,
    loss,
    train_loader: DataLoader,
    optimizer: Optimizer,
    device: str = "cpu",
    loss_key: str = "total",
    n_samples: int = 10
) -> defaultdict[str, List[float]]:
    model.train()

    stats = defaultdict(list)
    for x, y in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        x = x.to(device)
        y = y.to(device)

        outputs = []
        for _ in range(n_samples):
            output = model(x)
            outputs.append(output)

        losses = loss(outputs, y)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(epoch: int, model: BaseModel, loss,  data_loader: DataLoader, device: str = "cpu") -> defaultdict[str, float]:
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc=f'Evaluating epoch {epoch}'):
            x = x.to(device)
            y = y.to(device) 
            preds = [model(x)]
            losses = loss(preds, y)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset) # type: ignore
    return stats


def check_samples_is_2d(samples: np.ndarray) -> bool:
    shape = samples.shape
    if len(shape) == 2 and shape[1] == 2:
        return True
    return False


def train_model(
    model: BaseModel,
    loss, 
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler] = None,
    device: str = "cpu",
    loss_key: str = "total_loss",
    n_samples: int = 10,
    visualize_samples: bool = True,
    logscale_y: bool = False,
    logscale_x: bool = False,
):

    train_losses: Dict[str, List[float]] = defaultdict(list)
    test_losses: Dict[str, List[float]] = defaultdict(list)
    model = model.to(device)
    print("Start of the training")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            epoch, model,loss, train_loader, optimizer, device, loss_key, n_samples
        )
        if scheduler is not None:
            scheduler.step()
        test_loss = eval_model(epoch, model, loss, test_loader, device)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])

        epoch_loss = np.mean(train_loss[loss_key])

        if visualize_samples:
            clear_output(wait=True)
            plot_training_curves(train_losses, test_losses, logscale_y, logscale_x)
            for key, value in train_losses.items():
                print(f"{key=} {value[-5:]}")
        else:
            print(f"Epoch: {epoch}, loss: {epoch_loss}")
    if visualize_samples:
        plot_training_curves(train_losses, test_losses, logscale_y = False)
    print("End of the training")
    return train_losses, test_losses