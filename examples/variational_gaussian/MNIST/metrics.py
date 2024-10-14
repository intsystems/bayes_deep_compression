import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tqdm.notebook import tqdm


def log_metrics(
        step: int, 
        net: nn.Module, 
        test_loader: DataLoader, 
        device: torch.device,
        writer: SummaryWriter = None
) -> dict:
    metrics = {}
    logits = []
    targets = []

    for imgs, target in tqdm(test_loader, desc="Test batchs", leave=True):
        imgs = imgs.to(device)

        logits.append(net(imgs))
        targets.append(target)

    logits = torch.concat(logits)
    targets = torch.concat(targets)

    metrics["Test/Cross_entropy"] = F.cross_entropy(logits, targets).item()
    metrics["Test/Accuracy"] = accuracy_score(targets, logits.argmax(dim=1))
    metrics["Test/Precision_macro"] = precision_score(targets, logits.argmax(dim=1), average="macro")
    metrics["Test/Recall_macro"] = recall_score(targets, logits.argmax(dim=1), average="macro")

    if writer is not None:
        for key, val in metrics.items():
            writer.add_scalar(key, val, step)

    return metrics


