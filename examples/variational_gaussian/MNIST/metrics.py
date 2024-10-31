import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt

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

    for imgs, target in tqdm(test_loader, desc="Test batchs", leave=False):
        imgs = imgs.to(device, torch.float32)

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

        # show img classification
        figs = []
        NUM_EXAMPLES = 5
        i = 0
        test_loader_iter = iter(test_loader)
        while i < NUM_EXAMPLES:
            imgs, target = next(test_loader_iter)
            prediction = net(imgs.to(device)).cpu().argmax(axis=1)

            for j in range(test_loader.batch_size):
                if i < NUM_EXAMPLES:
                    fig, ax = plt.subplots()
                    ax.imshow(imgs[j].squeeze().numpy(), cmap="gray")
                    ax.set_title(f"Target = {target[j].item()}; prediction = {prediction[j].item()}")

                    figs.append(fig)

                    i += 1
                else:
                    break

        writer.add_figure("Test/Examples", figs, step)

    return metrics
