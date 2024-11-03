from pathlib import Path

import torch
import torch.nn as nn


class ModelWriter:
    def save(self, model: nn.Module, path: Path):
        torch.save(model.state_dict(), path)
