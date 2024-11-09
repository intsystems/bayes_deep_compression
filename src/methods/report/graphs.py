from itertools import cycle
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from src.methods.report.base import BasePlot

TICKS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16


class TrainingCurves:
    def __init__(self): ...
    def plot_training_curves(
        train_losses: Dict[str, List[float]],
        test_losses: Dict[str, List[float]],
        logscale_y: bool = False,
        logscale_x: bool = False,
    ) -> None:
        colors_list = cycle(mpl.colormaps["Paired_r"].colors)
        n_train = len(train_losses[list(train_losses.keys())[0]])
        n_test = len(test_losses[list(train_losses.keys())[0]])
        x_train = np.linspace(0, n_test - 1, n_train)
        x_test = np.arange(n_test)

        plt.figure()
        for key, value in train_losses.items():
            plt.plot(
                x_train, value, label=key + "_train", color=next(colors_list)
            )  # , alpha=0.8)

        for key, value in test_losses.items():
            plt.plot(
                x_test, value, label=key + "_test", color=next(colors_list)
            )  # , alpha=0.8)

        if logscale_y:
            plt.semilogy()

        if logscale_x:
            plt.semilogx()

        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
        plt.ylabel("Loss", fontsize=LABEL_FONT_SIZE)
        plt.xticks(fontsize=TICKS_FONT_SIZE)
        plt.yticks(fontsize=TICKS_FONT_SIZE)
        plt.grid()
        plt.show()


class TensorBoardPlot(BasePlot): ...


class ShrinkagePlot(BasePlot):
    """
    Show dependency with shrinking number of weights
    """


class MetricShrinkagePlot(ShrinkagePlot):
    """
    Show metric degradation with more aggressive shrinkage approach
    """
