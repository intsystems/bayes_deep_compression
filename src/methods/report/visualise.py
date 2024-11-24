from collections import defaultdict
from itertools import cycle
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from IPython.display import clear_output

from src.methods.report.base import BaseReport
TICKS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16


def plot_training_curves(
    train_losses: Dict[str, List[float]],
    test_losses: Dict[str, List[float]],
    logscale_y: bool = False,
    logscale_x: bool = False,
) -> None:
    colors_list = cycle(mpl.colormaps["Paired_r"].colors)
    n_train = len(train_losses[list(train_losses.keys())[0]])
    n_test = len(test_losses[list(test_losses.keys())[0]])
    x_train = np.arange(n_train) #np.linspace(0, n_test - 1, n_train)
    x_test = np.arange(n_test)
    plt.figure()
    for key, value in train_losses.items():
        plt.plot(x_train, value, label=key + "_train", color=next(colors_list)) #, alpha=0.8)

    for key, value in test_losses.items():
        plt.plot(x_test, value, label=key + "_test", color=next(colors_list)) #, alpha=0.8)

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


class VarPlotReport(BaseReport):
    def __init__(self, report_names: List[str] = ['total_loss', 'kl_loss'],
                    val_report_names:List[str] = ['val_total_loss'],
                    logscale_x: bool = False,
                    logscale_y: bool = False
                ):
        self.train_losses: Dict[str, List[float]] = defaultdict(list)
        self.val_losses: Dict[str, List[float]] = defaultdict(list)
        self.logscale_y: bool = logscale_y
        self.logscale_x: bool = logscale_x
        self.report_names = report_names
        self.val_report_names = val_report_names

    def __call__(self, callback: dict) -> None:
        for key in callback.keys():
            for dct, report_names in zip([self.train_losses, self.val_losses], 
                                        [self.report_names, self.val_report_names]):
                if key in report_names:
                    value = callback[key]
                    if not isinstance(value, float):
                        value = value.detach().cpu()
                    dct[key].append(value)

        clear_output(wait=True)
        plot_training_curves(self.train_losses,
                             self.val_losses,
                             self.logscale_y,
                             self.logscale_x)
