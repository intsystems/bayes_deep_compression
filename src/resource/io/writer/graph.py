from pathlib import Path

import matplotlib.pyplot as plt


class FigureSaver:
    def save(self, figure: plt.Figure, path: Path):
        figure.savefig(path)
