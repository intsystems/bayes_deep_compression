import matplotlib.pyplot as plt
from pathlib import Path


class FigureSaver:
    def save(self, figure: plt.Figure, path: Path):
        figure.savefig(path)
