from src.methods.report.graphs import BasePlot
from src.methods.report.metrics import BaseMetric


class GraphChain:
    def __init__(self, graphs: list[BasePlot]):
        self.graphs = graphs


class MetricsChain:
    def __init__(self, metric: list[BaseMetric]): ...
