class BasePlot:
    """
    Base class for graph API
    """


class TensorBoardPlot(BasePlot): ...


class ShrinkagePlot(BasePlot):
    """
    Show dependency with shrinking number of weights
    """


class MetricShrinkagePlot(ShrinkagePlot):
    """
    Show metric degradation with more aggressive shrinkage approach
    """