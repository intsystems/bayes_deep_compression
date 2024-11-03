from src.methods.report.base import BasePlot




class TensorBoardPlot(BasePlot): ...


class ShrinkagePlot(BasePlot):
    """
    Show dependency with shrinking number of weights
    """


class MetricShrinkagePlot(ShrinkagePlot):
    """
    Show metric degradation with more aggressive shrinkage approach
    """
