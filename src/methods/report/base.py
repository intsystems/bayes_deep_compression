class BaseReport:
    ...

class BasePlot(BaseReport):
    """
    Base class for graph API
    """

class BaseMetric(BaseReport):
    ...

class ReportChain:
    def __init__(self, reports: list[BaseReport]):
        self.reports = reports

    def report(self):
        ...