class BaseReport:
    def __call__(self, callback): ...


class BasePlot(BaseReport):
    """
    Base class for graph API
    """


class BaseMetric(BaseReport): ...


class ReportChain:
    def __init__(self, reports: list[BaseReport]):
        self.reports = reports

    def report(self, callback) -> None:
        for report in self.reports:
            report(callback)
        return
