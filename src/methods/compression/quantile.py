from src.methods.compression.base import BasePruner


class QuantilePruner(BasePruner):
    """
    Similar to p-test cut's weight
    if corresponding confidence interval doesn't
    contain zero
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def prune(self): ...
