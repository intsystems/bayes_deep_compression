class QuantileTrimmer:
    def __init__(self, q: float):
        self.q = q

    def prune(self): ...
