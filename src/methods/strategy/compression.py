from src.methods.strategy.base import BaseStrategy
from csv import DictReader


class CompressionStrategy(BaseStrategy):
    def compress(self): ...
