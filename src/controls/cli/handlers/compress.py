from src.cli.handlers.base import BaseHandler
from src.compression.strategy.compression import CompressionStrategy


class CompressHandler(BaseHandler):
    def __init__(self, strategy: CompressionStrategy): ...

    def compress_linear_model(self): ...

    def compress_arbitary_model(self): ...
