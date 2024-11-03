from src.cli.handlers.eager import ScratchHandler
from src.cli.routers.base import BaseRouter


class ScratchRouter(BaseRouter[ScratchHandler]):
    def __init__(self, handler: ScratchHandler):
        self.command("elbo", help="Model should be in input folder")(handler)
