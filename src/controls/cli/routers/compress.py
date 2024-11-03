from src.cli.handlers.eager import EagerHandler
from src.cli.routers.base import BaseRouter


class ScratchRouter(BaseRouter[EagerHandler]):
    def __init__(self, handler: EagerHandler):
        self.command("elbo", help="Model should be in input folder")(handler)
