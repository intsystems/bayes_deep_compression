from src.cli.handlers.eager import EagerHandler
from src.cli.routers.base import BaseRouter


class EagerRouter(BaseRouter[EagerHandler]):
    NAME = "eager"

    def __init__(self, handler: EagerHandler):
        super().__init__(handler)
        self.command("elbo")(handler.elbo_arch)
