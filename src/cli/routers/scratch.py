from src.controls.handlers.scratch import ScratchHandler
from src.controls.routers.base import BaseRouter


class ScratchRouter(BaseRouter[ScratchHandler]):
    def __init__(self, handler: ScratchHandler):
        self.command("elbo")(handler)
