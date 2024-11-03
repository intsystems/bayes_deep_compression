from src.cli.handlers.base import BaseHandler


class EagerHandler(BaseHandler):
    def elbo_arch(self): ...
