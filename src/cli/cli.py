from typer import Typer
from src.cli.routers.base import BaseRouter


class Cli(Typer):
    def __init__(self, routers: list[BaseRouter]):
        super().__init__()
        for router in routers:
            self.add_typer(router)
