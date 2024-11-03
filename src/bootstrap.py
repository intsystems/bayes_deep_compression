from src.cli.cli import Cli
from src.cli.routers.eager import EagerRouter
from src.cli.handlers.eager import EagerHandler


def bootsrap():
    Cli(routers=[EagerRouter(EagerHandler())])()
