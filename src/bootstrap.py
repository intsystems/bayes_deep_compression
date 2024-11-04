from src.cli.cli import Cli
from src.cli.handlers.eager import EagerHandler
from src.cli.routers.eager import EagerRouter


def bootsrap():
    Cli(routers=[EagerRouter(EagerHandler())])()
