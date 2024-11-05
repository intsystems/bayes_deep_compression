from src.controls.cli.cli import Cli
from src.controls.cli.handlers.eager import EagerHandler
from src.controls.cli.routers.eager import EagerRouter


def bootsrap():
    Cli(routers=[EagerRouter(EagerHandler())])()
