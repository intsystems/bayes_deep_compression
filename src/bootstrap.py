from src.cli.cli import Cli
from src.cli.routers.eager import EagerRouter

def bootsrap():
    Cli(
       routers=[
            EagerRouter()
       ]
    )()