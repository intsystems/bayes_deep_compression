from src.controls.handlers.base import BaseHandler

from typing import 
from typer import Typer

class BaseRouter(Typer):
    def __init__(self, handler:  BaseHandler):
        ...