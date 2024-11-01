from typing import Generic, TypeVar

from typer import Typer

from src.controls.handlers.base import BaseHandler

HandlerT = TypeVar("HandlerT", bound=BaseHandler)


class BaseRouter(Typer, Generic[HandlerT]):
    def __init__(self, handler: HandlerT): ...
