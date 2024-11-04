from typing import ClassVar, Generic, TypeVar

from typer import Typer

from src.cli.handlers.base import BaseHandler

HandlerT = TypeVar("HandlerT", bound=BaseHandler)


class BaseRouter(Typer, Generic[HandlerT]):
    NAME: ClassVar[str]

    def __init__(self, handler: HandlerT):
        super().__init__(name=self.NAME)
