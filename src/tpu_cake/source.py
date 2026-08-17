from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import FileLineColLoc, IntAttr, StringAttr
from xdsl.ir import Operation


@dataclass(frozen=True)
class SourceLocation:
    path: str
    line: int
    column: int

    def __post_init__(self) -> None:
        if not self.path or self.line <= 0 or self.column <= 0:
            raise ValueError("source locations require a path and positive line and column")

    def to_xdsl(self) -> FileLineColLoc:
        return FileLineColLoc(StringAttr(self.path), IntAttr(self.line), IntAttr(self.column))


def attach_source(operation: Operation, source: SourceLocation | None) -> Operation:
    if source is not None:
        operation.location = source.to_xdsl()
    return operation
