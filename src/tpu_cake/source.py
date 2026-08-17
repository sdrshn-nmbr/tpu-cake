from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import FileLineColLoc, IntAttr, ModuleOp, StringAttr
from xdsl.ir import Operation
from xdsl.utils.exceptions import VerifyException


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


def _source_location(operation: Operation) -> str | None:
    location = str(operation.location)
    if location != "loc(unknown)":
        return location
    for operand in operation.operands:
        owner = operand.owner
        if isinstance(owner, Operation):
            location = str(owner.location)
            if location != "loc(unknown)":
                return location
    return None


def verify_with_sources(module: ModuleOp) -> None:
    for operation in module.walk():
        if operation.regions:
            continue
        try:
            operation.verify(verify_nested_ops=False)
        except VerifyException as error:
            location = _source_location(operation)
            if location is None:
                raise
            raise VerifyException(
                f"{error}: {operation.name} at {location}"
            ) from error
    module.verify()
