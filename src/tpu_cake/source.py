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


class SourceAwareVerifyException(VerifyException):
    pass


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


def _source_context(operation: Operation) -> tuple[str, ...]:
    entries: list[str] = []
    seen: set[tuple[str, str]] = set()
    for nested in operation.walk():
        location = _source_location(nested)
        if location is None:
            continue
        entry = (nested.name, location)
        if entry not in seen:
            seen.add(entry)
            entries.append(f"{nested.name} at {location}")
    return tuple(entries)


def source_aware_error(message: str, *operations: Operation) -> VerifyException:
    entries: list[str] = []
    for operation in operations:
        location = str(operation.location)
        if location != "loc(unknown)":
            entries.append(f"{operation.name} at {location}")
    if not entries:
        return SourceAwareVerifyException(message)
    return SourceAwareVerifyException(
        f"{message}: relevant source sites: {'; '.join(dict.fromkeys(entries))}"
    )


def _raise_with_sources(operation: Operation, error: VerifyException) -> None:
    if isinstance(error, SourceAwareVerifyException):
        raise error
    context = _source_context(operation)
    if not context:
        raise error
    raise VerifyException(
        f"{error}: available source context: {'; '.join(context)}"
    ) from error


def verify_with_sources(module: ModuleOp) -> None:
    for operation in module.walk():
        try:
            operation.verify(verify_nested_ops=False)
        except VerifyException as error:
            _raise_with_sources(operation, error)
    try:
        module.verify()
    except VerifyException as error:
        _raise_with_sources(module, error)
