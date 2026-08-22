from __future__ import annotations

from collections import deque
from collections.abc import Callable

from jax._src.interpreters import mlir
from jaxlib.mlir import ir


def as_operation(value: object) -> ir.Operation | None:
    if isinstance(value, ir.Operation):
        return value
    operation = getattr(value, "operation", None)
    return operation if isinstance(operation, ir.Operation) else None


def _walk(operation: ir.Operation) -> tuple[ir.Operation, ...]:
    operations: list[ir.Operation] = []

    def visit(current: ir.Operation) -> None:
        operations.append(current)
        for region in current.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    visit(operation)
    return tuple(operations)


def _result_reaches(
    result: ir.Value,
    predicate: Callable[[ir.Operation], bool],
    *,
    blocked: tuple[ir.Operation, ...] = (),
) -> bool:
    pending = deque([result])
    visited: set[ir.Value] = set()
    while pending:
        value = pending.popleft()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            consumer = as_operation(use.owner)
            if consumer is None:
                raise ValueError("STABLEHLO_INVALID_USE")
            if any(consumer == operation for operation in blocked):
                continue
            if predicate(consumer):
                return True
            if consumer.name in {"sdy.return", "stablehlo.return"}:
                parent = as_operation(consumer.parent)
                if parent is not None and use.operand_number < len(parent.results):
                    pending.append(parent.results[use.operand_number])
            pending.extend(consumer.results)
    return False


def _result_reaches_return(result: ir.Value) -> bool:
    return _result_reaches(result, lambda operation: operation.name == "func.return")


class StableHloInspector:
    def __init__(
        self,
        context: ir.Context,
        module: ir.Module,
        functions: tuple[ir.Operation, ...],
        public_main: ir.Operation,
    ) -> None:
        self._context = context
        self._module = module
        self._functions = functions
        self._public_main = public_main

    @classmethod
    def parse(cls, text: str) -> StableHloInspector:
        try:
            context = mlir.make_ir_context()
            with context:
                module = ir.Module.parse(text)
                module.operation.verify()
                functions = tuple(
                    operation.operation
                    for operation in module.body
                    if operation.operation.name == "func.func"
                )
                entries = tuple(
                    function
                    for function in functions
                    if str(function.attributes.get("sym_name")) == '"main"'
                    and str(function.attributes.get("sym_visibility")) == '"public"'
                )
                if not entries:
                    raise ValueError("STABLEHLO_PUBLIC_MAIN_MISSING")
                if len(entries) != 1:
                    raise ValueError("STABLEHLO_PUBLIC_MAIN_AMBIGUOUS")
                return cls(context, module, functions, entries[0])
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("STABLEHLO_INVALID") from error

    def public_main_operation_count(self, name: str) -> int:
        return sum(operation.name == name for operation in self.operations(self._public_main))

    def live_public_main_operation_count(self, name: str) -> int:
        return sum(
            operation.name == name
            and any(_result_reaches_return(result) for result in operation.results)
            for operation in self.operations(self._public_main)
        )

    def live_collective_counts(self) -> dict[str, int]:
        return {
            name: self.live_public_main_operation_count(f"stablehlo.{name}")
            for name in ("all_gather", "all_reduce", "reduce_scatter")
        }

    @property
    def functions(self) -> tuple[ir.Operation, ...]:
        return self._functions

    @property
    def public_main(self) -> ir.Operation:
        return self._public_main

    def operations(self, root: ir.Operation) -> tuple[ir.Operation, ...]:
        return _walk(root)

    def result_reaches_return(self, result: ir.Value) -> bool:
        return _result_reaches_return(result)

    def result_reaches_operation(self, result: ir.Value, target: ir.Operation) -> bool:
        return _result_reaches(result, lambda operation: operation == target)

    def result_reaches_return_avoiding(
        self,
        result: ir.Value,
        blocked: tuple[ir.Operation, ...],
    ) -> bool:
        return _result_reaches(
            result,
            lambda operation: operation.name == "func.return",
            blocked=blocked,
        )
