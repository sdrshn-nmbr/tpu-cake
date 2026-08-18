from __future__ import annotations

from collections import deque

import ml_dtypes
import numpy as np
from jax._src.interpreters import mlir
from jaxlib.mlir import ir

BF16_UNIT_ROUNDOFF = 2.0**-8
SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA = "bf16-forward-numerical-v1"
_REGION_TERMINATORS = frozenset({"sdy.return", "stablehlo.return"})


def _is_bf16_tensor(value: ir.Value) -> bool:
    value_type = value.type.maybe_downcast()
    return isinstance(value_type, ir.RankedTensorType) and isinstance(
        value_type.element_type.maybe_downcast(), ir.BF16Type
    )


def _as_operation(value: object) -> ir.Operation | None:
    if isinstance(value, ir.Operation):
        return value
    operation = getattr(value, "operation", None)
    return operation if isinstance(operation, ir.Operation) else None


def _result_reaches_function_return(result: ir.Value) -> bool:
    pending = deque([result])
    visited: set[ir.Value] = set()
    while pending:
        value = pending.popleft()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            consumer = _as_operation(use.owner)
            assert consumer is not None
            if consumer.name == "func.return":
                return True
            if consumer.name in _REGION_TERMINATORS:
                parent = _as_operation(consumer.parent)
                if parent is not None and use.operand_number < len(parent.results):
                    pending.append(parent.results[use.operand_number])
            pending.extend(consumer.results)
    return False


def _function_operations(function: ir.Operation) -> tuple[ir.Operation, ...]:
    operations: list[ir.Operation] = []

    def visit(operation: ir.Operation) -> None:
        operations.append(operation)
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    visit(function)
    return tuple(operations)


def validate_strict_silu_stablehlo(stablehlo: str, *, expected_count: int) -> None:
    if expected_count <= 0:
        raise ValueError("strict SiLU StableHLO expected count must be positive")
    try:
        with mlir.make_ir_context():
            module = ir.Module.parse(stablehlo)
            module.operation.verify()
            strict_chains = 0
            for top_level in module.body:
                if top_level.operation.name != "func.func":
                    continue
                operations = _function_operations(top_level.operation)
                silu_calls = tuple(
                    operation
                    for operation in operations
                    if operation.name == "func.call"
                    and str(operation.attributes["callee"]) == "@silu"
                )
                for silu_call in silu_calls:
                    if len(silu_call.operands) != 1 or len(silu_call.results) != 1:
                        raise ValueError("strict SiLU call must have one input and one result")
                    source = silu_call.operands[0]
                    result = silu_call.results[0]
                    if not _is_bf16_tensor(source) or not _is_bf16_tensor(result):
                        raise ValueError("strict SiLU call must use BF16 tensors")
                    input_barrier = _as_operation(source.owner)
                    if input_barrier is None or input_barrier.name != (
                        "stablehlo.optimization_barrier"
                    ):
                        raise ValueError("strict SiLU StableHLO is missing its input barrier")
                    if tuple(_as_operation(use.owner) for use in source.uses) != (silu_call,):
                        raise ValueError("strict SiLU input barrier must feed only its SiLU call")
                    result_uses = tuple(result.uses)
                    if (
                        len(result_uses) != 1
                        or _as_operation(result_uses[0].owner).name
                        != "stablehlo.optimization_barrier"
                    ):
                        raise ValueError("strict SiLU result must feed only its result barrier")
                    result_barrier = _as_operation(result_uses[0].owner)
                    assert result_barrier is not None
                    if len(result_barrier.results) != 1:
                        raise ValueError(
                            "strict SiLU StableHLO is missing its unique result barrier"
                        )
                    barrier_result = result_barrier.results[0]
                    barrier_uses = tuple(barrier_result.uses)
                    if (
                        len(barrier_uses) != 1
                        or _as_operation(barrier_uses[0].owner).name != "stablehlo.multiply"
                    ):
                        raise ValueError(
                            "strict SiLU result barrier must feed exactly one BF16 multiply"
                        )
                    multiply = _as_operation(barrier_uses[0].owner)
                    assert multiply is not None
                    if not all(_is_bf16_tensor(value) for value in multiply.operands):
                        raise ValueError(
                            "strict SiLU result barrier must feed exactly one BF16 multiply"
                        )
                    if len(multiply.results) != 1 or not _is_bf16_tensor(multiply.results[0]):
                        raise ValueError(
                            "strict SiLU result barrier must feed exactly one BF16 multiply"
                        )
                    if not _result_reaches_function_return(multiply.results[0]):
                        raise ValueError("strict SiLU BF16 multiply must reach its function return")
                    strict_chains += 1
            if strict_chains != expected_count:
                raise ValueError(
                    f"strict SiLU StableHLO expected {expected_count} calls, found {strict_chains}"
                )
    except ir.MLIRError as error:
        raise ValueError("strict SiLU StableHLO is not valid MLIR") from error


def rounded_mathematical_silu_bf16(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.dtype(ml_dtypes.bfloat16):
        raise TypeError("mathematical SiLU reference requires BF16 input")
    source = value.astype(np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("mathematical SiLU reference requires finite input")
    sigmoid = np.empty_like(source)
    nonnegative = source >= 0
    sigmoid[nonnegative] = 1.0 / (1.0 + np.exp(-source[nonnegative]))
    exponential = np.exp(source[~nonnegative])
    sigmoid[~nonnegative] = exponential / (1.0 + exponential)
    return np.asarray(source * sigmoid, dtype=ml_dtypes.bfloat16)
