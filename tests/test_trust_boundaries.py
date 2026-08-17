from __future__ import annotations

from collections.abc import Callable

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.ir import Operation
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.distributed_tensor import ReduceScatterOp
from tpu_cake.dialects.tpu_schedule import CollectiveReduceScatterOp
from tpu_cake.lowering import lower_distributed_matmul
from tpu_cake.source import verify_with_sources
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def _operation(module, operation_type: type[Operation]) -> Operation:
    return next(operation for operation in module.walk() if isinstance(operation, operation_type))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda operation: operation.properties.__setitem__(
            "axes", ArrayAttr((StringAttr("missing"),))
        ),
        lambda operation: operation.properties.__setitem__(
            "scatter_dimensions", ArrayAttr((StringAttr("missing"),))
        ),
        lambda operation: operation.properties.__setitem__("reducer", StringAttr("max")),
    ),
)
def test_distributed_program_mutations_fail_closed(
    mutate: Callable[[Operation], None],
) -> None:
    module = distributed_matmul_schedule()
    operation = _operation(module, ReduceScatterOp)
    mutate(operation)

    with pytest.raises(VerifyException):
        verify_with_sources(module)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda operation: operation.properties.__setitem__(
            "mesh_axis", StringAttr("missing")
        ),
        lambda operation: operation.properties.__setitem__("group_size", IntAttr(7)),
        lambda operation: operation.properties.__setitem__(
            "reducer", StringAttr("product")
        ),
    ),
)
def test_physical_schedule_mutations_fail_closed(
    mutate: Callable[[Operation], None],
) -> None:
    module = lower_distributed_matmul(distributed_matmul_schedule())
    operation = _operation(module, CollectiveReduceScatterOp)
    mutate(operation)

    with pytest.raises(VerifyException):
        verify_with_sources(module)
