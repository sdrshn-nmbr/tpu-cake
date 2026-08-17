import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, bf16, f32
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import (
    CollectiveReduceScatterOp,
    KernelOp,
    MxuMatmulOp,
    TopologyAttr,
)
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.lowering import MatmulTile, UnsupportedLoweringError, lower_distributed_matmul
from tpu_cake.source import SourceLocation
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def test_distributed_matmul_lowers_to_verified_physical_schedule() -> None:
    schedule = lower_distributed_matmul(distributed_matmul_schedule())
    schedule.verify()
    operations = list(schedule.walk())
    assert sum(isinstance(operation, MxuMatmulOp) for operation in operations) == 1
    assert sum(isinstance(operation, CollectiveReduceScatterOp) for operation in operations) == 1
    matmul = next(operation for operation in operations if isinstance(operation, MxuMatmulOp))
    collective = next(
        operation for operation in operations if isinstance(operation, CollectiveReduceScatterOp)
    )
    assert str(matmul.location) == 'loc("tpu_cake/workloads/distributed_matmul.py":18:15)'
    assert str(collective.location) == 'loc("tpu_cake/workloads/distributed_matmul.py":26:14)'


def test_lowering_rejects_unsupported_collective_with_source() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("unsupported", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    result = builder.all_reduce(
        partial,
        tensor(f32, (("M", 16), ("N", 16))),
        axes=("t",),
        source=SourceLocation("seqax/model.py", 41, 9),
    )
    module = builder.module(result)
    with pytest.raises(
        UnsupportedLoweringError,
        match=r"dtensor.all_reduce at loc\(\"seqax/model.py\":41:9\)",
    ):
        lower_distributed_matmul(module)


def test_lowering_rejects_rank_three_einsum_at_source() -> None:
    lhs = tensor(
        bf16,
        (("B", 2), ("M", 16), ("K", 32)),
        sharding={"K": ("t",)},
    )
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("unsupported", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(
            f32,
            (("B", 2), ("M", 16), ("N", 16)),
            pending_reductions={"t": "sum"},
        ),
        contracting_dimension="K",
        source=SourceLocation("seqax/model.py", 52, 7),
    )
    result = builder.reduce_scatter(
        partial,
        tensor(
            f32,
            (("B", 2), ("M", 16), ("N", 16)),
            sharding={"N": ("t",)},
        ),
        axes=("t",),
        scatter_dimensions=("N",),
    )
    module = builder.module(result)
    with pytest.raises(
        UnsupportedLoweringError,
        match=r"rank-2.*dtensor.einsum_local at loc\(\"seqax/model.py\":52:7\)",
    ):
        lower_distributed_matmul(module)


@pytest.mark.parametrize(
    ("property_name", "replacement", "message"),
    (
        ("mesh_axis", StringAttr("missing"), "unknown mesh axis missing"),
        ("group_size", IntAttr(2), "group size must match"),
    ),
)
def test_physical_collective_must_match_kernel_mesh(
    property_name: str,
    replacement: StringAttr | IntAttr,
    message: str,
) -> None:
    schedule = lower_distributed_matmul(distributed_matmul_schedule())
    collective = next(
        operation
        for operation in schedule.walk()
        if isinstance(operation, CollectiveReduceScatterOp)
    )
    collective.properties[property_name] = replacement
    with pytest.raises(VerifyException, match=message):
        if property_name == "group_size":
            kernel = next(
                operation for operation in schedule.walk() if isinstance(operation, KernelOp)
            )
            kernel.verify_()
        else:
            schedule.verify()


def test_tile_must_divide_the_local_matmul_shape() -> None:
    with pytest.raises(VerifyException, match="tile dimensions must divide"):
        lower_distributed_matmul(
            distributed_matmul_schedule(mesh_size=4, m=256, k=512, n=256),
            tile=MatmulTile(96, 128),
        )


def test_physical_collective_requires_a_routed_topology_link() -> None:
    schedule = lower_distributed_matmul(distributed_matmul_schedule())
    kernel = next(operation for operation in schedule.walk() if isinstance(operation, KernelOp))
    assert kernel.topology is not None

    with pytest.raises(VerifyException, match="route references an unknown link"):
        TopologyAttr(
            kernel.topology.devices,
            ArrayAttr(()),
            kernel.topology.collective_plans,
        )
