import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, bf16, f16, f32
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.distributed_tensor import MeshAttr, PendingReductionsAttr
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def test_distributed_matmul_verifies_and_hashes_stably() -> None:
    first = distributed_matmul_schedule()
    second = distributed_matmul_schedule()
    first.verify()
    assert canonical_module_text(first) == canonical_module_text(second)
    assert schedule_sha256(first) == schedule_sha256(second)


def test_missing_reduction_is_rejected() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    with pytest.raises(VerifyException, match="partially reduced"):
        builder.module(partial)


def test_incorrect_reduce_scatter_sharding_is_rejected() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    result = builder.reduce_scatter(
        partial,
        tensor(f32, (("M", 16), ("N", 16)), sharding={"M": ("t",)}),
        axes=("t",),
        scatter_dimensions=("N",),
    )
    with pytest.raises(VerifyException, match="incorrect sharding"):
        builder.module(result)


def test_unknown_mesh_axis_is_rejected() -> None:
    lhs = tensor(bf16, (("M", 16),), sharding={"M": ("missing",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs,))
    with pytest.raises(VerifyException, match="unknown mesh axis"):
        builder.module(builder.inputs[0])


def test_einsum_cannot_drop_noncontracted_sharding() -> None:
    lhs = tensor(
        bf16,
        (("M", 16), ("K", 32)),
        sharding={"M": ("d",), "K": ("t",)},
    )
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"d": 2, "t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    result = builder.reduce_scatter(
        partial,
        tensor(f32, (("M", 16), ("N", 16)), sharding={"N": ("t",)}),
        axes=("t",),
        scatter_dimensions=("N",),
    )
    with pytest.raises(VerifyException, match="preserve sharding"):
        builder.module(result)


def test_einsum_cannot_invent_noncontracted_sharding() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"d": 2, "t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(
            f32,
            (("M", 16), ("N", 16)),
            sharding={"M": ("d",)},
            pending_reductions={"t": "sum"},
        ),
        contracting_dimension="K",
    )
    result = builder.reduce_scatter(
        partial,
        tensor(
            f32,
            (("M", 16), ("N", 16)),
            sharding={"M": ("d",), "N": ("t",)},
        ),
        axes=("t",),
        scatter_dimensions=("N",),
    )
    with pytest.raises(VerifyException, match="preserve sharding"):
        builder.module(result)


def test_mesh_map_order_does_not_change_identity() -> None:
    input_spec = tensor(bf16, (("M", 16),), sharding={"M": ("d",)})
    first = DistributedProgramBuilder("identity", {"d": 2, "t": 4}, (input_spec,))
    second = DistributedProgramBuilder("identity", {"t": 4, "d": 2}, (input_spec,))
    assert schedule_sha256(first.module(first.inputs[0])) == schedule_sha256(
        second.module(second.inputs[0])
    )


def test_einsum_rejects_mismatched_local_contraction_extent() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)))
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    complete = builder.all_reduce(
        partial,
        tensor(f32, (("M", 16), ("N", 16))),
        axes=("t",),
    )
    with pytest.raises(VerifyException, match="identical local sharding"):
        builder.module(complete)


def test_einsum_rejects_unsupported_accumulation_type() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f16, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
        accumulation_type=f16,  # type: ignore[arg-type]
    )
    with pytest.raises(VerifyException, match="accumulation must be f32"):
        builder.module(partial)


def test_mesh_rejects_noncanonical_axis_order() -> None:
    with pytest.raises(VerifyException, match="canonical lexical order"):
        MeshAttr(
            ArrayAttr((StringAttr("t"), StringAttr("d"))),
            ArrayAttr((IntAttr(4), IntAttr(2))),
        )


def test_pending_reductions_reject_noncanonical_axis_order() -> None:
    with pytest.raises(VerifyException, match="canonical lexical order"):
        PendingReductionsAttr(
            ArrayAttr((StringAttr("t"), StringAttr("d"))),
            ArrayAttr((StringAttr("sum"), StringAttr("sum"))),
        )


def test_all_reduce_rejects_duplicate_axes() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    complete = builder.all_reduce(
        partial,
        tensor(f32, (("M", 16), ("N", 16))),
        axes=("t", "t"),
    )
    with pytest.raises(VerifyException, match="axes must be unique"):
        builder.module(complete)
