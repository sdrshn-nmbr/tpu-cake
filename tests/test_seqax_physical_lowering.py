from collections import Counter
from dataclasses import replace

import pytest
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    KernelOp,
    MxuEinsumOp,
    VectorComputeOp,
)
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.lowering import TPU7X_TARGET, UnsupportedLoweringError
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import SeqaxFeedForwardFusion, seqax_forward_schedule

SMALL_SEQAX = {
    "batch": 2,
    "sequence": 4,
    "model": 8,
    "vocabulary": 16,
    "feed_forward": 16,
    "query_groups": 2,
    "key_value_heads": 4,
    "head": 4,
    "layers": 2,
    "data_mesh": 2,
    "tensor_mesh": 4,
    "rope_max_timescale": 256,
}


def test_complete_seqax_forward_lowers_to_canonical_physical_schedule() -> None:
    distributed = seqax_forward_schedule(**SMALL_SEQAX)
    first = lower_seqax_forward_to_physical(distributed)
    second = lower_seqax_forward_to_physical(seqax_forward_schedule(**SMALL_SEQAX))

    first.module.verify()
    assert first.distributed_schedule_sha256 == schedule_sha256(distributed)
    assert first.physical_schedule_sha256 == schedule_sha256(first.module)
    assert first.physical_schedule_sha256 == second.physical_schedule_sha256
    assert canonical_module_text(first.module) == canonical_module_text(second.module)
    assert first.unrolled_layer_count == 2
    assert first.operation_count > 50

    inventory = Counter(operation.name for operation in first.module.walk())
    assert inventory["tpu_schedule.kernel"] == 1
    assert inventory["tpu_schedule.mxu_einsum"] == 8 * 2 + 1
    assert inventory["tpu_schedule.vector_compute"] > 30
    kernel = next(operation for operation in first.module.walk() if isinstance(operation, KernelOp))
    assert kernel.target.data == "tpu7x"
    assert any(isinstance(operation, MxuEinsumOp) for operation in kernel.body.block.ops)
    assert any(isinstance(operation, VectorComputeOp) for operation in kernel.body.block.ops)


def test_fused_silu_multiply_changes_the_executed_physical_schedule() -> None:
    separate = lower_seqax_forward_to_physical(seqax_forward_schedule(**SMALL_SEQAX)).module
    fused = lower_seqax_forward_to_physical(
        seqax_forward_schedule(
            **SMALL_SEQAX,
            feed_forward_fusion=SeqaxFeedForwardFusion.SILU_MULTIPLY,
        )
    ).module
    separate_functions = tuple(
        operation.function.data
        for operation in separate.walk()
        if isinstance(operation, VectorComputeOp)
    )
    fused_functions = tuple(
        operation.function.data
        for operation in fused.walk()
        if isinstance(operation, VectorComputeOp)
    )

    fused.verify()
    assert separate_functions.count("silu") == SMALL_SEQAX["layers"]
    assert separate_functions.count("multiply") == SMALL_SEQAX["layers"]
    assert fused_functions.count("silu_multiply") == SMALL_SEQAX["layers"]
    assert "silu" not in fused_functions
    assert "multiply" not in fused_functions
    assert len(fused_functions) + SMALL_SEQAX["layers"] == len(separate_functions)
    assert schedule_sha256(fused) != schedule_sha256(separate)


def test_scan_unrolling_derives_lifetimes_beyond_sixteen_layers() -> None:
    parameters = dict(SMALL_SEQAX)
    parameters["layers"] = 16
    physical = lower_seqax_forward_to_physical(seqax_forward_schedule(**parameters)).module

    physical.verify()
    allocations = tuple(
        operation for operation in physical.walk() if isinstance(operation, AllocOp)
    )
    lifetimes = tuple(
        operation.buffer.type.lifetime
        for operation in allocations
        if isinstance(operation.buffer.type, BufferType)
    )
    assert max(value.end.data for value in lifetimes) > 512
    assert all(value.start.data <= value.end.data for value in lifetimes)


def test_explicit_einsum_tiles_must_cover_every_region_exactly() -> None:
    distributed = seqax_forward_schedule(**SMALL_SEQAX)
    default = lower_seqax_forward_to_physical(distributed).module
    tiles = tuple(
        (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
        for operation in default.walk()
        if isinstance(operation, MxuEinsumOp)
    )

    with pytest.raises(UnsupportedLoweringError, match="missing an MXU einsum tile"):
        lower_seqax_forward_to_physical(distributed, einsum_tiles=tiles[:-1])
    with pytest.raises(UnsupportedLoweringError, match="unused MXU einsum tile"):
        lower_seqax_forward_to_physical(distributed, einsum_tiles=(*tiles, (1, 1, 1)))


def test_pallas_accumulator_scratch_is_charged_to_vmem_capacity() -> None:
    distributed = seqax_forward_schedule(**SMALL_SEQAX)
    constrained = replace(TPU7X_TARGET, vmem_capacity_bytes=2040)

    with pytest.raises(
        VerifyException,
        match="VMEM capacity exceeded at stage 33: 2168 > 2040",
    ):
        lower_seqax_forward_to_physical(distributed, target=constrained).module.verify()


def test_tpu_pallas_tiles_reject_compiler_illegal_partial_blocks() -> None:
    distributed = seqax_forward_schedule(**SMALL_SEQAX)
    default = lower_seqax_forward_to_physical(distributed).module
    tiles = tuple(
        (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
        for operation in default.walk()
        if isinstance(operation, MxuEinsumOp)
    )
    first_m, first_k, first_n = tiles[0]

    with pytest.raises(
        VerifyException,
        match="TPU Pallas tile M must span M or be divisible by 8",
    ):
        lower_seqax_forward_to_physical(
            distributed,
            einsum_tiles=((first_m // 2, first_k, first_n), *tiles[1:]),
        )
