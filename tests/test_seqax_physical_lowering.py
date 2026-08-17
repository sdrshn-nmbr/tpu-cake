from collections import Counter

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    KernelOp,
    MxuEinsumOp,
    VectorComputeOp,
)
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule

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


def test_scan_unrolling_derives_lifetimes_beyond_sixteen_layers() -> None:
    parameters = dict(SMALL_SEQAX)
    parameters["layers"] = 16
    physical = lower_seqax_forward_to_physical(
        seqax_forward_schedule(**parameters)
    ).module

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
