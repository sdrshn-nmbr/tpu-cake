import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, bf16
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    DmaStartOp,
    DmaWaitOp,
    KernelOp,
    MemorySpace,
    Ownership,
    SemaphoreAllocOp,
    YieldOp,
)
from tpu_cake.frontend import KernelBuilder, buffer
from tpu_cake.workloads import inkling_rpa_schedule, matmul_schedule


def test_vertical_workload_schedules_verify() -> None:
    matmul_schedule().verify()
    inkling_rpa_schedule().verify()


def test_buffer_rejects_mismatched_logical_rank() -> None:
    with pytest.raises(ValueError, match="equal rank"):
        buffer((16, 32), "M", bf16, memory=MemorySpace.VMEM)


def test_buffer_rejects_invalid_layout() -> None:
    with pytest.raises(ValueError, match="rank permutation"):
        buffer((16, 32), "M K", bf16, memory=MemorySpace.VMEM, layout=(0, 0))


def test_buffer_rejects_incorrect_numeric_logical_dimension() -> None:
    with pytest.raises(VerifyException, match="does not match physical size"):
        buffer((16, 32), "15 K", bf16, memory=MemorySpace.VMEM).to_type()


def _dma_kernel(
    *, first_lifetime: tuple[int, int], second_lifetime: tuple[int, int], capacity: int
) -> KernelOp:
    external = buffer(
        (16, 16),
        "M N",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    ).to_type()
    block = Block(arg_types=[external])
    first = AllocOp(
        buffer((16, 16), "M N", bf16, memory=MemorySpace.VMEM, lifetime=first_lifetime).to_type(),
        "first",
    )
    second = AllocOp(
        buffer((16, 16), "M N", bf16, memory=MemorySpace.VMEM, lifetime=second_lifetime).to_type(),
        "second",
    )
    first_semaphore = SemaphoreAllocOp()
    second_semaphore = SemaphoreAllocOp()
    first_dma = DmaStartOp(block.args[0], first, first_semaphore, 0)
    second_dma = DmaStartOp(block.args[0], second, second_semaphore, 1)
    block.add_ops(
        [
            first,
            second,
            first_semaphore,
            second_semaphore,
            first_dma,
            DmaWaitOp(first_dma, 0),
            second_dma,
            DmaWaitOp(second_dma, 1),
            YieldOp(),
        ]
    )
    return KernelOp(
        "lifetime",
        "tpu7x",
        capacity,
        1 << 16,
        ArrayAttr[StringAttr](()),
        ArrayAttr[IntAttr](()),
        Region(block),
    )


def test_peak_memory_allows_nonoverlapping_allocations() -> None:
    _dma_kernel(first_lifetime=(0, 0), second_lifetime=(1, 1), capacity=512).verify()


def test_peak_memory_rejects_overlapping_allocations() -> None:
    with pytest.raises(VerifyException, match="VMEM capacity exceeded at stage 0"):
        _dma_kernel(first_lifetime=(0, 1), second_lifetime=(0, 1), capacity=512).verify()


def test_use_outside_buffer_lifetime_is_rejected() -> None:
    with pytest.raises(VerifyException, match="outside lifetime"):
        _dma_kernel(first_lifetime=(1, 1), second_lifetime=(1, 1), capacity=1024).verify()


def _view_builder(*, capacity: int = 512) -> tuple[KernelBuilder, AllocOp]:
    base_spec = buffer(
        (16, 16),
        "M N",
        bf16,
        memory=MemorySpace.VMEM,
        lifetime=(0, 2),
    )
    builder = KernelBuilder(
        "views",
        "tpu7x",
        (),
        vmem_capacity_bytes=capacity,
        smem_capacity_bytes=1024,
    )
    return builder, builder.alloc(base_spec, "base")


def _tile(*, lifetime: tuple[int, int] = (0, 1)):
    return buffer(
        (8, 8),
        "m_tile n_tile",
        bf16,
        memory=MemorySpace.VMEM,
        lifetime=lifetime,
    )


def test_tile_views_alias_without_consuming_additional_capacity() -> None:
    builder, base = _view_builder(capacity=512)
    builder.view(base, _tile(), offsets=(0, 0), alias_group="read_window")
    builder.view(base, _tile(), offsets=(8, 8), alias_group="other_window")

    builder.module().verify()


def test_tile_view_rejects_out_of_bounds_region() -> None:
    builder, base = _view_builder()
    builder.view(base, _tile(), offsets=(12, 12), alias_group="bad")

    with pytest.raises(VerifyException, match="exceeds its base buffer bounds"):
        builder.module()


def test_tile_view_lifetime_must_fit_base_lifetime() -> None:
    builder, base = _view_builder()
    builder.view(base, _tile(lifetime=(1, 3)), offsets=(0, 0), alias_group="bad")

    with pytest.raises(VerifyException, match="lifetime must be contained"):
        builder.module()


def test_overlapping_live_views_need_one_alias_group() -> None:
    builder, base = _view_builder()
    builder.view(base, _tile(), offsets=(0, 0), alias_group="lhs")
    builder.view(base, _tile(), offsets=(4, 4), alias_group="rhs")

    with pytest.raises(VerifyException, match="same alias group"):
        builder.module()


def test_overlapping_views_in_one_declared_alias_group_verify() -> None:
    builder, base = _view_builder()
    builder.view(base, _tile(), offsets=(0, 0), alias_group="rotating_buffer")
    builder.view(base, _tile(), offsets=(4, 4), alias_group="rotating_buffer")

    builder.module().verify()
