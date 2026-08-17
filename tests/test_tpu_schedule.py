import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, bf16, f32
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


def test_kernel_rejects_more_inflight_dmas_than_declared_engines() -> None:
    external = buffer(
        (8, 8),
        "M N",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    )
    builder = KernelBuilder(
        "dma_pressure",
        "tpu7x",
        (external, external, external),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        dma_engine_count=2,
    )
    transfers = []
    for index, source in enumerate(builder.inputs):
        destination = builder.alloc(
            buffer(
                (8, 8),
                f"m{index} n{index}",
                bf16,
                memory=MemorySpace.VMEM,
                lifetime=(0, 1),
            ),
            f"tile_{index}",
        )
        semaphore = builder.semaphore()
        transfers.append(builder.dma_start(source, destination, semaphore, stage=0))
    for transfer in transfers:
        builder.dma_wait(transfer, stage=1)

    with pytest.raises(VerifyException, match="DMA engine capacity exceeded"):
        builder.module()


def test_consumer_cannot_read_a_tile_before_dma_wait() -> None:
    lhs_external = buffer(
        (8, 8),
        "M K",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    )
    rhs_external = buffer(
        (8, 8),
        "K N",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    )
    builder = KernelBuilder(
        "missing_wait",
        "tpu7x",
        (lhs_external, rhs_external),
        vmem_capacity_bytes=4096,
        smem_capacity_bytes=1024,
    )
    lhs = builder.alloc(
        buffer((8, 8), "M K", bf16, memory=MemorySpace.VMEM, lifetime=(0, 1)),
        "lhs",
    )
    rhs = builder.alloc(
        buffer((8, 8), "K N", bf16, memory=MemorySpace.VMEM, lifetime=(0, 1)),
        "rhs",
    )
    accumulator = builder.alloc(
        buffer(
            (8, 8),
            "M N",
            f32,
            memory=MemorySpace.VMEM,
            lifetime=(0, 1),
        ),
        "accumulator",
    )
    lhs_semaphore = builder.semaphore()
    rhs_semaphore = builder.semaphore()
    lhs_dma = builder.dma_start(builder.inputs[0], lhs, lhs_semaphore, stage=0)
    rhs_dma = builder.dma_start(builder.inputs[1], rhs, rhs_semaphore, stage=0)
    builder.matmul(lhs, rhs, accumulator, stage=0)
    builder.dma_wait(lhs_dma, stage=1)
    builder.dma_wait(rhs_dma, stage=1)

    with pytest.raises(VerifyException, match="before its producing operation completes"):
        builder.module()


def test_alias_group_cannot_hide_concurrent_overlapping_dma_writes() -> None:
    external = buffer(
        (8, 8),
        "X Y",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    )
    builder = KernelBuilder(
        "write_hazard",
        "tpu7x",
        (external, external),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
    )
    base = builder.alloc(
        buffer((16, 16), "M N", bf16, memory=MemorySpace.VMEM, lifetime=(0, 1)),
        "base",
    )
    first = builder.view(base, _tile(), offsets=(0, 0), alias_group="declared_alias")
    second = builder.view(base, _tile(), offsets=(4, 4), alias_group="declared_alias")
    first_dma = builder.dma_start(builder.inputs[0], first, builder.semaphore(), stage=0)
    second_dma = builder.dma_start(builder.inputs[1], second, builder.semaphore(), stage=0)
    builder.dma_wait(first_dma, stage=1)
    builder.dma_wait(second_dma, stage=1)

    with pytest.raises(VerifyException, match="concurrent DMA writes"):
        builder.module()


def test_nested_views_are_normalized_to_the_root_allocation() -> None:
    builder, base = _view_builder()
    parent = builder.view(base, _tile(), offsets=(0, 0), alias_group="outer")
    child = buffer(
        (4, 4),
        "child_m child_n",
        bf16,
        memory=MemorySpace.VMEM,
        lifetime=(0, 1),
    )
    builder.view(parent, child, offsets=(0, 0), alias_group="inner")

    with pytest.raises(VerifyException, match="same alias group"):
        builder.module()


def test_disjoint_strided_views_do_not_false_positive_as_overlapping() -> None:
    builder = KernelBuilder(
        "strided_views",
        "tpu7x",
        (),
        vmem_capacity_bytes=32,
        smem_capacity_bytes=32,
    )
    base = builder.alloc(
        buffer((16,), "M", bf16, memory=MemorySpace.VMEM, lifetime=(0, 0)),
        "base",
    )
    tile = buffer((8,), "tile", bf16, memory=MemorySpace.VMEM, lifetime=(0, 0))
    builder.view(base, tile, offsets=(0,), strides=(2,), alias_group="even")
    builder.view(base, tile, offsets=(1,), strides=(2,), alias_group="odd")

    builder.module().verify()


def test_capacity_sweep_includes_declared_lifetime_endpoints() -> None:
    builder = KernelBuilder(
        "late_lifetime",
        "tpu7x",
        (),
        vmem_capacity_bytes=512,
        smem_capacity_bytes=32,
    )
    for role in ("first", "second"):
        builder.alloc(
            buffer(
                (16, 16),
                f"{role}_m {role}_n",
                bf16,
                memory=MemorySpace.VMEM,
                lifetime=(5, 5),
            ),
            role,
        )

    with pytest.raises(VerifyException, match="VMEM capacity exceeded at stage 5"):
        builder.module()
