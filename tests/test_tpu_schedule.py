import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, bf16, f32
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    CollectivePlanAttr,
    CollectiveReduceScatterOp,
    DmaStartOp,
    DmaWaitOp,
    KernelOp,
    LinkAttr,
    MemorySpace,
    MxuMatmulOp,
    Ownership,
    PipelineLoopOp,
    PipelineYieldOp,
    SemaphoreAllocOp,
    TopologyAttr,
    YieldOp,
    rectilinear_topology,
)
from tpu_cake.frontend import KernelBuilder, buffer
from tpu_cake.lowering import lower_distributed_matmul
from tpu_cake.workloads import inkling_rpa_schedule, matmul_schedule
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def test_vertical_workload_schedules_verify() -> None:
    matmul_schedule().verify()
    inkling_rpa_schedule().verify()


def test_rectilinear_topology_materializes_devices_links_and_collective_groups() -> None:
    topology = rectilinear_topology(
        ("d", "t"),
        (2, 2),
        {"d": 400_000_000_000, "t": 600_000_000_000},
    )

    assert len(topology.devices) == 4
    assert len(topology.links) == 4
    assert tuple(plan.plan_id.data for plan in topology.collective_plans) == (
        "axis:d",
        "axis:t",
    )
    assert all(len(plan.groups) == 2 for plan in topology.collective_plans)


def test_topology_rejects_links_to_unknown_devices() -> None:
    topology = rectilinear_topology(("t",), (2,), {"t": 600_000_000_000})
    invalid = LinkAttr(
        StringAttr("link:0-9"),
        IntAttr(0),
        IntAttr(9),
        IntAttr(600_000_000_000),
        IntAttr(1),
    )
    with pytest.raises(VerifyException, match="unknown device"):
        TopologyAttr(
            topology.devices,
            ArrayAttr((*topology.links, invalid)),
            topology.collective_plans,
        )


def test_collective_groups_must_follow_their_mesh_axis() -> None:
    topology = rectilinear_topology(
        ("d", "t"),
        (2, 2),
        {"d": 400_000_000_000, "t": 600_000_000_000},
    )
    d_plan, t_plan = topology.collective_plans
    swapped = TopologyAttr(
        topology.devices,
        topology.links,
        ArrayAttr(
            (
                CollectivePlanAttr(d_plan.plan_id, d_plan.mesh_axis, t_plan.groups),
                CollectivePlanAttr(t_plan.plan_id, t_plan.mesh_axis, d_plan.groups),
            )
        ),
    )
    builder = KernelBuilder(
        "bad_topology",
        "tpu7x",
        (),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        mesh={"d": 2, "t": 2},
        topology=swapped,
    )

    with pytest.raises(VerifyException, match="follow their declared mesh axis"):
        builder.module()


def test_new_schedules_use_only_the_structured_topology_schema() -> None:
    module = matmul_schedule()
    kernel = next(operation for operation in module.walk() if isinstance(operation, KernelOp))

    assert kernel.physical_schema is not None
    assert kernel.physical_schema.data == "structured-topology-v2"
    assert kernel.topology is not None
    assert kernel.interconnect is None


def test_collectives_cannot_overbook_one_physical_link() -> None:
    module = lower_distributed_matmul(distributed_matmul_schedule())
    kernel = next(operation for operation in module.walk() if isinstance(operation, KernelOp))
    collective = next(
        operation
        for operation in module.walk()
        if isinstance(operation, CollectiveReduceScatterOp)
    )
    duplicate = CollectiveReduceScatterOp(
        collective.source,
        collective.destination,
        stage=collective.stage.data,
        mesh_axis=collective.mesh_axis.data,
        group_size=collective.group_size.data,
        scatter_dimension=collective.scatter_dimension.data,
        reducer=collective.reducer.data,
    )
    assert collective.parent is not None
    collective.parent.insert_op_after(duplicate, collective)
    kernel.properties["ici_link_count"] = IntAttr(2)

    with pytest.raises(VerifyException, match="topology link .* capacity exceeded"):
        module.verify()


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


def _pipeline_matmul_kernel(
    *,
    initiation_interval: int,
    accumulator_rotations: int,
    vmem_capacity_bytes: int = 4096,
):
    lhs_external = buffer(
        (8, 8),
        "M K",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 3),
    )
    rhs_external = buffer(
        (8, 8),
        "K N",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 3),
    )
    accumulator_external = buffer(
        (8, 8),
        "M N",
        f32,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 3),
    )
    builder = KernelBuilder(
        "pipeline",
        "tpu7x",
        (lhs_external, rhs_external, accumulator_external),
        vmem_capacity_bytes=vmem_capacity_bytes,
        smem_capacity_bytes=1024,
        dma_engine_count=3,
        mxu_count=1,
    )
    local_specs = (
        buffer((8, 8), "M K", bf16, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        buffer((8, 8), "K N", bf16, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        buffer((8, 8), "M N", f32, memory=MemorySpace.VMEM, lifetime=(0, 2)),
    )
    captures = []
    transfers = []
    for index, (source, spec) in enumerate(
        zip(builder.inputs, local_specs, strict=True)
    ):
        destination = builder.alloc(spec, f"capture_{index}")
        transfers.append(
            builder.dma_start(source, destination, builder.semaphore(), stage=0)
        )
        captures.append(destination)
    for transfer in transfers:
        builder.dma_wait(transfer, stage=1)
    body = Block(arg_types=[capture.buffer.type for capture in captures])
    body.add_ops(
        [
            MxuMatmulOp(body.args[0], body.args[1], body.args[2], 0),
            MxuMatmulOp(body.args[0], body.args[1], body.args[2], 1),
            PipelineYieldOp(*body.args),
        ]
    )
    builder.block.add_op(
        PipelineLoopOp(
            tuple(captures),
            body,
            trip_count=4,
            initiation_interval=initiation_interval,
            pipeline_stages=3,
            rotation_counts=(1, 1, accumulator_rotations),
        )
    )
    return builder.module()


def test_pipeline_loop_proves_rotation_and_overlapped_resource_capacity() -> None:
    _pipeline_matmul_kernel(initiation_interval=2, accumulator_rotations=2).verify()
    with pytest.raises(VerifyException, match="needs 2 rotating buffers"):
        _pipeline_matmul_kernel(initiation_interval=2, accumulator_rotations=1)
    with pytest.raises(VerifyException, match="exceeds MXU capacity"):
        _pipeline_matmul_kernel(initiation_interval=1, accumulator_rotations=3)


def test_pipeline_rotation_banks_are_charged_to_vmem_capacity() -> None:
    with pytest.raises(
        VerifyException,
        match="VMEM capacity exceeded at stage 0: 768 > 600",
    ):
        _pipeline_matmul_kernel(
            initiation_interval=2,
            accumulator_rotations=2,
            vmem_capacity_bytes=600,
        )
