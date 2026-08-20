from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from xdsl.dialects.builtin import BFloat16Type, Float16Type, Float32Type, ModuleOp, UnknownLoc
from xdsl.ir import Block, Operation, SSAValue

from tpu_cake.cost_model import HardwareRateModel, tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    CollectiveKind,
    CollectiveOp,
    CollectiveReduceScatterOp,
    DmaStartOp,
    DmaWaitOp,
    FusedRaggedPagedAttentionOp,
    KernelOp,
    MemorySpace,
    MxuEinsumOp,
    MxuMatmulOp,
    Ownership,
    PipelineLoopOp,
    PipelineYieldOp,
    RaggedPagedAttentionOp,
    RemoteDmaStartOp,
    RemoteDmaWaitOp,
    SemaphoreAllocOp,
    VectorComputeOp,
    ViewOp,
    YieldOp,
    buffer_bytes,
    mxu_accumulator_scratch_bytes,
    pipeline_resident_memory_bytes,
)
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)
from tpu_cake.physical_geometry import mxu_geometry

PHYSICAL_KERNEL_RESOURCE_SCHEMA = "physical-kernel-resource-v1"


class UnsupportedPhysicalCostModelError(ValueError):
    pass


class PhysicalMxuRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    region_index: int = Field(ge=0)
    operation_id: str = Field(min_length=1)
    operation: str = Field(pattern=r"^tpu_schedule\.mxu_(einsum|matmul)$")
    input_dtype: str = Field(pattern=r"^(bf16|f16|f32)$")
    source_location: str = Field(min_length=1)
    stage: int = Field(ge=0)
    executions: int = Field(gt=0)
    batch: int = Field(gt=0)
    m: int = Field(gt=0)
    k: int = Field(gt=0)
    n: int = Field(gt=0)
    tile_m: int = Field(gt=0)
    tile_k: int = Field(gt=0)
    tile_n: int = Field(gt=0)
    grid: tuple[int, int, int, int]
    tile_programs_per_execution: int = Field(gt=0)
    total_tile_programs: int = Field(gt=0)
    flops_per_execution: int = Field(gt=0)
    total_flops: int = Field(gt=0)
    accumulator_scratch_bytes: int = Field(ge=0)
    pending_reduction_axes: tuple[str, ...]

    @model_validator(mode="after")
    def geometry_is_consistent(self) -> PhysicalMxuRegion:
        if (
            self.grid
            != (
                self.batch,
                self.m // self.tile_m,
                self.n // self.tile_n,
                self.k // self.tile_k,
            )
            or self.tile_programs_per_execution != math.prod(self.grid)
            or self.total_tile_programs != self.tile_programs_per_execution * self.executions
            or self.flops_per_execution != 2 * self.batch * self.m * self.k * self.n
            or self.total_flops != self.flops_per_execution * self.executions
        ):
            raise ValueError("PHYSICAL_MXU_GEOMETRY_MISMATCH")
        return self


class PhysicalVectorWork(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    stage: int = Field(ge=0)
    function: str = Field(min_length=1)
    executions: int = Field(gt=0)
    total_output_elements: int = Field(gt=0)
    scalar_flops: int = Field(ge=0)
    special_function_ops: int = Field(ge=0)
    index_and_compare_ops: int = Field(ge=0)


class PhysicalMemoryLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    external_hbm_resident_bytes_per_device: int = Field(ge=0)
    external_hbm_input_footprint_bytes_per_device: int = Field(ge=0)
    external_hbm_output_footprint_bytes_per_device: int = Field(ge=0)
    explicit_hbm_dma_read_bytes_per_device: int = Field(ge=0)
    explicit_hbm_dma_write_bytes_per_device: int = Field(ge=0)
    explicit_local_dma_bytes_per_device: int = Field(ge=0)
    allocated_vmem_bytes_per_device: int = Field(ge=0)
    allocated_smem_bytes_per_device: int = Field(ge=0)
    peak_live_vmem_bytes_per_device: int = Field(ge=0)
    peak_live_vmem_scope: str = Field(min_length=1)
    peak_live_vmem_stage: int = Field(ge=0)
    peak_live_smem_bytes_per_device: int = Field(ge=0)
    peak_live_smem_scope: str = Field(min_length=1)
    peak_live_smem_stage: int = Field(ge=0)
    pipeline_rotation_vmem_bytes_per_device: int = Field(ge=0)
    pipeline_rotation_smem_bytes_per_device: int = Field(ge=0)


class PhysicalCollectiveTraffic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    kind: CollectiveKind
    stage: int = Field(ge=0)
    executions: int = Field(gt=0)
    mesh_axis: str = Field(min_length=1)
    group_size: int = Field(gt=1)
    payload_bytes_per_device: int = Field(gt=0)
    ring_equivalent_bidirectional_bytes_per_device: Decimal = Field(gt=0)
    total_ring_equivalent_bidirectional_bytes_per_device: Decimal = Field(gt=0)
    collective_plan: str
    route_link_ids: tuple[str, ...]
    minimum_participating_link_bandwidth_bytes_per_second: int | None = Field(default=None, gt=0)
    per_link_traffic_derivable: bool = False

    @model_validator(mode="after")
    def authority_is_explicit(self) -> PhysicalCollectiveTraffic:
        if self.per_link_traffic_derivable:
            raise ValueError("PHYSICAL_COLLECTIVE_LINK_TRAFFIC_IS_NOT_DERIVABLE")
        if (
            self.total_ring_equivalent_bidirectional_bytes_per_device
            != self.ring_equivalent_bidirectional_bytes_per_device * self.executions
        ):
            raise ValueError("PHYSICAL_COLLECTIVE_EXECUTION_COUNT_MISMATCH")
        return self


class PhysicalRemoteDmaTraffic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    stage: int = Field(ge=0)
    executions: int = Field(gt=0)
    transfer_plan: str = Field(min_length=1)
    payload_bytes_per_route: int = Field(gt=0)
    route_count: int = Field(gt=0)
    aggregate_link_bytes: int = Field(gt=0)
    route_ids: tuple[str, ...]
    route_link_ids: tuple[str, ...]
    bottleneck_bandwidth_bytes_per_second: int = Field(gt=0)


class PhysicalStageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: str = Field(min_length=1)
    stage: int = Field(ge=0)
    operation_ids: tuple[str, ...]
    active_dma: int = Field(ge=0)
    active_remote_dma: int = Field(ge=0)
    active_mxu: int = Field(ge=0)
    active_vector: int = Field(ge=0)
    active_ici: int = Field(ge=0)
    live_vmem_bytes_per_device: int = Field(ge=0)
    live_smem_bytes_per_device: int = Field(ge=0)
    link_channel_uses: tuple[tuple[str, int], ...]


class PhysicalResourcePeak(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: str = Field(min_length=1)
    observed: int = Field(ge=0)
    capacity: int = Field(gt=0)
    scope: str = Field(min_length=1)
    stage: int = Field(ge=0)

    @model_validator(mode="after")
    def capacity_is_respected(self) -> PhysicalResourcePeak:
        if self.observed > self.capacity:
            raise ValueError("PHYSICAL_RESOURCE_CAPACITY_EXCEEDED")
        return self


class PhysicalDeviceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: int = Field(ge=0)
    coordinates: tuple[int, ...]
    mxu_flops: int = Field(ge=0)
    vector_scalar_flops: int = Field(ge=0)
    hbm_dma_bytes: int = Field(ge=0)
    collective_ring_equivalent_bytes: Decimal = Field(ge=0)
    remote_dma_endpoint_bytes: int = Field(ge=0)


class PhysicalLinkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    link_id: str = Field(min_length=1)
    source_device: int = Field(ge=0)
    destination_device: int = Field(ge=0)
    bandwidth_bytes_per_second: int = Field(gt=0)
    channel_capacity: int = Field(gt=0)
    peak_concurrent_channels: int = Field(ge=0)
    collective_operation_ids: tuple[str, ...]
    collective_traffic_derivable: bool = False
    remote_dma_operation_ids: tuple[str, ...]
    exact_remote_dma_link_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def link_claims_are_consistent(self) -> PhysicalLinkRecord:
        if self.collective_traffic_derivable:
            raise ValueError("PHYSICAL_COLLECTIVE_LINK_TRAFFIC_IS_NOT_DERIVABLE")
        if self.peak_concurrent_channels > self.channel_capacity:
            raise ValueError("PHYSICAL_LINK_CHANNEL_CAPACITY_EXCEEDED")
        return self


class PhysicalImbalanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: str = Field(min_length=1)
    minimum: Decimal = Field(ge=0)
    maximum: Decimal = Field(ge=0)
    maximum_to_minimum_ratio: Decimal | None = Field(default=None, ge=1)
    derivable: bool
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def ratio_is_consistent(self) -> PhysicalImbalanceRecord:
        expected = None if self.minimum == 0 else self.maximum / self.minimum
        if self.maximum < self.minimum or self.maximum_to_minimum_ratio != expected:
            raise ValueError("PHYSICAL_IMBALANCE_RATIO_MISMATCH")
        if self.derivable != (expected is not None):
            raise ValueError("PHYSICAL_IMBALANCE_DERIVABILITY_MISMATCH")
        return self


class PhysicalKernelResourceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^physical-kernel-resource-v1$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kernel_name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    topology_authority: str = Field(min_length=1)
    execution_authority: str = Field(pattern=r"^static-declared-schedule-only$")
    mesh_axes: tuple[tuple[str, int], ...]
    device_count: int = Field(gt=0)
    canonical_operation_inventory: tuple[tuple[str, int], ...]
    operation_executions: tuple[tuple[str, int], ...]
    hardware: HardwareRateModel
    mxu_regions: tuple[PhysicalMxuRegion, ...]
    vector_work: tuple[PhysicalVectorWork, ...]
    memory: PhysicalMemoryLedger
    collectives: tuple[PhysicalCollectiveTraffic, ...]
    remote_dmas: tuple[PhysicalRemoteDmaTraffic, ...]
    stages: tuple[PhysicalStageRecord, ...]
    resource_peaks: tuple[PhysicalResourcePeak, ...]
    devices: tuple[PhysicalDeviceRecord, ...]
    links: tuple[PhysicalLinkRecord, ...]
    imbalance: tuple[PhysicalImbalanceRecord, ...]
    priced_compute_time_floor_ns: Decimal = Field(ge=0)
    priced_hbm_time_floor_ns: Decimal = Field(ge=0)
    collective_ring_equivalent_time_scenario_ns: Decimal = Field(ge=0)
    remote_dma_exact_endpoint_time_floor_ns: Decimal = Field(ge=0)
    remote_dma_exact_link_time_floor_ns: Decimal = Field(ge=0)
    combined_ici_injection_time_scenario_ns: Decimal = Field(ge=0)
    priced_ici_time_scenario_ns: Decimal = Field(ge=0)
    priced_overlapped_resource_scenario_ns: Decimal = Field(ge=0)
    priced_serial_resource_scenario_ns: Decimal = Field(ge=0)
    predicted_limiting_priced_resource: str = Field(pattern=r"^(none|compute|hbm|ici)$")
    unpriced_work: tuple[str, ...]
    assumptions: tuple[str, ...]
    omissions: tuple[str, ...]
    metrics: tuple[Metric, ...]


def _location(operation: Operation) -> str:
    return "unknown" if isinstance(operation.location, UnknownLoc) else str(operation.location)


def _root(value: SSAValue) -> SSAValue:
    while isinstance(value.owner, ViewOp):
        value = value.owner.base
    return value


def _operation_stage(operation: Operation) -> int | None:
    stage = getattr(operation, "stage", None)
    return None if stage is None else stage.data


def _mxu_input_dtype(operation: MxuMatmulOp | MxuEinsumOp) -> str:
    lhs = operation.lhs.type
    assert isinstance(lhs, BufferType)
    element_type = lhs.storage.element_type
    if isinstance(element_type, BFloat16Type):
        return "bf16"
    if isinstance(element_type, Float16Type):
        return "f16"
    if isinstance(element_type, Float32Type):
        return "f32"
    raise UnsupportedPhysicalCostModelError(f"unsupported MXU input dtype {element_type}")


def _wait_stage(start: DmaStartOp | RemoteDmaStartOp) -> int:
    expected = DmaWaitOp if isinstance(start, DmaStartOp) else RemoteDmaWaitOp
    waits = tuple(use.operation for use in start.token.uses if isinstance(use.operation, expected))
    if len(waits) != 1:
        raise UnsupportedPhysicalCostModelError(f"{start.name} must have exactly one matching wait")
    return waits[0].stage.data


def _configuration(operation: VectorComputeOp) -> dict[str, str]:
    return dict(value.data.split("=", 1) for value in operation.configuration)


def _vector_counts(operation: VectorComputeOp) -> tuple[int, int, int, int]:
    output = operation.output.type
    assert isinstance(output, BufferType)
    elements = math.prod(output.storage.get_shape())
    function = operation.function.data
    configuration = _configuration(operation)
    scalar = special = index = 0
    if function in {"add", "multiply"}:
        scalar = elements
    elif function == "silu":
        scalar = 4 * elements
        special = elements
    elif function == "silu_multiply":
        scalar = 5 * elements
        special = elements
    elif function == "exp":
        special = elements
    elif function == "rms_norm":
        names = tuple(value.data for value in output.shape.dimensions)
        dimension = configuration["dimension"]
        if dimension not in names:
            raise UnsupportedPhysicalCostModelError(
                "physical RMSNorm dimension is absent from its output"
            )
        rows = elements // output.storage.get_shape()[names.index(dimension)]
        scalar = 4 * elements
        special = rows
    elif function == "rms_norm_partial":
        value = operation.inputs[0].type
        assert isinstance(value, BufferType)
        scalar = 2 * math.prod(value.storage.get_shape())
    elif function == "rms_norm_apply":
        statistics = operation.inputs[1].type
        assert isinstance(statistics, BufferType)
        scalar = 2 * elements
        special = math.prod(statistics.storage.get_shape())
    elif function == "rotary_embedding":
        scalar = 3 * elements
        special = elements
    elif function == "masked_softmax":
        names = tuple(value.data for value in output.shape.dimensions)
        dimension = configuration["dimension"]
        if dimension not in names:
            raise UnsupportedPhysicalCostModelError(
                "physical softmax dimension is absent from its output"
            )
        rows = elements // output.storage.get_shape()[names.index(dimension)]
        scalar = 3 * elements - rows
        special = elements
        index = elements
    elif function in {
        "cast",
        "embedding_lookup",
        "packed_causal_mask",
        "rename_dimension",
        "slice",
    }:
        index = elements
    else:
        raise UnsupportedPhysicalCostModelError(
            f"no physical vector work convention for {function!r}"
        )
    return elements, scalar, special, index


def _collective_kind(operation: CollectiveOp | CollectiveReduceScatterOp) -> CollectiveKind:
    if isinstance(operation, CollectiveReduceScatterOp):
        return CollectiveKind.REDUCE_SCATTER
    return operation.kind.data


def _collective_bytes(
    operation: CollectiveOp | CollectiveReduceScatterOp,
) -> tuple[int, Decimal]:
    source = operation.source.type
    destination = operation.destination.type
    assert isinstance(source, BufferType) and isinstance(destination, BufferType)
    source_bytes = buffer_bytes(source)
    destination_bytes = buffer_bytes(destination)
    group_size = operation.group_size.data
    kind = _collective_kind(operation)
    if kind is CollectiveKind.ALL_GATHER:
        payload = destination_bytes
        traffic = Decimal(2 * destination_bytes * (group_size - 1)) / Decimal(group_size)
    elif kind is CollectiveKind.REDUCE_SCATTER:
        payload = source_bytes
        traffic = Decimal(2 * source_bytes * (group_size - 1)) / Decimal(group_size)
    elif kind is CollectiveKind.ALL_REDUCE:
        payload = source_bytes
        traffic = Decimal(4 * source_bytes * (group_size - 1)) / Decimal(group_size)
    elif kind is CollectiveKind.ALL_TO_ALL:
        payload = source_bytes
        traffic = Decimal(2 * source_bytes * (group_size - 1)) / Decimal(group_size)
    else:
        raise UnsupportedPhysicalCostModelError(f"unsupported collective kind {kind}")
    return payload, traffic


def _operation_identities(module: ModuleOp) -> dict[Operation, str]:
    return {
        operation: f"{index:04d}:{operation.name}" for index, operation in enumerate(module.walk())
    }


def _kernel(module: ModuleOp) -> KernelOp:
    module.verify()
    operations = tuple(module.body.block.ops)
    if len(operations) != 1 or not isinstance(operations[0], KernelOp):
        raise UnsupportedPhysicalCostModelError(
            "physical cost model expects one top-level TPU kernel"
        )
    return operations[0]


def _executed_leaf_operations(
    block: Block,
    *,
    multiplier: int = 1,
) -> tuple[tuple[Operation, int], ...]:
    result: list[tuple[Operation, int]] = []
    for operation in block.ops:
        if isinstance(operation, PipelineLoopOp):
            result.extend(
                _executed_leaf_operations(
                    operation.body.block,
                    multiplier=multiplier * operation.trip_count.data,
                )
            )
        elif not isinstance(
            operation,
            (
                AllocOp,
                ViewOp,
                SemaphoreAllocOp,
                DmaWaitOp,
                RemoteDmaWaitOp,
                PipelineYieldOp,
                YieldOp,
            ),
        ):
            result.append((operation, multiplier))
    return tuple(result)


def _rotation_copies(kernel: KernelOp) -> tuple[tuple[BufferType, int], ...]:
    copies: list[tuple[BufferType, int]] = []
    for operation in kernel.body.block.ops:
        if not isinstance(operation, PipelineLoopOp):
            continue
        roots = tuple(_root(value) for value in operation.captures)
        for value, rotation in zip(roots, operation.rotation_counts, strict=True):
            buffer = value.type
            assert isinstance(buffer, BufferType)
            if buffer.ownership.data is not Ownership.EXTERNAL and rotation.data > 1:
                copies.append((buffer, rotation.data - 1))
    return tuple(copies)


def _storage_buffers(kernel: KernelOp) -> tuple[BufferType, ...]:
    buffers = [value.type for value in kernel.body.block.args if isinstance(value.type, BufferType)]
    buffers.extend(
        operation.buffer.type
        for operation in kernel.body.block.ops
        if isinstance(operation, AllocOp) and isinstance(operation.buffer.type, BufferType)
    )
    return tuple(buffers)


def _link_uses(
    operation: CollectiveOp | CollectiveReduceScatterOp | RemoteDmaStartOp,
    kernel: KernelOp,
) -> tuple[str, ...]:
    if kernel.topology is None:
        return ()
    if isinstance(operation, RemoteDmaStartOp):
        plan = kernel.topology.transfer_plans_by_id()[operation.transfer_plan.data]
        return tuple(sorted(link.data for route in plan.routes for link in route.route_link_ids))
    plan_value = operation.collective_plan
    if plan_value is None:
        return ()
    plan = kernel.topology.plans_by_id()[plan_value.data]
    return tuple(sorted(link.data for group in plan.groups for link in group.route_link_ids))


def _top_level_stages(
    kernel: KernelOp,
    operation_ids: dict[Operation, str],
) -> tuple[PhysicalStageRecord, ...]:
    operations = tuple(kernel.body.block.ops)
    local_buffers = tuple(
        buffer
        for buffer in _storage_buffers(kernel)
        if buffer.ownership.data is not Ownership.EXTERNAL
    )
    rotation_copies = _rotation_copies(kernel)
    max_stage = max(
        (
            *(_operation_stage(operation) or 0 for operation in operations),
            *(buffer.lifetime.end.data for buffer in local_buffers),
        ),
        default=0,
    )
    records: list[PhysicalStageRecord] = []
    for stage in range(max_stage + 1):
        scheduled = tuple(
            operation for operation in operations if _operation_stage(operation) == stage
        )
        active_dma = tuple(
            operation
            for operation in operations
            if isinstance(operation, DmaStartOp)
            and operation.stage.data <= stage <= _wait_stage(operation)
        )
        active_remote = tuple(
            operation
            for operation in operations
            if isinstance(operation, RemoteDmaStartOp)
            and operation.stage.data <= stage <= _wait_stage(operation)
        )
        mxu = sum(isinstance(operation, (MxuMatmulOp, MxuEinsumOp)) for operation in scheduled)
        vector = sum(isinstance(operation, VectorComputeOp) for operation in scheduled)
        collectives = tuple(
            operation
            for operation in scheduled
            if isinstance(operation, (CollectiveOp, CollectiveReduceScatterOp))
        )
        link_counts: Counter[str] = Counter()
        for operation in (*active_remote, *collectives):
            link_counts.update(_link_uses(operation, kernel))
        live_by_space = {
            space: sum(
                buffer_bytes(buffer)
                for buffer in local_buffers
                if buffer.space.data is space
                and buffer.lifetime.start.data <= stage <= buffer.lifetime.end.data
            )
            + sum(
                buffer_bytes(buffer) * copies
                for buffer, copies in rotation_copies
                if buffer.space.data is space
                and buffer.lifetime.start.data <= stage <= buffer.lifetime.end.data
            )
            for space in (MemorySpace.VMEM, MemorySpace.SMEM)
        }
        live_by_space[MemorySpace.VMEM] += sum(
            mxu_accumulator_scratch_bytes(operation)
            for operation in scheduled
            if isinstance(operation, (MxuMatmulOp, MxuEinsumOp))
        )
        records.append(
            PhysicalStageRecord(
                scope="kernel",
                stage=stage,
                operation_ids=tuple(operation_ids[operation] for operation in scheduled),
                active_dma=len(active_dma),
                active_remote_dma=len(active_remote),
                active_mxu=mxu,
                active_vector=vector,
                active_ici=len(collectives),
                live_vmem_bytes_per_device=live_by_space[MemorySpace.VMEM],
                live_smem_bytes_per_device=live_by_space[MemorySpace.SMEM],
                link_channel_uses=tuple(sorted(link_counts.items())),
            )
        )
    return tuple(records)


def _pipeline_stages(
    kernel: KernelOp,
    operation_ids: dict[Operation, str],
) -> tuple[PhysicalStageRecord, ...]:
    records: list[PhysicalStageRecord] = []
    for pipeline in (
        operation for operation in kernel.body.block.ops if isinstance(operation, PipelineLoopOp)
    ):
        operations = tuple(pipeline.body.block.ops)
        horizon = (
            pipeline.trip_count.data - 1
        ) * pipeline.initiation_interval.data + pipeline.pipeline_stages.data
        resident_bytes = pipeline_resident_memory_bytes(kernel)
        for absolute_stage in range(horizon):
            operation_labels: list[str] = []
            active_dma = active_remote = active_mxu = active_vector = active_ici = 0
            accumulator_scratch_bytes = 0
            link_counts: Counter[str] = Counter()
            for iteration in range(pipeline.trip_count.data):
                logical_stage = absolute_stage - iteration * pipeline.initiation_interval.data
                if logical_stage < 0 or logical_stage >= pipeline.pipeline_stages.data:
                    continue
                for operation in operations:
                    operation_stage = _operation_stage(operation)
                    if isinstance(operation, DmaStartOp) and (
                        operation.stage.data <= logical_stage <= _wait_stage(operation)
                    ):
                        active_dma += 1
                    if isinstance(operation, RemoteDmaStartOp) and (
                        operation.stage.data <= logical_stage <= _wait_stage(operation)
                    ):
                        active_remote += 1
                        link_counts.update(_link_uses(operation, kernel))
                    if operation_stage != logical_stage:
                        continue
                    operation_labels.append(f"{operation_ids[operation]}@{iteration}")
                    active_mxu += isinstance(operation, (MxuMatmulOp, MxuEinsumOp))
                    if isinstance(operation, (MxuMatmulOp, MxuEinsumOp)):
                        accumulator_scratch_bytes += mxu_accumulator_scratch_bytes(operation)
                    active_vector += isinstance(operation, VectorComputeOp)
                    if isinstance(operation, (CollectiveOp, CollectiveReduceScatterOp)):
                        active_ici += 1
                        link_counts.update(_link_uses(operation, kernel))
            records.append(
                PhysicalStageRecord(
                    scope=operation_ids[pipeline],
                    stage=absolute_stage,
                    operation_ids=tuple(operation_labels),
                    active_dma=active_dma,
                    active_remote_dma=active_remote,
                    active_mxu=active_mxu,
                    active_vector=active_vector,
                    active_ici=active_ici,
                    live_vmem_bytes_per_device=(
                        resident_bytes[MemorySpace.VMEM] + accumulator_scratch_bytes
                    ),
                    live_smem_bytes_per_device=resident_bytes[MemorySpace.SMEM],
                    link_channel_uses=tuple(sorted(link_counts.items())),
                )
            )
    return tuple(records)


def _peak(
    stages: tuple[PhysicalStageRecord, ...],
    *,
    resource: str,
    capacity: int,
) -> PhysicalResourcePeak:
    field = {
        "dma": "active_dma",
        "remote_dma": "active_remote_dma",
        "mxu": "active_mxu",
        "vector": "active_vector",
        "ici": "active_ici",
    }[resource]
    record = max(stages, key=lambda value: getattr(value, field))
    return PhysicalResourcePeak(
        resource=resource,
        observed=getattr(record, field),
        capacity=capacity,
        scope=record.scope,
        stage=record.stage,
    )


def _metric(
    name: str,
    value: Decimal | int,
    unit: Unit,
    source: MetricSource,
    formula_name: str,
    expression: str,
    *,
    scope: str = "one declared physical kernel invocation on one TPU device",
    numerator: Quantity | None = None,
    denominator: Quantity | None = None,
) -> Metric:
    return Metric(
        name=name,
        quantity=Quantity(value=Decimal(value), unit=unit),
        kind=MeasurementKind.ESTIMATED,
        interval=MeasurementInterval(scope=scope),
        sources=(source,),
        formula=FormulaIdentity(name=formula_name, version="1", expression=expression),
        numerator=numerator,
        denominator=denominator,
    )


def _imbalance(resource: str, values: tuple[Decimal, ...], reason: str) -> PhysicalImbalanceRecord:
    minimum = min(values)
    maximum = max(values)
    ratio = None if minimum == 0 else maximum / minimum
    return PhysicalImbalanceRecord(
        resource=resource,
        minimum=minimum,
        maximum=maximum,
        maximum_to_minimum_ratio=ratio,
        derivable=ratio is not None,
        reason=reason,
    )


def analyze_physical_kernel(
    module: ModuleOp,
    *,
    hardware: HardwareRateModel,
    expected_schedule_sha256: str | None = None,
) -> PhysicalKernelResourceReport:
    kernel = _kernel(module)
    if kernel.target.data != "tpu7x" or hardware != tpu7x_tensorcore_rates():
        raise UnsupportedPhysicalCostModelError(
            "physical-kernel-resource-v1 requires the exact TPU7x hardware rate authority"
        )
    physical_hash = schedule_sha256(module)
    source = physical_schedule_source(module)
    if expected_schedule_sha256 is not None and expected_schedule_sha256 != physical_hash:
        raise UnsupportedPhysicalCostModelError(
            "physical schedule hash does not match the expected schedule"
        )
    if kernel.argument_modes is None:
        raise UnsupportedPhysicalCostModelError(
            "physical cost model requires explicit kernel argument modes"
        )
    if kernel.topology is None or kernel.topology_authority is None:
        raise UnsupportedPhysicalCostModelError(
            "physical cost model requires structured topology authority"
        )
    operation_ids = _operation_identities(module)
    executed = _executed_leaf_operations(kernel.body.block)
    unsupported = tuple(
        operation
        for operation, _ in executed
        if isinstance(operation, (RaggedPagedAttentionOp, FusedRaggedPagedAttentionOp))
    )
    if unsupported:
        raise UnsupportedPhysicalCostModelError(
            f"physical cost model has no work convention for {unsupported[0].name}"
        )
    supported = (
        DmaStartOp,
        RemoteDmaStartOp,
        MxuMatmulOp,
        MxuEinsumOp,
        VectorComputeOp,
        CollectiveOp,
        CollectiveReduceScatterOp,
    )
    unknown = tuple(operation for operation, _ in executed if not isinstance(operation, supported))
    if unknown:
        raise UnsupportedPhysicalCostModelError(
            f"physical cost model does not account for {unknown[0].name}"
        )

    mesh = tuple(
        (name.data, size.data)
        for name, size in zip(
            kernel.mesh_axis_names,
            kernel.mesh_axis_sizes,
            strict=True,
        )
    )
    device_count = math.prod(size for _, size in mesh)
    static_inventory = tuple(sorted(Counter(op.name for op in module.walk()).items()))
    execution_inventory: Counter[str] = Counter(
        {
            "tpu_schedule.pipeline_loop": sum(
                isinstance(operation, PipelineLoopOp) for operation in kernel.body.block.ops
            )
        }
    )
    for operation, executions in executed:
        execution_inventory[operation.name] += executions

    mxu_regions: list[PhysicalMxuRegion] = []
    vector_work: list[PhysicalVectorWork] = []
    hbm_dma_read = hbm_dma_write = local_dma = 0
    collective_records: list[PhysicalCollectiveTraffic] = []
    remote_records: list[PhysicalRemoteDmaTraffic] = []
    remote_link_bytes: Counter[str] = Counter()
    remote_link_operations: dict[str, set[str]] = {}
    collective_link_operations: dict[str, set[str]] = {}
    remote_endpoint_bytes = [0 for _ in range(device_count)]
    links_by_id = kernel.topology.links_by_id()
    plans_by_id = kernel.topology.plans_by_id()
    transfers_by_id = kernel.topology.transfer_plans_by_id()

    for operation, executions in executed:
        operation_id = operation_ids[operation]
        if isinstance(operation, DmaStartOp):
            source_type = operation.source.type
            destination_type = operation.destination.type
            assert isinstance(source_type, BufferType) and isinstance(destination_type, BufferType)
            byte_count = buffer_bytes(source_type) * executions
            if source_type.space.data is MemorySpace.HBM:
                hbm_dma_read += byte_count
            elif destination_type.space.data is MemorySpace.HBM:
                hbm_dma_write += byte_count
            else:
                local_dma += byte_count
        elif isinstance(operation, RemoteDmaStartOp):
            source_type = operation.source.type
            assert isinstance(source_type, BufferType)
            payload = buffer_bytes(source_type)
            plan = transfers_by_id[operation.transfer_plan.data]
            route_ids = tuple(route.route_id.data for route in plan.routes)
            route_links = tuple(
                sorted({link.data for route in plan.routes for link in route.route_link_ids})
            )
            aggregate = 0
            for route in plan.routes:
                route_bytes = payload * executions
                remote_endpoint_bytes[route.source_device.data] += route_bytes
                remote_endpoint_bytes[route.destination_device.data] += route_bytes
                for link in route.route_link_ids:
                    remote_link_bytes[link.data] += route_bytes
                    remote_link_operations.setdefault(link.data, set()).add(operation_id)
                    aggregate += route_bytes
            remote_records.append(
                PhysicalRemoteDmaTraffic(
                    operation_id=operation_id,
                    source_location=_location(operation),
                    stage=operation.stage.data,
                    executions=executions,
                    transfer_plan=operation.transfer_plan.data,
                    payload_bytes_per_route=payload,
                    route_count=len(plan.routes),
                    aggregate_link_bytes=aggregate,
                    route_ids=route_ids,
                    route_link_ids=route_links,
                    bottleneck_bandwidth_bytes_per_second=min(
                        links_by_id[link].bandwidth_bytes_per_second.data for link in route_links
                    ),
                )
            )
        elif isinstance(operation, (MxuMatmulOp, MxuEinsumOp)):
            geometry = mxu_geometry(operation)
            pending = (
                tuple(value.data for value in operation.pending_reduction_axes)
                if isinstance(operation, MxuEinsumOp)
                else ()
            )
            mxu_regions.append(
                PhysicalMxuRegion(
                    region_index=len(mxu_regions),
                    operation_id=operation_id,
                    operation=operation.name,
                    input_dtype=_mxu_input_dtype(operation),
                    source_location=_location(operation),
                    stage=operation.stage.data,
                    executions=executions,
                    batch=geometry.batch,
                    m=geometry.m,
                    k=geometry.k,
                    n=geometry.n,
                    tile_m=geometry.tile_m,
                    tile_k=geometry.tile_k,
                    tile_n=geometry.tile_n,
                    grid=geometry.grid,
                    tile_programs_per_execution=geometry.tile_program_count,
                    total_tile_programs=geometry.tile_program_count * executions,
                    flops_per_execution=geometry.flops,
                    total_flops=geometry.flops * executions,
                    accumulator_scratch_bytes=mxu_accumulator_scratch_bytes(operation),
                    pending_reduction_axes=pending,
                )
            )
        elif isinstance(operation, VectorComputeOp):
            elements, scalar, special, index = _vector_counts(operation)
            vector_work.append(
                PhysicalVectorWork(
                    operation_id=operation_id,
                    source_location=_location(operation),
                    stage=operation.stage.data,
                    function=operation.function.data,
                    executions=executions,
                    total_output_elements=elements * executions,
                    scalar_flops=scalar * executions,
                    special_function_ops=special * executions,
                    index_and_compare_ops=index * executions,
                )
            )
        elif isinstance(operation, (CollectiveOp, CollectiveReduceScatterOp)):
            payload, traffic = _collective_bytes(operation)
            plan_id = "" if operation.collective_plan is None else operation.collective_plan.data
            route_links: tuple[str, ...] = ()
            bottleneck = None
            if plan_id:
                plan = plans_by_id[plan_id]
                route_links = tuple(
                    sorted({link.data for group in plan.groups for link in group.route_link_ids})
                )
                bottleneck = min(
                    links_by_id[link].bandwidth_bytes_per_second.data for link in route_links
                )
                for link in route_links:
                    collective_link_operations.setdefault(link, set()).add(operation_id)
            collective_records.append(
                PhysicalCollectiveTraffic(
                    operation_id=operation_id,
                    source_location=_location(operation),
                    kind=_collective_kind(operation),
                    stage=operation.stage.data,
                    executions=executions,
                    mesh_axis=operation.mesh_axis.data,
                    group_size=operation.group_size.data,
                    payload_bytes_per_device=payload,
                    ring_equivalent_bidirectional_bytes_per_device=traffic,
                    total_ring_equivalent_bidirectional_bytes_per_device=traffic * executions,
                    collective_plan=plan_id,
                    route_link_ids=route_links,
                    minimum_participating_link_bandwidth_bytes_per_second=bottleneck,
                    per_link_traffic_derivable=False,
                )
            )

    storage = _storage_buffers(kernel)
    modes = tuple(value.data for value in kernel.argument_modes)
    arguments = tuple(kernel.body.block.args)
    external_resident = sum(
        buffer_bytes(argument.type)
        for argument in arguments
        if isinstance(argument.type, BufferType)
    )
    external_input_footprint = sum(
        buffer_bytes(argument.type)
        for argument, mode in zip(arguments, modes, strict=True)
        if mode in {"input", "inout"} and isinstance(argument.type, BufferType)
    )
    external_output_footprint = sum(
        buffer_bytes(argument.type)
        for argument, mode in zip(arguments, modes, strict=True)
        if mode in {"output", "inout"} and isinstance(argument.type, BufferType)
    )
    allocated_vmem = sum(
        buffer_bytes(buffer)
        for buffer in storage
        if buffer.ownership.data is not Ownership.EXTERNAL and buffer.space.data is MemorySpace.VMEM
    )
    allocated_smem = sum(
        buffer_bytes(buffer)
        for buffer in storage
        if buffer.ownership.data is not Ownership.EXTERNAL and buffer.space.data is MemorySpace.SMEM
    )
    rotation_copies = _rotation_copies(kernel)
    rotation_vmem = sum(
        buffer_bytes(buffer) * copies
        for buffer, copies in rotation_copies
        if buffer.space.data is MemorySpace.VMEM
    )
    rotation_smem = sum(
        buffer_bytes(buffer) * copies
        for buffer, copies in rotation_copies
        if buffer.space.data is MemorySpace.SMEM
    )
    stages = (*_top_level_stages(kernel, operation_ids), *_pipeline_stages(kernel, operation_ids))
    if not stages:
        raise UnsupportedPhysicalCostModelError("physical schedule has no resource stages")
    vmem_peak_stage = max(stages, key=lambda value: value.live_vmem_bytes_per_device)
    smem_peak_stage = max(stages, key=lambda value: value.live_smem_bytes_per_device)
    memory = PhysicalMemoryLedger(
        external_hbm_resident_bytes_per_device=external_resident,
        external_hbm_input_footprint_bytes_per_device=external_input_footprint,
        external_hbm_output_footprint_bytes_per_device=external_output_footprint,
        explicit_hbm_dma_read_bytes_per_device=hbm_dma_read,
        explicit_hbm_dma_write_bytes_per_device=hbm_dma_write,
        explicit_local_dma_bytes_per_device=local_dma,
        allocated_vmem_bytes_per_device=allocated_vmem,
        allocated_smem_bytes_per_device=allocated_smem,
        peak_live_vmem_bytes_per_device=vmem_peak_stage.live_vmem_bytes_per_device,
        peak_live_vmem_scope=vmem_peak_stage.scope,
        peak_live_vmem_stage=vmem_peak_stage.stage,
        peak_live_smem_bytes_per_device=smem_peak_stage.live_smem_bytes_per_device,
        peak_live_smem_scope=smem_peak_stage.scope,
        peak_live_smem_stage=smem_peak_stage.stage,
        pipeline_rotation_vmem_bytes_per_device=rotation_vmem,
        pipeline_rotation_smem_bytes_per_device=rotation_smem,
    )
    resource_peaks = (
        _peak(stages, resource="dma", capacity=kernel.dma_engine_count.data),
        _peak(
            stages,
            resource="remote_dma",
            capacity=kernel.remote_dma_engine_count.data
            if kernel.remote_dma_engine_count is not None
            else 1,
        ),
        _peak(stages, resource="mxu", capacity=kernel.mxu_count.data),
        _peak(stages, resource="vector", capacity=kernel.vector_unit_count.data),
        _peak(stages, resource="ici", capacity=kernel.ici_link_count.data),
    )
    peak_link_channels: Counter[str] = Counter()
    for stage in stages:
        for link, uses in stage.link_channel_uses:
            peak_link_channels[link] = max(peak_link_channels[link], uses)
    link_records = tuple(
        PhysicalLinkRecord(
            link_id=link.link_id.data,
            source_device=link.source_device.data,
            destination_device=link.destination_device.data,
            bandwidth_bytes_per_second=link.bandwidth_bytes_per_second.data,
            channel_capacity=link.channel_count.data,
            peak_concurrent_channels=peak_link_channels[link.link_id.data],
            collective_operation_ids=tuple(
                sorted(collective_link_operations.get(link.link_id.data, set()))
            ),
            collective_traffic_derivable=False,
            remote_dma_operation_ids=tuple(
                sorted(remote_link_operations.get(link.link_id.data, set()))
            ),
            exact_remote_dma_link_bytes=remote_link_bytes[link.link_id.data],
        )
        for link in kernel.topology.links
    )
    total_mxu_flops = sum(region.total_flops for region in mxu_regions)
    mxu_flops_by_dtype = Counter[str]()
    for region in mxu_regions:
        mxu_flops_by_dtype[region.input_dtype] += region.total_flops
    priced_bf16_mxu_flops = mxu_flops_by_dtype["bf16"]
    total_vector_flops = sum(value.scalar_flops for value in vector_work)
    total_collective_bytes = sum(
        value.total_ring_equivalent_bidirectional_bytes_per_device for value in collective_records
    )
    hbm_bytes = hbm_dma_read + hbm_dma_write
    device_records = tuple(
        PhysicalDeviceRecord(
            device_id=device.device_id.data,
            coordinates=tuple(value.data for value in device.coordinates),
            mxu_flops=total_mxu_flops,
            vector_scalar_flops=total_vector_flops,
            hbm_dma_bytes=hbm_bytes,
            collective_ring_equivalent_bytes=total_collective_bytes,
            remote_dma_endpoint_bytes=remote_endpoint_bytes[device.device_id.data],
        )
        for device in kernel.topology.devices
    )
    imbalance = (
        _imbalance(
            "mxu_flops",
            tuple(Decimal(value.mxu_flops) for value in device_records),
            "Static SPMD local shapes assign the same declared MXU work to every device.",
        ),
        _imbalance(
            "hbm_dma_bytes",
            tuple(Decimal(value.hbm_dma_bytes) for value in device_records),
            "Explicit local HBM DMA operations execute symmetrically on every device.",
        ),
        _imbalance(
            "communication_endpoint_bytes",
            tuple(
                value.collective_ring_equivalent_bytes + Decimal(value.remote_dma_endpoint_bytes)
                for value in device_records
            ),
            "Collective bytes are ring-equivalent per-device scenarios; remote DMA endpoints come from exact declared routes.",
        ),
    )
    compute_ns = (
        Decimal(priced_bf16_mxu_flops)
        * Decimal(1_000_000_000)
        / Decimal(hardware.compute_flops_per_second)
    )
    hbm_ns = Decimal(hbm_bytes) * Decimal(1_000_000_000) / Decimal(hardware.hbm_bytes_per_second)
    collective_scenario_ns = (
        total_collective_bytes * Decimal(1_000_000_000) / Decimal(hardware.ici_bytes_per_second)
    )
    remote_endpoint_floor_ns = (
        Decimal(max(remote_endpoint_bytes, default=0))
        * Decimal(1_000_000_000)
        / Decimal(hardware.ici_bytes_per_second)
    )
    combined_injection_scenario_ns = (
        (total_collective_bytes + Decimal(max(remote_endpoint_bytes, default=0)))
        * Decimal(1_000_000_000)
        / Decimal(hardware.ici_bytes_per_second)
    )
    remote_floor_ns = max(
        (
            Decimal(value.exact_remote_dma_link_bytes)
            * Decimal(1_000_000_000)
            / Decimal(value.bandwidth_bytes_per_second)
            for value in link_records
        ),
        default=Decimal(0),
    )
    ici_scenario_ns = max(
        combined_injection_scenario_ns,
        remote_floor_ns,
    )
    overlapped_ns = max(compute_ns, hbm_ns, ici_scenario_ns)
    serial_ns = compute_ns + hbm_ns + ici_scenario_ns
    limiting = (
        "none"
        if overlapped_ns == 0
        else max(
            (("compute", compute_ns), ("hbm", hbm_ns), ("ici", ici_scenario_ns)),
            key=lambda value: value[1],
        )[0]
    )
    total_tile_programs = sum(region.total_tile_programs for region in mxu_regions)
    peak_vmem_ratio = Decimal(memory.peak_live_vmem_bytes_per_device) / Decimal(
        kernel.vmem_capacity_bytes.data
    )
    metrics = (
        _metric(
            "physical_priced_bf16_mxu_flops_per_device",
            priced_bf16_mxu_flops,
            Unit.FLOP,
            source,
            "physical_priced_bf16_mxu_flops",
            "sum(2*batch*M*K*N*executions where input_dtype=bf16)",
        ),
        _metric(
            "physical_mxu_tile_programs_per_device",
            total_tile_programs,
            Unit.COUNT,
            source,
            "physical_mxu_tile_grid",
            "sum(batch*(M/tile_m)*(N/tile_n)*(K/tile_k)*executions)",
        ),
        _metric(
            "physical_vector_scalar_flops_per_device",
            total_vector_flops,
            Unit.FLOP,
            source,
            "physical_vector_work_conventions",
            "sum(function_scalar_flops(output_elements)*executions)",
        ),
        _metric(
            "physical_explicit_hbm_dma_bytes_per_device",
            hbm_bytes,
            Unit.BYTE,
            source,
            "physical_explicit_hbm_dma",
            "sum(HBM_to_local_payload_bytes+local_to_HBM_payload_bytes)*executions",
        ),
        _metric(
            "physical_ring_equivalent_collective_bytes_per_device",
            total_collective_bytes,
            Unit.BYTE,
            source,
            "physical_ring_equivalent_collectives",
            "sum(kind_ring_equivalent_payload_bytes*executions)",
        ),
        _metric(
            "physical_peak_live_vmem_bytes_per_device",
            memory.peak_live_vmem_bytes_per_device,
            Unit.BYTE,
            source,
            "physical_inclusive_lifetimes",
            "max(sum(live_root_allocations+pipeline_rotations+declared_MXU_scratch))",
        ),
        _metric(
            "physical_peak_vmem_capacity_ratio",
            peak_vmem_ratio,
            Unit.RATIO,
            source,
            "physical_vmem_capacity_ratio",
            "peak_live_vmem_bytes/vmem_capacity_bytes",
            numerator=Quantity(
                value=Decimal(memory.peak_live_vmem_bytes_per_device), unit=Unit.BYTE
            ),
            denominator=Quantity(value=Decimal(kernel.vmem_capacity_bytes.data), unit=Unit.BYTE),
        ),
        _metric(
            "physical_priced_compute_time_floor",
            compute_ns,
            Unit.NANOSECOND,
            source,
            "physical_bf16_mxu_compute_floor",
            "physical_priced_bf16_mxu_flops/advertised_bf16_mxu_flops_per_second",
        ),
        _metric(
            "physical_priced_hbm_time_floor",
            hbm_ns,
            Unit.NANOSECOND,
            source,
            "physical_explicit_hbm_floor",
            "explicit_hbm_dma_bytes/advertised_hbm_bytes_per_second",
        ),
        _metric(
            "physical_collective_ring_equivalent_time_scenario",
            collective_scenario_ns,
            Unit.NANOSECOND,
            source,
            "physical_collective_ring_equivalent_scenario",
            "ring_equivalent_collective_bytes/advertised_ici_injection_bytes_per_second",
        ),
        _metric(
            "physical_remote_dma_exact_endpoint_time_floor",
            remote_endpoint_floor_ns,
            Unit.NANOSECOND,
            source,
            "physical_remote_dma_endpoint_floor",
            "max(remote_dma_endpoint_bytes)/advertised_ici_injection_bytes_per_second",
            scope="one declared physical kernel invocation across the full device mesh",
        ),
        _metric(
            "physical_remote_dma_exact_link_time_floor",
            remote_floor_ns,
            Unit.NANOSECOND,
            source,
            "physical_remote_dma_link_floor",
            "max(exact_remote_dma_link_bytes/link_bandwidth_bytes_per_second)",
            scope="one declared physical kernel invocation across the full device mesh",
        ),
        _metric(
            "physical_combined_ici_injection_time_scenario",
            combined_injection_scenario_ns,
            Unit.NANOSECOND,
            source,
            "physical_combined_ici_injection_scenario",
            "(ring_equivalent_collective_bytes+max_device_remote_dma_endpoint_bytes)/advertised_ici_injection_bytes_per_second",
            scope="one declared physical kernel invocation across the full device mesh",
        ),
        _metric(
            "physical_priced_ici_time_scenario",
            ici_scenario_ns,
            Unit.NANOSECOND,
            source,
            "physical_ici_scenario",
            "max(combined_ici_injection_time_scenario,remote_dma_exact_link_time_floor)",
            scope="one declared physical kernel invocation across the full device mesh",
        ),
        _metric(
            "physical_priced_overlapped_resource_scenario",
            overlapped_ns,
            Unit.NANOSECOND,
            source,
            "physical_overlapped_resource_scenario",
            "max(compute_time_floor,hbm_time_floor,ici_time_scenario)",
            scope="one declared physical kernel invocation across the full device mesh",
        ),
        _metric(
            "physical_priced_serial_resource_scenario",
            serial_ns,
            Unit.NANOSECOND,
            source,
            "physical_serial_resource_scenario",
            "compute_time_floor+hbm_time_floor+ici_time_scenario",
            scope="one declared physical kernel invocation across the full device mesh",
        ),
    )
    return PhysicalKernelResourceReport(
        schema_version=PHYSICAL_KERNEL_RESOURCE_SCHEMA,
        physical_schedule_sha256=physical_hash,
        kernel_name=kernel.sym_name.data,
        target=kernel.target.data,
        topology_authority=kernel.topology_authority.data,
        execution_authority="static-declared-schedule-only",
        mesh_axes=mesh,
        device_count=device_count,
        canonical_operation_inventory=static_inventory,
        operation_executions=tuple(
            sorted((name, count) for name, count in execution_inventory.items() if count)
        ),
        hardware=hardware,
        mxu_regions=tuple(mxu_regions),
        vector_work=tuple(vector_work),
        memory=memory,
        collectives=tuple(collective_records),
        remote_dmas=tuple(remote_records),
        stages=tuple(stages),
        resource_peaks=resource_peaks,
        devices=device_records,
        links=link_records,
        imbalance=imbalance,
        priced_compute_time_floor_ns=compute_ns,
        priced_hbm_time_floor_ns=hbm_ns,
        collective_ring_equivalent_time_scenario_ns=collective_scenario_ns,
        remote_dma_exact_endpoint_time_floor_ns=remote_endpoint_floor_ns,
        remote_dma_exact_link_time_floor_ns=remote_floor_ns,
        combined_ici_injection_time_scenario_ns=combined_injection_scenario_ns,
        priced_ici_time_scenario_ns=ici_scenario_ns,
        priced_overlapped_resource_scenario_ns=overlapped_ns,
        priced_serial_resource_scenario_ns=serial_ns,
        predicted_limiting_priced_resource=limiting,
        unpriced_work=(
            f"vector_scalar_flops={total_vector_flops}",
            f"special_function_ops={sum(value.special_function_ops for value in vector_work)}",
            f"index_and_compare_ops={sum(value.index_and_compare_ops for value in vector_work)}",
            f"tile_programs={total_tile_programs}",
            *tuple(
                f"mxu_flops[{dtype}]={flops}"
                for dtype, flops in sorted(mxu_flops_by_dtype.items())
                if dtype != "bf16" and flops
            ),
            "kernel launch, synchronization, collective startup, and compiler overhead",
        ),
        assumptions=(
            "The report describes the verified static physical schedule and does not measure compiler execution.",
            "MXU multiply-add counts as two floating-point operations.",
            "HBM traffic includes only explicit physical DMA payloads.",
            "Collective bytes are ring-equivalent bidirectional per-device scenarios.",
            "Pipeline work and traffic multiply by trip count; rotation storage does not.",
            "Pipeline loops conservatively co-reside every uncaptured kernel-local allocation because they have no outer absolute stage.",
            "Views alias their allocation roots and never add storage capacity.",
        ),
        omissions=(
            "Collective plans identify participating links but not an exact per-link byte schedule.",
            "Vector and special-function work have no external hardware rate and remain unpriced.",
            "Only BF16 MXU FLOPs use the advertised TPU7x rate; F16 and F32 MXU work remain unpriced.",
            "DMA, vector, and collective execution may be delegated to JAX/XLA rather than owned Mosaic kernels.",
            "Fusion savings require comparison of two verified physical schedules and are not inferred from labels.",
            "No measured calibration or predictive validation is applied to this static report.",
        ),
        metrics=metrics,
    )


def validate_physical_kernel_report(
    report: PhysicalKernelResourceReport,
    *,
    module: ModuleOp,
    hardware: HardwareRateModel,
) -> None:
    expected = analyze_physical_kernel(
        module,
        hardware=hardware,
        expected_schedule_sha256=report.physical_schedule_sha256,
    )
    if report != expected:
        raise ValueError("PHYSICAL_KERNEL_RESOURCE_REPORT_REPLAY_MISMATCH")


def write_physical_kernel_report(
    output: Path,
    *,
    module: ModuleOp,
    hardware: HardwareRateModel,
) -> PhysicalKernelResourceReport:
    output = output.resolve()
    if output.exists():
        raise ValueError("PHYSICAL_KERNEL_RESOURCE_OUTPUT_EXISTS")
    report = analyze_physical_kernel(module, hardware=hardware)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def physical_schedule_source(module: ModuleOp) -> MetricSource:
    physical_hash = schedule_sha256(module)
    if hashlib.sha256(canonical_module_text(module).encode()).hexdigest() != physical_hash:
        raise UnsupportedPhysicalCostModelError("canonical physical schedule hash drift")
    return MetricSource(
        artifact_sha256=physical_hash,
        artifact_path=f"physical/{physical_hash}.xdsl",
        tool="tpu-cake",
        field="canonical-physical-schedule-xdsl",
    )
