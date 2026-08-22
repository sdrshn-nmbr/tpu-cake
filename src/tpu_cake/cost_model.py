from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.dialects.tpu_schedule import CollectiveImplementation
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)
from tpu_cake.pallas_lowering import PallasMatmulPlan


class HardwareRateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    compute_flops_per_second: int = Field(gt=0)
    hbm_bytes_per_second: int = Field(gt=0)
    ici_bytes_per_second: int = Field(gt=0)
    compute_status: str = Field(min_length=1)
    hbm_status: str = Field(min_length=1)
    ici_status: str = Field(min_length=1)
    source_url: str = Field(min_length=1)


class MatmulPhysicalCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operations_per_device: int = Field(gt=0)
    hbm_read_bytes_per_device: int = Field(gt=0)
    hbm_write_bytes_per_device: int = Field(gt=0)
    ici_bidirectional_bytes_per_device: int = Field(gt=0)
    peak_live_vmem_bytes_per_device: int = Field(gt=0)
    collective_hbm_scratch_bytes_per_device: int = Field(ge=0)
    collective_vmem_scratch_bytes_per_device: int = Field(ge=0)
    collective_dma_semaphore_count: int = Field(ge=0)
    collective_capacity_semaphore_count: int = Field(ge=0)
    collective_startup_semaphore_count: int = Field(ge=0)
    collective_startup_barrier_phases: int = Field(ge=0)
    collective_remote_half_output_copy_count: int = Field(ge=0)


class MatmulCostModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mesh_size: int = Field(gt=0)
    m: int = Field(gt=0)
    k: int = Field(gt=0)
    n: int = Field(gt=0)
    tile_m: int = Field(gt=0)
    tile_k: int = Field(gt=0)
    tile_n: int = Field(gt=0)
    collective_implementation: CollectiveImplementation | None = None
    collective_link_bandwidths: tuple[tuple[str, int], ...] = ()
    hardware: HardwareRateModel

    @model_validator(mode="after")
    def dimensions_are_compatible(self) -> MatmulCostModelInput:
        if self.k % self.mesh_size or self.n % self.mesh_size:
            raise ValueError("distributed matmul K and N must divide the mesh size")
        if self.m % self.tile_m or self.n % self.tile_n:
            raise ValueError("matmul tiles must divide M and N")
        if self.tile_k != self.k // self.mesh_size:
            raise ValueError("matmul tile K must equal the local K extent")
        link_ids = tuple(link_id for link_id, _ in self.collective_link_bandwidths)
        if link_ids != tuple(sorted(set(link_ids))) or any(
            bandwidth <= 0 for _, bandwidth in self.collective_link_bandwidths
        ):
            raise ValueError(
                "collective links must be unique, ordered, and have positive bandwidth"
            )
        return self


class CostModelReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: HardwareRateModel
    counts: MatmulPhysicalCounts
    predicted_limiting_resource: str
    assumptions: tuple[str, ...]
    metrics: tuple[Metric, ...]


def tpu7x_tensorcore_rates() -> HardwareRateModel:
    return HardwareRateModel(
        name="TPU7x TensorCore share",
        scope="one JAX TPU device, assumed one half of a two-TensorCore TPU7x chip",
        compute_flops_per_second=1_153_500_000_000_000,
        hbm_bytes_per_second=3_690_000_000_000,
        ici_bytes_per_second=600_000_000_000,
        compute_status="derived from advertised per-chip BF16 peak divided by two",
        hbm_status="derived from advertised per-chip HBM bandwidth divided by two",
        ici_status="derived from advertised per-chip bidirectional ICI bandwidth divided by two",
        source_url="https://docs.cloud.google.com/tpu/docs/tpu7x",
    )


def _metric(
    name: str,
    value: Decimal | int,
    unit: Unit,
    source: MetricSource,
    formula_name: str,
    expression: str,
) -> Metric:
    return Metric(
        name=name,
        quantity=Quantity(value=Decimal(value), unit=unit),
        kind=MeasurementKind.ESTIMATED,
        interval=MeasurementInterval(scope="one distributed matmul on one JAX TPU device"),
        sources=(source,),
        formula=FormulaIdentity(name=formula_name, version="1", expression=expression),
    )


def estimate_distributed_matmul(
    plan: PallasMatmulPlan,
    *,
    hardware: HardwareRateModel,
    source: MetricSource,
) -> CostModelReport:
    m, local_k = plan.lhs_local_shape
    _, n = plan.rhs_local_shape
    output_elements = plan.output_local_shape[0] * plan.output_local_shape[1]
    partial_elements = plan.partial_local_shape[0] * plan.partial_local_shape[1]
    operations = 2 * m * local_k * n
    partial_bytes = 4 * partial_elements
    hbm_read = 2 * (m * local_k + local_k * n) + partial_bytes
    hbm_write = partial_bytes + 4 * output_elements
    generic_ici_bidirectional = 2 * partial_bytes * (plan.mesh_size - 1) // plan.mesh_size
    ici_bidirectional = (
        plan.collective_remote_bidirectional_endpoint_bytes
        if plan.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING
        else generic_ici_bidirectional
    )
    tile_input_bytes = 2 * (plan.tile_m * plan.tile_k + plan.tile_k * plan.tile_n)
    tile_output_bytes = 4 * plan.tile_m * plan.tile_n
    collective_peak_live_vmem = (
        partial_bytes
        + 4 * output_elements
        + plan.collective_accumulator_vmem_bytes
        if plan.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING
        else 0
    )
    peak_live_vmem = max(
        tile_input_bytes + tile_output_bytes,
        collective_peak_live_vmem,
    )
    counts = MatmulPhysicalCounts(
        operations_per_device=operations,
        hbm_read_bytes_per_device=hbm_read,
        hbm_write_bytes_per_device=hbm_write,
        ici_bidirectional_bytes_per_device=ici_bidirectional,
        peak_live_vmem_bytes_per_device=peak_live_vmem,
        collective_hbm_scratch_bytes_per_device=plan.collective_hbm_scratch_bytes,
        collective_vmem_scratch_bytes_per_device=plan.collective_accumulator_vmem_bytes,
        collective_dma_semaphore_count=plan.collective_dma_semaphore_count,
        collective_capacity_semaphore_count=plan.collective_capacity_semaphore_count,
        collective_startup_semaphore_count=plan.collective_startup_semaphore_count,
        collective_startup_barrier_phases=plan.collective_startup_barrier_phases,
        collective_remote_half_output_copy_count=(
            plan.collective_remote_half_output_copy_count
        ),
    )
    hbm_bytes = hbm_read + hbm_write
    compute_ns = (
        Decimal(operations) * Decimal(1_000_000_000) / Decimal(hardware.compute_flops_per_second)
    )
    memory_ns = Decimal(hbm_bytes) * Decimal(1_000_000_000) / Decimal(hardware.hbm_bytes_per_second)
    effective_ici_bandwidth = hardware.ici_bytes_per_second
    if plan.collective_link_bandwidths:
        effective_ici_bandwidth = min(
            effective_ici_bandwidth,
            min(bandwidth for _, bandwidth in plan.collective_link_bandwidths),
        )
    communication_ns = (
        Decimal(ici_bidirectional)
        * Decimal(1_000_000_000)
        / Decimal(effective_ici_bandwidth)
    )
    lower_bound_ns = max(compute_ns, memory_ns, communication_ns)
    serial_ns = compute_ns + memory_ns + communication_ns
    limiting = max(
        (("compute", compute_ns), ("hbm", memory_ns), ("ici", communication_ns)),
        key=lambda value: value[1],
    )[0]
    topology_metrics = (
        (
            _metric(
                "declared_collective_bottleneck_bandwidth",
                min(bandwidth for _, bandwidth in plan.collective_link_bandwidths),
                Unit.BYTE_PER_SECOND,
                source,
                "declared_collective_bottleneck_bandwidth",
                "min(declared_collective_link_bandwidths)",
            ),
        )
        if plan.collective_link_bandwidths
        else ()
    )
    metrics = (
        _metric("operations_per_device", operations, Unit.FLOP, source, "matmul_flops", "2*M*K*N"),
        _metric(
            "hbm_bytes_per_device",
            hbm_bytes,
            Unit.BYTE,
            source,
            "matmul_algorithmic_hbm_bytes",
            "sizeof(lhs)+sizeof(rhs)+sizeof(output)",
        ),
        _metric(
            "ici_bidirectional_bytes_per_device",
            ici_bidirectional,
            Unit.BYTE,
            source,
            (
                "pallas_remote_dma_endpoint_bytes"
                if plan.collective_implementation
                is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING
                else "ring_reduce_scatter_bytes"
            ),
            (
                "declared_bidirectional_remote_dma_endpoint_bytes"
                if plan.collective_implementation
                is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING
                else "2*sizeof(partial)*(P-1)/P"
            ),
        ),
        _metric(
            "collective_hbm_scratch_bytes_per_device",
            plan.collective_hbm_scratch_bytes,
            Unit.BYTE,
            source,
            "collective_hbm_scratch_capacity",
            "typed_collective_plan.hbm_scratch_bytes",
        ),
        _metric(
            "collective_vmem_scratch_bytes_per_device",
            plan.collective_accumulator_vmem_bytes,
            Unit.BYTE,
            source,
            "collective_vmem_scratch_capacity",
            "typed_collective_plan.vmem_scratch_bytes",
        ),
        _metric(
            "collective_semaphore_count",
            plan.collective_dma_semaphore_count
            + plan.collective_capacity_semaphore_count
            + plan.collective_startup_semaphore_count,
            Unit.COUNT,
            source,
            "collective_semaphore_count",
            "dma_semaphores+capacity_semaphores+startup_semaphores",
        ),
        _metric(
            "arithmetic_intensity",
            Decimal(operations) / Decimal(hbm_bytes),
            Unit.FLOP_PER_BYTE,
            source,
            "arithmetic_intensity",
            "operations/hbm_bytes",
        ),
        _metric(
            "compute_time_floor",
            compute_ns,
            Unit.NANOSECOND,
            source,
            "compute_floor",
            "operations/compute_rate",
        ),
        _metric(
            "hbm_time_floor",
            memory_ns,
            Unit.NANOSECOND,
            source,
            "hbm_floor",
            "hbm_bytes/hbm_bandwidth",
        ),
        _metric(
            "ici_time_floor",
            communication_ns,
            Unit.NANOSECOND,
            source,
            "ici_floor",
            "ici_bidirectional_bytes/ici_bidirectional_bandwidth",
        ),
        _metric(
            "idealized_time_floor",
            lower_bound_ns,
            Unit.NANOSECOND,
            source,
            "overlapped_resource_floor",
            "max(compute_time,hbm_time,ici_time)",
        ),
        _metric(
            "serial_resource_time",
            serial_ns,
            Unit.NANOSECOND,
            source,
            "serial_resource_time",
            "compute_time+hbm_time+ici_time",
        ),
        *topology_metrics,
    )
    native_collective = (
        plan.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING
    )
    topology_assumptions = (
        (
            (
                "The typed Pallas plan fixes logical remote-DMA endpoints; hardware maps "
                "those logical neighbors onto physical links."
                if native_collective
                else "Declared topology is a static cost-model constraint; XLA selects the "
                "executed collective route."
            ),
            (
                "ICI time uses the smaller of the hardware rate and the slowest declared "
                "collective link bandwidth."
            ),
        )
        if plan.collective_link_bandwidths
        else ()
    )
    return CostModelReport(
        schedule_sha256=plan.schedule_sha256,
        hardware=hardware,
        counts=counts,
        predicted_limiting_resource=limiting,
        assumptions=(
            "A multiply-add counts as two BF16 floating-point operations.",
            "Per-JAX-device rates are one half of advertised per-chip rates.",
            (
                "HBM bytes include the materialized partial passed between the matmul and "
                "owned collective Pallas calls; native scratch traffic is unpriced."
                if native_collective
                else "HBM bytes include one materialization of the Pallas partial before "
                "XLA reduce-scatter."
            ),
            "HBM bytes exclude additional compiler reloads, padding, and layout conversions.",
            (
                "ICI bytes use the typed native ring's exact bidirectional endpoint bytes."
                if native_collective
                else "Reduce-scatter uses a ring-equivalent bidirectional byte lower bound."
            ),
            "Launch, synchronization, collective startup, and compiler overhead are omitted.",
            *topology_assumptions,
        ),
        metrics=metrics,
    )


def estimate_distributed_matmul_input(
    model_input: MatmulCostModelInput,
    *,
    source: MetricSource,
) -> CostModelReport:
    local_k = model_input.k // model_input.mesh_size
    plan = PallasMatmulPlan(
        name="distributed_matmul_physical",
        schedule_sha256=model_input.schedule_sha256,
        mesh_axis="t",
        mesh_size=model_input.mesh_size,
        lhs_local_shape=(model_input.m, local_k),
        rhs_local_shape=(local_k, model_input.n),
        partial_local_shape=(model_input.m, model_input.n),
        output_local_shape=(model_input.m, model_input.n // model_input.mesh_size),
        lhs_sharding=("", "t"),
        rhs_sharding=("t", ""),
        output_sharding=("", "t"),
        scatter_dimension=1,
        tile_m=model_input.tile_m,
        tile_k=model_input.tile_k,
        tile_n=model_input.tile_n,
        collective_link_bandwidths=model_input.collective_link_bandwidths,
        collective_implementation=model_input.collective_implementation,
    )
    return estimate_distributed_matmul(plan, hardware=model_input.hardware, source=source)
