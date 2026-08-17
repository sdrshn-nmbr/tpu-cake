from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

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
    hbm_read = 2 * (m * local_k + local_k * n)
    hbm_write = 4 * output_elements
    partial_bytes = 4 * partial_elements
    ici_bidirectional = 2 * partial_bytes * (plan.mesh_size - 1) // plan.mesh_size
    peak_live_vmem = max(hbm_read, partial_bytes + 4 * output_elements)
    counts = MatmulPhysicalCounts(
        operations_per_device=operations,
        hbm_read_bytes_per_device=hbm_read,
        hbm_write_bytes_per_device=hbm_write,
        ici_bidirectional_bytes_per_device=ici_bidirectional,
        peak_live_vmem_bytes_per_device=peak_live_vmem,
    )
    hbm_bytes = hbm_read + hbm_write
    compute_ns = (
        Decimal(operations) * Decimal(1_000_000_000) / Decimal(hardware.compute_flops_per_second)
    )
    memory_ns = Decimal(hbm_bytes) * Decimal(1_000_000_000) / Decimal(hardware.hbm_bytes_per_second)
    communication_ns = (
        Decimal(ici_bidirectional) * Decimal(1_000_000_000) / Decimal(hardware.ici_bytes_per_second)
    )
    lower_bound_ns = max(compute_ns, memory_ns, communication_ns)
    serial_ns = compute_ns + memory_ns + communication_ns
    limiting = max(
        (("compute", compute_ns), ("hbm", memory_ns), ("ici", communication_ns)),
        key=lambda value: value[1],
    )[0]
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
            "ring_reduce_scatter_bytes",
            "2*sizeof(partial)*(P-1)/P",
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
    )
    return CostModelReport(
        schedule_sha256=plan.schedule_sha256,
        hardware=hardware,
        counts=counts,
        predicted_limiting_resource=limiting,
        assumptions=(
            "A multiply-add counts as two BF16 floating-point operations.",
            "Per-JAX-device rates are one half of advertised per-chip rates.",
            "HBM bytes are algorithmic minima and exclude compiler reloads and padding.",
            "Reduce-scatter uses a ring-equivalent bidirectional byte lower bound.",
            "Launch, synchronization, collective startup, and compiler overhead are omitted.",
        ),
        metrics=metrics,
    )
