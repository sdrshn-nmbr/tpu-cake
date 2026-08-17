from decimal import Decimal

from tpu_cake.cost_model import (
    MatmulCostModelInput,
    estimate_distributed_matmul,
    estimate_distributed_matmul_input,
    tpu7x_tensorcore_rates,
)
from tpu_cake.lowering import lower_distributed_matmul
from tpu_cake.metrics import MetricSource
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def test_distributed_matmul_cost_model_has_units_formulas_and_provenance() -> None:
    plan = lower_physical_matmul_to_pallas(
        lower_distributed_matmul(distributed_matmul_schedule(mesh_size=8, m=128, k=1024, n=1024))
    )
    report = estimate_distributed_matmul(
        plan,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256="a" * 64,
            artifact_path="model-input.json",
            tool="tpu-cake",
            field="distributed-matmul-v1",
        ),
    )
    assert report.predicted_limiting_resource == "ici"
    assert report.counts.operations_per_device == 33_554_432
    assert report.counts.hbm_read_bytes_per_device == 819_200
    assert report.counts.hbm_write_bytes_per_device == 589_824
    assert report.counts.ici_bidirectional_bytes_per_device == 917_504
    assert all(metric.formula is not None for metric in report.metrics)
    intensity = next(metric for metric in report.metrics if metric.name == "arithmetic_intensity")
    assert intensity.quantity.value > Decimal(20)

    replayed = estimate_distributed_matmul_input(
        MatmulCostModelInput(
            schedule_sha256=plan.schedule_sha256,
            mesh_size=plan.mesh_size,
            m=plan.global_lhs_shape[0],
            k=plan.global_lhs_shape[1],
            n=plan.global_rhs_shape[1],
            tile_m=plan.tile_m,
            tile_k=plan.tile_k,
            tile_n=plan.tile_n,
            collective_link_bandwidths=plan.collective_link_bandwidths,
            hardware=tpu7x_tensorcore_rates(),
        ),
        source=report.metrics[0].sources[0],
    )
    assert replayed == report


def test_declared_slow_link_changes_the_communication_floor() -> None:
    model_input = MatmulCostModelInput(
        schedule_sha256="a" * 64,
        mesh_size=8,
        m=128,
        k=1024,
        n=1024,
        tile_m=128,
        tile_k=128,
        tile_n=128,
        collective_link_bandwidths=(("slow", 1),),
        hardware=tpu7x_tensorcore_rates(),
    )
    report = estimate_distributed_matmul_input(
        model_input,
        source=MetricSource(
            artifact_sha256="b" * 64,
            artifact_path="model-input.json",
            tool="tpu-cake",
            field="distributed-matmul-v1",
        ),
    )

    communication = next(
        metric for metric in report.metrics if metric.name == "ici_time_floor"
    )
    assert communication.quantity.value == Decimal(917_504_000_000_000)
