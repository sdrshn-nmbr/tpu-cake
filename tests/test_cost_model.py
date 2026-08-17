from decimal import Decimal

from tpu_cake.cost_model import estimate_distributed_matmul, tpu7x_tensorcore_rates
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
    assert report.counts.hbm_read_bytes_per_device == 294_912
    assert report.counts.hbm_write_bytes_per_device == 65_536
    assert report.counts.ici_bidirectional_bytes_per_device == 917_504
    assert all(metric.formula is not None for metric in report.metrics)
    intensity = next(metric for metric in report.metrics if metric.name == "arithmetic_intensity")
    assert intensity.quantity.value > Decimal(90)
