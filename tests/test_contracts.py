from decimal import Decimal

import pytest
from pydantic import ValidationError

from tpu_cake.contracts import (
    CorrectnessResult,
    ProfileExpectation,
    RunReceipt,
    RuntimeIdentity,
    SemanticPropertyResult,
)
from tpu_cake.metrics import (
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)
from tpu_cake.workloads import matmul_experiment


def _source() -> MetricSource:
    return MetricSource(
        artifact_sha256="0" * 64,
        artifact_path="trace.xplane.pb",
        tool="XProf",
        field="duration",
    )


def test_profile_expectation_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ProfileExpectation.model_validate(
            {"name": "decode", "stage": "steady_decode", "unexpected": True}
        )


def test_experiment_identity_is_stable() -> None:
    assert matmul_experiment().experiment_id == matmul_experiment().experiment_id


def test_derived_metric_requires_formula() -> None:
    with pytest.raises(ValidationError, match="formula identity"):
        Metric(
            name="duration",
            quantity=Quantity(value=Decimal(1), unit=Unit.MICROSECOND),
            kind=MeasurementKind.DERIVED,
            interval=MeasurementInterval(scope="kernel"),
            sources=(_source(),),
        )


def test_ratio_requires_numerator_and_denominator() -> None:
    with pytest.raises(ValidationError, match="numerator and denominator"):
        Metric(
            name="mfu",
            quantity=Quantity(value=Decimal("0.5"), unit=Unit.RATIO),
            kind=MeasurementKind.MEASURED,
            interval=MeasurementInterval(scope="kernel"),
            sources=(_source(),),
        )


def test_run_receipt_is_immutable() -> None:
    experiment = matmul_experiment()
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        schedule_sha256=experiment.schedule_sha256,
        status="passed",
        runtime=RuntimeIdentity(python="3.13"),
        correctness=CorrectnessResult(passed=True, oracle="numpy"),
        required_semantic_properties=(),
        metrics=(),
        artifacts=(),
    )
    with pytest.raises(ValidationError):
        receipt.status = "failed"


def test_passed_receipt_requires_all_semantic_properties() -> None:
    experiment = matmul_experiment()
    with pytest.raises(ValidationError, match="every required semantic property"):
        RunReceipt(
            experiment_id=experiment.experiment_id,
            schedule_sha256=experiment.schedule_sha256,
            status="passed",
            runtime=RuntimeIdentity(python="3.13"),
            correctness=CorrectnessResult(passed=True, oracle="numpy"),
            required_semantic_properties=("batch_permutation_invariance",),
            metrics=(),
            artifacts=(),
        )


def test_passed_correctness_rejects_failed_semantic_property() -> None:
    with pytest.raises(ValidationError, match="failed semantic property"):
        CorrectnessResult(
            passed=True,
            oracle="numpy",
            semantic_properties=(
                SemanticPropertyResult(
                    property="batch_permutation_invariance",
                    passed=False,
                    maximum_absolute_error=1,
                    details="slot-dependent output",
                ),
            ),
        )
