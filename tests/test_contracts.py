import hashlib
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.contracts import (
    PHASE_REQUIRED_ROLES,
    ArtifactReference,
    CorrectnessResult,
    EvidencePhase,
    KernelExperiment,
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
from tpu_cake.receipt import validate_receipt
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


def test_experiment_json_round_trips_without_serializing_derived_identity() -> None:
    experiment = matmul_experiment()
    encoded = experiment.model_dump_json(exclude_computed_fields=True)

    assert "experiment_id" not in encoded
    assert KernelExperiment.model_validate_json(encoded) == experiment


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
        status="rejected",
        runtime=RuntimeIdentity(python="3.13"),
        correctness=CorrectnessResult(passed=True, oracle="numpy"),
        required_semantic_properties=(),
        metrics=(),
        artifacts=(),
        phases=(),
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
            phases=(),
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


def _complete_receipt(root: Path) -> tuple[RunReceipt, KernelExperiment]:
    experiment = matmul_experiment()
    artifacts = []
    phases = []
    for phase, roles in PHASE_REQUIRED_ROLES.items():
        phase_paths = []
        for role in roles:
            path = root / phase.value / f"{role.value}.artifact"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{phase.value}:{role.value}")
            relative = str(path.relative_to(root))
            phase_paths.append(relative)
            artifacts.append(
                ArtifactReference(
                    path=relative,
                    size_bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    role=role,
                )
            )
        phases.append(EvidencePhase(name=phase, artifact_paths=tuple(phase_paths)))
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        schedule_sha256=experiment.schedule_sha256,
        status="passed",
        runtime=RuntimeIdentity(python="3.13"),
        correctness=CorrectnessResult(passed=True, oracle="exact"),
        required_semantic_properties=(),
        metrics=(),
        artifacts=tuple(artifacts),
        phases=tuple(phases),
    )
    return receipt, experiment


def test_receipt_survives_moving_its_complete_bundle(tmp_path) -> None:
    original = tmp_path / "original"
    receipt, experiment = _complete_receipt(original)
    moved = tmp_path / "moved"
    original.rename(moved)

    validate_receipt(receipt, experiment, root=moved)


def test_receipt_rejects_a_mutated_required_artifact(tmp_path) -> None:
    root = tmp_path / "bundle"
    receipt, experiment = _complete_receipt(root)
    (root / receipt.artifacts[0].path).write_text("tampered")

    with pytest.raises(ValueError, match="size changed|hash changed"):
        validate_receipt(receipt, experiment, root=root)


def test_receipt_rejects_a_deleted_required_artifact(tmp_path) -> None:
    root = tmp_path / "bundle"
    receipt, experiment = _complete_receipt(root)
    (root / receipt.artifacts[0].path).unlink()

    with pytest.raises(ValueError, match="artifact is missing"):
        validate_receipt(receipt, experiment, root=root)


def test_receipt_rejects_a_phase_with_only_generic_role_coverage(tmp_path) -> None:
    receipt, _ = _complete_receipt(tmp_path / "bundle")
    payload = receipt.model_dump(mode="json")
    trace = next(phase for phase in payload["phases"] if phase["name"] == "trace")
    removed = next(
        artifact
        for artifact in payload["artifacts"]
        if artifact["path"] in trace["artifact_paths"]
        and artifact["role"] == "trace_result"
    )
    trace["artifact_paths"].remove(removed["path"])
    aggregate = next(phase for phase in payload["phases"] if phase["name"] == "aggregate")
    aggregate["artifact_paths"].append(removed["path"])

    with pytest.raises(ValidationError, match="phase trace is missing roles"):
        RunReceipt.model_validate(payload)
