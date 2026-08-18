import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import tpu_cake.seqax_cost_calibration as calibration
from tpu_cake.cli import _parser
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)
from tpu_cake.runner import RunMode
from tpu_cake.seqax_cost_calibration import (
    SeqaxCostCalibrationContract,
    _fit_coefficients,
    default_seqax_cost_calibration_contract,
    derive_seqax_cost_calibration,
    validate_seqax_cost_calibration,
    write_seqax_cost_calibration,
)


def _metric(
    name: str,
    value: str,
    kind: MeasurementKind,
    source_sha256: str,
    source_name: str | None = None,
) -> Metric:
    source_name = source_name or name
    return Metric(
        name=name,
        quantity=Quantity(value=Decimal(value), unit=Unit.NANOSECOND),
        kind=kind,
        interval=MeasurementInterval(scope="one forward"),
        sources=(
            MetricSource(
                artifact_sha256=source_sha256,
                artifact_path=f"{source_name}.artifact",
                tool="test",
                field=source_name,
            ),
        ),
        formula=FormulaIdentity(name="test", version="1", expression=name),
    )


def _synthetic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    contract = default_seqax_cost_calibration_contract()
    profile_root = tmp_path / "profile"
    (profile_root / "surface" / "cost").mkdir(parents=True)
    (profile_root / "receipt.json").write_text("profile")
    (profile_root / "surface" / "receipt.json").write_text("surface")
    values = {
        "tiny": (2, "4.84", "4.0", "220781.3625", "232147.659", "219736.382"),
        "wider": (2, "24.8266666667", "19.94", "217599.6775", "229367.347", "204342.1745"),
        "deeper": (4, "46.0533333333", "38.20", "349468.225", "363554.622", "357722.689"),
    }
    schedules = {name: str(index + 1) * 64 for index, name in enumerate(values)}
    metrics = []
    results = []
    costs = {}
    for index, (name, (_layers, floor, materialized, trace, p90, counters)) in enumerate(
        values.items()
    ):
        trace_sha = f"{index + 4:x}" * 64
        counter_sha = f"{index + 7:x}" * 64
        metrics.extend(
            (
                _metric(
                    f"{name}_trace_median_compiled_forward_duration",
                    trace,
                    MeasurementKind.DERIVED,
                    trace_sha,
                    f"{name}_trace_module",
                ),
                _metric(
                    f"{name}_trace_p90_compiled_forward_duration",
                    p90,
                    MeasurementKind.DERIVED,
                    trace_sha,
                    f"{name}_trace_module",
                ),
                _metric(
                    f"{name}_counters_median_compiled_forward_duration",
                    counters,
                    MeasurementKind.DERIVED,
                    counter_sha,
                ),
            )
        )
        costs[name] = SimpleNamespace(
            schedule_sha256=schedules[name],
            metrics=(
                _metric(
                    "seqax_idealized_time_floor",
                    floor,
                    MeasurementKind.ESTIMATED,
                    schedules[name],
                ),
                _metric(
                    "seqax_materialized_hbm_time",
                    materialized,
                    MeasurementKind.ESTIMATED,
                    schedules[name],
                ),
            ),
        )
        for mode in (RunMode.TRACE, RunMode.COUNTERS):
            results.append(
                SimpleNamespace(
                    invocation=SimpleNamespace(
                        scenario=name,
                        mode=mode,
                        schedule_sha256=schedules[name],
                        runtime=contract.runtime,
                        device_kind=contract.device_kind,
                        device_count=contract.device_count,
                        measured_iterations=contract.measured_iterations,
                        input_placement=contract.input_placement,
                        execution_scope=contract.execution_scope,
                    )
                )
            )
        (profile_root / "surface" / "cost" / f"{name}.json").write_text(
            json.dumps({"scenario": name})
        )
    receipt = SimpleNamespace(
        surface_id=contract.surface_id,
        surface_receipt_sha256=contract.surface_receipt_sha256,
        results=tuple(results),
        metrics=tuple(metrics),
    )
    monkeypatch.setattr(
        calibration.SeqaxSurfaceProfileReceipt,
        "model_validate_json",
        classmethod(lambda _class, _value: receipt),
    )
    monkeypatch.setattr(
        calibration,
        "validate_seqax_surface_profile_receipt",
        lambda _receipt, *, root: None,
    )
    monkeypatch.setattr(
        calibration.SeqaxCostModelReport,
        "model_validate_json",
        classmethod(lambda _class, value: costs[json.loads(value)["scenario"]]),
    )

    def sha256(path: Path) -> str:
        if path == profile_root / "receipt.json":
            return contract.surface_profile_receipt_sha256
        if path == profile_root / "surface" / "receipt.json":
            return contract.surface_receipt_sha256
        if path.parent.name == "cost":
            return "a" * 64
        raise AssertionError(path)

    monkeypatch.setattr(calibration, "_sha256", sha256)
    return profile_root, contract, receipt, costs


def test_calibration_contract_file_is_canonical() -> None:
    path = Path(__file__).resolve().parents[1] / "contracts" / "seqax-cost-calibration-v1.json"
    saved = SeqaxCostCalibrationContract.model_validate_json(path.read_text())
    expected = default_seqax_cost_calibration_contract()

    assert saved == expected
    assert (
        expected.contract_id == "baade8381ad18888e0b1940a750f48af72d80108ae2afbf803109207d145fb3d"
    )


def test_layer_residual_fit_is_exact_and_requires_distinct_layer_counts() -> None:
    fixed, per_layer = _fit_coefficients(
        (2, 2, 4),
        (Decimal(220), Decimal(218), Decimal(350)),
        (Decimal(5), Decimal(25), Decimal(46)),
        decimal_places=6,
    )

    assert fixed == Decimal("104.000000")
    assert per_layer == Decimal("50.000000")
    with pytest.raises(ValueError, match="MULTIPLE_LAYER_COUNTS"):
        _fit_coefficients(
            (2, 2, 2),
            (Decimal(1), Decimal(2), Decimal(3)),
            (Decimal(0), Decimal(0), Decimal(0)),
            decimal_places=6,
        )


def test_calibration_replays_bound_profile_and_exposes_no_predictive_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root, contract, _receipt, _costs = _synthetic_evidence(tmp_path, monkeypatch)

    report = derive_seqax_cost_calibration(profile_root, contract)

    assert tuple(point.scenario for point in report.points) == contract.scenarios
    assert report.fit_within_declared_surface_error
    assert report.cross_mode_consistent
    assert report.status == "descriptive-in-surface-fit-only"
    assert not report.predictive_validation
    assert report.fixed_residual_ns == Decimal("88929.201666666600")
    assert report.per_layer_residual_ns == Decimal("65123.242500000025")
    assert report.maximum_in_surface_relative_error < Decimal("0.01")
    assert report.maximum_cross_mode_relative_difference > Decimal("0.06")
    assert any("no causal attribution" in limitation for limitation in report.limitations)
    assert any("point estimates only" in limitation for limitation in report.limitations)

    validate_seqax_cost_calibration(
        report,
        profile_root=profile_root,
        contract=contract,
    )
    forged = report.model_copy(update={"fixed_residual_ns": Decimal(0)})
    with pytest.raises(ValueError, match="REPORT_REPLAY_MISMATCH"):
        validate_seqax_cost_calibration(
            forged,
            profile_root=profile_root,
            contract=contract,
        )


def test_calibration_rejects_cross_mode_schedule_and_runtime_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root, contract, receipt, _costs = _synthetic_evidence(tmp_path, monkeypatch)
    receipt.results += (receipt.results[0],)
    with pytest.raises(ValueError, match="RESULT_SET_MISMATCH"):
        derive_seqax_cost_calibration(profile_root, contract)

    profile_root, contract, receipt, _costs = _synthetic_evidence(
        tmp_path / "schedule", monkeypatch
    )
    receipt.results[1].invocation.schedule_sha256 = "f" * 64
    with pytest.raises(ValueError, match="SCHEDULE_MISMATCH"):
        derive_seqax_cost_calibration(profile_root, contract)

    profile_root, contract, receipt, _costs = _synthetic_evidence(tmp_path / "other", monkeypatch)
    receipt.results[0].invocation.runtime = receipt.results[0].invocation.runtime.model_copy(
        update={"jax": "forged"}
    )
    with pytest.raises(ValueError, match="EXECUTION_IDENTITY_MISMATCH"):
        derive_seqax_cost_calibration(profile_root, contract)


def test_calibration_requires_external_contract_and_new_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root, contract, _receipt, _costs = _synthetic_evidence(tmp_path, monkeypatch)
    wrong = contract.model_copy(update={"maximum_in_surface_relative_error": Decimal("0.03")})
    with pytest.raises(ValueError, match="EXTERNAL_CONTRACT_MISMATCH"):
        derive_seqax_cost_calibration(profile_root, wrong)

    report = derive_seqax_cost_calibration(profile_root, contract)
    monkeypatch.setattr(
        calibration, "derive_seqax_cost_calibration", lambda *_args, **_kwargs: report
    )
    with pytest.raises(ValueError, match="OUTPUT_OVERLAPS_PROFILE"):
        write_seqax_cost_calibration(
            profile_root / "derived.json",
            profile_root=profile_root,
            contract=contract,
        )
    assert not (profile_root / "derived.json").exists()
    output = tmp_path / "derived" / "report.json"
    assert (
        write_seqax_cost_calibration(
            output,
            profile_root=profile_root,
            contract=contract,
        )
        == report
    )
    assert calibration.SeqaxCostCalibrationReport.model_validate_json(output.read_text()) == report
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        write_seqax_cost_calibration(
            output,
            profile_root=profile_root,
            contract=contract,
        )


def test_cost_calibration_commands_require_external_contract() -> None:
    parser = _parser()
    derive = parser.parse_args(
        [
            "calibrate-seqax-cost",
            "profile",
            "--contract",
            "contract.json",
            "--output",
            "report.json",
        ]
    )
    verify = parser.parse_args(
        [
            "verify-seqax-cost-calibration",
            "report.json",
            "--profile-root",
            "profile",
            "--contract",
            "contract.json",
        ]
    )

    assert derive.command == "calibrate-seqax-cost"
    assert verify.command == "verify-seqax-cost-calibration"
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate-seqax-cost", "profile", "--output", "report.json"])
