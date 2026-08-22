from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceDesignReport,
    MatmulCollectiveSurfaceSplit,
    SurfaceCalibrationObservation,
    default_matmul_collective_surface_design_contract,
    derive_matmul_collective_surface_design_report,
    fit_surface_model,
    validate_matmul_collective_surface_design_report,
    write_matmul_collective_surface_design_report,
)

CONTRACT = Path("contracts/matmul-collective-surface-design-v1.json")


def test_surface_design_contract_is_canonical_and_keeps_holdout_untouched() -> None:
    saved = MatmulCollectiveSurfaceDesignContract.model_validate_json(CONTRACT.read_text())

    assert saved == default_matmul_collective_surface_design_contract()
    assert len(saved.design_id) == 64
    assert tuple(value.name for value in saved.calibration_scenarios) == tuple(
        f"calibration-{index}" for index in range(16)
    )
    assert tuple(value.name for value in saved.holdout_scenarios) == tuple(
        f"holdout-{index}" for index in range(4)
    )
    assert not any((value.m, value.k, value.n) == (1024, 65536, 1024) for value in saved.scenarios)
    assert saved.prediction_rule_predeclared
    assert saved.acceptance_bounds_predeclared
    assert saved.supports_shape_interpolation
    assert not saved.supports_shape_extrapolation
    assert saved.feature_scale_divisor_ns == 1_000.0
    assert saved.fit_rule.startswith("joint-nonnegative-affine")
    prior_shapes = {(value.m, value.k, value.n) for value in saved.prior_observation_inventory}
    assert not prior_shapes & {(value.m, value.k, value.n) for value in saved.holdout_scenarios}


def test_surface_design_replays_distinct_plans_and_full_rank_features() -> None:
    contract = default_matmul_collective_surface_design_contract()
    report = derive_matmul_collective_surface_design_report(contract)

    assert report.design_id == contract.design_id
    assert len(report.arms) == 2 * len(contract.scenarios)
    assert report.calibration_feature_rank == 6
    assert report.calibration_condition_number < contract.maximum_condition_number
    assert report.calibration_compute_hbm_correlation < contract.maximum_compute_hbm_correlation
    assert report.holdouts_inside_calibration_hull
    assert {value.predicted_limiting_resource for value in report.calibration_arms} >= {
        "compute",
        "hbm",
        "ici",
    }
    by_scenario: dict[str, set[str]] = {}
    for arm in report.arms:
        by_scenario.setdefault(arm.scenario_name, set()).add(arm.physical_schedule_sha256)
    assert all(len(values) == 2 for values in by_scenario.values())
    assert max(value.peak_live_vmem_bytes_per_device for value in report.arms) < 64 * 1024**2
    assert {value.collective_hbm_scratch_bytes_per_device > 0 for value in report.arms} == {
        False,
        True,
    }
    for strategy in contract.strategies:
        arms = tuple(value for value in report.calibration_arms if value.strategy is strategy)
        for resource in ("compute", "hbm", "ici"):
            assert sum(value.predicted_limiting_resource == resource for value in arms) >= 4


def test_surface_contract_rejects_moving_a_holdout_into_calibration() -> None:
    contract = default_matmul_collective_surface_design_contract()
    payload = contract.model_dump(mode="json", exclude_computed_fields=True)
    payload["scenarios"][-1]["split"] = MatmulCollectiveSurfaceSplit.CALIBRATION

    with pytest.raises(ValidationError, match="split/name mismatch"):
        MatmulCollectiveSurfaceDesignContract.model_validate_json(json.dumps(payload))


def test_surface_contract_rejects_a_previously_observed_holdout() -> None:
    contract = default_matmul_collective_surface_design_contract()
    payload = contract.model_dump(mode="json", exclude_computed_fields=True)
    holdout = payload["scenarios"][-1]
    payload["prior_observation_inventory"][0] |= {
        "m": holdout["m"],
        "k": holdout["k"],
        "n": holdout["n"],
    }

    with pytest.raises(ValidationError, match="holdout was previously observed"):
        MatmulCollectiveSurfaceDesignContract.model_validate_json(json.dumps(payload))


def test_surface_design_rejects_any_holdout_outside_the_calibration_hull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_prediction._holdouts_inside_calibration_hull",
        lambda _contract: False,
    )

    with pytest.raises(ValueError, match="HOLDOUT_EXTRAPOLATION"):
        derive_matmul_collective_surface_design_report(contract)


def test_surface_design_report_is_write_once_and_independently_replayable(
    tmp_path: Path,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    output = tmp_path / "design.json"

    expected = write_matmul_collective_surface_design_report(output, contract)
    observed = MatmulCollectiveSurfaceDesignReport.model_validate_json(output.read_text())
    validate_matmul_collective_surface_design_report(observed, contract)
    assert observed == expected

    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        write_matmul_collective_surface_design_report(output, contract)

    tampered = observed.model_copy(update={"physical_authority": "tampered"})
    with pytest.raises(ValueError, match="REPORT_REPLAY_MISMATCH"):
        validate_matmul_collective_surface_design_report(tampered, contract)


def _synthetic_calibration(
    contract: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCalibrationObservation, ...]:
    report = derive_matmul_collective_surface_design_report(contract)
    coefficients = np.asarray([100_000.0, 110_000.0, 2.0, 3.0, 4.0, 4.5])
    observations = []
    for arm in report.calibration_arms:
        is_first = arm.strategy is contract.strategies[0]
        divisor = contract.feature_scale_divisor_ns
        ici = float(arm.ici_time_floor_ns) / divisor
        features = np.asarray(
            [
                float(is_first),
                float(not is_first),
                float(arm.compute_time_floor_ns) / divisor,
                float(arm.hbm_time_floor_ns) / divisor,
                ici if is_first else 0.0,
                0.0 if is_first else ici,
            ]
        )
        observations.append(
            SurfaceCalibrationObservation(
                scenario_name=arm.scenario_name,
                strategy=arm.strategy,
                median_ns=float(features @ coefficients),
            )
        )
    return tuple(observations)


def test_surface_fit_uses_only_the_exact_calibration_inventory() -> None:
    contract = default_matmul_collective_surface_design_contract()
    observations = _synthetic_calibration(contract)

    model = fit_surface_model(contract, observations)

    assert model.coefficient_names == (
        "xla_intercept",
        "pallas_intercept",
        "shared_compute",
        "shared_hbm",
        "xla_ici",
        "pallas_ici",
    )
    assert model.maximum_calibration_relative_error < 1e-10
    leaked = SurfaceCalibrationObservation(
        scenario_name=contract.holdout_scenarios[0].name,
        strategy=contract.strategies[0],
        median_ns=1.0,
    )
    with pytest.raises(ValueError, match="CALIBRATION_INVENTORY_MISMATCH"):
        fit_surface_model(contract, (*observations, leaked))


def test_surface_fit_rejects_rank_deficient_physical_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    observations = _synthetic_calibration(contract)
    report = derive_matmul_collective_surface_design_report(contract)
    collapsed = tuple(
        value.model_copy(
            update={
                "compute_time_floor_ns": report.calibration_arms[0].compute_time_floor_ns,
                "hbm_time_floor_ns": report.calibration_arms[0].hbm_time_floor_ns,
                "ici_time_floor_ns": report.calibration_arms[0].ici_time_floor_ns,
            }
        )
        for value in report.arms
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_prediction.derive_matmul_collective_surface_design_report",
        lambda _contract: report.model_copy(update={"arms": collapsed}),
    )

    with pytest.raises(ValueError, match="CALIBRATION_FEATURE_RANK"):
        fit_surface_model(contract, observations)
