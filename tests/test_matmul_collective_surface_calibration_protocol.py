from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.matmul_collective_surface_calibration_protocol import (
    MatmulCollectiveSurfaceCalibrationProtocol,
    default_matmul_collective_surface_calibration_protocol,
    load_matmul_collective_surface_calibration_protocol,
)
from tpu_cake.matmul_collective_surface_prediction import (
    default_matmul_collective_surface_design_contract,
)

CONTRACT = Path("contracts/matmul-collective-surface-calibration-v1.json")


def test_calibration_protocol_is_canonical_and_binds_verified_correctness() -> None:
    saved = MatmulCollectiveSurfaceCalibrationProtocol.model_validate_json(CONTRACT.read_text())
    expected = default_matmul_collective_surface_calibration_protocol()

    assert saved == expected
    assert len(saved.protocol_id) == 64
    assert saved.correctness_parent.attempt_id.startswith("02c589e")
    assert saved.correctness_parent.archive_sha256.startswith("c2f0c459")
    assert saved.correctness_parent.evidence_sha256.startswith("92aaed8b")
    assert saved.correctness_parent.independent_replay_required is True
    assert saved.scenarios == tuple(f"calibration-{index}" for index in range(16))
    assert all(not value.startswith("holdout-") for value in saved.scenarios)
    assert saved.allow_holdout_materialization is False
    assert saved.allow_retry is False
    assert saved.strategy_order(0)[0] is saved.first_timed_strategy
    assert saved.warmup_strategy_order(0)[0] is saved.first_warmup_strategy
    assert saved.permanent_claim_key == (
        "a6334e879bc6d1d2cb9389e28456c86d077722024f4869d3536ca8748da1dc84"
    )


def test_calibration_protocol_defines_exact_balanced_execution_order() -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()

    assert protocol.scenario_order(0) == protocol.scenarios
    assert protocol.scenario_order(7) == protocol.scenarios
    assert protocol.scenario_order(8) == tuple(reversed(protocol.scenarios))
    assert protocol.scenario_order(15) == tuple(reversed(protocol.scenarios))
    orders = tuple(protocol.strategy_order(index) for index in range(protocol.paired_rounds))
    assert sum(order == protocol.strategies for order in orders) == 8
    assert sum(order == tuple(reversed(protocol.strategies)) for order in orders) == 8
    assert all(left != right for left, right in pairwise(orders))
    assert protocol.paired_rounds * len(protocol.scenarios) * len(protocol.strategies) * 5 == 2560
    for scenario_index in range(len(protocol.scenarios)):
        warmups = protocol.warmup_strategy_order(scenario_index)
        assert len(warmups) == 20
        assert sum(value is protocol.strategies[0] for value in warmups) == 10
        assert sum(value is protocol.strategies[1] for value in warmups) == 10
        assert all(left is not right for left, right in pairwise(warmups))


@pytest.mark.parametrize("round_index", [-1, 16])
def test_calibration_protocol_rejects_out_of_range_rounds(round_index: int) -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()

    with pytest.raises(ValueError, match="ROUND_INVALID"):
        protocol.scenario_order(round_index)
    with pytest.raises(ValueError, match="ROUND_INVALID"):
        protocol.strategy_order(round_index)


@pytest.mark.parametrize("scenario_index", [-1, 16])
def test_calibration_protocol_rejects_out_of_range_scenarios(scenario_index: int) -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()

    with pytest.raises(ValueError, match="SCENARIO_INVALID"):
        protocol.warmup_strategy_order(scenario_index)


@pytest.mark.parametrize("replicate_index", [-1, 10000])
def test_calibration_protocol_rejects_out_of_range_bootstrap_replicates(
    replicate_index: int,
) -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()

    with pytest.raises(ValueError, match="BOOTSTRAP_REPLICATE_INVALID"):
        protocol.bootstrap_round_indices(replicate_index)


def test_calibration_protocol_bootstrap_indices_are_exact_and_replayable() -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()

    assert protocol.bootstrap_round_indices(0) == (
        7,
        0,
        12,
        13,
        10,
        15,
        8,
        6,
        11,
        0,
        12,
        8,
        3,
        5,
        12,
        14,
    )
    assert protocol.bootstrap_round_indices(9999) == (
        9,
        9,
        10,
        4,
        11,
        8,
        0,
        0,
        9,
        11,
        12,
        1,
        10,
        5,
        11,
        4,
    )


def test_calibration_protocol_rejects_parent_rebinding_and_holdout_leakage() -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()
    payload = protocol.model_dump(mode="json", exclude_computed_fields=True)
    payload["correctness_parent"]["evidence_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="PROTOCOL_MISMATCH"):
        MatmulCollectiveSurfaceCalibrationProtocol.model_validate_json(json.dumps(payload))

    payload = protocol.model_dump(mode="json", exclude_computed_fields=True)
    payload["scenarios"][-1] = "holdout-0"
    with pytest.raises(ValidationError, match="PROTOCOL_MISMATCH"):
        MatmulCollectiveSurfaceCalibrationProtocol.model_validate_json(json.dumps(payload))


def test_calibration_protocol_load_requires_canonical_design(tmp_path: Path) -> None:
    design = default_matmul_collective_surface_design_contract()
    path = tmp_path / "calibration.json"
    path.write_text(CONTRACT.read_text())

    assert (
        load_matmul_collective_surface_calibration_protocol(path, design).design_id
        == design.design_id
    )

    payload = design.model_dump(mode="json", exclude_computed_fields=True)
    payload["maximum_condition_number"] = 19.0
    forged_design = design.model_construct(**payload)
    with pytest.raises(ValueError, match="DESIGN_MISMATCH"):
        load_matmul_collective_surface_calibration_protocol(path, forged_design)
