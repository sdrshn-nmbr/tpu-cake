from __future__ import annotations

import pytest
from pydantic import ValidationError

from tpu_cake.matmul_collective_surface_calibration_evidence import (
    SurfaceCalibrationCallSample,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    default_matmul_collective_surface_calibration_protocol,
)
from tpu_cake.matmul_collective_surface_calibration_seal import (
    SurfaceCalibrationArmObservation,
    SurfaceHoldoutPredictionInterval,
    _prediction_width_gate,
    derive_surface_calibration_observations,
    seal_surface_calibration_observations,
)
from tpu_cake.matmul_collective_surface_prediction import (
    default_matmul_collective_surface_design_contract,
)
from tpu_cake.runner import MatmulCollectiveStrategy


def _samples() -> tuple[SurfaceCalibrationCallSample, ...]:
    protocol = default_matmul_collective_surface_calibration_protocol()
    scenario_indices = {value: index for index, value in enumerate(protocol.scenarios)}
    samples = []
    sequence = 0
    clock = 100
    for round_index in range(protocol.paired_rounds):
        for scenario_position, scenario in enumerate(protocol.scenario_order(round_index), start=1):
            scenario_index = scenario_indices[scenario]
            for arm_position, strategy in enumerate(protocol.strategy_order(round_index), start=1):
                strategy_index = protocol.strategies.index(strategy)
                median = (
                    100_000
                    + scenario_index * 4_000
                    + strategy_index * 2_000
                    + (round_index % 4) * 20
                )
                for call_index, offset in enumerate((-2, -1, 0, 1, 2)):
                    sequence += 1
                    duration = median + offset
                    samples.append(
                        SurfaceCalibrationCallSample(
                            sequence=sequence,
                            round_index=round_index,
                            scenario_name=scenario,
                            scenario_position=scenario_position,
                            strategy=strategy,
                            arm_position=arm_position,
                            call_index=call_index,
                            resident_pair_sha256="1" * 64,
                            invocation_nonce="2" * 64,
                            worker_pid=42,
                            start_ns=clock,
                            stop_ns=clock + duration,
                            duration_ns=duration,
                        )
                    )
                    clock += duration + 1
    return tuple(samples)


def test_calibration_observations_recompute_every_round_and_arm_median() -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()
    design = default_matmul_collective_surface_design_contract()

    observations = derive_surface_calibration_observations(_samples(), protocol, design)

    assert len(observations) == 32
    assert (
        observations[0].round_medians_ns
        == (
            100_000,
            100_020,
            100_040,
            100_060,
        )
        * 4
    )
    assert observations[0].median_ns == 100_030.0
    assert observations[-1].median_ns == 162_030.0


def test_calibration_seal_replays_bootstrap_predictions_and_width_gate() -> None:
    protocol = default_matmul_collective_surface_calibration_protocol()
    design = default_matmul_collective_surface_design_contract()
    observations = derive_surface_calibration_observations(_samples(), protocol, design)

    seal = seal_surface_calibration_observations(
        observations,
        protocol,
        design,
        calibration_evidence_sha256="3" * 64,
    )
    replay = seal_surface_calibration_observations(
        observations,
        protocol,
        design,
        calibration_evidence_sha256="3" * 64,
    )

    assert seal == replay
    assert len(seal.holdout_predictions) == 8
    assert len(seal.strategy_predictions) == 4
    assert seal.bootstrap_sample_count == 10_000
    assert all(
        len(value) == 64
        for value in (
            seal.bootstrap_index_sha256,
            seal.bootstrap_coefficient_sha256,
            seal.bootstrap_prediction_sha256,
            seal.bootstrap_improvement_sha256,
            seal.seal_sha256,
        )
    )
    assert seal.width_gate_passed
    assert seal.holdout_authorization == "pending_independent_replay"


def test_calibration_arm_observation_rejects_a_forged_median() -> None:
    with pytest.raises(ValidationError, match="ARM_MEDIAN_MISMATCH"):
        SurfaceCalibrationArmObservation(
            scenario_name="calibration-0",
            strategy=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
            round_medians_ns=tuple(range(100, 116)),
            median_ns=108.0,
        )


def test_prediction_width_gate_is_inclusive_only_at_declared_boundary() -> None:
    def intervals(upper: float) -> tuple[SurfaceHoldoutPredictionInterval, ...]:
        return tuple(
            SurfaceHoldoutPredictionInterval(
                scenario_name=f"holdout-{index // 2}",
                strategy=(
                    MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
                    if index % 2 == 0
                    else MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING
                ),
                point_prediction_ns=100.0,
                lower_99pct_ns=90.0,
                upper_99pct_ns=upper,
                relative_width=(upper - 90.0) / 100.0,
            )
            for index in range(8)
        )

    assert _prediction_width_gate(intervals(110.0), 0.2)
    assert not _prediction_width_gate(intervals(110.000_000_1), 0.2)
