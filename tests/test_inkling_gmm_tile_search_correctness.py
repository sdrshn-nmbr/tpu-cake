from __future__ import annotations

import numpy as np
import pytest

from tpu_cake.inkling_gmm_tile_search import GmmArmName, GmmPolicyPair
from tpu_cake.inkling_gmm_tile_search_correctness import (
    GmmCorrectnessGateError,
    GmmStageOutputs,
    compare_active_spans,
    fixed_order_fp32_row_matmul,
    measure_operand_sentinels,
    measure_outputs,
    selected_row_cpu_oracle,
    unique_correctness_policies,
    validate_cpu_values,
)


def test_correctness_policy_inventory_contains_each_screen_policy_once() -> None:
    policies = unique_correctness_policies()

    assert len(policies) == 9
    assert len({policy.name for policy in policies}) == 9
    assert policies[0] == GmmPolicyPair(
        gate_up=GmmArmName.INCUMBENT,
        down=GmmArmName.INCUMBENT,
    )
    assert all(
        policy.gate_up is GmmArmName.INCUMBENT or policy.down is GmmArmName.INCUMBENT
        for policy in policies
    )


def test_fixed_order_matmul_really_accumulates_left_to_right_in_fp32() -> None:
    lhs = np.asarray([1e20, -1e20, 1.0], dtype=np.float32)
    rhs = np.ones((3, 1), dtype=np.float32)

    actual = fixed_order_fp32_row_matmul(lhs, rhs)

    assert actual.dtype == np.float32
    assert actual.tolist() == [1.0]


def test_selected_row_oracle_covers_gate_up_silu_and_down() -> None:
    inputs = np.asarray([[1.0, 2.0]], dtype=np.float32)
    gate = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    up = np.asarray([[[2.0, 1.0], [1.0, 3.0]]], dtype=np.float32)
    down = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)

    oracle = selected_row_cpu_oracle(
        inputs,
        gate,
        up,
        down,
        row_index=0,
        local_expert_index=0,
        down_columns=(0, 2),
    )
    expected_gate = np.asarray([1.0, 2.0], dtype=np.float32)
    expected_up = np.asarray([4.0, 7.0], dtype=np.float32)
    hidden = expected_gate / (np.float32(1.0) + np.exp(-expected_gate)) * expected_up
    expected_down = np.asarray(
        [hidden[0] + np.float32(4.0) * hidden[1], 3 * hidden[0] + 6 * hidden[1]],
        dtype=np.float32,
    )

    assert np.array_equal(oracle.gate, expected_gate)
    assert np.array_equal(oracle.up, expected_up)
    assert np.allclose(oracle.down, expected_down, rtol=0, atol=1e-6)


def test_active_span_checks_discriminate_one_bit_and_down_garbage() -> None:
    baseline = GmmStageOutputs(
        gate=np.zeros((2, 3, 2), dtype=np.float32),
        up=np.zeros((2, 3, 2), dtype=np.float32),
        down=np.zeros((2, 3, 3), dtype=np.float32),
    )
    spans = ((0, 1), (1, 3))
    candidate_gate = baseline.gate.copy()
    candidate_gate[1, 2, 0] = np.nextafter(np.float32(0), np.float32(1))
    candidate = GmmStageOutputs(
        gate=candidate_gate,
        up=baseline.up.copy(),
        down=baseline.down.copy(),
    )

    with pytest.raises(GmmCorrectnessGateError, match="ACTIVE_SPAN_MISMATCH"):
        compare_active_spans(baseline, candidate, spans, policy_name="candidate")

    outside_down = baseline.down.copy()
    outside_down[0, 2, 0] = 1.0
    with pytest.raises(GmmCorrectnessGateError, match="DOWN_OUTSIDE_LOCAL_SPAN"):
        measure_outputs(
            GmmStageOutputs(
                gate=baseline.gate,
                up=baseline.up,
                down=outside_down,
            ),
            spans,
        )


def test_output_measurement_checks_only_meaningful_gate_up_rows_for_finiteness() -> None:
    gate = np.zeros((1, 2, 2), dtype=np.float32)
    gate[0, 1, 0] = np.nan
    outputs = GmmStageOutputs(
        gate=gate,
        up=np.zeros((1, 2, 2), dtype=np.float32),
        down=np.zeros((1, 2, 3), dtype=np.float32),
    )

    measurement = measure_outputs(outputs, ((0, 1),))

    assert measurement.gate.nonfinite_count == 0
    assert measurement.down.outside_nonzero_count == 0
    with pytest.raises(GmmCorrectnessGateError, match="NONFINITE_ACTIVE_OUTPUT"):
        measure_outputs(outputs, ((0, 2),))


def test_cpu_tolerance_uses_frozen_combined_absolute_and_relative_bound() -> None:
    within = validate_cpu_values(
        np.asarray([10.0, 0.0], dtype=np.float32),
        np.asarray([10.21, 0.019], dtype=np.float32),
        stage="down",
        absolute_tolerance=0.02,
        relative_tolerance=0.02,
    )

    assert within.maximum_absolute_error == pytest.approx(0.21)
    with pytest.raises(GmmCorrectnessGateError, match="CPU_ORACLE_MISMATCH"):
        validate_cpu_values(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.021], dtype=np.float32),
            stage="down",
            absolute_tolerance=0.02,
            relative_tolerance=0.02,
        )


def test_operand_sentinels_are_seeded_compact_and_discriminating() -> None:
    arrays = {
        "inputs": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "gate": np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4),
        "up": np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4),
        "down": np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4),
    }

    first = measure_operand_sentinels(arrays, seed=17, sentinel_count=8)
    second = measure_operand_sentinels(arrays, seed=17, sentinel_count=8)
    changed = dict(arrays)
    changed["inputs"] = arrays["inputs"] + np.float32(1.0)
    third = measure_operand_sentinels(changed, seed=17, sentinel_count=8)

    assert first == second
    assert tuple(item.name for item in first) == ("inputs", "gate", "up", "down")
    assert all(len(item.indices) == 8 for item in first)
    assert first[0].sentinel_sha256 != third[0].sentinel_sha256
