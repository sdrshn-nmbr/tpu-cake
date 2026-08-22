from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.matmul_collective_surface_correctness import (
    CORRECTNESS_PATTERN_SCHEMA,
    CORRECTNESS_PATTERNS,
)
from tpu_cake.matmul_collective_surface_correctness_oracle import (
    ORACLE_PATTERN_SCHEMA,
    ORACLE_PATTERNS,
)
from tpu_cake.matmul_collective_surface_correctness_protocol import (
    PROTOCOL_PATTERN_SCHEMA,
    PROTOCOL_PATTERNS,
    MatmulCollectiveSurfaceCorrectnessProtocol,
    default_matmul_collective_surface_correctness_protocol,
)

CONTRACT_PATH = Path("contracts/matmul-collective-surface-correctness-v1.json")


def test_committed_correctness_protocol_is_canonical_and_parent_bound() -> None:
    committed = MatmulCollectiveSurfaceCorrectnessProtocol.model_validate_json(
        CONTRACT_PATH.read_text()
    )
    expected = default_matmul_collective_surface_correctness_protocol()

    assert committed == expected
    assert committed.parent_compile.source_commit == "6dead4dfa23e912fa6352452d1a9480cca9d1f7b"
    assert len(committed.scenarios) == 20
    assert len(committed.calibration_scenarios) == 16
    assert len(committed.holdout_scenarios) == 4
    assert len(committed.patterns.ordered_patterns) == 5
    assert CORRECTNESS_PATTERN_SCHEMA == ORACLE_PATTERN_SCHEMA == PROTOCOL_PATTERN_SCHEMA
    assert CORRECTNESS_PATTERNS == ORACLE_PATTERNS == PROTOCOL_PATTERNS


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("patterns", "constant_formula"), "A=0;B=0;C=0"),
        (("patterns", "signed_lhs_sequence"), [1] * 16),
        (("parent_compile", "manifest_file_sha256"), "0" * 64),
        (("absolute_tolerance",), 1.0),
        (("strategies",), ["pallas_bidirectional_ring", "xla_reduce_scatter"]),
        (("initial_execution_split",), "holdout"),
    ),
)
def test_correctness_protocol_rejects_rebinding(path, value) -> None:
    payload = json.loads(CONTRACT_PATH.read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((ValidationError, ValueError)):
        MatmulCollectiveSurfaceCorrectnessProtocol.model_validate(payload)
