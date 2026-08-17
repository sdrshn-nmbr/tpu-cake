from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.runner import MatmulRunResult, RunMode
from tpu_cake.search import (
    MatmulSearchCandidate,
    MatmulSearchContract,
    run_matmul_search,
)


def _contract() -> MatmulSearchContract:
    return MatmulSearchContract(
        mesh_size=4,
        m=256,
        k=512,
        n=256,
        warmup_iterations=1,
        measured_iterations=5,
        rounds=5,
        bootstrap_samples=1000,
        candidates=(
            MatmulSearchCandidate(name="whole", tile_m=256, tile_n=256),
            MatmulSearchCandidate(name="tile-128", tile_m=128, tile_n=128),
        ),
    )


def _result(name: str, samples: tuple[int, ...], *, lhs: str = "1" * 64) -> MatmulRunResult:
    return MatmulRunResult(
        run_id=("2" if name == "whole" else "3") * 64,
        mode=RunMode.TIMING,
        backend="tpu",
        device_kind="TPU v7x",
        device_count=4,
        schedule_sha256=("4" if name == "whole" else "5") * 64,
        pallas_source_sha256="6" * 64,
        lhs_sha256=lhs,
        rhs_sha256="7" * 64,
        output_sha256="8" * 64,
        passed=True,
        maximum_absolute_error=0,
        maximum_relative_error=0,
        compile_duration_ns=1,
        warmup_iterations=1,
        measured_iterations=len(samples),
        samples_ns=samples,
        median_ns=int(sum(samples) / len(samples)),
        p90_ns=max(samples),
        coefficient_of_variation=0,
        runtime=RuntimeIdentity(python="3.13"),
        artifacts=(),
    )


def _write_source_state(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "source_state.json").write_text(
        json.dumps({"git_dirty": False, "git_commit": "a" * 40, "uv_lock_sha256": "b" * 64})
    )


def test_search_alternates_order_and_promotes_only_clear_winner(tmp_path, monkeypatch) -> None:
    def fake_load(_path, _contract, candidate, *, interpret):
        assert interpret is False
        _write_source_state(_path)
        samples = (
            (100, 101, 99, 100, 100)
            if candidate.name == "whole"
            else (70, 71, 69, 70, 70)
        )
        return _result(candidate.name, samples)

    monkeypatch.setattr("tpu_cake.search._load_or_run", fake_load)
    result = run_matmul_search(tmp_path / "search", _contract())
    assert result.execution_orders == (
        ("whole", "tile-128"),
        ("tile-128", "whole"),
        ("whole", "tile-128"),
        ("tile-128", "whole"),
        ("whole", "tile-128"),
    )
    assert result.winner == "tile-128"
    assert result.candidates[1].improvement_confidence_interval[0] > 0


def test_search_rejects_unmatched_inputs(tmp_path, monkeypatch) -> None:
    def fake_load(_path, _contract, candidate, *, interpret):
        _write_source_state(_path)
        lhs = "1" * 64 if candidate.name == "whole" else "9" * 64
        return _result(candidate.name, (100, 100, 100), lhs=lhs)

    monkeypatch.setattr("tpu_cake.search._load_or_run", fake_load)
    with pytest.raises(ValueError, match="INPUTS_ARE_NOT_MATCHED"):
        run_matmul_search(tmp_path / "search", _contract())


def test_search_contract_rejects_nondividing_tile() -> None:
    with pytest.raises(ValidationError, match="does not divide"):
        MatmulSearchContract.model_validate(
            {
                **_contract().model_dump(exclude={"search_id"}),
                "candidates": [
                    {"name": "whole", "tile_m": 256, "tile_n": 256},
                    {"name": "bad", "tile_m": 96, "tile_n": 128},
                ],
            }
        )
