from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity
from tpu_cake.runner import MatmulCollectiveStrategy, MatmulRunResult, RunMode
from tpu_cake.search import (
    MatmulSearchCandidate,
    MatmulSearchContract,
    _resolve_artifact,
    run_matmul_search,
    validate_matmul_search_result,
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


def _result(
    name: str,
    samples: tuple[int, ...],
    *,
    lhs: str = "1" * 64,
    output: str = "8" * 64,
) -> MatmulRunResult:
    return MatmulRunResult(
        run_id=("2" if name == "whole" else "3") * 64,
        mode=RunMode.TIMING,
        backend="tpu",
        device_kind="TPU v7x",
        device_count=4,
        collective_strategy=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
        schedule_sha256=("4" if name == "whole" else "5") * 64,
        pallas_source_sha256="6" * 64,
        lhs_sha256=lhs,
        rhs_sha256="7" * 64,
        output_sha256=output,
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
    def fake_load(_path, _contract, candidate, *, interpret, recompute_schedule):
        assert interpret is False
        assert recompute_schedule is True
        _write_source_state(_path)
        samples = (
            (100, 101, 99, 100, 100)
            if candidate.name == "whole"
            else (70, 71, 69, 70, 70)
        )
        result = _result(candidate.name, samples)
        (_path / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
        return result

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
    validate_matmul_search_result(tmp_path / "search", _contract(), result)
    with pytest.raises(ValueError, match="DOES_NOT_MATCH_VERIFIED_RUNS"):
        validate_matmul_search_result(
            tmp_path / "search",
            _contract(),
            result.model_copy(update={"winner": "whole"}),
        )


def test_search_rejects_unmatched_inputs(tmp_path, monkeypatch) -> None:
    def fake_load(_path, _contract, candidate, *, interpret, recompute_schedule):
        assert recompute_schedule is True
        _write_source_state(_path)
        lhs = "1" * 64 if candidate.name == "whole" else "9" * 64
        result = _result(candidate.name, (100, 100, 100), lhs=lhs)
        (_path / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
        return result

    monkeypatch.setattr("tpu_cake.search._load_or_run", fake_load)
    with pytest.raises(ValueError, match="INPUTS_ARE_NOT_MATCHED"):
        run_matmul_search(tmp_path / "search", _contract())


def test_search_rejects_nondeterministic_repeats_within_one_candidate(
    tmp_path, monkeypatch
) -> None:
    def fake_load(_path, _contract, candidate, *, interpret, recompute_schedule):
        _write_source_state(_path)
        round_index = int(_path.parent.name.removeprefix("round-"))
        output = (
            ("8" if round_index == 0 else "9") * 64
            if candidate.name == "whole"
            else "8" * 64
        )
        result = _result(candidate.name, (100, 100, 100), output=output)
        (_path / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
        return result

    monkeypatch.setattr("tpu_cake.search._load_or_run", fake_load)
    with pytest.raises(ValueError, match="OUTPUTS_ARE_NOT_DETERMINISTIC"):
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


def test_saved_search_contract_round_trips_through_strict_schema() -> None:
    contract = _contract()
    encoded = contract.model_dump_json(indent=2, exclude_computed_fields=True)
    assert MatmulSearchContract.model_validate_json(encoded) == contract


def test_search_artifact_resolution_is_confined_to_the_run(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    real = run / "physical.xdsl"
    real.write_text("real")
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    (shadow_root / "physical.xdsl").write_text("shadow")
    monkeypatch.chdir(shadow_root)
    artifact = ArtifactReference(
        path="physical.xdsl",
        size_bytes=4,
        sha256="0" * 64,
        role=ArtifactRole.PHYSICAL_IR,
    )

    assert _resolve_artifact(run, artifact) == real


def test_search_artifact_resolution_rejects_symlinks(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside")
    (run / "physical.xdsl").symlink_to(outside)
    artifact = ArtifactReference(
        path="physical.xdsl",
        size_bytes=7,
        sha256="0" * 64,
        role=ArtifactRole.PHYSICAL_IR,
    )

    with pytest.raises(ValueError, match="ARTIFACT_SYMLINK_FORBIDDEN"):
        _resolve_artifact(run, artifact)
