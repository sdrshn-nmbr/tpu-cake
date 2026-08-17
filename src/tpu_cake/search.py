from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.ledger import RunState, read_ledger_history
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.runner import MatmulRunResult, RunMode, run_distributed_matmul
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


class MatmulSearchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    tile_m: int = Field(gt=0)
    tile_n: int = Field(gt=0)


class MatmulSearchContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mesh_size: int = Field(gt=0)
    m: int = Field(gt=0)
    k: int = Field(gt=0)
    n: int = Field(gt=0)
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    rounds: int = Field(ge=5)
    bootstrap_samples: int = Field(default=10_000, ge=1_000)
    minimum_practical_improvement: float = Field(default=0.01, gt=0, lt=1)
    candidates: tuple[MatmulSearchCandidate, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def candidates_are_unique_and_legal(self) -> MatmulSearchContract:
        names = [candidate.name for candidate in self.candidates]
        tiles = [(candidate.tile_m, candidate.tile_n) for candidate in self.candidates]
        if len(names) != len(set(names)) or len(tiles) != len(set(tiles)):
            raise ValueError("search candidates need unique names and tile shapes")
        for candidate in self.candidates:
            if self.m % candidate.tile_m or self.n % candidate.tile_n:
                raise ValueError(f"candidate {candidate.name} does not divide M and N")
        return self

    @computed_field
    @property
    def search_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"search_id"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class CandidateStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    tile_m: int
    tile_n: int
    run_count: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    median_ns: float = Field(gt=0)
    p90_ns: int = Field(gt=0)
    median_absolute_deviation_ns: float = Field(ge=0)
    coefficient_of_variation: float = Field(ge=0)
    improvement_over_baseline: float | None = None
    improvement_confidence_interval: tuple[float, float] | None = None
    promotable: bool


class MatmulSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: str
    winner: str | None
    matched_input_sha256: tuple[str, str]
    execution_orders: tuple[tuple[str, ...], ...]
    candidates: tuple[CandidateStatistics, ...]
    run_results: tuple[str, ...]


def _percentile(samples: list[int], fraction: float) -> int:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _improvement_interval(
    paired_round_improvements: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    values = np.asarray(paired_round_improvements, dtype=np.float64)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = generator.choice(values, len(values), replace=True)
        estimates[index] = np.median(draw)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _validate_artifact(path: Path, size_bytes: int, sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"SEARCH_ARTIFACT_MISSING path={path}")
    if path.stat().st_size != size_bytes:
        raise ValueError(f"SEARCH_ARTIFACT_SIZE_CHANGED path={path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"SEARCH_ARTIFACT_HASH_CHANGED path={path}")


def _validate_resumed_result(
    path: Path,
    contract: MatmulSearchContract,
    candidate: MatmulSearchCandidate,
    result: MatmulRunResult,
    *,
    interpret: bool,
) -> None:
    invocation = json.loads((path / "invocation.json").read_text())
    expected_invocation = {
        "mode": RunMode.TIMING.value,
        "mesh_size": contract.mesh_size,
        "m": contract.m,
        "k": contract.k,
        "n": contract.n,
        "warmup_iterations": contract.warmup_iterations,
        "measured_iterations": contract.measured_iterations,
        "tile_m": candidate.tile_m,
        "tile_n": candidate.tile_n,
        "interpret": interpret,
    }
    if invocation != expected_invocation:
        raise ValueError(f"STALE_SEARCH_INVOCATION candidate={candidate.name}")
    plan = lower_physical_matmul_to_pallas(
        lower_distributed_matmul(
            distributed_matmul_schedule(
                mesh_size=contract.mesh_size,
                m=contract.m,
                k=contract.k,
                n=contract.n,
            ),
            tile=MatmulTile(candidate.tile_m, candidate.tile_n),
        )
    )
    expected_backend = "cpu" if interpret else "tpu"
    if (
        result.mode is not RunMode.TIMING
        or result.schedule_sha256 != plan.schedule_sha256
        or result.pallas_source_sha256 != plan.source_sha256()
        or result.backend != expected_backend
        or result.device_count != contract.mesh_size
        or result.warmup_iterations != contract.warmup_iterations
        or result.measured_iterations != contract.measured_iterations
        or len(result.samples_ns) != contract.measured_iterations
    ):
        raise ValueError(f"STALE_SEARCH_RESULT candidate={candidate.name}")
    for artifact in result.artifacts:
        artifact_path = Path(artifact.path)
        _validate_artifact(artifact_path, artifact.size_bytes, artifact.sha256)
    ledger_path = path / "ledger.sqlite"
    if not ledger_path.is_file():
        raise ValueError(f"SEARCH_LEDGER_MISSING candidate={candidate.name}")
    history = tuple(event.state for event in read_ledger_history(ledger_path, result.run_id))
    if history != (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.TIMED,
    ):
        raise ValueError(f"STALE_SEARCH_LEDGER candidate={candidate.name}")


def _load_or_run(
    path: Path,
    contract: MatmulSearchContract,
    candidate: MatmulSearchCandidate,
    *,
    interpret: bool,
) -> MatmulRunResult:
    result_path = path / "result.json"
    if result_path.is_file():
        result = MatmulRunResult.model_validate_json(result_path.read_text())
        _validate_resumed_result(path, contract, candidate, result, interpret=interpret)
        return result
    if path.exists():
        raise ValueError(f"INCOMPLETE_SEARCH_RUN path={path}")
    return run_distributed_matmul(
        path,
        mode=RunMode.TIMING,
        mesh_size=contract.mesh_size,
        m=contract.m,
        k=contract.k,
        n=contract.n,
        warmup_iterations=contract.warmup_iterations,
        measured_iterations=contract.measured_iterations,
        tile_m=candidate.tile_m,
        tile_n=candidate.tile_n,
        interpret=interpret,
    )


def run_matmul_search(
    output_dir: Path,
    contract: MatmulSearchContract,
    *,
    interpret: bool = False,
    write_result: bool = True,
) -> MatmulSearchResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    contract_text = contract.model_dump_json(indent=2, exclude_computed_fields=True) + "\n"
    if contract_path.exists():
        if contract_path.read_text() != contract_text:
            raise ValueError("SEARCH_CONTRACT_CHANGED")
    else:
        contract_path.write_text(contract_text)

    run_results: list[tuple[int, str, MatmulRunResult, Path]] = []
    execution_orders: list[tuple[str, ...]] = []
    for round_index in range(contract.rounds):
        candidates = contract.candidates if round_index % 2 == 0 else contract.candidates[::-1]
        execution_orders.append(tuple(candidate.name for candidate in candidates))
        for candidate in candidates:
            path = output_dir / f"round-{round_index:02d}" / candidate.name
            run_results.append(
                (
                    round_index,
                    candidate.name,
                    _load_or_run(path, contract, candidate, interpret=interpret),
                    path,
                )
            )

    if not all(result.passed for _, _, result, _ in run_results):
        failed = [name for _, name, result, _ in run_results if not result.passed]
        raise ValueError(f"SEARCH_CORRECTNESS_FAILED candidates={failed}")
    input_hashes = {
        (result.lhs_sha256, result.rhs_sha256) for _, _, result, _ in run_results
    }
    if len(input_hashes) != 1:
        raise ValueError("SEARCH_INPUTS_ARE_NOT_MATCHED")

    samples_by_candidate: dict[str, list[int]] = {
        candidate.name: [] for candidate in contract.candidates
    }
    runtime_identities = {result.runtime for _, _, result, _ in run_results}
    if len(runtime_identities) != 1:
        raise ValueError("SEARCH_RUNTIME_IDENTITIES_ARE_NOT_MATCHED")
    source_states = [
        json.loads((path / "source_state.json").read_text())
        for _, _, _, path in run_results
    ]
    if any(state["git_dirty"] for state in source_states):
        raise ValueError("SEARCH_REQUIRES_CLEAN_COMMITTED_SOURCE")
    source_identities = {
        (state["git_commit"], state["uv_lock_sha256"]) for state in source_states
    }
    if len(source_identities) != 1:
        raise ValueError("SEARCH_SOURCE_IDENTITIES_ARE_NOT_MATCHED")
    round_medians: dict[str, dict[int, float]] = {
        candidate.name: {} for candidate in contract.candidates
    }
    for round_index, name, result, _ in run_results:
        samples_by_candidate[name].extend(result.samples_ns)
        round_medians[name][round_index] = float(statistics.median(result.samples_ns))
    baseline_name = contract.candidates[0].name
    baseline_samples = samples_by_candidate[baseline_name]
    baseline_median = float(statistics.median(baseline_samples))
    statistics_by_candidate: list[CandidateStatistics] = []
    for candidate in contract.candidates:
        values = samples_by_candidate[candidate.name]
        median = float(statistics.median(values))
        deviation = float(statistics.median(abs(value - median) for value in values))
        coefficient = statistics.pstdev(values) / statistics.mean(values)
        if candidate.name == baseline_name:
            improvement = None
            interval = None
            promotable = True
        else:
            paired_improvements = [
                (round_medians[baseline_name][round_index] - round_medians[candidate.name][round_index])
                / round_medians[baseline_name][round_index]
                for round_index in range(contract.rounds)
            ]
            improvement = float(statistics.median(paired_improvements))
            seed = int(hashlib.sha256(candidate.name.encode()).hexdigest()[:16], 16)
            interval = _improvement_interval(
                paired_improvements,
                samples=contract.bootstrap_samples,
                seed=seed,
            )
            promotable = interval[0] > contract.minimum_practical_improvement
        statistics_by_candidate.append(
            CandidateStatistics(
                name=candidate.name,
                tile_m=candidate.tile_m,
                tile_n=candidate.tile_n,
                run_count=contract.rounds,
                sample_count=len(values),
                median_ns=median,
                p90_ns=_percentile(values, 0.9),
                median_absolute_deviation_ns=deviation,
                coefficient_of_variation=coefficient,
                improvement_over_baseline=improvement,
                improvement_confidence_interval=interval,
                promotable=promotable,
            )
        )
    promoted = [
        candidate
        for candidate in statistics_by_candidate[1:]
        if candidate.promotable and candidate.median_ns < baseline_median
    ]
    winner = min(promoted, key=lambda candidate: candidate.median_ns).name if promoted else None
    result = MatmulSearchResult(
        search_id=contract.search_id,
        baseline=baseline_name,
        winner=winner,
        matched_input_sha256=next(iter(input_hashes)),
        execution_orders=tuple(execution_orders),
        candidates=tuple(statistics_by_candidate),
        run_results=tuple(str(path.relative_to(output_dir)) for _, _, _, path in run_results),
    )
    if write_result:
        (output_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def validate_matmul_search_result(
    output_dir: Path,
    contract: MatmulSearchContract,
    expected: MatmulSearchResult,
    *,
    interpret: bool = False,
) -> None:
    observed = run_matmul_search(
        output_dir,
        contract,
        interpret=interpret,
        write_result=False,
    )
    if observed != expected:
        raise ValueError("SEARCH_RESULT_DOES_NOT_MATCH_VERIFIED_RUNS")
