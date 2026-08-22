from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.artifacts import file_sha256, resolve_recorded_artifact
from tpu_cake.contracts import ArtifactReference, ArtifactRole, experiment_artifact_json
from tpu_cake.identity import (
    LEGACY_SEMANTIC_IDENTITY_SCHEMA,
    SEMANTIC_IDENTITY_SCHEMA,
    array_sha256,
    model_identity_sha256,
    semantic_sha256,
)
from tpu_cake.ledger import RunState, payload_sha256, read_ledger_history
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.pallas_lowering import (
    PALLAS_EXECUTION_SCHEMA,
    lower_physical_matmul_to_pallas,
    validate_saved_pallas_plan,
)
from tpu_cake.runner import (
    MatmulCollectiveStrategy,
    MatmulRunResult,
    RunMode,
    run_distributed_matmul,
    validate_profiler_contract,
)
from tpu_cake.workloads.distributed_matmul import (
    distributed_matmul_experiment,
    distributed_matmul_schedule,
)


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
        return model_identity_sha256(self, exclude={"search_id"})


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
    digest = file_sha256(path)
    if digest != sha256:
        raise ValueError(f"SEARCH_ARTIFACT_HASH_CHANGED path={path}")


def _resolve_artifact(run_path: Path, artifact: ArtifactReference) -> Path:
    return resolve_recorded_artifact(
        run_path,
        artifact.path,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
    )


def _named_artifact(
    run_path: Path,
    result: MatmulRunResult,
    name: str,
    role: ArtifactRole,
) -> Path:
    matches = [
        artifact
        for artifact in result.artifacts
        if Path(artifact.path).name == name and artifact.role is role
    ]
    if len(matches) != 1:
        raise ValueError(f"SEARCH_ARTIFACT_CONTRACT_MISMATCH name={name} role={role.value}")
    return _resolve_artifact(run_path, matches[0])


def _validate_saved_run_evidence(
    run_path: Path,
    contract: MatmulSearchContract,
    candidate: MatmulSearchCandidate,
    result: MatmulRunResult,
) -> None:
    physical = _named_artifact(run_path, result, "physical.xdsl", ArtifactRole.PHYSICAL_IR)
    pallas = _named_artifact(run_path, result, "lowered_pallas.py", ArtifactRole.PALLAS_SOURCE)
    if file_sha256(physical) != result.schedule_sha256:
        raise ValueError(f"SEARCH_SCHEDULE_ARTIFACT_MISMATCH candidate={candidate.name}")
    if file_sha256(pallas) != result.pallas_source_sha256:
        raise ValueError(f"SEARCH_PALLAS_ARTIFACT_MISMATCH candidate={candidate.name}")
    plan = validate_saved_pallas_plan(
        physical,
        pallas,
        schedule_sha256=result.schedule_sha256,
        pallas_source_sha256=result.pallas_source_sha256,
    )
    if (plan.tile_m, plan.tile_n) != (candidate.tile_m, candidate.tile_n):
        raise ValueError(f"SEARCH_PHYSICAL_TILE_MISMATCH candidate={candidate.name}")
    if plan.mesh_size != contract.mesh_size:
        raise ValueError(f"SEARCH_PALLAS_PLAN_MISMATCH candidate={candidate.name}")

    lhs = np.load(
        _named_artifact(run_path, result, "lhs.npy", ArtifactRole.CORRECTNESS_INPUT),
        allow_pickle=False,
    )
    rhs = np.load(
        _named_artifact(run_path, result, "rhs.npy", ArtifactRole.CORRECTNESS_INPUT),
        allow_pickle=False,
    )
    output = np.load(
        _named_artifact(run_path, result, "output.npy", ArtifactRole.CORRECTNESS_OUTPUT),
        allow_pickle=False,
    )
    oracle = np.load(
        _named_artifact(run_path, result, "oracle.npy", ArtifactRole.ORACLE_OUTPUT),
        allow_pickle=False,
    )
    if lhs.shape != (contract.m, contract.k) or rhs.shape != (contract.k, contract.n):
        raise ValueError(f"SEARCH_INPUT_SHAPE_MISMATCH candidate={candidate.name}")
    if output.shape != (contract.m, contract.n) or oracle.shape != output.shape:
        raise ValueError(f"SEARCH_OUTPUT_SHAPE_MISMATCH candidate={candidate.name}")
    if (
        array_sha256(lhs) != result.lhs_sha256
        or array_sha256(rhs) != result.rhs_sha256
        or array_sha256(output) != result.output_sha256
    ):
        raise ValueError(f"SEARCH_ARRAY_IDENTITY_MISMATCH candidate={candidate.name}")
    absolute = np.abs(output - oracle)
    denominator = np.maximum(np.abs(oracle), np.finfo(np.float32).tiny)
    maximum_absolute_error = float(absolute.max())
    maximum_relative_error = float((absolute / denominator).max())
    if not math.isclose(
        maximum_absolute_error, result.maximum_absolute_error, rel_tol=0, abs_tol=1e-12
    ) or not math.isclose(
        maximum_relative_error, result.maximum_relative_error, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"SEARCH_REPORTED_ERROR_MISMATCH candidate={candidate.name}")
    passed = bool(np.allclose(output, oracle, atol=1e-3, rtol=1e-3))
    if result.passed is not passed:
        raise ValueError(f"SEARCH_CORRECTNESS_VERDICT_MISMATCH candidate={candidate.name}")

    samples = list(result.samples_ns)
    expected_median = int(statistics.median(samples))
    expected_p90 = _percentile(samples, 0.9)
    expected_coefficient = (
        statistics.pstdev(samples) / statistics.mean(samples)
        if len(samples) > 1 and statistics.mean(samples)
        else None
    )
    if result.median_ns != expected_median or result.p90_ns != expected_p90:
        raise ValueError(f"SEARCH_TIMING_STATISTIC_MISMATCH candidate={candidate.name}")
    if expected_coefficient is None:
        if result.coefficient_of_variation is not None:
            raise ValueError(f"SEARCH_TIMING_STATISTIC_MISMATCH candidate={candidate.name}")
    elif result.coefficient_of_variation is None or not math.isclose(
        result.coefficient_of_variation,
        expected_coefficient,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(f"SEARCH_TIMING_STATISTIC_MISMATCH candidate={candidate.name}")

    experiment_path = _named_artifact(run_path, result, "experiment.json", ArtifactRole.EXPERIMENT)
    expected_experiment = distributed_matmul_experiment(
        schedule_sha256=result.schedule_sha256,
        mesh_size=contract.mesh_size,
        m=contract.m,
        k=contract.k,
        n=contract.n,
        warmup_iterations=contract.warmup_iterations,
        measured_iterations=contract.measured_iterations,
        collective_strategy=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER.value,
    )
    if experiment_path.read_text() != experiment_artifact_json(expected_experiment) + "\n":
        raise ValueError(f"SEARCH_EXPERIMENT_MISMATCH candidate={candidate.name}")

    profiler_path = _named_artifact(
        run_path, result, "profiler_config.json", ArtifactRole.PROFILER_CONFIG
    )
    validate_profiler_contract(RunMode.TIMING, json.loads(profiler_path.read_text()))
    invocation_path = _named_artifact(run_path, result, "invocation.json", ArtifactRole.INVOCATION)
    invocation = json.loads(invocation_path.read_text())
    identity_schema = invocation.get("identity_schema", LEGACY_SEMANTIC_IDENTITY_SCHEMA)
    source_state_path = _named_artifact(
        run_path, result, "source_state.json", ArtifactRole.SOURCE_STATE
    )
    source_diff_path = _named_artifact(
        run_path, result, "source_diff.patch", ArtifactRole.SOURCE_DIFF
    )
    source_state = json.loads(source_state_path.read_text())
    if source_state.get("git_dirty") is not False:
        raise ValueError(f"SEARCH_SOURCE_IS_DIRTY candidate={candidate.name}")
    if (
        source_state.get("source_diff_sha256")
        != file_sha256(source_diff_path)
    ):
        raise ValueError(f"SEARCH_SOURCE_DIFF_MISMATCH candidate={candidate.name}")

    expected_run_id = semantic_sha256(
        "distributed-matmul-run",
        RunMode.TIMING.value,
        str(contract.mesh_size),
        str(contract.m),
        str(contract.k),
        str(contract.n),
        str(candidate.tile_m),
        str(candidate.tile_n),
        schema=identity_schema,
    )
    if result.run_id != expected_run_id:
        raise ValueError(f"SEARCH_RUN_ID_MISMATCH candidate={candidate.name}")

    distributed = _named_artifact(run_path, result, "distributed.xdsl", ArtifactRole.DISTRIBUTED_IR)
    stablehlo = _named_artifact(run_path, result, "stablehlo.txt", ArtifactRole.STABLEHLO)
    compiler_hlo = _named_artifact(run_path, result, "compiler_hlo.txt", ArtifactRole.COMPILER_HLO)
    ledger = _named_artifact(run_path, result, "ledger.sqlite", ArtifactRole.EXECUTION_LEDGER)
    created_payload = {
        "mode": RunMode.TIMING.value,
        "mesh_size": contract.mesh_size,
        "m": contract.m,
        "k": contract.k,
        "n": contract.n,
        "tile_m": candidate.tile_m,
        "tile_n": candidate.tile_n,
    }
    if "identity_schema" in invocation:
        created_payload["identity_schema"] = identity_schema
    if "pallas_execution_schema" in invocation:
        created_payload["pallas_execution_schema"] = invocation["pallas_execution_schema"]
    expected_payloads = (
        created_payload,
        {"distributed_ir_sha256": file_sha256(distributed)},
        {
            "physical_ir_sha256": result.schedule_sha256,
            "schedule_sha256": result.schedule_sha256,
            "pallas_source_sha256": result.pallas_source_sha256,
            **(
                {"pallas_execution_schema": invocation["pallas_execution_schema"]}
                if "pallas_execution_schema" in invocation
                else {}
            ),
        },
        {
            "stablehlo_sha256": file_sha256(stablehlo),
            "compiler_hlo_sha256": file_sha256(compiler_hlo),
            "compile_duration_ns": result.compile_duration_ns,
        },
        {
            "lhs_sha256": result.lhs_sha256,
            "rhs_sha256": result.rhs_sha256,
            "output_sha256": result.output_sha256,
            "oracle_sha256": array_sha256(oracle),
        },
        {
            "measured_iterations": result.measured_iterations,
            "warmup_iterations": result.warmup_iterations,
            "median_ns": result.median_ns,
            "p90_ns": result.p90_ns,
            "sample_count": len(samples),
        },
    )
    history = read_ledger_history(ledger, result.run_id)
    expected_states = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.TIMED,
    )
    if tuple(event.state for event in history) != expected_states or tuple(
        event.payload_sha256 for event in history
    ) != tuple(payload_sha256(payload) for payload in expected_payloads):
        raise ValueError(f"SEARCH_LEDGER_EVIDENCE_MISMATCH candidate={candidate.name}")


def _validate_resumed_result(
    path: Path,
    contract: MatmulSearchContract,
    candidate: MatmulSearchCandidate,
    result: MatmulRunResult,
    *,
    interpret: bool,
    recompute_schedule: bool,
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
        "collective_strategy": MatmulCollectiveStrategy.XLA_REDUCE_SCATTER.value,
        "interpret": interpret,
    }
    identity_schema = invocation.get("identity_schema", LEGACY_SEMANTIC_IDENTITY_SCHEMA)
    if "identity_schema" in invocation:
        if identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError(f"STALE_SEARCH_IDENTITY_SCHEMA candidate={candidate.name}")
        expected_invocation["identity_schema"] = identity_schema
    if "pallas_execution_schema" in invocation:
        if invocation["pallas_execution_schema"] != PALLAS_EXECUTION_SCHEMA:
            raise ValueError(f"STALE_SEARCH_PALLAS_SCHEMA candidate={candidate.name}")
        expected_invocation["pallas_execution_schema"] = PALLAS_EXECUTION_SCHEMA
    if invocation != expected_invocation:
        raise ValueError(f"STALE_SEARCH_INVOCATION candidate={candidate.name}")
    expected_backend = "cpu" if interpret else "tpu"
    if (
        result.mode is not RunMode.TIMING
        or result.backend != expected_backend
        or result.device_count != contract.mesh_size
        or result.collective_strategy is not MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
        or result.warmup_iterations != contract.warmup_iterations
        or result.measured_iterations != contract.measured_iterations
        or len(result.samples_ns) != contract.measured_iterations
    ):
        raise ValueError(f"STALE_SEARCH_RESULT candidate={candidate.name}")
    if recompute_schedule:
        plan = lower_physical_matmul_to_pallas(
            lower_distributed_matmul(
                distributed_matmul_schedule(
                    mesh_size=contract.mesh_size,
                    m=contract.m,
                    k=contract.k,
                    n=contract.n,
                ),
                tile=MatmulTile(candidate.tile_m, candidate.tile_n),
                collective_implementation=(
                    MatmulCollectiveStrategy.XLA_REDUCE_SCATTER.lowering_implementation()
                ),
            )
        )
        if (
            result.schedule_sha256 != plan.schedule_sha256
            or result.pallas_source_sha256 != plan.source_sha256()
        ):
            raise ValueError(f"STALE_SEARCH_RESULT candidate={candidate.name}")
    for artifact in result.artifacts:
        artifact_path = _resolve_artifact(path, artifact)
        _validate_artifact(artifact_path, artifact.size_bytes, artifact.sha256)
    _validate_saved_run_evidence(path, contract, candidate, result)
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
    recompute_schedule: bool = True,
) -> MatmulRunResult:
    result_path = path / "result.json"
    if result_path.is_file():
        result = MatmulRunResult.model_validate_json(result_path.read_text())
        _validate_resumed_result(
            path,
            contract,
            candidate,
            result,
            interpret=interpret,
            recompute_schedule=recompute_schedule,
        )
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
        collective_strategy=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
        interpret=interpret,
    )


def run_matmul_search(
    output_dir: Path,
    contract: MatmulSearchContract,
    *,
    interpret: bool = False,
    write_result: bool = True,
    recompute_schedules: bool = True,
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
                    _load_or_run(
                        path,
                        contract,
                        candidate,
                        interpret=interpret,
                        recompute_schedule=recompute_schedules,
                    ),
                    path,
                )
            )

    if not all(result.passed for _, _, result, _ in run_results):
        failed = [name for _, name, result, _ in run_results if not result.passed]
        raise ValueError(f"SEARCH_CORRECTNESS_FAILED candidates={failed}")
    input_hashes = {(result.lhs_sha256, result.rhs_sha256) for _, _, result, _ in run_results}
    if len(input_hashes) != 1:
        raise ValueError("SEARCH_INPUTS_ARE_NOT_MATCHED")
    output_hashes_by_candidate: dict[str, set[str]] = {
        candidate.name: set() for candidate in contract.candidates
    }
    for _, name, result, _ in run_results:
        output_hashes_by_candidate[name].add(result.output_sha256)
    if any(len(hashes) != 1 for hashes in output_hashes_by_candidate.values()):
        raise ValueError("SEARCH_OUTPUTS_ARE_NOT_DETERMINISTIC")

    samples_by_candidate: dict[str, list[int]] = {
        candidate.name: [] for candidate in contract.candidates
    }
    runtime_identities = {result.runtime for _, _, result, _ in run_results}
    if len(runtime_identities) != 1:
        raise ValueError("SEARCH_RUNTIME_IDENTITIES_ARE_NOT_MATCHED")
    source_states = [
        json.loads((path / "source_state.json").read_text()) for _, _, _, path in run_results
    ]
    if any(state["git_dirty"] for state in source_states):
        raise ValueError("SEARCH_REQUIRES_CLEAN_COMMITTED_SOURCE")
    source_identities = {(state["git_commit"], state["uv_lock_sha256"]) for state in source_states}
    if len(source_identities) != 1:
        raise ValueError("SEARCH_SOURCE_IDENTITIES_ARE_NOT_MATCHED")
    candidate_code_identities: dict[str, set[tuple[str, str]]] = {
        candidate.name: set() for candidate in contract.candidates
    }
    for _, name, run_result, _ in run_results:
        candidate_code_identities[name].add(
            (run_result.schedule_sha256, run_result.pallas_source_sha256)
        )
    if any(len(identities) != 1 for identities in candidate_code_identities.values()):
        raise ValueError("SEARCH_CANDIDATE_CODE_IDENTITIES_ARE_NOT_STABLE")
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
                (
                    round_medians[baseline_name][round_index]
                    - round_medians[candidate.name][round_index]
                )
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
    recompute_schedules: bool = True,
) -> None:
    required_paths = {
        f"round-{round_index:02d}/{candidate.name}"
        for round_index in range(contract.rounds)
        for candidate in contract.candidates
    }
    if set(expected.run_results) != required_paths or any(
        not (output_dir / path / "result.json").is_file() for path in required_paths
    ):
        raise ValueError("SEARCH_RESULT_RUN_SET_IS_INCOMPLETE")
    observed = run_matmul_search(
        output_dir,
        contract,
        interpret=interpret,
        write_result=False,
        recompute_schedules=recompute_schedules,
    )
    if observed != expected:
        raise ValueError("SEARCH_RESULT_DOES_NOT_MATCH_VERIFIED_RUNS")
