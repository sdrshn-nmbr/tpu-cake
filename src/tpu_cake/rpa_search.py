from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from xprof import profile_data

from tpu_cake.contracts import RuntimeIdentity, SourceFileContract
from tpu_cake.rpa_bundle import validate_fused_rpa_run
from tpu_cake.rpa_runner import run_fused_rpa
from tpu_cake.runner import RunMode
from tpu_cake.workloads.inkling_rpa import inkling_fused_rpa_experiment


class RpaSearchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    query_block_size: int = Field(gt=0)
    kv_block_size: int = Field(gt=0)
    query_cluster_size: int = Field(gt=0)
    kv_cluster_size: int = Field(gt=0)

    @computed_field
    @property
    def block_sizes(self) -> tuple[int, int, int, int]:
        return (
            self.query_block_size,
            self.kv_block_size,
            self.query_cluster_size,
            self.kv_cluster_size,
        )

    @model_validator(mode="after")
    def block_sizes_are_legal_for_the_fixed_rpa_fixture(self) -> RpaSearchCandidate:
        if self.query_block_size % self.query_cluster_size:
            raise ValueError("query cluster size must divide query block size")
        if self.kv_block_size % self.kv_cluster_size:
            raise ValueError("KV cluster size must divide KV block size")
        if self.kv_block_size % 16 or self.kv_cluster_size % 16:
            raise ValueError("KV block and cluster sizes must be divisible by page size 16")
        return self


class RpaSearchProfilerAdvancedConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tpu_num_chips_to_profile_per_task: int = Field(gt=0)


class RpaSearchProfilerContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: str = Field(pattern="^trace$")
    raise_error_on_start_failure: bool
    enable_hlo_proto: bool
    host_tracer_level: int = Field(ge=0)
    python_tracer_level: int = Field(ge=0)
    advanced_configuration: RpaSearchProfilerAdvancedConfiguration
    libtpu_init_args: str


class RpaSearchContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: str = Field(min_length=1)
    seed: int = 97
    warmup_iterations: int = 5
    measured_iterations: int = 50
    rounds: int = Field(ge=5)
    confirmation_rounds: int = Field(ge=6)
    bootstrap_samples: int = Field(default=10_000, ge=1_000)
    minimum_practical_improvement: float = Field(default=0.01, gt=0, lt=1)
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    profiler: RpaSearchProfilerContract
    candidates: tuple[RpaSearchCandidate, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def candidates_and_protocol_are_unambiguous(self) -> RpaSearchContract:
        names = [candidate.name for candidate in self.candidates]
        blocks = [candidate.block_sizes for candidate in self.candidates]
        if len(names) != len(set(names)) or len(blocks) != len(set(blocks)):
            raise ValueError("RPA search candidates need unique names and block sizes")
        if self.baseline not in names:
            raise ValueError("RPA search baseline must name one candidate")
        if (self.seed, self.warmup_iterations, self.measured_iterations) != (97, 5, 50):
            raise ValueError("RPA search must use the fixed evidence protocol 97/5/50")
        if self.rounds % (2 * len(self.candidates)):
            raise ValueError(
                "RPA search rounds must complete forward and reverse Latin squares"
            )
        if self.confirmation_rounds % 2:
            raise ValueError("RPA confirmation rounds must balance both run orders")
        return self

    @computed_field
    @property
    def search_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class RpaCandidateStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    block_sizes: tuple[int, int, int, int]
    run_count: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    median_run_duration_ns: float = Field(gt=0)
    p90_run_median_ns: int = Field(gt=0)
    median_absolute_deviation_ns: float = Field(ge=0)
    coefficient_of_variation: float = Field(ge=0)
    improvement_over_baseline: float
    improvement_confidence_interval: tuple[float, float]
    promotable: bool


class RpaDeviceTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_fragment: str
    durations_ns: tuple[float, ...] = Field(min_length=1)
    median_ns: float = Field(gt=0)
    p90_ns: float = Field(gt=0)
    xplane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RpaSearchRunEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    candidate: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device_timing_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device_timing: RpaDeviceTiming


class RpaSearchExecutionIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profiler_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    execution_scope: str
    backend_manifest: tuple[SourceFileContract, ...]
    backend_executor: str
    backend_executor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RpaConfirmationStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: str
    candidate: str
    run_count: int = Field(gt=0)
    execution_orders: tuple[tuple[str, str], ...]
    median_improvement: float
    improvement_confidence_interval: tuple[float, float]
    confirmed: bool


class RpaIncompleteAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    file_count: int = Field(gt=0)
    total_size_bytes: int = Field(gt=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RpaSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: str
    provisional_winner: str | None
    winner: str | None
    confirmation: RpaConfirmationStatistics | None
    execution_identity: RpaSearchExecutionIdentity
    matched_input_sha256: tuple[str, ...]
    execution_orders: tuple[tuple[str, ...], ...]
    candidates: tuple[RpaCandidateStatistics, ...]
    runs: tuple[RpaSearchRunEvidence, ...]
    incomplete_attempts: tuple[RpaIncompleteAttempt, ...]


def _execution_orders(contract: RpaSearchContract) -> tuple[tuple[str, ...], ...]:
    names = tuple(candidate.name for candidate in contract.candidates)
    orders = []
    for round_index in range(contract.rounds):
        square = round_index // len(names)
        basis = names if square % 2 == 0 else tuple(reversed(names))
        offset = round_index % len(names)
        order = basis[offset:] + basis[:offset]
        orders.append(order)
    return tuple(orders)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_profiler_config_sha256(
    path: Path,
    contract: RpaSearchContract,
) -> str:
    if json.loads(path.read_text()) != contract.profiler.model_dump(mode="json"):
        raise ValueError("RPA_SEARCH_PROFILER_CONFIG_MISMATCH")
    return _sha256(path)


def _device_timing(
    run_root: Path,
    candidate: RpaSearchCandidate,
    measured_iterations: int,
) -> RpaDeviceTiming:
    xplanes = tuple((run_root / "profile").rglob("*.xplane.pb"))
    if len(xplanes) != 1:
        raise ValueError(f"RPA_SEARCH_XPLANE_COUNT_MISMATCH path={run_root}")
    event_fragment = (
        f"RPAd-p_16-bq_{candidate.query_block_size}_{candidate.query_cluster_size}"
        f"-bkv_{candidate.kv_block_size}_{candidate.kv_cluster_size}"
    )
    profile = profile_data.ProfileData.from_file(xplanes[0])
    try:
        durations = tuple(
            float(event.duration_ns)
            for plane in profile.planes
            if plane.name == "/device:TPU:0"
            for line in plane.lines
            if line.name == "XLA Ops"
            for event in line.events
            if event.name.startswith(f"%{event_fragment}")
            and 'custom_call_target="tpu_custom_call"' in event.name
        )
    finally:
        profile.close()
    if len(durations) != measured_iterations or any(value <= 0 for value in durations):
        raise ValueError(
            "RPA_SEARCH_DECODE_EVENT_PROTOCOL_MISMATCH "
            f"candidate={candidate.name} expected={measured_iterations} "
            f"observed={len(durations)}"
        )
    ordered = sorted(durations)
    p90_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))
    return RpaDeviceTiming(
        event_fragment=event_fragment,
        durations_ns=durations,
        median_ns=float(statistics.median(durations)),
        p90_ns=float(ordered[p90_index]),
        xplane_sha256=_sha256(xplanes[0]),
    )


def _improvement_interval(
    paired_improvements: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    values = np.asarray(paired_improvements, dtype=np.float64)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = np.median(
            generator.choice(values, len(values), replace=True)
        )
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _statistics(
    contract: RpaSearchContract,
    results: dict[str, list[RpaDeviceTiming]],
) -> tuple[RpaCandidateStatistics, ...]:
    baseline_runs = results[contract.baseline]
    baseline_medians = [result.median_ns for result in baseline_runs]
    statistics_by_candidate = []
    for candidate_index, candidate in enumerate(contract.candidates):
        candidate_runs = results[candidate.name]
        run_medians = [result.median_ns for result in candidate_runs]
        paired = [
            (baseline - observed) / baseline
            for baseline, observed in zip(baseline_medians, run_medians, strict=True)
        ]
        interval = _improvement_interval(
            paired,
            samples=contract.bootstrap_samples,
            seed=int(contract.search_id[:16], 16) ^ candidate_index,
        )
        median_duration = float(statistics.median(run_medians))
        median_improvement = float(statistics.median(paired))
        deviation = float(
            statistics.median(abs(value - median_duration) for value in run_medians)
        )
        coefficient = (
            statistics.pstdev(run_medians) / statistics.mean(run_medians)
            if len(run_medians) > 1 and statistics.mean(run_medians)
            else 0.0
        )
        statistics_by_candidate.append(
            RpaCandidateStatistics(
                name=candidate.name,
                block_sizes=candidate.block_sizes,
                run_count=len(candidate_runs),
                sample_count=sum(len(result.durations_ns) for result in candidate_runs),
                median_run_duration_ns=median_duration,
                p90_run_median_ns=_percentile(
                    [round(value) for value in run_medians], 0.9
                ),
                median_absolute_deviation_ns=deviation,
                coefficient_of_variation=coefficient,
                improvement_over_baseline=median_improvement,
                improvement_confidence_interval=interval,
                promotable=(
                    candidate.name != contract.baseline
                    and interval[0] > contract.minimum_practical_improvement
                ),
            )
        )
    return tuple(statistics_by_candidate)


def _confirmation_statistics(
    contract: RpaSearchContract,
    candidate: str,
    execution_orders: tuple[tuple[str, str], ...],
    baseline_runs: list[RpaDeviceTiming],
    candidate_runs: list[RpaDeviceTiming],
) -> RpaConfirmationStatistics:
    paired = [
        (baseline.median_ns - observed.median_ns) / baseline.median_ns
        for baseline, observed in zip(baseline_runs, candidate_runs, strict=True)
    ]
    interval = _improvement_interval(
        paired,
        samples=contract.bootstrap_samples,
        seed=int(contract.search_id[16:32], 16),
    )
    return RpaConfirmationStatistics(
        baseline=contract.baseline,
        candidate=candidate,
        run_count=len(paired),
        execution_orders=execution_orders,
        median_improvement=float(statistics.median(paired)),
        improvement_confidence_interval=interval,
        confirmed=interval[0] > contract.minimum_practical_improvement,
    )


def _expected_result(
    root: Path,
    contract: RpaSearchContract,
    *,
    require_confirmation: bool = True,
) -> RpaSearchResult:
    candidates = {candidate.name: candidate for candidate in contract.candidates}
    results: dict[str, list[RpaDeviceTiming]] = {
        candidate.name: [] for candidate in contract.candidates
    }
    run_evidence = []
    matched_inputs: tuple[str, ...] | None = None
    shared_execution_identity: RpaSearchExecutionIdentity | None = None
    candidate_execution_identities: dict[str, tuple[object, ...]] = {}

    def read_run(relative: Path, name: str) -> RpaDeviceTiming:
        nonlocal matched_inputs, shared_execution_identity
        candidate = candidates[name]
        run_root = root / relative
        result = validate_fused_rpa_run(
            run_root,
            inkling_fused_rpa_experiment(candidate.block_sizes),
            RunMode.TRACE,
        )
        if matched_inputs is None:
            matched_inputs = result.input_sha256
        elif result.input_sha256 != matched_inputs:
            raise ValueError("RPA_SEARCH_INPUT_IDENTITY_MISMATCH")
        source_state_path = run_root / "source_state.json"
        source_state = json.loads(source_state_path.read_text())
        profiler_config_path = run_root / "profiler_config.json"
        execution_identity = RpaSearchExecutionIdentity(
            source_state_sha256=_sha256(source_state_path),
            profiler_config_sha256=_validated_profiler_config_sha256(
                profiler_config_path, contract
            ),
            git_commit=source_state["git_commit"],
            uv_lock_sha256=source_state["uv_lock_sha256"],
            runtime=result.runtime,
            backend=result.backend,
            device_kind=result.device_kind,
            device_count=result.device_count,
            execution_scope=result.execution_scope,
            backend_manifest=result.backend_manifest,
            backend_executor=result.backend_executor,
            backend_executor_sha256=result.backend_executor_sha256,
        )
        if shared_execution_identity is None:
            shared_execution_identity = execution_identity
        elif execution_identity != shared_execution_identity:
            raise ValueError("RPA_SEARCH_EXECUTION_IDENTITY_MISMATCH")
        if (
            result.runtime != contract.runtime
            or result.backend != contract.backend
            or result.device_kind != contract.device_kind
            or result.device_count != contract.device_count
        ):
            raise ValueError("RPA_SEARCH_DECLARED_RUNTIME_MISMATCH")
        candidate_execution_identity = (
            result.schedule_sha256,
            result.pallas_source_sha256,
            result.stablehlo_sha256,
            result.compiler_hlo_sha256,
            result.output_sha256,
            result.oracle_sha256,
        )
        previous_candidate_identity = candidate_execution_identities.setdefault(
            name, candidate_execution_identity
        )
        if candidate_execution_identity != previous_candidate_identity:
            raise ValueError(
                f"RPA_SEARCH_CANDIDATE_EXECUTION_IDENTITY_MISMATCH candidate={name}"
            )
        timing_path = run_root / "device_timing.json"
        saved_timing = RpaDeviceTiming.model_validate_json(timing_path.read_text())
        expected_timing = _device_timing(
            run_root,
            candidate,
            contract.measured_iterations,
        )
        if saved_timing != expected_timing:
            raise ValueError(
                f"RPA_SEARCH_DEVICE_TIMING_REPLAY_MISMATCH candidate={name}"
            )
        run_evidence.append(
            RpaSearchRunEvidence(
                path=relative.as_posix(),
                candidate=name,
                result_sha256=_sha256(run_root / "result.json"),
                device_timing_sha256=_sha256(timing_path),
                device_timing=saved_timing,
            )
        )
        return saved_timing

    for round_index, order in enumerate(_execution_orders(contract)):
        for position, name in enumerate(order):
            relative = Path(f"round-{round_index:02d}") / f"{position:02d}-{name}"
            results[name].append(read_run(relative, name))
    candidate_statistics = _statistics(contract, results)
    promotable = [item for item in candidate_statistics if item.promotable]
    provisional_winner = (
        max(promotable, key=lambda item: item.improvement_over_baseline).name
        if promotable
        else None
    )
    confirmation = None
    winner = None
    if provisional_winner is not None and require_confirmation:
        confirmation_orders = tuple(
            (
                (contract.baseline, provisional_winner)
                if round_index % 2 == 0
                else (provisional_winner, contract.baseline)
            )
            for round_index in range(contract.confirmation_rounds)
        )
        confirmation_results = {contract.baseline: [], provisional_winner: []}
        for round_index, order in enumerate(confirmation_orders):
            for position, name in enumerate(order):
                relative = (
                    Path("confirmation")
                    / f"round-{round_index:02d}"
                    / f"{position:02d}-{name}"
                )
                confirmation_results[name].append(read_run(relative, name))
        confirmation = _confirmation_statistics(
            contract,
            provisional_winner,
            confirmation_orders,
            confirmation_results[contract.baseline],
            confirmation_results[provisional_winner],
        )
        winner = provisional_winner if confirmation.confirmed else None
    if matched_inputs is None or shared_execution_identity is None:
        raise ValueError("RPA_SEARCH_HAS_NO_RUNS")
    return RpaSearchResult(
        search_id=contract.search_id,
        baseline=contract.baseline,
        provisional_winner=provisional_winner,
        winner=winner,
        confirmation=confirmation,
        execution_identity=shared_execution_identity,
        matched_input_sha256=matched_inputs,
        execution_orders=_execution_orders(contract),
        candidates=candidate_statistics,
        runs=tuple(run_evidence),
        incomplete_attempts=_incomplete_attempts(root),
    )


def validate_rpa_search_result(
    root: Path,
    expected_contract: RpaSearchContract,
) -> RpaSearchResult:
    root = root.resolve()
    contract = RpaSearchContract.model_validate_json((root / "contract.json").read_text())
    if contract != expected_contract:
        raise ValueError("RPA_SEARCH_CONTRACT_MISMATCH")
    saved = RpaSearchResult.model_validate_json((root / "result.json").read_text())
    expected = _expected_result(root, contract)
    if saved != expected:
        raise ValueError("RPA_SEARCH_RESULT_REPLAY_MISMATCH")
    return saved


def _incomplete_attempts(search_root: Path) -> tuple[RpaIncompleteAttempt, ...]:
    incomplete_root = search_root / "incomplete"
    if not incomplete_root.exists():
        return ()
    attempts = []
    for attempt in sorted(path for path in incomplete_root.rglob("*-attempt-*") if path.is_dir()):
        files = sorted(path for path in attempt.rglob("*") if path.is_file())
        manifest = [
            {
                "path": path.relative_to(attempt).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ]
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        attempts.append(
            RpaIncompleteAttempt(
                path=attempt.relative_to(search_root).as_posix(),
                file_count=len(files),
                total_size_bytes=sum(item["size_bytes"] for item in manifest),
                manifest_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        )
    return tuple(attempts)


def _archive_incomplete_run(
    search_root: Path,
    run_root: Path,
    reason: str,
) -> Path:
    (run_root / "failure.json").write_text(
        json.dumps({"reason": reason}, sort_keys=True, indent=2) + "\n"
    )
    relative = run_root.relative_to(search_root)
    archive_parent = search_root / "incomplete" / relative.parent
    archive_parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        archive = archive_parent / f"{relative.name}-attempt-{attempt:02d}"
        if not archive.exists():
            run_root.rename(archive)
            return archive
        attempt += 1


def run_rpa_search(
    root: Path,
    contract: RpaSearchContract,
    *,
    kernel: Callable[..., tuple[jax.Array, jax.Array]],
    backend_manifest: tuple[tuple[str, str], ...],
) -> RpaSearchResult:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "contract.json"
    contract_text = contract.model_dump_json(
        indent=2, exclude_computed_fields=True
    ) + "\n"
    if contract_path.exists():
        if contract_path.read_text() != contract_text:
            raise ValueError("RPA_SEARCH_CONTRACT_CHANGED")
    else:
        contract_path.write_text(contract_text)
    candidates = {candidate.name: candidate for candidate in contract.candidates}

    def ensure_run(relative: Path, name: str) -> None:
        candidate = candidates[name]
        run_root = root / relative
        if (run_root / "result.json").exists():
            validate_fused_rpa_run(
                run_root,
                inkling_fused_rpa_experiment(candidate.block_sizes),
                RunMode.TRACE,
            )
            expected_timing = _device_timing(
                run_root,
                candidate,
                contract.measured_iterations,
            )
            timing_path = run_root / "device_timing.json"
            if not timing_path.exists():
                timing_path.write_text(
                    expected_timing.model_dump_json(indent=2) + "\n"
                )
            saved_timing = RpaDeviceTiming.model_validate_json(timing_path.read_text())
            if saved_timing != expected_timing:
                raise ValueError(
                    f"RPA_SEARCH_DEVICE_TIMING_REPLAY_MISMATCH candidate={name}"
                )
            return
        if run_root.exists():
            _archive_incomplete_run(
                root,
                run_root,
                "incomplete run found during resume",
            )
        try:
            run_fused_rpa(
                run_root,
                mode=RunMode.TRACE,
                kernel=kernel,
                backend_manifest=backend_manifest,
                seed=contract.seed,
                warmup_iterations=contract.warmup_iterations,
                measured_iterations=contract.measured_iterations,
                decode_block_sizes=candidate.block_sizes,
            )
        except Exception as error:
            if run_root.exists():
                _archive_incomplete_run(
                    root,
                    run_root,
                    f"{type(error).__name__}: {error}",
                )
            raise
        validate_fused_rpa_run(
            run_root,
            inkling_fused_rpa_experiment(candidate.block_sizes),
            RunMode.TRACE,
        )
        timing = _device_timing(
            run_root,
            candidate,
            contract.measured_iterations,
        )
        (run_root / "device_timing.json").write_text(
            timing.model_dump_json(indent=2) + "\n"
        )

    for round_index, order in enumerate(_execution_orders(contract)):
        for position, name in enumerate(order):
            ensure_run(
                Path(f"round-{round_index:02d}") / f"{position:02d}-{name}",
                name,
            )
    interim = _expected_result(root, contract, require_confirmation=False)
    if interim.provisional_winner is not None:
        for round_index in range(contract.confirmation_rounds):
            order = (
                (contract.baseline, interim.provisional_winner)
                if round_index % 2 == 0
                else (interim.provisional_winner, contract.baseline)
            )
            for position, name in enumerate(order):
                ensure_run(
                    Path("confirmation")
                    / f"round-{round_index:02d}"
                    / f"{position:02d}-{name}",
                    name,
                )
    result = _expected_result(root, contract)
    (root / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    return validate_rpa_search_result(root, contract)
