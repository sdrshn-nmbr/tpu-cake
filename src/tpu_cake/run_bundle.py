from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    CorrectnessResult,
    EvidencePhase,
    EvidencePhaseName,
    KernelExperiment,
    RunReceipt,
    RunStatus,
    SearchProvenance,
)
from tpu_cake.cost_model import CostModelReport
from tpu_cake.ledger import RunState, read_ledger_history
from tpu_cake.receipt import _relative_json, counter_expectation, validate_receipt
from tpu_cake.receipt_metrics import build_receipt_metrics
from tpu_cake.runner import MatmulRunResult, RunMode, _source_state
from tpu_cake.search import (
    MatmulSearchContract,
    MatmulSearchResult,
    validate_matmul_search_result,
)
from tpu_cake.workloads.distributed_matmul import distributed_matmul_experiment
from tpu_cake.xprof_evidence import assess_capture
from tpu_cake.xprof_export import export_xprof_capture


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(root: Path, path: Path, role: ArtifactRole) -> ArtifactReference:
    path = path.resolve()
    return ArtifactReference(
        path=str(path.relative_to(root.resolve())),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
    )


def _load_result(path: Path, expected_mode: RunMode) -> MatmulRunResult:
    result = MatmulRunResult.model_validate_json(path.read_text())
    if result.mode is not expected_mode:
        raise ValueError(f"RUN_MODE_MISMATCH expected={expected_mode.value} observed={result.mode}")
    return result


def _validate_execution_ledger(path: Path, result: MatmulRunResult) -> None:
    if not path.is_file():
        raise ValueError(f"EXECUTION_LEDGER_MISSING path={path}")
    terminal = {
        RunMode.TIMING: RunState.TIMED,
        RunMode.TRACE: RunState.TRACED,
        RunMode.COUNTERS: RunState.COUNTERED,
    }[result.mode]
    expected = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        terminal,
    )
    observed = tuple(event.state for event in read_ledger_history(path, result.run_id))
    if observed != expected:
        raise ValueError(
            f"EXECUTION_LEDGER_HISTORY_MISMATCH mode={result.mode.value} "
            f"expected={expected} observed={observed}"
        )


def _ensure_exports(mode_root: Path) -> None:
    if list(mode_root.rglob("hlo_stats.json")):
        return
    export_xprof_capture(mode_root / "profile", mode_root / "xprof")


def build_distributed_matmul_receipt(
    root: Path, *, search_root: Path | None = None
) -> RunReceipt:
    root = root.resolve()
    finalizer_root = root / "finalizer"
    _source_state(Path(__file__).resolve().parents[2], finalizer_root)
    finalizer_state = json.loads((finalizer_root / "source_state.json").read_text())
    if finalizer_state["git_dirty"]:
        raise ValueError("RECEIPT_FINALIZER_REQUIRES_CLEAN_COMMITTED_SOURCE")
    timing = _load_result(root / "timing" / "result.json", RunMode.TIMING)
    trace = _load_result(root / "trace" / "result.json", RunMode.TRACE)
    counters = _load_result(root / "counters" / "result.json", RunMode.COUNTERS)
    for mode, result in (("timing", timing), ("trace", trace), ("counters", counters)):
        _validate_execution_ledger(root / mode / "ledger.sqlite", result)
    identities = {
        (result.schedule_sha256, result.lhs_sha256, result.rhs_sha256, result.output_sha256)
        for result in (timing, trace, counters)
    }
    if len(identities) != 1:
        raise ValueError("RUNS_DO_NOT_SHARE_SCHEDULE_INPUTS_AND_OUTPUT")
    model_input = json.loads((root / "timing" / "cost_model_input.json").read_text())
    expected_experiment = distributed_matmul_experiment(
        schedule_sha256=timing.schedule_sha256,
        mesh_size=int(model_input["mesh_size"]),
        m=int(model_input["m"]),
        k=int(model_input["k"]),
        n=int(model_input["n"]),
        warmup_iterations=timing.warmup_iterations,
        measured_iterations=timing.measured_iterations,
    )
    experiment_path = root / "timing" / "experiment.json"
    experiment = KernelExperiment.model_validate_json(experiment_path.read_text())
    if experiment != expected_experiment:
        raise ValueError("PREDECLARED_EXPERIMENT_DOES_NOT_MATCH_TIMING_RESULT")
    source_states = [
        json.loads((root / mode / "source_state.json").read_text())
        for mode in ("timing", "trace", "counters")
    ]
    if any(state["git_dirty"] for state in source_states):
        raise ValueError("RECEIPT_REQUIRES_CLEAN_COMMITTED_SOURCE")
    source_identities = {
        (state["git_commit"], state["uv_lock_sha256"]) for state in source_states
    }
    if len(source_identities) != 1:
        raise ValueError("RUNS_DO_NOT_SHARE_SOURCE_AND_DEPENDENCY_IDENTITY")

    search_provenance = None
    search_artifact_specs: list[tuple[Path, ArtifactRole]] = []
    if search_root is not None:
        search_root = search_root.resolve()
        if not search_root.is_relative_to(root):
            raise ValueError("SEARCH_EVIDENCE_MUST_BE_INSIDE_RUN_ROOT")
        contract_path = search_root / "contract.json"
        result_path = search_root / "result.json"
        contract = MatmulSearchContract.model_validate_json(contract_path.read_text())
        search_result = MatmulSearchResult.model_validate_json(result_path.read_text())
        validate_matmul_search_result(
            search_root,
            contract,
            search_result,
            recompute_schedules=False,
        )
        if search_result.winner is None:
            raise ValueError("SEARCH_DID_NOT_PROMOTE_A_WINNER")
        winner = next(
            candidate
            for candidate in contract.candidates
            if candidate.name == search_result.winner
        )
        expected_shape = (
            int(model_input["mesh_size"]),
            int(model_input["m"]),
            int(model_input["k"]),
            int(model_input["n"]),
            timing.warmup_iterations,
            timing.measured_iterations,
        )
        if (
            contract.mesh_size,
            contract.m,
            contract.k,
            contract.n,
            contract.warmup_iterations,
            contract.measured_iterations,
        ) != expected_shape:
            raise ValueError("SEARCH_WORKLOAD_DOES_NOT_MATCH_FINALIST")
        invocation = json.loads((root / "timing" / "invocation.json").read_text())
        if (invocation.get("tile_m"), invocation.get("tile_n")) != (
            winner.tile_m,
            winner.tile_n,
        ):
            raise ValueError("SEARCH_WINNER_TILE_DOES_NOT_MATCH_FINALIST")
        winner_results = [
            MatmulRunResult.model_validate_json(
                (search_root / run_path / "result.json").read_text()
            )
            for run_path in search_result.run_results
            if Path(run_path).name == winner.name
        ]
        if {result.schedule_sha256 for result in winner_results} != {
            timing.schedule_sha256
        }:
            raise ValueError("SEARCH_WINNER_SCHEDULE_DOES_NOT_MATCH_FINALIST")
        search_sources = [
            json.loads((search_root / run_path / "source_state.json").read_text())
            for run_path in search_result.run_results
        ]
        search_source_identities = {
            (state["git_commit"], state["uv_lock_sha256"]) for state in search_sources
        }
        if search_source_identities != source_identities:
            raise ValueError("SEARCH_SOURCE_DOES_NOT_MATCH_FINALIST")
        search_provenance = SearchProvenance(
            search_id=contract.search_id,
            winner=winner.name,
            tile_m=winner.tile_m,
            tile_n=winner.tile_n,
            winner_schedule_sha256=timing.schedule_sha256,
            contract_sha256=_sha256(contract_path),
            result_sha256=_sha256(result_path),
            run_count=len(search_result.run_results),
        )
        for path in sorted(search_root.rglob("*")):
            if not path.is_file():
                continue
            role = (
                ArtifactRole.SEARCH_CONTRACT
                if path == contract_path
                else ArtifactRole.SEARCH_RESULT
                if path == result_path
                else ArtifactRole.SEARCH_EVIDENCE
            )
            search_artifact_specs.append((path, role))

    _ensure_exports(root / "trace")
    _ensure_exports(root / "counters")
    trace_assessment = assess_capture(root / "trace", experiment.profile)
    counter_assessment = assess_capture(root / "counters", counter_expectation(experiment))
    assessment_path = root / "profile_assessment.json"
    assessment_payload = _relative_json(
        {
            "timing_trace": trace_assessment.model_dump(mode="json"),
            "counter_trace": counter_assessment.model_dump(mode="json"),
        },
        root,
    )
    assessment_path.write_text(
        json.dumps(
            assessment_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    passed = all(result.passed for result in (timing, trace, counters))
    passed = passed and trace_assessment.accepted and counter_assessment.accepted

    trace_xplane = next((root / "trace" / "profile").rglob("*.xplane.pb"))
    counter_xplane = next((root / "counters" / "profile").rglob("*.xplane.pb"))
    trace_hlo_stats = next((root / "trace").rglob("hlo_stats.json"))
    counter_hlo_stats = next((root / "counters").rglob("hlo_stats.json"))
    artifact_specs = [
        (experiment_path, ArtifactRole.EXPERIMENT),
        (root / "timing" / "distributed.xdsl", ArtifactRole.DISTRIBUTED_IR),
        (root / "timing" / "physical.xdsl", ArtifactRole.PHYSICAL_IR),
        (root / "timing" / "lowered_pallas.py", ArtifactRole.PALLAS_SOURCE),
        (root / "timing" / "stablehlo.txt", ArtifactRole.STABLEHLO),
        (root / "timing" / "compiler_hlo.txt", ArtifactRole.COMPILER_HLO),
        (root / "timing" / "lhs.npy", ArtifactRole.CORRECTNESS_INPUT),
        (root / "timing" / "rhs.npy", ArtifactRole.CORRECTNESS_INPUT),
        (root / "timing" / "output.npy", ArtifactRole.CORRECTNESS_OUTPUT),
        (root / "timing" / "oracle.npy", ArtifactRole.ORACLE_OUTPUT),
        (root / "timing" / "result.json", ArtifactRole.TIMING_SAMPLES),
        (root / "trace" / "result.json", ArtifactRole.TRACE_RESULT),
        (root / "counters" / "result.json", ArtifactRole.COUNTER_RESULT),
        (root / "timing" / "ledger.sqlite", ArtifactRole.EXECUTION_LEDGER),
        (root / "trace" / "ledger.sqlite", ArtifactRole.EXECUTION_LEDGER),
        (root / "counters" / "ledger.sqlite", ArtifactRole.EXECUTION_LEDGER),
        (trace_xplane, ArtifactRole.TIMING_TRACE),
        (counter_xplane, ArtifactRole.COUNTER_TRACE),
        (trace_hlo_stats, ArtifactRole.HLO_STATS),
        (counter_hlo_stats, ArtifactRole.HLO_STATS),
        (assessment_path, ArtifactRole.PROFILE_ASSESSMENT),
        (root / "timing" / "cost_model_input.json", ArtifactRole.COST_MODEL_INPUT),
        (root / "timing" / "cost_model.json", ArtifactRole.COST_MODEL),
        (root / "roofline-input.json", ArtifactRole.ROOFLINE_INPUT),
        (root / "roofline" / "report.md", ArtifactRole.ROOFLINE_REPORT),
        (root / "roofline" / "metrics.json", ArtifactRole.ROOFLINE_METRICS),
        (root / "timing" / "invocation.json", ArtifactRole.INVOCATION),
        (root / "trace" / "invocation.json", ArtifactRole.INVOCATION),
        (root / "counters" / "invocation.json", ArtifactRole.INVOCATION),
        (root / "timing" / "profiler_config.json", ArtifactRole.PROFILER_CONFIG),
        (root / "trace" / "profiler_config.json", ArtifactRole.PROFILER_CONFIG),
        (root / "counters" / "profiler_config.json", ArtifactRole.PROFILER_CONFIG),
        (root / "timing" / "source_state.json", ArtifactRole.SOURCE_STATE),
        (root / "trace" / "source_state.json", ArtifactRole.SOURCE_STATE),
        (root / "counters" / "source_state.json", ArtifactRole.SOURCE_STATE),
        (root / "timing" / "source_diff.patch", ArtifactRole.SOURCE_DIFF),
        (root / "trace" / "source_diff.patch", ArtifactRole.SOURCE_DIFF),
        (root / "counters" / "source_diff.patch", ArtifactRole.SOURCE_DIFF),
        (finalizer_root / "source_state.json", ArtifactRole.SOURCE_STATE),
        (finalizer_root / "source_diff.patch", ArtifactRole.SOURCE_DIFF),
        *search_artifact_specs,
    ]
    artifact_roles = {path.resolve(): role for path, role in artifact_specs}
    for phase, result in (
        ("timing", timing),
        ("trace", trace),
        ("counters", counters),
    ):
        for artifact in result.artifacts:
            path = (root / phase / Path(artifact.path).name).resolve()
            previous = artifact_roles.setdefault(path, artifact.role)
            if previous is not artifact.role:
                raise ValueError(
                    f"RUN_ARTIFACT_ROLE_CONFLICT path={path} first={previous.value} "
                    f"second={artifact.role.value}"
                )
    artifact_specs = list(artifact_roles.items())
    artifacts = tuple(_reference(root, path, role) for path, role in artifact_specs)
    phase_paths: dict[EvidencePhaseName, list[str]] = {
        phase: [] for phase in EvidencePhaseName
    }
    for artifact in artifacts:
        first_component = Path(artifact.path).parts[0]
        phase = (
            EvidencePhaseName(first_component)
            if first_component in {"timing", "trace", "counters", "finalizer"}
            else EvidencePhaseName.AGGREGATE
        )
        phase_paths[phase].append(artifact.path)
    phases = tuple(
        EvidencePhase(name=phase, artifact_paths=tuple(paths))
        for phase, paths in phase_paths.items()
    )
    cost_report = CostModelReport.model_validate_json(
        (root / "timing" / "cost_model.json").read_text()
    )
    metrics = build_receipt_metrics(
        root,
        timing,
        cost_report,
        trace_assessment,
        counter_assessment,
    )
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        schedule_sha256=timing.schedule_sha256,
        status=RunStatus.PASSED if passed else RunStatus.REJECTED,
        runtime=timing.runtime,
        correctness=CorrectnessResult(
            passed=all(result.passed for result in (timing, trace, counters)),
            oracle=experiment.workload.numerical.reference,
            maximum_absolute_error=max(
                result.maximum_absolute_error for result in (timing, trace, counters)
            ),
            maximum_relative_error=max(
                result.maximum_relative_error for result in (timing, trace, counters)
            ),
        ),
        required_semantic_properties=(),
        metrics=metrics,
        artifacts=artifacts,
        phases=phases,
        search_provenance=search_provenance,
    )
    if receipt.status is RunStatus.PASSED:
        validate_receipt(receipt, experiment, root=root)
    receipt_path = root / "receipt.json"
    receipt_path.write_text(receipt.model_dump_json(indent=2) + "\n")

    return receipt
