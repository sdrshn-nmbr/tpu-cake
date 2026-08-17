from __future__ import annotations

import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    CorrectnessResult,
    EvidencePhase,
    EvidencePhaseName,
    KernelExperiment,
    ProfileExpectation,
    RunReceipt,
    RunStatus,
)
from tpu_cake.cost_model import CostModelReport
from tpu_cake.ledger import ExperimentLedger, RunState
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)
from tpu_cake.receipt import validate_receipt
from tpu_cake.runner import MatmulRunResult, RunMode, _source_state
from tpu_cake.workloads.distributed_matmul import distributed_matmul_experiment
from tpu_cake.xprof_evidence import assess_capture, capture_metrics
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
    with ExperimentLedger(path) as ledger:
        observed = tuple(event.state for event in ledger.history(result.run_id))
    if observed != expected:
        raise ValueError(
            f"EXECUTION_LEDGER_HISTORY_MISMATCH mode={result.mode.value} "
            f"expected={expected} observed={observed}"
        )


def _ensure_exports(mode_root: Path) -> None:
    if list(mode_root.rglob("hlo_stats.json")):
        return
    export_xprof_capture(mode_root / "profile", mode_root / "xprof")


def _timing_metrics(root: Path, result: MatmulRunResult) -> tuple[Metric, ...]:
    if not result.samples_ns:
        raise ValueError("TIMING_RUN_HAS_NO_SAMPLES")
    timing_path = root / "timing" / "result.json"
    source = MetricSource(
        artifact_sha256=_sha256(timing_path),
        artifact_path=str(timing_path.relative_to(root)),
        tool="tpu-cake",
        field="samples_ns",
    )
    interval = MeasurementInterval(
        scope=f"{len(result.samples_ns)} warm synchronized distributed matmul executions"
    )
    ordered = sorted(result.samples_ns)
    p90_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))
    definitions = (
        (
            "median_device_duration",
            Decimal(statistics.median(result.samples_ns)),
            "median",
            "median(samples_ns)",
        ),
        (
            "p90_device_duration",
            Decimal(ordered[p90_index]),
            "nearest_rank_p90",
            "sort(samples_ns)[round((N-1)*0.9)]",
        ),
    )
    return tuple(
        Metric(
            name=name,
            quantity=Quantity(value=value, unit=Unit.NANOSECOND),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(name=formula, version="1", expression=expression),
        )
        for name, value, formula, expression in definitions
    )


def _roofline_metrics(root: Path) -> tuple[Metric, ...]:
    path = root / "roofline" / "metrics.json"
    payload = json.loads(path.read_text())
    source = MetricSource(
        artifact_sha256=_sha256(path),
        artifact_path=str(path.relative_to(root)),
        tool="roofline skill",
        field="kernel roofline metrics",
    )
    interval = MeasurementInterval(scope="one synchronized distributed matmul execution")
    lower_bound = Decimal(str(payload["lower_bound_time_s"]))
    measured = Decimal(str(payload["measured_time_s"]))
    return (
        Metric(
            name="roofline_two_resource_time_floor",
            quantity=Quantity(value=lower_bound, unit=Unit.SECOND),
            kind=MeasurementKind.ESTIMATED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="roofline_two_resource_floor",
                version="1",
                expression="max(operations/compute_peak,total_bytes/hbm_bandwidth)",
            ),
        ),
        Metric(
            name="measured_to_two_resource_floor",
            quantity=Quantity(value=measured / lower_bound, unit=Unit.RATIO),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="measured_to_two_resource_floor",
                version="1",
                expression="measured_time/two_resource_time_floor",
            ),
            numerator=Quantity(value=measured, unit=Unit.SECOND),
            denominator=Quantity(value=lower_bound, unit=Unit.SECOND),
        ),
    )


def _three_resource_gap_metric(
    root: Path,
    timing: MatmulRunResult,
    cost_report: CostModelReport,
) -> Metric:
    if timing.median_ns is None:
        raise ValueError("TIMING_RUN_HAS_NO_MEDIAN")
    floor_metric = next(
        metric for metric in cost_report.metrics if metric.name == "idealized_time_floor"
    )
    measured = Decimal(timing.median_ns)
    floor = floor_metric.quantity.value
    timing_path = root / "timing" / "result.json"
    return Metric(
        name="measured_to_compute_hbm_ici_floor",
        quantity=Quantity(value=measured / floor, unit=Unit.RATIO),
        kind=MeasurementKind.DERIVED,
        interval=MeasurementInterval(scope="one synchronized distributed matmul execution"),
        sources=(
            MetricSource(
                artifact_sha256=_sha256(timing_path),
                artifact_path=str(timing_path.relative_to(root)),
                tool="tpu-cake",
                field="median_ns",
            ),
            *floor_metric.sources,
        ),
        formula=FormulaIdentity(
            name="measured_to_compute_hbm_ici_floor",
            version="1",
            expression="measured_median_ns/max(compute_floor,hbm_floor,ici_floor)",
        ),
        numerator=Quantity(value=measured, unit=Unit.NANOSECOND),
        denominator=Quantity(value=floor, unit=Unit.NANOSECOND),
    )


def _relative_json(value: object, root: Path) -> object:
    if isinstance(value, dict):
        return {key: _relative_json(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_json(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), ".")
    return value


def build_distributed_matmul_receipt(root: Path) -> RunReceipt:
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

    _ensure_exports(root / "trace")
    _ensure_exports(root / "counters")
    trace_assessment = assess_capture(root / "trace", experiment.profile)
    counter_contract = experiment.profile.model_dump()
    counter_contract.update(
        require_tensor_core_activity=True,
        require_hbm_read_counters=True,
        require_hbm_write_counters=True,
        require_cycle_counters=True,
        minimum_counter_device_planes=experiment.target.chip_count,
        required_timed_hlo_markers=tuple(
            marker
            for marker in experiment.profile.required_timed_hlo_markers
            if marker != "pallas_call"
        ),
    )
    counter_expectation = ProfileExpectation.model_validate(counter_contract)
    counter_assessment = assess_capture(root / "counters", counter_expectation)
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
    artifact_specs = (
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
    )
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
    metrics = list(_timing_metrics(root, timing))
    metrics.extend(_roofline_metrics(root))
    metrics.extend(cost_report.metrics)
    metrics.append(_three_resource_gap_metric(root, timing, cost_report))
    metrics.extend(
        metric.model_copy(update={"name": f"timing_trace.{metric.name}"})
        for metric in capture_metrics(trace_assessment.capture)
    )
    metrics.extend(
        metric.model_copy(update={"name": f"counter_trace.{metric.name}"})
        for metric in capture_metrics(counter_assessment.capture)
    )
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        schedule_sha256=timing.schedule_sha256,
        status=RunStatus.PASSED if passed else RunStatus.REJECTED,
        runtime=timing.runtime,
        correctness=CorrectnessResult(
            passed=all(result.passed for result in (timing, trace, counters)),
            oracle="NumPy float32 matmul over the exact BF16 device inputs",
            maximum_absolute_error=max(
                result.maximum_absolute_error for result in (timing, trace, counters)
            ),
            maximum_relative_error=max(
                result.maximum_relative_error for result in (timing, trace, counters)
            ),
        ),
        required_semantic_properties=(),
        metrics=tuple(metrics),
        artifacts=artifacts,
        phases=phases,
    )
    if receipt.status is RunStatus.PASSED:
        validate_receipt(receipt, experiment, root=root)
    receipt_path = root / "receipt.json"
    receipt_path.write_text(receipt.model_dump_json(indent=2) + "\n")

    return receipt
