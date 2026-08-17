from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from decimal import Decimal
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from tpu_cake.artifacts import resolve_recorded_artifact
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    CorrectnessResult,
    EvidencePhase,
    EvidencePhaseName,
    EvidenceProfile,
    KernelExperiment,
    RunReceipt,
    RunStatus,
)
from tpu_cake.dialects.tpu_schedule import TPUSchedule
from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, semantic_sha256
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)
from tpu_cake.receipt import _relative_json, _source_identity, counter_expectation
from tpu_cake.rpa_lowering import lower_inkling_rpa_to_pallas
from tpu_cake.rpa_runner import (
    FusedRpaRunResult,
    fused_rpa_outputs_pass,
    validate_fused_rpa_run_protocol,
)
from tpu_cake.runner import RunMode, _source_state, validate_profiler_contract
from tpu_cake.workloads.inkling_rpa import (
    inkling_fused_rpa_experiment,
    inkling_fused_rpa_inputs,
    inkling_fused_rpa_reference,
)
from tpu_cake.xprof_evidence import assess_capture, capture_metrics, count_profile_events
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
        path=path.relative_to(root.resolve()).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
    )


def _load_declared_array(path: Path, dtype: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if dtype == "bfloat16" and value.dtype == np.dtype("V2"):
        value = value.view(np.dtype(jnp.bfloat16))
    return value


def _load_result(path: Path, mode: RunMode) -> FusedRpaRunResult:
    result = FusedRpaRunResult.model_validate_json(path.read_text())
    if result.mode is not mode:
        raise ValueError(
            f"RPA_RUN_MODE_MISMATCH expected={mode.value} observed={result.mode.value}"
        )
    return result


def _phase_identity(result: FusedRpaRunResult) -> tuple[object, ...]:
    return (
        result.schedule_sha256,
        result.pallas_source_sha256,
        result.stablehlo_sha256,
        result.compiler_hlo_sha256,
        result.input_sha256,
        result.output_sha256,
        result.oracle_sha256,
        result.backend_manifest,
        result.backend_executor,
        result.backend_executor_sha256,
        result.runtime,
        result.backend,
        result.device_kind,
        result.device_count,
        result.execution_scope,
    )


def _require_shared_canonical_identity(
    results: tuple[FusedRpaRunResult, ...], experiment: KernelExperiment
) -> None:
    if (
        len({_phase_identity(result) for result in results}) != 1
        or any(result.schedule_sha256 != experiment.schedule_sha256 for result in results)
    ):
        raise ValueError("RPA_RUNS_DO_NOT_SHARE_CANONICAL_EXECUTION_IDENTITY")


def _require_timing_sample_protocol(result: FusedRpaRunResult) -> None:
    if len(result.samples_ns) != result.measured_iterations or any(
        sample <= 0 for sample in result.samples_ns
    ):
        raise ValueError("RPA_TIMING_SAMPLE_PROTOCOL_MISMATCH")


def _parse_saved_plan(physical_path: Path, lowered_path: Path, result: FusedRpaRunResult):
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(TPUSchedule)
    module = Parser(
        context,
        physical_path.read_text(),
        name=str(physical_path),
    ).parse_module()
    plan = lower_inkling_rpa_to_pallas(module)
    if (
        plan.schedule_sha256 != result.schedule_sha256
        or schedule_sha256(module) != result.schedule_sha256
        or _sha256(physical_path) != result.schedule_sha256
        or plan.source_sha256() != result.pallas_source_sha256
        or _sha256(lowered_path) != result.pallas_source_sha256
        or lowered_path.read_text() != plan.render_executable_source()
    ):
        raise ValueError("RPA_SAVED_LOWERING_IDENTITY_MISMATCH")
    return plan


def _result_artifacts(
    root: Path,
    receipt: RunReceipt,
    phase: str,
    result: FusedRpaRunResult,
) -> dict[str, Path]:
    receipt_artifacts = {artifact.path: artifact for artifact in receipt.artifacts}
    resolved: dict[str, Path] = {}
    for artifact in result.artifacts:
        path = resolve_recorded_artifact(
            root / phase,
            artifact.path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        receipt_artifact = receipt_artifacts.get(relative)
        if receipt_artifact is None or receipt_artifact != artifact.model_copy(
            update={"path": relative}
        ):
            raise ValueError(f"RPA_RESULT_ARTIFACT_RECEIPT_MISMATCH phase={phase} path={relative}")
        if path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise ValueError(f"RPA_RESULT_ARTIFACT_CONTENT_MISMATCH phase={phase} path={relative}")
        resolved[artifact.path] = path
    return resolved


def _validate_phase(
    root: Path,
    receipt: RunReceipt,
    experiment: KernelExperiment,
    phase: str,
    result: FusedRpaRunResult,
) -> tuple[float, float]:
    artifacts = _result_artifacts(root, receipt, phase, result)
    required_names = {
        "invocation.json",
        "profiler_config.json",
        "backend_manifest.json",
        "source_state.json",
        "source_diff.patch",
        "experiment.json",
        "physical.xdsl",
        "lowered_pallas.py",
        "preflight.json",
        "stablehlo.txt",
        "compiler_hlo.txt",
        "ledger.sqlite",
    }
    missing = required_names - artifacts.keys()
    if missing:
        raise ValueError(f"RPA_RESULT_ARTIFACT_SET_INCOMPLETE phase={phase} missing={missing}")
    plan = _parse_saved_plan(
        artifacts["physical.xdsl"],
        artifacts["lowered_pallas.py"],
        result,
    )
    if (
        result.backend != "tpu"
        or re.fullmatch(r"tpu(?: v)?7x(?: lite)?", result.device_kind.strip().lower()) is None
        or result.device_count < experiment.target.chip_count
        or result.execution_scope != plan.execution_scope
        or experiment.workload.execution is None
        or result.execution_scope != experiment.workload.execution.scope
    ):
        raise ValueError(f"RPA_EXECUTION_IDENTITY_MISMATCH phase={phase}")
    invocation = json.loads(artifacts["invocation.json"].read_text())
    expected_invocation = {
        "identity_schema": SEMANTIC_IDENTITY_SCHEMA,
        "mode": result.mode.value,
        "seed": 97,
        "warmup_iterations": result.warmup_iterations,
        "measured_iterations": result.measured_iterations,
        "execution_scope": result.execution_scope,
        "schedule_sha256": result.schedule_sha256,
        "pallas_source_sha256": result.pallas_source_sha256,
        "backend_executor": result.backend_executor,
        "backend_executor_sha256": result.backend_executor_sha256,
    }
    if invocation != expected_invocation:
        raise ValueError(f"RPA_INVOCATION_RESULT_MISMATCH phase={phase}")
    validate_fused_rpa_run_protocol(
        seed=invocation["seed"],
        warmup_iterations=result.warmup_iterations,
        measured_iterations=result.measured_iterations,
    )
    validate_profiler_contract(
        result.mode,
        json.loads(artifacts["profiler_config.json"].read_text()),
    )
    _source_identity(
        artifacts["source_state.json"],
        artifacts["source_diff.patch"],
        require_clean=receipt.status is RunStatus.PASSED,
    )
    saved_experiment = KernelExperiment.model_validate_json(
        artifacts["experiment.json"].read_text()
    )
    if saved_experiment != experiment:
        raise ValueError(f"RPA_EXPERIMENT_MISMATCH phase={phase}")
    manifest = json.loads(artifacts["backend_manifest.json"].read_text())
    expected_manifest = {
        "source_revision": plan.backend_repository_revision,
        "files": [list(item) for item in plan.backend_manifest],
    }
    if (
        manifest != expected_manifest
        or tuple((source.path, source.sha256) for source in result.backend_manifest)
        != plan.backend_manifest
    ):
        raise ValueError(f"RPA_BACKEND_MANIFEST_MISMATCH phase={phase}")
    if (
        result.backend_executor
        != f"{plan.backend_module}.{plan.backend_executor_qualname}"
        or result.backend_executor_sha256 != plan.backend_sha256
        or result.stablehlo_sha256 != _sha256(artifacts["stablehlo.txt"])
        or result.compiler_hlo_sha256 != _sha256(artifacts["compiler_hlo.txt"])
    ):
        raise ValueError(f"RPA_EXECUTED_BACKEND_IDENTITY_MISMATCH phase={phase}")
    preflight = json.loads(artifacts["preflight.json"].read_text())
    if preflight.get("passed") is not True or result.preflight_passed is not True:
        raise ValueError(f"RPA_PREFLIGHT_NOT_PROVEN phase={phase}")

    input_paths = tuple(
        artifacts[f"inputs/{index:02d}-{tensor.name}.npy"]
        for index, tensor in enumerate(experiment.workload.inputs)
    )
    output_paths = tuple(
        artifacts[f"outputs/{index:02d}-{tensor.name}.npy"]
        for index, tensor in enumerate(experiment.workload.outputs)
    )
    oracle_paths = tuple(
        artifacts[f"oracle/{index:02d}-{tensor.name}.npy"]
        for index, tensor in enumerate(experiment.workload.outputs)
    )
    inputs = tuple(
        _load_declared_array(path, tensor.dtype)
        for path, tensor in zip(input_paths, experiment.workload.inputs, strict=True)
    )
    outputs = tuple(
        _load_declared_array(path, tensor.dtype)
        for path, tensor in zip(output_paths, experiment.workload.outputs, strict=True)
    )
    saved_oracles = tuple(
        _load_declared_array(path, tensor.dtype)
        for path, tensor in zip(oracle_paths, experiment.workload.outputs, strict=True)
    )
    for value, tensor in zip(inputs, experiment.workload.inputs, strict=True):
        if value.shape != tensor.shape or str(value.dtype) != tensor.dtype:
            raise ValueError(f"RPA_INPUT_CONTRACT_MISMATCH phase={phase} tensor={tensor.name}")
    for value, tensor in zip(outputs, experiment.workload.outputs, strict=True):
        if value.shape != tensor.shape or str(value.dtype) != tensor.dtype:
            raise ValueError(f"RPA_OUTPUT_CONTRACT_MISMATCH phase={phase} tensor={tensor.name}")
    if tuple(_sha256(path) for path in input_paths) != result.input_sha256:
        raise ValueError(f"RPA_INPUT_IDENTITY_MISMATCH phase={phase}")
    if tuple(_sha256(path) for path in output_paths) != result.output_sha256:
        raise ValueError(f"RPA_OUTPUT_IDENTITY_MISMATCH phase={phase}")
    if tuple(_sha256(path) for path in oracle_paths) != result.oracle_sha256:
        raise ValueError(f"RPA_ORACLE_IDENTITY_MISMATCH phase={phase}")
    expected_inputs = tuple(np.asarray(value) for value in inkling_fused_rpa_inputs(97))
    if any(
        not np.array_equal(saved, expected)
        for saved, expected in zip(inputs, expected_inputs, strict=True)
    ):
        raise ValueError(f"RPA_DETERMINISTIC_INPUT_REPLAY_MISMATCH phase={phase}")
    device_inputs = tuple(jnp.asarray(value) for value in inputs)
    plan.preflight(*device_inputs)
    recomputed_oracles = tuple(
        np.asarray(value) for value in inkling_fused_rpa_reference(device_inputs)
    )
    if any(
        not np.array_equal(saved, recomputed)
        for saved, recomputed in zip(saved_oracles, recomputed_oracles, strict=True)
    ):
        raise ValueError(f"RPA_ORACLE_REPLAY_MISMATCH phase={phase}")
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    for output, oracle in zip(outputs, recomputed_oracles, strict=True):
        output_f32 = output.astype(np.float32)
        oracle_f32 = oracle.astype(np.float32)
        absolute = np.abs(output_f32 - oracle_f32)
        denominator = np.maximum(np.abs(oracle_f32), np.finfo(np.float32).tiny)
        absolute_errors.append(float(absolute.max()))
        relative_errors.append(float((absolute / denominator).max()))
    if any(
        not math.isclose(observed, declared, rel_tol=0, abs_tol=1e-12)
        for observed, declared in zip(
            absolute_errors,
            result.maximum_absolute_errors,
            strict=True,
        )
    ) or any(
        not math.isclose(observed, declared, rel_tol=0, abs_tol=1e-12)
        for observed, declared in zip(
            relative_errors,
            result.maximum_relative_errors,
            strict=True,
        )
    ):
        raise ValueError(f"RPA_REPORTED_ERROR_MISMATCH phase={phase}")
    passed = fused_rpa_outputs_pass(
        outputs,
        recomputed_oracles,
        output_atol=experiment.workload.numerical.absolute_tolerance,
        output_rtol=experiment.workload.numerical.relative_tolerance,
    )
    if passed is not result.passed:
        raise ValueError(f"RPA_CORRECTNESS_VERDICT_MISMATCH phase={phase}")
    if not passed:
        raise ValueError(f"RPA_PHASE_CORRECTNESS_FAILED phase={phase}")

    expected_run_id = semantic_sha256(
        "inkling-fused-rpa-run",
        result.mode.value,
        str(invocation["seed"]),
        str(result.warmup_iterations),
        str(result.measured_iterations),
        result.schedule_sha256,
        result.pallas_source_sha256,
        result.backend_executor,
        result.backend_executor_sha256,
    )
    if expected_run_id != result.run_id:
        raise ValueError(f"RPA_RUN_ID_MISMATCH phase={phase}")
    terminal_state = {
        RunMode.TIMING: RunState.TIMED,
        RunMode.TRACE: RunState.TRACED,
        RunMode.COUNTERS: RunState.COUNTERED,
    }[result.mode]
    expected_states = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        terminal_state,
    )
    terminal_payload: dict[str, object] = {
        "measured_iterations": result.measured_iterations,
        "warmup_iterations": result.warmup_iterations,
    }
    if result.mode is RunMode.TIMING:
        samples = list(result.samples_ns)
        _require_timing_sample_protocol(result)
        expected_median = int(statistics.median(samples))
        ordered = sorted(samples)
        expected_p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
        expected_coefficient = statistics.pstdev(samples) / statistics.mean(samples)
        if (
            result.median_ns != expected_median
            or result.p90_ns != expected_p90
            or result.coefficient_of_variation is None
            or not math.isclose(
                result.coefficient_of_variation,
                expected_coefficient,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("RPA_TIMING_STATISTIC_MISMATCH")
        terminal_payload.update(
            median_ns=result.median_ns,
            p90_ns=result.p90_ns,
            sample_count=len(samples),
        )
    else:
        if result.samples_ns or any(
            value is not None
            for value in (result.median_ns, result.p90_ns, result.coefficient_of_variation)
        ):
            raise ValueError(f"RPA_PROFILE_CONTAINS_TIMING_CLAIMS phase={phase}")
        xplanes = sorted((root / phase / "profile").rglob("*.xplane.pb"))
        if len(xplanes) != 1:
            raise ValueError(f"RPA_PROFILE_XPLANE_COUNT_MISMATCH phase={phase}")
        observed_steps = count_profile_events(
            root / phase / "profile", "inkling_fused_rpa"
        )
        if observed_steps != result.measured_iterations:
            raise ValueError(
                f"RPA_PROFILE_STEP_COUNT_MISMATCH phase={phase} "
                f"expected={result.measured_iterations} observed={observed_steps}"
            )
        terminal_payload.update(
            xplane_sha256=_sha256(xplanes[0]),
            xplane_size_bytes=xplanes[0].stat().st_size,
        )
    expected_payloads = (
        expected_invocation,
        {"physical_ir_sha256": _sha256(artifacts["physical.xdsl"])},
        {
            "schedule_sha256": result.schedule_sha256,
            "pallas_source_sha256": result.pallas_source_sha256,
            "backend_manifest": [list(item) for item in plan.backend_manifest],
            "execution_scope": result.execution_scope,
        },
        {
            "stablehlo_sha256": _sha256(artifacts["stablehlo.txt"]),
            "compiler_hlo_sha256": _sha256(artifacts["compiler_hlo.txt"]),
            "compile_duration_ns": result.compile_duration_ns,
        },
        {
            "output_shapes": [list(value.shape) for value in outputs],
            "maximum_errors": [
                [absolute, relative]
                for absolute, relative in zip(
                    absolute_errors,
                    relative_errors,
                    strict=True,
                )
            ],
        },
        terminal_payload,
    )
    history = read_ledger_history(artifacts["ledger.sqlite"], result.run_id)
    if tuple(event.state for event in history) != expected_states or tuple(
        event.payload_sha256 for event in history
    ) != tuple(ExperimentLedger.payload_sha256(payload) for payload in expected_payloads):
        raise ValueError(f"RPA_LEDGER_EVIDENCE_MISMATCH phase={phase}")
    return max(absolute_errors), max(relative_errors)


def _prefix_capture_metrics(
    prefix: str, metrics: tuple[Metric, ...], root: Path
) -> tuple[Metric, ...]:
    prefixed = []
    for metric in metrics:
        sources = tuple(
            source.model_copy(
                update={
                    "artifact_path": Path(source.artifact_path)
                    .resolve()
                    .relative_to(root.resolve())
                    .as_posix()
                }
            )
            for source in metric.sources
        )
        prefixed.append(
            metric.model_copy(update={"name": f"{prefix}_{metric.name}", "sources": sources})
        )
    return tuple(prefixed)


def _timing_metrics(root: Path, result: FusedRpaRunResult) -> tuple[Metric, ...]:
    result_path = root / "timing" / "result.json"
    source = MetricSource(
        artifact_sha256=_sha256(result_path),
        artifact_path="timing/result.json",
        tool="tpu-cake",
        field="samples_ns",
    )
    interval = MeasurementInterval(scope="one synchronized local-shard fused RPA invocation")
    assert result.median_ns is not None
    assert result.p90_ns is not None
    assert result.coefficient_of_variation is not None
    mean = Decimal(str(statistics.mean(result.samples_ns)))
    deviation = Decimal(str(statistics.pstdev(result.samples_ns)))
    return (
        Metric(
            name="median_synchronized_invocation_duration",
            quantity=Quantity(value=Decimal(result.median_ns), unit=Unit.NANOSECOND),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="sample_median",
                version="1",
                expression="median(samples_ns)",
            ),
        ),
        Metric(
            name="p90_synchronized_invocation_duration",
            quantity=Quantity(value=Decimal(result.p90_ns), unit=Unit.NANOSECOND),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="nearest_rank_p90",
                version="1",
                expression="sorted(samples_ns)[round((n-1)*0.9)]",
            ),
        ),
        Metric(
            name="synchronized_invocation_coefficient_of_variation",
            quantity=Quantity(
                value=Decimal(str(result.coefficient_of_variation)),
                unit=Unit.RATIO,
            ),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="population_coefficient_of_variation",
                version="1",
                expression="population_standard_deviation(samples_ns) / mean(samples_ns)",
            ),
            numerator=Quantity(value=deviation, unit=Unit.NANOSECOND),
            denominator=Quantity(value=mean, unit=Unit.NANOSECOND),
        ),
    )


def _ensure_exports(mode_root: Path) -> None:
    if not list(mode_root.rglob("hlo_stats.json")):
        export_xprof_capture(mode_root / "profile", mode_root / "xprof")


def build_fused_rpa_receipt(root: Path) -> RunReceipt:
    root = root.resolve()
    finalizer_root = root / "finalizer"
    _source_state(Path(__file__).resolve().parents[2], finalizer_root)
    timing = _load_result(root / "timing" / "result.json", RunMode.TIMING)
    trace = _load_result(root / "trace" / "result.json", RunMode.TRACE)
    counters = _load_result(root / "counters" / "result.json", RunMode.COUNTERS)
    experiment = inkling_fused_rpa_experiment()
    _require_shared_canonical_identity((timing, trace, counters), experiment)
    source_identities = {
        _source_identity(
            root / phase / "source_state.json",
            root / phase / "source_diff.patch",
        )
        for phase in ("timing", "trace", "counters", "finalizer")
    }
    if len(source_identities) != 1:
        raise ValueError("RPA_RUNS_DO_NOT_SHARE_SOURCE_IDENTITY")
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
    assessment_path.write_text(json.dumps(assessment_payload, indent=2, sort_keys=True) + "\n")
    artifact_roles: dict[Path, ArtifactRole] = {}
    for phase, result in (
        ("timing", timing),
        ("trace", trace),
        ("counters", counters),
    ):
        for artifact in result.artifacts:
            path = (root / phase / artifact.path).resolve()
            previous = artifact_roles.setdefault(path, artifact.role)
            if previous is not artifact.role:
                raise ValueError(f"RPA_ARTIFACT_ROLE_CONFLICT path={path}")
        artifact_roles[root / phase / "result.json"] = {
            "timing": ArtifactRole.TIMING_SAMPLES,
            "trace": ArtifactRole.TRACE_RESULT,
            "counters": ArtifactRole.COUNTER_RESULT,
        }[phase]
    for phase in ("trace", "counters"):
        for path in sorted((root / phase).rglob("*")):
            if not path.is_file() or path.resolve() in artifact_roles:
                continue
            role = (
                ArtifactRole.HLO_STATS
                if path.name == "hlo_stats.json"
                else ArtifactRole.XPROF_EXPORT
            )
            artifact_roles[path.resolve()] = role
    artifact_roles[assessment_path] = ArtifactRole.PROFILE_ASSESSMENT
    artifact_roles[finalizer_root / "source_state.json"] = ArtifactRole.SOURCE_STATE
    artifact_roles[finalizer_root / "source_diff.patch"] = ArtifactRole.SOURCE_DIFF
    artifacts = tuple(
        _reference(root, path, role)
        for path, role in sorted(artifact_roles.items(), key=lambda item: str(item[0]))
    )
    phase_paths: dict[EvidencePhaseName, list[str]] = {phase: [] for phase in EvidencePhaseName}
    for artifact in artifacts:
        first = Path(artifact.path).parts[0]
        phase = (
            EvidencePhaseName(first)
            if first in {"timing", "trace", "counters", "finalizer"}
            else EvidencePhaseName.AGGREGATE
        )
        phase_paths[phase].append(artifact.path)
    phases = tuple(
        EvidencePhase(name=phase, artifact_paths=tuple(paths))
        for phase, paths in phase_paths.items()
    )
    metrics = (
        *_timing_metrics(root, timing),
        *_prefix_capture_metrics("trace", capture_metrics(trace_assessment.capture), root),
        *_prefix_capture_metrics("counter", capture_metrics(counter_assessment.capture), root),
    )
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        evidence_profile=EvidenceProfile.OPAQUE_RPA_ADAPTER,
        schedule_sha256=timing.schedule_sha256,
        status=(
            RunStatus.PASSED
            if all(result.passed for result in (timing, trace, counters))
            and trace_assessment.accepted
            and counter_assessment.accepted
            else RunStatus.REJECTED
        ),
        runtime=timing.runtime,
        correctness=CorrectnessResult(
            passed=all(result.passed for result in (timing, trace, counters)),
            oracle=experiment.workload.numerical.reference,
            maximum_absolute_error=max(
                error
                for result in (timing, trace, counters)
                for error in result.maximum_absolute_errors
            ),
            maximum_relative_error=max(
                error
                for result in (timing, trace, counters)
                for error in result.maximum_relative_errors
            ),
        ),
        required_semantic_properties=(),
        metrics=metrics,
        artifacts=artifacts,
        phases=phases,
    )
    validate_fused_rpa_receipt(receipt, experiment, root=root)
    (root / "receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    return receipt


def validate_fused_rpa_receipt(
    receipt: RunReceipt,
    experiment: KernelExperiment,
    *,
    root: Path,
) -> None:
    root = root.resolve()
    if receipt.evidence_profile is not EvidenceProfile.OPAQUE_RPA_ADAPTER:
        raise ValueError("RPA_RECEIPT_PROFILE_MISMATCH")
    if receipt.experiment_id != experiment.experiment_id:
        raise ValueError("RPA_RECEIPT_EXPERIMENT_MISMATCH")
    for artifact in receipt.artifacts:
        path = resolve_recorded_artifact(
            root,
            artifact.path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
        if path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise ValueError(f"RPA_RECEIPT_ARTIFACT_MISMATCH path={artifact.path}")
    timing = _load_result(root / "timing" / "result.json", RunMode.TIMING)
    trace = _load_result(root / "trace" / "result.json", RunMode.TRACE)
    counters = _load_result(root / "counters" / "result.json", RunMode.COUNTERS)
    _require_shared_canonical_identity((timing, trace, counters), experiment)
    source_identities = {
        _source_identity(
            root / phase / "source_state.json",
            root / phase / "source_diff.patch",
        )
        for phase in ("timing", "trace", "counters", "finalizer")
    }
    if len(source_identities) != 1:
        raise ValueError("RPA_RUNS_DO_NOT_SHARE_SOURCE_IDENTITY")
    errors = tuple(
        _validate_phase(root, receipt, experiment, phase, result)
        for phase, result in (
            ("timing", timing),
            ("trace", trace),
            ("counters", counters),
        )
    )
    if receipt.schedule_sha256 != timing.schedule_sha256:
        raise ValueError("RPA_RECEIPT_SCHEDULE_MISMATCH")
    if receipt.runtime != timing.runtime or any(
        result.runtime != timing.runtime for result in (trace, counters)
    ):
        raise ValueError("RPA_RECEIPT_RUNTIME_MISMATCH")
    phase_results = (timing, trace, counters)
    all_phases_passed = all(result.passed for result in phase_results)
    if (
        receipt.correctness.passed is not all_phases_passed
        or receipt.correctness.oracle != experiment.workload.numerical.reference
        or receipt.required_semantic_properties != experiment.workload.numerical.semantic_properties
        or receipt.correctness.semantic_properties
    ):
        raise ValueError("RPA_RECEIPT_CORRECTNESS_CONTRACT_MISMATCH")
    if receipt.correctness.maximum_absolute_error != max(error[0] for error in errors):
        raise ValueError("RPA_RECEIPT_ABSOLUTE_ERROR_MISMATCH")
    if receipt.correctness.maximum_relative_error != max(error[1] for error in errors):
        raise ValueError("RPA_RECEIPT_RELATIVE_ERROR_MISMATCH")
    trace_assessment = assess_capture(root / "trace", experiment.profile)
    counter_assessment = assess_capture(root / "counters", counter_expectation(experiment))
    expected_status = (
        RunStatus.PASSED
        if all_phases_passed and trace_assessment.accepted and counter_assessment.accepted
        else RunStatus.REJECTED
    )
    if receipt.status is not expected_status:
        raise ValueError("RPA_RECEIPT_STATUS_MISMATCH")
    assessment_artifact = next(
        artifact
        for artifact in receipt.artifacts
        if artifact.role is ArtifactRole.PROFILE_ASSESSMENT
    )
    expected_assessment = _relative_json(
        {
            "timing_trace": trace_assessment.model_dump(mode="json"),
            "counter_trace": counter_assessment.model_dump(mode="json"),
        },
        root,
    )
    if json.loads((root / assessment_artifact.path).read_text()) != expected_assessment:
        raise ValueError("RPA_PROFILE_ASSESSMENT_REPLAY_MISMATCH")
    expected_metrics = (
        *_timing_metrics(root, timing),
        *_prefix_capture_metrics("trace", capture_metrics(trace_assessment.capture), root),
        *_prefix_capture_metrics("counter", capture_metrics(counter_assessment.capture), root),
    )
    if receipt.metrics != expected_metrics:
        raise ValueError("RPA_RECEIPT_METRICS_REPLAY_MISMATCH")
    artifacts_by_path = {artifact.path: artifact for artifact in receipt.artifacts}
    for metric in receipt.metrics:
        for source in metric.sources:
            artifact = artifacts_by_path.get(source.artifact_path)
            if artifact is None or artifact.sha256 != source.artifact_sha256:
                raise ValueError(
                    f"RPA_METRIC_SOURCE_NOT_BOUND metric={metric.name} path={source.artifact_path}"
                )
