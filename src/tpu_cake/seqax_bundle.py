from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from decimal import Decimal
from pathlib import Path

import numpy as np
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xprof import profile_data

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
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.distributed_tensor import DistributedTensor
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, array_sha256, semantic_sha256
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
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
from tpu_cake.receipt import _relative_json, _source_identity
from tpu_cake.runner import RunMode, _source_state, validate_profiler_contract
from tpu_cake.seqax_cost_model import SeqaxCostModelReport, estimate_seqax_forward
from tpu_cake.seqax_runner import (
    SEQAX_EVIDENCE_MEASURED_ITERATIONS,
    SEQAX_EVIDENCE_PARAMETERS,
    SEQAX_EVIDENCE_SEED,
    SEQAX_EVIDENCE_WARMUP_ITERATIONS,
    SEQAX_OUTPUT_ATOL,
    SEQAX_OUTPUT_RTOL,
    SeqaxForwardInvocation,
    SeqaxForwardRunResult,
    expected_seqax_profiler_contract,
)
from tpu_cake.workloads.seqax_forward import (
    seqax_forward_experiment,
    seqax_forward_schedule,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_inputs,
    seqax_forward_reference,
)
from tpu_cake.xprof_evidence import assess_capture, capture_metrics, count_profile_events
from tpu_cake.xprof_export import export_xprof_capture

_PHASES = ("timing", "trace", "counters")
_PROFILE_MARKERS = ("all-gather", "reduce_scatter", "dot_general")


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


def _trusted_experiment() -> KernelExperiment:
    module = seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    plan = lower_distributed_program_to_jax_mesh(module)
    return seqax_forward_experiment(
        plan,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        absolute_tolerance=SEQAX_OUTPUT_ATOL,
        relative_tolerance=SEQAX_OUTPUT_RTOL,
    )


def _counter_experiment(experiment: KernelExperiment) -> KernelExperiment:
    return experiment.model_copy(
        update={
            "profile": experiment.profile.model_copy(
                update={
                    "require_hbm_read_counters": True,
                    "require_hbm_write_counters": True,
                    "require_cycle_counters": True,
                    "minimum_counter_device_planes": experiment.target.chip_count,
                }
            )
        }
    )


def _load_result(root: Path, phase: str) -> SeqaxForwardRunResult:
    result = SeqaxForwardRunResult.model_validate_json(
        (root / phase / "result.json").read_text()
    )
    if result.mode.value != phase:
        raise ValueError(
            f"SEQAX_RUN_MODE_MISMATCH phase={phase} observed={result.mode.value}"
        )
    return result


def _parse_plan(path: Path):
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(DistributedTensor)
    module = Parser(context, path.read_text(), name=str(path)).parse_module()
    module.verify()
    return module, lower_distributed_program_to_jax_mesh(module)


def _phase_identity(result: SeqaxForwardRunResult) -> tuple[object, ...]:
    return (
        result.schedule_sha256,
        result.jax_source_sha256,
        result.plan_manifest_sha256,
        result.stablehlo_sha256,
        result.compiler_hlo_sha256,
        result.input_sha256,
        result.output_sha256,
        result.oracle_sha256,
        result.runtime,
        result.backend,
        result.device_kind,
        result.device_count,
        result.mesh,
        result.execution_scope,
    )


def _result_artifacts(
    root: Path,
    receipt: RunReceipt,
    phase: str,
    result: SeqaxForwardRunResult,
) -> dict[str, Path]:
    receipt_by_path = {artifact.path: artifact for artifact in receipt.artifacts}
    resolved: dict[str, Path] = {}
    paths = tuple(artifact.path for artifact in result.artifacts)
    if len(paths) != len(set(paths)):
        raise ValueError(f"SEQAX_RESULT_ARTIFACT_PATHS_NOT_UNIQUE phase={phase}")
    for artifact in result.artifacts:
        expected_role = _expected_result_role(artifact.path)
        if artifact.role is not expected_role:
            raise ValueError(
                f"SEQAX_RESULT_ARTIFACT_ROLE_MISMATCH phase={phase} "
                f"path={artifact.path} expected={expected_role.value} "
                f"observed={artifact.role.value}"
            )
        path = resolve_recorded_artifact(
            root / phase,
            artifact.path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
        relative = path.relative_to(root).as_posix()
        expected = artifact.model_copy(update={"path": relative})
        if receipt_by_path.get(relative) != expected:
            raise ValueError(
                f"SEQAX_RESULT_ARTIFACT_RECEIPT_MISMATCH phase={phase} path={relative}"
            )
        resolved[artifact.path] = path
    return resolved


def _expected_result_role(path: str) -> ArtifactRole:
    fixed = {
        "experiment.json": ArtifactRole.EXPERIMENT,
        "invocation.json": ArtifactRole.INVOCATION,
        "profiler_config.json": ArtifactRole.PROFILER_CONFIG,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "lowered_jax.py": ArtifactRole.JAX_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
        "cost_model.json": ArtifactRole.COST_MODEL,
        "stablehlo.txt": ArtifactRole.STABLEHLO,
        "compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "outputs/00.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "oracle/00.npy": ArtifactRole.ORACLE_OUTPUT,
    }
    if path in fixed:
        return fixed[path]
    if path.startswith("inputs/") and path.endswith(".npy"):
        index = path.removeprefix("inputs/").removesuffix(".npy")
        if index.isdigit() and 0 <= int(index) < len(_trusted_experiment().workload.inputs):
            return ArtifactRole.CORRECTNESS_INPUT
    raise ValueError(f"SEQAX_RESULT_ARTIFACT_PATH_UNRECOGNIZED path={path}")


def _load_arrays(paths: tuple[Path, ...]) -> tuple[np.ndarray, ...]:
    return tuple(np.load(path, allow_pickle=False) for path in paths)


def _validate_array_contract(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    phase: str,
) -> None:
    if value.shape != shape or str(value.dtype) != dtype:
        raise ValueError(
            f"SEQAX_TENSOR_CONTRACT_MISMATCH phase={phase} tensor={name} "
            f"expected_shape={shape} observed_shape={value.shape} "
            f"expected_dtype={dtype} observed_dtype={value.dtype}"
        )


def _errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    observed = actual.astype(np.float32)
    reference = expected.astype(np.float32)
    absolute = np.abs(observed - reference)
    denominator = np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
    return float(absolute.max()), float((absolute / denominator).max())


def _validate_phase(
    root: Path,
    receipt: RunReceipt,
    experiment: KernelExperiment,
    phase: str,
    result: SeqaxForwardRunResult,
) -> tuple[float, float]:
    if (
        result.warmup_iterations != SEQAX_EVIDENCE_WARMUP_ITERATIONS
        or result.measured_iterations != SEQAX_EVIDENCE_MEASURED_ITERATIONS
    ):
        raise ValueError(f"SEQAX_RESULT_PROTOCOL_MISMATCH phase={phase}")
    artifacts = _result_artifacts(root, receipt, phase, result)
    required = {
        "experiment.json",
        "invocation.json",
        "profiler_config.json",
        "source_state.json",
        "source_diff.patch",
        "distributed.xdsl",
        "lowered_jax.py",
        "plan_manifest.json",
        "cost_model.json",
        "stablehlo.txt",
        "compiler_hlo.txt",
        "ledger.sqlite",
    }
    missing = required - artifacts.keys()
    if missing:
        raise ValueError(
            f"SEQAX_RESULT_ARTIFACT_SET_INCOMPLETE phase={phase} missing={sorted(missing)}"
        )
    module, plan = _parse_plan(artifacts["distributed.xdsl"])
    if (
        plan.schedule_sha256 != result.schedule_sha256
        or _sha256(artifacts["distributed.xdsl"]) != result.schedule_sha256
        or plan.source_sha256() != result.jax_source_sha256
        or artifacts["lowered_jax.py"].read_text() != plan.render_executable_source()
        or _sha256(artifacts["lowered_jax.py"]) != result.jax_source_sha256
        or json.loads(artifacts["plan_manifest.json"].read_text()) != plan.manifest()
        or _sha256(artifacts["plan_manifest.json"]) != result.plan_manifest_sha256
    ):
        raise ValueError(f"SEQAX_SAVED_LOWERING_IDENTITY_MISMATCH phase={phase}")
    saved_experiment = KernelExperiment.model_validate_json(
        artifacts["experiment.json"].read_text()
    )
    if saved_experiment != experiment:
        raise ValueError(f"SEQAX_EXPERIMENT_MISMATCH phase={phase}")
    expected_invocation = SeqaxForwardInvocation(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        execution_schema=plan.schema,
        mode=result.mode,
        seed=SEQAX_EVIDENCE_SEED,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        parameters=SEQAX_EVIDENCE_PARAMETERS,
        schedule_sha256=plan.schedule_sha256,
        jax_source_sha256=plan.source_sha256(),
        execution_scope=plan.execution_scope,
    )
    invocation = SeqaxForwardInvocation.model_validate_json(
        artifacts["invocation.json"].read_text()
    )
    if invocation != expected_invocation:
        raise ValueError(f"SEQAX_INVOCATION_MISMATCH phase={phase}")
    profiler_contract = json.loads(artifacts["profiler_config.json"].read_text())
    validate_profiler_contract(result.mode, profiler_contract)
    if profiler_contract != expected_seqax_profiler_contract(result.mode):
        raise ValueError(f"SEQAX_PROFILER_CONTRACT_MISMATCH phase={phase}")
    _source_identity(
        artifacts["source_state.json"],
        artifacts["source_diff.patch"],
        require_clean=True,
    )
    if (
        result.backend != "tpu"
        or re.fullmatch(
            r"tpu(?: v)?7x(?: lite)?", result.device_kind.strip().lower()
        )
        is None
        or result.device_count != plan.device_count
        or result.mesh != plan.mesh_axes
        or result.execution_scope != plan.execution_scope
        or result.stablehlo_sha256 != _sha256(artifacts["stablehlo.txt"])
        or result.compiler_hlo_sha256 != _sha256(artifacts["compiler_hlo.txt"])
    ):
        raise ValueError(f"SEQAX_EXECUTION_IDENTITY_MISMATCH phase={phase}")
    compiler_hlo = artifacts["compiler_hlo.txt"].read_text()
    if not all(marker in compiler_hlo for marker in _PROFILE_MARKERS):
        raise ValueError(f"SEQAX_COMPILER_HLO_DISTRIBUTION_MISSING phase={phase}")
    expected_cost = estimate_seqax_forward(
        module,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=result.schedule_sha256,
            artifact_path="distributed.xdsl",
            tool="tpu-cake",
            field="seqax-distributed-forward-v1",
        ),
        expected_schedule_sha256=result.schedule_sha256,
    )
    saved_cost = SeqaxCostModelReport.model_validate_json(
        artifacts["cost_model.json"].read_text()
    )
    if saved_cost != expected_cost:
        raise ValueError(f"SEQAX_COST_MODEL_REPLAY_MISMATCH phase={phase}")

    input_paths = tuple(
        artifacts[f"inputs/{index:02d}.npy"]
        for index in range(len(experiment.workload.inputs))
    )
    output_paths = (artifacts["outputs/00.npy"],)
    oracle_paths = (artifacts["oracle/00.npy"],)
    inputs = _load_arrays(input_paths)
    outputs = _load_arrays(output_paths)
    saved_oracles = _load_arrays(oracle_paths)
    for value, contract in zip(inputs, experiment.workload.inputs, strict=True):
        _validate_array_contract(
            value,
            name=contract.name,
            shape=contract.shape,
            dtype=contract.dtype,
            phase=phase,
        )
    for values, label in ((outputs, "output"), (saved_oracles, "oracle")):
        for value, contract in zip(
            values,
            experiment.workload.outputs,
            strict=True,
        ):
            _validate_array_contract(
                value,
                name=f"{label}:{contract.name}",
                shape=contract.shape,
                dtype=contract.dtype,
                phase=phase,
            )
    expected_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=SEQAX_EVIDENCE_SEED,
            **SEQAX_EVIDENCE_PARAMETERS,
        )
    )
    if any(
        not np.array_equal(saved, expected)
        for saved, expected in zip(inputs, expected_inputs, strict=True)
    ):
        raise ValueError(f"SEQAX_DETERMINISTIC_INPUT_REPLAY_MISMATCH phase={phase}")
    expected_oracle = np.asarray(
        seqax_forward_reference(inputs, **SEQAX_EVIDENCE_PARAMETERS)
    )
    if not np.array_equal(saved_oracles[0], expected_oracle):
        raise ValueError(f"SEQAX_ORACLE_REPLAY_MISMATCH phase={phase}")
    if (
        tuple(array_sha256(value) for value in inputs) != result.input_sha256
        or tuple(array_sha256(value) for value in outputs) != result.output_sha256
        or tuple(array_sha256(value) for value in saved_oracles) != result.oracle_sha256
    ):
        raise ValueError(f"SEQAX_ARRAY_IDENTITY_MISMATCH phase={phase}")
    maximum_absolute_error, maximum_relative_error = _errors(
        outputs[0], expected_oracle
    )
    passed = np.allclose(
        outputs[0],
        expected_oracle,
        atol=SEQAX_OUTPUT_ATOL,
        rtol=SEQAX_OUTPUT_RTOL,
    )
    if (
        passed is not result.passed
        or not math.isclose(
            maximum_absolute_error,
            result.maximum_absolute_error,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            maximum_relative_error,
            result.maximum_relative_error,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"SEQAX_CORRECTNESS_REPLAY_MISMATCH phase={phase}")
    if not passed:
        raise ValueError(f"SEQAX_PHASE_CORRECTNESS_FAILED phase={phase}")
    expected_run_id = semantic_sha256(
        "seqax-distributed-forward-run-v1",
        result.mode.value,
        plan.schedule_sha256,
        plan.source_sha256(),
        str(SEQAX_EVIDENCE_SEED),
    )
    if result.run_id != expected_run_id:
        raise ValueError(f"SEQAX_RUN_ID_MISMATCH phase={phase}")
    terminal = {
        RunMode.TIMING: RunState.TIMED,
        RunMode.TRACE: RunState.TRACED,
        RunMode.COUNTERS: RunState.COUNTERED,
    }[result.mode]
    terminal_payload: dict[str, object] = {
        "warmup_iterations": SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        "measured_iterations": SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        "mesh": dict(plan.mesh_axes),
    }
    if result.mode is RunMode.TIMING:
        if len(result.samples_ns) != SEQAX_EVIDENCE_MEASURED_ITERATIONS or any(
            sample <= 0 for sample in result.samples_ns
        ):
            raise ValueError("SEQAX_TIMING_SAMPLE_PROTOCOL_MISMATCH")
        median = statistics.median(result.samples_ns)
        ordered = sorted(result.samples_ns)
        p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
        coefficient = statistics.pstdev(result.samples_ns) / statistics.mean(
            result.samples_ns
        )
        if (
            result.median_ns != median
            or result.p90_ns != p90
            or result.coefficient_of_variation is None
            or not math.isclose(
                result.coefficient_of_variation,
                coefficient,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("SEQAX_TIMING_STATISTIC_MISMATCH")
        terminal_payload.update(
            median_ns=median,
            p90_ns=p90,
            coefficient_of_variation=coefficient,
        )
    else:
        if result.samples_ns or any(
            value is not None
            for value in (
                result.median_ns,
                result.p90_ns,
                result.coefficient_of_variation,
            )
        ):
            raise ValueError(f"SEQAX_PROFILE_CONTAINS_TIMING_CLAIMS phase={phase}")
        observed_steps = count_profile_events(root / phase / "profile", "seqax_forward")
        if observed_steps != SEQAX_EVIDENCE_MEASURED_ITERATIONS:
            raise ValueError(
                f"SEQAX_PROFILE_STEP_COUNT_MISMATCH phase={phase} "
                f"expected={SEQAX_EVIDENCE_MEASURED_ITERATIONS} observed={observed_steps}"
            )
        terminal_payload["profile_root"] = "profile"
    expected_payloads = (
        invocation.model_dump(mode="json"),
        {"schedule_sha256": plan.schedule_sha256},
        {
            "schedule_sha256": plan.schedule_sha256,
            "jax_source_sha256": result.jax_source_sha256,
            "plan_manifest_sha256": result.plan_manifest_sha256,
            "execution_scope": plan.execution_scope,
        },
        {
            "stablehlo_sha256": result.stablehlo_sha256,
            "compiler_hlo_sha256": result.compiler_hlo_sha256,
            "compile_duration_ns": result.compile_duration_ns,
        },
        {
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
        },
        terminal_payload,
    )
    history = read_ledger_history(artifacts["ledger.sqlite"], result.run_id)
    expected_states = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        terminal,
    )
    if tuple(event.state for event in history) != expected_states or tuple(
        event.payload_sha256 for event in history
    ) != tuple(ExperimentLedger.payload_sha256(payload) for payload in expected_payloads):
        raise ValueError(f"SEQAX_LEDGER_REPLAY_MISMATCH phase={phase}")
    return maximum_absolute_error, maximum_relative_error


def _ensure_exports(root: Path, phase: str) -> None:
    if not list((root / phase).rglob("hlo_stats.json")):
        export_xprof_capture(root / phase / "profile", root / phase / "xprof")


def _prefix_metrics(
    prefix: str,
    metrics: tuple[Metric, ...],
    root: Path,
) -> tuple[Metric, ...]:
    result = []
    for metric in metrics:
        sources = tuple(
            source.model_copy(
                update={
                    "artifact_path": Path(source.artifact_path)
                    .resolve()
                    .relative_to(root)
                    .as_posix()
                }
            )
            for source in metric.sources
        )
        result.append(
            metric.model_copy(
                update={"name": f"{prefix}_{metric.name}", "sources": sources}
            )
        )
    return tuple(result)


def _timing_metrics(root: Path, result: SeqaxForwardRunResult) -> tuple[Metric, ...]:
    source = MetricSource(
        artifact_sha256=_sha256(root / "timing/result.json"),
        artifact_path="timing/result.json",
        tool="tpu-cake",
        field="samples_ns",
    )
    interval = MeasurementInterval(scope="one synchronized eight-device Seqax forward")
    assert result.p90_ns is not None
    assert result.coefficient_of_variation is not None
    mean = Decimal(str(statistics.mean(result.samples_ns)))
    deviation = Decimal(str(statistics.pstdev(result.samples_ns)))
    return (
        Metric(
            name="median_synchronized_forward_duration",
            quantity=Quantity(
                value=Decimal(str(statistics.median(result.samples_ns))),
                unit=Unit.NANOSECOND,
            ),
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
            name="p90_synchronized_forward_duration",
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
            name="synchronized_forward_coefficient_of_variation",
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


def _step_metric(root: Path, phase: str, result: SeqaxForwardRunResult) -> Metric:
    xplane = next((root / phase / "profile").rglob("*.xplane.pb"))
    count = count_profile_events(root / phase / "profile", "seqax_forward")
    return Metric(
        name=f"{phase}_seqax_forward_count",
        quantity=Quantity(value=Decimal(count), unit=Unit.COUNT),
        kind=MeasurementKind.MEASURED,
        interval=MeasurementInterval(
            scope=f"the complete {result.measured_iterations}-forward {phase} capture"
        ),
        sources=(
            MetricSource(
                artifact_sha256=_sha256(xplane),
                artifact_path=xplane.relative_to(root).as_posix(),
                tool="XPlane",
                field="StepTraceAnnotation(seqax_forward)",
            ),
        ),
    )


def _bound_program_id(assessment) -> str:
    matches = tuple(
        program.program_id
        for program in assessment.capture.programs
        if program.program_id in assessment.capture.timed_program_ids
        and program.timed_self_us > 0
        and all(program.marker_counts.get(marker, 0) > 0 for marker in _PROFILE_MARKERS)
    )
    if len(matches) != 1:
        raise ValueError(f"SEQAX_BOUND_TIMED_PROGRAM_COUNT_MISMATCH observed={matches}")
    return matches[0]


def _validate_capture_topology(assessment, *, counters: bool) -> None:
    observed_planes = {
        plane.name
        for plane in assessment.capture.planes
        if plane.name.startswith("/device:TPU:") and "SparseCore" not in plane.name
    }
    expected_planes = {f"/device:TPU:{index}" for index in range(8)}
    if observed_planes != expected_planes:
        raise ValueError(
            f"SEQAX_TPU_PLANE_SET_MISMATCH expected={sorted(expected_planes)} "
            f"observed={sorted(observed_planes)}"
        )
    observed_counter_cores = set(
        assessment.capture.counters.periodic_samples_per_tpu_core
    )
    expected_counter_cores = {"0", "2", "4", "6"} if counters else set()
    if observed_counter_cores != expected_counter_cores:
        raise ValueError(
            f"SEQAX_COUNTER_CORE_SET_MISMATCH expected={sorted(expected_counter_cores)} "
            f"observed={sorted(observed_counter_cores)}"
        )


def _module_durations(
    root: Path,
    phase: str,
    assessment,
    expected_count: int,
) -> tuple[float, ...]:
    program_id = _bound_program_id(assessment)
    xplane = next((root / phase / "profile").rglob("*.xplane.pb"))
    expected_name = f"jit_execute({program_id})"
    profile = profile_data.ProfileData.from_file(xplane)
    try:
        durations = tuple(
            float(event.duration_ns)
            for plane in profile.planes
            if plane.name == "/device:TPU:0"
            for line in plane.lines
            if line.name == "XLA Modules"
            for event in line.events
            if event.name == expected_name
        )
    finally:
        profile.close()
    if len(durations) != expected_count or any(value <= 0 for value in durations):
        raise ValueError(
            f"SEQAX_MODULE_EXECUTION_COUNT_MISMATCH phase={phase} "
            f"program={program_id} expected={expected_count} observed={len(durations)}"
        )
    return durations


def _module_metrics(
    root: Path,
    phase: str,
    result: SeqaxForwardRunResult,
    assessment,
) -> tuple[Metric, ...]:
    durations = _module_durations(
        root,
        phase,
        assessment,
        SEQAX_EVIDENCE_MEASURED_ITERATIONS,
    )
    ordered = sorted(durations)
    p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
    xplane = next((root / phase / "profile").rglob("*.xplane.pb"))
    source = MetricSource(
        artifact_sha256=_sha256(xplane),
        artifact_path=xplane.relative_to(root).as_posix(),
        tool="XPlane",
        field=f"/device:TPU:0/XLA Modules/jit_execute({_bound_program_id(assessment)})",
    )
    interval = MeasurementInterval(scope="one compiled eight-device Seqax forward module")
    return (
        Metric(
            name=f"{phase}_median_compiled_forward_duration",
            quantity=Quantity(
                value=Decimal(str(statistics.median(durations))),
                unit=Unit.NANOSECOND,
            ),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="sample_median",
                version="1",
                expression="median(jit_execute_duration_ns)",
            ),
        ),
        Metric(
            name=f"{phase}_p90_compiled_forward_duration",
            quantity=Quantity(value=Decimal(str(p90)), unit=Unit.NANOSECOND),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="nearest_rank_p90",
                version="1",
                expression="sorted(duration_ns)[round((n-1)*0.9)]",
            ),
        ),
    )


def _cost_metrics(root: Path) -> tuple[Metric, ...]:
    report = SeqaxCostModelReport.model_validate_json(
        (root / "timing/cost_model.json").read_text()
    )
    return tuple(
        metric.model_copy(
            update={
                "name": f"cost_{metric.name}",
                "sources": tuple(
                    source.model_copy(
                        update={"artifact_path": f"timing/{source.artifact_path}"}
                    )
                    for source in metric.sources
                ),
            }
        )
        for metric in report.metrics
    )


def _metrics(
    root: Path,
    timing: SeqaxForwardRunResult,
    trace_assessment,
    counter_assessment,
    trace: SeqaxForwardRunResult,
    counters: SeqaxForwardRunResult,
) -> tuple[Metric, ...]:
    return (
        *_timing_metrics(root, timing),
        _step_metric(root, "trace", trace),
        _step_metric(root, "counters", counters),
        *_module_metrics(root, "trace", trace, trace_assessment),
        *_module_metrics(root, "counters", counters, counter_assessment),
        *_cost_metrics(root),
        *_prefix_metrics("trace", capture_metrics(trace_assessment.capture), root),
        *_prefix_metrics("counter", capture_metrics(counter_assessment.capture), root),
    )


def _artifact_roles(
    root: Path,
    results: dict[str, SeqaxForwardRunResult],
    assessment_path: Path,
) -> dict[Path, ArtifactRole]:
    roles: dict[Path, ArtifactRole] = {}
    for phase, result in results.items():
        for artifact in result.artifacts:
            path = (root / phase / artifact.path).resolve()
            previous = roles.setdefault(path, artifact.role)
            if previous is not artifact.role:
                raise ValueError(f"SEQAX_ARTIFACT_ROLE_CONFLICT path={path}")
        roles[(root / phase / "result.json").resolve()] = {
            "timing": ArtifactRole.TIMING_SAMPLES,
            "trace": ArtifactRole.TRACE_RESULT,
            "counters": ArtifactRole.COUNTER_RESULT,
        }[phase]
    for phase, role in (
        ("trace", ArtifactRole.TIMING_TRACE),
        ("counters", ArtifactRole.COUNTER_TRACE),
    ):
        for path in sorted((root / phase).rglob("*")):
            if not path.is_file() or path.resolve() in roles:
                continue
            if path.name == "hlo_stats.json":
                selected_role = ArtifactRole.HLO_STATS
            elif path.suffixes[-2:] == [".xplane", ".pb"]:
                selected_role = role
            else:
                selected_role = ArtifactRole.XPROF_EXPORT
            roles[path.resolve()] = selected_role
    roles[assessment_path.resolve()] = ArtifactRole.PROFILE_ASSESSMENT
    roles[(root / "finalizer/source_state.json").resolve()] = ArtifactRole.SOURCE_STATE
    roles[(root / "finalizer/source_diff.patch").resolve()] = ArtifactRole.SOURCE_DIFF
    return roles


def _phases(artifacts: tuple[ArtifactReference, ...]) -> tuple[EvidencePhase, ...]:
    grouped: dict[EvidencePhaseName, list[str]] = {
        phase: [] for phase in EvidencePhaseName
    }
    for artifact in artifacts:
        first = Path(artifact.path).parts[0]
        phase = (
            EvidencePhaseName(first)
            if first in {*_PHASES, "finalizer"}
            else EvidencePhaseName.AGGREGATE
        )
        grouped[phase].append(artifact.path)
    return tuple(
        EvidencePhase(name=phase, artifact_paths=tuple(paths))
        for phase, paths in grouped.items()
    )


def build_seqax_forward_receipt(root: Path, *, write_receipt: bool = True) -> RunReceipt:
    root = root.resolve()
    finalizer = root / "finalizer"
    _source_state(Path(__file__).resolve().parents[2], finalizer)
    results = {phase: _load_result(root, phase) for phase in _PHASES}
    if len({_phase_identity(result) for result in results.values()}) != 1:
        raise ValueError("SEQAX_RUNS_DO_NOT_SHARE_EXECUTION_IDENTITY")
    source_identities = {
        _source_identity(
            root / phase / "source_state.json",
            root / phase / "source_diff.patch",
            require_clean=True,
        )
        for phase in (*_PHASES, "finalizer")
    }
    if len(source_identities) != 1:
        raise ValueError("SEQAX_RUNS_DO_NOT_SHARE_SOURCE_IDENTITY")
    for phase in ("trace", "counters"):
        _ensure_exports(root, phase)
    experiment = _trusted_experiment()
    trace_assessment = assess_capture(root / "trace", experiment.profile)
    counter_assessment = assess_capture(
        root / "counters", _counter_experiment(experiment).profile
    )
    _validate_capture_topology(trace_assessment, counters=False)
    _validate_capture_topology(counter_assessment, counters=True)
    assessment_path = root / "profile_assessment.json"
    assessment_path.write_text(
        json.dumps(
            _relative_json(
                {
                    "timing_trace": trace_assessment.model_dump(mode="json"),
                    "counter_trace": counter_assessment.model_dump(mode="json"),
                },
                root,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    roles = _artifact_roles(root, results, assessment_path)
    artifacts = tuple(
        _reference(root, path, role)
        for path, role in sorted(roles.items(), key=lambda item: str(item[0]))
    )
    timing = results["timing"]
    metrics = _metrics(
        root,
        timing,
        trace_assessment,
        counter_assessment,
        results["trace"],
        results["counters"],
    )
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        evidence_profile=EvidenceProfile.SEQAX_DISTRIBUTED_FORWARD,
        schedule_sha256=timing.schedule_sha256,
        status=(
            RunStatus.PASSED
            if all(result.passed for result in results.values())
            and trace_assessment.accepted
            and counter_assessment.accepted
            else RunStatus.REJECTED
        ),
        runtime=timing.runtime,
        correctness=CorrectnessResult(
            passed=all(result.passed for result in results.values()),
            oracle=experiment.workload.numerical.reference,
            maximum_absolute_error=max(
                result.maximum_absolute_error for result in results.values()
            ),
            maximum_relative_error=max(
                result.maximum_relative_error for result in results.values()
            ),
        ),
        required_semantic_properties=(),
        metrics=metrics,
        artifacts=artifacts,
        phases=_phases(artifacts),
    )
    validate_seqax_forward_receipt(receipt, root=root)
    if write_receipt:
        temporary = root / "receipt.json.tmp"
        temporary.write_text(receipt.model_dump_json(indent=2) + "\n")
        temporary.replace(root / "receipt.json")
    return receipt


def validate_seqax_forward_receipt(receipt: RunReceipt, *, root: Path) -> None:
    root = root.resolve()
    experiment = _trusted_experiment()
    if (
        receipt.evidence_profile is not EvidenceProfile.SEQAX_DISTRIBUTED_FORWARD
        or receipt.experiment_id != experiment.experiment_id
        or receipt.schedule_sha256 != experiment.schedule_sha256
    ):
        raise ValueError("SEQAX_RECEIPT_EXPERIMENT_IDENTITY_MISMATCH")
    for artifact in receipt.artifacts:
        resolve_recorded_artifact(
            root,
            artifact.path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
    results = {phase: _load_result(root, phase) for phase in _PHASES}
    if len({_phase_identity(result) for result in results.values()}) != 1:
        raise ValueError("SEQAX_RUNS_DO_NOT_SHARE_EXECUTION_IDENTITY")
    expected_artifacts = tuple(
        _reference(root, path, role)
        for path, role in sorted(
            _artifact_roles(root, results, root / "profile_assessment.json").items(),
            key=lambda item: str(item[0]),
        )
    )
    if receipt.artifacts != expected_artifacts:
        raise ValueError("SEQAX_RECEIPT_ARTIFACT_MANIFEST_MISMATCH")
    if receipt.phases != _phases(expected_artifacts):
        raise ValueError("SEQAX_RECEIPT_PHASE_PARTITION_MISMATCH")
    source_identities = {
        _source_identity(
            root / phase / "source_state.json",
            root / phase / "source_diff.patch",
            require_clean=receipt.status is RunStatus.PASSED,
        )
        for phase in (*_PHASES, "finalizer")
    }
    if len(source_identities) != 1:
        raise ValueError("SEQAX_RUNS_DO_NOT_SHARE_SOURCE_IDENTITY")
    errors = tuple(
        _validate_phase(root, receipt, experiment, phase, results[phase])
        for phase in _PHASES
    )
    trace_assessment = assess_capture(root / "trace", experiment.profile)
    counter_assessment = assess_capture(
        root / "counters", _counter_experiment(experiment).profile
    )
    _validate_capture_topology(trace_assessment, counters=False)
    _validate_capture_topology(counter_assessment, counters=True)
    expected_status = (
        RunStatus.PASSED
        if all(result.passed for result in results.values())
        and trace_assessment.accepted
        and counter_assessment.accepted
        else RunStatus.REJECTED
    )
    if receipt.status is not expected_status:
        raise ValueError("SEQAX_RECEIPT_STATUS_MISMATCH")
    if (
        receipt.runtime != results["timing"].runtime
        or any(result.runtime != receipt.runtime for result in results.values())
        or receipt.correctness.passed
        is not all(result.passed for result in results.values())
        or receipt.correctness.oracle != experiment.workload.numerical.reference
        or receipt.correctness.maximum_absolute_error != max(error[0] for error in errors)
        or receipt.correctness.maximum_relative_error != max(error[1] for error in errors)
        or receipt.required_semantic_properties
        or receipt.correctness.semantic_properties
    ):
        raise ValueError("SEQAX_RECEIPT_CORRECTNESS_OR_RUNTIME_MISMATCH")
    assessment = next(
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
    if json.loads((root / assessment.path).read_text()) != expected_assessment:
        raise ValueError("SEQAX_PROFILE_ASSESSMENT_REPLAY_MISMATCH")
    expected_metrics = _metrics(
        root,
        results["timing"],
        trace_assessment,
        counter_assessment,
        results["trace"],
        results["counters"],
    )
    if receipt.metrics != expected_metrics:
        raise ValueError("SEQAX_RECEIPT_METRICS_REPLAY_MISMATCH")
    artifacts_by_path = {artifact.path: artifact for artifact in receipt.artifacts}
    for metric in receipt.metrics:
        for source in metric.sources:
            artifact = artifacts_by_path.get(source.artifact_path)
            if artifact is None or artifact.sha256 != source.artifact_sha256:
                raise ValueError(
                    f"SEQAX_METRIC_SOURCE_NOT_BOUND metric={metric.name} "
                    f"path={source.artifact_path}"
                )
