from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from xprof import profile_data

from tpu_cake.artifacts import resolve_bundle_artifact
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
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, array_sha256, semantic_sha256
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
from tpu_cake.seqax_pallas_lowering import (
    lower_seqax_physical_to_pallas,
)
from tpu_cake.seqax_pallas_runner import (
    SeqaxPallasInvocation,
    SeqaxPallasRunResult,
    _errors,
    _physical_collective_counts,
    _validate_compiled_program,
    seqax_physical_pallas_experiment,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_runner import (
    SEQAX_EVIDENCE_MEASURED_ITERATIONS,
    SEQAX_EVIDENCE_PARAMETERS,
    SEQAX_EVIDENCE_SEED,
    SEQAX_EVIDENCE_WARMUP_ITERATIONS,
    SEQAX_OUTPUT_ATOL,
    SEQAX_OUTPUT_RTOL,
    expected_seqax_profiler_contract,
)
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)
from tpu_cake.xprof_evidence import assess_capture, capture_metrics
from tpu_cake.xprof_export import DEFAULT_TOOLS, XProfExportManifest, export_xprof_capture

_PHASES = ("timing", "trace", "counters")
_PROFILE_MARKERS = ("pallas_call", "all-gather", "reduce_scatter")
_STEP_EVENT = "seqax_physical_pallas_forward"


class _XProfDerivedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifacts: tuple[ArtifactReference, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_plan():
    distributed = seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    physical = lower_seqax_forward_to_physical(distributed).module
    return distributed, physical, lower_seqax_physical_to_pallas(distributed, physical)


def _trusted_experiment() -> KernelExperiment:
    return seqax_physical_pallas_experiment(_trusted_plan()[2])


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


def _canonical_profile_assessment(value: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(value))
    for key in ("timing_trace", "counter_trace"):
        assessment = normalized.get(key)
        if not isinstance(assessment, dict):
            continue
        capture = assessment.get("capture")
        if not isinstance(capture, dict):
            continue
        program_ids = capture.get("timed_program_ids")
        if isinstance(program_ids, list):
            capture["timed_program_ids"] = sorted(program_ids)
    return normalized


def _reference(root: Path, path: Path, role: ArtifactRole) -> ArtifactReference:
    path = path.resolve()
    return ArtifactReference(
        path=path.relative_to(root.resolve()).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
    )


def _load_result(root: Path, phase: str) -> SeqaxPallasRunResult:
    result = SeqaxPallasRunResult.model_validate_json((root / phase / "result.json").read_text())
    if result.mode.value != phase:
        raise ValueError(
            f"SEQAX_PALLAS_RUN_MODE_MISMATCH phase={phase} observed={result.mode.value}"
        )
    return result


def _phase_identity(result: SeqaxPallasRunResult) -> tuple[object, ...]:
    return (
        result.distributed_schedule_sha256,
        result.physical_schedule_sha256,
        result.pallas_source_sha256,
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


def _expected_result_role(path: str) -> ArtifactRole:
    fixed = {
        "experiment.json": ArtifactRole.EXPERIMENT,
        "invocation.json": ArtifactRole.INVOCATION,
        "profiler_config.json": ArtifactRole.PROFILER_CONFIG,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "physical.xdsl": ArtifactRole.PHYSICAL_IR,
        "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
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
        if (
            index.isdigit()
            and index == f"{int(index):02d}"
            and 0 <= int(index) < len(_trusted_experiment().workload.inputs)
        ):
            return ArtifactRole.CORRECTNESS_INPUT
    if path.endswith(".xplane.pb") and path.startswith("profile/"):
        raise ValueError("profile XPlane role depends on the phase")
    if path.startswith("profile/") and path.endswith(".trace.json.gz"):
        return ArtifactRole.PROFILE_AUXILIARY
    raise ValueError(f"SEQAX_PALLAS_RESULT_ARTIFACT_PATH_UNRECOGNIZED path={path}")


def _result_artifacts(
    root: Path,
    phase: str,
    result: SeqaxPallasRunResult,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    paths = tuple(artifact.path for artifact in result.artifacts)
    if len(paths) != len(set(paths)):
        raise ValueError(f"SEQAX_PALLAS_RESULT_ARTIFACT_PATHS_NOT_UNIQUE phase={phase}")
    expected_paths = {
        "experiment.json",
        "invocation.json",
        "profiler_config.json",
        "source_state.json",
        "source_diff.patch",
        "distributed.xdsl",
        "physical.xdsl",
        "lowered_pallas.py",
        "plan_manifest.json",
        "stablehlo.txt",
        "compiler_hlo.txt",
        "ledger.sqlite",
        "outputs/00.npy",
        "oracle/00.npy",
        *(f"inputs/{index:02d}.npy" for index in range(len(_trusted_experiment().workload.inputs))),
    }
    profile_paths = tuple(
        path for path in paths if path.startswith("profile/") and path.endswith(".xplane.pb")
    )
    auxiliary_profile_paths = tuple(
        path for path in paths if path.startswith("profile/") and path not in profile_paths
    )
    for path in auxiliary_profile_paths:
        if _expected_result_role(path) is not ArtifactRole.PROFILE_AUXILIARY:
            raise ValueError(f"SEQAX_PALLAS_PROFILE_AUXILIARY_PATH_UNRECOGNIZED path={path}")
    if phase == "timing":
        if profile_paths:
            raise ValueError("SEQAX_PALLAS_TIMING_RESULT_HAS_PROFILE_XPLANE")
    elif len(profile_paths) != 1:
        raise ValueError(
            f"SEQAX_PALLAS_PROFILE_XPLANE_COUNT_MISMATCH phase={phase} observed={profile_paths}"
        )
    expected_paths.update(profile_paths)
    expected_paths.update(auxiliary_profile_paths)
    if set(paths) != expected_paths:
        raise ValueError(
            f"SEQAX_PALLAS_RESULT_ARTIFACT_SET_MISMATCH phase={phase} "
            f"missing={sorted(expected_paths - set(paths))} "
            f"extra={sorted(set(paths) - expected_paths)}"
        )
    for artifact in result.artifacts:
        if artifact.path.endswith(".xplane.pb") and artifact.path.startswith("profile/"):
            expected_role = (
                ArtifactRole.TIMING_TRACE if phase == "trace" else ArtifactRole.COUNTER_TRACE
            )
        else:
            expected_role = _expected_result_role(artifact.path)
        if artifact.role is not expected_role:
            raise ValueError(
                f"SEQAX_PALLAS_RESULT_ARTIFACT_ROLE_MISMATCH phase={phase} "
                f"path={artifact.path} expected={expected_role.value} "
                f"observed={artifact.role.value}"
            )
        resolved[artifact.path] = _resolve_result_artifact(root, phase, artifact)
    return resolved


def _resolve_result_artifact(
    root: Path,
    phase: str,
    artifact: ArtifactReference,
) -> Path:
    path = resolve_bundle_artifact(root / phase, artifact.path)
    if not path.is_file():
        raise ValueError(f"SEQAX_PALLAS_RESULT_ARTIFACT_MISSING phase={phase} path={artifact.path}")
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError(
            f"SEQAX_PALLAS_RESULT_ARTIFACT_SIZE_MISMATCH phase={phase} path={artifact.path}"
        )
    if _sha256(path) != artifact.sha256:
        raise ValueError(
            f"SEQAX_PALLAS_RESULT_ARTIFACT_HASH_MISMATCH phase={phase} path={artifact.path}"
        )
    return path


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
            f"SEQAX_PALLAS_ARRAY_ABI_MISMATCH phase={phase} name={name} "
            f"expected={shape}/{dtype} observed={value.shape}/{value.dtype}"
        )


def _validate_phase(
    root: Path,
    experiment: KernelExperiment,
    phase: str,
    result: SeqaxPallasRunResult,
) -> tuple[float, float]:
    if (
        result.warmup_iterations != SEQAX_EVIDENCE_WARMUP_ITERATIONS
        or result.measured_iterations != SEQAX_EVIDENCE_MEASURED_ITERATIONS
    ):
        raise ValueError(f"SEQAX_PALLAS_RESULT_PROTOCOL_MISMATCH phase={phase}")
    artifacts = _result_artifacts(root, phase, result)
    required = {
        "experiment.json",
        "invocation.json",
        "profiler_config.json",
        "source_state.json",
        "source_diff.patch",
        "distributed.xdsl",
        "physical.xdsl",
        "lowered_pallas.py",
        "plan_manifest.json",
        "stablehlo.txt",
        "compiler_hlo.txt",
        "ledger.sqlite",
    }
    missing = required - artifacts.keys()
    if missing:
        raise ValueError(
            f"SEQAX_PALLAS_RESULT_ARTIFACT_SET_INCOMPLETE phase={phase} missing={sorted(missing)}"
        )
    distributed, physical, plan = _trusted_plan()
    if (
        artifacts["distributed.xdsl"].read_text() != plan.canonical_distributed_xdsl
        or artifacts["physical.xdsl"].read_text() != plan.canonical_physical_xdsl
        or _sha256(artifacts["distributed.xdsl"]) != plan.distributed_schedule_sha256
        or _sha256(artifacts["physical.xdsl"]) != plan.physical_schedule_sha256
        or artifacts["lowered_pallas.py"].read_text() != plan.render_executable_source()
        or _sha256(artifacts["lowered_pallas.py"]) != plan.source_sha256()
        or json.loads(artifacts["plan_manifest.json"].read_text()) != plan.manifest()
        or result.distributed_schedule_sha256 != plan.distributed_schedule_sha256
        or result.physical_schedule_sha256 != plan.physical_schedule_sha256
        or result.pallas_source_sha256 != plan.source_sha256()
        or result.plan_manifest_sha256 != _sha256(artifacts["plan_manifest.json"])
    ):
        raise ValueError(f"SEQAX_PALLAS_SAVED_LOWERING_IDENTITY_MISMATCH phase={phase}")
    distributed.verify()
    physical.verify()
    saved_experiment = KernelExperiment.model_validate_json(
        artifacts["experiment.json"].read_text()
    )
    if saved_experiment != experiment:
        raise ValueError(f"SEQAX_PALLAS_EXPERIMENT_MISMATCH phase={phase}")
    expected_invocation = SeqaxPallasInvocation(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        execution_schema=plan.schema,
        mode=result.mode,
        seed=SEQAX_EVIDENCE_SEED,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        parameters=SEQAX_EVIDENCE_PARAMETERS,
        distributed_schedule_sha256=plan.distributed_schedule_sha256,
        physical_schedule_sha256=plan.physical_schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
        pallas_region_count=plan.pallas_region_count,
        execution_scope=plan.execution_scope,
    )
    invocation = SeqaxPallasInvocation.model_validate_json(artifacts["invocation.json"].read_text())
    if invocation != expected_invocation:
        raise ValueError(f"SEQAX_PALLAS_INVOCATION_MISMATCH phase={phase}")
    profiler_contract = json.loads(artifacts["profiler_config.json"].read_text())
    validate_profiler_contract(result.mode, profiler_contract)
    if profiler_contract != expected_seqax_profiler_contract(result.mode):
        raise ValueError(f"SEQAX_PALLAS_PROFILER_CONTRACT_MISMATCH phase={phase}")
    _source_identity(
        artifacts["source_state.json"],
        artifacts["source_diff.patch"],
        require_clean=True,
    )
    if (
        result.backend != "tpu"
        or re.fullmatch(r"tpu(?: v)?7x", result.device_kind.strip().lower()) is None
        or result.device_count != plan.device_count
        or result.mesh != plan.mesh
        or result.execution_scope != plan.execution_scope
        or result.stablehlo_sha256 != _sha256(artifacts["stablehlo.txt"])
        or result.compiler_hlo_sha256 != _sha256(artifacts["compiler_hlo.txt"])
    ):
        raise ValueError(f"SEQAX_PALLAS_EXECUTION_IDENTITY_MISMATCH phase={phase}")
    _validate_compiled_program(
        artifacts["stablehlo.txt"].read_text(),
        artifacts["compiler_hlo.txt"].read_text(),
        pallas_region_count=plan.pallas_region_count,
        all_gather_count=_physical_collective_counts(physical)[0],
        reduce_scatter_count=_physical_collective_counts(physical)[1],
    )

    input_paths = tuple(
        artifacts[f"inputs/{index:02d}.npy"] for index in range(len(experiment.workload.inputs))
    )
    inputs = tuple(np.load(path, allow_pickle=False) for path in input_paths)
    outputs = (np.load(artifacts["outputs/00.npy"], allow_pickle=False),)
    saved_oracles = (np.load(artifacts["oracle/00.npy"], allow_pickle=False),)
    for value, contract in zip(inputs, experiment.workload.inputs, strict=True):
        _validate_array_contract(
            value,
            name=contract.name,
            shape=contract.shape,
            dtype=contract.dtype,
            phase=phase,
        )
    for values, label in ((outputs, "output"), (saved_oracles, "oracle")):
        for value, contract in zip(values, experiment.workload.outputs, strict=True):
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
        raise ValueError(f"SEQAX_PALLAS_INPUT_REPLAY_MISMATCH phase={phase}")
    expected_oracle = np.asarray(
        seqax_forward_canonical_reference(inputs, **SEQAX_EVIDENCE_PARAMETERS)
    )
    if not np.array_equal(saved_oracles[0], expected_oracle):
        raise ValueError(f"SEQAX_PALLAS_ORACLE_REPLAY_MISMATCH phase={phase}")
    if (
        tuple(array_sha256(value) for value in inputs) != result.input_sha256
        or tuple(array_sha256(value) for value in outputs) != result.output_sha256
        or tuple(array_sha256(value) for value in saved_oracles) != result.oracle_sha256
    ):
        raise ValueError(f"SEQAX_PALLAS_ARRAY_IDENTITY_MISMATCH phase={phase}")
    maximum_absolute_error, maximum_relative_error = _errors(outputs[0], expected_oracle)
    passed = np.allclose(
        outputs[0],
        expected_oracle,
        atol=SEQAX_OUTPUT_ATOL,
        rtol=SEQAX_OUTPUT_RTOL,
    )
    if (
        passed is not result.correctness_passed
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
        raise ValueError(f"SEQAX_PALLAS_CORRECTNESS_REPLAY_MISMATCH phase={phase}")
    if not passed:
        raise ValueError(f"SEQAX_PALLAS_PHASE_CORRECTNESS_FAILED phase={phase}")
    expected_run_id = semantic_sha256(
        "seqax-physical-pallas-forward-run-v1",
        result.mode.value,
        plan.distributed_schedule_sha256,
        plan.physical_schedule_sha256,
        plan.source_sha256(),
        str(SEQAX_EVIDENCE_SEED),
    )
    if result.run_id != expected_run_id:
        raise ValueError(f"SEQAX_PALLAS_RUN_ID_MISMATCH phase={phase}")
    terminal = {
        RunMode.TIMING: RunState.TIMED,
        RunMode.TRACE: RunState.TRACED,
        RunMode.COUNTERS: RunState.COUNTERED,
    }[result.mode]
    terminal_payload: dict[str, object] = {
        "warmup_iterations": SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        "measured_iterations": SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        "mesh": dict(plan.mesh),
    }
    if result.mode is RunMode.TIMING:
        if len(result.samples_ns) != SEQAX_EVIDENCE_MEASURED_ITERATIONS or any(
            sample <= 0 for sample in result.samples_ns
        ):
            raise ValueError("SEQAX_PALLAS_TIMING_SAMPLE_PROTOCOL_MISMATCH")
        median = statistics.median(result.samples_ns)
        ordered = sorted(result.samples_ns)
        p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
        coefficient = statistics.pstdev(result.samples_ns) / statistics.mean(result.samples_ns)
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
            raise ValueError("SEQAX_PALLAS_TIMING_STATISTIC_MISMATCH")
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
            raise ValueError(f"SEQAX_PALLAS_PROFILE_CONTAINS_TIMING_CLAIMS phase={phase}")
        xplane = next(
            artifact
            for artifact in result.artifacts
            if artifact.role
            is (ArtifactRole.TIMING_TRACE if phase == "trace" else ArtifactRole.COUNTER_TRACE)
        )
        terminal_payload.update(
            profile_root="profile",
            xplane_sha256=xplane.sha256,
            xplane_size_bytes=xplane.size_bytes,
            profile_auxiliary_artifacts=tuple(
                {
                    "path": artifact.path,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                }
                for artifact in sorted(
                    (
                        artifact
                        for artifact in result.artifacts
                        if artifact.role is ArtifactRole.PROFILE_AUXILIARY
                    ),
                    key=lambda artifact: artifact.path,
                )
            ),
        )
    expected_payloads = (
        expected_invocation.model_dump(mode="json"),
        {
            "distributed_schedule_sha256": plan.distributed_schedule_sha256,
            "physical_schedule_sha256": plan.physical_schedule_sha256,
        },
        {
            "physical_schedule_sha256": plan.physical_schedule_sha256,
            "pallas_source_sha256": result.pallas_source_sha256,
            "plan_manifest_sha256": result.plan_manifest_sha256,
            "pallas_region_count": plan.pallas_region_count,
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
        raise ValueError(f"SEQAX_PALLAS_LEDGER_REPLAY_MISMATCH phase={phase}")
    return maximum_absolute_error, maximum_relative_error


def _validate_xprof_exports(
    output_root: Path,
    *,
    phase_root: Path,
    expected_xplane: Path,
) -> None:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("SEQAX_PALLAS_XPROF_MANIFEST_MISSING")
    manifest = XProfExportManifest.model_validate_json(manifest_path.read_text())
    expected_xplane_path = expected_xplane.relative_to(phase_root)
    if manifest.xplane != expected_xplane_path:
        raise ValueError("SEQAX_PALLAS_XPROF_XPLANE_IDENTITY_MISMATCH")
    if tuple(sorted(set(manifest.available_tools))) != manifest.available_tools:
        raise ValueError("SEQAX_PALLAS_XPROF_AVAILABLE_TOOLS_NOT_CANONICAL")
    derived_manifest_path = output_root / "derived_manifest.json"
    if not derived_manifest_path.is_file():
        raise ValueError("SEQAX_PALLAS_XPROF_DERIVED_MANIFEST_MISSING")
    derived_manifest = _XProfDerivedManifest.model_validate_json(derived_manifest_path.read_text())
    derived_paths: set[str] = set()
    for artifact in derived_manifest.artifacts:
        if (
            artifact.role is not ArtifactRole.XPROF_EXPORT
            or not artifact.path.startswith("derived/")
            or not artifact.path.endswith((".hlo_proto.pb", "/ALL_HOSTS.op_stats_v2.pb"))
            or artifact.path in derived_paths
        ):
            raise ValueError("SEQAX_PALLAS_XPROF_DERIVED_ARTIFACT_INVALID")
        path = (output_root / artifact.path).resolve()
        if not path.is_relative_to(output_root.resolve()) or path.is_symlink():
            raise ValueError("SEQAX_PALLAS_XPROF_DERIVED_ARTIFACT_UNSAFE")
        if (
            not path.is_file()
            or path.stat().st_size != artifact.size_bytes
            or _sha256(path) != artifact.sha256
        ):
            raise ValueError("SEQAX_PALLAS_XPROF_DERIVED_ARTIFACT_MISMATCH")
        derived_paths.add(artifact.path)
    if not any(path.endswith(".hlo_proto.pb") for path in derived_paths):
        raise ValueError("SEQAX_PALLAS_XPROF_HLO_PROTO_MISSING")
    if not any(path.endswith("/ALL_HOSTS.op_stats_v2.pb") for path in derived_paths):
        raise ValueError("SEQAX_PALLAS_XPROF_OP_STATS_MISSING")
    expected_tools = tuple(tool for tool in DEFAULT_TOOLS if tool in set(manifest.available_tools))
    observed_tools = tuple(export.tool for export in manifest.exports)
    if observed_tools != expected_tools or len(observed_tools) != len(set(observed_tools)):
        raise ValueError("SEQAX_PALLAS_XPROF_TOOL_SET_MISMATCH")
    expected_names: set[str] = set()
    for export in manifest.exports:
        suffix = ".json" if export.mime_type == "application/json" else ".bin"
        expected_output = Path("xprof") / f"{export.tool}{suffix}"
        if export.output != expected_output:
            raise ValueError(f"SEQAX_PALLAS_XPROF_OUTPUT_PATH_MISMATCH tool={export.tool}")
        expected_names.add(expected_output.name)
    expected = {"manifest.json", "derived_manifest.json", *expected_names, *derived_paths}
    observed = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if observed != expected or len(expected) != (
        len(manifest.exports) + len(derived_manifest.artifacts) + 2
    ):
        raise ValueError("SEQAX_PALLAS_XPROF_EXPORT_SET_MISMATCH")
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_PALLAS_XPROF_SYMLINK path={path}")
    for export in manifest.exports:
        path = output_root / export.output.name
        if path.stat().st_size != export.size_bytes:
            raise ValueError(f"SEQAX_PALLAS_XPROF_EXPORT_SIZE_MISMATCH tool={export.tool}")
    if not any(export.tool == "hlo_stats" for export in manifest.exports):
        raise ValueError("SEQAX_PALLAS_XPROF_HLO_STATS_MISSING")


def _export_xprof_isolated(
    expected_xplane: Path,
    temporary: Path,
) -> XProfExportManifest:
    staging = temporary / ".xprof-input"
    staging.mkdir(parents=True)
    staged_xplane = staging / expected_xplane.name
    shutil.copy2(expected_xplane, staged_xplane)
    manifest = export_xprof_capture(staging, temporary)
    derived_root = temporary / "derived"
    derived_root.mkdir()
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path == staged_xplane:
            continue
        relative = path.relative_to(staging)
        if not relative.as_posix().endswith((".hlo_proto.pb", "ALL_HOSTS.op_stats_v2.pb")):
            raise ValueError(f"SEQAX_PALLAS_XPROF_DERIVED_ARTIFACT_UNRECOGNIZED path={relative}")
        destination = derived_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, destination)
    shutil.rmtree(staging)
    derived_manifest = _XProfDerivedManifest(
        artifacts=tuple(
            _reference(temporary, path, ArtifactRole.XPROF_EXPORT)
            for path in sorted(derived_root.rglob("*"))
            if path.is_file()
        )
    )
    (temporary / "derived_manifest.json").write_text(
        derived_manifest.model_dump_json(indent=2) + "\n"
    )
    return manifest


def _ensure_exports(
    root: Path,
    phase: str,
    result: SeqaxPallasRunResult,
) -> None:
    phase_root = root / phase
    result_artifacts = _result_artifacts(root, phase, result)
    xplanes = tuple(
        path
        for declared, path in result_artifacts.items()
        if declared.startswith("profile/") and declared.endswith(".xplane.pb")
    )
    if len(xplanes) != 1:
        raise ValueError(
            f"SEQAX_PALLAS_PROFILE_XPLANE_COUNT_MISMATCH phase={phase} observed={xplanes}"
        )
    expected_xplane = xplanes[0]
    output_root = phase_root / "xprof"
    if output_root.exists():
        _validate_xprof_exports(
            output_root,
            phase_root=phase_root,
            expected_xplane=expected_xplane,
        )
        return
    temporary = phase_root / f"xprof.tmp-{time.time_ns()}"
    try:
        manifest = _export_xprof_isolated(expected_xplane, temporary)
        portable = manifest.model_copy(
            update={
                "xplane": expected_xplane.relative_to(phase_root),
                "exports": tuple(
                    export.model_copy(update={"output": Path("xprof") / export.output.name})
                    for export in manifest.exports
                ),
            }
        )
        (temporary / "manifest.json").write_text(
            json.dumps(portable.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _validate_xprof_exports(
            temporary,
            phase_root=phase_root,
            expected_xplane=expected_xplane,
        )
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            archive = root.parent / (f"{root.name}-{phase}-xprof-failed-{time.time_ns()}")
            temporary.rename(archive)
        raise


def _bound_program(assessment):
    matches = tuple(
        program
        for program in assessment.capture.programs
        if program.program_id in assessment.capture.timed_program_ids
        and program.timed_self_us > 0
        and all(program.marker_counts.get(marker, 0) > 0 for marker in _PROFILE_MARKERS)
    )
    if len(matches) != 1:
        raise ValueError(f"SEQAX_PALLAS_BOUND_TIMED_PROGRAM_COUNT_MISMATCH observed={matches}")
    return matches[0]


def _bound_program_id(assessment) -> str:
    return _bound_program(assessment).program_id


def _validate_capture(assessment, *, counters: bool) -> None:
    if not assessment.accepted:
        raise ValueError(
            "SEQAX_PALLAS_PROFILE_REJECTED "
            f"findings={[(value.code, value.severity.value) for value in assessment.findings]}"
        )
    _bound_program_id(assessment)
    observed_planes = {
        plane.name
        for plane in assessment.capture.planes
        if plane.name.startswith("/device:TPU:") and "SparseCore" not in plane.name
    }
    expected_planes = {f"/device:TPU:{index}" for index in range(8)}
    if observed_planes != expected_planes:
        raise ValueError(
            f"SEQAX_PALLAS_TPU_PLANE_SET_MISMATCH expected={sorted(expected_planes)} "
            f"observed={sorted(observed_planes)}"
        )
    observed_counter_cores = set(assessment.capture.counters.periodic_samples_per_tpu_core)
    expected_counter_cores = {"0", "2", "4", "6"} if counters else set()
    if observed_counter_cores != expected_counter_cores:
        raise ValueError(
            f"SEQAX_PALLAS_COUNTER_CORE_SET_MISMATCH "
            f"expected={sorted(expected_counter_cores)} "
            f"observed={sorted(observed_counter_cores)}"
        )
    if counters:
        names = assessment.capture.counters.periodic_counter_names
        if not any(name.startswith("COUNT_MXU_BUSY") for name in names):
            raise ValueError("SEQAX_PALLAS_MXU_PERIODIC_COUNTER_MISSING")


def _profile_event_replay(root: Path, phase: str, assessment) -> tuple[float, ...]:
    program = _bound_program(assessment)
    xplane = next((root / phase / "profile").rglob("*.xplane.pb"))
    profile = profile_data.ProfileData.from_file(xplane)
    try:
        steps = 0
        durations: list[float] = []
        for plane in profile.planes:
            for line in plane.lines:
                for event in line.events:
                    steps += event.name == _STEP_EVENT
                    if (
                        plane.name == "/device:TPU:0"
                        and line.name == "XLA Modules"
                        and event.name == program.name
                    ):
                        durations.append(float(event.duration_ns))
    finally:
        profile.close()
    if steps != SEQAX_EVIDENCE_MEASURED_ITERATIONS:
        raise ValueError(
            f"SEQAX_PALLAS_PROFILE_STEP_COUNT_MISMATCH phase={phase} "
            f"expected={SEQAX_EVIDENCE_MEASURED_ITERATIONS} observed={steps}"
        )
    if len(durations) != SEQAX_EVIDENCE_MEASURED_ITERATIONS or any(
        value <= 0 for value in durations
    ):
        raise ValueError(
            f"SEQAX_PALLAS_MODULE_EXECUTION_COUNT_MISMATCH phase={phase} "
            f"expected={SEQAX_EVIDENCE_MEASURED_ITERATIONS} observed={len(durations)}"
        )
    return tuple(durations)


def _timing_metrics(root: Path, result: SeqaxPallasRunResult) -> tuple[Metric, ...]:
    source = MetricSource(
        artifact_sha256=_sha256(root / "timing/result.json"),
        artifact_path="timing/result.json",
        tool="tpu-cake",
        field="samples_ns",
    )
    interval = MeasurementInterval(
        scope="one synchronized resident-input eight-device physical Pallas Seqax forward"
    )
    assert result.p90_ns is not None
    assert result.coefficient_of_variation is not None
    mean = Decimal(str(statistics.mean(result.samples_ns)))
    deviation = Decimal(str(statistics.pstdev(result.samples_ns)))
    return (
        Metric(
            name="median_synchronized_physical_pallas_forward_duration",
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
            name="p90_synchronized_physical_pallas_forward_duration",
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
            name="synchronized_physical_pallas_forward_coefficient_of_variation",
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


def _prefix_metrics(prefix: str, metrics: tuple[Metric, ...], root: Path) -> tuple[Metric, ...]:
    return tuple(
        metric.model_copy(
            update={
                "name": f"{prefix}_{metric.name}",
                "sources": tuple(
                    source.model_copy(
                        update={
                            "artifact_path": Path(source.artifact_path)
                            .resolve()
                            .relative_to(root)
                            .as_posix()
                        }
                    )
                    for source in metric.sources
                ),
            }
        )
        for metric in metrics
    )


def _profile_metrics(
    root: Path,
    phase: str,
    assessment,
    durations: tuple[float, ...],
) -> tuple[Metric, ...]:
    ordered = sorted(durations)
    p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
    xplane = next((root / phase / "profile").rglob("*.xplane.pb"))
    source = MetricSource(
        artifact_sha256=_sha256(xplane),
        artifact_path=xplane.relative_to(root).as_posix(),
        tool="XPlane",
        field=f"/device:TPU:0/XLA Modules/{_bound_program(assessment).name}",
    )
    interval = MeasurementInterval(
        scope="one profiler-instrumented compiled eight-device physical Pallas forward"
    )
    step_source = MetricSource(
        artifact_sha256=_sha256(xplane),
        artifact_path=xplane.relative_to(root).as_posix(),
        tool="XPlane",
        field=f"StepTraceAnnotation({_STEP_EVENT})",
    )
    return (
        Metric(
            name=f"{phase}_physical_pallas_forward_count",
            quantity=Quantity(
                value=Decimal(SEQAX_EVIDENCE_MEASURED_ITERATIONS),
                unit=Unit.COUNT,
            ),
            kind=MeasurementKind.MEASURED,
            interval=MeasurementInterval(scope=f"the complete {phase} capture"),
            sources=(step_source,),
        ),
        Metric(
            name=f"{phase}_median_compiled_physical_pallas_forward_duration",
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
                expression="median(bound_xla_module_duration_ns)",
            ),
        ),
        Metric(
            name=f"{phase}_p90_compiled_physical_pallas_forward_duration",
            quantity=Quantity(value=Decimal(str(p90)), unit=Unit.NANOSECOND),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="nearest_rank_p90",
                version="1",
                expression="sorted(bound_xla_module_duration_ns)[round((n-1)*0.9)]",
            ),
        ),
    )


def _metrics(root: Path, results: dict[str, SeqaxPallasRunResult], replays):
    trace, trace_durations = replays["trace"]
    counters, counter_durations = replays["counters"]
    return (
        *_timing_metrics(root, results["timing"]),
        *_profile_metrics(root, "trace", trace, trace_durations),
        *_profile_metrics(root, "counters", counters, counter_durations),
        *_prefix_metrics("trace", capture_metrics(trace.capture), root),
        *_prefix_metrics("counter", capture_metrics(counters.capture), root),
    )


def _artifact_roles(
    root: Path,
    results: dict[str, SeqaxPallasRunResult],
    assessment_path: Path,
) -> dict[Path, ArtifactRole]:
    roles: dict[Path, ArtifactRole] = {}
    for phase, result in results.items():
        for artifact in result.artifacts:
            path = (root / phase / artifact.path).resolve()
            previous = roles.setdefault(path, artifact.role)
            if previous is not artifact.role:
                raise ValueError(f"SEQAX_PALLAS_ARTIFACT_ROLE_CONFLICT path={path}")
        roles[(root / phase / "result.json").resolve()] = {
            "timing": ArtifactRole.TIMING_SAMPLES,
            "trace": ArtifactRole.TRACE_RESULT,
            "counters": ArtifactRole.COUNTER_RESULT,
        }[phase]
    for phase in ("trace", "counters"):
        output_root = root / phase / "xprof"
        result_artifacts = _result_artifacts(root, phase, results[phase])
        xplanes = tuple(
            path
            for declared, path in result_artifacts.items()
            if declared.startswith("profile/") and declared.endswith(".xplane.pb")
        )
        if len(xplanes) != 1:
            raise ValueError(f"SEQAX_PALLAS_PROFILE_XPLANE_COUNT_MISMATCH phase={phase}")
        _validate_xprof_exports(
            output_root,
            phase_root=root / phase,
            expected_xplane=xplanes[0],
        )
        for path in sorted(output_root.rglob("*")):
            if not path.is_file():
                continue
            role = (
                ArtifactRole.HLO_STATS
                if path.name == "hlo_stats.json"
                else ArtifactRole.XPROF_EXPORT
            )
            roles[path.resolve()] = role
    roles[assessment_path.resolve()] = ArtifactRole.PROFILE_ASSESSMENT
    roles[(root / "finalizer/source_state.json").resolve()] = ArtifactRole.SOURCE_STATE
    roles[(root / "finalizer/source_diff.patch").resolve()] = ArtifactRole.SOURCE_DIFF
    receipt_path = (root / "receipt.json").resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"SEQAX_PALLAS_SYMLINK_ARTIFACT path={path}")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved not in roles and resolved != receipt_path:
            raise ValueError(f"SEQAX_PALLAS_UNMANIFESTED_ARTIFACT path={path}")
    return roles


def _phases(artifacts: tuple[ArtifactReference, ...]) -> tuple[EvidencePhase, ...]:
    grouped: dict[EvidencePhaseName, list[str]] = {phase: [] for phase in EvidencePhaseName}
    for artifact in artifacts:
        first = Path(artifact.path).parts[0]
        phase = (
            EvidencePhaseName(first)
            if first in {*_PHASES, "finalizer"}
            else EvidencePhaseName.AGGREGATE
        )
        grouped[phase].append(artifact.path)
    return tuple(
        EvidencePhase(name=phase, artifact_paths=tuple(paths)) for phase, paths in grouped.items()
    )


def _profile_replays(root: Path, experiment: KernelExperiment):
    trace = assess_capture(root / "trace", experiment.profile)
    counters = assess_capture(root / "counters", _counter_experiment(experiment).profile)
    _validate_capture(trace, counters=False)
    _validate_capture(counters, counters=True)
    replays = {}
    for phase, assessment in (("trace", trace), ("counters", counters)):
        replays[phase] = (assessment, _profile_event_replay(root, phase, assessment))
    return replays


def _require_owned_safe_root(root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    protected = {Path("/").resolve(), Path.home().resolve(), repo_root.resolve()}
    if root in protected or repo_root.resolve().is_relative_to(root):
        raise ValueError(f"SEQAX_PALLAS_UNSAFE_RUN_ROOT path={root}")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_PALLAS_RUN_ROOT_NOT_DIRECTORY path={root}")
    allowed_top_level = {
        "timing",
        "trace",
        "counters",
        "finalizer",
        "profile_assessment.json",
        "receipt.json",
    }
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in allowed_top_level)
    if unexpected:
        raise ValueError(f"SEQAX_PALLAS_UNOWNED_RUN_ROOT entries={unexpected}")
    for write_target in (
        root / "finalizer",
        root / "profile_assessment.json",
        root / "receipt.json",
        root / "receipt.json.tmp",
    ):
        if write_target.is_symlink():
            raise ValueError(f"SEQAX_PALLAS_WRITE_TARGET_SYMLINK path={write_target}")
        if write_target.is_file() and write_target.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_PALLAS_WRITE_TARGET_HARDLINK path={write_target}")
    for phase in _PHASES:
        phase_root = root / phase
        if phase_root.is_symlink() or not phase_root.is_dir():
            raise ValueError(f"SEQAX_PALLAS_PHASE_ROOT_NOT_OWNED phase={phase} path={phase_root}")
        for marker in ("invocation.json", "result.json"):
            if not (phase_root / marker).is_file():
                raise ValueError(
                    f"SEQAX_PALLAS_RUN_ROOT_OWNERSHIP_MISSING phase={phase} marker={marker}"
                )
    finalizer = root / "finalizer"
    if finalizer.exists():
        for path in finalizer.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"SEQAX_PALLAS_WRITE_TARGET_SYMLINK path={path}")
            if path.is_file() and path.stat().st_nlink != 1:
                raise ValueError(f"SEQAX_PALLAS_WRITE_TARGET_HARDLINK path={path}")
        allowed = {"source_state.json", "source_diff.patch"}
        observed = {
            path.relative_to(finalizer).as_posix()
            for path in finalizer.rglob("*")
            if path.is_file()
        }
        if observed - allowed:
            raise ValueError(
                f"SEQAX_PALLAS_UNOWNED_FINALIZER_FILES files={sorted(observed - allowed)}"
            )


def _preflight_phase_files(
    root: Path,
    phase: str,
    result: SeqaxPallasRunResult,
) -> None:
    phase_root = root / phase
    result_artifacts = _result_artifacts(root, phase, result)
    expected = {path.resolve() for path in result_artifacts.values()}
    expected.add((phase_root / "result.json").resolve())
    xprof = phase_root / "xprof"
    if xprof.exists():
        if phase == "timing":
            raise ValueError("SEQAX_PALLAS_TIMING_HAS_XPROF_EXPORTS")
        xplanes = tuple(
            path
            for declared, path in result_artifacts.items()
            if declared.startswith("profile/") and declared.endswith(".xplane.pb")
        )
        if len(xplanes) != 1:
            raise ValueError(f"SEQAX_PALLAS_PROFILE_XPLANE_COUNT_MISMATCH phase={phase}")
        _validate_xprof_exports(
            xprof,
            phase_root=phase_root,
            expected_xplane=xplanes[0],
        )
        expected.update(path.resolve() for path in xprof.rglob("*") if path.is_file())
    observed: set[Path] = set()
    for path in phase_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_PALLAS_SYMLINK_ARTIFACT path={path}")
        if path.is_file():
            observed.add(path.resolve())
    if observed != expected:
        raise ValueError(
            f"SEQAX_PALLAS_PHASE_FILE_SET_MISMATCH phase={phase} "
            f"missing={sorted(str(path) for path in expected - observed)} "
            f"extra={sorted(str(path) for path in observed - expected)}"
        )


def _require_clean_repository() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_PALLAS_FINALIZER_SOURCE_IS_DIRTY status={status}")


def build_seqax_pallas_receipt(root: Path, *, write_receipt: bool = True) -> RunReceipt:
    root = root.absolute()
    if root.is_symlink():
        raise ValueError(f"SEQAX_PALLAS_RUN_ROOT_SYMLINK path={root}")
    root = root.resolve()
    receipt_path = root / "receipt.json"
    if receipt_path.is_symlink():
        raise ValueError(f"SEQAX_PALLAS_WRITE_TARGET_SYMLINK path={receipt_path}")
    if receipt_path.exists():
        receipt = RunReceipt.model_validate_json(receipt_path.read_text())
        validate_seqax_pallas_receipt(receipt, root=root)
        return receipt
    _require_owned_safe_root(root)
    results = {phase: _load_result(root, phase) for phase in _PHASES}
    if len({_phase_identity(result) for result in results.values()}) != 1:
        raise ValueError("SEQAX_PALLAS_RUNS_DO_NOT_SHARE_EXECUTION_IDENTITY")
    experiment = _trusted_experiment()
    errors = tuple(_validate_phase(root, experiment, phase, results[phase]) for phase in _PHASES)
    for phase in _PHASES:
        _preflight_phase_files(root, phase, results[phase])
    phase_source_identities = {
        _source_identity(
            root / phase / "source_state.json",
            root / phase / "source_diff.patch",
            require_clean=True,
        )
        for phase in _PHASES
    }
    if len(phase_source_identities) != 1:
        raise ValueError("SEQAX_PALLAS_RUNS_DO_NOT_SHARE_SOURCE_IDENTITY")
    _require_clean_repository()
    finalizer = root / "finalizer"
    _source_state(Path(__file__).resolve().parents[2], finalizer)
    source_identities = {
        _source_identity(
            root / phase / "source_state.json",
            root / phase / "source_diff.patch",
            require_clean=True,
        )
        for phase in (*_PHASES, "finalizer")
    }
    if len(source_identities) != 1:
        raise ValueError("SEQAX_PALLAS_RUNS_DO_NOT_SHARE_SOURCE_IDENTITY")
    for phase in ("trace", "counters"):
        _ensure_exports(root, phase, results[phase])
    replays = _profile_replays(root, experiment)
    trace = replays["trace"][0]
    counters = replays["counters"][0]
    assessment_path = root / "profile_assessment.json"
    assessment_payload = _canonical_profile_assessment(
        _relative_json(
            {
                "timing_trace": trace.model_dump(mode="json"),
                "counter_trace": counters.model_dump(mode="json"),
            },
            root,
        )
    )
    assessment_path.write_text(json.dumps(assessment_payload, indent=2, sort_keys=True) + "\n")
    roles = _artifact_roles(root, results, assessment_path)
    artifacts = tuple(
        _reference(root, path, role)
        for path, role in sorted(roles.items(), key=lambda item: str(item[0]))
    )
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        evidence_profile=EvidenceProfile.SEQAX_PHYSICAL_PALLAS_FORWARD,
        schedule_sha256=results["timing"].physical_schedule_sha256,
        status=RunStatus.PASSED,
        runtime=results["timing"].runtime,
        correctness=CorrectnessResult(
            passed=True,
            oracle=experiment.workload.numerical.reference,
            maximum_absolute_error=max(value[0] for value in errors),
            maximum_relative_error=max(value[1] for value in errors),
        ),
        required_semantic_properties=(),
        metrics=_metrics(root, results, replays),
        artifacts=artifacts,
        phases=_phases(artifacts),
    )
    validate_seqax_pallas_receipt(receipt, root=root)
    if write_receipt:
        temporary = root / "receipt.json.tmp"
        temporary.write_text(receipt.model_dump_json(indent=2) + "\n")
        temporary.replace(receipt_path)
    return receipt


def validate_seqax_pallas_receipt(receipt: RunReceipt, *, root: Path) -> None:
    root = root.resolve()
    experiment = _trusted_experiment()
    if (
        receipt.evidence_profile is not EvidenceProfile.SEQAX_PHYSICAL_PALLAS_FORWARD
        or receipt.experiment_id != experiment.experiment_id
        or receipt.schedule_sha256 != experiment.schedule_sha256
        or receipt.status is not RunStatus.PASSED
    ):
        raise ValueError("SEQAX_PALLAS_RECEIPT_EXPERIMENT_IDENTITY_MISMATCH")
    for artifact in receipt.artifacts:
        path = resolve_bundle_artifact(root, artifact.path)
        if (
            not path.is_file()
            or path.stat().st_size != artifact.size_bytes
            or _sha256(path) != artifact.sha256
        ):
            raise ValueError(
                f"SEQAX_PALLAS_RECEIPT_ARTIFACT_IDENTITY_MISMATCH path={artifact.path}"
            )
    results = {phase: _load_result(root, phase) for phase in _PHASES}
    if len({_phase_identity(result) for result in results.values()}) != 1:
        raise ValueError("SEQAX_PALLAS_RUNS_DO_NOT_SHARE_EXECUTION_IDENTITY")
    expected_artifacts = tuple(
        _reference(root, path, role)
        for path, role in sorted(
            _artifact_roles(root, results, root / "profile_assessment.json").items(),
            key=lambda item: str(item[0]),
        )
    )
    if receipt.artifacts != expected_artifacts:
        raise ValueError("SEQAX_PALLAS_RECEIPT_ARTIFACT_MANIFEST_MISMATCH")
    if receipt.phases != _phases(expected_artifacts):
        raise ValueError("SEQAX_PALLAS_RECEIPT_PHASE_PARTITION_MISMATCH")
    source_identities = {
        _source_identity(
            root / phase / "source_state.json",
            root / phase / "source_diff.patch",
            require_clean=True,
        )
        for phase in (*_PHASES, "finalizer")
    }
    if len(source_identities) != 1:
        raise ValueError("SEQAX_PALLAS_RUNS_DO_NOT_SHARE_SOURCE_IDENTITY")
    errors = tuple(_validate_phase(root, experiment, phase, results[phase]) for phase in _PHASES)
    replays = _profile_replays(root, experiment)
    trace = replays["trace"][0]
    counters = replays["counters"][0]
    if (
        receipt.runtime != results["timing"].runtime
        or any(result.runtime != receipt.runtime for result in results.values())
        or not all(result.correctness_passed for result in results.values())
        or receipt.correctness.passed is not True
        or receipt.correctness.oracle != experiment.workload.numerical.reference
        or receipt.correctness.maximum_absolute_error != max(value[0] for value in errors)
        or receipt.correctness.maximum_relative_error != max(value[1] for value in errors)
        or receipt.required_semantic_properties
        or receipt.correctness.semantic_properties
    ):
        raise ValueError("SEQAX_PALLAS_RECEIPT_CORRECTNESS_OR_RUNTIME_MISMATCH")
    assessment_artifact = next(
        artifact
        for artifact in receipt.artifacts
        if artifact.role is ArtifactRole.PROFILE_ASSESSMENT
    )
    expected_assessment = _canonical_profile_assessment(
        _relative_json(
            {
                "timing_trace": trace.model_dump(mode="json"),
                "counter_trace": counters.model_dump(mode="json"),
            },
            root,
        )
    )
    saved_assessment = _canonical_profile_assessment(
        json.loads((root / assessment_artifact.path).read_text())
    )
    if saved_assessment != expected_assessment:
        raise ValueError("SEQAX_PALLAS_PROFILE_ASSESSMENT_REPLAY_MISMATCH")
    expected_metrics = _metrics(root, results, replays)
    if receipt.metrics != expected_metrics:
        raise ValueError("SEQAX_PALLAS_RECEIPT_METRICS_REPLAY_MISMATCH")
    artifacts_by_path = {artifact.path: artifact for artifact in receipt.artifacts}
    for metric in receipt.metrics:
        for source in metric.sources:
            artifact = artifacts_by_path.get(source.artifact_path)
            if artifact is None or artifact.sha256 != source.artifact_sha256:
                raise ValueError(
                    f"SEQAX_PALLAS_METRIC_SOURCE_NOT_BOUND metric={metric.name} "
                    f"path={source.artifact_path}"
                )
