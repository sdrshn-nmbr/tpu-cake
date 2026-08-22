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
from typing import Any

import jax
import numpy as np
from jax.sharding import NamedSharding
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.artifacts import file_sha256 as _sha256
from tpu_cake.artifacts import resolve_recorded_artifact
from tpu_cake.artifacts import save_relative_array_artifact as _save_array
from tpu_cake.artifacts import write_relative_text_artifact as _write_text
from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    ProfileExpectation,
    RuntimeIdentity,
    WorkloadStage,
)
from tpu_cake.identity import array_sha256, semantic_seed, semantic_sha256
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.ledger import RunState, payload_sha256, read_ledger_history
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
from tpu_cake.runner import (
    RunMode,
    _profiler_options,
    _record_event,
    _runtime_identity,
    _source_state,
)
from tpu_cake.seqax_runner import expected_seqax_profiler_contract
from tpu_cake.seqax_surface import (
    SEQAX_SURFACE_ATOL,
    SEQAX_SURFACE_RTOL,
    SeqaxSurfaceReceipt,
    _compiler_hlo,
    seqax_forward_workload_surface,
    validate_seqax_surface_receipt,
)
from tpu_cake.seqax_surface import (
    _artifact_roles as _surface_artifact_roles,
)
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)
from tpu_cake.xprof_evidence import (
    XPlaneIndex,
    assess_capture,
    capture_metrics,
    count_profile_events,
)
from tpu_cake.xprof_export import XProfExportManifest, export_xprof_capture

SEQAX_SURFACE_PROFILE_SCHEMA = "tpu-cake-seqax-surface-profile-v1"
SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS = 5
SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS = 50
SEQAX_SURFACE_PROFILE_MODES = (RunMode.TRACE, RunMode.COUNTERS)
_PROFILE_MARKERS = ("all-gather", "reduce_scatter", "dot_general")


class SeqaxSurfaceProfileInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^tpu-cake-seqax-surface-profile-v1$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario: str
    mode: RunMode
    seed: int
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    jax_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profiler_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: tuple[str, ...]
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_placement: str
    execution_scope: str
    runtime: RuntimeIdentity
    device_kind: str
    device_count: int = Field(gt=0)
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def protocol_is_fixed(self) -> SeqaxSurfaceProfileInvocation:
        if (
            self.warmup_iterations != SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS
            or self.measured_iterations != SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS
            or self.mode not in SEQAX_SURFACE_PROFILE_MODES
            or self.input_placement != "resident-named-sharding-before-warmup"
        ):
            raise ValueError("SEQAX_SURFACE_PROFILE_PROTOCOL_MISMATCH")
        return self


class SeqaxSurfaceProfileRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation: SeqaxSurfaceProfileInvocation
    backend: str
    mesh: tuple[tuple[str, int], ...]
    plan_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: tuple[str, ...]
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)
    compile_duration_ns: int = Field(ge=0)
    artifacts: tuple[ArtifactReference, ...]


class SeqaxSurfaceProfileReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^tpu-cake-seqax-surface-profile-v1$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: tuple[SeqaxSurfaceProfileRunResult, ...]
    metrics: tuple[Metric, ...]
    artifacts: tuple[ArtifactReference, ...]
    accepted: bool


def _json_sha256(value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _execution_identity_sha256(
    *,
    surface_receipt_sha256: str,
    scenario: str,
    schedule_sha256: str,
    jax_source_sha256: str,
    stablehlo_sha256: str,
    compiler_hlo_sha256: str,
    input_sha256: tuple[str, ...],
    output_sha256: str,
    oracle_sha256: str,
    runtime: RuntimeIdentity,
    device_kind: str,
    device_count: int,
) -> str:
    return semantic_sha256(
        SEQAX_SURFACE_PROFILE_SCHEMA,
        surface_receipt_sha256,
        scenario,
        schedule_sha256,
        jax_source_sha256,
        stablehlo_sha256,
        compiler_hlo_sha256,
        *input_sha256,
        output_sha256,
        oracle_sha256,
        json.dumps(runtime.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
        device_kind,
        str(device_count),
    )


def _artifact(root: Path, path: Path, role: ArtifactRole) -> ArtifactReference:
    return ArtifactReference(
        path=path.resolve().relative_to(root.resolve()).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
    )


def _write_json(
    root: Path,
    relative: Path,
    value: object,
    role: ArtifactRole,
) -> ArtifactReference:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return _artifact(root, path, role)


def _scenario(name: str):
    matches = tuple(
        scenario for scenario in seqax_forward_workload_surface().scenarios if scenario.name == name
    )
    if len(matches) != 1:
        raise ValueError(f"SEQAX_SURFACE_PROFILE_SCENARIO_UNKNOWN scenario={name}")
    return matches[0]


def _surface_receipt(surface_root: Path) -> SeqaxSurfaceReceipt:
    surface_root = surface_root.resolve()
    receipt = SeqaxSurfaceReceipt.model_validate_json((surface_root / "receipt.json").read_text())
    validate_seqax_surface_receipt(receipt, root=surface_root)
    if not receipt.candidate_promoted:
        raise ValueError("SEQAX_SURFACE_PROFILE_REQUIRES_PROMOTED_CANDIDATE")
    return receipt


def _expectation(scenario: str, mode: RunMode) -> ProfileExpectation:
    counters = mode is RunMode.COUNTERS
    return ProfileExpectation(
        name=f"seqax-surface-{scenario}-{mode.value}",
        stage=WorkloadStage.CONTROL,
        minimum_tpu_device_planes=8,
        require_tensor_core_activity=False,
        require_hbm_read_counters=counters,
        require_hbm_write_counters=counters,
        require_cycle_counters=counters,
        minimum_counter_device_planes=4 if counters else 0,
        required_timed_hlo_markers=_PROFILE_MARKERS,
    )


def _errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    observed = actual.astype(np.float32)
    reference = expected.astype(np.float32)
    absolute = np.abs(observed - reference)
    denominator = np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
    return float(absolute.max()), float((absolute / denominator).max())


def _phase_identity(result: SeqaxSurfaceProfileRunResult) -> tuple[object, ...]:
    invocation = result.invocation
    return (
        invocation.surface_id,
        invocation.surface_receipt_sha256,
        invocation.scenario,
        invocation.seed,
        invocation.schedule_sha256,
        invocation.jax_source_sha256,
        invocation.stablehlo_sha256,
        invocation.compiler_hlo_sha256,
        invocation.execution_identity_sha256,
        invocation.input_placement,
        invocation.execution_scope,
        invocation.runtime,
        invocation.device_kind,
        invocation.device_count,
        result.backend,
        result.mesh,
        result.plan_manifest_sha256,
        result.input_sha256,
        result.output_sha256,
        result.oracle_sha256,
    )


def _expected_result_role(path: str) -> ArtifactRole:
    fixed = {
        "invocation.json": ArtifactRole.INVOCATION,
        "profiler_config.json": ArtifactRole.PROFILER_CONFIG,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "lowered_jax.py": ArtifactRole.JAX_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
        "stablehlo.txt": ArtifactRole.STABLEHLO,
        "compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "oracle.npy": ArtifactRole.ORACLE_OUTPUT,
    }
    if path in fixed:
        return fixed[path]
    if re.fullmatch(r"inputs/\d{2}\.npy", path):
        return ArtifactRole.CORRECTNESS_INPUT
    raise ValueError(f"SEQAX_SURFACE_PROFILE_RESULT_PATH_UNKNOWN path={path}")


def _validate_phase_paths(output_dir: Path, surface_root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(output_dir == path or output_dir in path.parents for path in protected):
        raise ValueError(f"SEQAX_SURFACE_PROFILE_UNSAFE_OUTPUT_PATH path={output_dir}")
    if (
        output_dir == surface_root
        or surface_root in output_dir.parents
        or output_dir in surface_root.parents
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_OUTPUT_OVERLAPS_SURFACE")


def _require_clean_repository() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        raise ValueError("SEQAX_SURFACE_PROFILE_SOURCE_MUST_BE_CLEAN")


def _prepare_phase_output(
    output_dir: Path,
    *,
    surface_root: Path,
    surface_id: str,
    surface_receipt_sha256: str,
    scenario: str,
    mode: RunMode,
) -> SeqaxSurfaceProfileRunResult | None:
    _validate_phase_paths(output_dir, surface_root)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return None
    if not any(output_dir.iterdir()):
        return None
    required = (
        output_dir / "invocation.json",
        output_dir / "profiler_config.json",
        output_dir / "source_state.json",
        output_dir / "source_diff.patch",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(f"SEQAX_SURFACE_PROFILE_OUTPUT_NOT_OWNED path={output_dir}")
    invocation = SeqaxSurfaceProfileInvocation.model_validate_json(
        (output_dir / "invocation.json").read_text()
    )
    if (
        invocation.surface_id != surface_id
        or invocation.surface_receipt_sha256 != surface_receipt_sha256
        or invocation.scenario != scenario
        or invocation.mode is not mode
    ):
        raise ValueError(f"SEQAX_SURFACE_PROFILE_OUTPUT_NOT_OWNED path={output_dir}")
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = SeqaxSurfaceProfileRunResult.model_validate_json(result_path.read_text())
        if result.invocation != invocation:
            raise ValueError("SEQAX_SURFACE_PROFILE_RESUME_IDENTITY_MISMATCH")
        _source_identity(
            output_dir / "source_state.json",
            output_dir / "source_diff.patch",
            require_clean=True,
        )
        if len(tuple((output_dir / "profile").rglob("*.xplane.pb"))) != 1:
            raise ValueError("SEQAX_SURFACE_PROFILE_RESUME_XPLANE_MISMATCH")
        return result
    archived = output_dir.with_name(f"{output_dir.name}.incomplete-{time.time_ns()}")
    output_dir.rename(archived)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"SEQAX_SURFACE_PROFILE_ARCHIVED_INCOMPLETE source={output_dir} archive={archived}")
    return None


def run_seqax_surface_profile_phase(
    output_dir: Path,
    *,
    surface_root: Path,
    scenario_name: str,
    mode: RunMode,
) -> SeqaxSurfaceProfileRunResult:
    if mode not in SEQAX_SURFACE_PROFILE_MODES:
        raise ValueError("SEQAX_SURFACE_PROFILE_REQUIRES_TRACE_OR_COUNTERS")
    if jax.default_backend() != "tpu":
        raise ValueError("Seqax surface profiling requires a TPU backend")
    output_dir = output_dir.resolve()
    surface_root = surface_root.resolve()
    _validate_phase_paths(output_dir, surface_root)
    _require_clean_repository()
    surface_receipt = _surface_receipt(surface_root)
    surface_receipt_sha256 = _sha256(surface_root / "receipt.json")
    surface = seqax_forward_workload_surface()
    scenario = _scenario(scenario_name)
    resumed = _prepare_phase_output(
        output_dir,
        surface_root=surface_root,
        surface_id=surface.surface_id,
        surface_receipt_sha256=surface_receipt_sha256,
        scenario=scenario.name,
        mode=mode,
    )
    if resumed is not None:
        return resumed
    module = seqax_forward_schedule(**scenario.parameters())
    module.verify()
    plan = lower_distributed_program_to_jax_mesh(module)
    devices = tuple(jax.devices())
    kinds = {device.device_kind for device in devices}
    if len(devices) != plan.device_count or kinds != {"TPU7x"}:
        raise ValueError(
            "SEQAX_SURFACE_PROFILE_DEVICE_MISMATCH "
            f"expected={plan.device_count}/TPU7x observed={len(devices)}/{sorted(kinds)}"
        )
    runtime = _runtime_identity()
    if (
        runtime != surface_receipt.invocation.runtime
        or len(devices) != surface_receipt.invocation.device_count
        or "TPU7x" != surface_receipt.invocation.device_kind
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_RUNTIME_DIFFERS_FROM_SURFACE")
    seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
    expected_stablehlo = surface_root / "hlo" / scenario.name / "candidate_stablehlo.txt"
    expected_compiler_hlo = surface_root / "hlo" / scenario.name / "candidate_compiler_hlo.txt"
    profiler_contract = expected_seqax_profiler_contract(mode)
    if runtime.xla != profiler_contract["libtpu_init_args"]:
        raise ValueError("SEQAX_SURFACE_PROFILE_LIBTPU_ARGS_MISMATCH")
    host_inputs = tuple(
        np.asarray(value) for value in seqax_forward_inputs(seed=seed, **scenario.parameters())
    )
    surface_inputs = tuple(
        np.load(
            surface_root / "inputs" / scenario.name / f"{index:02d}.npy",
            allow_pickle=False,
        )
        for index in range(len(host_inputs))
    )
    if any(
        saved.shape != expected.shape
        or saved.dtype != expected.dtype
        or not np.array_equal(saved, expected)
        for saved, expected in zip(surface_inputs, host_inputs, strict=True)
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_INPUT_BINDING_MISMATCH")
    surface_output = np.load(
        surface_root / "outputs" / scenario.name / "candidate.npy",
        allow_pickle=False,
    )
    surface_oracle = np.load(
        surface_root / "oracle" / f"{scenario.name}.npy",
        allow_pickle=False,
    )
    input_sha256 = tuple(array_sha256(value) for value in host_inputs)
    output_sha256 = array_sha256(surface_output)
    oracle_sha256 = array_sha256(surface_oracle)
    stablehlo_sha256 = _sha256(expected_stablehlo)
    compiler_hlo_sha256 = _sha256(expected_compiler_hlo)
    execution_identity_sha256 = _execution_identity_sha256(
        surface_receipt_sha256=surface_receipt_sha256,
        scenario=scenario.name,
        schedule_sha256=plan.schedule_sha256,
        jax_source_sha256=plan.source_sha256(),
        stablehlo_sha256=stablehlo_sha256,
        compiler_hlo_sha256=compiler_hlo_sha256,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        oracle_sha256=oracle_sha256,
        runtime=runtime,
        device_kind="TPU7x",
        device_count=len(devices),
    )
    profiler_config_sha256 = _json_sha256(profiler_contract)
    run_id = semantic_sha256(
        SEQAX_SURFACE_PROFILE_SCHEMA,
        execution_identity_sha256,
        mode.value,
        profiler_config_sha256,
    )
    invocation = SeqaxSurfaceProfileInvocation(
        schema_version=SEQAX_SURFACE_PROFILE_SCHEMA,
        surface_id=surface.surface_id,
        surface_receipt_sha256=surface_receipt_sha256,
        scenario=scenario.name,
        mode=mode,
        seed=seed,
        warmup_iterations=SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS,
        schedule_sha256=plan.schedule_sha256,
        jax_source_sha256=plan.source_sha256(),
        stablehlo_sha256=stablehlo_sha256,
        compiler_hlo_sha256=compiler_hlo_sha256,
        profiler_config_sha256=profiler_config_sha256,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        oracle_sha256=oracle_sha256,
        execution_identity_sha256=execution_identity_sha256,
        input_placement="resident-named-sharding-before-warmup",
        execution_scope=plan.execution_scope,
        runtime=runtime,
        device_kind="TPU7x",
        device_count=len(devices),
        run_id=run_id,
    )
    source_artifacts = _source_state(Path(__file__).resolve().parents[2], output_dir)
    source_state = json.loads((output_dir / "source_state.json").read_text())
    if source_state["git_dirty"]:
        raise ValueError("SEQAX_SURFACE_PROFILE_SOURCE_MUST_BE_CLEAN")
    artifacts = [
        _write_json(
            output_dir,
            Path("invocation.json"),
            invocation.model_dump(mode="json"),
            ArtifactRole.INVOCATION,
        ),
        _write_json(
            output_dir,
            Path("profiler_config.json"),
            profiler_contract,
            ArtifactRole.PROFILER_CONFIG,
        ),
        *source_artifacts,
    ]
    ledger_path = output_dir / "ledger.sqlite"
    _record_event(ledger_path, run_id, RunState.CREATED, invocation.model_dump(mode="json"))
    distributed = _write_text(
        output_dir,
        Path("distributed.xdsl"),
        canonical_text(module),
        ArtifactRole.DISTRIBUTED_IR,
    )
    artifacts.append(distributed)
    _record_event(
        ledger_path,
        run_id,
        RunState.VERIFIED,
        {"schedule_sha256": plan.schedule_sha256},
    )
    lowered_source = _write_text(
        output_dir,
        Path("lowered_jax.py"),
        plan.render_executable_source(),
        ArtifactRole.JAX_SOURCE,
    )
    manifest = _write_json(
        output_dir,
        Path("plan_manifest.json"),
        plan.manifest(),
        ArtifactRole.PLAN_MANIFEST,
    )
    artifacts.extend((lowered_source, manifest))
    _record_event(
        ledger_path,
        run_id,
        RunState.LOWERED,
        {
            "jax_source_sha256": lowered_source.sha256,
            "plan_manifest_sha256": manifest.sha256,
            "execution_scope": plan.execution_scope,
        },
    )
    executable, mesh = plan.build(devices=devices)
    resident_inputs = tuple(
        jax.device_put(value, NamedSharding(mesh, spec))
        for value, spec in zip(host_inputs, plan.input_partition_specs, strict=True)
    )
    compile_started = time.perf_counter_ns()
    lowered = executable.lower(*resident_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo")) + "\n"
    compiler_hlo = _compiler_hlo(lowered) + "\n"
    compiled = lowered.compile()
    compile_duration_ns = time.perf_counter_ns() - compile_started
    stablehlo_artifact = _write_text(
        output_dir,
        Path("stablehlo.txt"),
        stablehlo,
        ArtifactRole.STABLEHLO,
    )
    compiler_hlo_artifact = _write_text(
        output_dir,
        Path("compiler_hlo.txt"),
        compiler_hlo,
        ArtifactRole.COMPILER_HLO,
    )
    artifacts.extend((stablehlo_artifact, compiler_hlo_artifact))
    if (
        stablehlo_artifact.sha256 != invocation.stablehlo_sha256
        or compiler_hlo_artifact.sha256 != invocation.compiler_hlo_sha256
    ):
        raise ValueError(f"SEQAX_SURFACE_PROFILE_HLO_BINDING_MISMATCH scenario={scenario.name}")
    _record_event(
        ledger_path,
        run_id,
        RunState.COMPILED,
        {
            "stablehlo_sha256": stablehlo_artifact.sha256,
            "compiler_hlo_sha256": compiler_hlo_artifact.sha256,
            "compile_duration_ns": compile_duration_ns,
        },
    )
    actual_device = compiled(*resident_inputs)[0]
    actual_device.block_until_ready()
    actual = np.asarray(jax.device_get(actual_device))
    oracle = np.asarray(
        seqax_forward_canonical_reference(
            host_inputs,
            quantization_decimals=surface.oracle_quantization_decimals,
            **scenario.parameters(),
        )
    )
    if (
        oracle.shape != surface_oracle.shape
        or oracle.dtype != surface_oracle.dtype
        or not np.array_equal(oracle, surface_oracle)
        or actual.shape != surface_output.shape
        or actual.dtype != surface_output.dtype
        or not np.array_equal(actual, surface_output)
    ):
        raise ValueError(f"SEQAX_SURFACE_PROFILE_OUTPUT_BINDING_MISMATCH scenario={scenario.name}")
    maximum_absolute_error, maximum_relative_error = _errors(actual, oracle)
    passed = bool(np.allclose(actual, oracle, atol=SEQAX_SURFACE_ATOL, rtol=SEQAX_SURFACE_RTOL))
    if not passed:
        raise ValueError(
            "SEQAX_SURFACE_PROFILE_CORRECTNESS_FAILED "
            f"scenario={scenario.name} absolute={maximum_absolute_error} "
            f"relative={maximum_relative_error}"
        )
    _record_event(
        ledger_path,
        run_id,
        RunState.CORRECT,
        {
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
        },
    )
    for _ in range(SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS):
        jax.block_until_ready(compiled(*resident_inputs))
    profile_root = output_dir / "profile"
    jax.profiler.start_trace(profile_root, profiler_options=_profiler_options(mode))
    event_name = f"seqax_surface_{scenario.name}"
    try:
        for step in range(SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS):
            with jax.profiler.StepTraceAnnotation(event_name, step_num=step):
                jax.block_until_ready(compiled(*resident_inputs))
    finally:
        jax.profiler.stop_trace()
    terminal = RunState.TRACED if mode is RunMode.TRACE else RunState.COUNTERED
    _record_event(
        ledger_path,
        run_id,
        terminal,
        {
            "event_name": event_name,
            "warmup_iterations": SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS,
            "measured_iterations": SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS,
            "profile_root": "profile",
        },
    )
    artifacts.extend(
        _save_array(
            output_dir,
            Path("inputs") / f"{index:02d}.npy",
            value,
            ArtifactRole.CORRECTNESS_INPUT,
        )
        for index, value in enumerate(host_inputs)
    )
    artifacts.extend(
        (
            _save_array(
                output_dir,
                Path("output.npy"),
                actual,
                ArtifactRole.CORRECTNESS_OUTPUT,
            ),
            _save_array(
                output_dir,
                Path("oracle.npy"),
                oracle,
                ArtifactRole.ORACLE_OUTPUT,
            ),
        )
    )
    artifacts.append(_artifact(output_dir, ledger_path, ArtifactRole.EXECUTION_LEDGER))
    result = SeqaxSurfaceProfileRunResult(
        invocation=invocation,
        backend=jax.default_backend(),
        mesh=tuple(plan.mesh_axes),
        plan_manifest_sha256=manifest.sha256,
        input_sha256=tuple(array_sha256(value) for value in host_inputs),
        output_sha256=array_sha256(actual),
        oracle_sha256=array_sha256(oracle),
        passed=True,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        compile_duration_ns=compile_duration_ns,
        artifacts=tuple(sorted(artifacts, key=lambda artifact: artifact.path)),
    )
    (output_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def _load_result(root: Path, scenario: str, mode: RunMode) -> SeqaxSurfaceProfileRunResult:
    result = SeqaxSurfaceProfileRunResult.model_validate_json(
        (root / scenario / mode.value / "result.json").read_text()
    )
    if result.invocation.scenario != scenario or result.invocation.mode is not mode:
        raise ValueError("SEQAX_SURFACE_PROFILE_RESULT_LOCATION_MISMATCH")
    return result


def _owned_bundle_results(root: Path) -> tuple[SeqaxSurfaceProfileRunResult, ...]:
    if not root.is_dir():
        raise ValueError(f"SEQAX_SURFACE_PROFILE_ROOT_NOT_OWNED path={root}")
    results = []
    for scenario in seqax_forward_workload_surface().scenarios:
        for mode in SEQAX_SURFACE_PROFILE_MODES:
            phase_root = root / scenario.name / mode.value
            if (
                not (phase_root / "invocation.json").is_file()
                or not (phase_root / "result.json").is_file()
            ):
                raise ValueError(f"SEQAX_SURFACE_PROFILE_ROOT_NOT_OWNED path={root}")
            results.append(_load_result(root, scenario.name, mode))
    return tuple(results)


def _profile_phase_root(root: Path, result: SeqaxSurfaceProfileRunResult) -> Path:
    return root / result.invocation.scenario / result.invocation.mode.value


def _canonical_assessment(value: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value))
    capture = normalized.get("capture")
    if isinstance(capture, dict) and isinstance(capture.get("timed_program_ids"), list):
        capture["timed_program_ids"] = sorted(capture["timed_program_ids"])
    return normalized


def _bound_program_id(assessment) -> str:
    matches = tuple(
        program.program_id
        for program in assessment.capture.programs
        if program.program_id in assessment.capture.timed_program_ids
        and program.timed_self_us > 0
        and all(program.marker_counts.get(marker, 0) > 0 for marker in _PROFILE_MARKERS)
    )
    if len(matches) != 1:
        raise ValueError(f"SEQAX_SURFACE_PROFILE_BOUND_PROGRAM_MISMATCH observed={matches}")
    return matches[0]


def _module_durations(phase_root: Path, assessment) -> tuple[float, ...]:
    program_id = _bound_program_id(assessment)
    xplanes = tuple((phase_root / "profile").rglob("*.xplane.pb"))
    if len(xplanes) != 1:
        raise ValueError("SEQAX_SURFACE_PROFILE_XPLANE_COUNT_MISMATCH")
    durations = tuple(
        float(event.duration_ns)
        for plane in XPlaneIndex.from_file(xplanes[0]).planes
        if plane.name == "/device:TPU:0"
        for line in plane.lines
        if line.name == "XLA Modules"
        for event in line.events
        if event.name == f"jit_execute({program_id})"
    )
    if len(durations) != SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS or any(
        duration <= 0 for duration in durations
    ):
        raise ValueError(
            "SEQAX_SURFACE_PROFILE_MODULE_COUNT_MISMATCH "
            f"program={program_id} observed={len(durations)}"
        )
    return durations


def _profile_metrics(
    root: Path, result: SeqaxSurfaceProfileRunResult, assessment
) -> tuple[Metric, ...]:
    scenario = result.invocation.scenario
    mode = result.invocation.mode.value
    prefix = f"{scenario}_{mode}"
    phase_root = _profile_phase_root(root, result)
    xplane = next((phase_root / "profile").rglob("*.xplane.pb"))
    durations = _module_durations(phase_root, assessment)
    ordered = sorted(durations)
    p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
    source = MetricSource(
        artifact_sha256=_sha256(xplane),
        artifact_path=xplane.relative_to(root).as_posix(),
        tool="XPlane",
        field=f"/device:TPU:0/XLA Modules/jit_execute({_bound_program_id(assessment)})",
    )
    interval = MeasurementInterval(scope=f"one compiled resident-input {scenario} forward")
    metrics = [
        Metric(
            name=f"{prefix}_captured_forward_count",
            quantity=Quantity(
                value=Decimal(SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS),
                unit=Unit.COUNT,
            ),
            kind=MeasurementKind.MEASURED,
            interval=MeasurementInterval(
                scope=(
                    f"the complete {SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS}-forward "
                    f"{scenario} {mode} capture"
                )
            ),
            sources=(
                source.model_copy(
                    update={"field": f"StepTraceAnnotation(seqax_surface_{scenario})"}
                ),
            ),
        ),
        Metric(
            name=f"{prefix}_median_compiled_forward_duration",
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
            name=f"{prefix}_p90_compiled_forward_duration",
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
    ]
    for metric in capture_metrics(assessment.capture):
        metrics.append(
            metric.model_copy(
                update={
                    "name": f"{prefix}_{metric.name}",
                    "sources": tuple(
                        source_item.model_copy(
                            update={
                                "artifact_path": Path(source_item.artifact_path)
                                .resolve()
                                .relative_to(root)
                                .as_posix()
                            }
                        )
                        for source_item in metric.sources
                    ),
                }
            )
        )
    return tuple(metrics)


def _validate_profile_topology(assessment, *, counters: bool) -> None:
    planes = {
        plane.name
        for plane in assessment.capture.planes
        if plane.name.startswith("/device:TPU:") and "SparseCore" not in plane.name
    }
    expected_planes = {f"/device:TPU:{index}" for index in range(8)}
    if planes != expected_planes:
        raise ValueError("SEQAX_SURFACE_PROFILE_TPU_PLANE_SET_MISMATCH")
    counter_cores = set(assessment.capture.counters.periodic_samples_per_tpu_core)
    expected_counter_cores = {"0", "2", "4", "6"} if counters else set()
    if counter_cores != expected_counter_cores:
        raise ValueError("SEQAX_SURFACE_PROFILE_COUNTER_CORE_SET_MISMATCH")


def _copy_surface(surface_root: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError("SEQAX_SURFACE_PROFILE_SURFACE_ALREADY_PRESENT")
    shutil.copytree(surface_root, destination, copy_function=shutil.copy2)


def _all_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def _validate_xprof_exports(output_root: Path) -> None:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("SEQAX_SURFACE_PROFILE_XPROF_MANIFEST_MISSING")
    manifest = XProfExportManifest.model_validate_json(manifest_path.read_text())
    expected = {"manifest.json", *(export.output.name for export in manifest.exports)}
    observed = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise ValueError("SEQAX_SURFACE_PROFILE_XPROF_EXPORT_SET_MISMATCH")
    for export in manifest.exports:
        path = output_root / export.output.name
        if path.stat().st_size != export.size_bytes:
            raise ValueError("SEQAX_SURFACE_PROFILE_XPROF_EXPORT_SIZE_MISMATCH")
    if not any(export.tool == "hlo_stats" for export in manifest.exports):
        raise ValueError("SEQAX_SURFACE_PROFILE_XPROF_HLO_STATS_MISSING")


def _ensure_atomic_xprof_exports(root: Path, phase_root: Path) -> None:
    output_root = phase_root / "xprof"
    if output_root.exists():
        try:
            _validate_xprof_exports(output_root)
            return
        except (ValueError, OSError):
            archive = root.parent / (
                f"{root.name}-{phase_root.parent.name}-{phase_root.name}-"
                f"xprof-incomplete-{time.time_ns()}"
            )
            output_root.rename(archive)
            print(f"SEQAX_SURFACE_PROFILE_ARCHIVED_XPROF source={output_root} archive={archive}")
    temporary = phase_root / f"xprof.tmp-{time.time_ns()}"
    try:
        manifest = export_xprof_capture(phase_root / "profile", temporary)
        portable = manifest.model_copy(
            update={
                "exports": tuple(
                    export.model_copy(update={"output": output_root / export.output.name})
                    for export in manifest.exports
                )
            }
        )
        (temporary / "manifest.json").write_text(
            json.dumps(portable.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _validate_xprof_exports(temporary)
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            archive = root.parent / (
                f"{root.name}-{phase_root.parent.name}-{phase_root.name}-"
                f"xprof-failed-{time.time_ns()}"
            )
            temporary.rename(archive)
        raise


def _role_for_bundle_path(
    path: str, results: tuple[SeqaxSurfaceProfileRunResult, ...]
) -> ArtifactRole:
    if path.startswith("surface/"):
        surface_path = path.removeprefix("surface/")
        if surface_path == "receipt.json":
            return ArtifactRole.SEARCH_RESULT
        surface = seqax_forward_workload_surface()
        return _surface_artifact_roles(surface)[surface_path]
    if path in {"finalizer/source_state.json", "finalizer/source_diff.patch"}:
        return (
            ArtifactRole.SOURCE_STATE
            if path.endswith("source_state.json")
            else ArtifactRole.SOURCE_DIFF
        )
    if path == "profile_assessment.json":
        return ArtifactRole.PROFILE_ASSESSMENT
    parts = Path(path).parts
    if len(parts) < 3:
        raise ValueError(f"SEQAX_SURFACE_PROFILE_BUNDLE_PATH_UNKNOWN path={path}")
    scenario, mode, relative = parts[0], parts[1], Path(*parts[2:]).as_posix()
    result = next(
        (
            candidate
            for candidate in results
            if candidate.invocation.scenario == scenario and candidate.invocation.mode.value == mode
        ),
        None,
    )
    if result is None:
        raise ValueError(f"SEQAX_SURFACE_PROFILE_RESULT_MISSING path={path}")
    if relative == "result.json":
        return ArtifactRole.TRACE_RESULT if mode == "trace" else ArtifactRole.COUNTER_RESULT
    result_roles = {artifact.path: artifact.role for artifact in result.artifacts}
    if relative in result_roles:
        return result_roles[relative]
    if relative.endswith(("-shm", "-wal")):
        raise ValueError(f"SEQAX_SURFACE_PROFILE_LEDGER_SIDECAR_PRESENT path={path}")
    if relative.endswith(".xplane.pb"):
        return ArtifactRole.TIMING_TRACE if mode == "trace" else ArtifactRole.COUNTER_TRACE
    if relative.endswith("hlo_stats.json"):
        return ArtifactRole.HLO_STATS
    if relative.startswith(("profile/", "xprof/")):
        return ArtifactRole.XPROF_EXPORT
    raise ValueError(f"SEQAX_SURFACE_PROFILE_PHASE_PATH_UNKNOWN path={path}")


def _bundle_artifacts(
    root: Path,
    results: tuple[SeqaxSurfaceProfileRunResult, ...],
) -> tuple[ArtifactReference, ...]:
    return tuple(
        _artifact(root, path, _role_for_bundle_path(path.relative_to(root).as_posix(), results))
        for path in _all_files(root)
        if path.resolve() != (root / "receipt.json").resolve()
    )


def build_seqax_surface_profile_receipt(
    root: Path,
    *,
    surface_root: Path,
    write_receipt: bool = True,
) -> SeqaxSurfaceProfileReceipt:
    root = root.resolve()
    surface_root = surface_root.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == path or root in path.parents for path in protected):
        raise ValueError(f"SEQAX_SURFACE_PROFILE_UNSAFE_ROOT path={root}")
    if root == surface_root or root in surface_root.parents or surface_root in root.parents:
        raise ValueError("SEQAX_SURFACE_PROFILE_ROOTS_MUST_BE_DISJOINT")
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        receipt = SeqaxSurfaceProfileReceipt.model_validate_json(receipt_path.read_text())
        validate_seqax_surface_profile_receipt(receipt, root=root)
        return receipt
    results = _owned_bundle_results(root)
    _require_clean_repository()
    source_surface_receipt = _surface_receipt(surface_root)
    copied_surface = root / "surface"
    if not copied_surface.exists():
        _copy_surface(surface_root, copied_surface)
    copied_surface_receipt = _surface_receipt(copied_surface)
    if copied_surface_receipt != source_surface_receipt:
        raise ValueError("SEQAX_SURFACE_PROFILE_COPIED_SURFACE_MISMATCH")
    finalizer = root / "finalizer"
    _source_state(Path(__file__).resolve().parents[2], finalizer)
    source_identities = {
        _source_identity(
            root / result.invocation.scenario / result.invocation.mode.value / "source_state.json",
            root / result.invocation.scenario / result.invocation.mode.value / "source_diff.patch",
            require_clean=True,
        )
        for result in results
    }
    source_identities.add(
        _source_identity(
            finalizer / "source_state.json",
            finalizer / "source_diff.patch",
            require_clean=True,
        )
    )
    if len(source_identities) != 1:
        raise ValueError("SEQAX_SURFACE_PROFILE_SOURCE_IDENTITY_MISMATCH")
    assessments: dict[str, dict[str, Any]] = {}
    metrics: list[Metric] = []
    for result in results:
        phase_root = _profile_phase_root(root, result)
        _ensure_atomic_xprof_exports(root, phase_root)
        assessment = assess_capture(
            phase_root,
            _expectation(result.invocation.scenario, result.invocation.mode),
        )
        if not assessment.accepted:
            raise ValueError(
                "SEQAX_SURFACE_PROFILE_CAPTURE_REJECTED "
                f"scenario={result.invocation.scenario} mode={result.invocation.mode.value}"
            )
        _validate_profile_topology(
            assessment,
            counters=result.invocation.mode is RunMode.COUNTERS,
        )
        event_name = f"seqax_surface_{result.invocation.scenario}"
        observed_steps = count_profile_events(phase_root / "profile", event_name)
        if observed_steps != SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS:
            raise ValueError(
                "SEQAX_SURFACE_PROFILE_STEP_COUNT_MISMATCH "
                f"scenario={result.invocation.scenario} mode={result.invocation.mode.value} "
                f"observed={observed_steps}"
            )
        _module_durations(phase_root, assessment)
        key = f"{result.invocation.scenario}/{result.invocation.mode.value}"
        assessments[key] = _canonical_assessment(
            _relative_json(assessment.model_dump(mode="json"), root)
        )
        metrics.extend(_profile_metrics(root, result, assessment))
    assessment_path = root / "profile_assessment.json"
    assessment_path.write_text(json.dumps(assessments, indent=2, sort_keys=True) + "\n")
    artifacts = _bundle_artifacts(root, results)
    surface_receipt_sha256 = _sha256(copied_surface / "receipt.json")
    receipt = SeqaxSurfaceProfileReceipt(
        schema_version=SEQAX_SURFACE_PROFILE_SCHEMA,
        surface_id=copied_surface_receipt.surface_id,
        surface_receipt_sha256=surface_receipt_sha256,
        results=results,
        metrics=tuple(metrics),
        artifacts=artifacts,
        accepted=True,
    )
    validate_seqax_surface_profile_receipt(receipt, root=root)
    if write_receipt:
        temporary = root / "receipt.json.tmp"
        temporary.write_text(receipt.model_dump_json(indent=2) + "\n")
        temporary.replace(root / "receipt.json")
    return receipt


def _validate_result(
    root: Path,
    receipt: SeqaxSurfaceProfileReceipt,
    result: SeqaxSurfaceProfileRunResult,
) -> None:
    phase_root = _profile_phase_root(root, result)
    invocation = result.invocation
    scenario = _scenario(invocation.scenario)
    surface = seqax_forward_workload_surface()
    module = seqax_forward_schedule(**scenario.parameters())
    plan = lower_distributed_program_to_jax_mesh(module)
    saved_invocation = SeqaxSurfaceProfileInvocation.model_validate_json(
        (phase_root / "invocation.json").read_text()
    )
    if saved_invocation != invocation:
        raise ValueError("SEQAX_SURFACE_PROFILE_INVOCATION_REPLAY_MISMATCH")
    profiler_contract = json.loads((phase_root / "profiler_config.json").read_text())
    copied_surface_receipt = _surface_receipt(root / "surface")
    if (
        profiler_contract != expected_seqax_profiler_contract(invocation.mode)
        or invocation.profiler_config_sha256 != _json_sha256(profiler_contract)
        or invocation.runtime.xla != profiler_contract["libtpu_init_args"]
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_PROFILER_CONTRACT_MISMATCH")
    if (
        invocation.surface_id != surface.surface_id
        or invocation.surface_receipt_sha256 != receipt.surface_receipt_sha256
        or invocation.seed != semantic_seed(surface.surface_id, scenario.name, "inputs")
        or invocation.schedule_sha256 != plan.schedule_sha256
        or invocation.jax_source_sha256 != plan.source_sha256()
        or invocation.execution_scope != plan.execution_scope
        or invocation.runtime != copied_surface_receipt.invocation.runtime
        or invocation.device_kind != "TPU7x"
        or invocation.device_kind != copied_surface_receipt.invocation.device_kind
        or invocation.device_count != copied_surface_receipt.invocation.device_count
        or result.backend != "tpu"
        or result.mesh != plan.mesh_axes
        or result.plan_manifest_sha256 != _sha256(phase_root / "plan_manifest.json")
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_EXECUTION_IDENTITY_MISMATCH")
    expected_artifacts = {artifact.path: artifact for artifact in result.artifacts}
    if len(expected_artifacts) != len(result.artifacts):
        raise ValueError("SEQAX_SURFACE_PROFILE_RESULT_ARTIFACT_DUPLICATE")
    for path, artifact in expected_artifacts.items():
        if artifact.role is not _expected_result_role(path):
            raise ValueError(f"SEQAX_SURFACE_PROFILE_RESULT_ROLE_MISMATCH path={path}")
        resolve_recorded_artifact(
            phase_root,
            path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
    if (
        (phase_root / "distributed.xdsl").read_text() != canonical_text(module)
        or (phase_root / "lowered_jax.py").read_text() != plan.render_executable_source()
        or json.loads((phase_root / "plan_manifest.json").read_text()) != plan.manifest()
        or _sha256(phase_root / "stablehlo.txt") != invocation.stablehlo_sha256
        or _sha256(phase_root / "compiler_hlo.txt") != invocation.compiler_hlo_sha256
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_LOWERING_REPLAY_MISMATCH")
    surface_root = root / "surface"
    if (
        _sha256(surface_root / "hlo" / scenario.name / "candidate_stablehlo.txt")
        != invocation.stablehlo_sha256
        or _sha256(surface_root / "hlo" / scenario.name / "candidate_compiler_hlo.txt")
        != invocation.compiler_hlo_sha256
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_SURFACE_HLO_MISMATCH")
    expected_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=invocation.seed, **scenario.parameters())
    )
    saved_inputs = tuple(
        np.load(phase_root / "inputs" / f"{index:02d}.npy", allow_pickle=False)
        for index in range(len(expected_inputs))
    )
    if any(
        saved.shape != expected.shape
        or saved.dtype != expected.dtype
        or not np.array_equal(saved, expected)
        for saved, expected in zip(saved_inputs, expected_inputs, strict=True)
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_INPUT_REPLAY_MISMATCH")
    actual = np.load(phase_root / "output.npy", allow_pickle=False)
    oracle = np.load(phase_root / "oracle.npy", allow_pickle=False)
    expected_oracle = np.asarray(
        seqax_forward_canonical_reference(
            expected_inputs,
            quantization_decimals=surface.oracle_quantization_decimals,
            **scenario.parameters(),
        )
    )
    surface_output = np.load(
        surface_root / "outputs" / scenario.name / "candidate.npy",
        allow_pickle=False,
    )
    if (
        oracle.shape != expected_oracle.shape
        or oracle.dtype != expected_oracle.dtype
        or not np.array_equal(oracle, expected_oracle)
        or actual.shape != surface_output.shape
        or actual.dtype != surface_output.dtype
        or not np.array_equal(actual, surface_output)
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_ARRAY_REPLAY_MISMATCH")
    input_sha256 = tuple(array_sha256(value) for value in expected_inputs)
    output_sha256 = array_sha256(actual)
    oracle_sha256 = array_sha256(oracle)
    execution_identity_sha256 = _execution_identity_sha256(
        surface_receipt_sha256=receipt.surface_receipt_sha256,
        scenario=scenario.name,
        schedule_sha256=plan.schedule_sha256,
        jax_source_sha256=plan.source_sha256(),
        stablehlo_sha256=invocation.stablehlo_sha256,
        compiler_hlo_sha256=invocation.compiler_hlo_sha256,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        oracle_sha256=oracle_sha256,
        runtime=invocation.runtime,
        device_kind=invocation.device_kind,
        device_count=invocation.device_count,
    )
    if (
        invocation.input_sha256 != input_sha256
        or invocation.output_sha256 != output_sha256
        or invocation.oracle_sha256 != oracle_sha256
        or invocation.execution_identity_sha256 != execution_identity_sha256
        or invocation.run_id
        != semantic_sha256(
            SEQAX_SURFACE_PROFILE_SCHEMA,
            execution_identity_sha256,
            invocation.mode.value,
            invocation.profiler_config_sha256,
        )
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_BOUND_IDENTITY_MISMATCH")
    maximum_absolute_error, maximum_relative_error = _errors(actual, expected_oracle)
    if (
        not result.passed
        or not np.allclose(
            actual,
            expected_oracle,
            atol=SEQAX_SURFACE_ATOL,
            rtol=SEQAX_SURFACE_RTOL,
        )
        or not math.isclose(
            result.maximum_absolute_error,
            maximum_absolute_error,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            result.maximum_relative_error,
            maximum_relative_error,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or result.input_sha256 != input_sha256
        or result.output_sha256 != output_sha256
        or result.oracle_sha256 != oracle_sha256
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_CORRECTNESS_REPLAY_MISMATCH")
    expected_payloads = (
        invocation.model_dump(mode="json"),
        {"schedule_sha256": plan.schedule_sha256},
        {
            "jax_source_sha256": plan.source_sha256(),
            "plan_manifest_sha256": result.plan_manifest_sha256,
            "execution_scope": plan.execution_scope,
        },
        {
            "stablehlo_sha256": invocation.stablehlo_sha256,
            "compiler_hlo_sha256": invocation.compiler_hlo_sha256,
            "compile_duration_ns": result.compile_duration_ns,
        },
        {
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
        },
        {
            "event_name": f"seqax_surface_{scenario.name}",
            "warmup_iterations": SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS,
            "measured_iterations": SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS,
            "profile_root": "profile",
        },
    )
    terminal = RunState.TRACED if invocation.mode is RunMode.TRACE else RunState.COUNTERED
    history = read_ledger_history(phase_root / "ledger.sqlite", invocation.run_id)
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
    ) != tuple(payload_sha256(payload) for payload in expected_payloads):
        raise ValueError("SEQAX_SURFACE_PROFILE_LEDGER_REPLAY_MISMATCH")


def validate_seqax_surface_profile_receipt(
    receipt: SeqaxSurfaceProfileReceipt,
    *,
    root: Path,
) -> None:
    root = root.resolve()
    surface = seqax_forward_workload_surface()
    copied_surface_receipt = _surface_receipt(root / "surface")
    if (
        receipt.schema_version != SEQAX_SURFACE_PROFILE_SCHEMA
        or receipt.surface_id != surface.surface_id
        or copied_surface_receipt.surface_id != receipt.surface_id
        or receipt.surface_receipt_sha256 != _sha256(root / "surface" / "receipt.json")
        or not receipt.accepted
    ):
        raise ValueError("SEQAX_SURFACE_PROFILE_RECEIPT_IDENTITY_MISMATCH")
    expected_keys = {
        (scenario.name, mode)
        for scenario in surface.scenarios
        for mode in SEQAX_SURFACE_PROFILE_MODES
    }
    observed_keys = {
        (result.invocation.scenario, result.invocation.mode) for result in receipt.results
    }
    if observed_keys != expected_keys or len(receipt.results) != len(expected_keys):
        raise ValueError("SEQAX_SURFACE_PROFILE_RESULT_SET_MISMATCH")
    by_key = {
        (result.invocation.scenario, result.invocation.mode): result for result in receipt.results
    }
    loaded_results = tuple(
        _load_result(root, scenario.name, mode)
        for scenario in surface.scenarios
        for mode in SEQAX_SURFACE_PROFILE_MODES
    )
    if tuple(receipt.results) != loaded_results:
        raise ValueError("SEQAX_SURFACE_PROFILE_RESULT_REPLAY_MISMATCH")
    for result in loaded_results:
        _validate_result(root, receipt, result)
    for scenario in surface.scenarios:
        trace = by_key[(scenario.name, RunMode.TRACE)]
        counters = by_key[(scenario.name, RunMode.COUNTERS)]
        trace_identity = _phase_identity(trace)
        counter_identity = _phase_identity(counters)
        if trace_identity != counter_identity:
            raise ValueError(
                f"SEQAX_SURFACE_PROFILE_CROSS_MODE_IDENTITY_MISMATCH scenario={scenario.name}"
            )
    source_identities = {
        _source_identity(
            _profile_phase_root(root, result) / "source_state.json",
            _profile_phase_root(root, result) / "source_diff.patch",
            require_clean=True,
        )
        for result in loaded_results
    }
    source_identities.add(
        _source_identity(
            root / "finalizer" / "source_state.json",
            root / "finalizer" / "source_diff.patch",
            require_clean=True,
        )
    )
    if len(source_identities) != 1:
        raise ValueError("SEQAX_SURFACE_PROFILE_SOURCE_REPLAY_MISMATCH")
    expected_assessments: dict[str, dict[str, Any]] = {}
    expected_metrics: list[Metric] = []
    for result in loaded_results:
        phase_root = _profile_phase_root(root, result)
        _validate_xprof_exports(phase_root / "xprof")
        assessment = assess_capture(
            phase_root,
            _expectation(result.invocation.scenario, result.invocation.mode),
        )
        if not assessment.accepted:
            raise ValueError("SEQAX_SURFACE_PROFILE_ASSESSMENT_REJECTED")
        _validate_profile_topology(
            assessment,
            counters=result.invocation.mode is RunMode.COUNTERS,
        )
        if (
            count_profile_events(
                phase_root / "profile",
                f"seqax_surface_{result.invocation.scenario}",
            )
            != SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS
        ):
            raise ValueError("SEQAX_SURFACE_PROFILE_EVENT_REPLAY_MISMATCH")
        _module_durations(phase_root, assessment)
        key = f"{result.invocation.scenario}/{result.invocation.mode.value}"
        expected_assessments[key] = _canonical_assessment(
            _relative_json(assessment.model_dump(mode="json"), root)
        )
        expected_metrics.extend(_profile_metrics(root, result, assessment))
    saved_assessments = json.loads((root / "profile_assessment.json").read_text())
    if saved_assessments != expected_assessments:
        raise ValueError("SEQAX_SURFACE_PROFILE_ASSESSMENT_REPLAY_MISMATCH")
    if receipt.metrics != tuple(expected_metrics):
        raise ValueError("SEQAX_SURFACE_PROFILE_METRIC_REPLAY_MISMATCH")
    expected_artifacts = _bundle_artifacts(root, loaded_results)
    if receipt.artifacts != expected_artifacts:
        raise ValueError("SEQAX_SURFACE_PROFILE_ARTIFACT_MANIFEST_MISMATCH")
    artifact_by_path = {artifact.path: artifact for artifact in receipt.artifacts}
    for artifact in receipt.artifacts:
        resolve_recorded_artifact(
            root,
            artifact.path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
    for metric in receipt.metrics:
        for source in metric.sources:
            artifact = artifact_by_path.get(source.artifact_path)
            if artifact is None or artifact.sha256 != source.artifact_sha256:
                raise ValueError(
                    f"SEQAX_SURFACE_PROFILE_METRIC_SOURCE_MISMATCH metric={metric.name}"
                )
    allowed = {artifact.path for artifact in receipt.artifacts} | {"receipt.json"}
    observed = {path.relative_to(root).as_posix() for path in _all_files(root)}
    if observed not in ({artifact.path for artifact in receipt.artifacts}, allowed):
        raise ValueError("SEQAX_SURFACE_PROFILE_CLOSED_WORLD_MISMATCH")
