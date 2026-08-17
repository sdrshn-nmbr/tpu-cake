from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    RuntimeIdentity,
    experiment_artifact_json,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, array_sha256, semantic_sha256
from tpu_cake.jax_lowering import (
    JAX_DISTRIBUTED_EXECUTION_SCHEMA,
    lower_distributed_program_to_jax_mesh,
)
from tpu_cake.ledger import RunState
from tpu_cake.metrics import MetricSource
from tpu_cake.runner import (
    RunMode,
    _percentile,
    _profiler_contract,
    _profiler_options,
    _record_event,
    _runtime_identity,
    _sha256,
    _source_state,
    _write_json,
    _write_text,
)
from tpu_cake.seqax_cost_model import estimate_seqax_forward
from tpu_cake.workloads.seqax_forward import (
    seqax_forward_experiment,
    seqax_forward_schedule,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)

SEQAX_EVIDENCE_SEED = 9173
SEQAX_EVIDENCE_WARMUP_ITERATIONS = 5
SEQAX_EVIDENCE_MEASURED_ITERATIONS = 50
SEQAX_EVIDENCE_PARAMETERS = {
    "batch": 2,
    "sequence": 4,
    "model": 8,
    "vocabulary": 16,
    "feed_forward": 16,
    "query_groups": 2,
    "key_value_heads": 4,
    "head": 4,
    "layers": 2,
    "data_mesh": 2,
    "tensor_mesh": 4,
    "rope_max_timescale": 256,
}
SEQAX_OUTPUT_ATOL = 0.006
SEQAX_OUTPUT_RTOL = 0.05
SEQAX_LIBTPU_INIT_ARGS = " --xla_tpu_use_enhanced_launch_barrier=true"


def expected_seqax_profiler_contract(mode: RunMode) -> dict[str, object]:
    advanced: dict[str, object] = {"tpu_num_chips_to_profile_per_task": 4}
    if mode is RunMode.COUNTERS:
        advanced.update(
            {
                "tpu_enable_periodic_counter_sampling": True,
                "tpu_tc_perf_counter_sampling_options": (
                    "interval_us:1 scaling:0 counter_size_bits:1 "
                    "indices:1 indices:3 indices:4 indices:10 indices:11 "
                    "indices:31 indices:32 indices:33 indices:34 indices:35 "
                    "indices:37 indices:38 indices:56 indices:57 indices:58 "
                    "indices:73 indices:74 indices:75 indices:105"
                ),
                "num_tensor_cores_to_trace_per_device": 1,
            }
        )
    return {
        "mode": mode.value,
        "raise_error_on_start_failure": True,
        "enable_hlo_proto": True,
        "host_tracer_level": 1,
        "python_tracer_level": 0,
        "advanced_configuration": advanced,
        "libtpu_init_args": SEQAX_LIBTPU_INIT_ARGS,
    }


class SeqaxForwardInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_schema: str
    execution_schema: str
    mode: RunMode
    seed: int
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    parameters: dict[str, int]
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    jax_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_scope: str

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxForwardInvocation:
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("SEQAX_IDENTITY_SCHEMA_MISMATCH")
        if self.execution_schema != JAX_DISTRIBUTED_EXECUTION_SCHEMA:
            raise ValueError("SEQAX_EXECUTION_SCHEMA_MISMATCH")
        if (
            self.seed != SEQAX_EVIDENCE_SEED
            or self.warmup_iterations != SEQAX_EVIDENCE_WARMUP_ITERATIONS
            or self.measured_iterations != SEQAX_EVIDENCE_MEASURED_ITERATIONS
            or self.parameters != SEQAX_EVIDENCE_PARAMETERS
        ):
            raise ValueError("SEQAX_EVIDENCE_PROTOCOL_MISMATCH")
        return self


class SeqaxForwardRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: RunMode
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    mesh: tuple[tuple[str, int], ...]
    execution_scope: str
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    jax_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: tuple[str, ...]
    output_sha256: tuple[str, ...]
    oracle_sha256: tuple[str, ...]
    passed: bool
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)
    compile_duration_ns: int = Field(ge=0)
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    samples_ns: tuple[int, ...]
    median_ns: float | None = Field(default=None, ge=0)
    p90_ns: int | None = Field(default=None, ge=0)
    coefficient_of_variation: float | None = Field(default=None, ge=0)
    runtime: RuntimeIdentity
    artifacts: tuple[ArtifactReference, ...]


def _artifact(root: Path, path: Path, role: ArtifactRole) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
    )


def _save_array(
    root: Path,
    path: Path,
    value: np.ndarray,
    role: ArtifactRole,
) -> ArtifactReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return _artifact(root, path, role)


def _compiler_hlo(lowered: Any) -> str:
    computation = lowered.compiler_ir(dialect="hlo")
    return computation.as_hlo_text() if hasattr(computation, "as_hlo_text") else str(computation)


def _errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    observed = actual.astype(np.float32)
    reference = expected.astype(np.float32)
    absolute = np.abs(observed - reference)
    denominator = np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
    return float(absolute.max()), float((absolute / denominator).max())


def run_seqax_forward(
    output_dir: Path,
    *,
    mode: RunMode,
) -> SeqaxForwardRunResult:
    if jax.default_backend() != "tpu":
        raise ValueError("Seqax forward device evidence requires a TPU backend")
    devices = tuple(jax.devices())
    output_dir.mkdir(parents=True, exist_ok=False)
    module = seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    plan = lower_distributed_program_to_jax_mesh(module)
    if len(devices) != plan.device_count:
        raise ValueError(
            f"SEQAX_DEVICE_COUNT_MISMATCH expected={plan.device_count} observed={len(devices)}"
        )
    source = plan.render_executable_source()
    experiment = seqax_forward_experiment(
        plan,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        absolute_tolerance=SEQAX_OUTPUT_ATOL,
        relative_tolerance=SEQAX_OUTPUT_RTOL,
    )
    profiler_contract = _profiler_contract(mode)
    if profiler_contract != expected_seqax_profiler_contract(mode):
        raise ValueError("SEQAX_PROFILER_CONTRACT_MISMATCH")
    invocation = SeqaxForwardInvocation(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        execution_schema=JAX_DISTRIBUTED_EXECUTION_SCHEMA,
        mode=mode,
        seed=SEQAX_EVIDENCE_SEED,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        parameters=SEQAX_EVIDENCE_PARAMETERS,
        schedule_sha256=plan.schedule_sha256,
        jax_source_sha256=plan.source_sha256(),
        execution_scope=plan.execution_scope,
    )
    run_id = semantic_sha256(
        "seqax-distributed-forward-run-v1",
        mode.value,
        plan.schedule_sha256,
        plan.source_sha256(),
        str(SEQAX_EVIDENCE_SEED),
    )
    artifacts = [
        _write_text(
            output_dir / "experiment.json",
            experiment_artifact_json(experiment) + "\n",
            ArtifactRole.EXPERIMENT,
        ),
        _write_json(
            output_dir / "invocation.json",
            invocation.model_dump(mode="json"),
            ArtifactRole.INVOCATION,
        ),
        _write_json(
            output_dir / "profiler_config.json",
            profiler_contract,
            ArtifactRole.PROFILER_CONFIG,
        ),
        *_source_state(Path(__file__).resolve().parents[2], output_dir),
    ]
    ledger_path = output_dir / "ledger.sqlite"
    _record_event(
        ledger_path,
        run_id,
        RunState.CREATED,
        invocation.model_dump(mode="json"),
    )
    module.verify()
    distributed_text = canonical_text(module)
    distributed_artifact = _write_text(
        output_dir / "distributed.xdsl",
        distributed_text,
        ArtifactRole.DISTRIBUTED_IR,
    )
    artifacts.append(distributed_artifact)
    _record_event(
        ledger_path,
        run_id,
        RunState.VERIFIED,
        {"schedule_sha256": plan.schedule_sha256},
    )
    source_artifact = _write_text(
        output_dir / "lowered_jax.py",
        source,
        ArtifactRole.JAX_SOURCE,
    )
    manifest_artifact = _write_json(
        output_dir / "plan_manifest.json",
        plan.manifest(),
        ArtifactRole.PLAN_MANIFEST,
    )
    artifacts.extend((source_artifact, manifest_artifact))
    cost_report = estimate_seqax_forward(
        module,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=distributed_artifact.sha256,
            artifact_path=distributed_artifact.path,
            tool="tpu-cake",
            field="seqax-distributed-forward-v1",
        ),
        expected_schedule_sha256=plan.schedule_sha256,
    )
    artifacts.append(
        _write_text(
            output_dir / "cost_model.json",
            cost_report.model_dump_json(indent=2) + "\n",
            ArtifactRole.COST_MODEL,
        )
    )
    _record_event(
        ledger_path,
        run_id,
        RunState.LOWERED,
        {
            "schedule_sha256": plan.schedule_sha256,
            "jax_source_sha256": source_artifact.sha256,
            "plan_manifest_sha256": manifest_artifact.sha256,
            "execution_scope": plan.execution_scope,
        },
    )

    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=SEQAX_EVIDENCE_SEED,
            **SEQAX_EVIDENCE_PARAMETERS,
        )
    )
    oracle = np.asarray(
        seqax_forward_canonical_reference(host_inputs, **SEQAX_EVIDENCE_PARAMETERS)
    )
    executable, mesh = plan.build(devices=devices)
    compile_inputs = tuple(jnp.asarray(value) for value in host_inputs)
    compile_started = time.perf_counter_ns()
    lowered = executable.lower(*compile_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiler_hlo = _compiler_hlo(lowered)
    compiled = lowered.compile()
    compile_duration_ns = time.perf_counter_ns() - compile_started
    stablehlo_artifact = _write_text(
        output_dir / "stablehlo.txt",
        stablehlo + "\n",
        ArtifactRole.STABLEHLO,
    )
    compiler_hlo_artifact = _write_text(
        output_dir / "compiler_hlo.txt",
        compiler_hlo + "\n",
        ArtifactRole.COMPILER_HLO,
    )
    artifacts.extend((stablehlo_artifact, compiler_hlo_artifact))
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

    (actual_device,) = compiled(*compile_inputs)
    actual_device.block_until_ready()
    actual = np.asarray(actual_device)
    maximum_absolute_error, maximum_relative_error = _errors(actual, oracle)
    passed = np.allclose(
        actual,
        oracle,
        atol=SEQAX_OUTPUT_ATOL,
        rtol=SEQAX_OUTPUT_RTOL,
    )
    if not passed:
        _record_event(
            ledger_path,
            run_id,
            RunState.REJECTED,
            {
                "maximum_absolute_error": maximum_absolute_error,
                "maximum_relative_error": maximum_relative_error,
            },
        )
        raise ValueError(
            "SEQAX_FORWARD_CORRECTNESS_FAILED "
            f"absolute={maximum_absolute_error} relative={maximum_relative_error}"
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

    for _ in range(SEQAX_EVIDENCE_WARMUP_ITERATIONS):
        jax.block_until_ready(compiled(*compile_inputs))
    samples: list[int] = []
    if mode is RunMode.TIMING:
        for _ in range(SEQAX_EVIDENCE_MEASURED_ITERATIONS):
            started = time.perf_counter_ns()
            jax.block_until_ready(compiled(*compile_inputs))
            samples.append(time.perf_counter_ns() - started)
    else:
        profile_root = output_dir / "profile"
        jax.profiler.start_trace(
            profile_root,
            profiler_options=_profiler_options(mode),
        )
        try:
            for step in range(SEQAX_EVIDENCE_MEASURED_ITERATIONS):
                with jax.profiler.StepTraceAnnotation("seqax_forward", step_num=step):
                    jax.block_until_ready(compiled(*compile_inputs))
        finally:
            jax.profiler.stop_trace()

    input_artifacts = tuple(
        _save_array(
            output_dir,
            output_dir / "inputs" / f"{index:02d}.npy",
            value,
            ArtifactRole.CORRECTNESS_INPUT,
        )
        for index, value in enumerate(host_inputs)
    )
    output_artifact = _save_array(
        output_dir,
        output_dir / "outputs" / "00.npy",
        actual,
        ArtifactRole.CORRECTNESS_OUTPUT,
    )
    oracle_artifact = _save_array(
        output_dir,
        output_dir / "oracle" / "00.npy",
        oracle,
        ArtifactRole.ORACLE_OUTPUT,
    )
    artifacts.extend((*input_artifacts, output_artifact, oracle_artifact))
    median_ns = statistics.median(samples) if samples else None
    p90_ns = _percentile(samples, 0.9) if samples else None
    coefficient = (
        statistics.pstdev(samples) / statistics.mean(samples)
        if len(samples) > 1 and statistics.mean(samples)
        else None
    )
    terminal = {
        RunMode.TIMING: RunState.TIMED,
        RunMode.TRACE: RunState.TRACED,
        RunMode.COUNTERS: RunState.COUNTERED,
    }[mode]
    terminal_payload: dict[str, Any] = {
        "warmup_iterations": SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        "measured_iterations": SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        "mesh": dict(mesh.shape),
    }
    if mode is RunMode.TIMING:
        terminal_payload.update(
            median_ns=median_ns,
            p90_ns=p90_ns,
            coefficient_of_variation=coefficient,
        )
    else:
        terminal_payload["profile_root"] = "profile"
    _record_event(ledger_path, run_id, terminal, terminal_payload)
    ledger_artifact = _artifact(output_dir, ledger_path, ArtifactRole.EXECUTION_LEDGER)
    artifacts.append(ledger_artifact)
    result = SeqaxForwardRunResult(
        run_id=run_id,
        mode=mode,
        backend=jax.default_backend(),
        device_kind=devices[0].device_kind,
        device_count=len(devices),
        mesh=tuple(plan.mesh_axes),
        execution_scope=plan.execution_scope,
        schedule_sha256=plan.schedule_sha256,
        jax_source_sha256=source_artifact.sha256,
        plan_manifest_sha256=manifest_artifact.sha256,
        stablehlo_sha256=stablehlo_artifact.sha256,
        compiler_hlo_sha256=compiler_hlo_artifact.sha256,
        input_sha256=tuple(array_sha256(value) for value in host_inputs),
        output_sha256=(array_sha256(actual),),
        oracle_sha256=(array_sha256(oracle),),
        passed=True,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        compile_duration_ns=compile_duration_ns,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        samples_ns=tuple(samples),
        median_ns=median_ns,
        p90_ns=p90_ns,
        coefficient_of_variation=coefficient,
        runtime=_runtime_identity(),
        artifacts=tuple(artifacts),
    )
    _write_text(
        output_dir / "result.json",
        result.model_dump_json(indent=2) + "\n",
        {
            RunMode.TIMING: ArtifactRole.TIMING_SAMPLES,
            RunMode.TRACE: ArtifactRole.TRACE_RESULT,
            RunMode.COUNTERS: ArtifactRole.COUNTER_RESULT,
        }[mode],
    )
    return result
