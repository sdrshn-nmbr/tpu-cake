from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tpu_cake.artifacts import artifact_reference as _artifact
from tpu_cake.artifacts import save_array_reference as _save_array
from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    RuntimeIdentity,
    SourceFileContract,
    experiment_artifact_json,
)
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, semantic_sha256
from tpu_cake.ledger import RunState
from tpu_cake.rpa_lowering import lower_inkling_rpa_to_pallas
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
from tpu_cake.workloads.inkling_rpa import (
    inkling_fused_rpa_experiment,
    inkling_fused_rpa_inputs,
    inkling_fused_rpa_reference,
    inkling_fused_rpa_schedule,
)


class FusedRpaRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: RunMode
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    execution_scope: str
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_manifest: tuple[SourceFileContract, ...]
    backend_executor: str
    backend_executor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preflight_passed: bool
    input_sha256: tuple[str, ...]
    output_sha256: tuple[str, ...]
    oracle_sha256: tuple[str, ...]
    passed: bool
    maximum_absolute_errors: tuple[float, float]
    maximum_relative_errors: tuple[float, float]
    compile_duration_ns: int = Field(ge=0)
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    samples_ns: tuple[int, ...]
    median_ns: int | None = Field(default=None, ge=0)
    p90_ns: int | None = Field(default=None, ge=0)
    coefficient_of_variation: float | None = Field(default=None, ge=0)
    runtime: RuntimeIdentity
    artifacts: tuple[ArtifactReference, ...]


def validate_fused_rpa_run_protocol(
    *,
    seed: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> None:
    protocol = inkling_fused_rpa_experiment().benchmark
    if (
        seed != 97
        or warmup_iterations != protocol.warmup_iterations
        or measured_iterations != protocol.measured_iterations
    ):
        raise ValueError(
            "fused RPA run must use its predeclared benchmark protocol: "
            "seed=97, "
            f"warmup={protocol.warmup_iterations}, measured={protocol.measured_iterations}"
        )


def _block_results(results: tuple[jax.Array, jax.Array]) -> None:
    jax.block_until_ready(results)


def _errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    actual_f32 = actual.astype(np.float32)
    expected_f32 = expected.astype(np.float32)
    absolute = np.abs(actual_f32 - expected_f32)
    denominator = np.maximum(np.abs(expected_f32), np.finfo(np.float32).tiny)
    return float(absolute.max()), float((absolute / denominator).max())


def fused_rpa_outputs_pass(
    actual: tuple[np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray],
    *,
    output_atol: float,
    output_rtol: float,
) -> bool:
    return np.allclose(
        actual[0], expected[0], atol=output_atol, rtol=output_rtol
    ) and np.array_equal(actual[1], expected[1])


def run_fused_rpa(
    output_dir: Path,
    *,
    mode: RunMode,
    kernel: Callable[..., tuple[jax.Array, jax.Array]],
    backend_manifest: tuple[tuple[str, str], ...],
    seed: int,
    warmup_iterations: int,
    measured_iterations: int,
    decode_block_sizes: tuple[int, int, int, int] = (8, 128, 8, 128),
) -> FusedRpaRunResult:
    if jax.default_backend() != "tpu":
        raise ValueError("fused Inkling RPA device evidence requires a TPU backend")
    validate_fused_rpa_run_protocol(
        seed=seed,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    schedule = inkling_fused_rpa_schedule(decode_block_sizes)
    plan = lower_inkling_rpa_to_pallas(schedule)
    experiment = inkling_fused_rpa_experiment(decode_block_sizes)
    if experiment.schedule_sha256 != plan.schedule_sha256:
        raise ValueError("RPA experiment schedule does not match the executable plan")
    if backend_manifest != plan.backend_manifest:
        raise ValueError("RPA runtime source manifest does not match the executable plan")
    backend_executor, backend_executor_sha256 = plan.validate_backend_callable(kernel)
    device_kind = jax.devices()[0].device_kind
    run_id = semantic_sha256(
        "inkling-fused-rpa-run",
        mode.value,
        str(seed),
        str(warmup_iterations),
        str(measured_iterations),
        plan.schedule_sha256,
        plan.source_sha256(),
        backend_executor,
        backend_executor_sha256,
    )
    invocation = {
        "identity_schema": SEMANTIC_IDENTITY_SCHEMA,
        "mode": mode.value,
        "seed": seed,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "execution_scope": plan.execution_scope,
        "schedule_sha256": plan.schedule_sha256,
        "pallas_source_sha256": plan.source_sha256(),
        "backend_executor": backend_executor,
        "backend_executor_sha256": backend_executor_sha256,
    }
    artifacts = [
        _write_json(output_dir / "invocation.json", invocation, ArtifactRole.INVOCATION),
        _write_json(
            output_dir / "profiler_config.json",
            _profiler_contract(mode),
            ArtifactRole.PROFILER_CONFIG,
        ),
        _write_json(
            output_dir / "backend_manifest.json",
            {"source_revision": plan.backend_repository_revision, "files": backend_manifest},
            ArtifactRole.BACKEND_MANIFEST,
        ),
        *_source_state(Path(__file__).resolve().parents[2], output_dir),
    ]
    ledger_path = output_dir / "ledger.sqlite"
    _record_event(
        ledger_path,
        run_id,
        RunState.CREATED,
        invocation,
    )
    schedule.verify()
    physical_text = canonical_text(schedule)
    _record_event(
        ledger_path,
        run_id,
        RunState.VERIFIED,
        {"physical_ir_sha256": hashlib.sha256(physical_text.encode()).hexdigest()},
    )
    artifacts.extend(
        (
            _write_text(
                output_dir / "experiment.json",
                experiment_artifact_json(experiment) + "\n",
                ArtifactRole.EXPERIMENT,
            ),
            _write_text(
                output_dir / "physical.xdsl",
                physical_text,
                ArtifactRole.PHYSICAL_IR,
            ),
            _write_text(
                output_dir / "lowered_pallas.py",
                plan.render_executable_source(),
                ArtifactRole.PALLAS_SOURCE,
            ),
        )
    )
    _record_event(
        ledger_path,
        run_id,
        RunState.LOWERED,
        {
            "schedule_sha256": plan.schedule_sha256,
            "pallas_source_sha256": plan.source_sha256(),
            "backend_manifest": list(backend_manifest),
            "execution_scope": plan.execution_scope,
        },
    )

    generated_inputs = inkling_fused_rpa_inputs(seed)
    host_inputs = tuple(np.asarray(value) for value in generated_inputs)
    plan.preflight(*generated_inputs)
    artifacts.append(
        _write_json(
            output_dir / "preflight.json",
            {
                "passed": True,
                "validator": "tpu_cake.rpa_lowering.FusedRpaPlan.preflight",
                "input_shapes": [list(value.shape) for value in host_inputs],
                "input_dtypes": [str(value.dtype) for value in host_inputs],
            },
            ArtifactRole.PREFLIGHT_RESULT,
        )
    )
    oracle_device = inkling_fused_rpa_reference(generated_inputs)
    oracle_host = tuple(np.asarray(value) for value in oracle_device)

    def execute(*inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return plan.invoke(
            kernel,
            *inputs,
            backend_manifest=backend_manifest,
            device_kind=device_kind,
        )

    executable = jax.jit(execute)
    compile_inputs = tuple(jnp.asarray(value) for value in host_inputs)
    compile_started = time.perf_counter_ns()
    lowered = executable.lower(*compile_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    hlo_computation = lowered.compiler_ir(dialect="hlo")
    compiler_hlo = (
        hlo_computation.as_hlo_text()
        if hasattr(hlo_computation, "as_hlo_text")
        else str(hlo_computation)
    )
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

    correctness_inputs = tuple(jnp.asarray(value) for value in host_inputs)
    actual_device = compiled(*correctness_inputs)
    _block_results(actual_device)
    actual_host = tuple(np.asarray(value) for value in actual_device)
    maximum_errors = tuple(
        _errors(actual, expected) for actual, expected in zip(actual_host, oracle_host, strict=True)
    )
    passed = fused_rpa_outputs_pass(
        actual_host,
        oracle_host,
        output_atol=experiment.workload.numerical.absolute_tolerance,
        output_rtol=experiment.workload.numerical.relative_tolerance,
    )
    if not passed:
        _record_event(
            ledger_path,
            run_id,
            RunState.REJECTED,
            {"reason": "numerical mismatch", "maximum_errors": maximum_errors},
        )
        raise ValueError(f"FUSED_RPA_CORRECTNESS_FAILED errors={maximum_errors}")
    _record_event(
        ledger_path,
        run_id,
        RunState.CORRECT,
        {
            "output_shapes": [list(value.shape) for value in actual_host],
            "maximum_errors": maximum_errors,
        },
    )

    def fresh_device_inputs() -> tuple[jax.Array, ...]:
        return tuple(jax.device_put(np.array(value, copy=True)) for value in host_inputs)

    warmup_inputs = tuple(fresh_device_inputs() for _ in range(warmup_iterations))
    measured_inputs = tuple(fresh_device_inputs() for _ in range(measured_iterations))
    jax.block_until_ready((*warmup_inputs, *measured_inputs))
    for inputs in warmup_inputs:
        outputs = compiled(*inputs)
        _block_results(outputs)
    samples: list[int] = []
    if mode is RunMode.TIMING:
        for inputs in measured_inputs:
            started = time.perf_counter_ns()
            outputs = compiled(*inputs)
            _block_results(outputs)
            samples.append(time.perf_counter_ns() - started)
    else:
        trace_dir = output_dir / "profile"
        jax.profiler.start_trace(trace_dir, profiler_options=_profiler_options(mode))
        try:
            for step, inputs in enumerate(measured_inputs):
                with jax.profiler.StepTraceAnnotation("inkling_fused_rpa", step_num=step):
                    outputs = compiled(*inputs)
                    _block_results(outputs)
        finally:
            jax.profiler.stop_trace()

    input_names = tuple(tensor.name for tensor in experiment.workload.inputs)
    output_names = tuple(tensor.name for tensor in experiment.workload.outputs)
    input_artifacts = tuple(
        _save_array(
            output_dir,
            output_dir / "inputs" / f"{index:02d}-{name}.npy",
            value,
            ArtifactRole.CORRECTNESS_INPUT,
        )
        for index, (name, value) in enumerate(zip(input_names, host_inputs, strict=True))
    )
    output_artifacts = tuple(
        _save_array(
            output_dir,
            output_dir / "outputs" / f"{index:02d}-{name}.npy",
            value,
            ArtifactRole.CORRECTNESS_OUTPUT,
        )
        for index, (name, value) in enumerate(zip(output_names, actual_host, strict=True))
    )
    oracle_artifacts = tuple(
        _save_array(
            output_dir,
            output_dir / "oracle" / f"{index:02d}-{name}.npy",
            value,
            ArtifactRole.ORACLE_OUTPUT,
        )
        for index, (name, value) in enumerate(zip(output_names, oracle_host, strict=True))
    )
    artifacts.extend((*input_artifacts, *output_artifacts, *oracle_artifacts))
    median_ns = int(statistics.median(samples)) if samples else None
    p90_ns = _percentile(samples, 0.9) if samples else None
    coefficient = (
        statistics.pstdev(samples) / statistics.mean(samples)
        if len(samples) > 1 and statistics.mean(samples)
        else None
    )
    terminal_state = {
        RunMode.TIMING: RunState.TIMED,
        RunMode.TRACE: RunState.TRACED,
        RunMode.COUNTERS: RunState.COUNTERED,
    }[mode]
    terminal_payload: dict[str, Any] = {
        "measured_iterations": measured_iterations,
        "warmup_iterations": warmup_iterations,
    }
    if mode is RunMode.TIMING:
        terminal_payload.update(
            median_ns=median_ns,
            p90_ns=p90_ns,
            sample_count=len(samples),
        )
    else:
        xplanes = sorted((output_dir / "profile").rglob("*.xplane.pb"))
        if len(xplanes) != 1:
            raise ValueError(f"PROFILE_XPLANE_COUNT_MISMATCH observed={xplanes}")
        trace_role = (
            ArtifactRole.TIMING_TRACE if mode is RunMode.TRACE else ArtifactRole.COUNTER_TRACE
        )
        artifacts.append(_artifact(output_dir, xplanes[0], trace_role))
        terminal_payload.update(
            xplane_sha256=_sha256(xplanes[0]),
            xplane_size_bytes=xplanes[0].stat().st_size,
        )
    _record_event(ledger_path, run_id, terminal_state, terminal_payload)
    artifacts.append(_artifact(output_dir, ledger_path, ArtifactRole.EXECUTION_LEDGER))
    result = FusedRpaRunResult(
        run_id=run_id,
        mode=mode,
        backend=jax.default_backend(),
        device_kind=device_kind,
        device_count=len(jax.devices()),
        execution_scope=plan.execution_scope,
        schedule_sha256=plan.schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
        stablehlo_sha256=stablehlo_artifact.sha256,
        compiler_hlo_sha256=compiler_hlo_artifact.sha256,
        backend_manifest=tuple(
            SourceFileContract(path=path, sha256=sha256) for path, sha256 in backend_manifest
        ),
        backend_executor=backend_executor,
        backend_executor_sha256=backend_executor_sha256,
        preflight_passed=True,
        input_sha256=tuple(artifact.sha256 for artifact in input_artifacts),
        output_sha256=tuple(artifact.sha256 for artifact in output_artifacts),
        oracle_sha256=tuple(artifact.sha256 for artifact in oracle_artifacts),
        passed=passed,
        maximum_absolute_errors=(maximum_errors[0][0], maximum_errors[1][0]),
        maximum_relative_errors=(maximum_errors[0][1], maximum_errors[1][1]),
        compile_duration_ns=compile_duration_ns,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
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
        ArtifactRole.TIMING_SAMPLES if mode is RunMode.TIMING else ArtifactRole.PROFILE_ASSESSMENT,
    )
    return result
