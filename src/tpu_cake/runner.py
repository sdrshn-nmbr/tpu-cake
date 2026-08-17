from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from enum import StrEnum
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from pydantic import BaseModel, ConfigDict, Field

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity
from tpu_cake.cost_model import estimate_distributed_matmul, tpu7x_tensorcore_rates
from tpu_cake.identity import array_sha256, semantic_sha256, workload_rng
from tpu_cake.lowering import lower_distributed_matmul
from tpu_cake.metrics import MetricSource
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


class RunMode(StrEnum):
    TIMING = "timing"
    TRACE = "trace"
    COUNTERS = "counters"


class MatmulRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: RunMode
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)
    compile_duration_ns: int = Field(ge=0)
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    samples_ns: tuple[int, ...]
    median_ns: int | None = Field(default=None, ge=0)
    p90_ns: int | None = Field(default=None, ge=0)
    coefficient_of_variation: float | None = Field(default=None, ge=0)
    runtime: RuntimeIdentity
    artifacts: tuple[ArtifactReference, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, value: str, role: ArtifactRole) -> ArtifactReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return ArtifactReference(
        path=str(path), size_bytes=path.stat().st_size, sha256=_sha256(path), role=role
    )


def _write_json(path: Path, value: object, role: ArtifactRole) -> ArtifactReference:
    return _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n", role)


def _runtime_identity() -> RuntimeIdentity:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return RuntimeIdentity(
        python=platform.python_version(),
        jax=jax.__version__,
        jaxlib=version("jaxlib"),
        libtpu=version("libtpu"),
        xla=os.environ.get("LIBTPU_INIT_ARGS"),
    )


def _profiler_options(mode: RunMode) -> jax.profiler.ProfileOptions:
    options = jax.profiler.ProfileOptions()
    options.raise_error_on_start_failure = True
    options.enable_hlo_proto = True
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    options.advanced_configuration = {
        "tpu_num_chips_to_profile_per_task": 4,
    }
    if mode is RunMode.COUNTERS:
        options.advanced_configuration.update(
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
    return options


def _percentile(samples: list[int], fraction: float) -> int:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def run_distributed_matmul(
    output_dir: Path,
    *,
    mode: RunMode,
    mesh_size: int,
    m: int,
    k: int,
    n: int,
    warmup_iterations: int,
    measured_iterations: int,
    interpret: bool = False,
) -> MatmulRunResult:
    if mode is not RunMode.TIMING and jax.default_backend() != "tpu":
        raise ValueError("device traces and hardware counters require a TPU backend")
    output_dir.mkdir(parents=True, exist_ok=False)
    distributed = distributed_matmul_schedule(mesh_size=mesh_size, m=m, k=k, n=n)
    physical = lower_distributed_matmul(distributed)
    plan = lower_physical_matmul_to_pallas(physical)
    artifacts = [
        _write_text(
            output_dir / "distributed.xdsl",
            canonical_text(distributed),
            ArtifactRole.DISTRIBUTED_IR,
        ),
        _write_text(
            output_dir / "physical.xdsl", canonical_text(physical), ArtifactRole.PHYSICAL_IR
        ),
        _write_text(
            output_dir / "lowered_pallas.py",
            plan.render_source(),
            ArtifactRole.PALLAS_SOURCE,
        ),
    ]
    model_input = {
        "schedule_sha256": plan.schedule_sha256,
        "mesh_size": mesh_size,
        "m": m,
        "k": k,
        "n": n,
        "hardware": tpu7x_tensorcore_rates().model_dump(mode="json"),
    }
    model_input_artifact = _write_json(
        output_dir / "cost_model_input.json", model_input, ArtifactRole.COST_MODEL_INPUT
    )
    artifacts.append(model_input_artifact)
    cost_report = estimate_distributed_matmul(
        plan,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=model_input_artifact.sha256,
            artifact_path=model_input_artifact.path,
            tool="tpu-cake",
            field="distributed-matmul-v1",
        ),
    )
    artifacts.append(
        _write_text(
            output_dir / "cost_model.json",
            cost_report.model_dump_json(indent=2) + "\n",
            ArtifactRole.COST_MODEL,
        )
    )
    executable, mesh = plan.build(interpret=interpret)
    generator = workload_rng(plan.schedule_sha256, "device-run", "attempt-0", "inputs")
    lhs_host = generator.normal(size=plan.global_lhs_shape).astype(np.float32)
    rhs_host = generator.normal(size=plan.global_rhs_shape).astype(np.float32)
    lhs = jax.device_put(
        jnp.asarray(lhs_host, dtype=jnp.bfloat16),
        NamedSharding(mesh, PartitionSpec(None, plan.mesh_axis)),
    )
    rhs = jax.device_put(
        jnp.asarray(rhs_host, dtype=jnp.bfloat16),
        NamedSharding(mesh, PartitionSpec(plan.mesh_axis, None)),
    )
    compile_start = time.perf_counter_ns()
    lowered = executable.lower(lhs, rhs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    hlo_computation = lowered.compiler_ir(dialect="hlo")
    hlo = (
        hlo_computation.as_hlo_text()
        if hasattr(hlo_computation, "as_hlo_text")
        else str(hlo_computation)
    )
    compiled = lowered.compile()
    compile_duration_ns = time.perf_counter_ns() - compile_start
    artifacts.extend(
        (
            _write_text(output_dir / "stablehlo.txt", stablehlo + "\n", ArtifactRole.STABLEHLO),
            _write_text(output_dir / "compiler_hlo.txt", hlo + "\n", ArtifactRole.COMPILER_HLO),
        )
    )
    for _ in range(warmup_iterations):
        compiled(lhs, rhs).block_until_ready()
    samples: list[int] = []
    if mode is RunMode.TIMING:
        for _ in range(measured_iterations):
            started = time.perf_counter_ns()
            compiled(lhs, rhs).block_until_ready()
            samples.append(time.perf_counter_ns() - started)
    else:
        trace_dir = output_dir / "profile"
        jax.profiler.start_trace(trace_dir, profiler_options=_profiler_options(mode))
        try:
            for step in range(measured_iterations):
                with jax.profiler.StepTraceAnnotation("distributed_matmul", step_num=step):
                    compiled(lhs, rhs).block_until_ready()
        finally:
            jax.profiler.stop_trace()
    actual = compiled(lhs, rhs)
    actual.block_until_ready()
    actual_host = np.asarray(actual)
    lhs_quantized = np.asarray(lhs).astype(np.float32)
    rhs_quantized = np.asarray(rhs).astype(np.float32)
    expected_host = lhs_quantized @ rhs_quantized
    absolute = np.abs(actual_host - expected_host)
    denominator = np.maximum(np.abs(expected_host), np.finfo(np.float32).tiny)
    maximum_absolute_error = float(absolute.max())
    maximum_relative_error = float((absolute / denominator).max())
    passed = bool(np.allclose(actual_host, expected_host, atol=1e-3, rtol=1e-3))
    np.save(output_dir / "lhs.npy", lhs_quantized)
    np.save(output_dir / "rhs.npy", rhs_quantized)
    np.save(output_dir / "output.npy", actual_host)
    np.save(output_dir / "oracle.npy", expected_host)
    array_roles = {
        "lhs.npy": ArtifactRole.CORRECTNESS_INPUT,
        "rhs.npy": ArtifactRole.CORRECTNESS_INPUT,
        "output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "oracle.npy": ArtifactRole.ORACLE_OUTPUT,
    }
    for name, role in array_roles.items():
        path = output_dir / name
        artifacts.append(
            ArtifactReference(
                path=str(path), size_bytes=path.stat().st_size, sha256=_sha256(path), role=role
            )
        )
    median_ns = int(statistics.median(samples)) if samples else None
    p90_ns = _percentile(samples, 0.9) if samples else None
    coefficient = (
        statistics.pstdev(samples) / statistics.mean(samples)
        if len(samples) > 1 and statistics.mean(samples)
        else None
    )
    result = MatmulRunResult(
        run_id=semantic_sha256(
            plan.schedule_sha256,
            mode.value,
            str(mesh_size),
            str(m),
            str(k),
            str(n),
        ),
        mode=mode,
        backend=jax.default_backend(),
        device_kind=jax.devices()[0].device_kind,
        device_count=len(jax.devices()),
        schedule_sha256=plan.schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
        lhs_sha256=array_sha256(lhs_quantized),
        rhs_sha256=array_sha256(rhs_quantized),
        output_sha256=array_sha256(actual_host),
        passed=passed,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
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
