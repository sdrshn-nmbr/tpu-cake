from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    BenchmarkProtocol,
    ExecutionContract,
    KernelExperiment,
    NumericalContract,
    ProfileExpectation,
    RuntimeIdentity,
    SearchPolicy,
    SourceFileContract,
    TargetHardware,
    TensorContract,
    WorkloadContract,
    WorkloadStage,
    experiment_artifact_json,
)
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, array_sha256, semantic_sha256
from tpu_cake.ledger import RunState
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
from tpu_cake.seqax_pallas_lowering import (
    SEQAX_PALLAS_EXECUTION_SCHEMA,
    SeqaxPallasPlan,
    lower_seqax_physical_to_pallas,
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
from tpu_cake.workloads.seqax_forward import (
    SEQAX_FORWARD_INPUT_NAMES,
    SEQAX_REVISION,
    seqax_forward_schedule,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)


class SeqaxPallasInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_schema: str
    execution_schema: str
    mode: RunMode
    seed: int
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    parameters: dict[str, int]
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_region_count: int = Field(gt=0)
    execution_scope: str

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxPallasInvocation:
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("SEQAX_PALLAS_IDENTITY_SCHEMA_MISMATCH")
        if self.execution_schema != SEQAX_PALLAS_EXECUTION_SCHEMA:
            raise ValueError("SEQAX_PALLAS_EXECUTION_SCHEMA_MISMATCH")
        if (
            self.seed != SEQAX_EVIDENCE_SEED
            or self.warmup_iterations != SEQAX_EVIDENCE_WARMUP_ITERATIONS
            or self.measured_iterations != SEQAX_EVIDENCE_MEASURED_ITERATIONS
            or self.parameters != SEQAX_EVIDENCE_PARAMETERS
        ):
            raise ValueError("SEQAX_PALLAS_EVIDENCE_PROTOCOL_MISMATCH")
        return self


class SeqaxPallasRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: RunMode
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    mesh: tuple[tuple[str, int], ...]
    execution_scope: str
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: tuple[str, ...]
    output_sha256: tuple[str, ...]
    oracle_sha256: tuple[str, ...]
    correctness_passed: bool
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


def _tensor_contract(name: str, contract: Any) -> TensorContract:
    return TensorContract(
        name=name,
        shape=tuple(size for _, size in contract.shape),
        logical_shape=tuple(dimension for dimension, _ in contract.shape),
        dtype=contract.dtype,
        sharding=tuple("+".join(axes) for axes in contract.declared_sharding),
    )


def seqax_physical_pallas_experiment(plan: SeqaxPallasPlan) -> KernelExperiment:
    if len(plan.input_contracts) != len(SEQAX_FORWARD_INPUT_NAMES):
        raise ValueError("SEQAX_PALLAS_INPUT_CONTRACT_COUNT_MISMATCH")
    if len(plan.output_contracts) != 1:
        raise ValueError("SEQAX_PALLAS_OUTPUT_CONTRACT_COUNT_MISMATCH")
    return KernelExperiment(
        workload=WorkloadContract(
            name="seqax-complete-physical-pallas-forward",
            stage=WorkloadStage.CONTROL,
            inputs=tuple(
                _tensor_contract(name, value)
                for name, value in zip(
                    SEQAX_FORWARD_INPUT_NAMES,
                    plan.input_contracts,
                    strict=True,
                )
            ),
            outputs=(_tensor_contract("logits", plan.output_contracts[0]),),
            numerical=NumericalContract(
                reference="canonical CPU JAX Seqax forward reference rounded to 1e-6",
                absolute_tolerance=SEQAX_OUTPUT_ATOL,
                relative_tolerance=SEQAX_OUTPUT_RTOL,
            ),
            execution=ExecutionContract(
                executor="tpu_cake.seqax_pallas_lowering.SeqaxPallasPlan.build",
                scope=plan.execution_scope,
                source_revision=SEQAX_REVISION,
                source_manifest=tuple(
                    SourceFileContract(path=path, sha256=sha256)
                    for path, sha256 in plan.implementation_manifest
                ),
            ),
        ),
        target=TargetHardware(
            accelerator="TPU7x",
            topology="mesh(d=2,t=4)",
            chip_count=4,
            vmem_budget_bytes_per_core=128 << 20,
            smem_budget_bytes_per_core=32 << 20,
            runtime_target="JAX shard_map with Pallas physical contractions",
        ),
        benchmark=BenchmarkProtocol(
            warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
            measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
            synchronization="block until every output shard is ready",
            statistic="median synchronized physical Pallas forward duration",
        ),
        search=SearchPolicy(
            objective_metric="median_synchronized_physical_pallas_forward_duration_ns",
        ),
        profile=ProfileExpectation(
            name="seqax-complete-physical-pallas-forward",
            stage=WorkloadStage.CONTROL,
            minimum_tpu_device_planes=plan.device_count,
            require_tensor_core_activity=False,
            required_timed_hlo_markers=(
                "pallas_call",
                "all-gather",
                "reduce_scatter",
            ),
        ),
        schedule_sha256=plan.physical_schedule_sha256,
    )


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


def _validate_compiled_program(
    stablehlo: str,
    compiler_hlo: str,
    *,
    pallas_region_count: int,
) -> None:
    stable_regions = stablehlo.count("seqax_named_einsum")
    compiler_regions = sum(
        'custom_call_target="tpu_custom_call"' in line
        for line in compiler_hlo.splitlines()
    )
    if stable_regions != pallas_region_count or compiler_regions != pallas_region_count:
        raise ValueError(
            "SEQAX_PALLAS_COMPILED_REGION_COUNT_MISMATCH "
            f"expected={pallas_region_count} stablehlo={stable_regions} "
            f"compiler_hlo={compiler_regions}"
        )
    for marker in ("all-gather", "reduce-scatter"):
        if marker not in compiler_hlo:
            raise ValueError(f"SEQAX_PALLAS_COMPILER_HLO_MISSING marker={marker}")


def _require_clean_repository(repo_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_PALLAS_SOURCE_IS_DIRTY status={status}")


def run_seqax_physical_pallas(
    output_dir: Path,
    *,
    mode: RunMode,
) -> SeqaxPallasRunResult:
    if jax.default_backend() != "tpu":
        raise ValueError("Seqax physical Pallas evidence requires a TPU backend")
    devices = tuple(jax.devices())
    if (
        len(devices) != 8
        or any(device.platform != "tpu" for device in devices)
        or any(device.device_kind not in {"TPU7x", "TPU v7x"} for device in devices)
    ):
        raise ValueError(
            "SEQAX_PALLAS_DEVICE_IDENTITY_MISMATCH "
            f"devices={[(device.platform, device.device_kind) for device in devices]}"
        )
    repo_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repo_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    distributed = seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    physical = lower_seqax_forward_to_physical(distributed).module
    plan = lower_seqax_physical_to_pallas(distributed, physical)
    if len(devices) != plan.device_count:
        raise ValueError(
            f"SEQAX_PALLAS_DEVICE_COUNT_MISMATCH expected={plan.device_count} "
            f"observed={len(devices)}"
        )
    source = plan.render_executable_source()
    experiment = seqax_physical_pallas_experiment(plan)
    profiler_contract = _profiler_contract(mode)
    if profiler_contract != expected_seqax_profiler_contract(mode):
        raise ValueError("SEQAX_PALLAS_PROFILER_CONTRACT_MISMATCH")
    invocation = SeqaxPallasInvocation(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        execution_schema=SEQAX_PALLAS_EXECUTION_SCHEMA,
        mode=mode,
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
    run_id = semantic_sha256(
        "seqax-physical-pallas-forward-run-v1",
        mode.value,
        plan.distributed_schedule_sha256,
        plan.physical_schedule_sha256,
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
        *_source_state(repo_root, output_dir),
    ]
    ledger_path = output_dir / "ledger.sqlite"
    _record_event(
        ledger_path,
        run_id,
        RunState.CREATED,
        invocation.model_dump(mode="json"),
    )
    distributed.verify()
    physical.verify()
    distributed_artifact = _write_text(
        output_dir / "distributed.xdsl",
        canonical_text(distributed),
        ArtifactRole.DISTRIBUTED_IR,
    )
    physical_artifact = _write_text(
        output_dir / "physical.xdsl",
        canonical_text(physical),
        ArtifactRole.PHYSICAL_IR,
    )
    artifacts.extend((distributed_artifact, physical_artifact))
    _record_event(
        ledger_path,
        run_id,
        RunState.VERIFIED,
        {
            "distributed_schedule_sha256": plan.distributed_schedule_sha256,
            "physical_schedule_sha256": plan.physical_schedule_sha256,
        },
    )
    source_artifact = _write_text(
        output_dir / "lowered_pallas.py",
        source,
        ArtifactRole.PALLAS_SOURCE,
    )
    manifest_artifact = _write_json(
        output_dir / "plan_manifest.json",
        plan.manifest(),
        ArtifactRole.PLAN_MANIFEST,
    )
    artifacts.extend((source_artifact, manifest_artifact))
    _record_event(
        ledger_path,
        run_id,
        RunState.LOWERED,
        {
            "physical_schedule_sha256": plan.physical_schedule_sha256,
            "pallas_source_sha256": source_artifact.sha256,
            "plan_manifest_sha256": manifest_artifact.sha256,
            "pallas_region_count": plan.pallas_region_count,
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
    namespace: dict[str, Any] = {}
    exec(compile(source, "<seqax-physical-pallas>", "exec"), namespace)  # noqa: S102
    replayed_plan = namespace["PLAN"]
    if replayed_plan.manifest() != plan.manifest():
        raise ValueError("SEQAX_PALLAS_SOURCE_REPLAY_MISMATCH")
    executable, mesh = namespace["build"](interpret=False, devices=devices)
    compile_inputs = tuple(
        jax.device_put(
            jnp.asarray(value),
            NamedSharding(mesh, contract.partition_spec()),
        )
        for value, contract in zip(host_inputs, plan.input_contracts, strict=True)
    )
    compile_started = time.perf_counter_ns()
    lowered = executable.lower(*compile_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiler_hlo = _compiler_hlo(lowered)
    _validate_compiled_program(
        stablehlo,
        compiler_hlo,
        pallas_region_count=plan.pallas_region_count,
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
            "SEQAX_PALLAS_CORRECTNESS_FAILED "
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
                with jax.profiler.StepTraceAnnotation(
                    "seqax_physical_pallas_forward",
                    step_num=step,
                ):
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
        xplanes = sorted((output_dir / "profile").rglob("*.xplane.pb"))
        if len(xplanes) != 1:
            raise ValueError(f"SEQAX_PALLAS_XPLANE_COUNT_MISMATCH observed={xplanes}")
        artifacts.append(
            _artifact(
                output_dir,
                xplanes[0],
                ArtifactRole.TIMING_TRACE
                if mode is RunMode.TRACE
                else ArtifactRole.COUNTER_TRACE,
            )
        )
        terminal_payload.update(
            profile_root="profile",
            xplane_sha256=_sha256(xplanes[0]),
            xplane_size_bytes=xplanes[0].stat().st_size,
        )
    _record_event(ledger_path, run_id, terminal, terminal_payload)
    ledger_artifact = _artifact(output_dir, ledger_path, ArtifactRole.EXECUTION_LEDGER)
    artifacts.append(ledger_artifact)
    result = SeqaxPallasRunResult(
        run_id=run_id,
        mode=mode,
        backend=jax.default_backend(),
        device_kind=devices[0].device_kind,
        device_count=len(devices),
        mesh=plan.mesh,
        execution_scope=plan.execution_scope,
        distributed_schedule_sha256=plan.distributed_schedule_sha256,
        physical_schedule_sha256=plan.physical_schedule_sha256,
        pallas_source_sha256=source_artifact.sha256,
        plan_manifest_sha256=manifest_artifact.sha256,
        stablehlo_sha256=stablehlo_artifact.sha256,
        compiler_hlo_sha256=compiler_hlo_artifact.sha256,
        input_sha256=tuple(array_sha256(value) for value in host_inputs),
        output_sha256=(array_sha256(actual),),
        oracle_sha256=(array_sha256(oracle),),
        correctness_passed=True,
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
