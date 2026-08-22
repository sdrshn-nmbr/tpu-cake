from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from xdsl.dialects.builtin import ModuleOp

from tpu_cake.artifacts import (
    build_artifact_manifest,
    validate_artifact_manifest,
)
from tpu_cake.artifacts import (
    file_sha256 as _sha256,
)
from tpu_cake.artifacts import save_array as _save_array
from tpu_cake.artifacts import text_sha256 as _text_sha256
from tpu_cake.artifacts import (
    write_json as _write_json,
)
from tpu_cake.artifacts import (
    write_text as _write_text,
)
from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import (
    CompilerCollectiveStrategyPoint,
    CompilerCollectiveStrategySurface,
    CompilerExecutableAnalysis,
    capture_compiler_analysis,
    validate_compiler_analysis,
    write_compiler_analysis,
)
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    ProfileExpectation,
    SourceFileContract,
    WorkloadStage,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import array_sha256, arrays_sha256, json_sha256, semantic_sha256
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.ledger import EvidenceRun, RunState, payload_sha256, read_ledger_history
from tpu_cake.metrics import MetricSource
from tpu_cake.runner import RunMode, _runtime_identity, _source_state
from tpu_cake.seqax_cost_model import SeqaxCostModelReport, estimate_seqax_forward
from tpu_cake.seqax_numerical import (
    _assess_output_arrays,
    assess_seqax_bf16_outputs,
    assess_seqax_cpu_reference_replay,
    decode_seqax_bf16_checkpoint,
    default_seqax_bf16_validation_contract,
    encode_seqax_bf16_checkpoint,
)
from tpu_cake.seqax_pallas_diagnostic import (
    SeqaxPallasDiagnosticAttribution,
    _attribution,
    _bound_program,
    _canonical_assessment,
    _capture_phase,
    _cost_metric,
    _export_xprof,
    _gviz_rows,
    _profile_files,
    _profile_replay,
    _validate_counter_evidence,
    _validate_xprof,
    _validate_xprof_replay,
)
from tpu_cake.seqax_pallas_lowering import (
    SeqaxPallasPlan,
    lower_seqax_physical_to_pallas,
    place_inputs,
)
from tpu_cake.seqax_pallas_runner import (
    _compiler_hlo,
    _physical_collective_inventory,
    _validate_compiled_program,
)
from tpu_cake.seqax_pallas_search import (
    SeqaxPallasDevice,
)
from tpu_cake.seqax_pallas_search import (
    seqax_pallas_device_inventory as _device_inventory,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_residual_profile import (
    SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT,
    SEQAX_RESIDUAL_PROFILE_ITERATIONS,
    SEQAX_RESIDUAL_PROFILE_SCHEMA,
    SEQAX_RESIDUAL_PROFILE_WARMUPS,
    SeqaxResidualCandidateResult,
    SeqaxResidualCorrectnessObservation,
    SeqaxResidualProfileCandidateContract,
    SeqaxResidualProfileCapture,
    SeqaxResidualProfileContract,
    SeqaxResidualProfileReceipt,
    SeqaxResidualProfileResult,
    SeqaxResidualProfileSummary,
    compare_residual_profiles,
    default_seqax_residual_profile_contract,
)
from tpu_cake.seqax_runner import expected_seqax_profiler_contract
from tpu_cake.stablehlo import StableHloInspector
from tpu_cake.workloads.seqax_forward import (
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)
from tpu_cake.xprof_evidence import assess_capture

_COMPILER_STRATEGY_SURFACE_COLUMNS = (96, 100, 104, 108, 112, 116, 120, 124, 128)


@dataclass(frozen=True)
class PreparedResidualProfile:
    expected: SeqaxResidualProfileCandidateContract
    distributed: ModuleOp
    physical: ModuleOp
    plan: SeqaxPallasPlan


@dataclass(frozen=True)
class CompiledResidualProfile:
    prepared: PreparedResidualProfile
    pallas_executable: Any
    control_executable: Any
    mesh: Any
    pallas_stablehlo: str
    pallas_compiler_hlo: str
    control_stablehlo: str
    control_compiler_hlo: str
    pallas_compiler_analysis: CompilerExecutableAnalysis
    control_compiler_analysis: CompilerExecutableAnalysis


def _canonical_hlo(value: str) -> str:
    return value.rstrip("\n") + "\n"


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_array(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"SEQAX_RESIDUAL_PROFILE_ARRAY_INVALID path={path}")
    return np.load(path, allow_pickle=False)


def _source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "canonical.py",
        package / "cli.py",
        package / "compiler_analysis.py",
        package / "contracts.py",
        package / "cost_model.py",
        package / "dtensor_interpreter.py",
        package / "evidence.py",
        package / "frontend.py",
        package / "identity.py",
        package / "jax_lowering.py",
        package / "ledger.py",
        package / "lowering.py",
        package / "metrics.py",
        package / "physical_cost_model.py",
        package / "physical_geometry.py",
        package / "runner.py",
        package / "seqax_cost_model.py",
        package / "seqax_numerical.py",
        package / "seqax_pallas_diagnostic.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_pallas_runner.py",
        package / "stablehlo.py",
        package / "seqax_pallas_search.py",
        package / "seqax_physical_execution.py",
        package / "seqax_physical_lowering.py",
        package / "seqax_residual_profile.py",
        package / "seqax_residual_profile_runner.py",
        package / "seqax_runner.py",
        package / "xprof_evidence.py",
        package / "xprof_export.py",
        package / "dialects" / "distributed_tensor.py",
        package / "dialects" / "tpu_schedule.py",
        package / "workloads" / "seqax_forward.py",
        package / "workloads" / "seqax_oracle.py",
    )
    return tuple(
        SourceFileContract(
            path=path.relative_to(package.parent).as_posix(),
            sha256=_sha256(path),
        )
        for path in paths
    )


def _require_clean_repository(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_RESIDUAL_PROFILE_SOURCE_DIRTY status={status}")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"SEQAX_RESIDUAL_PROFILE_PATH_SYMLINK path={current}")


def _require_compilation_root(repository_root: Path) -> None:
    if repository_root.resolve() != Path(SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT):
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT_MISMATCH "
            f"expected={SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT} observed={repository_root}"
        )


def _require_safe_new_root(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if root.exists() or root.is_symlink():
        raise ValueError(f"SEQAX_RESIDUAL_PROFILE_ROOT_EXISTS path={root}")
    if any(root == value or root in value.parents for value in protected) or (
        repository_root in root.parents
    ):
        raise ValueError(f"SEQAX_RESIDUAL_PROFILE_UNSAFE_ROOT path={root}")


def _preflight_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_RESIDUAL_PROFILE_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_RESIDUAL_PROFILE_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_RESIDUAL_PROFILE_HARDLINK path={path}")


def _parameters(contract: SeqaxResidualProfileContract) -> dict[str, int | Any]:
    parameters = dict(contract.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    return parameters


def _json_sha256(value: object) -> str:
    return json_sha256(value)


def _prepare_candidates(
    contract: SeqaxResidualProfileContract,
) -> tuple[PreparedResidualProfile, ...]:
    prepared = []
    for expected in contract.candidates:
        distributed = seqax_forward_schedule(
            **_parameters(contract),
            residual_norm_strategy=expected.candidate,
        )
        physical = lower_seqax_forward_to_physical(distributed).module
        plan = lower_seqax_physical_to_pallas(distributed, physical)
        observed_collectives = _physical_collective_inventory(physical)
        observed = (
            plan.distributed_schedule_sha256,
            plan.physical_schedule_sha256,
            plan.source_sha256(),
            _json_sha256(plan.manifest()),
            plan.pallas_region_count,
            observed_collectives,
        )
        required = (
            expected.distributed_schedule_sha256,
            expected.physical_schedule_sha256,
            expected.pallas_source_sha256,
            expected.pallas_manifest_sha256,
            expected.expected_pallas_regions,
            (
                expected.expected_all_gathers,
                expected.expected_all_reduces,
                expected.expected_reduce_scatters,
            ),
        )
        if observed != required:
            raise ValueError(
                "SEQAX_RESIDUAL_PROFILE_PLAN_IDENTITY_MISMATCH "
                f"candidate={expected.candidate} expected={required} observed={observed}"
            )
        prepared.append(
            PreparedResidualProfile(
                expected=expected,
                distributed=distributed,
                physical=physical,
                plan=plan,
            )
        )
    return tuple(prepared)


def _resident_inputs(
    host_inputs: tuple[np.ndarray, ...],
    prepared: PreparedResidualProfile,
    mesh: Any,
) -> tuple[jax.Array, ...]:
    return place_inputs(host_inputs, prepared.plan.input_contracts, mesh)


def _compile(
    prepared: PreparedResidualProfile,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
    *,
    enforce_hlo_identity: bool = True,
) -> CompiledResidualProfile:
    pallas_callable, mesh = prepared.plan.build(interpret=False, devices=devices)
    resident = _resident_inputs(host_inputs, prepared, mesh)
    pallas_lowered = pallas_callable.lower(*resident)
    pallas_stablehlo = _canonical_hlo(str(pallas_lowered.compiler_ir(dialect="stablehlo")))
    pallas_pre_optimization_hlo = _canonical_hlo(_compiler_hlo(pallas_lowered))
    expected = prepared.expected
    _validate_compiled_program(
        pallas_stablehlo,
        pallas_pre_optimization_hlo,
        pallas_region_count=prepared.plan.pallas_region_count,
        pallas_vector_region_count=prepared.plan.pallas_vector_region_count,
        all_gather_count=expected.expected_all_gathers,
        all_reduce_count=expected.expected_all_reduces,
        reduce_scatter_count=expected.expected_reduce_scatters,
    )
    collective_counts = StableHloInspector.parse(pallas_stablehlo).live_collective_counts()
    stable_counts = tuple(
        collective_counts[name] for name in ("all_gather", "all_reduce", "reduce_scatter")
    )
    expected_counts = (
        expected.expected_all_gathers,
        expected.expected_all_reduces,
        expected.expected_reduce_scatters,
    )
    if stable_counts != expected_counts:
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_STABLEHLO_COLLECTIVE_MISMATCH "
            f"candidate={expected.candidate} expected={expected_counts} observed={stable_counts}"
        )
    control_callable, _control_mesh = lower_distributed_program_to_jax_mesh(
        prepared.distributed
    ).build(devices=devices)
    control_lowered = control_callable.lower(*resident)
    control_stablehlo = _canonical_hlo(str(control_lowered.compiler_ir(dialect="stablehlo")))
    pallas_executable = pallas_lowered.compile()
    control_executable = control_lowered.compile()
    pallas_compiler_hlo = _canonical_hlo(pallas_executable.as_text())
    control_compiler_hlo = _canonical_hlo(control_executable.as_text())
    identities = (
        _text_sha256(pallas_stablehlo),
        _text_sha256(control_stablehlo),
    )
    required_identities = (
        expected.pallas_stablehlo_sha256,
        expected.control_stablehlo_sha256,
    )
    if enforce_hlo_identity and identities != required_identities:
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_COMPILED_IDENTITY_MISMATCH "
            f"candidate={expected.candidate} expected={required_identities} observed={identities}"
        )
    pallas_compiler_analysis = capture_compiler_analysis(
        pallas_executable,
        stablehlo=pallas_stablehlo.rstrip("\n"),
        compiler_hlo=pallas_compiler_hlo.rstrip("\n"),
    )
    if pallas_compiler_analysis.collectives != expected.expected_pallas_compiler_collectives:
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_COMPILER_COLLECTIVE_MISMATCH "
            f"candidate={expected.candidate} "
            f"expected={expected.expected_pallas_compiler_collectives} "
            f"observed={pallas_compiler_analysis.collectives}"
        )
    return CompiledResidualProfile(
        prepared=prepared,
        pallas_executable=pallas_executable,
        control_executable=control_executable,
        mesh=mesh,
        pallas_stablehlo=pallas_stablehlo,
        pallas_compiler_hlo=pallas_compiler_hlo,
        control_stablehlo=control_stablehlo,
        control_compiler_hlo=control_compiler_hlo,
        pallas_compiler_analysis=pallas_compiler_analysis,
        control_compiler_analysis=capture_compiler_analysis(
            control_executable,
            stablehlo=control_stablehlo.rstrip("\n"),
            compiler_hlo=control_compiler_hlo.rstrip("\n"),
        ),
    )


def _capture_compiler_strategy_surface(
    root: Path,
    devices: tuple[Any, ...],
) -> CompilerCollectiveStrategySurface:
    mesh = Mesh(np.asarray(devices, dtype=object).reshape(2, 4), ("d", "t"))
    points = []
    for columns in _COMPILER_STRATEGY_SURFACE_COLUMNS:
        value = jax.device_put(
            jnp.ones((128, columns), jnp.bfloat16),
            NamedSharding(mesh, P()),
        )
        residual = jax.device_put(
            jnp.ones((128, columns), jnp.bfloat16),
            NamedSharding(mesh, P(None, "t")),
        )

        def boundary(argument: jax.Array, local_residual: jax.Array) -> jax.Array:
            shard = jax.lax.psum_scatter(
                argument,
                "t",
                scatter_dimension=1,
                tiled=True,
            ) * jnp.bfloat16(0.25)
            shard = (shard + local_residual) * jnp.bfloat16(0.5)
            return jax.lax.all_gather(
                shard,
                "t",
                axis=1,
                tiled=True,
                to="invarying",
            )

        mapped = jax.shard_map(
            boundary,
            mesh=mesh,
            in_specs=(P(), P(None, "t")),
            out_specs=P(),
            axis_names={"t"},
        )
        lowered = jax.jit(mapped).lower(value, residual)
        stablehlo = _canonical_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
        executable = lowered.compile()
        compiler_hlo = _canonical_hlo(executable.as_text())
        analysis = capture_compiler_analysis(
            executable,
            stablehlo=stablehlo.rstrip("\n"),
            compiler_hlo=compiler_hlo.rstrip("\n"),
        )
        output = np.asarray(executable(value, residual))
        output_f32 = output.astype(np.float32)
        if not np.array_equal(output_f32, np.ones((128, columns), dtype=np.float32)):
            raise ValueError(f"SEQAX_RESIDUAL_COMPILER_SURFACE_OUTPUT_MISMATCH columns={columns}")
        point_root = root / "compiler_strategy_surface" / str(columns)
        _write_text(point_root / "stablehlo.txt", stablehlo)
        _write_text(point_root / "compiler_hlo.txt", compiler_hlo)
        write_compiler_analysis(point_root / "compiler_analysis.json", analysis)
        _save_array(point_root / "output.npy", output_f32)
        points.append(
            CompilerCollectiveStrategyPoint(
                rows=128,
                columns=columns,
                payload_bytes_per_device=128 * columns * 2,
                stablehlo_sha256=_sha256(point_root / "stablehlo.txt"),
                compiler_hlo_sha256=_sha256(point_root / "compiler_hlo.txt"),
                compiler_analysis_sha256=_sha256(point_root / "compiler_analysis.json"),
                output_sha256=_sha256(point_root / "output.npy"),
                collectives=analysis.collectives,
            )
        )
    surface = CompilerCollectiveStrategySurface(
        mesh_axes=(("d", 2), ("t", 4)),
        dtype="bfloat16",
        points=tuple(points),
    )
    _write_json(root / "compiler_strategy_surface.json", surface.model_dump(mode="json"))
    return surface


def _validate_compiler_strategy_surface(root: Path) -> CompilerCollectiveStrategySurface:
    surface = CompilerCollectiveStrategySurface.model_validate_json(
        (root / "compiler_strategy_surface.json").read_text()
    )
    if tuple(point.columns for point in surface.points) != _COMPILER_STRATEGY_SURFACE_COLUMNS:
        raise ValueError("SEQAX_RESIDUAL_COMPILER_SURFACE_POINTS_MISMATCH")
    for point in surface.points:
        point_root = root / "compiler_strategy_surface" / str(point.columns)
        analysis = validate_compiler_analysis(
            point_root / "compiler_analysis.json",
            stablehlo_path=point_root / "stablehlo.txt",
            compiler_hlo_path=point_root / "compiler_hlo.txt",
        )
        output = _load_array(point_root / "output.npy")
        if (
            point.stablehlo_sha256 != _sha256(point_root / "stablehlo.txt")
            or point.compiler_hlo_sha256 != _sha256(point_root / "compiler_hlo.txt")
            or point.compiler_analysis_sha256 != _sha256(point_root / "compiler_analysis.json")
            or point.output_sha256 != _sha256(point_root / "output.npy")
            or point.collectives != analysis.collectives
            or not np.array_equal(
                output,
                np.ones((point.rows, point.columns), dtype=np.float32),
            )
        ):
            raise ValueError(
                f"SEQAX_RESIDUAL_COMPILER_SURFACE_REPLAY_MISMATCH columns={point.columns}"
            )
    return surface


def _execute(executable: Any, inputs: tuple[jax.Array, ...]) -> np.ndarray:
    outputs = executable(*inputs)
    jax.block_until_ready(outputs)
    if len(outputs) != 1:
        raise ValueError("SEQAX_RESIDUAL_PROFILE_OUTPUT_COUNT_MISMATCH")
    return np.asarray(outputs[0])


def _validate_devices(devices: tuple[Any, ...], contract: SeqaxResidualProfileContract) -> None:
    if (
        jax.default_backend() != contract.backend
        or len(devices) != contract.device_count
        or tuple(device.id for device in devices) != tuple(range(contract.device_count))
        or any(device.platform != "tpu" for device in devices)
        or any(device.device_kind not in {"TPU7x", "TPU v7x"} for device in devices)
        or any(device.process_index != 0 for device in devices)
    ):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_DEVICE_MISMATCH")


def _validate_verifier_runtime() -> None:
    numerical_runtime = default_seqax_bf16_validation_contract().runtime
    observed = (
        ".".join(platform.python_version().split(".")[:2]),
        jax.__version__,
        importlib.metadata.version("jaxlib"),
        importlib.metadata.version("ml_dtypes"),
    )
    expected = (
        numerical_runtime.python_major_minor,
        numerical_runtime.jax,
        numerical_runtime.jaxlib,
        numerical_runtime.ml_dtypes,
    )
    if observed != expected:
        raise ValueError(
            f"SEQAX_RESIDUAL_PROFILE_VERIFIER_RUNTIME_MISMATCH expected={expected} observed={observed}"
        )


def _expected_profile(
    expected: SeqaxResidualProfileCandidateContract,
    mode: RunMode,
) -> ProfileExpectation:
    required = ["pallas_call", "all-gather", "all-reduce", "reduce_scatter"]
    return ProfileExpectation(
        name=f"seqax-residual-profile-{expected.candidate}",
        stage=WorkloadStage.CONTROL,
        minimum_tpu_device_planes=8,
        require_tensor_core_activity=False,
        require_hbm_read_counters=mode is RunMode.COUNTERS,
        require_hbm_write_counters=mode is RunMode.COUNTERS,
        require_cycle_counters=mode is RunMode.COUNTERS,
        minimum_counter_device_planes=4 if mode is RunMode.COUNTERS else 0,
        required_timed_hlo_markers=tuple(required),
        forbidden_timed_hlo_fragments=(),
    )


def _cost_report(
    prepared: PreparedResidualProfile,
    compiler_analysis: CompilerExecutableAnalysis,
) -> SeqaxCostModelReport:
    report = estimate_seqax_forward(
        prepared.distributed,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=prepared.plan.distributed_schedule_sha256,
            artifact_path=f"candidates/{prepared.expected.candidate}/distributed.xdsl",
            tool="tpu-cake",
            field="canonical distributed tensor program",
        ),
        expected_schedule_sha256=prepared.plan.distributed_schedule_sha256,
    )
    return report.model_copy(update={"compiler_collectives": compiler_analysis.collectives})


def _profile_summary(
    *,
    expected: SeqaxResidualProfileCandidateContract,
    mode: RunMode,
    attribution: SeqaxPallasDiagnosticAttribution,
    hlo_stats: Path,
) -> SeqaxResidualProfileSummary:
    rows = tuple(
        row for row in _gviz_rows(hlo_stats) if str(row.get("program_id")) == attribution.program_id
    )
    semantic_all_gathers = sum(row.get("category") == "all-gather" for row in rows)
    all_reduce_rows = tuple(row for row in rows if row.get("category") == "all-reduce")
    semantic_all_reduces = len(all_reduce_rows)
    semantic_reduce_scatters = sum(row.get("category") == "reduce-scatter" for row in rows)
    expected_all_reduce_occurrences = attribution.module_execution_count * 8
    if any(
        int(row.get("occurrences") or 0) != expected_all_reduce_occurrences
        or not math.isfinite(float(row.get("avg_self_time") or 0))
        or float(row.get("avg_self_time") or 0) <= 0
        for row in all_reduce_rows
    ):
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_ALL_REDUCE_EVIDENCE_MISMATCH "
            f"candidate={expected.candidate} mode={mode} "
            f"expected_occurrences={expected_all_reduce_occurrences}"
        )
    all_reduce_ns = sum(float(row["avg_self_time"]) * 1_000 for row in all_reduce_rows)
    completion_rows = sum(
        row.get("category") == "async-done"
        and str(row.get("hlo_op_name", "")).startswith(
            ("all-gather", "all-reduce", "reduce-scatter")
        )
        and "call-done" in str(row.get("hlo_op_name", ""))
        for row in rows
    )
    summary = SeqaxResidualProfileSummary(
        candidate=expected.candidate,
        mode=mode,
        module_execution_count=attribution.module_execution_count,
        module_median_duration_ns=attribution.module_median_duration_ns,
        module_p90_duration_ns=attribution.module_p90_duration_ns,
        pallas_average_self_time_sum_ns_per_device=(
            attribution.pallas_average_self_time_sum_ns_per_device
        ),
        collective_completion_average_self_time_sum_ns_per_device=(
            attribution.collective_completion_average_self_time_sum_ns_per_device
        ),
        all_reduce_average_self_time_sum_ns_per_device=all_reduce_ns,
        semantic_all_gather_rows=semantic_all_gathers,
        semantic_all_reduce_rows=semantic_all_reduces,
        semantic_reduce_scatter_rows=semantic_reduce_scatters,
        async_collective_completion_rows=completion_rows,
        static_all_gathers=expected.expected_all_gathers,
        static_all_reduces=expected.expected_all_reduces,
        static_reduce_scatters=expected.expected_reduce_scatters,
        pallas_regions=expected.expected_pallas_regions,
        ring_equivalent_ici_bytes_per_device=(
            expected.expected_ring_equivalent_ici_bytes_per_device
        ),
    )
    observed = (
        summary.semantic_all_gather_rows,
        summary.semantic_all_reduce_rows,
        summary.semantic_reduce_scatter_rows,
        summary.async_collective_completion_rows,
    )
    required = (
        expected.expected_semantic_all_gather_rows,
        expected.expected_semantic_all_reduce_rows,
        expected.expected_semantic_reduce_scatter_rows,
        expected.expected_async_collective_completion_rows,
    )
    if observed != required:
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_OBSERVED_COLLECTIVE_INVENTORY_MISMATCH "
            f"candidate={expected.candidate} mode={mode} expected={required} observed={observed}"
        )
    return summary


def _capture_record(
    *,
    candidate_root: Path,
    expected: SeqaxResidualProfileCandidateContract,
    mode: RunMode,
    xplane: Path,
    assessment: Any,
    attribution: SeqaxPallasDiagnosticAttribution,
) -> SeqaxResidualProfileCapture:
    phase_root = candidate_root / mode.value
    program_id, _program_name = _bound_program(assessment)
    summary = _profile_summary(
        expected=expected,
        mode=mode,
        attribution=attribution,
        hlo_stats=phase_root / "xprof" / "hlo_stats.json",
    )
    counters = assessment.capture.counters
    if mode is RunMode.TRACE:
        periodic_names: tuple[str, ...] = ()
        samples: dict[str, int] = {}
        hbm_read = hbm_write = cycles = 0
    else:
        periodic_names = counters.periodic_counter_names
        samples = counters.periodic_samples_per_tpu_core
        hbm_read = counters.hbm_read_names
        hbm_write = counters.hbm_write_names
        cycles = counters.cycle_names
    return SeqaxResidualProfileCapture(
        candidate=expected.candidate,
        mode=mode,
        step_event=(
            expected.trace_step_event if mode is RunMode.TRACE else expected.counter_step_event
        ),
        profiler_config_sha256=_sha256(phase_root / "profiler_config.json"),
        xplane_sha256=_sha256(xplane),
        assessment_sha256=_sha256(phase_root / "profile_assessment.json"),
        attribution_sha256=_sha256(phase_root / "attribution.json"),
        program_id=program_id,
        summary=summary,
        periodic_counter_names=periodic_names,
        periodic_counter_samples_per_core=samples,
        hbm_read_counter_names=hbm_read,
        hbm_write_counter_names=hbm_write,
        cycle_counter_names=cycles,
    )


def _capture_candidate_phase(
    *,
    candidate_root: Path,
    expected: SeqaxResidualProfileCandidateContract,
    compiled: CompiledResidualProfile,
    resident: tuple[jax.Array, ...],
    mode: RunMode,
    cost_report: SeqaxCostModelReport,
) -> SeqaxResidualProfileCapture:
    for _ in range(SEQAX_RESIDUAL_PROFILE_WARMUPS):
        jax.block_until_ready(compiled.pallas_executable(*resident))
    step_event = expected.trace_step_event if mode is RunMode.TRACE else expected.counter_step_event
    phase_root = candidate_root / mode.value
    profiler_contract = expected_seqax_profiler_contract(mode)
    if profiler_contract != (
        default_seqax_residual_profile_contract(_runtime_identity()).trace_profiler_config
        if mode is RunMode.TRACE
        else default_seqax_residual_profile_contract(_runtime_identity()).counter_profiler_config
    ):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_PROFILER_CONTRACT_MISMATCH")
    xplane, _assessment, _steps, durations = _capture_phase(
        phase_root,
        SimpleNamespace(compiled=compiled.pallas_executable),
        resident,
        mode,
        step_event=step_event,
        iterations=SEQAX_RESIDUAL_PROFILE_ITERATIONS,
    )
    assessment = assess_capture(phase_root, _expected_profile(expected, mode))
    if not assessment.accepted:
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_CAPTURE_REJECTED "
            f"candidate={expected.candidate} mode={mode} "
            f"findings={[value.code for value in assessment.findings]}"
        )
    if mode is RunMode.COUNTERS:
        _validate_counter_evidence(assessment)
    _write_json(
        phase_root / "profile_assessment.json",
        _canonical_assessment(assessment.model_dump(mode="json")),
    )
    program_id, _program_name = _bound_program(assessment)
    attribution = _attribution(
        physical=compiled.prepared.physical,
        program_id=program_id,
        durations=durations,
        hlo_stats=phase_root / "xprof" / "hlo_stats.json",
        cost_report=cost_report,
        iterations=SEQAX_RESIDUAL_PROFILE_ITERATIONS,
        collective_categories=("all-gather", "reduce-scatter"),
        synchronous_collective_categories=("all-reduce",) if expected.expected_all_reduces else (),
    )
    _write_json(phase_root / "attribution.json", attribution.model_dump(mode="json"))
    return _capture_record(
        candidate_root=candidate_root,
        expected=expected,
        mode=mode,
        xplane=xplane,
        assessment=assessment,
        attribution=attribution,
    )


def _save_inputs(
    root: Path,
    seed: int,
    inputs: tuple[np.ndarray, ...],
    scenario: Any,
) -> None:
    for index, (value, tensor) in enumerate(zip(inputs, scenario.inputs, strict=True)):
        stored = (
            encode_seqax_bf16_checkpoint(value, tensor)
            if tensor.dtype == "bfloat16"
            else np.asarray(value)
        )
        _save_array(root / "correctness" / str(seed) / "inputs" / f"{index:02d}.npy", stored)


def _load_inputs(root: Path, seed: int, scenario: Any) -> tuple[np.ndarray, ...]:
    values = []
    for index, tensor in enumerate(scenario.inputs):
        stored = _load_array(root / "correctness" / str(seed) / "inputs" / f"{index:02d}.npy")
        values.append(
            decode_seqax_bf16_checkpoint(stored, tensor) if tensor.dtype == "bfloat16" else stored
        )
    return tuple(values)


def _correctness_observation(
    *,
    root: Path,
    compiled: CompiledResidualProfile,
    host_inputs: tuple[np.ndarray, ...],
    seed: int,
) -> SeqaxResidualCorrectnessObservation:
    numerical = default_seqax_bf16_validation_contract()
    scenario = next(
        value for value in numerical.scenarios if value.name == "calibration-m256-b2-s1-l1"
    )
    resident = _resident_inputs(host_inputs, compiled.prepared, compiled.mesh)
    pallas = _execute(compiled.pallas_executable, resident)
    control = _execute(compiled.control_executable, resident)
    cpu = seqax_forward_canonical_reference(
        host_inputs,
        quantization_decimals=numerical.policy.cpu_reference_quantization_decimals,
        **scenario.parameters.model_dump(),
    )
    assessment = assess_seqax_bf16_outputs(
        pallas,
        control,
        seed=seed,
        inputs=host_inputs,
        policy=numerical.policy,
        scenario=scenario,
    )
    seed_root = root / "correctness" / str(seed)
    _save_inputs(root, seed, host_inputs, scenario)
    _save_array(seed_root / "cpu.npy", cpu)
    _save_array(seed_root / "control.npy", control)
    _save_array(seed_root / "pallas.npy", pallas)
    return SeqaxResidualCorrectnessObservation(
        candidate=compiled.prepared.expected.candidate,
        seed=seed,
        input_sha256=arrays_sha256(host_inputs),
        cpu_output_sha256=array_sha256(cpu),
        control_output_sha256=array_sha256(control),
        pallas_output_sha256=array_sha256(pallas),
        assessment=assessment,
    )


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    fixed = {
        "contract.json": ArtifactRole.EXPERIMENT,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "source_manifest.json": ArtifactRole.BACKEND_MANIFEST,
        "compiler_strategy_surface.json": ArtifactRole.COST_MODEL_INPUT,
        "comparison.json": ArtifactRole.SEARCH_EVIDENCE,
        "result.json": ArtifactRole.TRACE_RESULT,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
    }
    if relative in fixed:
        return fixed[relative]
    if relative.startswith("compiler_strategy_surface/"):
        surface_roles = {
            "stablehlo.txt": ArtifactRole.STABLEHLO,
            "compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
            "compiler_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
            "output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        }
        if path.name in surface_roles:
            return surface_roles[path.name]
        raise ValueError(f"SEQAX_RESIDUAL_PROFILE_ARTIFACT_UNRECOGNIZED path={relative}")
    if not relative.startswith("candidates/"):
        raise ValueError(f"SEQAX_RESIDUAL_PROFILE_ARTIFACT_UNRECOGNIZED path={relative}")
    name = path.name
    roles = {
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "physical.xdsl": ArtifactRole.PHYSICAL_IR,
        "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
        "pallas_stablehlo.txt": ArtifactRole.STABLEHLO,
        "pallas_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "control_stablehlo.txt": ArtifactRole.STABLEHLO,
        "control_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "pallas_compiler_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
        "control_compiler_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
        "cost_model.json": ArtifactRole.COST_MODEL,
        "timing_output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "cpu.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "control.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "pallas.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "profiler_config.json": ArtifactRole.PROFILER_CONFIG,
        "profile_assessment.json": ArtifactRole.PROFILE_ASSESSMENT,
        "attribution.json": ArtifactRole.SEARCH_EVIDENCE,
    }
    if name in roles:
        return roles[name]
    if "/correctness/" in relative and "/inputs/" in relative and name.endswith(".npy"):
        return ArtifactRole.CORRECTNESS_INPUT
    if "/trace/profile/" in relative and relative.endswith(".xplane.pb"):
        return ArtifactRole.TIMING_TRACE
    if "/counters/profile/" in relative and relative.endswith(".xplane.pb"):
        return ArtifactRole.COUNTER_TRACE
    if "/profile/" in relative and relative.endswith(".trace.json.gz"):
        return ArtifactRole.PROFILE_AUXILIARY
    if "/xprof/" in relative:
        return ArtifactRole.XPROF_EXPORT
    raise ValueError(f"SEQAX_RESIDUAL_PROFILE_ARTIFACT_UNRECOGNIZED path={relative}")


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    return build_artifact_manifest(
        root,
        role_for_path=_artifact_role,
    )


def _validate_manifest(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    validate_artifact_manifest(
        root,
        artifacts,
        role_for_path=_artifact_role,
        duplicate_error="SEQAX_RESIDUAL_PROFILE_CLOSED_WORLD_MISMATCH",
        closed_world_error="SEQAX_RESIDUAL_PROFILE_CLOSED_WORLD_MISMATCH",
        mismatch_error=lambda path: f"SEQAX_RESIDUAL_PROFILE_ARTIFACT_MISMATCH path={path}",
    )


def _validate_source(root: Path, result: SeqaxResidualProfileResult) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = tuple(
        SourceFileContract.model_validate(value)
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    state = json.loads((root / "source_state.json").read_text())
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if (
        status
        or manifest != result.source_manifest
        or manifest != _source_manifest()
        or result.source_state_sha256 != _sha256(root / "source_state.json")
        or result.source_manifest_sha256 != _sha256(root / "source_manifest.json")
        or state.get("git_commit") != current_commit
        or state.get("git_dirty") is not False
        or state.get("git_status") != []
        or state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
        or (root / "source_diff.patch").read_bytes() != b""
    ):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_SOURCE_MISMATCH")
    for source in manifest:
        blob = subprocess.run(
            ["git", "show", f"{current_commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(f"SEQAX_RESIDUAL_PROFILE_SOURCE_BLOB_MISMATCH path={source.path}")


def _expected_plan_files(root: Path, prepared: PreparedResidualProfile) -> None:
    candidate_root = root / "candidates" / prepared.expected.candidate
    expected = prepared.expected
    if (
        (candidate_root / "distributed.xdsl").read_text() != canonical_text(prepared.distributed)
        or (candidate_root / "physical.xdsl").read_text() != canonical_text(prepared.physical)
        or (candidate_root / "lowered_pallas.py").read_text()
        != prepared.plan.render_executable_source()
        or json.loads((candidate_root / "plan_manifest.json").read_text())
        != prepared.plan.manifest()
        or _sha256(candidate_root / "pallas_stablehlo.txt") != expected.pallas_stablehlo_sha256
        or _sha256(candidate_root / "control_stablehlo.txt") != expected.control_stablehlo_sha256
    ):
        raise ValueError(
            f"SEQAX_RESIDUAL_PROFILE_PLAN_REPLAY_MISMATCH candidate={expected.candidate}"
        )
    stablehlo = StableHloInspector.parse((candidate_root / "pallas_stablehlo.txt").read_text())
    collective_counts = stablehlo.live_collective_counts()
    replayed_collectives = tuple(
        collective_counts[name] for name in ("all_gather", "all_reduce", "reduce_scatter")
    )
    expected_collectives = (
        expected.expected_all_gathers,
        expected.expected_all_reduces,
        expected.expected_reduce_scatters,
    )
    if replayed_collectives != expected_collectives:
        raise ValueError(
            "SEQAX_RESIDUAL_PROFILE_STABLEHLO_REPLAY_MISMATCH "
            f"candidate={expected.candidate} "
            f"expected={expected_collectives} observed={replayed_collectives}"
        )
    pallas_analysis = validate_compiler_analysis(
        candidate_root / "pallas_compiler_analysis.json",
        stablehlo_path=candidate_root / "pallas_stablehlo.txt",
        compiler_hlo_path=candidate_root / "pallas_compiler_hlo.txt",
    )
    validate_compiler_analysis(
        candidate_root / "control_compiler_analysis.json",
        stablehlo_path=candidate_root / "control_stablehlo.txt",
        compiler_hlo_path=candidate_root / "control_compiler_hlo.txt",
    )
    if pallas_analysis.collectives != expected.expected_pallas_compiler_collectives:
        raise ValueError(
            f"SEQAX_RESIDUAL_PROFILE_COMPILER_REPLAY_MISMATCH candidate={expected.candidate}"
        )


def _replay_correctness(
    *,
    root: Path,
    prepared: PreparedResidualProfile,
    saved: tuple[SeqaxResidualCorrectnessObservation, ...],
) -> None:
    numerical = default_seqax_bf16_validation_contract()
    scenario = next(
        value for value in numerical.scenarios if value.name == "calibration-m256-b2-s1-l1"
    )
    candidate_root = root / "candidates" / prepared.expected.candidate
    replayed = []
    for observation in saved:
        inputs = _load_inputs(candidate_root, observation.seed, scenario)
        expected_inputs = tuple(
            np.asarray(value)
            for value in seqax_forward_inputs(
                seed=observation.seed,
                **scenario.parameters.model_dump(),
            )
        )
        if any(
            not np.array_equal(actual, expected)
            for actual, expected in zip(inputs, expected_inputs, strict=True)
        ):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_INPUT_REPLAY_MISMATCH seed={observation.seed}"
            )
        seed_root = candidate_root / "correctness" / str(observation.seed)
        producer_cpu = _load_array(seed_root / "cpu.npy")
        control = _load_array(seed_root / "control.npy")
        pallas = _load_array(seed_root / "pallas.npy")
        fresh_cpu = seqax_forward_canonical_reference(
            inputs,
            quantization_decimals=numerical.policy.cpu_reference_quantization_decimals,
            **scenario.parameters.model_dump(),
        )
        cpu_replay = assess_seqax_cpu_reference_replay(
            producer_cpu,
            fresh_cpu,
            policy=numerical.policy,
            scenario=scenario,
        )
        producer_assessment = _assess_output_arrays(
            pallas,
            control,
            producer_cpu,
            policy=numerical.policy,
            scenario=scenario,
        )
        fresh_assessment = _assess_output_arrays(
            pallas,
            control,
            fresh_cpu,
            policy=numerical.policy,
            scenario=scenario,
        )
        if (
            not cpu_replay.within_bounds
            or not producer_assessment.final_outputs_satisfy_policy
            or not fresh_assessment.final_outputs_satisfy_policy
        ):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_PORTABLE_CORRECTNESS_FAILED seed={observation.seed}"
            )
        replayed.append(
            SeqaxResidualCorrectnessObservation(
                candidate=prepared.expected.candidate,
                seed=observation.seed,
                input_sha256=arrays_sha256(inputs),
                cpu_output_sha256=array_sha256(producer_cpu),
                control_output_sha256=array_sha256(control),
                pallas_output_sha256=array_sha256(pallas),
                assessment=producer_assessment,
            )
        )
    if tuple(replayed) != saved:
        raise ValueError(
            f"SEQAX_RESIDUAL_PROFILE_CORRECTNESS_REPLAY_MISMATCH candidate={prepared.expected.candidate}"
        )


def _replay_candidate_profiles(
    *,
    root: Path,
    prepared: PreparedResidualProfile,
    cost_report: SeqaxCostModelReport,
) -> tuple[SeqaxResidualProfileCapture, SeqaxResidualProfileCapture]:
    candidate_root = root / "candidates" / prepared.expected.candidate
    records = []
    with tempfile.TemporaryDirectory(prefix="tpu-cake-residual-profile-") as directory:
        replay_parent = Path(directory)
        for mode in (RunMode.TRACE, RunMode.COUNTERS):
            phase_root = candidate_root / mode.value
            if json.loads((phase_root / "profiler_config.json").read_text()) != (
                expected_seqax_profiler_contract(mode)
            ):
                raise ValueError(
                    "SEQAX_RESIDUAL_PROFILE_PROFILER_REPLAY_MISMATCH "
                    f"candidate={prepared.expected.candidate} mode={mode}"
                )
            xplane, _trace_files = _profile_files(phase_root / "profile")
            _validate_xprof(phase_root, xplane)
            replay_root = replay_parent / mode.value
            replay_profile = replay_root / "profile"
            replay_xplane = replay_profile / xplane.relative_to(phase_root / "profile")
            replay_xplane.parent.mkdir(parents=True)
            shutil.copy2(xplane, replay_xplane)
            for trace_file in (phase_root / "profile").rglob("*.trace.json.gz"):
                destination = replay_profile / trace_file.relative_to(phase_root / "profile")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(trace_file, destination)
            _export_xprof(replay_profile, replay_root / "xprof")
            _validate_xprof(replay_root, replay_xplane)
            _validate_xprof_replay(phase_root, replay_root)
            expectation = _expected_profile(prepared.expected, mode)
            saved_assessment = assess_capture(phase_root, expectation)
            assessment = assess_capture(replay_root, expectation)
            if (
                not saved_assessment.accepted
                or not assessment.accepted
                or _canonical_assessment(saved_assessment.model_dump(mode="json"))
                != _canonical_assessment(assessment.model_dump(mode="json"))
            ):
                raise ValueError(
                    "SEQAX_RESIDUAL_PROFILE_ASSESSMENT_REPLAY_MISMATCH "
                    f"candidate={prepared.expected.candidate} mode={mode}"
                )
            if mode is RunMode.COUNTERS:
                _validate_counter_evidence(saved_assessment)
                _validate_counter_evidence(assessment)
            if json.loads((phase_root / "profile_assessment.json").read_text()) != (
                _canonical_assessment(assessment.model_dump(mode="json"))
            ):
                raise ValueError(
                    "SEQAX_RESIDUAL_PROFILE_SAVED_ASSESSMENT_MISMATCH "
                    f"candidate={prepared.expected.candidate} mode={mode}"
                )
            program_id, program_name = _bound_program(assessment)
            step_event = (
                prepared.expected.trace_step_event
                if mode is RunMode.TRACE
                else prepared.expected.counter_step_event
            )
            _steps, durations = _profile_replay(
                xplane,
                program_name,
                step_event=step_event,
                iterations=SEQAX_RESIDUAL_PROFILE_ITERATIONS,
            )
            attribution = _attribution(
                physical=prepared.physical,
                program_id=program_id,
                durations=durations,
                hlo_stats=replay_root / "xprof" / "hlo_stats.json",
                cost_report=cost_report,
                iterations=SEQAX_RESIDUAL_PROFILE_ITERATIONS,
                collective_categories=("all-gather", "reduce-scatter"),
                synchronous_collective_categories=("all-reduce",)
                if prepared.expected.expected_all_reduces
                else (),
            )
            saved_attribution = SeqaxPallasDiagnosticAttribution.model_validate_json(
                (phase_root / "attribution.json").read_text()
            )
            if saved_attribution != attribution:
                raise ValueError(
                    "SEQAX_RESIDUAL_PROFILE_ATTRIBUTION_REPLAY_MISMATCH "
                    f"candidate={prepared.expected.candidate} mode={mode}"
                )
            records.append(
                _capture_record(
                    candidate_root=candidate_root,
                    expected=prepared.expected,
                    mode=mode,
                    xplane=xplane,
                    assessment=assessment,
                    attribution=attribution,
                )
            )
    return records[0], records[1]


def _ledger_payloads(
    *,
    contract: SeqaxResidualProfileContract,
    result: SeqaxResidualProfileResult,
    root: Path,
    require_accepted: bool,
) -> tuple[tuple[RunState, dict[str, object]], ...]:
    payloads: tuple[tuple[RunState, dict[str, object]], ...] = (
        (
            RunState.CREATED,
            {
                "profile_id": contract.profile_id,
                "devices": [value.model_dump(mode="json") for value in result.devices],
            },
        ),
        (
            RunState.VERIFIED,
            {
                "numerical_contract_id": contract.numerical_contract_id,
                "distributed_schedules": {
                    value.candidate: value.distributed_schedule_sha256
                    for value in result.candidates
                },
            },
        ),
        (
            RunState.LOWERED,
            {
                "physical_schedules": {
                    value.candidate: value.physical_schedule_sha256 for value in result.candidates
                },
                "pallas_sources": {
                    value.candidate: value.pallas_source_sha256 for value in result.candidates
                },
            },
        ),
        (
            RunState.COMPILED,
            {
                "compiled_hlo": {
                    value.candidate: {
                        "pallas_stablehlo_sha256": value.pallas_stablehlo_sha256,
                        "pallas_compiler_hlo_sha256": value.pallas_compiler_hlo_sha256,
                        "control_stablehlo_sha256": value.control_stablehlo_sha256,
                        "control_compiler_hlo_sha256": value.control_compiler_hlo_sha256,
                    }
                    for value in result.candidates
                }
            },
        ),
        (
            RunState.CORRECT,
            {
                "correctness": {
                    value.candidate: [
                        {
                            "seed": observation.seed,
                            "input_sha256": observation.input_sha256,
                            "cpu_output_sha256": observation.cpu_output_sha256,
                            "control_output_sha256": observation.control_output_sha256,
                            "pallas_output_sha256": observation.pallas_output_sha256,
                        }
                        for observation in value.correctness
                    ]
                    for value in result.candidates
                }
            },
        ),
        (
            RunState.COUNTERED,
            {
                "captures": {
                    value.candidate: {
                        "trace_xplane_sha256": value.trace.xplane_sha256,
                        "counter_xplane_sha256": value.counters.xplane_sha256,
                        "trace_attribution_sha256": value.trace.attribution_sha256,
                        "counter_attribution_sha256": value.counters.attribution_sha256,
                    }
                    for value in result.candidates
                },
                "comparison": result.comparison.model_dump(mode="json"),
            },
        ),
    )
    if require_accepted:
        payloads += ((RunState.ACCEPTED, {"result_sha256": _sha256(root / "result.json")}),)
    return payloads


def capture_seqax_residual_profile_hlo_identities(
    contract: SeqaxResidualProfileContract,
) -> dict[str, dict[str, str]]:
    repository_root = Path(__file__).resolve().parents[2]
    runtime = _runtime_identity()
    if (
        contract.hlo_identity_status != "pending"
        or runtime != contract.runtime
        or contract != default_seqax_residual_profile_contract(runtime)
    ):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_HLO_CAPTURE_CONTRACT_MISMATCH")
    _require_compilation_root(repository_root)
    _require_clean_repository(repository_root)
    devices = tuple(jax.devices())
    _validate_devices(devices, contract)
    scenario = next(
        value
        for value in default_seqax_bf16_validation_contract().scenarios
        if value.name == "calibration-m256-b2-s1-l1"
    )
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **scenario.parameters.model_dump(),
        )
    )
    compiled = tuple(
        _compile(value, host_inputs, devices, enforce_hlo_identity=False)
        for value in _prepare_candidates(contract)
    )
    return {
        value.prepared.expected.candidate.value: {
            "pallas_stablehlo_sha256": _text_sha256(value.pallas_stablehlo),
            "pallas_compiler_hlo_sha256": _text_sha256(value.pallas_compiler_hlo),
            "control_stablehlo_sha256": _text_sha256(value.control_stablehlo),
            "control_compiler_hlo_sha256": _text_sha256(value.control_compiler_hlo),
        }
        for value in compiled
    }


def _validate(
    root: Path,
    trusted_contract: SeqaxResidualProfileContract,
    *,
    require_accepted: bool,
) -> SeqaxResidualProfileResult:
    if trusted_contract.hlo_identity_status != "pinned":
        raise ValueError("SEQAX_RESIDUAL_PROFILE_HLO_IDENTITIES_PENDING")
    _preflight_root(root)
    saved_contract = SeqaxResidualProfileContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != trusted_contract:
        raise ValueError("SEQAX_RESIDUAL_PROFILE_CONTRACT_MISMATCH")
    result = SeqaxResidualProfileResult.model_validate_json((root / "result.json").read_text())
    expected_devices = tuple(
        SeqaxPallasDevice(
            id=index,
            process_index=0,
            platform="tpu",
            device_kind=trusted_contract.device_kind,
        )
        for index in range(trusted_contract.device_count)
    )
    if (
        result.profile_id != trusted_contract.profile_id
        or result.numerical_contract_id != trusted_contract.numerical_contract_id
        or result.runtime != trusted_contract.runtime
        or result.devices != expected_devices
    ):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_RESULT_IDENTITY_MISMATCH")
    _validate_source(root, result)
    expected_run_id = semantic_sha256(
        SEQAX_RESIDUAL_PROFILE_SCHEMA,
        trusted_contract.profile_id,
        _sha256(root / "source_state.json"),
        _sha256(root / "source_manifest.json"),
        _json_sha256([value.model_dump(mode="json") for value in result.devices]),
    )
    if result.run_id != expected_run_id:
        raise ValueError("SEQAX_RESIDUAL_PROFILE_RUN_ID_MISMATCH")
    _validate_compiler_strategy_surface(root)
    if result.compiler_strategy_surface_sha256 != _sha256(root / "compiler_strategy_surface.json"):
        raise ValueError("SEQAX_RESIDUAL_COMPILER_SURFACE_IDENTITY_MISMATCH")
    prepared = _prepare_candidates(trusted_contract)
    replayed_candidates = []
    for value, saved in zip(prepared, result.candidates, strict=True):
        expected = value.expected
        candidate_root = root / "candidates" / expected.candidate
        if (
            saved.candidate is not expected.candidate
            or saved.distributed_schedule_sha256 != expected.distributed_schedule_sha256
            or saved.physical_schedule_sha256 != expected.physical_schedule_sha256
            or saved.pallas_source_sha256 != expected.pallas_source_sha256
            or saved.pallas_manifest_sha256 != expected.pallas_manifest_sha256
            or saved.pallas_stablehlo_sha256 != expected.pallas_stablehlo_sha256
            or saved.control_stablehlo_sha256 != expected.control_stablehlo_sha256
            or saved.pallas_compiler_analysis_sha256
            != _sha256(candidate_root / "pallas_compiler_analysis.json")
            or saved.control_compiler_analysis_sha256
            != _sha256(candidate_root / "control_compiler_analysis.json")
        ):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_CANDIDATE_IDENTITY_MISMATCH candidate={expected.candidate}"
            )
        _expected_plan_files(root, value)
        _replay_correctness(root=root, prepared=value, saved=saved.correctness)
        compiler_analysis = validate_compiler_analysis(
            candidate_root / "pallas_compiler_analysis.json",
            stablehlo_path=candidate_root / "pallas_stablehlo.txt",
            compiler_hlo_path=candidate_root / "pallas_compiler_hlo.txt",
        )
        cost_report = _cost_report(value, compiler_analysis)
        if (
            json.loads((candidate_root / "cost_model.json").read_text())
            != cost_report.model_dump(mode="json")
            or saved.cost_model_sha256 != _sha256(candidate_root / "cost_model.json")
            or int(_cost_metric(cost_report, "seqax_ici_bidirectional_bytes_per_device"))
            != expected.expected_ring_equivalent_ici_bytes_per_device
        ):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_COST_REPLAY_MISMATCH candidate={expected.candidate}"
            )
        timing_inputs = _load_inputs(
            candidate_root,
            trusted_contract.timing_seed,
            next(
                scenario
                for scenario in default_seqax_bf16_validation_contract().scenarios
                if scenario.name == "calibration-m256-b2-s1-l1"
            ),
        )
        timing_output = _load_array(candidate_root / "timing_output.npy")
        timing_observation = next(
            item for item in saved.correctness if item.seed == trusted_contract.timing_seed
        )
        if (
            saved.timing_input_sha256 != arrays_sha256(timing_inputs)
            or saved.timing_output_sha256 != array_sha256(timing_output)
            or saved.timing_output_sha256 != timing_observation.pallas_output_sha256
        ):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_TIMING_OUTPUT_MISMATCH candidate={expected.candidate}"
            )
        trace, counters = _replay_candidate_profiles(
            root=root,
            prepared=value,
            cost_report=cost_report,
        )
        replayed_candidates.append(saved.model_copy(update={"trace": trace, "counters": counters}))
    if tuple(replayed_candidates) != result.candidates:
        raise ValueError("SEQAX_RESIDUAL_PROFILE_CANDIDATE_REPLAY_MISMATCH")
    comparison = compare_residual_profiles(result.candidates[0], result.candidates[1])
    if result.comparison != comparison or json.loads(
        (root / "comparison.json").read_text()
    ) != comparison.model_dump(mode="json"):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_COMPARISON_REPLAY_MISMATCH")
    history = read_ledger_history(root / "ledger.sqlite", result.run_id)
    payloads = _ledger_payloads(
        contract=trusted_contract,
        result=result,
        root=root,
        require_accepted=require_accepted,
    )
    if tuple(value.state for value in history) != tuple(state for state, _payload in payloads):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_LEDGER_STATE_MISMATCH")
    if tuple(value.payload_sha256 for value in history) != tuple(
        payload_sha256(payload) for _state, payload in payloads
    ):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_LEDGER_PAYLOAD_MISMATCH")
    if require_accepted:
        receipt = SeqaxResidualProfileReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        _validate_manifest(root, receipt.artifacts)
        expected_receipt = SeqaxResidualProfileReceipt(
            status="passed",
            profile_id=trusted_contract.profile_id,
            run_id=result.run_id,
            result_sha256=_sha256(root / "result.json"),
            ledger_sha256=_sha256(root / "ledger.sqlite"),
            artifacts=_artifact_manifest(root),
        )
        if receipt != expected_receipt:
            raise ValueError("SEQAX_RESIDUAL_PROFILE_RECEIPT_MISMATCH")
    else:
        _validate_manifest(root, _artifact_manifest(root))
    return result


def run_seqax_residual_profile(
    root: Path,
    contract: SeqaxResidualProfileContract,
) -> SeqaxResidualProfileResult:
    _reject_symlink_components(root)
    if root.is_symlink():
        raise ValueError("SEQAX_RESIDUAL_PROFILE_ROOT_SYMLINK")
    root = root.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    runtime = _runtime_identity()
    if runtime != contract.runtime or contract != default_seqax_residual_profile_contract(runtime):
        raise ValueError("SEQAX_RESIDUAL_PROFILE_EXTERNAL_CONTRACT_MISMATCH")
    if contract.hlo_identity_status != "pinned":
        raise ValueError("SEQAX_RESIDUAL_PROFILE_HLO_IDENTITIES_PENDING")
    if (root / "receipt.json").is_file():
        return validate_seqax_residual_profile(root, contract)
    _require_compilation_root(repository_root)
    _require_clean_repository(repository_root)
    _require_safe_new_root(root)
    devices = tuple(jax.devices())
    _validate_devices(devices, contract)
    root.mkdir(parents=True)
    _write_json(
        root / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    _source_state(repository_root, root)
    manifest = _source_manifest()
    _write_json(
        root / "source_manifest.json",
        [value.model_dump(mode="json") for value in manifest],
    )
    device_inventory = _device_inventory(devices)
    run_id = semantic_sha256(
        SEQAX_RESIDUAL_PROFILE_SCHEMA,
        contract.profile_id,
        _sha256(root / "source_state.json"),
        _sha256(root / "source_manifest.json"),
        _json_sha256([value.model_dump(mode="json") for value in device_inventory]),
    )
    ledger_path = root / "ledger.sqlite"
    evidence_run = EvidenceRun(ledger_path, run_id)
    evidence_run.create(
        {
            "profile_id": contract.profile_id,
            "devices": [value.model_dump(mode="json") for value in device_inventory],
        }
    )
    prepared = _prepare_candidates(contract)
    evidence_run.transition(
        RunState.VERIFIED,
        {
            "numerical_contract_id": contract.numerical_contract_id,
            "distributed_schedules": {
                value.expected.candidate: value.plan.distributed_schedule_sha256
                for value in prepared
            },
        },
    )
    for value in prepared:
        candidate_root = root / "candidates" / value.expected.candidate
        _write_text(candidate_root / "distributed.xdsl", canonical_text(value.distributed))
        _write_text(candidate_root / "physical.xdsl", canonical_text(value.physical))
        _write_text(candidate_root / "lowered_pallas.py", value.plan.render_executable_source())
        _write_json(candidate_root / "plan_manifest.json", value.plan.manifest())
    evidence_run.transition(
        RunState.LOWERED,
        {
            "physical_schedules": {
                value.expected.candidate: value.plan.physical_schedule_sha256 for value in prepared
            },
            "pallas_sources": {
                value.expected.candidate: value.plan.source_sha256() for value in prepared
            },
        },
    )
    scenario = next(
        value
        for value in default_seqax_bf16_validation_contract().scenarios
        if value.name == "calibration-m256-b2-s1-l1"
    )
    timing_host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **scenario.parameters.model_dump(),
        )
    )
    compiled = tuple(_compile(value, timing_host_inputs, devices) for value in prepared)
    for value in compiled:
        candidate_root = root / "candidates" / value.prepared.expected.candidate
        _write_text(candidate_root / "pallas_stablehlo.txt", value.pallas_stablehlo)
        _write_text(candidate_root / "pallas_compiler_hlo.txt", value.pallas_compiler_hlo)
        _write_text(candidate_root / "control_stablehlo.txt", value.control_stablehlo)
        _write_text(candidate_root / "control_compiler_hlo.txt", value.control_compiler_hlo)
        write_compiler_analysis(
            candidate_root / "pallas_compiler_analysis.json",
            value.pallas_compiler_analysis,
        )
        write_compiler_analysis(
            candidate_root / "control_compiler_analysis.json",
            value.control_compiler_analysis,
        )
    _capture_compiler_strategy_surface(root, devices)
    evidence_run.transition(
        RunState.COMPILED,
        {
            "compiled_hlo": {
                value.prepared.expected.candidate: {
                    "pallas_stablehlo_sha256": _text_sha256(value.pallas_stablehlo),
                    "pallas_compiler_hlo_sha256": _text_sha256(value.pallas_compiler_hlo),
                    "control_stablehlo_sha256": _text_sha256(value.control_stablehlo),
                    "control_compiler_hlo_sha256": _text_sha256(value.control_compiler_hlo),
                }
                for value in compiled
            }
        },
    )
    executions = []
    correctness_by_candidate = {}
    for value in compiled:
        candidate_root = root / "candidates" / value.prepared.expected.candidate
        observations = []
        for seed in contract.correctness_seeds:
            host_inputs = tuple(
                np.asarray(item)
                for item in seqax_forward_inputs(
                    seed=seed,
                    **scenario.parameters.model_dump(),
                )
            )
            observations.append(
                _correctness_observation(
                    root=candidate_root,
                    compiled=value,
                    host_inputs=host_inputs,
                    seed=seed,
                )
            )
        correctness_by_candidate[value.prepared.expected.candidate] = tuple(observations)
        resident = _resident_inputs(timing_host_inputs, value.prepared, value.mesh)
        timing_output = _execute(value.pallas_executable, resident)
        timing_observation = next(
            item for item in observations if item.seed == contract.timing_seed
        )
        if array_sha256(timing_output) != timing_observation.pallas_output_sha256:
            raise ValueError(
                "SEQAX_RESIDUAL_PROFILE_TIMING_PRECHECK_MISMATCH "
                f"candidate={value.prepared.expected.candidate}"
            )
        _save_array(candidate_root / "timing_output.npy", timing_output)
        cost_report = _cost_report(value.prepared, value.pallas_compiler_analysis)
        if int(_cost_metric(cost_report, "seqax_ici_bidirectional_bytes_per_device")) != (
            value.prepared.expected.expected_ring_equivalent_ici_bytes_per_device
        ):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_COST_MISMATCH candidate={value.prepared.expected.candidate}"
            )
        _write_json(candidate_root / "cost_model.json", cost_report.model_dump(mode="json"))
        executions.append((value, resident, timing_output, cost_report))
    correctness_payload = {
        candidate: [
            {
                "seed": item.seed,
                "input_sha256": item.input_sha256,
                "cpu_output_sha256": item.cpu_output_sha256,
                "control_output_sha256": item.control_output_sha256,
                "pallas_output_sha256": item.pallas_output_sha256,
            }
            for item in observations
        ]
        for candidate, observations in correctness_by_candidate.items()
    }
    evidence_run.transition(RunState.CORRECT, {"correctness": correctness_payload})
    candidate_results = []
    for value, resident, timing_output, cost_report in executions:
        expected = value.prepared.expected
        candidate_root = root / "candidates" / expected.candidate
        trace = _capture_candidate_phase(
            candidate_root=candidate_root,
            expected=expected,
            compiled=value,
            resident=resident,
            mode=RunMode.TRACE,
            cost_report=cost_report,
        )
        if not np.array_equal(_execute(value.pallas_executable, resident), timing_output):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_POST_TRACE_OUTPUT_MISMATCH candidate={expected.candidate}"
            )
        counters = _capture_candidate_phase(
            candidate_root=candidate_root,
            expected=expected,
            compiled=value,
            resident=resident,
            mode=RunMode.COUNTERS,
            cost_report=cost_report,
        )
        if not np.array_equal(_execute(value.pallas_executable, resident), timing_output):
            raise ValueError(
                f"SEQAX_RESIDUAL_PROFILE_POST_COUNTER_OUTPUT_MISMATCH candidate={expected.candidate}"
            )
        candidate_results.append(
            SeqaxResidualCandidateResult(
                candidate=expected.candidate,
                distributed_schedule_sha256=value.prepared.plan.distributed_schedule_sha256,
                physical_schedule_sha256=value.prepared.plan.physical_schedule_sha256,
                pallas_source_sha256=value.prepared.plan.source_sha256(),
                pallas_manifest_sha256=_json_sha256(value.prepared.plan.manifest()),
                pallas_stablehlo_sha256=_sha256(candidate_root / "pallas_stablehlo.txt"),
                pallas_compiler_hlo_sha256=_sha256(candidate_root / "pallas_compiler_hlo.txt"),
                control_stablehlo_sha256=_sha256(candidate_root / "control_stablehlo.txt"),
                control_compiler_hlo_sha256=_sha256(candidate_root / "control_compiler_hlo.txt"),
                pallas_compiler_analysis_sha256=_sha256(
                    candidate_root / "pallas_compiler_analysis.json"
                ),
                control_compiler_analysis_sha256=_sha256(
                    candidate_root / "control_compiler_analysis.json"
                ),
                cost_model_sha256=_sha256(candidate_root / "cost_model.json"),
                timing_input_sha256=arrays_sha256(timing_host_inputs),
                timing_output_sha256=array_sha256(timing_output),
                correctness=correctness_by_candidate[expected.candidate],
                trace=trace,
                counters=counters,
            )
        )
    comparison = compare_residual_profiles(candidate_results[0], candidate_results[1])
    _write_json(root / "comparison.json", comparison.model_dump(mode="json"))
    result = SeqaxResidualProfileResult(
        profile_id=contract.profile_id,
        run_id=run_id,
        numerical_contract_id=contract.numerical_contract_id,
        runtime=runtime,
        devices=device_inventory,
        source_state_sha256=_sha256(root / "source_state.json"),
        source_manifest_sha256=_sha256(root / "source_manifest.json"),
        source_manifest=manifest,
        compiler_strategy_surface_sha256=_sha256(root / "compiler_strategy_surface.json"),
        candidates=tuple(candidate_results),
        comparison=comparison,
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    evidence_run.transition(
        RunState.COUNTERED,
        _ledger_payloads(
            contract=contract,
            result=result,
            root=root,
            require_accepted=False,
        )[-1][1],
    )
    evidence_run.seal("SEQAX_RESIDUAL_PROFILE_LEDGER_SIDECARS")
    _validate(root, contract, require_accepted=False)
    evidence_run.transition(
        RunState.ACCEPTED,
        {"result_sha256": _sha256(root / "result.json")},
    )
    evidence_run.seal("SEQAX_RESIDUAL_PROFILE_LEDGER_SIDECARS")
    receipt = SeqaxResidualProfileReceipt(
        status="passed",
        profile_id=contract.profile_id,
        run_id=run_id,
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(root / "ledger.sqlite"),
        artifacts=_artifact_manifest(root),
    )
    _write_json_atomic(root / "receipt.json", receipt.model_dump(mode="json"))
    return _validate(root, contract, require_accepted=True)


def validate_seqax_residual_profile(
    root: Path,
    trusted_contract: SeqaxResidualProfileContract,
) -> SeqaxResidualProfileResult:
    _reject_symlink_components(root)
    if root.is_symlink():
        raise ValueError("SEQAX_RESIDUAL_PROFILE_ROOT_SYMLINK")
    root = root.resolve()
    runtime = _runtime_identity()
    canonical = default_seqax_residual_profile_contract(trusted_contract.runtime)
    _validate_verifier_runtime()
    if trusted_contract != canonical or runtime.jax != trusted_contract.runtime.jax:
        raise ValueError("SEQAX_RESIDUAL_PROFILE_EXTERNAL_CONTRACT_MISMATCH")
    return _validate(root, trusted_contract, require_accepted=True)
