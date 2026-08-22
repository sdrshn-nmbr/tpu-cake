from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from tpu_cake.compiler_analysis import CompilerExecutableAnalysis, capture_compiler_analysis
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.matmul_collective_surface_executor import (
    EXECUTOR_SOURCE_PATH,
    WORKER_SOURCE_PATH,
    SurfaceCompileAbstractInputABI,
    SurfaceCompileCaptureEnvelope,
    SurfaceCompileDevice,
    SurfaceCompileExecutionAuthority,
    SurfaceCompileWorkerRequest,
    SurfaceCompileWorkerResult,
    _analysis_path,
    _canonical_contract,
    _compiler_environment,
    _executor_source_blob,
    _executor_source_sha256,
    _expected_capture_paths,
    _file_sha256,
    _metadata,
    _verifier_source_sha256,
    _worker_source_blob,
    _worker_source_sha256,
    _write_bytes_exclusive,
    _write_model_exclusive,
    validate_execution_authority,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    derive_matmul_collective_surface_design_report,
)
from tpu_cake.matmul_collective_surface_runner import (
    SURFACE_EXECUTABLE_DEPENDENCIES,
    CompileCaptureRecord,
    _runtime_identity,
    capture_surface_source_authority,
    derive_surface_input_identities,
    make_compile_capture_record,
)
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.runner import MatmulCollectiveStrategy
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def _validate_loaded_tpu_cake_sources(
    repository_root: Path,
    source_commit: str,
    source_blobs: dict[str, bytes],
) -> None:
    expected_blobs = {
        **{path: source_blobs[path] for path in SURFACE_EXECUTABLE_DEPENDENCIES},
        EXECUTOR_SOURCE_PATH.removeprefix("src/"): _executor_source_blob(
            repository_root, source_commit
        ),
        WORKER_SOURCE_PATH.removeprefix("src/"): _worker_source_blob(
            repository_root, source_commit
        ),
    }
    source_root = (repository_root / "src").resolve()
    observed_paths: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if name != "tpu_cake" and not name.startswith("tpu_cake."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LOADED_SOURCE_MISSING")
        path = Path(module_file)
        try:
            relative = path.resolve().relative_to(source_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "MATMUL_COLLECTIVE_SURFACE_COMPILE_LOADED_SOURCE_OUTSIDE_ROOT"
            ) from error
        if path.is_symlink() or relative not in expected_blobs:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LOADED_SOURCE_UNDECLARED")
        if path.read_bytes() != expected_blobs[relative]:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LOADED_SOURCE_HASH_MISMATCH")
        observed_paths.add(relative)
    worker_relative = WORKER_SOURCE_PATH.removeprefix("src/")
    running_worker = Path(__file__)
    if (
        running_worker.is_symlink()
        or running_worker.resolve() != (repository_root / WORKER_SOURCE_PATH).resolve()
        or running_worker.read_bytes() != expected_blobs[worker_relative]
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LOADED_SOURCE_HASH_MISMATCH")
    observed_paths.add(worker_relative)
    required = {
        "tpu_cake/__init__.py",
        EXECUTOR_SOURCE_PATH.removeprefix("src/"),
        worker_relative,
    }
    if not required <= observed_paths:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LOADED_SOURCE_INCOMPLETE")


def _device_inventory() -> tuple[SurfaceCompileDevice, ...]:
    return tuple(
        SurfaceCompileDevice(
            id=int(device.id),
            process_index=int(device.process_index),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        for device in jax.devices()
    )


def capture_execution_authority(
    repository_root: Path,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCompileExecutionAuthority, dict[str, bytes]]:
    source, source_blobs = capture_surface_source_authority(repository_root, contract)
    authority = SurfaceCompileExecutionAuthority(
        source=source,
        executor_source_sha256=_executor_source_sha256(repository_root, source.source_commit),
        worker_source_sha256=_worker_source_sha256(repository_root, source.source_commit),
        verifier_source_sha256=_verifier_source_sha256(repository_root, source.source_commit),
        project=_metadata("project/project-id"),
        zone=_metadata("instance/zone").rsplit("/", maxsplit=1)[-1],
        hostname=_metadata("instance/name"),
        numeric_project_id=_metadata("project/numeric-project-id"),
        instance_id=_metadata("instance/id"),
        instance_hostname=_metadata("instance/hostname"),
        machine_type=_metadata("instance/machine-type").rsplit("/", maxsplit=1)[-1],
        cpu_platform=_metadata("instance/cpu-platform"),
        backend=jax.default_backend(),
        runtime=_runtime_identity().model_dump(mode="python"),
        compiler_environment=_compiler_environment(contract),
        devices=_device_inventory(),
    )
    validate_execution_authority(authority, contract, source_blobs)
    if (
        Path(__file__).resolve() != (repository_root / WORKER_SOURCE_PATH).resolve()
        or _file_sha256(Path(__file__)) != authority.worker_source_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_RUNNING_WORKER_SOURCE_MISMATCH")
    _validate_loaded_tpu_cake_sources(repository_root, source.source_commit, source_blobs)
    return authority, source_blobs


def _compile_arm(
    contract: MatmulCollectiveSurfaceDesignContract,
    scenario_name: str,
    strategy: MatmulCollectiveStrategy,
    repetition: int,
) -> tuple[CompileCaptureRecord, CompilerExecutableAnalysis]:
    scenario = next(value for value in contract.scenarios if value.name == scenario_name)
    distributed = distributed_matmul_schedule(
        mesh_size=contract.mesh_size,
        m=scenario.m,
        k=scenario.k,
        n=scenario.n,
    )
    distributed.verify()
    physical = lower_distributed_matmul(
        distributed,
        tile=MatmulTile(scenario.tile_m, scenario.tile_n),
        collective_implementation=strategy.lowering_implementation(),
    )
    plan = lower_physical_matmul_to_pallas(physical)
    executable, mesh = plan.build(interpret=False)
    if plan.mesh_axis != "t":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_MESH_AXIS_MISMATCH")
    abstract_inputs = (
        jax.ShapeDtypeStruct(
            (scenario.m, scenario.k),
            jnp.bfloat16,
            sharding=NamedSharding(mesh, PartitionSpec(None, plan.mesh_axis)),
        ),
        jax.ShapeDtypeStruct(
            (scenario.k, scenario.n),
            jnp.bfloat16,
            sharding=NamedSharding(mesh, PartitionSpec(plan.mesh_axis, None)),
        ),
    )
    lowered = executable.lower(*abstract_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiled = lowered.compile()
    compiler_hlo = compiled.as_text()
    if not isinstance(compiler_hlo, str) or not compiler_hlo:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_HLO_UNAVAILABLE")
    arm = next(
        value
        for value in derive_matmul_collective_surface_design_report(contract).arms
        if value.scenario_name == scenario_name and value.strategy is strategy
    )
    input_contract = next(
        value.input_contract_sha256
        for value in derive_surface_input_identities(contract)
        if value.scenario_name == scenario_name
    )
    capture = make_compile_capture_record(
        scenario_name=scenario_name,
        strategy=strategy,
        repetition=repetition,
        input_contract_sha256=input_contract,
        distributed_schedule_sha256=arm.distributed_schedule_sha256,
        physical_schedule_sha256=arm.physical_schedule_sha256,
        pallas_source_sha256=arm.pallas_source_sha256,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
    )
    analysis = capture_compiler_analysis(
        compiled,
        stablehlo=capture.stablehlo.rstrip("\n"),
        compiler_hlo=capture.compiler_hlo.rstrip("\n"),
    )
    return capture, analysis


def execute_worker(root: Path, request_path: Path) -> SurfaceCompileWorkerResult:
    request = SurfaceCompileWorkerRequest.model_validate_json(request_path.read_text())
    cache_root = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    if (
        request.compilation_cache_schema != "isolated-empty-temporary-directory-v1"
        or not cache_root.is_absolute()
        or not cache_root.is_dir()
        or cache_root.is_symlink()
        or any(cache_root.iterdir())
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_CACHE_INVALID")
    started = json.loads((request_path.parent / "STARTED.json").read_text())
    if started != {
        "attempt_id": request.attempt_id,
        "invocation_nonce": request.invocation_nonce,
        "repetition": request.repetition,
        "state": "started",
    }:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_START_CLAIM_MISMATCH")
    authority = SurfaceCompileExecutionAuthority.model_validate_json(
        (root / "execution_authority.json").read_text()
    )
    observed_authority, source_blobs = capture_execution_authority(
        Path(request.contract.compilation_source_root), request.contract
    )
    if observed_authority != authority or request.authority_sha256 != authority.authority_sha256:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_AUTHORITY_MISMATCH")
    captures: list[SurfaceCompileCaptureEnvelope] = []
    for scenario in request.contract.scenarios:
        for strategy in request.contract.strategies:
            capture, analysis = _compile_arm(
                request.contract,
                scenario.name,
                strategy,
                request.repetition,
            )
            stablehlo_path, compiler_hlo_path = _expected_capture_paths(
                request.repetition, scenario.name, strategy
            )
            analysis_path = _analysis_path(request.repetition, scenario.name, strategy)
            _write_bytes_exclusive(root / stablehlo_path, capture.stablehlo.encode())
            _write_bytes_exclusive(root / compiler_hlo_path, capture.compiler_hlo.encode())
            _write_model_exclusive(root / analysis_path, analysis)
            captures.append(
                SurfaceCompileCaptureEnvelope(
                    capture=capture,
                    abstract_input_abi=SurfaceCompileAbstractInputABI(
                        lhs_shape=(scenario.m, scenario.k),
                        lhs_dtype="bfloat16",
                        lhs_sharding="PartitionSpec(None, 't')",
                        rhs_shape=(scenario.k, scenario.n),
                        rhs_dtype="bfloat16",
                        rhs_sharding="PartitionSpec('t', None)",
                        output_shape=(scenario.m, scenario.n),
                        output_dtype="float32",
                        output_sharding="PartitionSpec(None, 't')",
                    ),
                    stablehlo_path=stablehlo_path,
                    compiler_hlo_path=compiler_hlo_path,
                    compiler_analysis_path=analysis_path,
                    compiler_analysis=analysis,
                )
            )
    _validate_loaded_tpu_cake_sources(
        Path(request.contract.compilation_source_root),
        authority.source.source_commit,
        source_blobs,
    )
    result = SurfaceCompileWorkerResult(
        attempt_id=request.attempt_id,
        repetition=request.repetition,
        invocation_nonce=request.invocation_nonce,
        worker_pid=os.getpid(),
        authority_sha256=authority.authority_sha256,
        captures=tuple(captures),
    )
    _write_model_exclusive(root / f"repetition-{request.repetition}/result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    probe_command = commands.add_parser("probe")
    probe_command.add_argument("--contract", required=True, type=Path)
    probe_command.add_argument("--output", required=True, type=Path)
    worker_command = commands.add_parser("worker")
    worker_command.add_argument("--root", required=True, type=Path)
    worker_command.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "probe":
        contract = _canonical_contract(args.contract)
        authority, _ = capture_execution_authority(Path(contract.compilation_source_root), contract)
        _write_model_exclusive(args.output, authority)
    else:
        execute_worker(args.root, args.request)


if __name__ == "__main__":
    main()
