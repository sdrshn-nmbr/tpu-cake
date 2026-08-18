from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from xdsl.dialects.builtin import BFloat16Type, Float32Type, ModuleOp

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, SourceFileContract
from tpu_cake.dialects.tpu_schedule import BufferType, MxuEinsumOp
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.jax_lowering import JaxTensorContract
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.runner import _runtime_identity, _source_state
from tpu_cake.seqax_pallas_lowering import (
    SeqaxPallasPlan,
    _einsum_tiles,
    _pallas_einsum,
    lower_seqax_physical_to_pallas,
)
from tpu_cake.seqax_pallas_runner import (
    _compiler_hlo,
    _errors,
    _physical_collective_counts,
    _validate_compiled_program,
)
from tpu_cake.seqax_pallas_search import (
    SeqaxPallasCandidateCorrectness,
    SeqaxPallasCandidatePlan,
    SeqaxPallasDevice,
    SeqaxPallasPrimitiveObservation,
    SeqaxPallasRoundObservation,
    SeqaxPallasSearchCandidate,
    SeqaxPallasSearchContract,
    SeqaxPallasSearchReceipt,
    SeqaxPallasSearchResult,
    candidate_statistics,
    candidate_tiles,
    confirmation_statistics,
    execution_orders,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_runner import SEQAX_OUTPUT_ATOL, SEQAX_OUTPUT_RTOL
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)


@dataclass(frozen=True)
class PreparedCandidate:
    candidate: SeqaxPallasSearchCandidate
    physical: ModuleOp
    plan: SeqaxPallasPlan
    tiles: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class CompiledCandidate:
    prepared: PreparedCandidate
    compiled: Any
    mesh: Any
    stablehlo: str
    compiler_hlo: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _search_source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "contracts.py",
        package / "identity.py",
        package / "ledger.py",
        package / "runner.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_pallas_runner.py",
        package / "seqax_pallas_search.py",
        package / "seqax_pallas_search_runner.py",
        package / "seqax_physical_lowering.py",
        package / "seqax_runner.py",
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def _save_primitive_input(path: Path, value: np.ndarray, dtype_name: str) -> None:
    if dtype_name == "bf16":
        if value.dtype != np.dtype(jnp.bfloat16):
            raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_STORAGE_DTYPE dtype={value.dtype}")
        _save_array(path, value.view(np.uint16))
        return
    if dtype_name == "f32":
        if value.dtype != np.dtype(np.float32):
            raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_STORAGE_DTYPE dtype={value.dtype}")
        _save_array(path, value)
        return
    raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_STORAGE_UNSUPPORTED dtype={dtype_name}")


def _load_primitive_input(path: Path, dtype_name: str) -> np.ndarray:
    stored = _load_array(path)
    if dtype_name == "bf16":
        if stored.dtype != np.dtype(np.uint16):
            raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_STORAGE_DTYPE dtype={stored.dtype}")
        return stored.view(np.dtype(jnp.bfloat16))
    if dtype_name == "f32":
        if stored.dtype != np.dtype(np.float32):
            raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_STORAGE_DTYPE dtype={stored.dtype}")
        return stored
    raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_STORAGE_UNSUPPORTED dtype={dtype_name}")


def _close_ledger(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    for sidecar in (
        path.with_name(f"{path.name}-shm"),
        path.with_name(f"{path.name}-wal"),
    ):
        if sidecar.exists():
            raise ValueError(f"SEQAX_PALLAS_SEARCH_LEDGER_SIDECAR path={sidecar}")


def _require_safe_new_root(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == value or root in value.parents for value in protected):
        raise ValueError(f"SEQAX_PALLAS_SEARCH_UNSAFE_ROOT path={root}")
    if root.exists():
        raise ValueError(f"SEQAX_PALLAS_SEARCH_ROOT_EXISTS path={root}")


def _require_clean_repository(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_PALLAS_SEARCH_SOURCE_DIRTY status={status}")


def _preflight_existing_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_PALLAS_SEARCH_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_PALLAS_SEARCH_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_PALLAS_SEARCH_HARDLINK path={path}")


def prepare_seqax_pallas_candidates(
    contract: SeqaxPallasSearchContract,
) -> tuple[ModuleOp, tuple[PreparedCandidate, ...]]:
    distributed = seqax_forward_schedule(**contract.parameters)
    full_physical = lower_seqax_forward_to_physical(distributed).module
    full_tiles = _einsum_tiles(full_physical)
    prepared = []
    for candidate in contract.candidates:
        tiles = candidate_tiles(full_tiles, candidate)
        changed = sum(left != right for left, right in zip(full_tiles, tiles, strict=True))
        if changed != candidate.expected_changed_regions:
            raise ValueError(
                "SEQAX_PALLAS_SEARCH_CHANGED_REGION_MISMATCH "
                f"candidate={candidate.name} expected={candidate.expected_changed_regions} "
                f"observed={changed}"
            )
        physical = lower_seqax_forward_to_physical(
            distributed,
            einsum_tiles=tiles,
        ).module
        prepared.append(
            PreparedCandidate(
                candidate=candidate,
                physical=physical,
                plan=lower_seqax_physical_to_pallas(distributed, physical),
                tiles=tiles,
            )
        )
    schedules = tuple(value.plan.physical_schedule_sha256 for value in prepared)
    sources = tuple(value.plan.source_sha256() for value in prepared)
    if len(set(schedules)) != len(prepared) or len(set(sources)) != len(prepared):
        raise ValueError("SEQAX_PALLAS_SEARCH_CANDIDATES_NOT_DISTINCT")
    return distributed, tuple(prepared)


def _compile_candidate(
    value: PreparedCandidate,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
) -> CompiledCandidate:
    executable, mesh = value.plan.build(interpret=False, devices=devices)
    resident_inputs = tuple(
        jax.device_put(
            jnp.asarray(array),
            NamedSharding(mesh, tensor.partition_spec()),
        )
        for array, tensor in zip(host_inputs, value.plan.input_contracts, strict=True)
    )
    lowered = executable.lower(*resident_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiler_hlo = _compiler_hlo(lowered)
    all_gather_count, reduce_scatter_count = _physical_collective_counts(value.physical)
    _validate_compiled_program(
        stablehlo,
        compiler_hlo,
        pallas_region_count=value.plan.pallas_region_count,
        all_gather_count=all_gather_count,
        reduce_scatter_count=reduce_scatter_count,
    )
    return CompiledCandidate(
        prepared=value,
        compiled=lowered.compile(),
        mesh=mesh,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
    )


def _names(buffer: BufferType) -> tuple[str, ...]:
    return tuple(value.data for value in buffer.shape.dimensions)


def _primitive_dtype(operation: MxuEinsumOp) -> tuple[jnp.dtype, str]:
    lhs_type = operation.lhs.type
    if not isinstance(lhs_type, BufferType):
        raise TypeError("Seqax Pallas primitive input is not a physical buffer")
    if isinstance(lhs_type.storage.element_type, BFloat16Type):
        return jnp.bfloat16, "bf16"
    if isinstance(lhs_type.storage.element_type, Float32Type):
        return jnp.float32, "f32"
    raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_DTYPE dtype={lhs_type.storage.element_type}")


def _numpy_einsum(
    operation: MxuEinsumOp,
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    lhs_type = operation.lhs.type
    rhs_type = operation.rhs.type
    output_type = operation.accumulator.type
    if not all(isinstance(value, BufferType) for value in (lhs_type, rhs_type, output_type)):
        raise TypeError("Seqax Pallas primitive requires physical buffer operands")
    lhs_names = _names(lhs_type)
    rhs_names = _names(rhs_type)
    output_names = _names(output_type)
    contracting_names = tuple(value.data for value in operation.contracting_dimensions)
    extents: dict[str, int] = {}
    for name, extent in zip(lhs_names, lhs.shape, strict=True):
        extents[name] = extent
    for name, extent in zip(rhs_names, rhs.shape, strict=True):
        if name in extents and extents[name] != extent:
            raise ValueError(f"SEQAX_PALLAS_SEARCH_PRIMITIVE_EXTENT_MISMATCH dimension={name}")
        extents[name] = extent
    output = np.empty(tuple(extents[name] for name in output_names), dtype=np.float32)
    contraction_shape = tuple(extents[name] for name in contracting_names)
    for output_index in np.ndindex(output.shape):
        indices = dict(zip(output_names, output_index, strict=True))
        accumulator = 0.0
        for contraction_index in np.ndindex(contraction_shape):
            indices.update(zip(contracting_names, contraction_index, strict=True))
            accumulator += float(lhs[tuple(indices[name] for name in lhs_names)]) * float(
                rhs[tuple(indices[name] for name in rhs_names)]
            )
        output[output_index] = accumulator
    return output


def _primitive_signature(operation: MxuEinsumOp) -> tuple[object, ...]:
    lhs_type = operation.lhs.type
    rhs_type = operation.rhs.type
    output_type = operation.accumulator.type
    if not all(isinstance(value, BufferType) for value in (lhs_type, rhs_type, output_type)):
        raise TypeError("Seqax Pallas primitive requires physical buffer operands")
    return (
        lhs_type.storage.get_shape(),
        _names(lhs_type),
        rhs_type.storage.get_shape(),
        _names(rhs_type),
        output_type.storage.get_shape(),
        _names(output_type),
        tuple(value.data for value in operation.contracting_dimensions),
        operation.tile_m.data,
        operation.tile_k.data,
        operation.tile_n.data,
        str(lhs_type.storage.element_type),
    )


def _regenerate_primitive_operands(
    operation: MxuEinsumOp,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    lhs_type = operation.lhs.type
    rhs_type = operation.rhs.type
    if not isinstance(lhs_type, BufferType) or not isinstance(rhs_type, BufferType):
        raise TypeError("Seqax Pallas primitive requires physical buffer operands")
    input_dtype, _name = _primitive_dtype(operation)
    generator = np.random.default_rng(seed)
    lhs = generator.normal(
        scale=0.08,
        size=lhs_type.storage.get_shape(),
    ).astype(np.float32)
    rhs = generator.normal(
        scale=0.08,
        size=rhs_type.storage.get_shape(),
    ).astype(np.float32)
    return lhs.astype(input_dtype), rhs.astype(input_dtype)


def _primitive_executor(
    operation: MxuEinsumOp,
    schedule_sha256: str,
):
    @jax.jit
    def execute(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return _pallas_einsum(
            operation,
            lhs,
            rhs,
            interpret=False,
            schedule_sha256_value=schedule_sha256,
        )

    return execute


def _primitive_observations(
    root: Path,
    contract: SeqaxPallasSearchContract,
    candidates: tuple[PreparedCandidate, ...],
) -> tuple[SeqaxPallasPrimitiveObservation, ...]:
    baseline = candidates[0]
    broadest = next(value for value in candidates if value.candidate.name == "split-kn")
    changed = tuple(
        operation
        for index, operation in enumerate(
            value for value in broadest.physical.walk() if isinstance(value, MxuEinsumOp)
        )
        if baseline.tiles[index] != broadest.tiles[index]
    )
    unique: dict[tuple[object, ...], MxuEinsumOp] = {}
    for operation in changed:
        unique.setdefault(_primitive_signature(operation), operation)
    if len(changed) != 7 or len(unique) != 6:
        raise ValueError(
            "SEQAX_PALLAS_SEARCH_PRIMITIVE_SURFACE_MISMATCH "
            f"regions={len(changed)} signatures={len(unique)}"
        )
    observations = []
    for shape_index, operation in enumerate(unique.values()):
        lhs_type = operation.lhs.type
        rhs_type = operation.rhs.type
        output_type = operation.accumulator.type
        if not all(isinstance(value, BufferType) for value in (lhs_type, rhs_type, output_type)):
            raise TypeError("Seqax Pallas primitive requires physical buffer operands")
        input_dtype, dtype_name = _primitive_dtype(operation)

        execute = _primitive_executor(
            operation,
            broadest.plan.physical_schedule_sha256,
        )

        for seed in contract.correctness_seeds:
            lhs, rhs = _regenerate_primitive_operands(operation, seed)
            lhs_device = jnp.asarray(lhs, dtype=input_dtype)
            rhs_device = jnp.asarray(rhs, dtype=input_dtype)
            actual_device = execute(lhs_device, rhs_device)
            actual_device.block_until_ready()
            actual = np.asarray(actual_device)
            reference = _numpy_einsum(
                operation,
                np.asarray(lhs_device),
                np.asarray(rhs_device),
            )
            maximum_absolute_error, maximum_relative_error = _errors(actual, reference)
            passed = bool(
                np.allclose(
                    actual,
                    reference,
                    atol=contract.primitive_absolute_tolerance,
                    rtol=contract.primitive_relative_tolerance,
                )
            )
            case_root = root / f"shape-{shape_index:02d}" / str(seed)
            _save_primitive_input(
                case_root / "lhs.npy",
                np.asarray(lhs_device),
                dtype_name,
            )
            _save_primitive_input(
                case_root / "rhs.npy",
                np.asarray(rhs_device),
                dtype_name,
            )
            _save_array(case_root / "actual.npy", actual)
            _save_array(case_root / "reference.npy", reference)
            observations.append(
                SeqaxPallasPrimitiveObservation(
                    shape_index=shape_index,
                    seed=seed,
                    lhs_shape=lhs_type.storage.get_shape(),
                    lhs_names=_names(lhs_type),
                    rhs_shape=rhs_type.storage.get_shape(),
                    rhs_names=_names(rhs_type),
                    output_shape=actual.shape,
                    output_names=_names(output_type),
                    contracting_dimensions=tuple(
                        value.data for value in operation.contracting_dimensions
                    ),
                    tiles=(
                        operation.tile_m.data,
                        operation.tile_k.data,
                        operation.tile_n.data,
                    ),
                    dtype=dtype_name,
                    lhs_sha256=array_sha256(np.asarray(lhs_device)),
                    rhs_sha256=array_sha256(np.asarray(rhs_device)),
                    actual_sha256=array_sha256(actual),
                    reference_sha256=array_sha256(reference),
                    maximum_absolute_error=maximum_absolute_error,
                    maximum_relative_error=maximum_relative_error,
                    passed=passed,
                )
            )
    if len(observations) != 30 or not all(value.passed for value in observations):
        raise ValueError("SEQAX_PALLAS_SEARCH_PRIMITIVE_ORACLE_FAILED")
    return tuple(observations)


def _resident_inputs(
    host_inputs: tuple[np.ndarray, ...],
    compiled: CompiledCandidate,
) -> tuple[jax.Array, ...]:
    return tuple(
        jax.device_put(
            jnp.asarray(value),
            NamedSharding(compiled.mesh, contract.partition_spec()),
        )
        for value, contract in zip(
            host_inputs,
            compiled.prepared.plan.input_contracts,
            strict=True,
        )
    )


def _execute(
    compiled: CompiledCandidate,
    inputs: tuple[jax.Array, ...],
) -> np.ndarray:
    outputs = compiled.compiled(*inputs)
    jax.block_until_ready(outputs)
    if len(outputs) != 1:
        raise ValueError("SEQAX_PALLAS_SEARCH_OUTPUT_COUNT_MISMATCH")
    return np.asarray(outputs[0])


def _candidate_correctness(
    root: Path,
    contract: SeqaxPallasSearchContract,
    candidates: tuple[CompiledCandidate, ...],
) -> tuple[SeqaxPallasCandidateCorrectness, ...]:
    inputs_by_seed = []
    oracle_by_seed = []
    outputs_by_name: dict[str, list[np.ndarray]] = {
        value.prepared.candidate.name: [] for value in candidates
    }
    for seed in contract.correctness_seeds:
        host_inputs = tuple(
            np.asarray(value) for value in seqax_forward_inputs(seed=seed, **contract.parameters)
        )
        oracle = np.asarray(seqax_forward_canonical_reference(host_inputs, **contract.parameters))
        seed_root = root / str(seed)
        for index, value in enumerate(host_inputs):
            _save_array(seed_root / "inputs" / f"{index:02d}.npy", value)
        _save_array(seed_root / "cpu_oracle.npy", oracle)
        inputs_by_seed.append(host_inputs)
        oracle_by_seed.append(oracle)
        for candidate in candidates:
            actual = _execute(candidate, _resident_inputs(host_inputs, candidate))
            _save_array(
                seed_root / "outputs" / f"{candidate.prepared.candidate.name}.npy",
                actual,
            )
            outputs_by_name[candidate.prepared.candidate.name].append(actual)
    baseline_outputs = outputs_by_name[contract.baseline]
    baseline_hashes = tuple(array_sha256(value) for value in baseline_outputs)
    correctness = []
    for candidate in candidates:
        name = candidate.prepared.candidate.name
        outputs = outputs_by_name[name]
        exact_parity = all(
            actual.shape == baseline.shape
            and actual.dtype == baseline.dtype
            and np.array_equal(actual, baseline)
            for actual, baseline in zip(outputs, baseline_outputs, strict=True)
        )
        if not exact_parity:
            raise ValueError(f"SEQAX_PALLAS_SEARCH_BASELINE_PARITY_FAILED candidate={name}")
        errors = tuple(
            _errors(actual, oracle) for actual, oracle in zip(outputs, oracle_by_seed, strict=True)
        )
        correctness.append(
            SeqaxPallasCandidateCorrectness(
                name=name,
                input_sha256=tuple(arrays_sha256(values) for values in inputs_by_seed),
                output_sha256=tuple(array_sha256(value) for value in outputs),
                baseline_output_sha256=baseline_hashes,
                exact_baseline_parity=True,
                cpu_oracle_sha256=tuple(array_sha256(value) for value in oracle_by_seed),
                cpu_oracle_maximum_absolute_error=tuple(value[0] for value in errors),
                cpu_oracle_maximum_relative_error=tuple(value[1] for value in errors),
                cpu_oracle_passed=tuple(
                    bool(
                        np.allclose(
                            actual,
                            oracle,
                            atol=SEQAX_OUTPUT_ATOL,
                            rtol=SEQAX_OUTPUT_RTOL,
                        )
                    )
                    for actual, oracle in zip(outputs, oracle_by_seed, strict=True)
                ),
            )
        )
    return tuple(correctness)


def _measure(
    candidate: CompiledCandidate,
    inputs: tuple[jax.Array, ...],
    iterations: int,
) -> tuple[int, ...]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        jax.block_until_ready(candidate.compiled(*inputs))
        samples.append(time.perf_counter_ns() - started)
    return tuple(samples)


def _timing_observations(
    contract: SeqaxPallasSearchContract,
    candidates: dict[str, CompiledCandidate],
    inputs: dict[str, tuple[jax.Array, ...]],
    orders: tuple[tuple[str, ...], ...],
) -> tuple[SeqaxPallasRoundObservation, ...]:
    observations = []
    for round_index, order in enumerate(orders):
        for position, name in enumerate(order):
            samples = _measure(
                candidates[name],
                inputs[name],
                contract.measured_iterations,
            )
            observations.append(
                SeqaxPallasRoundObservation(
                    round_index=round_index,
                    position=position,
                    candidate=name,
                    samples_ns=samples,
                    median_ns=float(statistics.median(samples)),
                )
            )
    return tuple(observations)


def run_seqax_pallas_search(
    root: Path,
    contract: SeqaxPallasSearchContract,
) -> SeqaxPallasSearchResult:
    root = root.resolve()
    _require_safe_new_root(root)
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
    runtime = _runtime_identity()
    if runtime != contract.runtime:
        raise ValueError("SEQAX_PALLAS_SEARCH_RUNTIME_MISMATCH")
    devices = tuple(jax.devices())
    if (
        jax.default_backend() != contract.backend
        or len(devices) != contract.device_count
        or any(device.platform != "tpu" for device in devices)
        or any(device.device_kind not in {"TPU7x", "TPU v7x"} for device in devices)
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_DEVICE_MISMATCH")
    root.mkdir(parents=True)
    _write_json(
        root / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    _source_state(repository_root, root)
    source_state_sha256 = _sha256(root / "source_state.json")
    ledger_path = root / "ledger.sqlite"
    run_id = semantic_sha256("seqax-pallas-tile-search-run-v1", contract.search_id)
    with ExperimentLedger(ledger_path) as ledger:
        ledger.create(run_id, {"search_id": contract.search_id})

    distributed, prepared = prepare_seqax_pallas_candidates(contract)
    _write_text(root / "distributed.xdsl", canonical_text(distributed))
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.VERIFIED,
            {
                "candidate_schedules": {
                    value.candidate.name: value.plan.physical_schedule_sha256 for value in prepared
                }
            },
        )
    plan_records = []
    for value in prepared:
        plan_root = root / "plans" / value.candidate.name
        _write_text(plan_root / "physical.xdsl", canonical_text(value.physical))
        _write_text(plan_root / "lowered_pallas.py", value.plan.render_executable_source())
        _write_json(plan_root / "plan_manifest.json", value.plan.manifest())
        _write_json(plan_root / "tiles.json", value.tiles)
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.LOWERED,
            {
                "pallas_sources": {
                    value.candidate.name: value.plan.source_sha256() for value in prepared
                }
            },
        )

    compile_host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **contract.parameters,
        )
    )
    compiled = tuple(_compile_candidate(value, compile_host_inputs, devices) for value in prepared)
    for value in compiled:
        plan_root = root / "plans" / value.prepared.candidate.name
        _write_text(plan_root / "stablehlo.txt", value.stablehlo + "\n")
        _write_text(plan_root / "compiler_hlo.txt", value.compiler_hlo + "\n")
        plan_records.append(
            SeqaxPallasCandidatePlan(
                name=value.prepared.candidate.name,
                policy=value.prepared.candidate.policy,
                tiles=value.prepared.tiles,
                changed_region_count=value.prepared.candidate.expected_changed_regions,
                physical_schedule_sha256=value.prepared.plan.physical_schedule_sha256,
                pallas_source_sha256=value.prepared.plan.source_sha256(),
                stablehlo_sha256=_sha256(plan_root / "stablehlo.txt"),
                compiler_hlo_sha256=_sha256(plan_root / "compiler_hlo.txt"),
            )
        )
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.COMPILED,
            {"plans": [value.model_dump(mode="json") for value in plan_records]},
        )

    primitive = _primitive_observations(root / "primitive", contract, prepared)
    correctness = _candidate_correctness(
        root / "correctness",
        contract,
        compiled,
    )
    _write_json(
        root / "primitive_observations.json",
        [value.model_dump(mode="json") for value in primitive],
    )
    _write_json(
        root / "correctness.json",
        [value.model_dump(mode="json") for value in correctness],
    )
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.CORRECT,
            {
                "primitive_case_count": len(primitive),
                "candidate_output_sha256": {
                    value.name: value.output_sha256 for value in correctness
                },
            },
        )

    timing_inputs = {
        value.prepared.candidate.name: _resident_inputs(compile_host_inputs, value)
        for value in compiled
    }
    compiled_by_name = {value.prepared.candidate.name: value for value in compiled}
    for name, value in compiled_by_name.items():
        for _ in range(contract.warmup_iterations):
            jax.block_until_ready(value.compiled(*timing_inputs[name]))
    orders = execution_orders(contract)
    rounds = _timing_observations(
        contract,
        compiled_by_name,
        timing_inputs,
        orders,
    )
    statistics_by_candidate = candidate_statistics(contract, rounds)
    promotable = tuple(value for value in statistics_by_candidate if value.promotable)
    provisional_winner = (
        min(promotable, key=lambda value: value.median_round_ns).name if promotable else None
    )
    confirmation_rounds: tuple[SeqaxPallasRoundObservation, ...] = ()
    confirmation = None
    winner = None
    if provisional_winner is not None:
        confirmation_orders = tuple(
            (
                (contract.baseline, provisional_winner)
                if round_index % 2 == 0
                else (provisional_winner, contract.baseline)
            )
            for round_index in range(contract.confirmation_rounds)
        )
        confirmation_rounds = _timing_observations(
            contract,
            compiled_by_name,
            timing_inputs,
            confirmation_orders,
        )
        confirmation = confirmation_statistics(
            contract,
            provisional_winner,
            confirmation_rounds,
        )
        winner = provisional_winner if confirmation.confirmed else None
    _write_json(
        root / "rounds.json",
        [value.model_dump(mode="json") for value in rounds],
    )
    _write_json(
        root / "confirmation_rounds.json",
        [value.model_dump(mode="json") for value in confirmation_rounds],
    )
    result = SeqaxPallasSearchResult(
        search_id=contract.search_id,
        baseline=contract.baseline,
        runtime=runtime,
        device_kind="TPU7x",
        device_count=len(devices),
        devices=tuple(
            SeqaxPallasDevice(
                id=device.id,
                process_index=device.process_index,
                platform=device.platform,
                device_kind=device.device_kind,
            )
            for device in devices
        ),
        timing_input_sha256=arrays_sha256(compile_host_inputs),
        source_state_sha256=source_state_sha256,
        search_source_manifest=_search_source_manifest(),
        plans=tuple(plan_records),
        primitive_observations=primitive,
        correctness=correctness,
        execution_orders=orders,
        rounds=rounds,
        candidates=statistics_by_candidate,
        provisional_winner=provisional_winner,
        confirmation_rounds=confirmation_rounds,
        confirmation=confirmation,
        winner=winner,
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.TIMED,
            {
                "round_count": len(rounds),
                "confirmation_round_count": len(confirmation_rounds),
                "provisional_winner": provisional_winner,
                "winner": winner,
            },
        )
    _close_ledger(ledger_path)
    _validate_seqax_pallas_search(
        root,
        contract,
        require_accepted=False,
    )
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.ACCEPTED,
            {
                "result_sha256": _sha256(root / "result.json"),
                "winner": winner,
            },
        )
    _close_ledger(ledger_path)
    _build_receipt(root, contract)
    return validate_seqax_pallas_search(root, contract)


def _load_array(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"SEQAX_PALLAS_SEARCH_ARRAY_MISSING path={path}")
    return np.load(path, allow_pickle=False)


def _compiler_tile_metadata(
    compiler_hlo: str,
) -> tuple[tuple[int, str, int, int, int], ...]:
    matches = re.findall(
        r"(?ms)^\s*(?:ROOT\s+)?pallas_call\.\d+\s*=\s*[^\n]+\s+custom-call\([^\n]*\),\s*"
        r'custom_call_target="tpu_custom_call"[^\n]*frontend_attributes=\{kernel_metadata=\{\s*'
        r'"region_index"\s*:\s*(\d+)\s*,\s*'
        r'"schedule_sha256"\s*:\s*"([0-9a-f]{64})"\s*,\s*'
        r'"tile_k"\s*:\s*(\d+)\s*,\s*'
        r'"tile_m"\s*:\s*(\d+)\s*,\s*'
        r'"tile_n"\s*:\s*(\d+)\s*\}\},\s*backend_config=',
        compiler_hlo,
    )
    records = tuple(
        (int(region), schedule, int(tile_m), int(tile_k), int(tile_n))
        for region, schedule, tile_k, tile_m, tile_n in matches
    )
    region_indices = tuple(record[0] for record in records)
    if len(set(region_indices)) != len(region_indices):
        raise ValueError("SEQAX_PALLAS_SEARCH_DUPLICATE_COMPILER_REGION")
    return tuple(sorted(records, key=lambda record: record[0]))


def _expected_files(
    root: Path,
    contract: SeqaxPallasSearchContract,
    primitive: tuple[SeqaxPallasPrimitiveObservation, ...],
    receipt_present: bool,
) -> set[Path]:
    expected = {
        root / "contract.json",
        root / "source_state.json",
        root / "source_diff.patch",
        root / "ledger.sqlite",
        root / "distributed.xdsl",
        root / "primitive_observations.json",
        root / "correctness.json",
        root / "rounds.json",
        root / "confirmation_rounds.json",
        root / "result.json",
    }
    for candidate in contract.candidates:
        candidate_root = root / "plans" / candidate.name
        expected.update(
            candidate_root / name
            for name in (
                "physical.xdsl",
                "lowered_pallas.py",
                "plan_manifest.json",
                "tiles.json",
                "stablehlo.txt",
                "compiler_hlo.txt",
            )
        )
    for observation in primitive:
        case_root = (
            root / "primitive" / f"shape-{observation.shape_index:02d}" / str(observation.seed)
        )
        expected.update(
            case_root / name for name in ("lhs.npy", "rhs.npy", "actual.npy", "reference.npy")
        )
    for seed in contract.correctness_seeds:
        seed_root = root / "correctness" / str(seed)
        expected.update(seed_root / "inputs" / f"{index:02d}.npy" for index in range(13))
        expected.add(seed_root / "cpu_oracle.npy")
        expected.update(
            seed_root / "outputs" / f"{candidate.name}.npy" for candidate in contract.candidates
        )
    if receipt_present:
        expected.add(root / "receipt.json")
    return {path.resolve() for path in expected}


def _validate_closed_world(root: Path, expected: set[Path]) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_PALLAS_SEARCH_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_PALLAS_SEARCH_HARDLINK path={path}")
    observed = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if observed != expected:
        missing = sorted(str(path) for path in expected - observed)
        extra = sorted(str(path) for path in observed - expected)
        raise ValueError(
            f"SEQAX_PALLAS_SEARCH_CLOSED_WORLD_MISMATCH missing={missing} extra={extra}"
        )


def _artifact_role(relative: Path) -> ArtifactRole:
    path = relative.as_posix()
    if path == "contract.json":
        return ArtifactRole.SEARCH_CONTRACT
    if path == "result.json":
        return ArtifactRole.SEARCH_RESULT
    if path == "ledger.sqlite":
        return ArtifactRole.EXECUTION_LEDGER
    if path == "source_state.json":
        return ArtifactRole.SOURCE_STATE
    if path == "source_diff.patch":
        return ArtifactRole.SOURCE_DIFF
    if path == "distributed.xdsl":
        return ArtifactRole.DISTRIBUTED_IR
    if path.endswith("/physical.xdsl"):
        return ArtifactRole.PHYSICAL_IR
    if path.endswith("/lowered_pallas.py"):
        return ArtifactRole.PALLAS_SOURCE
    if path.endswith("/plan_manifest.json"):
        return ArtifactRole.PLAN_MANIFEST
    if path.endswith("/stablehlo.txt"):
        return ArtifactRole.STABLEHLO
    if path.endswith("/compiler_hlo.txt"):
        return ArtifactRole.COMPILER_HLO
    if "/inputs/" in path or path.endswith(("/lhs.npy", "/rhs.npy")):
        return ArtifactRole.CORRECTNESS_INPUT
    if path.endswith(("/cpu_oracle.npy", "/reference.npy")):
        return ArtifactRole.ORACLE_OUTPUT
    if "/outputs/" in path or path.endswith("/actual.npy"):
        return ArtifactRole.CORRECTNESS_OUTPUT
    if path in {"rounds.json", "confirmation_rounds.json"}:
        return ArtifactRole.TIMING_SAMPLES
    return ArtifactRole.SEARCH_EVIDENCE


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    artifacts = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root)
        if relative.as_posix() == "receipt.json":
            continue
        artifacts.append(
            ArtifactReference(
                path=relative.as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
                role=_artifact_role(relative),
            )
        )
    return tuple(artifacts)


def _build_receipt(
    root: Path,
    contract: SeqaxPallasSearchContract,
) -> SeqaxPallasSearchReceipt:
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        raise ValueError("SEQAX_PALLAS_SEARCH_RECEIPT_ALREADY_EXISTS")
    receipt = SeqaxPallasSearchReceipt(
        search_id=contract.search_id,
        status="passed",
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(root / "ledger.sqlite"),
        artifacts=_artifact_manifest(root),
    )
    _write_json(receipt_path, receipt.model_dump(mode="json"))
    return receipt


def _validate_primitive_replay(
    root: Path,
    contract: SeqaxPallasSearchContract,
    prepared: tuple[PreparedCandidate, ...],
    saved: tuple[SeqaxPallasPrimitiveObservation, ...],
) -> None:
    broadest = next(value for value in prepared if value.candidate.name == "split-kn")
    baseline = next(value for value in prepared if value.candidate.name == contract.baseline)
    operations = tuple(
        operation
        for index, operation in enumerate(
            value for value in broadest.physical.walk() if isinstance(value, MxuEinsumOp)
        )
        if baseline.tiles[index] != broadest.tiles[index]
    )
    unique: dict[tuple[object, ...], MxuEinsumOp] = {}
    for operation in operations:
        unique.setdefault(_primitive_signature(operation), operation)
    expected_keys = tuple(
        (shape_index, seed)
        for shape_index in range(len(unique))
        for seed in contract.correctness_seeds
    )
    observed_keys = tuple((value.shape_index, value.seed) for value in saved)
    if len(unique) != 6 or observed_keys != expected_keys:
        raise ValueError("SEQAX_PALLAS_SEARCH_PRIMITIVE_CASE_SET_MISMATCH")
    for observation in saved:
        operation = tuple(unique.values())[observation.shape_index]
        lhs_type = operation.lhs.type
        rhs_type = operation.rhs.type
        output_type = operation.accumulator.type
        if not all(isinstance(value, BufferType) for value in (lhs_type, rhs_type, output_type)):
            raise TypeError("Seqax Pallas primitive requires physical buffer operands")
        case_root = root / f"shape-{observation.shape_index:02d}" / str(observation.seed)
        _input_dtype, expected_dtype = _primitive_dtype(operation)
        lhs = _load_primitive_input(case_root / "lhs.npy", expected_dtype)
        rhs = _load_primitive_input(case_root / "rhs.npy", expected_dtype)
        actual = _load_array(case_root / "actual.npy")
        reference = _load_array(case_root / "reference.npy")
        expected_lhs, expected_rhs = _regenerate_primitive_operands(
            operation,
            observation.seed,
        )
        expected_reference = _numpy_einsum(operation, lhs, rhs)
        expected_tiles = (
            operation.tile_m.data,
            operation.tile_k.data,
            operation.tile_n.data,
        )
        if (
            tuple(lhs.shape) != lhs_type.storage.get_shape()
            or tuple(rhs.shape) != rhs_type.storage.get_shape()
            or tuple(actual.shape) != output_type.storage.get_shape()
            or tuple(reference.shape) != output_type.storage.get_shape()
            or lhs.dtype != expected_lhs.dtype
            or rhs.dtype != expected_rhs.dtype
            or actual.dtype != np.dtype(np.float32)
            or reference.dtype != np.dtype(np.float32)
            or not np.array_equal(lhs, expected_lhs)
            or not np.array_equal(rhs, expected_rhs)
            or observation.lhs_shape != tuple(lhs.shape)
            or observation.rhs_shape != tuple(rhs.shape)
            or observation.output_shape != tuple(actual.shape)
            or observation.lhs_names != _names(lhs_type)
            or observation.rhs_names != _names(rhs_type)
            or observation.output_names != _names(output_type)
            or observation.contracting_dimensions
            != tuple(value.data for value in operation.contracting_dimensions)
            or observation.dtype != expected_dtype
            or observation.tiles != expected_tiles
        ):
            raise ValueError("SEQAX_PALLAS_SEARCH_PRIMITIVE_ABI_MISMATCH")
        if (
            observation.lhs_sha256 != array_sha256(lhs)
            or observation.rhs_sha256 != array_sha256(rhs)
            or observation.actual_sha256 != array_sha256(actual)
            or observation.reference_sha256 != array_sha256(reference)
            or not np.array_equal(reference, expected_reference)
        ):
            raise ValueError("SEQAX_PALLAS_SEARCH_PRIMITIVE_ARRAY_REPLAY_MISMATCH")
        absolute, relative = _errors(actual, expected_reference)
        passed = bool(
            np.allclose(
                actual,
                expected_reference,
                atol=contract.primitive_absolute_tolerance,
                rtol=contract.primitive_relative_tolerance,
            )
        )
        if (
            not math.isclose(
                absolute,
                observation.maximum_absolute_error,
                rel_tol=0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                relative,
                observation.maximum_relative_error,
                rel_tol=0,
                abs_tol=1e-15,
            )
            or observation.passed is not passed
            or not passed
        ):
            raise ValueError("SEQAX_PALLAS_SEARCH_PRIMITIVE_VERDICT_MISMATCH")


def _validate_output_abi(
    output: np.ndarray,
    contract: JaxTensorContract,
    candidate: str,
) -> None:
    expected_shape = tuple(size for _, size in contract.shape)
    expected_dtype = {
        "f32": np.dtype(np.float32),
        "float32": np.dtype(np.float32),
        "bf16": np.dtype(jnp.bfloat16),
        "bfloat16": np.dtype(jnp.bfloat16),
    }.get(contract.dtype)
    if expected_dtype is None:
        raise ValueError(f"SEQAX_PALLAS_SEARCH_OUTPUT_DTYPE_UNSUPPORTED dtype={contract.dtype}")
    if output.shape != expected_shape or output.dtype != expected_dtype:
        raise ValueError(f"SEQAX_PALLAS_SEARCH_OUTPUT_ABI_MISMATCH candidate={candidate}")


def _validate_cpu_oracle_replay(
    saved: np.ndarray,
    fresh: np.ndarray,
    absolute_tolerance: float,
) -> None:
    if (
        saved.shape != fresh.shape
        or saved.dtype != fresh.dtype
        or not np.isfinite(saved).all()
        or not np.isfinite(fresh).all()
        or not np.allclose(saved, fresh, atol=absolute_tolerance, rtol=0)
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_ORACLE_REPLAY_MISMATCH")


def _portable_cpu_oracle_passed(
    outputs: list[np.ndarray],
    saved_oracles: list[np.ndarray],
    fresh_oracles: list[np.ndarray],
) -> tuple[bool, ...]:
    saved = tuple(
        bool(np.allclose(output, oracle, atol=SEQAX_OUTPUT_ATOL, rtol=SEQAX_OUTPUT_RTOL))
        for output, oracle in zip(outputs, saved_oracles, strict=True)
    )
    fresh = tuple(
        bool(np.allclose(output, oracle, atol=SEQAX_OUTPUT_ATOL, rtol=SEQAX_OUTPUT_RTOL))
        for output, oracle in zip(outputs, fresh_oracles, strict=True)
    )
    if saved != fresh:
        raise ValueError("SEQAX_PALLAS_SEARCH_ORACLE_VERDICT_PORTABILITY_MISMATCH")
    return saved


def _validate_correctness_replay(
    root: Path,
    contract: SeqaxPallasSearchContract,
    prepared: tuple[PreparedCandidate, ...],
    saved: tuple[SeqaxPallasCandidateCorrectness, ...],
) -> None:
    if tuple(value.name for value in saved) != tuple(
        candidate.name for candidate in contract.candidates
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_CORRECTNESS_CANDIDATE_MISMATCH")
    inputs_by_seed = []
    oracles = []
    fresh_oracles = []
    outputs: dict[str, list[np.ndarray]] = {candidate.name: [] for candidate in contract.candidates}
    output_contracts = {value.candidate.name: value.plan.output_contracts[0] for value in prepared}
    for seed in contract.correctness_seeds:
        expected_inputs = tuple(
            np.asarray(value) for value in seqax_forward_inputs(seed=seed, **contract.parameters)
        )
        saved_inputs = tuple(
            _load_array(root / str(seed) / "inputs" / f"{index:02d}.npy")
            for index in range(len(expected_inputs))
        )
        if any(
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not np.array_equal(actual, expected)
            for actual, expected in zip(saved_inputs, expected_inputs, strict=True)
        ):
            raise ValueError("SEQAX_PALLAS_SEARCH_INPUT_REPLAY_MISMATCH")
        expected_oracle = np.asarray(
            seqax_forward_canonical_reference(expected_inputs, **contract.parameters)
        )
        saved_oracle = _load_array(root / str(seed) / "cpu_oracle.npy")
        _validate_cpu_oracle_replay(
            saved_oracle,
            expected_oracle,
            contract.cpu_oracle_replay_absolute_tolerance,
        )
        inputs_by_seed.append(saved_inputs)
        oracles.append(saved_oracle)
        fresh_oracles.append(expected_oracle)
        for candidate in contract.candidates:
            output = _load_array(root / str(seed) / "outputs" / f"{candidate.name}.npy")
            output_contract = output_contracts[candidate.name]
            _validate_output_abi(output, output_contract, candidate.name)
            outputs[candidate.name].append(output)
    baseline = outputs[contract.baseline]
    baseline_hashes = tuple(array_sha256(value) for value in baseline)
    for record in saved:
        candidate_outputs = outputs[record.name]
        exact = all(
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and np.array_equal(actual, expected)
            for actual, expected in zip(candidate_outputs, baseline, strict=True)
        )
        errors = tuple(
            _errors(actual, oracle)
            for actual, oracle in zip(candidate_outputs, oracles, strict=True)
        )
        portable_cpu_oracle_passed = _portable_cpu_oracle_passed(
            candidate_outputs,
            oracles,
            fresh_oracles,
        )
        expected_record = SeqaxPallasCandidateCorrectness(
            name=record.name,
            input_sha256=tuple(arrays_sha256(value) for value in inputs_by_seed),
            output_sha256=tuple(array_sha256(value) for value in candidate_outputs),
            baseline_output_sha256=baseline_hashes,
            exact_baseline_parity=exact,
            cpu_oracle_sha256=tuple(array_sha256(value) for value in oracles),
            cpu_oracle_maximum_absolute_error=tuple(value[0] for value in errors),
            cpu_oracle_maximum_relative_error=tuple(value[1] for value in errors),
            cpu_oracle_passed=portable_cpu_oracle_passed,
        )
        if record != expected_record or not exact:
            raise ValueError(
                f"SEQAX_PALLAS_SEARCH_CORRECTNESS_REPLAY_MISMATCH candidate={record.name}"
            )


def _validate_seqax_pallas_search(
    root: Path,
    trusted_contract: SeqaxPallasSearchContract,
    *,
    require_accepted: bool,
) -> SeqaxPallasSearchResult:
    if root.is_symlink():
        raise ValueError(f"SEQAX_PALLAS_SEARCH_ROOT_INVALID path={root}")
    root = root.resolve()
    _preflight_existing_root(root)
    saved_contract = SeqaxPallasSearchContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != trusted_contract:
        raise ValueError("SEQAX_PALLAS_SEARCH_CONTRACT_MISMATCH")
    result = SeqaxPallasSearchResult.model_validate_json((root / "result.json").read_text())
    if (
        result.search_id != trusted_contract.search_id
        or result.baseline != trusted_contract.baseline
        or result.runtime != trusted_contract.runtime
        or result.device_kind != trusted_contract.device_kind
        or result.device_count != trusted_contract.device_count
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_RESULT_IDENTITY_MISMATCH")
    if (
        len(result.devices) != trusted_contract.device_count
        or tuple(value.id for value in result.devices)
        != tuple(range(trusted_contract.device_count))
        or any(value.platform != "tpu" for value in result.devices)
        or any(value.device_kind not in {"TPU7x", "TPU v7x"} for value in result.devices)
        or len({value.process_index for value in result.devices}) != 1
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_DEVICE_INVENTORY_MISMATCH")
    expected_timing_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=trusted_contract.timing_seed,
            **trusted_contract.parameters,
        )
    )
    if (
        trusted_contract.timing_seed not in trusted_contract.correctness_seeds
        or result.timing_input_sha256 != arrays_sha256(expected_timing_inputs)
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_TIMING_INPUT_MISMATCH")
    source_state_path = root / "source_state.json"
    source_state = json.loads(source_state_path.read_text())
    repository_root = Path(__file__).resolve().parents[2]
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        result.source_state_sha256 != _sha256(source_state_path)
        or source_state.get("git_dirty") is not False
        or source_state.get("git_status") != []
        or source_state.get("git_commit") != current_commit
        or source_state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
        or result.search_source_manifest != _search_source_manifest()
        or (root / "source_diff.patch").read_bytes() != b""
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_SOURCE_STATE_MISMATCH")

    distributed, prepared = prepare_seqax_pallas_candidates(trusted_contract)
    if (root / "distributed.xdsl").read_text() != canonical_text(distributed):
        raise ValueError("SEQAX_PALLAS_SEARCH_DISTRIBUTED_IR_MISMATCH")
    if len(result.plans) != len(prepared):
        raise ValueError("SEQAX_PALLAS_SEARCH_PLAN_COUNT_MISMATCH")
    for record, expected in zip(result.plans, prepared, strict=True):
        plan_root = root / "plans" / expected.candidate.name
        stablehlo = (plan_root / "stablehlo.txt").read_text()
        compiler_hlo = (plan_root / "compiler_hlo.txt").read_text()
        all_gather_count, reduce_scatter_count = _physical_collective_counts(expected.physical)
        _validate_compiled_program(
            stablehlo,
            compiler_hlo,
            pallas_region_count=expected.plan.pallas_region_count,
            all_gather_count=all_gather_count,
            reduce_scatter_count=reduce_scatter_count,
        )
        observed_tiles = _compiler_tile_metadata(compiler_hlo)
        expected_tiles = tuple(
            (index, expected.plan.physical_schedule_sha256, *tiles)
            for index, tiles in enumerate(expected.tiles)
        )
        if observed_tiles != expected_tiles:
            raise ValueError(f"SEQAX_PALLAS_SEARCH_COMPILER_TILE_MISMATCH candidate={record.name}")
        expected_record = SeqaxPallasCandidatePlan(
            name=expected.candidate.name,
            policy=expected.candidate.policy,
            tiles=expected.tiles,
            changed_region_count=expected.candidate.expected_changed_regions,
            physical_schedule_sha256=expected.plan.physical_schedule_sha256,
            pallas_source_sha256=expected.plan.source_sha256(),
            stablehlo_sha256=_sha256(plan_root / "stablehlo.txt"),
            compiler_hlo_sha256=_sha256(plan_root / "compiler_hlo.txt"),
        )
        if (
            record != expected_record
            or (plan_root / "physical.xdsl").read_text() != canonical_text(expected.physical)
            or (plan_root / "lowered_pallas.py").read_text()
            != expected.plan.render_executable_source()
            or json.loads((plan_root / "plan_manifest.json").read_text())
            != expected.plan.manifest()
            or tuple(tuple(value) for value in json.loads((plan_root / "tiles.json").read_text()))
            != expected.tiles
        ):
            raise ValueError(f"SEQAX_PALLAS_SEARCH_PLAN_REPLAY_MISMATCH candidate={record.name}")

    primitive_json = tuple(
        SeqaxPallasPrimitiveObservation.model_validate(value)
        for value in json.loads((root / "primitive_observations.json").read_text())
    )
    if primitive_json != result.primitive_observations:
        raise ValueError("SEQAX_PALLAS_SEARCH_PRIMITIVE_RESULT_MISMATCH")
    _validate_primitive_replay(
        root / "primitive",
        trusted_contract,
        prepared,
        primitive_json,
    )
    correctness_json = tuple(
        SeqaxPallasCandidateCorrectness.model_validate(value)
        for value in json.loads((root / "correctness.json").read_text())
    )
    if correctness_json != result.correctness:
        raise ValueError("SEQAX_PALLAS_SEARCH_CORRECTNESS_RESULT_MISMATCH")
    _validate_correctness_replay(
        root / "correctness",
        trusted_contract,
        prepared,
        correctness_json,
    )

    rounds = tuple(
        SeqaxPallasRoundObservation.model_validate(value)
        for value in json.loads((root / "rounds.json").read_text())
    )
    confirmation_rounds = tuple(
        SeqaxPallasRoundObservation.model_validate(value)
        for value in json.loads((root / "confirmation_rounds.json").read_text())
    )
    if rounds != result.rounds or confirmation_rounds != result.confirmation_rounds:
        raise ValueError("SEQAX_PALLAS_SEARCH_TIMING_RESULT_MISMATCH")
    statistics_by_candidate = candidate_statistics(trusted_contract, rounds)
    if statistics_by_candidate != result.candidates:
        raise ValueError("SEQAX_PALLAS_SEARCH_STATISTICS_REPLAY_MISMATCH")
    promotable = tuple(value for value in statistics_by_candidate if value.promotable)
    provisional = (
        min(promotable, key=lambda value: value.median_round_ns).name if promotable else None
    )
    expected_confirmation = (
        confirmation_statistics(trusted_contract, provisional, confirmation_rounds)
        if provisional is not None
        else None
    )
    expected_winner = (
        provisional
        if expected_confirmation is not None and expected_confirmation.confirmed
        else None
    )
    if (
        result.execution_orders != execution_orders(trusted_contract)
        or result.provisional_winner != provisional
        or result.confirmation != expected_confirmation
        or result.winner != expected_winner
        or (provisional is None and confirmation_rounds)
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_SELECTION_REPLAY_MISMATCH")

    run_id = semantic_sha256("seqax-pallas-tile-search-run-v1", trusted_contract.search_id)
    ledger_payloads = [
        (RunState.CREATED, {"search_id": trusted_contract.search_id}),
        (
            RunState.VERIFIED,
            {
                "candidate_schedules": {
                    value.candidate.name: value.plan.physical_schedule_sha256 for value in prepared
                }
            },
        ),
        (
            RunState.LOWERED,
            {
                "pallas_sources": {
                    value.candidate.name: value.plan.source_sha256() for value in prepared
                }
            },
        ),
        (
            RunState.COMPILED,
            {"plans": [value.model_dump(mode="json") for value in result.plans]},
        ),
        (
            RunState.CORRECT,
            {
                "primitive_case_count": len(result.primitive_observations),
                "candidate_output_sha256": {
                    value.name: value.output_sha256 for value in result.correctness
                },
            },
        ),
        (
            RunState.TIMED,
            {
                "round_count": len(result.rounds),
                "confirmation_round_count": len(result.confirmation_rounds),
                "provisional_winner": result.provisional_winner,
                "winner": result.winner,
            },
        ),
    ]
    if require_accepted:
        ledger_payloads.append(
            (
                RunState.ACCEPTED,
                {
                    "result_sha256": _sha256(root / "result.json"),
                    "winner": result.winner,
                },
            )
        )
    history = read_ledger_history(root / "ledger.sqlite", run_id)
    if tuple(event.state for event in history) != tuple(value[0] for value in ledger_payloads):
        raise ValueError("SEQAX_PALLAS_SEARCH_LEDGER_STATE_MISMATCH")
    if tuple(event.payload_sha256 for event in history) != tuple(
        ExperimentLedger.payload_sha256(value[1]) for value in ledger_payloads
    ):
        raise ValueError("SEQAX_PALLAS_SEARCH_LEDGER_PAYLOAD_MISMATCH")

    expected_files = _expected_files(
        root,
        trusted_contract,
        result.primitive_observations,
        require_accepted,
    )
    _validate_closed_world(root, expected_files)
    if require_accepted:
        receipt = SeqaxPallasSearchReceipt.model_validate_json((root / "receipt.json").read_text())
        expected_receipt = SeqaxPallasSearchReceipt(
            search_id=trusted_contract.search_id,
            status="passed",
            result_sha256=_sha256(root / "result.json"),
            ledger_sha256=_sha256(root / "ledger.sqlite"),
            artifacts=_artifact_manifest(root),
        )
        if receipt != expected_receipt:
            raise ValueError("SEQAX_PALLAS_SEARCH_RECEIPT_REPLAY_MISMATCH")
    return result


def validate_seqax_pallas_search(
    root: Path,
    trusted_contract: SeqaxPallasSearchContract,
) -> SeqaxPallasSearchResult:
    return _validate_seqax_pallas_search(
        root,
        trusted_contract,
        require_accepted=True,
    )
