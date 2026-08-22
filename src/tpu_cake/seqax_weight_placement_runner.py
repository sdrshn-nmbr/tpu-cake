from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from xdsl.dialects.builtin import ModuleOp

from tpu_cake.artifacts import (
    build_artifact_manifest,
)
from tpu_cake.artifacts import (
    file_sha256 as _sha256,
)
from tpu_cake.artifacts import (
    text_file_sha256 as _text_sha256,
)
from tpu_cake.artifacts import (
    write_json as _write_json,
)
from tpu_cake.artifacts import (
    write_text as _write_text,
)
from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import (
    CompilerExecutableAnalysis,
    capture_compiler_analysis,
    validate_compiler_analysis,
    write_compiler_analysis,
)
from tpu_cake.contracts import ArtifactReference, ArtifactRole, SourceFileContract
from tpu_cake.dialects.distributed_tensor import AllGatherOp
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import EvidenceRun, ExperimentLedger, RunState, read_ledger_history
from tpu_cake.runner import _runtime_identity, _source_state
from tpu_cake.seqax_pallas_lowering import (
    SeqaxPallasPlan,
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
    SeqaxPallasDevice,
    SeqaxPallasRoundObservation,
    candidate_statistics,
    confirmation_statistics,
    execution_orders,
)
from tpu_cake.seqax_pallas_search_runner import (
    _compiler_tile_metadata,
    _portable_cpu_oracle_passed,
    _validate_cpu_oracle_replay,
    _validate_output_abi,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementCandidate,
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementName,
    SeqaxWeightPlacementPlan,
    SeqaxWeightPlacementReceipt,
    SeqaxWeightPlacementResult,
    SeqaxWeightResidencyObservation,
    default_seqax_weight_placement_contract,
    parameter_residency_bytes_per_device,
)
from tpu_cake.stablehlo import StableHloInspector
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)


@dataclass(frozen=True)
class PreparedPlacement:
    candidate: SeqaxWeightPlacementCandidate
    distributed: ModuleOp
    physical: ModuleOp
    plan: SeqaxPallasPlan


@dataclass(frozen=True)
class CompiledPlacement:
    prepared: PreparedPlacement
    executable: Any
    mesh: Any
    stablehlo: str
    compiler_hlo: str
    compiler_analysis: CompilerExecutableAnalysis


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def _load_array(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_ARRAY_INVALID path={path}")
    return np.load(path, allow_pickle=False)


def _source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "artifacts.py",
        package / "canonical.py",
        package / "cli.py",
        package / "compiler_analysis.py",
        package / "contracts.py",
        package / "dtensor_interpreter.py",
        package / "identity.py",
        package / "jax_lowering.py",
        package / "ledger.py",
        package / "lowering.py",
        package / "physical_geometry.py",
        package / "runner.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_pallas_runner.py",
        package / "seqax_pallas_search.py",
        package / "seqax_pallas_search_runner.py",
        package / "seqax_physical_execution.py",
        package / "seqax_physical_lowering.py",
        package / "seqax_runner.py",
        package / "stablehlo.py",
        package / "seqax_weight_confirmation.py",
        package / "seqax_weight_confirmation_runner.py",
        package / "seqax_weight_placement.py",
        package / "seqax_weight_placement_runner.py",
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
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_SOURCE_DIRTY status={status}")


def _require_safe_new_root(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == value or root in value.parents for value in protected):
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_UNSAFE_ROOT path={root}")
    if root.exists():
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_ROOT_EXISTS path={root}")


def _preflight_existing_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_HARDLINK path={path}")


def prepare_weight_placement_candidates(
    contract: SeqaxWeightPlacementContract,
) -> tuple[PreparedPlacement, ...]:
    prepared = []
    for candidate in contract.candidates:
        distributed = seqax_forward_schedule(
            **contract.parameters,
            weight_data_placement=candidate.policy.schedule_policy(),
        )
        physical = lower_seqax_forward_to_physical(distributed).module
        plan = lower_seqax_physical_to_pallas(distributed, physical)
        high_level_all_gathers = sum(
            isinstance(operation, AllGatherOp) for operation in distributed.walk()
        )
        all_gathers, reduce_scatters = _physical_collective_counts(physical)
        parameter_bytes = parameter_residency_bytes_per_device(
            plan.input_contracts,
            mesh=dict(plan.mesh),
        )
        observed = (
            high_level_all_gathers,
            all_gathers + reduce_scatters,
            all_gathers,
            parameter_bytes,
        )
        expected = (
            candidate.expected_high_level_all_gathers,
            candidate.expected_physical_collectives,
            candidate.expected_stablehlo_all_gathers,
            candidate.expected_parameter_bytes_per_device,
        )
        if observed != expected or plan.pallas_region_count != 9:
            raise ValueError(
                "SEQAX_WEIGHT_PLACEMENT_PLAN_CONTRACT_MISMATCH "
                f"candidate={candidate.name} expected={expected} observed={observed} "
                f"pallas_regions={plan.pallas_region_count}"
            )
        prepared.append(
            PreparedPlacement(
                candidate=candidate,
                distributed=distributed,
                physical=physical,
                plan=plan,
            )
        )
    identities = tuple(
        (
            value.plan.distributed_schedule_sha256,
            value.plan.physical_schedule_sha256,
            value.plan.source_sha256(),
        )
        for value in prepared
    )
    if len(set(identities)) != len(identities):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_PLANS_NOT_DISTINCT")
    return tuple(prepared)


def _resident_inputs(
    host_inputs: tuple[np.ndarray, ...],
    prepared: PreparedPlacement,
    mesh: Any,
) -> tuple[jax.Array, ...]:
    return tuple(
        jax.device_put(
            jnp.asarray(value),
            NamedSharding(mesh, tensor.partition_spec()),
        )
        for value, tensor in zip(host_inputs, prepared.plan.input_contracts, strict=True)
    )


def _compile(
    prepared: PreparedPlacement,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
) -> CompiledPlacement:
    callable_, mesh = prepared.plan.build(interpret=False, devices=devices)
    resident = _resident_inputs(host_inputs, prepared, mesh)
    lowered = callable_.lower(*resident)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    pre_optimization_hlo = _compiler_hlo(lowered)
    all_gathers, reduce_scatters = _physical_collective_counts(prepared.physical)
    _validate_compiled_program(
        stablehlo,
        pre_optimization_hlo,
        pallas_region_count=prepared.plan.pallas_region_count,
        pallas_vector_region_count=prepared.plan.pallas_vector_region_count,
        all_gather_count=all_gathers,
        reduce_scatter_count=reduce_scatters,
    )
    stablehlo_all_gathers = StableHloInspector.parse(stablehlo).live_public_main_operation_count(
        "stablehlo.all_gather"
    )
    if stablehlo_all_gathers != prepared.candidate.expected_stablehlo_all_gathers:
        raise ValueError(
            f"SEQAX_WEIGHT_PLACEMENT_STABLEHLO_GATHER_MISMATCH candidate={prepared.candidate.name}"
        )
    executable = lowered.compile()
    compiler_hlo = executable.as_text()
    return CompiledPlacement(
        prepared=prepared,
        executable=executable,
        mesh=mesh,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
        compiler_analysis=capture_compiler_analysis(
            executable,
            stablehlo=stablehlo,
            compiler_hlo=compiler_hlo,
        ),
    )


def _execute(
    compiled: CompiledPlacement,
    inputs: tuple[jax.Array, ...],
) -> np.ndarray:
    outputs = compiled.executable(*inputs)
    jax.block_until_ready(outputs)
    if len(outputs) != 1:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_OUTPUT_COUNT_MISMATCH")
    return np.asarray(outputs[0])


def _device_inventory(devices: tuple[Any, ...]) -> tuple[SeqaxPallasDevice, ...]:
    return tuple(
        SeqaxPallasDevice(
            id=device.id,
            process_index=device.process_index,
            platform=device.platform,
            device_kind=device.device_kind,
        )
        for device in devices
    )


def _validate_devices(devices: tuple[Any, ...], contract: SeqaxWeightPlacementContract) -> None:
    if (
        jax.default_backend() != contract.backend
        or len(devices) != contract.device_count
        or tuple(device.id for device in devices) != tuple(range(contract.device_count))
        or any(device.platform != "tpu" for device in devices)
        or any(device.device_kind not in {"TPU7x", "TPU v7x"} for device in devices)
        or len({device.process_index for device in devices}) != 1
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DEVICE_MISMATCH")


def probe_weight_placement_memory(
    candidate_name: SeqaxWeightPlacementName,
) -> SeqaxWeightResidencyObservation:
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime = _runtime_identity()
    contract = default_seqax_weight_placement_contract(runtime)
    devices = tuple(jax.devices())
    _validate_devices(devices, contract)
    prepared = next(
        value
        for value in prepare_weight_placement_candidates(contract)
        if value.candidate.name is candidate_name
    )
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=contract.timing_seed, **contract.parameters)
    )
    compiled = _compile(prepared, host_inputs, devices)
    resident = _resident_inputs(host_inputs, prepared, compiled.mesh)
    output = _execute(compiled, resident)
    stats = tuple(device.memory_stats() for device in devices)
    if any(value is None for value in stats):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_MEMORY_STATS_UNAVAILABLE")
    limits = tuple(int(value["bytes_limit"]) for value in stats if value is not None)
    peaks = tuple(int(value["peak_bytes_in_use"]) for value in stats if value is not None)
    largest = tuple(int(value["largest_alloc_size"]) for value in stats if value is not None)
    return SeqaxWeightResidencyObservation(
        candidate=candidate_name,
        runtime=runtime,
        devices=_device_inventory(devices),
        distributed_schedule_sha256=prepared.plan.distributed_schedule_sha256,
        physical_schedule_sha256=prepared.plan.physical_schedule_sha256,
        pallas_source_sha256=prepared.plan.source_sha256(),
        stablehlo_sha256=_text_sha256(compiled.stablehlo),
        compiler_hlo_sha256=_text_sha256(compiled.compiler_hlo),
        source_commit=source_commit,
        source_manifest=_source_manifest(),
        timing_input_sha256=arrays_sha256(host_inputs),
        output_sha256=array_sha256(output),
        parameter_bytes_per_device=prepared.candidate.expected_parameter_bytes_per_device,
        device_bytes_limit=limits,
        peak_bytes_in_use=peaks,
        largest_allocation_bytes=largest,
        isolated_process=True,
        fits_observed_device_memory=all(peak <= limit for peak, limit in zip(peaks, limits)),
    )


def _isolated_memory_observations(
    contract: SeqaxWeightPlacementContract,
    repository_root: Path,
) -> tuple[SeqaxWeightResidencyObservation, ...]:
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observations = []
    for candidate in contract.candidates:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "tpu_cake.cli",
                "probe-seqax-weight-placement-memory",
                "--candidate",
                candidate.name,
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise ValueError(
                "SEQAX_WEIGHT_PLACEMENT_MEMORY_SUBPROCESS_FAILED "
                f"candidate={candidate.name} returncode={process.returncode} "
                f"stdout={process.stdout!r} stderr={process.stderr!r}"
            )
        prefix = "SEQAX_WEIGHT_PLACEMENT_MEMORY_JSON="
        records = [
            line.removeprefix(prefix)
            for line in process.stdout.splitlines()
            if line.startswith(prefix)
        ]
        if len(records) != 1:
            raise ValueError(
                "SEQAX_WEIGHT_PLACEMENT_MEMORY_SUBPROCESS_OUTPUT "
                f"candidate={candidate.name} stdout={process.stdout!r} stderr={process.stderr!r}"
            )
        observation = SeqaxWeightResidencyObservation.model_validate_json(records[0])
        if (
            observation.candidate is not candidate.name
            or observation.runtime != contract.runtime
            or observation.source_commit != source_commit
            or observation.source_manifest != _source_manifest()
            or observation.parameter_bytes_per_device
            != candidate.expected_parameter_bytes_per_device
            or not observation.fits_observed_device_memory
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_MEMORY_CONTRACT_MISMATCH candidate={candidate.name}"
            )
        observations.append(observation)
    return tuple(observations)


def _cpu_oracle_verdicts(
    outputs: list[np.ndarray],
    oracles: list[np.ndarray],
) -> tuple[bool, ...]:
    return _portable_cpu_oracle_passed(outputs, oracles, oracles)


def _candidate_correctness(
    root: Path,
    contract: SeqaxWeightPlacementContract,
    compiled: tuple[CompiledPlacement, ...],
) -> tuple[SeqaxPallasCandidateCorrectness, ...]:
    inputs_by_seed: list[tuple[np.ndarray, ...]] = []
    oracles: list[np.ndarray] = []
    outputs: dict[str, list[np.ndarray]] = {value.prepared.candidate.name: [] for value in compiled}
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
        oracles.append(oracle)
        for candidate in compiled:
            actual = _execute(
                candidate,
                _resident_inputs(host_inputs, candidate.prepared, candidate.mesh),
            )
            _save_array(
                seed_root / "outputs" / f"{candidate.prepared.candidate.name}.npy",
                actual,
            )
            outputs[candidate.prepared.candidate.name].append(actual)
    baseline = outputs[contract.baseline]
    baseline_hashes = tuple(array_sha256(value) for value in baseline)
    records = []
    for candidate in compiled:
        name = candidate.prepared.candidate.name
        actuals = outputs[name]
        exact = all(
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and np.array_equal(actual, expected)
            for actual, expected in zip(actuals, baseline, strict=True)
        )
        if not exact:
            raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_PARITY_FAILED candidate={name}")
        errors = tuple(
            _errors(actual, oracle) for actual, oracle in zip(actuals, oracles, strict=True)
        )
        records.append(
            SeqaxPallasCandidateCorrectness(
                name=name,
                input_sha256=tuple(arrays_sha256(value) for value in inputs_by_seed),
                output_sha256=tuple(array_sha256(value) for value in actuals),
                baseline_output_sha256=baseline_hashes,
                exact_baseline_parity=True,
                cpu_oracle_sha256=tuple(array_sha256(value) for value in oracles),
                cpu_oracle_maximum_absolute_error=tuple(value[0] for value in errors),
                cpu_oracle_maximum_relative_error=tuple(value[1] for value in errors),
                cpu_oracle_passed=_cpu_oracle_verdicts(actuals, oracles),
            )
        )
    if any(
        value.cpu_oracle_passed != contract.expected_incumbent_cpu_oracle_passed
        for value in records
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_ORACLE_SCOPE_MISMATCH")
    return tuple(records)


def _measure(
    compiled: CompiledPlacement,
    inputs: tuple[jax.Array, ...],
    iterations: int,
) -> tuple[int, ...]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        jax.block_until_ready(compiled.executable(*inputs))
        samples.append(time.perf_counter_ns() - started)
    return tuple(samples)


def _timing_observations(
    contract: SeqaxWeightPlacementContract,
    compiled: dict[str, CompiledPlacement],
    resident_inputs: dict[str, tuple[jax.Array, ...]],
    orders: tuple[tuple[str, ...], ...],
) -> tuple[SeqaxPallasRoundObservation, ...]:
    observations = []
    for round_index, order in enumerate(orders):
        for position, name in enumerate(order):
            samples = _measure(compiled[name], resident_inputs[name], contract.measured_iterations)
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


def _confirmation_orders(
    contract: SeqaxWeightPlacementContract,
    candidate: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (contract.baseline, candidate) if round_index % 2 == 0 else (candidate, contract.baseline)
        for round_index in range(contract.confirmation_rounds)
    )


def _plan_record(root: Path, value: CompiledPlacement) -> SeqaxWeightPlacementPlan:
    plan_root = root / "plans" / value.prepared.candidate.name
    stablehlo_all_gathers = StableHloInspector.parse(
        value.stablehlo
    ).live_public_main_operation_count("stablehlo.all_gather")
    return SeqaxWeightPlacementPlan(
        candidate=value.prepared.candidate.name,
        policy=value.prepared.candidate.policy,
        distributed_schedule_sha256=value.prepared.plan.distributed_schedule_sha256,
        physical_schedule_sha256=value.prepared.plan.physical_schedule_sha256,
        pallas_source_sha256=value.prepared.plan.source_sha256(),
        stablehlo_sha256=_sha256(plan_root / "stablehlo.txt"),
        compiler_hlo_sha256=_sha256(plan_root / "compiler_hlo.txt"),
        high_level_all_gathers=sum(
            isinstance(operation, AllGatherOp) for operation in value.prepared.distributed.walk()
        ),
        physical_collectives=sum(_physical_collective_counts(value.prepared.physical)),
        stablehlo_all_gathers=stablehlo_all_gathers,
        pallas_regions=value.prepared.plan.pallas_region_count,
        parameter_bytes_per_device=parameter_residency_bytes_per_device(
            value.prepared.plan.input_contracts,
            mesh=dict(value.prepared.plan.mesh),
        ),
    )


def run_seqax_weight_placement(
    root: Path,
    contract: SeqaxWeightPlacementContract,
) -> SeqaxWeightPlacementResult:
    root = root.resolve()
    _require_safe_new_root(root)
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
    runtime = _runtime_identity()
    if runtime != contract.runtime:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_RUNTIME_MISMATCH")
    memory = _isolated_memory_observations(contract, repository_root)
    devices = tuple(jax.devices())
    _validate_devices(devices, contract)
    root.mkdir(parents=True)
    _write_json(
        root / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    _source_state(repository_root, root)
    source_state_sha256 = _sha256(root / "source_state.json")
    _write_json(root / "memory.json", [value.model_dump(mode="json") for value in memory])
    run_id = semantic_sha256("seqax-weight-placement-run-v1", contract.search_id)
    ledger_path = root / "ledger.sqlite"
    evidence_run = EvidenceRun(ledger_path, run_id)
    evidence_run.create({"search_id": contract.search_id})

    prepared = prepare_weight_placement_candidates(contract)
    evidence_run.transition(
        RunState.VERIFIED,
        {
            "distributed_schedules": {
                value.candidate.name: value.plan.distributed_schedule_sha256 for value in prepared
            }
        },
    )
    for value in prepared:
        plan_root = root / "plans" / value.candidate.name
        _write_text(plan_root / "distributed.xdsl", canonical_text(value.distributed))
        _write_text(plan_root / "physical.xdsl", canonical_text(value.physical))
        _write_text(plan_root / "lowered_pallas.py", value.plan.render_executable_source())
        _write_json(plan_root / "plan_manifest.json", value.plan.manifest())
    evidence_run.transition(
        RunState.LOWERED,
        {
            "pallas_sources": {
                value.candidate.name: value.plan.source_sha256() for value in prepared
            }
        },
    )

    timing_host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=contract.timing_seed, **contract.parameters)
    )
    compiled = tuple(_compile(value, timing_host_inputs, devices) for value in prepared)
    for value in compiled:
        plan_root = root / "plans" / value.prepared.candidate.name
        _write_text(plan_root / "stablehlo.txt", value.stablehlo + "\n")
        _write_text(plan_root / "compiler_hlo.txt", value.compiler_hlo + "\n")
        write_compiler_analysis(
            plan_root / "compiler_analysis.json",
            value.compiler_analysis,
        )
    plan_records = tuple(_plan_record(root, value) for value in compiled)
    evidence_run.transition(
        RunState.COMPILED,
        {"plans": [value.model_dump(mode="json") for value in plan_records]},
    )

    correctness = _candidate_correctness(root / "correctness", contract, compiled)
    _write_json(
        root / "correctness.json",
        [value.model_dump(mode="json") for value in correctness],
    )
    evidence_run.transition(
        RunState.CORRECT,
        {
            "candidate_output_sha256": {value.name: value.output_sha256 for value in correctness},
            "cpu_oracle_passed": correctness[0].cpu_oracle_passed,
        },
    )

    compiled_by_name = {value.prepared.candidate.name: value for value in compiled}
    timing_inputs = {
        name: _resident_inputs(timing_host_inputs, value.prepared, value.mesh)
        for name, value in compiled_by_name.items()
    }
    for name, value in compiled_by_name.items():
        for _ in range(contract.warmup_iterations):
            jax.block_until_ready(value.executable(*timing_inputs[name]))
    orders = execution_orders(contract)
    rounds = _timing_observations(contract, compiled_by_name, timing_inputs, orders)
    statistics_by_candidate = candidate_statistics(contract, rounds)
    promotable = tuple(value for value in statistics_by_candidate if value.promotable)
    provisional = (
        min(promotable, key=lambda value: value.median_round_ns).name if promotable else None
    )
    confirmation_rounds: tuple[SeqaxPallasRoundObservation, ...] = ()
    confirmation = None
    winner = None
    if provisional is not None:
        confirmation_rounds = _timing_observations(
            contract,
            compiled_by_name,
            timing_inputs,
            _confirmation_orders(contract, provisional),
        )
        confirmation = confirmation_statistics(contract, provisional, confirmation_rounds)
        winner = provisional if confirmation.confirmed else None
    _write_json(root / "rounds.json", [value.model_dump(mode="json") for value in rounds])
    _write_json(
        root / "confirmation_rounds.json",
        [value.model_dump(mode="json") for value in confirmation_rounds],
    )
    result = SeqaxWeightPlacementResult(
        search_id=contract.search_id,
        baseline=contract.baseline,
        runtime=runtime,
        devices=_device_inventory(devices),
        timing_input_sha256=arrays_sha256(timing_host_inputs),
        source_state_sha256=source_state_sha256,
        source_manifest=_source_manifest(),
        plans=plan_records,
        memory=memory,
        correctness=correctness,
        execution_orders=orders,
        rounds=rounds,
        candidates=statistics_by_candidate,
        provisional_winner=provisional,
        confirmation_rounds=confirmation_rounds,
        confirmation=confirmation,
        winner=winner,
        correctness_scope="incumbent-bit-exact",
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    evidence_run.transition(
        RunState.TIMED,
        {
            "round_count": len(rounds),
            "confirmation_round_count": len(confirmation_rounds),
            "provisional_winner": provisional,
            "winner": winner,
        },
    )
    evidence_run.seal("SEQAX_WEIGHT_PLACEMENT_LEDGER_SIDECARS paths={paths}")
    _validate(root, contract, require_accepted=False)
    evidence_run.transition(
        RunState.ACCEPTED,
        {"result_sha256": _sha256(root / "result.json"), "winner": winner},
    )
    evidence_run.seal("SEQAX_WEIGHT_PLACEMENT_LEDGER_SIDECARS paths={paths}")
    _build_receipt(root, contract)
    return validate_seqax_weight_placement(root, contract)


def _expected_files(
    root: Path,
    contract: SeqaxWeightPlacementContract,
    *,
    receipt_present: bool,
) -> set[Path]:
    expected = {
        root / "contract.json",
        root / "source_state.json",
        root / "source_diff.patch",
        root / "memory.json",
        root / "ledger.sqlite",
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
                "distributed.xdsl",
                "physical.xdsl",
                "lowered_pallas.py",
                "plan_manifest.json",
                "stablehlo.txt",
                "compiler_hlo.txt",
                "compiler_analysis.json",
            )
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
    return {value.resolve() for value in expected}


def _validate_closed_world(root: Path, expected: set[Path]) -> None:
    _preflight_existing_root(root)
    observed = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if observed != expected:
        raise ValueError(
            "SEQAX_WEIGHT_PLACEMENT_CLOSED_WORLD_MISMATCH "
            f"missing={sorted(map(str, expected - observed))} "
            f"extra={sorted(map(str, observed - expected))}"
        )


def _artifact_role(path: Path) -> ArtifactRole:
    value = path.as_posix()
    if value == "contract.json":
        return ArtifactRole.SEARCH_CONTRACT
    if value == "result.json":
        return ArtifactRole.SEARCH_RESULT
    if value == "ledger.sqlite":
        return ArtifactRole.EXECUTION_LEDGER
    if value == "source_state.json":
        return ArtifactRole.SOURCE_STATE
    if value == "source_diff.patch":
        return ArtifactRole.SOURCE_DIFF
    if value.endswith("/distributed.xdsl"):
        return ArtifactRole.DISTRIBUTED_IR
    if value.endswith("/physical.xdsl"):
        return ArtifactRole.PHYSICAL_IR
    if value.endswith("/lowered_pallas.py"):
        return ArtifactRole.PALLAS_SOURCE
    if value.endswith("/plan_manifest.json"):
        return ArtifactRole.PLAN_MANIFEST
    if value.endswith("/stablehlo.txt"):
        return ArtifactRole.STABLEHLO
    if value.endswith("/compiler_hlo.txt"):
        return ArtifactRole.COMPILER_HLO
    if value.endswith("/compiler_analysis.json"):
        return ArtifactRole.SEARCH_EVIDENCE
    if "/inputs/" in value:
        return ArtifactRole.CORRECTNESS_INPUT
    if value.endswith("/cpu_oracle.npy"):
        return ArtifactRole.ORACLE_OUTPUT
    if "/outputs/" in value:
        return ArtifactRole.CORRECTNESS_OUTPUT
    if value in {"rounds.json", "confirmation_rounds.json"}:
        return ArtifactRole.TIMING_SAMPLES
    return ArtifactRole.SEARCH_EVIDENCE


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    return build_artifact_manifest(root, role_for_path=_artifact_role)


def _build_receipt(
    root: Path,
    contract: SeqaxWeightPlacementContract,
) -> SeqaxWeightPlacementReceipt:
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_RECEIPT_EXISTS")
    receipt = SeqaxWeightPlacementReceipt(
        search_id=contract.search_id,
        status="passed",
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(root / "ledger.sqlite"),
        artifacts=_artifact_manifest(root),
    )
    _write_json(receipt_path, receipt.model_dump(mode="json"))
    return receipt


def _validate_correctness(
    root: Path,
    contract: SeqaxWeightPlacementContract,
    prepared: tuple[PreparedPlacement, ...],
    records: tuple[SeqaxPallasCandidateCorrectness, ...],
) -> None:
    expected_names = tuple(value.name for value in contract.candidates)
    if tuple(value.name for value in records) != expected_names:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_CORRECTNESS_NAMES_MISMATCH")
    inputs_by_seed = []
    saved_oracles = []
    fresh_oracles = []
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in expected_names}
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
            saved.shape != expected.shape
            or saved.dtype != expected.dtype
            or not np.array_equal(saved, expected)
            for saved, expected in zip(saved_inputs, expected_inputs, strict=True)
        ):
            raise ValueError("SEQAX_WEIGHT_PLACEMENT_INPUT_REPLAY_MISMATCH")
        fresh_oracle = np.asarray(
            seqax_forward_canonical_reference(expected_inputs, **contract.parameters)
        )
        saved_oracle = _load_array(root / str(seed) / "cpu_oracle.npy")
        _validate_cpu_oracle_replay(
            saved_oracle,
            fresh_oracle,
            contract.cpu_oracle_replay_absolute_tolerance,
        )
        inputs_by_seed.append(saved_inputs)
        saved_oracles.append(saved_oracle)
        fresh_oracles.append(fresh_oracle)
        for name in expected_names:
            output = _load_array(root / str(seed) / "outputs" / f"{name}.npy")
            _validate_output_abi(output, output_contracts[name], name)
            outputs[name].append(output)
    baseline = outputs[contract.baseline]
    baseline_hashes = tuple(array_sha256(value) for value in baseline)
    for record in records:
        actuals = outputs[record.name]
        exact = all(
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and np.array_equal(actual, expected)
            for actual, expected in zip(actuals, baseline, strict=True)
        )
        errors = tuple(
            _errors(actual, oracle) for actual, oracle in zip(actuals, saved_oracles, strict=True)
        )
        expected_record = SeqaxPallasCandidateCorrectness(
            name=record.name,
            input_sha256=tuple(arrays_sha256(value) for value in inputs_by_seed),
            output_sha256=tuple(array_sha256(value) for value in actuals),
            baseline_output_sha256=baseline_hashes,
            exact_baseline_parity=exact,
            cpu_oracle_sha256=tuple(array_sha256(value) for value in saved_oracles),
            cpu_oracle_maximum_absolute_error=tuple(value[0] for value in errors),
            cpu_oracle_maximum_relative_error=tuple(value[1] for value in errors),
            cpu_oracle_passed=_portable_cpu_oracle_passed(
                actuals,
                saved_oracles,
                fresh_oracles,
            ),
        )
        if record != expected_record or not exact:
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_CORRECTNESS_REPLAY_MISMATCH candidate={record.name}"
            )
        if record.cpu_oracle_passed != contract.expected_incumbent_cpu_oracle_passed:
            raise ValueError("SEQAX_WEIGHT_PLACEMENT_ORACLE_SCOPE_MISMATCH")


def _validate(
    root: Path,
    trusted_contract: SeqaxWeightPlacementContract,
    *,
    require_accepted: bool,
) -> SeqaxWeightPlacementResult:
    _preflight_existing_root(root)
    saved_contract = SeqaxWeightPlacementContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != trusted_contract:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_CONTRACT_MISMATCH")
    result = SeqaxWeightPlacementResult.model_validate_json((root / "result.json").read_text())
    expected_names = tuple(value.name for value in trusted_contract.candidates)
    if (
        result.search_id != trusted_contract.search_id
        or result.baseline is not trusted_contract.baseline
        or result.runtime != trusted_contract.runtime
        or tuple(value.candidate for value in result.plans) != expected_names
        or result.correctness_scope != "incumbent-bit-exact"
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_RESULT_IDENTITY_MISMATCH")
    if (
        tuple(value.id for value in result.devices) != tuple(range(trusted_contract.device_count))
        or any(value.platform != "tpu" for value in result.devices)
        or any(value.device_kind not in {"TPU7x", "TPU v7x"} for value in result.devices)
        or len({value.process_index for value in result.devices}) != 1
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DEVICE_INVENTORY_MISMATCH")
    timing_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=trusted_contract.timing_seed,
            **trusted_contract.parameters,
        )
    )
    if (
        trusted_contract.timing_seed not in trusted_contract.correctness_seeds
        or result.timing_input_sha256 != arrays_sha256(timing_inputs)
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_TIMING_INPUT_MISMATCH")
    repository_root = Path(__file__).resolve().parents[2]
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_state_path = root / "source_state.json"
    source_state = json.loads(source_state_path.read_text())
    if (
        result.source_state_sha256 != _sha256(source_state_path)
        or source_state.get("git_dirty") is not False
        or source_state.get("git_status") != []
        or source_state.get("git_commit") != current_commit
        or source_state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
        or result.source_manifest != _source_manifest()
        or (root / "source_diff.patch").read_bytes() != b""
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_SOURCE_STATE_MISMATCH")

    prepared = prepare_weight_placement_candidates(trusted_contract)
    if len(result.plans) != len(prepared):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_PLAN_COUNT_MISMATCH")
    for record, expected in zip(result.plans, prepared, strict=True):
        plan_root = root / "plans" / expected.candidate.name
        stablehlo = (plan_root / "stablehlo.txt").read_text()
        compiler_hlo = (plan_root / "compiler_hlo.txt").read_text()
        validate_compiler_analysis(
            plan_root / "compiler_analysis.json",
            stablehlo_path=plan_root / "stablehlo.txt",
            compiler_hlo_path=plan_root / "compiler_hlo.txt",
        )
        all_gathers, reduce_scatters = _physical_collective_counts(expected.physical)
        _validate_compiled_program(
            stablehlo,
            compiler_hlo,
            pallas_region_count=expected.plan.pallas_region_count,
            pallas_vector_region_count=expected.plan.pallas_vector_region_count,
            all_gather_count=all_gathers,
            reduce_scatter_count=reduce_scatters,
        )
        stablehlo_all_gathers = StableHloInspector.parse(
            stablehlo
        ).live_public_main_operation_count("stablehlo.all_gather")
        if stablehlo_all_gathers != expected.candidate.expected_stablehlo_all_gathers:
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_STABLEHLO_GATHER_MISMATCH candidate={record.candidate}"
            )
        observed_tiles = _compiler_tile_metadata(compiler_hlo)
        expected_tiles = tuple(
            (index, expected.plan.physical_schedule_sha256, *tiles)
            for index, tiles in enumerate(
                (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
                for operation in expected.physical.walk()
                if operation.name == "tpu_schedule.mxu_einsum"
            )
        )
        if observed_tiles != expected_tiles:
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_COMPILER_TILE_MISMATCH candidate={record.candidate}"
            )
        expected_record = SeqaxWeightPlacementPlan(
            candidate=expected.candidate.name,
            policy=expected.candidate.policy,
            distributed_schedule_sha256=expected.plan.distributed_schedule_sha256,
            physical_schedule_sha256=expected.plan.physical_schedule_sha256,
            pallas_source_sha256=expected.plan.source_sha256(),
            stablehlo_sha256=_sha256(plan_root / "stablehlo.txt"),
            compiler_hlo_sha256=_sha256(plan_root / "compiler_hlo.txt"),
            high_level_all_gathers=sum(
                isinstance(operation, AllGatherOp) for operation in expected.distributed.walk()
            ),
            physical_collectives=all_gathers + reduce_scatters,
            stablehlo_all_gathers=stablehlo_all_gathers,
            pallas_regions=expected.plan.pallas_region_count,
            parameter_bytes_per_device=parameter_residency_bytes_per_device(
                expected.plan.input_contracts,
                mesh=dict(expected.plan.mesh),
            ),
        )
        if (
            record != expected_record
            or (plan_root / "distributed.xdsl").read_text() != canonical_text(expected.distributed)
            or (plan_root / "physical.xdsl").read_text() != canonical_text(expected.physical)
            or (plan_root / "lowered_pallas.py").read_text()
            != expected.plan.render_executable_source()
            or json.loads((plan_root / "plan_manifest.json").read_text())
            != expected.plan.manifest()
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_PLAN_REPLAY_MISMATCH candidate={record.candidate}"
            )

    memory = tuple(
        SeqaxWeightResidencyObservation.model_validate(value)
        for value in json.loads((root / "memory.json").read_text())
    )
    if memory != result.memory or tuple(value.candidate for value in memory) != expected_names:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_MEMORY_RESULT_MISMATCH")
    for observation, candidate in zip(memory, trusted_contract.candidates, strict=True):
        expected_plan = next(value for value in prepared if value.candidate.name is candidate.name)
        timing_seed_index = trusted_contract.correctness_seeds.index(trusted_contract.timing_seed)
        expected_output_sha256 = next(
            value.output_sha256[timing_seed_index]
            for value in result.correctness
            if value.name == candidate.name
        )
        if (
            observation.parameter_bytes_per_device != candidate.expected_parameter_bytes_per_device
            or observation.runtime != trusted_contract.runtime
            or observation.devices != result.devices
            or observation.distributed_schedule_sha256
            != expected_plan.plan.distributed_schedule_sha256
            or observation.physical_schedule_sha256 != expected_plan.plan.physical_schedule_sha256
            or observation.pallas_source_sha256 != expected_plan.plan.source_sha256()
            or observation.stablehlo_sha256
            != next(
                value.stablehlo_sha256
                for value in result.plans
                if value.candidate is candidate.name
            )
            or observation.compiler_hlo_sha256
            != next(
                value.compiler_hlo_sha256
                for value in result.plans
                if value.candidate is candidate.name
            )
            or observation.source_commit != current_commit
            or observation.source_manifest != _source_manifest()
            or observation.timing_input_sha256 != arrays_sha256(timing_inputs)
            or observation.output_sha256 != expected_output_sha256
            or not observation.isolated_process
            or not observation.fits_observed_device_memory
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_MEMORY_REPLAY_MISMATCH candidate={candidate.name}"
            )

    correctness = tuple(
        SeqaxPallasCandidateCorrectness.model_validate(value)
        for value in json.loads((root / "correctness.json").read_text())
    )
    if correctness != result.correctness:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_CORRECTNESS_RESULT_MISMATCH")
    _validate_correctness(root / "correctness", trusted_contract, prepared, correctness)

    rounds = tuple(
        SeqaxPallasRoundObservation.model_validate(value)
        for value in json.loads((root / "rounds.json").read_text())
    )
    confirmation_rounds = tuple(
        SeqaxPallasRoundObservation.model_validate(value)
        for value in json.loads((root / "confirmation_rounds.json").read_text())
    )
    if rounds != result.rounds or confirmation_rounds != result.confirmation_rounds:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_TIMING_RESULT_MISMATCH")
    candidate_stats = candidate_statistics(trusted_contract, rounds)
    promotable = tuple(value for value in candidate_stats if value.promotable)
    provisional = (
        min(promotable, key=lambda value: value.median_round_ns).name if promotable else None
    )
    expected_confirmation = (
        confirmation_statistics(trusted_contract, provisional, confirmation_rounds)
        if provisional is not None
        else None
    )
    winner = (
        provisional
        if expected_confirmation is not None and expected_confirmation.confirmed
        else None
    )
    if (
        result.execution_orders != execution_orders(trusted_contract)
        or result.candidates != candidate_stats
        or result.provisional_winner != provisional
        or result.confirmation != expected_confirmation
        or result.winner != winner
        or (provisional is None and confirmation_rounds)
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_SELECTION_REPLAY_MISMATCH")

    run_id = semantic_sha256("seqax-weight-placement-run-v1", trusted_contract.search_id)
    ledger_payloads = (
        (RunState.CREATED, {"search_id": trusted_contract.search_id}),
        (
            RunState.VERIFIED,
            {
                "distributed_schedules": {
                    value.candidate.name: value.plan.distributed_schedule_sha256
                    for value in prepared
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
                "candidate_output_sha256": {
                    value.name: value.output_sha256 for value in result.correctness
                },
                "cpu_oracle_passed": result.correctness[0].cpu_oracle_passed,
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
    )
    if require_accepted:
        ledger_payloads += (
            (
                RunState.ACCEPTED,
                {"result_sha256": _sha256(root / "result.json"), "winner": result.winner},
            ),
        )
    history = read_ledger_history(root / "ledger.sqlite", run_id)
    if tuple(value.state for value in history) != tuple(value[0] for value in ledger_payloads):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_LEDGER_STATES_MISMATCH")
    if tuple(value.payload_sha256 for value in history) != tuple(
        ExperimentLedger.payload_sha256(payload) for _state, payload in ledger_payloads
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_LEDGER_PAYLOAD_MISMATCH")

    _validate_closed_world(
        root,
        _expected_files(root, trusted_contract, receipt_present=require_accepted),
    )
    if require_accepted:
        receipt = SeqaxWeightPlacementReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        if (
            receipt.search_id != trusted_contract.search_id
            or receipt.result_sha256 != _sha256(root / "result.json")
            or receipt.ledger_sha256 != _sha256(root / "ledger.sqlite")
            or receipt.artifacts != _artifact_manifest(root)
        ):
            raise ValueError("SEQAX_WEIGHT_PLACEMENT_RECEIPT_MISMATCH")
    return result


def validate_seqax_weight_placement(
    root: Path,
    trusted_contract: SeqaxWeightPlacementContract,
) -> SeqaxWeightPlacementResult:
    if root.is_symlink():
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_ROOT_INVALID path={root}")
    root = root.resolve()
    _preflight_existing_root(root)
    canonical = default_seqax_weight_placement_contract(trusted_contract.runtime)
    if trusted_contract != canonical:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_EXTERNAL_CONTRACT_MISMATCH")
    return _validate(root, trusted_contract, require_accepted=True)
