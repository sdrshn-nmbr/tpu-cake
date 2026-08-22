from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from tpu_cake.artifacts import (
    build_artifact_manifest,
)
from tpu_cake.artifacts import (
    file_sha256 as _sha256,
)
from tpu_cake.artifacts import save_array as _save_array
from tpu_cake.artifacts import (
    write_json as _write_json,
)
from tpu_cake.artifacts import (
    write_text as _write_text,
)
from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import (
    validate_compiler_analysis,
    write_compiler_analysis,
)
from tpu_cake.contracts import ArtifactReference, ArtifactRole, SourceFileContract
from tpu_cake.dialects.distributed_tensor import AllGatherOp
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import EvidenceRun, RunState, payload_sha256, read_ledger_history
from tpu_cake.runner import _runtime_identity, _source_state
from tpu_cake.seqax_pallas_runner import (
    _physical_collective_counts,
    _validate_compiled_program,
)
from tpu_cake.seqax_pallas_search import (
    SeqaxPallasCandidateCorrectness,
    SeqaxPallasRoundObservation,
)
from tpu_cake.seqax_pallas_search_runner import (
    _compiler_tile_metadata,
    _validate_output_abi,
)
from tpu_cake.seqax_weight_confirmation import (
    SeqaxWeightConfirmationContract,
    SeqaxWeightConfirmationPlanIdentity,
    SeqaxWeightConfirmationReceipt,
    SeqaxWeightConfirmationResult,
    base_weight_placement_contract,
    confirmation_orders,
    confirmation_statistics,
    default_seqax_weight_confirmation_contract,
)
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementPlan,
    parameter_residency_bytes_per_device,
)
from tpu_cake.seqax_weight_placement_runner import (
    CompiledPlacement,
    PreparedPlacement,
    _candidate_correctness,
    _compile,
    _device_inventory,
    _execute,
    _load_array,
    _plan_record,
    _preflight_existing_root,
    _require_clean_repository,
    _require_safe_new_root,
    _resident_inputs,
    _source_manifest,
    _validate_correctness,
    _validate_devices,
    prepare_weight_placement_candidates,
)
from tpu_cake.stablehlo import StableHloInspector
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


def _validate_source_blobs(
    repository_root: Path,
    commit: str,
    manifest: tuple[SourceFileContract, ...],
) -> None:
    for source in manifest:
        blob = subprocess.run(
            ["git", "show", f"{commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(f"SEQAX_WEIGHT_CONFIRMATION_SOURCE_BLOB_MISMATCH path={source.path}")


def _validate_accepted_search_plans(
    contract: SeqaxWeightConfirmationContract,
    plans: tuple[SeqaxWeightPlacementPlan, ...],
) -> None:
    plan_identities = tuple(
        SeqaxWeightConfirmationPlanIdentity(
            candidate=value.candidate,
            distributed_schedule_sha256=value.distributed_schedule_sha256,
            physical_schedule_sha256=value.physical_schedule_sha256,
            pallas_source_sha256=value.pallas_source_sha256,
            stablehlo_sha256=value.stablehlo_sha256,
            compiler_hlo_sha256=value.compiler_hlo_sha256,
        )
        for value in plans
    )
    if plan_identities != contract.accepted_search_plans:
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_ACCEPTED_SEARCH_PLAN_MISMATCH")


def _timing_observations(
    contract: SeqaxWeightConfirmationContract,
    compiled: dict[str, CompiledPlacement],
    resident_inputs: dict[str, tuple[jax.Array, ...]],
) -> tuple[SeqaxPallasRoundObservation, ...]:
    observations = []
    for round_index, order in enumerate(confirmation_orders(contract)):
        for position, name in enumerate(order):
            samples = []
            for _ in range(contract.measured_iterations):
                started = time.perf_counter_ns()
                jax.block_until_ready(compiled[name].executable(*resident_inputs[name]))
                samples.append(time.perf_counter_ns() - started)
            observations.append(
                SeqaxPallasRoundObservation(
                    round_index=round_index,
                    position=position,
                    candidate=name,
                    samples_ns=tuple(samples),
                    median_ns=float(statistics.median(samples)),
                )
            )
    return tuple(observations)


def _expected_files(root: Path, *, receipt_present: bool) -> set[Path]:
    contract = SeqaxWeightConfirmationContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    base = base_weight_placement_contract(contract)
    expected = {
        root / "contract.json",
        root / "source_state.json",
        root / "source_diff.patch",
        root / "ledger.sqlite",
        root / "correctness.json",
        root / "rounds.json",
        root / "post_timing_outputs" / "sharded.npy",
        root / "post_timing_outputs" / "embedding-mlp.npy",
        root / "result.json",
    }
    for candidate in base.candidates:
        plan_root = root / "plans" / candidate.name
        expected.update(
            plan_root / name
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
            seed_root / "outputs" / f"{candidate.name}.npy" for candidate in base.candidates
        )
    if receipt_present:
        expected.add(root / "receipt.json")
    return {value.resolve() for value in expected}


def _validate_closed_world(root: Path, expected: set[Path]) -> None:
    _preflight_existing_root(root)
    observed = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if observed != expected:
        raise ValueError(
            "SEQAX_WEIGHT_CONFIRMATION_CLOSED_WORLD_MISMATCH "
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
    if "/outputs/" in value or value.startswith("post_timing_outputs/"):
        return ArtifactRole.CORRECTNESS_OUTPUT
    if value == "rounds.json":
        return ArtifactRole.TIMING_SAMPLES
    return ArtifactRole.SEARCH_EVIDENCE


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    return build_artifact_manifest(root, role_for_path=_artifact_role)


def run_seqax_weight_confirmation(
    root: Path,
    contract: SeqaxWeightConfirmationContract,
) -> SeqaxWeightConfirmationResult:
    root = root.resolve()
    _require_safe_new_root(root)
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
    runtime = _runtime_identity()
    if runtime != contract.runtime:
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_RUNTIME_MISMATCH")
    devices = tuple(jax.devices())
    base = base_weight_placement_contract(contract)
    _validate_devices(devices, base)
    root.mkdir(parents=True)
    _write_json(
        root / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    _source_state(repository_root, root)
    source_state_sha256 = _sha256(root / "source_state.json")
    run_id = semantic_sha256("seqax-weight-confirmation-run-v1", contract.confirmation_id)
    ledger_path = root / "ledger.sqlite"
    evidence_run = EvidenceRun(ledger_path, run_id)
    evidence_run.create({"confirmation_id": contract.confirmation_id})

    prepared = prepare_weight_placement_candidates(base)
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

    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=contract.timing_seed, **contract.parameters)
    )
    compiled = tuple(_compile(value, host_inputs, devices) for value in prepared)
    for value in compiled:
        plan_root = root / "plans" / value.prepared.candidate.name
        _write_text(plan_root / "stablehlo.txt", value.stablehlo + "\n")
        _write_text(plan_root / "compiler_hlo.txt", value.compiler_hlo + "\n")
        write_compiler_analysis(
            plan_root / "compiler_analysis.json",
            value.compiler_analysis,
        )
    plans = tuple(_plan_record(root, value) for value in compiled)
    evidence_run.transition(
        RunState.COMPILED,
        {"plans": [value.model_dump(mode="json") for value in plans]},
    )

    correctness = _candidate_correctness(root / "correctness", base, compiled)
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
    resident = {
        name: _resident_inputs(host_inputs, value.prepared, value.mesh)
        for name, value in compiled_by_name.items()
    }
    for name, value in compiled_by_name.items():
        for _ in range(contract.warmup_iterations):
            jax.block_until_ready(value.executable(*resident[name]))
    rounds = _timing_observations(contract, compiled_by_name, resident)
    statistics_record = confirmation_statistics(contract, rounds)
    post_outputs = tuple(
        _execute(compiled_by_name[name], resident[name])
        for name in (contract.baseline, contract.candidate)
    )
    for name, output in zip((contract.baseline, contract.candidate), post_outputs, strict=True):
        _save_array(root / "post_timing_outputs" / f"{name}.npy", output)
    timing_index = contract.correctness_seeds.index(contract.timing_seed)
    expected_post_hashes = tuple(
        next(value for value in correctness if value.name == name).output_sha256[timing_index]
        for name in (contract.baseline, contract.candidate)
    )
    post_hashes = tuple(array_sha256(value) for value in post_outputs)
    if post_hashes != expected_post_hashes or not np.array_equal(post_outputs[0], post_outputs[1]):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_POST_TIMING_PARITY_FAILED")
    _write_json(root / "rounds.json", [value.model_dump(mode="json") for value in rounds])
    winner = contract.candidate if statistics_record.confirmed else None
    result = SeqaxWeightConfirmationResult(
        confirmation_id=contract.confirmation_id,
        runtime=runtime,
        devices=_device_inventory(devices),
        timing_input_sha256=arrays_sha256(host_inputs),
        source_state_sha256=source_state_sha256,
        source_manifest=_source_manifest(),
        plans=plans,
        correctness=correctness,
        execution_orders=confirmation_orders(contract),
        rounds=rounds,
        post_timing_output_sha256=post_hashes,
        statistics=statistics_record,
        winner=winner,
        correctness_scope="incumbent-bit-exact",
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    evidence_run.transition(
        RunState.TIMED,
        {
            "round_count": contract.paired_rounds,
            "winner": winner,
            "confidence_level": contract.confidence_level,
        },
    )
    evidence_run.seal("SEQAX_WEIGHT_CONFIRMATION_LEDGER_SIDECARS")
    _validate(root, contract, require_accepted=False)
    evidence_run.transition(
        RunState.ACCEPTED,
        {"result_sha256": _sha256(root / "result.json"), "winner": winner},
    )
    evidence_run.seal("SEQAX_WEIGHT_CONFIRMATION_LEDGER_SIDECARS")
    receipt = SeqaxWeightConfirmationReceipt(
        confirmation_id=contract.confirmation_id,
        status="passed",
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(ledger_path),
        artifacts=_artifact_manifest(root),
    )
    _write_json(root / "receipt.json", receipt.model_dump(mode="json"))
    return validate_seqax_weight_confirmation(root, contract)


def _validate_plans(
    root: Path,
    base: SeqaxWeightPlacementContract,
    result: SeqaxWeightConfirmationResult,
) -> tuple[PreparedPlacement, ...]:
    prepared = prepare_weight_placement_candidates(base)
    if tuple(value.candidate for value in result.plans) != tuple(
        value.candidate.name for value in prepared
    ):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_PLAN_NAMES_MISMATCH")
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
                "SEQAX_WEIGHT_CONFIRMATION_STABLEHLO_GATHER_MISMATCH "
                f"candidate={expected.candidate.name}"
            )
        expected_tiles = tuple(
            (index, expected.plan.physical_schedule_sha256, *tiles)
            for index, tiles in enumerate(
                (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
                for operation in expected.physical.walk()
                if operation.name == "tpu_schedule.mxu_einsum"
            )
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
            or _compiler_tile_metadata(compiler_hlo) != expected_tiles
            or (plan_root / "distributed.xdsl").read_text() != canonical_text(expected.distributed)
            or (plan_root / "physical.xdsl").read_text() != canonical_text(expected.physical)
            or (plan_root / "lowered_pallas.py").read_text()
            != expected.plan.render_executable_source()
            or json.loads((plan_root / "plan_manifest.json").read_text())
            != expected.plan.manifest()
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_CONFIRMATION_PLAN_REPLAY_MISMATCH candidate={record.candidate}"
            )
    return prepared


def _validate(
    root: Path,
    contract: SeqaxWeightConfirmationContract,
    *,
    require_accepted: bool,
) -> SeqaxWeightConfirmationResult:
    _preflight_existing_root(root)
    saved = SeqaxWeightConfirmationContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved != contract:
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_CONTRACT_MISMATCH")
    result = SeqaxWeightConfirmationResult.model_validate_json((root / "result.json").read_text())
    if (
        result.confirmation_id != contract.confirmation_id
        or result.runtime != contract.runtime
        or result.correctness_scope != "incumbent-bit-exact"
        or tuple(value.id for value in result.devices) != tuple(range(contract.device_count))
        or any(value.platform != "tpu" for value in result.devices)
        or any(value.device_kind not in {"TPU7x", "TPU v7x"} for value in result.devices)
        or len({value.process_index for value in result.devices}) != 1
    ):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_RESULT_IDENTITY_MISMATCH")
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
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
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_SOURCE_STATE_MISMATCH")
    _validate_source_blobs(repository_root, current_commit, result.source_manifest)
    expected_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=contract.timing_seed, **contract.parameters)
    )
    if result.timing_input_sha256 != arrays_sha256(expected_inputs):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_TIMING_INPUT_MISMATCH")

    base = base_weight_placement_contract(contract)
    prepared = _validate_plans(root, base, result)
    _validate_accepted_search_plans(contract, result.plans)
    correctness = tuple(
        SeqaxPallasCandidateCorrectness.model_validate(value)
        for value in json.loads((root / "correctness.json").read_text())
    )
    if correctness != result.correctness:
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_CORRECTNESS_RESULT_MISMATCH")
    _validate_correctness(root / "correctness", base, prepared, correctness)

    rounds = tuple(
        SeqaxPallasRoundObservation.model_validate(value)
        for value in json.loads((root / "rounds.json").read_text())
    )
    statistics_record = confirmation_statistics(contract, rounds)
    if (
        rounds != result.rounds
        or result.execution_orders != confirmation_orders(contract)
        or result.statistics != statistics_record
        or result.winner != (contract.candidate if statistics_record.confirmed else None)
    ):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_STATISTICS_REPLAY_MISMATCH")
    timing_index = contract.correctness_seeds.index(contract.timing_seed)
    outputs = tuple(
        _load_array(root / "post_timing_outputs" / f"{name}.npy")
        for name in (contract.baseline, contract.candidate)
    )
    output_contracts = {value.candidate.name: value.plan.output_contracts[0] for value in prepared}
    for name, output in zip((contract.baseline, contract.candidate), outputs, strict=True):
        _validate_output_abi(output, output_contracts[name], name)
    expected_hashes = tuple(
        next(value for value in correctness if value.name == name).output_sha256[timing_index]
        for name in (contract.baseline, contract.candidate)
    )
    observed_hashes = tuple(array_sha256(value) for value in outputs)
    if (
        result.post_timing_output_sha256 != observed_hashes
        or observed_hashes != expected_hashes
        or not np.array_equal(outputs[0], outputs[1])
    ):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_POST_TIMING_REPLAY_MISMATCH")

    run_id = semantic_sha256("seqax-weight-confirmation-run-v1", contract.confirmation_id)
    ledger_payloads = (
        (RunState.CREATED, {"confirmation_id": contract.confirmation_id}),
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
        (RunState.COMPILED, {"plans": [value.model_dump(mode="json") for value in result.plans]}),
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
                "round_count": contract.paired_rounds,
                "winner": result.winner,
                "confidence_level": contract.confidence_level,
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
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_LEDGER_STATES_MISMATCH")
    if tuple(value.payload_sha256 for value in history) != tuple(
        payload_sha256(payload) for _state, payload in ledger_payloads
    ):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_LEDGER_PAYLOAD_MISMATCH")

    _validate_closed_world(root, _expected_files(root, receipt_present=require_accepted))
    if require_accepted:
        receipt = SeqaxWeightConfirmationReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        if (
            receipt.confirmation_id != contract.confirmation_id
            or receipt.result_sha256 != _sha256(root / "result.json")
            or receipt.ledger_sha256 != _sha256(root / "ledger.sqlite")
            or receipt.artifacts != _artifact_manifest(root)
        ):
            raise ValueError("SEQAX_WEIGHT_CONFIRMATION_RECEIPT_MISMATCH")
    return result


def validate_seqax_weight_confirmation(
    root: Path,
    contract: SeqaxWeightConfirmationContract,
) -> SeqaxWeightConfirmationResult:
    if root.is_symlink():
        raise ValueError(f"SEQAX_WEIGHT_CONFIRMATION_ROOT_INVALID path={root}")
    root = root.resolve()
    _preflight_existing_root(root)
    if contract != default_seqax_weight_confirmation_contract(contract.runtime):
        raise ValueError("SEQAX_WEIGHT_CONFIRMATION_EXTERNAL_CONTRACT_MISMATCH")
    return _validate(root, contract, require_accepted=True)
