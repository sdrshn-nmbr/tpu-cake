from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import statistics
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import numpy as np

from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import write_compiler_analysis
from tpu_cake.contracts import ArtifactReference, ArtifactRole, SourceFileContract
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history, seal_ledger
from tpu_cake.runner import _runtime_identity, _source_state
from tpu_cake.seqax_numerical import default_seqax_bf16_validation_contract
from tpu_cake.seqax_pallas_search_runner import _validate_output_abi
from tpu_cake.seqax_residual_confirmation import (
    SEQAX_RESIDUAL_CONFIRMATION_SCHEMA,
    SeqaxResidualConfirmationContract,
    SeqaxResidualConfirmationPlan,
    SeqaxResidualConfirmationReceipt,
    SeqaxResidualConfirmationResult,
    SeqaxResidualConfirmationRunIdentity,
    SeqaxResidualTimingOutput,
    SeqaxResidualTimingRound,
    confirmation_orders,
    confirmation_statistics,
    default_seqax_residual_confirmation_contract,
)
from tpu_cake.seqax_residual_profile import default_seqax_residual_profile_contract
from tpu_cake.seqax_residual_profile_runner import (
    CompiledResidualProfile,
    PreparedResidualProfile,
    _compile,
    _correctness_observation,
    _device_inventory,
    _execute,
    _expected_plan_files,
    _json_sha256,
    _load_array,
    _prepare_candidates,
    _replay_correctness,
    _resident_inputs,
    _save_array,
    _sha256,
    _validate_devices,
    _validate_verifier_runtime,
    _write_json,
    _write_text,
)
from tpu_cake.seqax_residual_profile_runner import (
    _source_manifest as _profile_source_manifest,
)
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


def _write_json_atomic_at(path: Path, value: object, temporary_parent: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_parent / (
        f".{path.parent.name}-{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_json_atomic_at(path, value, path.parent.parent)


def _write_sibling_json_atomic(path: Path, value: object) -> None:
    _write_json_atomic_at(path, value, path.parent)


def _source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    additions = (
        package / "seqax_residual_confirmation.py",
        package / "seqax_residual_confirmation_runner.py",
    )
    manifest = {
        value.path: value
        for value in (
            *_profile_source_manifest(),
            *(
                SourceFileContract(
                    path=path.relative_to(package.parent).as_posix(),
                    sha256=_sha256(path),
                )
                for path in additions
            ),
        )
    }
    return tuple(manifest[path] for path in sorted(manifest))


def _require_clean_repository(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_SOURCE_DIRTY status={status}")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_PATH_SYMLINK path={current}")


def _require_safe_root(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == value or root in value.parents for value in protected) or (
        repository_root in root.parents
    ):
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_UNSAFE_ROOT path={root}")


def _require_compilation_root(
    repository_root: Path,
    contract: SeqaxResidualConfirmationContract,
) -> None:
    if repository_root.resolve() != Path(contract.compilation_source_root):
        raise ValueError(
            "SEQAX_RESIDUAL_CONFIRMATION_COMPILATION_ROOT_MISMATCH "
            f"expected={contract.compilation_source_root} observed={repository_root}"
        )


def _preflight_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_HARDLINK path={path}")


@contextmanager
def _exclusive_run_lock(root: Path) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-residual-confirmation-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_root_stat = lock_root.lstat()
    if (
        not stat.S_ISDIR(lock_root_stat.st_mode)
        or lock_root_stat.st_uid != os.getuid()
        or lock_root_stat.st_mode & 0o077
    ):
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_LOCK_ROOT_INVALID path={lock_root}")
    lock_name = hashlib.sha256(str(root).encode()).hexdigest()
    descriptor = os.open(
        lock_root / f"{lock_name}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        lock_info = os.fstat(descriptor)
        if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1:
            raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_LOCK_FILE_INVALID path={root}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_RUN_LOCKED path={root}") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _read_resume_ledger(path: Path) -> tuple[tuple[str, RunState], ...]:
    sidecars = (path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm"))
    if not any(value.exists() for value in sidecars):
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT run_id, state FROM events ORDER BY sequence"
            ).fetchall()
    else:
        with tempfile.TemporaryDirectory(prefix="seqax-residual-confirmation-ledger-") as directory:
            temporary = Path(directory) / path.name
            shutil.copy2(path, temporary)
            for sidecar in sidecars:
                if sidecar.exists():
                    shutil.copy2(sidecar, temporary.with_name(sidecar.name))
            uri = f"{temporary.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                rows = connection.execute(
                    "SELECT run_id, state FROM events ORDER BY sequence"
                ).fetchall()
    return tuple((run_id, RunState(state)) for run_id, state in rows)


def _expected_files(
    root: Path,
    contract: SeqaxResidualConfirmationContract,
    *,
    receipt_present: bool,
) -> set[Path]:
    expected = {
        root / "run_identity.json",
        root / "contract.json",
        root / "source_state.json",
        root / "source_diff.patch",
        root / "source_manifest.json",
        root / "ledger.sqlite",
        root / "correctness.json",
        root / "rounds.json",
        root / "result.json",
    }
    plan_files = (
        "distributed.xdsl",
        "physical.xdsl",
        "lowered_pallas.py",
        "plan_manifest.json",
        "pallas_stablehlo.txt",
        "pallas_compiler_hlo.txt",
        "control_stablehlo.txt",
        "control_compiler_hlo.txt",
        "pallas_compiler_analysis.json",
        "control_compiler_analysis.json",
        "pre_timing_output.npy",
        "post_timing_output.npy",
    )
    for candidate in contract.plans:
        candidate_root = root / "candidates" / candidate.candidate
        expected.update(candidate_root / name for name in plan_files)
        for seed in contract.correctness_seeds:
            seed_root = candidate_root / "correctness" / str(seed)
            expected.update(seed_root / "inputs" / f"{index:02d}.npy" for index in range(13))
            expected.update(seed_root / name for name in ("cpu.npy", "control.npy", "pallas.npy"))
    if receipt_present:
        expected.add(root / "receipt.json")
    return {path.resolve() for path in expected}


def _prepare_output_root(
    root: Path,
    identity: SeqaxResidualConfirmationRunIdentity,
    contract: SeqaxResidualConfirmationContract,
) -> tuple[RunState, ...]:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return ()
    _preflight_root(root)
    if not any(root.iterdir()):
        return ()
    identity_path = root / "run_identity.json"
    if not identity_path.is_file():
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ROOT_NOT_OWNED path={root}")
    if (
        SeqaxResidualConfirmationRunIdentity.model_validate_json(identity_path.read_text())
        != identity
    ):
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ROOT_NOT_OWNED path={root}")
    if (root / "receipt.json").exists():
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ACCEPTED_ROOT_NOT_RETRYABLE path={root}")
    if (root / "failure.json").exists():
        return _archive_incomplete_root(root)
    allowed = _expected_files(root, contract, receipt_present=False)
    ledger_path = root / "ledger.sqlite"
    allowed.update(
        {
            ledger_path.with_name(f"{ledger_path.name}-wal").resolve(),
            ledger_path.with_name(f"{ledger_path.name}-shm").resolve(),
        }
    )
    observed = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if not observed.issubset(allowed):
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ROOT_NOT_OWNED path={root}")
    if not ledger_path.exists():
        return ()
    try:
        rows = _read_resume_ledger(ledger_path)
    except (OSError, sqlite3.Error, ValueError):
        return _archive_incomplete_root(root)
    states = tuple(state for run_id, state in rows if run_id == identity.run_id)
    if len(states) != len(rows):
        return _archive_incomplete_root(root)
    expected = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.TIMED,
        RunState.ACCEPTED,
    )
    if states != expected[: len(states)]:
        return _archive_incomplete_root(root)
    print(f"SEQAX_RESIDUAL_CONFIRMATION_RESUMING run_id={identity.run_id} root={root}")
    return states


def _archive_incomplete_root(root: Path) -> tuple[RunState, ...]:
    while True:
        archived = root.with_name(f"{root.name}.incomplete-{time.time_ns()}")
        if not archived.exists() and not archived.is_symlink():
            break
    root.rename(archived)
    root.mkdir(parents=True, exist_ok=False)
    print(f"SEQAX_RESIDUAL_CONFIRMATION_ARCHIVED source={root} archive={archived}")
    return ()


def _record_state(
    ledger_path: Path,
    run_id: str,
    state: RunState,
    payload: Mapping[str, object],
) -> None:
    expected_hash = ExperimentLedger.payload_sha256(payload)
    if ledger_path.exists():
        existing = {value.state: value for value in read_ledger_history(ledger_path, run_id)}
        if state in existing:
            if existing[state].payload_sha256 != expected_hash:
                raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_LEDGER_CONFLICT state={state}")
            return
    with ExperimentLedger(ledger_path) as ledger:
        if state is RunState.CREATED:
            ledger.create(run_id, payload)
        else:
            ledger.transition(run_id, state, payload)


def _record_failure(root: Path, run_id: str, error: Exception) -> None:
    receipt_path = root / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        failure_path = root.with_name(f"{root.name}.failure.json")
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_sibling_json_atomic(
                failure_path,
                {
                    "run_id": run_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        print(f"SEQAX_RESIDUAL_CONFIRMATION_POST_RECEIPT_FAILURE path={failure_path}")
        return
    if not (root / "run_identity.json").is_file():
        return
    _write_json_atomic(
        root / "failure.json",
        {
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    ledger_path = root / "ledger.sqlite"
    if ledger_path.exists():
        with ExperimentLedger(ledger_path) as ledger:
            state = ledger.current_state(run_id)
            if state not in {None, RunState.ACCEPTED, RunState.REJECTED}:
                ledger.transition(
                    run_id,
                    RunState.REJECTED,
                    {"error_type": type(error).__name__, "error": str(error)},
                )
        seal_ledger(ledger_path, "SEQAX_RESIDUAL_PROFILE_LEDGER_SIDECARS")


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    fixed = {
        "run_identity.json": ArtifactRole.EXPERIMENT,
        "contract.json": ArtifactRole.EXPERIMENT,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "source_manifest.json": ArtifactRole.BACKEND_MANIFEST,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "correctness.json": ArtifactRole.SEARCH_EVIDENCE,
        "rounds.json": ArtifactRole.TIMING_SAMPLES,
        "result.json": ArtifactRole.SEARCH_RESULT,
    }
    if relative in fixed:
        return fixed[relative]
    if not relative.startswith("candidates/"):
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ARTIFACT_UNRECOGNIZED path={relative}")
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
        "pre_timing_output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "post_timing_output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "cpu.npy": ArtifactRole.ORACLE_OUTPUT,
        "control.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "pallas.npy": ArtifactRole.CORRECTNESS_OUTPUT,
    }
    if path.name in roles:
        return roles[path.name]
    if "/correctness/" in relative and "/inputs/" in relative and relative.endswith(".npy"):
        return ArtifactRole.CORRECTNESS_INPUT
    raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ARTIFACT_UNRECOGNIZED path={relative}")


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    return tuple(
        ArtifactReference(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            role=_artifact_role(path.relative_to(root)),
        )
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.name not in {"receipt.json", "failure.json"}
    )


def _validate_manifest(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    declared = tuple(value.path for value in artifacts)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "receipt.json"
    }
    if len(declared) != len(set(declared)) or set(declared) != observed:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_CLOSED_WORLD_MISMATCH")
    for artifact in artifacts:
        path = root / artifact.path
        if (
            path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != artifact.size_bytes
            or _sha256(path) != artifact.sha256
            or _artifact_role(Path(artifact.path)) is not artifact.role
        ):
            raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ARTIFACT_MISMATCH path={artifact.path}")


def _plan_record(prepared: PreparedResidualProfile) -> SeqaxResidualConfirmationPlan:
    expected = prepared.expected
    return SeqaxResidualConfirmationPlan(
        candidate=expected.candidate,
        distributed_schedule_sha256=prepared.plan.distributed_schedule_sha256,
        physical_schedule_sha256=prepared.plan.physical_schedule_sha256,
        pallas_source_sha256=prepared.plan.source_sha256(),
        pallas_manifest_sha256=_json_sha256(prepared.plan.manifest()),
        pallas_stablehlo_sha256=expected.pallas_stablehlo_sha256,
        control_stablehlo_sha256=expected.control_stablehlo_sha256,
        expected_pallas_compiler_collectives=expected.expected_pallas_compiler_collectives,
        pallas_regions=expected.expected_pallas_regions,
        all_gathers=expected.expected_all_gathers,
        all_reduces=expected.expected_all_reduces,
        reduce_scatters=expected.expected_reduce_scatters,
    )


def _timing_observations(
    contract: SeqaxResidualConfirmationContract,
    compiled: dict[SeqaxResidualNormStrategy, CompiledResidualProfile],
    resident: dict[SeqaxResidualNormStrategy, tuple[jax.Array, ...]],
) -> tuple[SeqaxResidualTimingRound, ...]:
    observations = []
    for round_index, order in enumerate(confirmation_orders(contract)):
        for position, candidate in enumerate(order):
            samples = []
            for _ in range(contract.measured_iterations):
                started = time.perf_counter_ns()
                outputs = compiled[candidate].pallas_executable(*resident[candidate])
                jax.block_until_ready(outputs)
                samples.append(time.perf_counter_ns() - started)
            observations.append(
                SeqaxResidualTimingRound(
                    round_index=round_index,
                    position=position,
                    candidate=candidate,
                    samples_ns=tuple(samples),
                    median_ns=float(statistics.median(samples)),
                )
            )
    return tuple(observations)


def _validate_source(root: Path, result: SeqaxResidualConfirmationResult) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = tuple(
        SourceFileContract.model_validate(value)
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    state = json.loads((root / "source_state.json").read_text())
    commit = subprocess.run(
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
        or state.get("git_commit") != commit
        or state.get("git_dirty") is not False
        or state.get("git_status") != []
        or state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
        or (root / "source_diff.patch").read_bytes() != b""
    ):
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_SOURCE_MISMATCH")
    for source in manifest:
        blob = subprocess.run(
            ["git", "show", f"{commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_SOURCE_BLOB_MISMATCH path={source.path}")


def _ledger_payloads(
    contract: SeqaxResidualConfirmationContract,
    result: SeqaxResidualConfirmationResult,
) -> tuple[tuple[RunState, dict[str, object]], ...]:
    payloads: tuple[tuple[RunState, dict[str, object]], ...] = (
        (
            RunState.CREATED,
            {
                "confirmation_id": contract.confirmation_id,
                "source_profile_receipt_sha256": contract.source_profile_receipt_sha256,
                "devices": [value.model_dump(mode="json") for value in result.devices],
            },
        ),
        (
            RunState.VERIFIED,
            {
                "numerical_contract_id": contract.numerical_contract_id,
                "distributed_schedules": {
                    value.candidate: value.distributed_schedule_sha256 for value in result.plans
                },
            },
        ),
        (
            RunState.LOWERED,
            {
                "physical_schedules": {
                    value.candidate: value.physical_schedule_sha256 for value in result.plans
                },
                "pallas_sources": {
                    value.candidate: value.pallas_source_sha256 for value in result.plans
                },
            },
        ),
        (
            RunState.COMPILED,
            {
                "plans": [value.model_dump(mode="json") for value in result.plans],
            },
        ),
        (
            RunState.CORRECT,
            {"outputs": {value.candidate: value.sha256 for value in result.pre_timing_outputs}},
        ),
        (
            RunState.TIMED,
            {
                "round_count": contract.paired_rounds,
                "winner": result.winner,
                "confidence_level": contract.confidence_level,
                "statistics": result.statistics.model_dump(mode="json"),
                "post_timing_outputs": {
                    value.candidate: value.sha256 for value in result.post_timing_outputs
                },
            },
        ),
    )
    return payloads


def _validate(
    root: Path,
    contract: SeqaxResidualConfirmationContract,
    *,
    require_accepted: bool,
    require_receipt: bool,
) -> SeqaxResidualConfirmationResult:
    _preflight_root(root)
    saved_contract = SeqaxResidualConfirmationContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != contract:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_CONTRACT_MISMATCH")
    identity = SeqaxResidualConfirmationRunIdentity.model_validate_json(
        (root / "run_identity.json").read_text()
    )
    result = SeqaxResidualConfirmationResult.model_validate_json((root / "result.json").read_text())
    source_state = json.loads((root / "source_state.json").read_text())
    if (
        identity.schema_version != SEQAX_RESIDUAL_CONFIRMATION_SCHEMA
        or identity.confirmation_id != contract.confirmation_id
        or identity.source_commit != source_state.get("git_commit")
    ):
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_RUN_IDENTITY_MISMATCH")
    expected_devices = tuple(
        type(result.devices[0])(
            id=index,
            process_index=0,
            platform="tpu",
            device_kind=contract.device_kind,
        )
        for index in range(contract.device_count)
    )
    if (
        result.confirmation_id != contract.confirmation_id
        or result.run_id != identity.run_id
        or result.source_profile_id != contract.source_profile_id
        or result.source_profile_receipt_sha256 != contract.source_profile_receipt_sha256
        or result.numerical_contract_id != contract.numerical_contract_id
        or result.runtime != contract.runtime
        or result.devices != expected_devices
    ):
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_RESULT_IDENTITY_MISMATCH")
    _validate_source(root, result)
    expected_run_id = semantic_sha256(
        SEQAX_RESIDUAL_CONFIRMATION_SCHEMA,
        contract.confirmation_id,
        identity.source_commit,
    )
    if result.run_id != expected_run_id:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_RUN_ID_MISMATCH")
    prepared = _prepare_candidates(default_seqax_residual_profile_contract(contract.runtime))
    expected_plans = tuple(_plan_record(value) for value in prepared)
    if result.plans != expected_plans:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_PLAN_IDENTITY_MISMATCH")
    for value in prepared:
        _expected_plan_files(root, value)
    correctness_file = tuple(
        type(result.correctness[0]).model_validate_json(json.dumps(value))
        for value in json.loads((root / "correctness.json").read_text())
    )
    if correctness_file != result.correctness:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_CORRECTNESS_FILE_MISMATCH")
    for value in prepared:
        saved = tuple(
            observation
            for observation in result.correctness
            if observation.candidate is value.expected.candidate
        )
        _replay_correctness(root=root, prepared=value, saved=saved)
    numerical = default_seqax_bf16_validation_contract()
    scenario = next(
        value for value in numerical.scenarios if value.name == "calibration-m256-b2-s1-l1"
    )
    timing_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **scenario.parameters.model_dump(),
        )
    )
    if result.timing_input_sha256 != arrays_sha256(timing_inputs):
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_TIMING_INPUT_MISMATCH")
    timing_observations = {
        value.candidate: value for value in result.correctness if value.seed == contract.timing_seed
    }
    pre_outputs = []
    post_outputs = []
    for value in prepared:
        candidate = value.expected.candidate
        candidate_root = root / "candidates" / candidate
        pre = _load_array(candidate_root / "pre_timing_output.npy")
        post = _load_array(candidate_root / "post_timing_output.npy")
        _validate_output_abi(pre, value.plan.output_contracts[0], str(candidate))
        _validate_output_abi(post, value.plan.output_contracts[0], str(candidate))
        expected_hash = timing_observations[candidate].pallas_output_sha256
        if array_sha256(pre) != expected_hash or array_sha256(post) != expected_hash:
            raise ValueError(
                f"SEQAX_RESIDUAL_CONFIRMATION_POST_TIMING_REPLAY_MISMATCH candidate={candidate}"
            )
        pre_outputs.append(SeqaxResidualTimingOutput(candidate=candidate, sha256=expected_hash))
        post_outputs.append(SeqaxResidualTimingOutput(candidate=candidate, sha256=expected_hash))
    rounds = tuple(
        SeqaxResidualTimingRound.model_validate_json(json.dumps(value))
        for value in json.loads((root / "rounds.json").read_text())
    )
    statistics_record = confirmation_statistics(contract, rounds)
    expected_winner = contract.candidate if statistics_record.confirmed else None
    if (
        result.pre_timing_outputs != tuple(pre_outputs)
        or result.post_timing_outputs != tuple(post_outputs)
        or result.execution_orders != confirmation_orders(contract)
        or result.rounds != rounds
        or result.statistics != statistics_record
        or result.winner != expected_winner
    ):
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_STATISTICS_REPLAY_MISMATCH")
    payloads = _expected_ledger_payloads(
        contract,
        result,
        root=root,
        include_accepted=require_accepted,
    )
    history = read_ledger_history(root / "ledger.sqlite", result.run_id)
    if tuple(value.state for value in history) != tuple(state for state, _ in payloads):
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_LEDGER_STATE_MISMATCH")
    if tuple(value.payload_sha256 for value in history) != tuple(
        ExperimentLedger.payload_sha256(payload) for _, payload in payloads
    ):
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_LEDGER_PAYLOAD_MISMATCH")
    expected_files = _expected_files(root, contract, receipt_present=require_receipt)
    observed_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_CLOSED_WORLD_MISMATCH")
    if require_receipt:
        receipt = SeqaxResidualConfirmationReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        _validate_manifest(root, receipt.artifacts)
        expected_receipt = SeqaxResidualConfirmationReceipt(
            confirmation_id=contract.confirmation_id,
            run_id=result.run_id,
            status="passed",
            result_sha256=_sha256(root / "result.json"),
            ledger_sha256=_sha256(root / "ledger.sqlite"),
            artifacts=_artifact_manifest(root),
        )
        if receipt != expected_receipt:
            raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_RECEIPT_MISMATCH")
    return result


def _expected_ledger_payloads(
    contract: SeqaxResidualConfirmationContract,
    result: SeqaxResidualConfirmationResult,
    *,
    root: Path,
    include_accepted: bool,
) -> tuple[tuple[RunState, dict[str, object]], ...]:
    payloads = _ledger_payloads(contract, result)
    if include_accepted:
        payloads += (
            (
                RunState.ACCEPTED,
                {
                    "result_sha256": _sha256(root / "result.json"),
                    "winner": result.winner,
                },
            ),
        )
    return payloads


def _publish_receipt(
    root: Path,
    contract: SeqaxResidualConfirmationContract,
    result: SeqaxResidualConfirmationResult,
) -> SeqaxResidualConfirmationResult:
    receipt = SeqaxResidualConfirmationReceipt(
        confirmation_id=contract.confirmation_id,
        run_id=result.run_id,
        status="passed",
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(root / "ledger.sqlite"),
        artifacts=_artifact_manifest(root),
    )
    _write_json_atomic(root / "receipt.json", receipt.model_dump(mode="json"))
    return _validate(
        root,
        contract,
        require_accepted=True,
        require_receipt=True,
    )


def _execute_confirmation(
    root: Path,
    contract: SeqaxResidualConfirmationContract,
    devices: tuple[Any, ...],
    identity: SeqaxResidualConfirmationRunIdentity,
) -> SeqaxResidualConfirmationResult:
    _write_json(
        root / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    repository_root = Path(__file__).resolve().parents[2]
    _source_state(repository_root, root)
    manifest = _source_manifest()
    _write_json(
        root / "source_manifest.json",
        [value.model_dump(mode="json") for value in manifest],
    )
    device_inventory = _device_inventory(devices)
    run_id = semantic_sha256(
        SEQAX_RESIDUAL_CONFIRMATION_SCHEMA,
        contract.confirmation_id,
        identity.source_commit,
    )
    if run_id != identity.run_id:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_RUN_ID_MISMATCH")
    ledger_path = root / "ledger.sqlite"
    created_payload = {
        "confirmation_id": contract.confirmation_id,
        "source_profile_receipt_sha256": contract.source_profile_receipt_sha256,
        "devices": [value.model_dump(mode="json") for value in device_inventory],
    }
    _record_state(ledger_path, run_id, RunState.CREATED, created_payload)
    profile_contract = default_seqax_residual_profile_contract(contract.runtime)
    prepared = _prepare_candidates(profile_contract)
    plans = tuple(_plan_record(value) for value in prepared)
    _record_state(
        ledger_path,
        run_id,
        RunState.VERIFIED,
        {
            "numerical_contract_id": contract.numerical_contract_id,
            "distributed_schedules": {
                value.candidate: value.distributed_schedule_sha256 for value in plans
            },
        },
    )
    for value in prepared:
        candidate_root = root / "candidates" / value.expected.candidate
        _write_text(candidate_root / "distributed.xdsl", canonical_text(value.distributed))
        _write_text(candidate_root / "physical.xdsl", canonical_text(value.physical))
        _write_text(candidate_root / "lowered_pallas.py", value.plan.render_executable_source())
        _write_json(candidate_root / "plan_manifest.json", value.plan.manifest())
    _record_state(
        ledger_path,
        run_id,
        RunState.LOWERED,
        {
            "physical_schedules": {
                value.candidate: value.physical_schedule_sha256 for value in plans
            },
            "pallas_sources": {value.candidate: value.pallas_source_sha256 for value in plans},
        },
    )
    numerical = default_seqax_bf16_validation_contract()
    scenario = next(
        value for value in numerical.scenarios if value.name == "calibration-m256-b2-s1-l1"
    )
    timing_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **scenario.parameters.model_dump(),
        )
    )
    compiled = tuple(_compile(value, timing_inputs, devices) for value in prepared)
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
    _record_state(
        ledger_path,
        run_id,
        RunState.COMPILED,
        {"plans": [value.model_dump(mode="json") for value in plans]},
    )
    observations = []
    pre_outputs = []
    resident: dict[SeqaxResidualNormStrategy, tuple[jax.Array, ...]] = {}
    compiled_by_candidate = {value.prepared.expected.candidate: value for value in compiled}
    for value in compiled:
        candidate = value.prepared.expected.candidate
        candidate_root = root / "candidates" / candidate
        for seed in contract.correctness_seeds:
            inputs = tuple(
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
                    host_inputs=inputs,
                    seed=seed,
                )
            )
        resident[candidate] = _resident_inputs(timing_inputs, value.prepared, value.mesh)
        output = _execute(value.pallas_executable, resident[candidate])
        expected_hash = next(
            item.pallas_output_sha256
            for item in observations
            if item.candidate is candidate and item.seed == contract.timing_seed
        )
        if array_sha256(output) != expected_hash:
            raise ValueError(
                f"SEQAX_RESIDUAL_CONFIRMATION_PRE_TIMING_MISMATCH candidate={candidate}"
            )
        _save_array(candidate_root / "pre_timing_output.npy", output)
        pre_outputs.append(SeqaxResidualTimingOutput(candidate=candidate, sha256=expected_hash))
    correctness = tuple(observations)
    _write_json(
        root / "correctness.json",
        [value.model_dump(mode="json") for value in correctness],
    )
    _record_state(
        ledger_path,
        run_id,
        RunState.CORRECT,
        {"outputs": {value.candidate: value.sha256 for value in pre_outputs}},
    )
    for candidate in (contract.baseline, contract.candidate):
        for _ in range(contract.warmup_iterations):
            outputs = compiled_by_candidate[candidate].pallas_executable(*resident[candidate])
            jax.block_until_ready(outputs)
    rounds = _timing_observations(contract, compiled_by_candidate, resident)
    statistics_record = confirmation_statistics(contract, rounds)
    _write_json(root / "rounds.json", [value.model_dump(mode="json") for value in rounds])
    post_outputs = []
    for candidate in (
        SeqaxResidualNormStrategy.STANDARD,
        SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
    ):
        output = _execute(compiled_by_candidate[candidate].pallas_executable, resident[candidate])
        expected_hash = next(value.sha256 for value in pre_outputs if value.candidate is candidate)
        if array_sha256(output) != expected_hash:
            raise ValueError(
                f"SEQAX_RESIDUAL_CONFIRMATION_POST_TIMING_MISMATCH candidate={candidate}"
            )
        _save_array(root / "candidates" / candidate / "post_timing_output.npy", output)
        post_outputs.append(SeqaxResidualTimingOutput(candidate=candidate, sha256=expected_hash))
    winner = contract.candidate if statistics_record.confirmed else None
    result = SeqaxResidualConfirmationResult(
        confirmation_id=contract.confirmation_id,
        run_id=run_id,
        source_profile_id=contract.source_profile_id,
        source_profile_receipt_sha256=contract.source_profile_receipt_sha256,
        numerical_contract_id=contract.numerical_contract_id,
        runtime=contract.runtime,
        devices=device_inventory,
        source_state_sha256=_sha256(root / "source_state.json"),
        source_manifest_sha256=_sha256(root / "source_manifest.json"),
        source_manifest=manifest,
        plans=plans,
        correctness=correctness,
        timing_input_sha256=arrays_sha256(timing_inputs),
        pre_timing_outputs=tuple(pre_outputs),
        execution_orders=confirmation_orders(contract),
        rounds=rounds,
        post_timing_outputs=tuple(post_outputs),
        statistics=statistics_record,
        winner=winner,
        claim_scope="fixed-model256-layer1-sequence1-bf16-pallas-performance",
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    _record_state(
        ledger_path,
        run_id,
        RunState.TIMED,
        _expected_ledger_payloads(
            contract,
            result,
            root=root,
            include_accepted=False,
        )[-1][1],
    )
    seal_ledger(ledger_path, "SEQAX_RESIDUAL_PROFILE_LEDGER_SIDECARS")
    _validate(
        root,
        contract,
        require_accepted=False,
        require_receipt=False,
    )
    _record_state(
        ledger_path,
        run_id,
        RunState.ACCEPTED,
        {
            "result_sha256": _sha256(root / "result.json"),
            "winner": winner,
        },
    )
    seal_ledger(ledger_path, "SEQAX_RESIDUAL_PROFILE_LEDGER_SIDECARS")
    return _publish_receipt(root, contract, result)


def run_seqax_residual_confirmation(
    root: Path,
    contract: SeqaxResidualConfirmationContract,
) -> SeqaxResidualConfirmationResult:
    canonical = default_seqax_residual_confirmation_contract(contract.runtime)
    if contract != canonical:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_EXTERNAL_CONTRACT_MISMATCH")
    _reject_symlink_components(root)
    if root.is_symlink():
        raise ValueError(f"SEQAX_RESIDUAL_CONFIRMATION_ROOT_INVALID path={root}")
    root = root.resolve()
    _require_safe_root(root)
    with _exclusive_run_lock(root):
        repository_root = Path(__file__).resolve().parents[2]
        _require_clean_repository(repository_root)
        if (root / "receipt.json").is_file():
            return validate_seqax_residual_confirmation(root, contract)
        _require_compilation_root(repository_root, contract)
        runtime = _runtime_identity()
        if runtime != contract.runtime:
            raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_RUNTIME_MISMATCH")
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        device_inventory = tuple(jax.devices())
        profile_contract = default_seqax_residual_profile_contract(contract.runtime)
        _validate_devices(device_inventory, profile_contract)
        identity_seed = semantic_sha256(
            SEQAX_RESIDUAL_CONFIRMATION_SCHEMA,
            contract.confirmation_id,
            source_commit,
        )
        identity = SeqaxResidualConfirmationRunIdentity(
            confirmation_id=contract.confirmation_id,
            run_id=identity_seed,
            source_commit=source_commit,
        )
        states = _prepare_output_root(root, identity, contract)
        _write_json_atomic(root / "run_identity.json", identity.model_dump(mode="json"))
        try:
            if states and states[-1] in {RunState.TIMED, RunState.ACCEPTED}:
                result = _validate(
                    root,
                    contract,
                    require_accepted=states[-1] is RunState.ACCEPTED,
                    require_receipt=False,
                )
                if states[-1] is RunState.TIMED:
                    _record_state(
                        root / "ledger.sqlite",
                        result.run_id,
                        RunState.ACCEPTED,
                        {
                            "result_sha256": _sha256(root / "result.json"),
                            "winner": result.winner,
                        },
                    )
                    seal_ledger(
                        root / "ledger.sqlite",
                        "SEQAX_RESIDUAL_PROFILE_LEDGER_SIDECARS",
                    )
                return _publish_receipt(root, contract, result)
            return _execute_confirmation(root, contract, device_inventory, identity)
        except Exception as error:
            _record_failure(root, identity.run_id, error)
            raise


def validate_seqax_residual_confirmation(
    root: Path,
    trusted_contract: SeqaxResidualConfirmationContract,
) -> SeqaxResidualConfirmationResult:
    canonical = default_seqax_residual_confirmation_contract(trusted_contract.runtime)
    if trusted_contract != canonical:
        raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_EXTERNAL_CONTRACT_MISMATCH")
    _reject_symlink_components(root)
    root = root.resolve()
    _validate_verifier_runtime()
    return _validate(
        root,
        trusted_contract,
        require_accepted=True,
        require_receipt=True,
    )
