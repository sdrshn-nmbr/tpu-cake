from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from pydantic import BaseModel, ConfigDict, Field, model_validator
from xdsl.utils.exceptions import VerifyException

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.jax_lowering import JaxDistributedMeshPlan, lower_distributed_program_to_jax_mesh
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.runner import _runtime_identity, _source_state
from tpu_cake.seqax_numerical import (
    SeqaxBf16NumericalAssessment,
    SeqaxBf16NumericalScenario,
    SeqaxBf16OutputAssessment,
    SeqaxBf16ValidationContract,
    SeqaxDiscriminatorClause,
    SeqaxInputMutation,
    SeqaxNumericalDiscriminator,
    _assess_output_arrays,
    _relative_l2,
    _row_scaled_max,
    _validate_strict_silu_stablehlo,
    assess_seqax_bf16_forward,
    assess_seqax_bf16_outputs,
    assess_seqax_cpu_reference_replay,
    canonical_seqax_stablehlo,
    decode_seqax_bf16_checkpoint,
    default_seqax_bf16_validation_contract,
    encode_seqax_bf16_checkpoint,
    mutate_seqax_forward_inputs,
    seqax_discriminator_clause,
    seqax_gate_projection_reference_float32,
    seqax_up_projection_reference_float32,
    validate_activation_mutant_stablehlo,
    validate_instrumented_strict_silu_stablehlo,
    validate_strict_silu_stablehlo,
)
from tpu_cake.seqax_pallas_lowering import (
    SeqaxPallasPlan,
    _parse_physical,
    lower_seqax_physical_to_pallas,
)
from tpu_cake.seqax_pallas_runner import (
    _compiler_hlo,
    _physical_collective_counts,
    _validate_compiled_program,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import SeqaxNumericalSemantics, seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import seqax_forward_canonical_reference, seqax_forward_inputs

SEQAX_BF16_RUN_SCHEMA = "seqax-bf16-forward-validation-run-v5"
_STRICT_MLP_CHECKPOINT_SPECS = (
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
)


class SeqaxBf16Device(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0)
    process_index: int = Field(ge=0)
    platform: str = Field(min_length=1)
    device_kind: str = Field(min_length=1)


class SeqaxBf16Runtime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    runtime: RuntimeIdentity
    ml_dtypes: str = Field(min_length=1)
    cpu_machine: str = Field(min_length=1)
    cpu_system: str = Field(min_length=1)


class SeqaxBf16RunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = Field(pattern=r"^seqax-bf16-forward-validation-run-v5$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class SeqaxBf16PlanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario: str = Field(min_length=1)
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_pallas_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_control_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_region_count: int = Field(gt=0)
    all_gather_count: int = Field(ge=0)
    reduce_scatter_count: int = Field(ge=0)
    strict_silu_count: int = Field(gt=0)
    strict_hidden_count: int = Field(gt=0)


class SeqaxBf16InstrumentationDifference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pallas_exact: bool
    control_exact: bool
    pallas_relative_l2: float = Field(ge=0)
    control_relative_l2: float = Field(ge=0)
    pallas_row_scaled_max: float = Field(ge=0)
    control_row_scaled_max: float = Field(ge=0)

    @property
    def exact(self) -> bool:
        return self.pallas_exact and self.control_exact


class SeqaxBf16SeedObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario: str = Field(min_length=1)
    seed: int = Field(ge=0)
    input_sha256: tuple[str, ...] = Field(min_length=13, max_length=13)
    cpu_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_pallas_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_control_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_rms_input_sha256: tuple[str, ...] = Field(min_length=1)
    control_rms_input_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_rms_mean_square_sha256: tuple[str, ...] = Field(min_length=1)
    control_rms_mean_square_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_rms_inverse_sha256: tuple[str, ...] = Field(min_length=1)
    control_rms_inverse_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_normalized_float32_sha256: tuple[str, ...] = Field(min_length=1)
    control_normalized_float32_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_normalized_input_sha256: tuple[str, ...] = Field(min_length=1)
    control_normalized_input_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_gate_float32_sha256: tuple[str, ...] = Field(min_length=1)
    control_gate_float32_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_gate_sha256: tuple[str, ...] = Field(min_length=1)
    control_gate_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_silu_sha256: tuple[str, ...] = Field(min_length=1)
    control_silu_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_up_float32_sha256: tuple[str, ...] = Field(min_length=1)
    control_up_float32_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_up_sha256: tuple[str, ...] = Field(min_length=1)
    control_up_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_hidden_sha256: tuple[str, ...] = Field(min_length=1)
    control_hidden_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_down_float32_sha256: tuple[str, ...] = Field(min_length=1)
    control_down_float32_sha256: tuple[str, ...] = Field(min_length=1)
    pallas_down_bfloat16_sha256: tuple[str, ...] = Field(min_length=1)
    control_down_bfloat16_sha256: tuple[str, ...] = Field(min_length=1)
    normal_assessment: SeqaxBf16OutputAssessment
    instrumented_assessment: SeqaxBf16NumericalAssessment
    instrumentation_difference: SeqaxBf16InstrumentationDifference

    @model_validator(mode="after")
    def input_hashes_are_valid(self) -> SeqaxBf16SeedObservation:
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.input_sha256):
            raise ValueError("Seqax BF16 input artifact hash is invalid")
        return self


class SeqaxBf16DiscriminatorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    discriminator: SeqaxNumericalDiscriminator
    clause: SeqaxDiscriminatorClause
    artifact_paths: tuple[str, ...] = Field(min_length=1)
    artifact_sha256: tuple[str, ...] = Field(min_length=1)
    rejected: bool
    failure: str = Field(min_length=1)

    @model_validator(mode="after")
    def artifacts_match(self) -> SeqaxBf16DiscriminatorObservation:
        if len(self.artifact_paths) != len(self.artifact_sha256):
            raise ValueError("Seqax BF16 discriminator artifact identity count mismatch")
        if self.clause is not seqax_discriminator_clause(self.discriminator):
            raise ValueError("Seqax BF16 discriminator clause mismatch")
        return self


class SeqaxBf16ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = Field(pattern=r"^seqax-bf16-forward-validation-run-v5$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: SeqaxBf16Runtime
    devices: tuple[SeqaxBf16Device, ...] = Field(min_length=8, max_length=8)
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    plans: tuple[SeqaxBf16PlanRecord, ...] = Field(min_length=10, max_length=10)
    observations: tuple[SeqaxBf16SeedObservation, ...] = Field(min_length=41, max_length=41)
    discriminators: tuple[SeqaxBf16DiscriminatorObservation, ...] = Field(min_length=16)
    passed: bool
    claim_scope: str = Field(pattern=r"^declared-surface-dual-jax-cpu-bf16-numerical-agreement-v1$")


class SeqaxBf16ValidationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = Field(pattern=r"^seqax-bf16-forward-validation-receipt-v5$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str = Field(pattern=r"^passed$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)


@dataclass(frozen=True)
class _CompiledPath:
    plan: SeqaxPallasPlan | JaxDistributedMeshPlan
    executable: Any
    mesh: Any
    stablehlo: str
    compiler_hlo: str


@dataclass(frozen=True)
class _InstrumentedPath:
    plan: SeqaxPallasPlan | JaxDistributedMeshPlan
    executable: Any
    mesh: Any
    stablehlo: str
    compiler_hlo: str


@dataclass(frozen=True)
class _CompiledScenario:
    scenario: SeqaxBf16NumericalScenario
    distributed: Any
    physical: Any
    pallas: _CompiledPath
    control: _CompiledPath
    instrumented_pallas: _InstrumentedPath
    instrumented_control: _InstrumentedPath
    record: SeqaxBf16PlanRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent.parent / (
        f".{path.parent.name}-{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value), allow_pickle=False)


def _load_array(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"SEQAX_BF16_ARRAY_INVALID path={path}")
    return np.load(path, allow_pickle=False)


def _source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "canonical.py",
        package / "cli.py",
        package / "contracts.py",
        package / "dtensor_interpreter.py",
        package / "identity.py",
        package / "jax_lowering.py",
        package / "ledger.py",
        package / "lowering.py",
        package / "physical_geometry.py",
        package / "runner.py",
        package / "seqax_numerical.py",
        package / "seqax_numerical_runner.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_pallas_runner.py",
        package / "seqax_physical_execution.py",
        package / "seqax_physical_lowering.py",
        package / "seqax_runner.py",
        package / "dialects" / "distributed_tensor.py",
        package / "dialects" / "tpu_schedule.py",
        package / "workloads" / "seqax_forward.py",
        package / "workloads" / "seqax_oracle.py",
    )
    return tuple(
        SourceFileContract(path=path.relative_to(package.parent).as_posix(), sha256=_sha256(path))
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
        raise ValueError(f"SEQAX_BF16_SOURCE_DIRTY status={status}")


def _require_safe_root(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == value or root in value.parents for value in protected) or (
        repository_root in root.parents
    ):
        raise ValueError(f"SEQAX_BF16_UNSAFE_ROOT path={root}")


@contextmanager
def _exclusive_run_lock(root: Path) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-seqax-bf16-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_root_stat = lock_root.lstat()
    if (
        not stat.S_ISDIR(lock_root_stat.st_mode)
        or lock_root_stat.st_uid != os.getuid()
        or lock_root_stat.st_mode & 0o077
    ):
        raise ValueError(f"SEQAX_BF16_LOCK_ROOT_INVALID path={lock_root}")
    lock_name = hashlib.sha256(str(root).encode()).hexdigest()
    descriptor = os.open(
        lock_root / f"{lock_name}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise ValueError(f"SEQAX_BF16_LOCK_FILE_INVALID path={root}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"SEQAX_BF16_RUN_LOCKED path={root}") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _prepare_output_root(
    root: Path,
    identity: SeqaxBf16RunIdentity,
    contract: SeqaxBf16ValidationContract,
) -> Path | None:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return None
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_BF16_ROOT_INVALID path={root}")
    if not any(root.iterdir()):
        return None
    _preflight_existing_root(root)
    identity_path = root / "run_identity.json"
    if not identity_path.is_file():
        raise ValueError(f"SEQAX_BF16_ROOT_NOT_OWNED path={root}")
    saved = SeqaxBf16RunIdentity.model_validate_json(identity_path.read_text())
    if saved != identity:
        raise ValueError(f"SEQAX_BF16_ROOT_NOT_OWNED path={root}")
    if (root / "receipt.json").exists():
        raise ValueError(f"SEQAX_BF16_ACCEPTED_ROOT_NOT_RETRYABLE path={root}")
    if _root_is_resumable(root, identity, contract):
        print(f"SEQAX_BF16_RESUMING run_id={identity.run_id} root={root}")
        return None
    while True:
        archived = root.with_name(f"{root.name}.incomplete-{time.time_ns()}")
        if not archived.exists() and not archived.is_symlink():
            break
    root.rename(archived)
    root.mkdir(parents=True, exist_ok=False)
    print(f"SEQAX_BF16_ARCHIVED_INCOMPLETE source={root} archive={archived}")
    return archived


def _root_is_resumable(
    root: Path,
    identity: SeqaxBf16RunIdentity,
    contract: SeqaxBf16ValidationContract,
) -> bool:
    if (root / "failure.json").exists() or (root / "failure.json").is_symlink():
        return False
    ledger_path = root / "ledger.sqlite"
    allowed = _expected_files(root, contract, receipt_present=False)
    allowed.update(
        {
            ledger_path.with_name(f"{ledger_path.name}-shm").resolve(),
            ledger_path.with_name(f"{ledger_path.name}-wal").resolve(),
        }
    )
    observed = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if not observed.issubset(allowed):
        return False
    if not ledger_path.exists():
        return True
    try:
        rows = _read_resume_ledger(ledger_path)
        if any(run_id != identity.run_id for run_id, _state in rows):
            return False
        states = tuple(RunState(state) for _run_id, state in rows)
    except (OSError, sqlite3.Error, ValueError):
        return False
    expected_prefix = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.VALIDATED,
        RunState.ACCEPTED,
    )
    return states == expected_prefix[: len(states)]


def _read_resume_ledger(ledger_path: Path) -> list[tuple[str, str]]:
    sidecars = (
        ledger_path.with_name(f"{ledger_path.name}-shm"),
        ledger_path.with_name(f"{ledger_path.name}-wal"),
    )
    if not any(path.exists() for path in sidecars):
        uri = f"{ledger_path.resolve().as_uri()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            return connection.execute(
                "SELECT run_id, state FROM events ORDER BY sequence"
            ).fetchall()
    with tempfile.TemporaryDirectory(prefix="seqax-bf16-ledger-inspection-") as directory:
        temporary_root = Path(directory)
        temporary_ledger = temporary_root / ledger_path.name
        shutil.copy2(ledger_path, temporary_ledger)
        for sidecar in sidecars:
            if sidecar.exists():
                shutil.copy2(sidecar, temporary_root / sidecar.name)
        uri = f"{temporary_ledger.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            return connection.execute(
                "SELECT run_id, state FROM events ORDER BY sequence"
            ).fetchall()


def _preflight_existing_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_BF16_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_BF16_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_BF16_HARDLINK path={path}")


def _close_ledger(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    sidecars = (path.with_name(f"{path.name}-shm"), path.with_name(f"{path.name}-wal"))
    if any(value.exists() for value in sidecars):
        raise ValueError("SEQAX_BF16_LEDGER_SIDECAR_PRESENT")


def _transition_or_replay(
    ledger: ExperimentLedger,
    run_id: str,
    state: RunState,
    payload: dict[str, object],
) -> None:
    expected_hash = ExperimentLedger.payload_sha256(payload)
    existing = next((event for event in ledger.history(run_id) if event.state is state), None)
    if existing is not None:
        if existing.payload_sha256 != expected_hash:
            raise ValueError(f"SEQAX_BF16_LEDGER_REPLAY_MISMATCH state={state.value}")
        return
    if state is RunState.CREATED:
        ledger.create(run_id, payload)
    else:
        ledger.transition(run_id, state, payload)


def _record_failure(root: Path, run_id: str, error: Exception) -> None:
    if (root / "receipt.json").exists() or (root / "receipt.json").is_symlink():
        failure_path = root.with_name(f"{root.name}.failure.json")
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_json(
                failure_path,
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "run_id": run_id,
                },
            )
        print(f"SEQAX_BF16_POST_RECEIPT_FAILURE path={failure_path}")
        return
    ledger_path = root / "ledger.sqlite"
    state: str | None = None
    if ledger_path.is_file() and not ledger_path.is_symlink() and ledger_path.stat().st_nlink == 1:
        try:
            with ExperimentLedger(ledger_path) as ledger:
                current = ledger.current_state(run_id)
                state = current.value if current is not None else None
                if current not in {None, RunState.ACCEPTED, RunState.REJECTED}:
                    ledger.transition(
                        run_id,
                        RunState.REJECTED,
                        {
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "previous_state": state,
                        },
                    )
            _close_ledger(ledger_path)
        except (OSError, sqlite3.Error, ValueError) as ledger_error:
            print(
                "SEQAX_BF16_FAILURE_LEDGER_ERROR "
                f"error={type(ledger_error).__name__}:{ledger_error}"
            )
    failure_path = root / "failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        print(f"SEQAX_BF16_FAILURE_RECORD_EXISTS path={failure_path}")
        return
    _write_json(
        failure_path,
        {
            "error_type": type(error).__name__,
            "message": str(error),
            "previous_state": state,
            "run_id": run_id,
        },
    )


def _device_inventory(devices: tuple[Any, ...]) -> tuple[SeqaxBf16Device, ...]:
    return tuple(
        SeqaxBf16Device(
            id=value.id,
            process_index=value.process_index,
            platform=value.platform,
            device_kind=value.device_kind,
        )
        for value in devices
    )


def _validate_devices(devices: tuple[Any, ...], contract: SeqaxBf16ValidationContract) -> None:
    inventory = _device_inventory(devices)
    if (
        len(inventory) != contract.device_count
        or tuple(value.id for value in inventory) != tuple(range(contract.device_count))
        or len({value.process_index for value in inventory}) != 1
        or any(value.platform != contract.backend for value in inventory)
        or any(value.device_kind not in {contract.device_kind, "TPU v7x"} for value in inventory)
    ):
        raise ValueError(f"SEQAX_BF16_DEVICE_MISMATCH devices={inventory}")


def _resident_inputs(host_inputs: tuple[np.ndarray, ...], plan: Any, mesh: Any) -> tuple[Any, ...]:
    return tuple(
        jax.device_put(
            jnp.asarray(value),
            NamedSharding(mesh, tensor_contract.partition_spec()),
        )
        for value, tensor_contract in zip(host_inputs, plan.input_contracts, strict=True)
    )


def _execute_outputs(executable: Any, inputs: tuple[Any, ...]) -> tuple[np.ndarray, ...]:
    outputs = executable(*inputs)
    jax.block_until_ready(outputs)
    return tuple(np.asarray(value) for value in outputs)


def _execute(executable: Any, inputs: tuple[Any, ...]) -> np.ndarray:
    outputs = _execute_outputs(executable, inputs)
    if len(outputs) != 1:
        raise ValueError("SEQAX_BF16_OUTPUT_COUNT_MISMATCH")
    return outputs[0]


def _compile_path(
    plan: SeqaxPallasPlan | JaxDistributedMeshPlan,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
    *,
    pallas: bool,
    interpret_pallas: bool = False,
) -> _CompiledPath:
    if pallas:
        callable_, mesh = plan.build(interpret=interpret_pallas, devices=devices)
    else:
        callable_, mesh = plan.build(devices=devices)
    resident = _resident_inputs(host_inputs, plan, mesh)
    lowered = callable_.lower(*resident)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiler_hlo = _compiler_hlo(lowered)
    return _CompiledPath(
        plan=plan,
        executable=lowered.compile(),
        mesh=mesh,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
    )


def _compile_instrumented_pallas(
    plan: SeqaxPallasPlan,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
    *,
    expected_layers: int,
    interpret: bool = False,
) -> _InstrumentedPath:
    callable_, mesh = plan.build_with_strict_mlp_checkpoints(
        expected_layers=expected_layers,
        checkpoint_specs=_STRICT_MLP_CHECKPOINT_SPECS,
        interpret=interpret,
        devices=devices,
    )
    resident = _resident_inputs(host_inputs, plan, mesh)
    lowered = callable_.lower(*resident)
    return _InstrumentedPath(
        plan=plan,
        executable=lowered.compile(),
        mesh=mesh,
        stablehlo=str(lowered.compiler_ir(dialect="stablehlo")),
        compiler_hlo=_compiler_hlo(lowered),
    )


def _compile_instrumented_control(
    plan: JaxDistributedMeshPlan,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
    *,
    expected_layers: int,
) -> _InstrumentedPath:
    callable_, mesh = plan.build_with_strict_mlp_checkpoints(
        expected_layers=expected_layers,
        checkpoint_specs=_STRICT_MLP_CHECKPOINT_SPECS,
        devices=devices,
    )
    resident = _resident_inputs(host_inputs, plan, mesh)
    lowered = callable_.lower(*resident)
    return _InstrumentedPath(
        plan=plan,
        executable=lowered.compile(),
        mesh=mesh,
        stablehlo=str(lowered.compiler_ir(dialect="stablehlo")),
        compiler_hlo=_compiler_hlo(lowered),
    )


def _split_instrumented_outputs(
    outputs: tuple[np.ndarray, ...],
    scenario: SeqaxBf16NumericalScenario,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    expected_count = 1 + 13 * scenario.parameters.layers
    if len(outputs) != expected_count:
        raise ValueError(
            "SEQAX_BF16_INSTRUMENTED_OUTPUT_COUNT_MISMATCH "
            f"expected={expected_count} observed={len(outputs)}"
        )
    checkpoints = outputs[1:]
    return (
        outputs[0],
        checkpoints[0::13],
        checkpoints[1::13],
        checkpoints[2::13],
        checkpoints[3::13],
        checkpoints[4::13],
        checkpoints[5::13],
        checkpoints[6::13],
        checkpoints[7::13],
        checkpoints[8::13],
        checkpoints[9::13],
        checkpoints[10::13],
        checkpoints[11::13],
        checkpoints[12::13],
    )


def _plan_root(root: Path, scenario: SeqaxBf16NumericalScenario) -> Path:
    return root / "plans" / scenario.name


def _prepare_scenario(
    root: Path,
    scenario: SeqaxBf16NumericalScenario,
    devices: tuple[Any, ...],
) -> _CompiledScenario:
    parameters = scenario.parameters.model_dump()
    host_inputs = tuple(
        np.asarray(value) for value in seqax_forward_inputs(seed=scenario.seeds[0], **parameters)
    )
    distributed = seqax_forward_schedule(
        **parameters,
        numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
    )
    physical = lower_seqax_forward_to_physical(distributed).module
    pallas_plan = lower_seqax_physical_to_pallas(distributed, physical)
    control_plan = lower_distributed_program_to_jax_mesh(distributed)
    pallas = _compile_path(pallas_plan, host_inputs, devices, pallas=True)
    control = _compile_path(control_plan, host_inputs, devices, pallas=False)
    validate_strict_silu_stablehlo(
        pallas.stablehlo,
        expected_count=parameters["layers"],
        expected_sha256=scenario.pallas_stablehlo_sha256,
    )
    validate_strict_silu_stablehlo(
        control.stablehlo,
        expected_count=parameters["layers"],
        expected_sha256=scenario.control_stablehlo_sha256,
    )
    all_gathers, reduce_scatters = _physical_collective_counts(physical)
    _validate_compiled_program(
        pallas.stablehlo,
        pallas.compiler_hlo,
        pallas_region_count=pallas_plan.pallas_region_count,
        pallas_vector_region_count=pallas_plan.pallas_vector_region_count,
        all_gather_count=all_gathers,
        reduce_scatter_count=reduce_scatters,
    )
    instrumented_pallas = _compile_instrumented_pallas(
        pallas_plan,
        host_inputs,
        devices,
        expected_layers=parameters["layers"],
    )
    instrumented_control = _compile_instrumented_control(
        control_plan,
        host_inputs,
        devices,
        expected_layers=parameters["layers"],
    )
    validate_instrumented_strict_silu_stablehlo(
        instrumented_pallas.stablehlo,
        expected_count=parameters["layers"],
        expected_sha256=scenario.instrumented_pallas_stablehlo_sha256,
    )
    validate_instrumented_strict_silu_stablehlo(
        instrumented_control.stablehlo,
        expected_count=parameters["layers"],
        expected_sha256=scenario.instrumented_control_stablehlo_sha256,
    )
    plan_root = _plan_root(root, scenario)
    files = {
        "distributed.xdsl": canonical_text(distributed),
        "physical.xdsl": canonical_text(physical),
        "lowered_pallas.py": pallas_plan.render_executable_source(),
        "lowered_control.py": control_plan.render_executable_source(),
        "pallas_stablehlo.txt": canonical_seqax_stablehlo(pallas.stablehlo),
        "pallas_compiler_hlo.txt": pallas.compiler_hlo + "\n",
        "control_stablehlo.txt": canonical_seqax_stablehlo(control.stablehlo),
        "control_compiler_hlo.txt": control.compiler_hlo + "\n",
        "instrumented_pallas_stablehlo.txt": canonical_seqax_stablehlo(
            instrumented_pallas.stablehlo
        ),
        "instrumented_pallas_compiler_hlo.txt": instrumented_pallas.compiler_hlo + "\n",
        "instrumented_control_stablehlo.txt": canonical_seqax_stablehlo(
            instrumented_control.stablehlo
        ),
        "instrumented_control_compiler_hlo.txt": instrumented_control.compiler_hlo + "\n",
    }
    for name, value in files.items():
        _write_text(plan_root / name, value)
    _write_json(plan_root / "pallas_manifest.json", pallas_plan.manifest())
    _write_json(plan_root / "control_manifest.json", control_plan.manifest())
    record = SeqaxBf16PlanRecord(
        scenario=scenario.name,
        distributed_schedule_sha256=pallas_plan.distributed_schedule_sha256,
        physical_schedule_sha256=pallas_plan.physical_schedule_sha256,
        pallas_source_sha256=pallas_plan.source_sha256(),
        control_source_sha256=control_plan.source_sha256(),
        pallas_stablehlo_sha256=_sha256(plan_root / "pallas_stablehlo.txt"),
        pallas_compiler_hlo_sha256=_sha256(plan_root / "pallas_compiler_hlo.txt"),
        control_stablehlo_sha256=_sha256(plan_root / "control_stablehlo.txt"),
        control_compiler_hlo_sha256=_sha256(plan_root / "control_compiler_hlo.txt"),
        instrumented_pallas_stablehlo_sha256=_sha256(
            plan_root / "instrumented_pallas_stablehlo.txt"
        ),
        instrumented_pallas_compiler_hlo_sha256=_sha256(
            plan_root / "instrumented_pallas_compiler_hlo.txt"
        ),
        instrumented_control_stablehlo_sha256=_sha256(
            plan_root / "instrumented_control_stablehlo.txt"
        ),
        instrumented_control_compiler_hlo_sha256=_sha256(
            plan_root / "instrumented_control_compiler_hlo.txt"
        ),
        pallas_region_count=pallas_plan.pallas_region_count,
        all_gather_count=all_gathers,
        reduce_scatter_count=reduce_scatters,
        strict_silu_count=parameters["layers"],
        strict_hidden_count=parameters["layers"],
    )
    _write_json(plan_root / "plan.json", record.model_dump(mode="json"))
    return _CompiledScenario(
        scenario=scenario,
        distributed=distributed,
        physical=physical,
        pallas=pallas,
        control=control,
        instrumented_pallas=instrumented_pallas,
        instrumented_control=instrumented_control,
        record=record,
    )


def _seed_root(root: Path, scenario: str, seed: int) -> Path:
    return root / "scenarios" / scenario / f"seed-{seed}"


def _checkpoint_hashes(values: tuple[np.ndarray, ...]) -> tuple[str, ...]:
    return tuple(array_sha256(value) for value in values)


def _save_checkpoints(
    root: Path,
    prefix: str,
    rms_inputs: tuple[np.ndarray, ...],
    rms_mean_square: tuple[np.ndarray, ...],
    rms_inverse: tuple[np.ndarray, ...],
    normalized_float32: tuple[np.ndarray, ...],
    normalized_inputs: tuple[np.ndarray, ...],
    gate_float32: tuple[np.ndarray, ...],
    gates: tuple[np.ndarray, ...],
    silus: tuple[np.ndarray, ...],
    up_float32: tuple[np.ndarray, ...],
    up: tuple[np.ndarray, ...],
    hidden: tuple[np.ndarray, ...],
    down_float32: tuple[np.ndarray, ...],
    down_bfloat16: tuple[np.ndarray, ...],
    scenario: SeqaxBf16NumericalScenario,
) -> None:
    for layer, (
        rms_input,
        rms_mean_square_value,
        rms_inverse_value,
        normalized_float32_value,
        normalized_input,
        gate_float32_value,
        gate,
        silu,
        up_float32_value,
        up_value,
        hidden_value,
        down_float32_value,
        down_bfloat16_value,
        rms_input_contract,
        rms_mean_square_contract,
        rms_inverse_contract,
        normalized_float32_contract,
        normalized_input_contract,
        gate_float32_contract,
        gate_contract,
        silu_contract,
        up_float32_contract,
        up_contract,
        hidden_contract,
        down_float32_contract,
        down_bfloat16_contract,
    ) in enumerate(
        zip(
            rms_inputs,
            rms_mean_square,
            rms_inverse,
            normalized_float32,
            normalized_inputs,
            gate_float32,
            gates,
            silus,
            up_float32,
            up,
            hidden,
            down_float32,
            down_bfloat16,
            scenario.rms_input_checkpoints,
            scenario.rms_mean_square_checkpoints,
            scenario.rms_inverse_checkpoints,
            scenario.normalized_float32_checkpoints,
            scenario.normalized_input_checkpoints,
            scenario.gate_float32_checkpoints,
            scenario.gate_checkpoints,
            scenario.silu_checkpoints,
            scenario.up_float32_checkpoints,
            scenario.up_checkpoints,
            scenario.hidden_checkpoints,
            scenario.down_float32_checkpoints,
            scenario.down_bfloat16_checkpoints,
            strict=True,
        )
    ):
        _save_array(
            root / "checkpoints" / f"{prefix}_rms_input_{layer:02d}.npy",
            encode_seqax_bf16_checkpoint(rms_input, rms_input_contract),
        )
        for label, value, contract in (
            ("RMS_MEAN_SQUARE", rms_mean_square_value, rms_mean_square_contract),
            ("RMS_INVERSE", rms_inverse_value, rms_inverse_contract),
            ("NORMALIZED_FLOAT32", normalized_float32_value, normalized_float32_contract),
        ):
            if value.dtype != np.float32 or value.shape != contract.shape:
                raise ValueError(f"SEQAX_BF16_{label}_CHECKPOINT_ABI_MISMATCH")
        _save_array(
            root / "checkpoints" / f"{prefix}_rms_mean_square_{layer:02d}.npy",
            rms_mean_square_value,
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_rms_inverse_{layer:02d}.npy",
            rms_inverse_value,
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_normalized_float32_{layer:02d}.npy",
            normalized_float32_value,
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_normalized_input_{layer:02d}.npy",
            encode_seqax_bf16_checkpoint(normalized_input, normalized_input_contract),
        )
        if gate_float32_value.dtype != np.float32 or (
            gate_float32_value.shape != gate_float32_contract.shape
        ):
            raise ValueError("SEQAX_BF16_GATE_FLOAT32_CHECKPOINT_ABI_MISMATCH")
        _save_array(
            root / "checkpoints" / f"{prefix}_gate_float32_{layer:02d}.npy",
            gate_float32_value,
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_gate_{layer:02d}.npy",
            encode_seqax_bf16_checkpoint(gate, gate_contract),
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_silu_{layer:02d}.npy",
            encode_seqax_bf16_checkpoint(silu, silu_contract),
        )
        if up_float32_value.dtype != np.float32 or (
            up_float32_value.shape != up_float32_contract.shape
        ):
            raise ValueError("SEQAX_BF16_UP_FLOAT32_CHECKPOINT_ABI_MISMATCH")
        _save_array(
            root / "checkpoints" / f"{prefix}_up_float32_{layer:02d}.npy",
            up_float32_value,
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_up_{layer:02d}.npy",
            encode_seqax_bf16_checkpoint(up_value, up_contract),
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_hidden_{layer:02d}.npy",
            encode_seqax_bf16_checkpoint(hidden_value, hidden_contract),
        )
        if down_float32_value.dtype != np.float32 or (
            down_float32_value.shape != down_float32_contract.shape
        ):
            raise ValueError("SEQAX_BF16_DOWN_FLOAT32_CHECKPOINT_ABI_MISMATCH")
        _save_array(
            root / "checkpoints" / f"{prefix}_down_float32_{layer:02d}.npy",
            down_float32_value,
        )
        _save_array(
            root / "checkpoints" / f"{prefix}_down_bfloat16_{layer:02d}.npy",
            encode_seqax_bf16_checkpoint(down_bfloat16_value, down_bfloat16_contract),
        )


def _instrumentation_difference(
    pallas: np.ndarray,
    control: np.ndarray,
    instrumented_pallas: np.ndarray,
    instrumented_control: np.ndarray,
    contract: SeqaxBf16ValidationContract,
) -> SeqaxBf16InstrumentationDifference:
    decimals = contract.policy.metric_quantization_decimals
    scale_floor = contract.policy.row_scale_floor
    return SeqaxBf16InstrumentationDifference(
        pallas_exact=bool(np.array_equal(pallas, instrumented_pallas)),
        control_exact=bool(np.array_equal(control, instrumented_control)),
        pallas_relative_l2=_relative_l2(
            pallas,
            instrumented_pallas,
            quantization_decimals=decimals,
        ),
        control_relative_l2=_relative_l2(
            control,
            instrumented_control,
            quantization_decimals=decimals,
        ),
        pallas_row_scaled_max=_row_scaled_max(
            pallas,
            instrumented_pallas,
            scale_floor=scale_floor,
            quantization_decimals=decimals,
        ),
        control_row_scaled_max=_row_scaled_max(
            control,
            instrumented_control,
            scale_floor=scale_floor,
            quantization_decimals=decimals,
        ),
    )


def _run_seed(
    root: Path,
    compiled: _CompiledScenario,
    seed: int,
    contract: SeqaxBf16ValidationContract,
) -> SeqaxBf16SeedObservation:
    scenario = compiled.scenario
    parameters = scenario.parameters.model_dump()
    host_inputs = tuple(
        np.asarray(value) for value in seqax_forward_inputs(seed=seed, **parameters)
    )
    seed_root = _seed_root(root, scenario.name, seed)
    for index, value in enumerate(host_inputs):
        _save_array(seed_root / "inputs" / f"{index:02d}.npy", value)
    cpu_reference = np.asarray(
        seqax_forward_canonical_reference(
            host_inputs,
            quantization_decimals=contract.policy.cpu_reference_quantization_decimals,
            **parameters,
        )
    )
    _save_array(seed_root / "cpu_reference.npy", cpu_reference)

    pallas_inputs = _resident_inputs(host_inputs, compiled.pallas.plan, compiled.pallas.mesh)
    control_inputs = _resident_inputs(host_inputs, compiled.control.plan, compiled.control.mesh)
    pallas_output = _execute(compiled.pallas.executable, pallas_inputs)
    control_output = _execute(compiled.control.executable, control_inputs)
    _save_array(seed_root / "pallas_output.npy", pallas_output)
    _save_array(seed_root / "control_output.npy", control_output)

    instrumented_pallas_inputs = _resident_inputs(
        host_inputs,
        compiled.instrumented_pallas.plan,
        compiled.instrumented_pallas.mesh,
    )
    instrumented_pallas_outputs = _execute_outputs(
        compiled.instrumented_pallas.executable,
        instrumented_pallas_inputs,
    )
    (
        instrumented_pallas_output,
        pallas_rms_inputs,
        pallas_rms_mean_square,
        pallas_rms_inverse,
        pallas_normalized_float32,
        pallas_normalized_inputs,
        pallas_gate_float32,
        pallas_gates,
        pallas_silus,
        pallas_up_float32,
        pallas_up,
        pallas_hidden,
        pallas_down_float32,
        pallas_down_bfloat16,
    ) = _split_instrumented_outputs(instrumented_pallas_outputs, scenario)
    instrumented_control_inputs = _resident_inputs(
        host_inputs,
        compiled.instrumented_control.plan,
        compiled.instrumented_control.mesh,
    )
    instrumented_control_outputs = _execute_outputs(
        compiled.instrumented_control.executable,
        instrumented_control_inputs,
    )
    (
        instrumented_control_output,
        control_rms_inputs,
        control_rms_mean_square,
        control_rms_inverse,
        control_normalized_float32,
        control_normalized_inputs,
        control_gate_float32,
        control_gates,
        control_silus,
        control_up_float32,
        control_up,
        control_hidden,
        control_down_float32,
        control_down_bfloat16,
    ) = _split_instrumented_outputs(instrumented_control_outputs, scenario)
    _save_array(seed_root / "instrumented_pallas_output.npy", instrumented_pallas_output)
    _save_array(seed_root / "instrumented_control_output.npy", instrumented_control_output)
    _save_checkpoints(
        seed_root,
        "pallas",
        pallas_rms_inputs,
        pallas_rms_mean_square,
        pallas_rms_inverse,
        pallas_normalized_float32,
        pallas_normalized_inputs,
        pallas_gate_float32,
        pallas_gates,
        pallas_silus,
        pallas_up_float32,
        pallas_up,
        pallas_hidden,
        pallas_down_float32,
        pallas_down_bfloat16,
        scenario,
    )
    _save_checkpoints(
        seed_root,
        "control",
        control_rms_inputs,
        control_rms_mean_square,
        control_rms_inverse,
        control_normalized_float32,
        control_normalized_inputs,
        control_gate_float32,
        control_gates,
        control_silus,
        control_up_float32,
        control_up,
        control_hidden,
        control_down_float32,
        control_down_bfloat16,
        scenario,
    )

    instrumentation_difference = _instrumentation_difference(
        pallas_output,
        control_output,
        instrumented_pallas_output,
        instrumented_control_output,
        contract,
    )
    evidence = {
        "seed": seed,
        "inputs": host_inputs,
        "pallas_rms_input_checkpoints": pallas_rms_inputs,
        "control_rms_input_checkpoints": control_rms_inputs,
        "pallas_rms_mean_square_checkpoints": pallas_rms_mean_square,
        "control_rms_mean_square_checkpoints": control_rms_mean_square,
        "pallas_rms_inverse_checkpoints": pallas_rms_inverse,
        "control_rms_inverse_checkpoints": control_rms_inverse,
        "pallas_normalized_float32_checkpoints": pallas_normalized_float32,
        "control_normalized_float32_checkpoints": control_normalized_float32,
        "pallas_normalized_input_checkpoints": pallas_normalized_inputs,
        "control_normalized_input_checkpoints": control_normalized_inputs,
        "pallas_gate_float32_checkpoints": pallas_gate_float32,
        "control_gate_float32_checkpoints": control_gate_float32,
        "pallas_gate_checkpoints": pallas_gates,
        "control_gate_checkpoints": control_gates,
        "pallas_silu_checkpoints": pallas_silus,
        "control_silu_checkpoints": control_silus,
        "pallas_up_float32_checkpoints": pallas_up_float32,
        "control_up_float32_checkpoints": control_up_float32,
        "pallas_up_checkpoints": pallas_up,
        "control_up_checkpoints": control_up,
        "pallas_hidden_checkpoints": pallas_hidden,
        "control_hidden_checkpoints": control_hidden,
        "pallas_down_float32_checkpoints": pallas_down_float32,
        "control_down_float32_checkpoints": control_down_float32,
        "pallas_down_bfloat16_checkpoints": pallas_down_bfloat16,
        "control_down_bfloat16_checkpoints": control_down_bfloat16,
        "policy": contract.policy,
        "scenario": scenario,
    }
    normal_assessment = assess_seqax_bf16_outputs(
        pallas_output,
        control_output,
        seed=seed,
        inputs=host_inputs,
        policy=contract.policy,
        scenario=scenario,
    )
    instrumented_assessment = assess_seqax_bf16_forward(
        instrumented_pallas_output,
        instrumented_control_output,
        **evidence,
    )
    if (
        not normal_assessment.final_outputs_satisfy_policy
        or not instrumented_assessment.final_outputs_satisfy_policy
        or not instrumented_assessment.checkpoint_values_consistent
    ):
        raise ValueError(
            f"SEQAX_BF16_SEED_REJECTED scenario={scenario.name} seed={seed} "
            f"normal_assessment={normal_assessment} "
            f"instrumented_assessment={instrumented_assessment}"
        )
    observation = SeqaxBf16SeedObservation(
        scenario=scenario.name,
        seed=seed,
        input_sha256=arrays_sha256(host_inputs),
        cpu_reference_sha256=array_sha256(cpu_reference),
        pallas_output_sha256=array_sha256(pallas_output),
        control_output_sha256=array_sha256(control_output),
        instrumented_pallas_output_sha256=array_sha256(instrumented_pallas_output),
        instrumented_control_output_sha256=array_sha256(instrumented_control_output),
        pallas_rms_input_sha256=_checkpoint_hashes(pallas_rms_inputs),
        control_rms_input_sha256=_checkpoint_hashes(control_rms_inputs),
        pallas_rms_mean_square_sha256=_checkpoint_hashes(pallas_rms_mean_square),
        control_rms_mean_square_sha256=_checkpoint_hashes(control_rms_mean_square),
        pallas_rms_inverse_sha256=_checkpoint_hashes(pallas_rms_inverse),
        control_rms_inverse_sha256=_checkpoint_hashes(control_rms_inverse),
        pallas_normalized_float32_sha256=_checkpoint_hashes(pallas_normalized_float32),
        control_normalized_float32_sha256=_checkpoint_hashes(control_normalized_float32),
        pallas_normalized_input_sha256=_checkpoint_hashes(pallas_normalized_inputs),
        control_normalized_input_sha256=_checkpoint_hashes(control_normalized_inputs),
        pallas_gate_float32_sha256=_checkpoint_hashes(pallas_gate_float32),
        control_gate_float32_sha256=_checkpoint_hashes(control_gate_float32),
        pallas_gate_sha256=_checkpoint_hashes(pallas_gates),
        control_gate_sha256=_checkpoint_hashes(control_gates),
        pallas_silu_sha256=_checkpoint_hashes(pallas_silus),
        control_silu_sha256=_checkpoint_hashes(control_silus),
        pallas_up_float32_sha256=_checkpoint_hashes(pallas_up_float32),
        control_up_float32_sha256=_checkpoint_hashes(control_up_float32),
        pallas_up_sha256=_checkpoint_hashes(pallas_up),
        control_up_sha256=_checkpoint_hashes(control_up),
        pallas_hidden_sha256=_checkpoint_hashes(pallas_hidden),
        control_hidden_sha256=_checkpoint_hashes(control_hidden),
        pallas_down_float32_sha256=_checkpoint_hashes(pallas_down_float32),
        control_down_float32_sha256=_checkpoint_hashes(control_down_float32),
        pallas_down_bfloat16_sha256=_checkpoint_hashes(pallas_down_bfloat16),
        control_down_bfloat16_sha256=_checkpoint_hashes(control_down_bfloat16),
        normal_assessment=normal_assessment,
        instrumented_assessment=instrumented_assessment,
        instrumentation_difference=instrumentation_difference,
    )
    _write_json(seed_root / "observation.json", observation.model_dump(mode="json"))
    return observation


def _remove_strict_barrier(stablehlo: str, *, input_barrier: bool) -> str:
    lines = stablehlo.splitlines()
    call_index = next(
        (index for index, line in enumerate(lines) if "func.call @silu(" in line),
        None,
    )
    if call_index is None:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_SILU_CALL_MISSING")
    call_match = re.search(
        r"(?P<result>%[A-Za-z0-9_]+) = func\.call @silu\((?P<input>%[A-Za-z0-9_]+)\)",
        lines[call_index],
    )
    if call_match is None:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_SILU_CALL_INVALID")
    if input_barrier:
        converted_input = call_match.group("input")
        conversion_index = next(
            (
                index
                for index, line in enumerate(lines[:call_index])
                if re.search(
                    rf"^\s*{re.escape(converted_input)} = stablehlo\.convert ",
                    line,
                )
            ),
            None,
        )
        if conversion_index is None:
            raise ValueError("SEQAX_BF16_DISCRIMINATOR_INPUT_PROMOTION_MISSING")
        conversion_match = re.search(
            r"stablehlo\.convert (?P<source>%[A-Za-z0-9_]+)",
            lines[conversion_index],
        )
        assert conversion_match is not None
        barrier_result = conversion_match.group("source")
        barrier_index = next(
            (
                index
                for index, line in enumerate(lines[:call_index])
                if re.search(
                    rf"^\s*{re.escape(barrier_result)} = stablehlo\.optimization_barrier ",
                    line,
                )
            ),
            None,
        )
        if barrier_index is None:
            raise ValueError("SEQAX_BF16_DISCRIMINATOR_INPUT_BARRIER_MISSING")
        source_match = re.search(
            r"stablehlo\.optimization_barrier (?P<source>%[A-Za-z0-9_]+)",
            lines[barrier_index],
        )
        assert source_match is not None
        replacement = source_match.group("source")
    else:
        call_result = call_match.group("result")
        conversion_index = next(
            (
                index
                for index, line in enumerate(lines[call_index + 1 :], start=call_index + 1)
                if "stablehlo.convert" in line
                and re.search(rf"\s{re.escape(call_result)}\s*:", line)
            ),
            None,
        )
        if conversion_index is None:
            raise ValueError("SEQAX_BF16_DISCRIMINATOR_OUTPUT_ROUNDING_MISSING")
        conversion_match = re.search(
            r"^\s*(?P<result>%[A-Za-z0-9_]+) =",
            lines[conversion_index],
        )
        assert conversion_match is not None
        converted_result = conversion_match.group("result")
        barrier_index = next(
            (
                index
                for index, line in enumerate(
                    lines[conversion_index + 1 :], start=conversion_index + 1
                )
                if "stablehlo.optimization_barrier" in line
                and re.search(rf"\s{re.escape(converted_result)}\s*:", line)
            ),
            None,
        )
        if barrier_index is None:
            raise ValueError("SEQAX_BF16_DISCRIMINATOR_OUTPUT_BARRIER_MISSING")
        result_match = re.search(r"^\s*(?P<result>%[A-Za-z0-9_]+) =", lines[barrier_index])
        assert result_match is not None
        barrier_result = result_match.group("result")
        replacement = converted_result
    del lines[barrier_index]
    return re.sub(rf"{re.escape(barrier_result)}\b", replacement, "\n".join(lines)) + "\n"


def _replace_silu_body(stablehlo: str, *, relu: bool) -> str:
    match = re.search(
        r"(?ms)^  func\.func private @silu\(%arg0: (?P<type>tensor<[^\n]+xf32>)\) "
        r"-> (?P=type) \{.*?^  \}",
        stablehlo,
    )
    if match is None:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_SILU_BODY_MISSING")
    tensor_type = match.group("type")
    if relu:
        replacement = f"""  func.func private @silu(%arg0: {tensor_type}) -> {tensor_type} {{
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> {tensor_type}
    %1 = stablehlo.maximum %arg0, %0 : {tensor_type}
    return %1 : {tensor_type}
  }}"""
    else:
        replacement = f"""  func.func private @silu(%arg0: {tensor_type}) -> {tensor_type} {{
    return %arg0 : {tensor_type}
  }}"""
    return stablehlo[: match.start()] + replacement + stablehlo[match.end() :]


def _remove_hidden_barrier(stablehlo: str) -> str:
    lines = stablehlo.splitlines()
    call_index = next(
        (index for index, line in enumerate(lines) if "func.call @silu(" in line),
        None,
    )
    if call_index is None:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_SILU_CALL_MISSING")
    multiply_index = next(
        (
            index
            for index, line in enumerate(lines[call_index + 1 :], start=call_index + 1)
            if "stablehlo.multiply" in line and "xbf16>" in line
        ),
        None,
    )
    if multiply_index is None:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_HIDDEN_MULTIPLY_MISSING")
    multiply_match = re.search(r"^\s*(?P<result>%[A-Za-z0-9_]+) =", lines[multiply_index])
    assert multiply_match is not None
    multiply_result = multiply_match.group("result")
    barrier_index = next(
        (
            index
            for index, line in enumerate(lines[multiply_index + 1 :], start=multiply_index + 1)
            if "stablehlo.optimization_barrier" in line
            and re.search(rf"\s{re.escape(multiply_result)}\s*:", line)
        ),
        None,
    )
    if barrier_index is None:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_HIDDEN_BARRIER_MISSING")
    barrier_match = re.search(r"^\s*(?P<result>%[A-Za-z0-9_]+) =", lines[barrier_index])
    assert barrier_match is not None
    barrier_result = barrier_match.group("result")
    del lines[barrier_index]
    return re.sub(rf"{re.escape(barrier_result)}\b", multiply_result, "\n".join(lines)) + "\n"


def _bypass_rms_norm_checkpoint(stablehlo: str) -> str:
    lines = stablehlo.splitlines()
    for index, line in enumerate(lines):
        if "sdy.return " not in line:
            continue
        prefix, result_types = line.split(" : ", maxsplit=1)
        indentation, operands_text = prefix.split("sdy.return ", maxsplit=1)
        operands = operands_text.split(", ")
        if len(operands) < 14:
            continue
        operands[1] = operands[5]
        lines[index] = f"{indentation}sdy.return {', '.join(operands)} : {result_types}"
        return "\n".join(lines) + "\n"
    raise ValueError("SEQAX_BF16_DISCRIMINATOR_RMS_RETURN_MISSING")


def _drop_reduction_collective(physical: str) -> str:
    removed = False
    lines = []
    for line in physical.splitlines():
        if not removed and '"tpu_schedule.collective"' in line and "reduce_scatter" in line:
            removed = True
            continue
        lines.append(line)
    if not removed:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_REDUCE_SCATTER_MISSING")
    return "\n".join(lines) + "\n"


def _load_saved_checkpoints(
    seed_root: Path,
    scenario: SeqaxBf16NumericalScenario,
    prefix: str,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    rms_inputs = tuple(
        decode_seqax_bf16_checkpoint(
            _load_array(seed_root / "checkpoints" / f"{prefix}_rms_input_{layer:02d}.npy"),
            contract,
        )
        for layer, contract in enumerate(scenario.rms_input_checkpoints)
    )
    rms_mean_square = tuple(
        _load_array(seed_root / "checkpoints" / f"{prefix}_rms_mean_square_{layer:02d}.npy")
        for layer in range(scenario.parameters.layers)
    )
    rms_inverse = tuple(
        _load_array(seed_root / "checkpoints" / f"{prefix}_rms_inverse_{layer:02d}.npy")
        for layer in range(scenario.parameters.layers)
    )
    normalized_float32 = tuple(
        _load_array(seed_root / "checkpoints" / f"{prefix}_normalized_float32_{layer:02d}.npy")
        for layer in range(scenario.parameters.layers)
    )
    normalized_inputs = tuple(
        decode_seqax_bf16_checkpoint(
            _load_array(seed_root / "checkpoints" / f"{prefix}_normalized_input_{layer:02d}.npy"),
            contract,
        )
        for layer, contract in enumerate(scenario.normalized_input_checkpoints)
    )
    gate_float32 = tuple(
        _load_array(seed_root / "checkpoints" / f"{prefix}_gate_float32_{layer:02d}.npy")
        for layer in range(scenario.parameters.layers)
    )
    gates = tuple(
        decode_seqax_bf16_checkpoint(
            _load_array(seed_root / "checkpoints" / f"{prefix}_gate_{layer:02d}.npy"),
            contract,
        )
        for layer, contract in enumerate(scenario.gate_checkpoints)
    )
    silus = tuple(
        decode_seqax_bf16_checkpoint(
            _load_array(seed_root / "checkpoints" / f"{prefix}_silu_{layer:02d}.npy"),
            contract,
        )
        for layer, contract in enumerate(scenario.silu_checkpoints)
    )
    up_float32 = tuple(
        _load_array(seed_root / "checkpoints" / f"{prefix}_up_float32_{layer:02d}.npy")
        for layer in range(scenario.parameters.layers)
    )
    up = tuple(
        decode_seqax_bf16_checkpoint(
            _load_array(seed_root / "checkpoints" / f"{prefix}_up_{layer:02d}.npy"),
            contract,
        )
        for layer, contract in enumerate(scenario.up_checkpoints)
    )
    hidden = tuple(
        decode_seqax_bf16_checkpoint(
            _load_array(seed_root / "checkpoints" / f"{prefix}_hidden_{layer:02d}.npy"),
            contract,
        )
        for layer, contract in enumerate(scenario.hidden_checkpoints)
    )
    down_float32 = tuple(
        _load_array(seed_root / "checkpoints" / f"{prefix}_down_float32_{layer:02d}.npy")
        for layer in range(scenario.parameters.layers)
    )
    down_bfloat16 = tuple(
        decode_seqax_bf16_checkpoint(
            _load_array(seed_root / "checkpoints" / f"{prefix}_down_bfloat16_{layer:02d}.npy"),
            contract,
        )
        for layer, contract in enumerate(scenario.down_bfloat16_checkpoints)
    )
    return (
        rms_inputs,
        rms_mean_square,
        rms_inverse,
        normalized_float32,
        normalized_inputs,
        gate_float32,
        gates,
        silus,
        up_float32,
        up,
        hidden,
        down_float32,
        down_bfloat16,
    )


def _mutation_failure(
    output: np.ndarray,
    *,
    clause: SeqaxDiscriminatorClause,
    contract: SeqaxBf16ValidationContract,
    scenario: SeqaxBf16NumericalScenario,
    seed: int,
    inputs: tuple[np.ndarray, ...],
    rms_inputs: tuple[np.ndarray, ...],
    rms_mean_square: tuple[np.ndarray, ...],
    rms_inverse: tuple[np.ndarray, ...],
    normalized_float32: tuple[np.ndarray, ...],
    normalized_inputs: tuple[np.ndarray, ...],
    gate_float32: tuple[np.ndarray, ...],
    gates: tuple[np.ndarray, ...],
    silus: tuple[np.ndarray, ...],
    up_float32: tuple[np.ndarray, ...],
    up: tuple[np.ndarray, ...],
    hidden: tuple[np.ndarray, ...],
    down_float32: tuple[np.ndarray, ...],
    down_bfloat16: tuple[np.ndarray, ...],
    cpu_reference: np.ndarray | None = None,
) -> str:
    try:
        assessment = assess_seqax_bf16_forward(
            output,
            output,
            seed=seed,
            inputs=inputs,
            pallas_rms_input_checkpoints=rms_inputs,
            control_rms_input_checkpoints=rms_inputs,
            pallas_rms_mean_square_checkpoints=rms_mean_square,
            control_rms_mean_square_checkpoints=rms_mean_square,
            pallas_rms_inverse_checkpoints=rms_inverse,
            control_rms_inverse_checkpoints=rms_inverse,
            pallas_normalized_float32_checkpoints=normalized_float32,
            control_normalized_float32_checkpoints=normalized_float32,
            pallas_normalized_input_checkpoints=normalized_inputs,
            control_normalized_input_checkpoints=normalized_inputs,
            pallas_gate_float32_checkpoints=gate_float32,
            control_gate_float32_checkpoints=gate_float32,
            pallas_gate_checkpoints=gates,
            control_gate_checkpoints=gates,
            pallas_silu_checkpoints=silus,
            control_silu_checkpoints=silus,
            pallas_up_float32_checkpoints=up_float32,
            control_up_float32_checkpoints=up_float32,
            pallas_up_checkpoints=up,
            control_up_checkpoints=up,
            pallas_hidden_checkpoints=hidden,
            control_hidden_checkpoints=hidden,
            pallas_down_float32_checkpoints=down_float32,
            control_down_float32_checkpoints=down_float32,
            pallas_down_bfloat16_checkpoints=down_bfloat16,
            control_down_bfloat16_checkpoints=down_bfloat16,
            policy=contract.policy,
            scenario=scenario,
        )
    except (TypeError, ValueError) as error:
        expected = {
            SeqaxDiscriminatorClause.FINITE_OUTPUT: "finite",
            SeqaxDiscriminatorClause.OUTPUT_DTYPE: "float32",
            SeqaxDiscriminatorClause.OUTPUT_SHAPE: "shape",
        }.get(clause)
        if expected is None or expected not in str(error):
            raise ValueError(
                f"SEQAX_BF16_DISCRIMINATOR_WRONG_FAILURE clause={clause} error={error}"
            ) from error
        return f"{type(error).__name__}: {error}"
    if cpu_reference is not None:
        assessment = assessment.model_copy(
            update=_assess_output_arrays(
                output,
                output,
                cpu_reference,
                policy=contract.policy,
                scenario=scenario,
            ).model_dump()
        )
    unit = contract.policy.unit_roundoff
    depth_scale = contract.policy.depth_scale(scenario.parameters.layers)
    clause_rejected = {
        SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY: (
            not assessment.final_outputs_satisfy_policy
        ),
        SeqaxDiscriminatorClause.ROW_SCALED_MAXIMUM: (
            assessment.cpu_pallas_row_scaled_max
            > contract.policy.cpu_row_scaled_max_units * unit * depth_scale
        ),
        SeqaxDiscriminatorClause.RELATIVE_L2: (
            assessment.cpu_pallas_relative_l2
            > contract.policy.cpu_relative_l2_units * unit * depth_scale
        ),
    }.get(clause, False)
    if not clause_rejected:
        raise ValueError(
            f"SEQAX_BF16_DISCRIMINATOR_FALSE_ACCEPT clause={clause} assessment={assessment}"
        )
    return f"{clause.value}: rejected metrics={assessment.model_dump(mode='json')}"


def _assess_checkpoint_mutation(
    baseline: np.ndarray,
    *,
    contract: SeqaxBf16ValidationContract,
    scenario: SeqaxBf16NumericalScenario,
    seed: int,
    inputs: tuple[np.ndarray, ...],
    rms_inputs: tuple[np.ndarray, ...],
    rms_mean_square: tuple[np.ndarray, ...],
    rms_inverse: tuple[np.ndarray, ...],
    normalized_float32: tuple[np.ndarray, ...],
    normalized_inputs: tuple[np.ndarray, ...],
    gate_float32: tuple[np.ndarray, ...],
    gates: tuple[np.ndarray, ...],
    silus: tuple[np.ndarray, ...],
    up_float32: tuple[np.ndarray, ...],
    up: tuple[np.ndarray, ...],
    hidden: tuple[np.ndarray, ...],
    down_float32: tuple[np.ndarray, ...],
    down_bfloat16: tuple[np.ndarray, ...],
    mutated_rms_inputs: tuple[np.ndarray, ...] | None = None,
    mutated_rms_mean_square: tuple[np.ndarray, ...] | None = None,
    mutated_rms_inverse: tuple[np.ndarray, ...] | None = None,
    mutated_normalized_float32: tuple[np.ndarray, ...] | None = None,
    mutated_normalized_inputs: tuple[np.ndarray, ...] | None = None,
    mutated_gate_float32: tuple[np.ndarray, ...] | None = None,
    mutated_gate: tuple[np.ndarray, ...] | None = None,
    mutated_up_float32: tuple[np.ndarray, ...] | None = None,
    mutated_up: tuple[np.ndarray, ...] | None = None,
    mutated_down_float32: tuple[np.ndarray, ...] | None = None,
    cpu_reference: np.ndarray | None = None,
) -> SeqaxBf16NumericalAssessment:
    assessment = assess_seqax_bf16_forward(
        baseline,
        baseline,
        seed=seed,
        inputs=inputs,
        pallas_rms_input_checkpoints=(
            rms_inputs if mutated_rms_inputs is None else mutated_rms_inputs
        ),
        control_rms_input_checkpoints=rms_inputs,
        pallas_rms_mean_square_checkpoints=(
            rms_mean_square if mutated_rms_mean_square is None else mutated_rms_mean_square
        ),
        control_rms_mean_square_checkpoints=rms_mean_square,
        pallas_rms_inverse_checkpoints=(
            rms_inverse if mutated_rms_inverse is None else mutated_rms_inverse
        ),
        control_rms_inverse_checkpoints=rms_inverse,
        pallas_normalized_float32_checkpoints=(
            normalized_float32 if mutated_normalized_float32 is None else mutated_normalized_float32
        ),
        control_normalized_float32_checkpoints=normalized_float32,
        pallas_normalized_input_checkpoints=(
            normalized_inputs if mutated_normalized_inputs is None else mutated_normalized_inputs
        ),
        control_normalized_input_checkpoints=normalized_inputs,
        pallas_gate_float32_checkpoints=(
            gate_float32 if mutated_gate_float32 is None else mutated_gate_float32
        ),
        control_gate_float32_checkpoints=gate_float32,
        pallas_gate_checkpoints=gates if mutated_gate is None else mutated_gate,
        control_gate_checkpoints=gates,
        pallas_silu_checkpoints=silus,
        control_silu_checkpoints=silus,
        pallas_up_float32_checkpoints=(
            up_float32 if mutated_up_float32 is None else mutated_up_float32
        ),
        control_up_float32_checkpoints=up_float32,
        pallas_up_checkpoints=up if mutated_up is None else mutated_up,
        control_up_checkpoints=up,
        pallas_hidden_checkpoints=hidden,
        control_hidden_checkpoints=hidden,
        pallas_down_float32_checkpoints=(
            down_float32 if mutated_down_float32 is None else mutated_down_float32
        ),
        control_down_float32_checkpoints=down_float32,
        pallas_down_bfloat16_checkpoints=down_bfloat16,
        control_down_bfloat16_checkpoints=down_bfloat16,
        policy=contract.policy,
        scenario=scenario,
    )
    if cpu_reference is None:
        return assessment
    return assessment.model_copy(
        update=_assess_output_arrays(
            baseline,
            baseline,
            cpu_reference,
            policy=contract.policy,
            scenario=scenario,
        ).model_dump()
    )


def _checkpoint_mutants(
    inputs: tuple[np.ndarray, ...],
    rms_inputs: tuple[np.ndarray, ...],
    rms_mean_square: tuple[np.ndarray, ...],
    rms_inverse: tuple[np.ndarray, ...],
    normalized_float32: tuple[np.ndarray, ...],
    normalized_inputs: tuple[np.ndarray, ...],
    gate_float32: tuple[np.ndarray, ...],
    gates: tuple[np.ndarray, ...],
    up_float32: tuple[np.ndarray, ...],
    up: tuple[np.ndarray, ...],
) -> dict[SeqaxNumericalDiscriminator, tuple[np.ndarray, ...]]:
    wrong_scale = (
        np.asarray(
            rms_inputs[0].astype(np.float32)
            * rms_inverse[0]
            * inputs[4][0].astype(np.float32)[::-1],
            dtype=np.float32,
        ),
        *(value.copy() for value in normalized_float32[1:]),
    )
    corrupt_mean_square = tuple(value.copy() for value in rms_mean_square)
    corrupt_mean_square[0].reshape(-1)[0] += np.float32(0.25)
    corrupt_inverse = tuple(value.copy() for value in rms_inverse)
    corrupt_inverse[0].reshape(-1)[0] *= np.float32(1.25)
    corrupt_normalized_float32 = tuple(value.copy() for value in normalized_float32)
    corrupt_normalized_float32[0].reshape(-1)[0] += np.float32(0.25)
    corrupt_normalized_bfloat16 = tuple(value.copy() for value in normalized_inputs)
    corrupt_normalized_bfloat16[0].view(np.uint16).reshape(-1)[0] ^= np.uint16(1)
    wrong_gate_weight = (
        seqax_gate_projection_reference_float32(normalized_inputs[0], inputs[9][0]),
        *gate_float32[1:],
    )
    corrupt_gate_float32 = tuple(value.copy() for value in gate_float32)
    corrupt_gate_float32[0].reshape(-1)[0] += np.float32(1e-3)
    corrupt_gate_bfloat16 = tuple(value.copy() for value in gates)
    corrupt_gate_bfloat16[0].view(np.uint16).reshape(-1)[0] ^= np.uint16(1)
    wrong_weight = (
        seqax_up_projection_reference_float32(normalized_inputs[0], inputs[8][0]),
        *up_float32[1:],
    )
    corrupt_float32 = tuple(value.copy() for value in up_float32)
    corrupt_float32[0].reshape(-1)[0] += np.float32(1e-3)
    corrupt_bfloat16 = tuple(value.copy() for value in up)
    corrupt_bfloat16[0].view(np.uint16).reshape(-1)[0] ^= np.uint16(1)
    return {
        SeqaxNumericalDiscriminator.WRONG_RMS_SCALE_CHECKPOINT: wrong_scale,
        SeqaxNumericalDiscriminator.CORRUPT_RMS_MEAN_SQUARE_CHECKPOINT: corrupt_mean_square,
        SeqaxNumericalDiscriminator.CORRUPT_RMS_INV_CHECKPOINT: corrupt_inverse,
        SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_FLOAT32_CHECKPOINT: (
            corrupt_normalized_float32
        ),
        SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT: (
            corrupt_normalized_bfloat16
        ),
        SeqaxNumericalDiscriminator.WRONG_GATE_WEIGHT_CHECKPOINT: wrong_gate_weight,
        SeqaxNumericalDiscriminator.CORRUPT_GATE_FLOAT32_CHECKPOINT: corrupt_gate_float32,
        SeqaxNumericalDiscriminator.CORRUPT_GATE_BFLOAT16_CHECKPOINT: corrupt_gate_bfloat16,
        SeqaxNumericalDiscriminator.WRONG_UP_WEIGHT_CHECKPOINT: wrong_weight,
        SeqaxNumericalDiscriminator.CORRUPT_UP_FLOAT32_CHECKPOINT: corrupt_float32,
        SeqaxNumericalDiscriminator.CORRUPT_UP_BFLOAT16_CHECKPOINT: corrupt_bfloat16,
    }


def _checkpoint_mutation_rejected(
    discriminator: SeqaxNumericalDiscriminator,
    assessment: SeqaxBf16NumericalAssessment,
) -> bool:
    return {
        SeqaxNumericalDiscriminator.WRONG_RMS_SCALE_CHECKPOINT: (
            not assessment.pallas_normalized_float32_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_RMS_MEAN_SQUARE_CHECKPOINT: (
            not assessment.pallas_rms_mean_square_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_RMS_INV_CHECKPOINT: (
            not assessment.pallas_rms_inverse_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_FLOAT32_CHECKPOINT: (
            not assessment.pallas_normalized_float32_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT: (
            not assessment.pallas_normalized_bfloat16_matches_float32
        ),
        SeqaxNumericalDiscriminator.WRONG_GATE_WEIGHT_CHECKPOINT: (
            not assessment.pallas_gate_float32_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_GATE_FLOAT32_CHECKPOINT: (
            not assessment.pallas_gate_float32_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_GATE_BFLOAT16_CHECKPOINT: (
            not assessment.pallas_gate_bfloat16_matches_float32
        ),
        SeqaxNumericalDiscriminator.WRONG_UP_WEIGHT_CHECKPOINT: (
            not assessment.pallas_up_float32_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_UP_FLOAT32_CHECKPOINT: (
            not assessment.pallas_up_float32_within_bound
        ),
        SeqaxNumericalDiscriminator.CORRUPT_UP_BFLOAT16_CHECKPOINT: (
            not assessment.pallas_up_bfloat16_matches_float32
        ),
    }[discriminator]


def _checkpoint_mutation_arguments(
    discriminator: SeqaxNumericalDiscriminator,
    mutant: tuple[np.ndarray, ...],
) -> dict[str, tuple[np.ndarray, ...]]:
    if discriminator is SeqaxNumericalDiscriminator.WRONG_RMS_SCALE_CHECKPOINT:
        return {"mutated_normalized_float32": mutant}
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_RMS_MEAN_SQUARE_CHECKPOINT:
        return {"mutated_rms_mean_square": mutant}
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_RMS_INV_CHECKPOINT:
        return {"mutated_rms_inverse": mutant}
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_FLOAT32_CHECKPOINT:
        return {"mutated_normalized_float32": mutant}
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT:
        return {"mutated_normalized_inputs": mutant}
    if discriminator in {
        SeqaxNumericalDiscriminator.WRONG_GATE_WEIGHT_CHECKPOINT,
        SeqaxNumericalDiscriminator.CORRUPT_GATE_FLOAT32_CHECKPOINT,
    }:
        return {"mutated_gate_float32": mutant}
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_GATE_BFLOAT16_CHECKPOINT:
        return {"mutated_gate": mutant}
    if discriminator in {
        SeqaxNumericalDiscriminator.WRONG_UP_WEIGHT_CHECKPOINT,
        SeqaxNumericalDiscriminator.CORRUPT_UP_FLOAT32_CHECKPOINT,
    }:
        return {"mutated_up_float32": mutant}
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_UP_BFLOAT16_CHECKPOINT:
        return {"mutated_up": mutant}
    raise ValueError(f"SEQAX_BF16_CHECKPOINT_DISCRIMINATOR_INVALID name={discriminator}")


def _checkpoint_mutant_storage(
    discriminator: SeqaxNumericalDiscriminator,
    mutant: np.ndarray,
    scenario: SeqaxBf16NumericalScenario,
) -> np.ndarray:
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT:
        return encode_seqax_bf16_checkpoint(mutant, scenario.normalized_input_checkpoints[0])
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_GATE_BFLOAT16_CHECKPOINT:
        return encode_seqax_bf16_checkpoint(mutant, scenario.gate_checkpoints[0])
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_UP_BFLOAT16_CHECKPOINT:
        return encode_seqax_bf16_checkpoint(mutant, scenario.up_checkpoints[0])
    return mutant


def _decode_checkpoint_mutant(
    discriminator: SeqaxNumericalDiscriminator,
    storage: np.ndarray,
    scenario: SeqaxBf16NumericalScenario,
) -> np.ndarray:
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT:
        return decode_seqax_bf16_checkpoint(storage, scenario.normalized_input_checkpoints[0])
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_GATE_BFLOAT16_CHECKPOINT:
        return decode_seqax_bf16_checkpoint(storage, scenario.gate_checkpoints[0])
    if discriminator is SeqaxNumericalDiscriminator.CORRUPT_UP_BFLOAT16_CHECKPOINT:
        return decode_seqax_bf16_checkpoint(storage, scenario.up_checkpoints[0])
    return storage


def _compile_activation_mutant(
    path: _CompiledPath,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
    *,
    pallas: bool,
    relu: bool,
    interpret_pallas: bool = False,
) -> tuple[str, np.ndarray]:
    original = jax.nn.silu

    def identity(value: Any) -> Any:
        return value

    jax.nn.silu = jax.nn.relu if relu else identity
    try:
        if pallas:
            if not isinstance(path.plan, SeqaxPallasPlan):
                raise TypeError("SEQAX_BF16_PALLAS_MUTANT_PLAN_MISMATCH")
            callable_, mesh = path.plan.build(interpret=interpret_pallas, devices=devices)
        else:
            if not isinstance(path.plan, JaxDistributedMeshPlan):
                raise TypeError("SEQAX_BF16_CONTROL_MUTANT_PLAN_MISMATCH")
            callable_, mesh = path.plan.build(devices=devices)
        resident = _resident_inputs(host_inputs, path.plan, mesh)
        lowered = callable_.lower(*resident)
    finally:
        jax.nn.silu = original
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    output = _execute(lowered.compile(), resident)
    return stablehlo, output


def _artifact_identities(
    root: Path, paths: tuple[Path, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(path.relative_to(root).as_posix() for path in paths),
        tuple(_sha256(path) for path in paths),
    )


def _activation_mutant_stablehlo_sha256(
    contract: SeqaxBf16ValidationContract,
    discriminator: SeqaxNumericalDiscriminator,
    label: str,
) -> str:
    expected = next(
        value
        for value in contract.activation_mutant_stablehlo
        if value.discriminator is discriminator
    )
    if label == "pallas":
        return expected.pallas_stablehlo_sha256
    if label == "control":
        return expected.control_stablehlo_sha256
    raise ValueError(f"SEQAX_BF16_ACTIVATION_PATH_INVALID path={label}")


def _run_discriminators(
    root: Path,
    contract: SeqaxBf16ValidationContract,
    compiled: _CompiledScenario,
    devices: tuple[Any, ...],
) -> tuple[SeqaxBf16DiscriminatorObservation, ...]:
    scenario = compiled.scenario
    seed = scenario.seeds[0]
    seed_root = _seed_root(root, scenario.name, seed)
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=seed, **scenario.parameters.model_dump())
    )
    baseline = _load_array(seed_root / "pallas_output.npy")
    (
        rms_inputs,
        rms_mean_square,
        rms_inverse,
        normalized_float32,
        normalized_inputs,
        gate_float32,
        gates,
        silus,
        up_float32,
        up,
        hidden,
        down_float32,
        down_bfloat16,
    ) = _load_saved_checkpoints(seed_root, scenario, "pallas")
    observations = []

    def record(
        discriminator: SeqaxNumericalDiscriminator,
        paths: tuple[Path, ...],
        failure: str,
    ) -> None:
        artifact_paths, artifact_sha256 = _artifact_identities(root, paths)
        observations.append(
            SeqaxBf16DiscriminatorObservation(
                discriminator=discriminator,
                clause=seqax_discriminator_clause(discriminator),
                artifact_paths=artifact_paths,
                artifact_sha256=artifact_sha256,
                rejected=True,
                failure=failure,
            )
        )

    pallas_hlo = canonical_seqax_stablehlo(compiled.pallas.stablehlo)
    hlo_mutants = {
        SeqaxNumericalDiscriminator.REMOVE_INPUT_BARRIER: _remove_strict_barrier(
            pallas_hlo, input_barrier=True
        ),
        SeqaxNumericalDiscriminator.REMOVE_OUTPUT_BARRIER: _remove_strict_barrier(
            pallas_hlo, input_barrier=False
        ),
        SeqaxNumericalDiscriminator.REMOVE_HIDDEN_BARRIER: _remove_hidden_barrier(pallas_hlo),
        SeqaxNumericalDiscriminator.IDENTITY_SILU: _replace_silu_body(pallas_hlo, relu=False),
        SeqaxNumericalDiscriminator.RELU_SILU: _replace_silu_body(pallas_hlo, relu=True),
    }
    hlo_mutants = {
        discriminator: canonical_seqax_stablehlo(mutant)
        for discriminator, mutant in hlo_mutants.items()
    }
    for discriminator, mutant in hlo_mutants.items():
        path = root / "discriminators" / discriminator / "mutant_stablehlo.txt"
        _write_text(path, mutant)
        try:
            _validate_strict_silu_stablehlo(
                mutant,
                expected_count=scenario.parameters.layers,
                instrumented=False,
            )
        except ValueError as error:
            failure = f"{type(error).__name__}: {error}"
        else:
            raise ValueError(f"SEQAX_BF16_HLO_DISCRIMINATOR_ACCEPTED name={discriminator}")
        paths = [path]
        if discriminator in {
            SeqaxNumericalDiscriminator.IDENTITY_SILU,
            SeqaxNumericalDiscriminator.RELU_SILU,
        }:
            for label, compiled_path, pallas in (
                ("pallas", compiled.pallas, True),
                ("control", compiled.control, False),
            ):
                runtime_hlo, runtime_output = _compile_activation_mutant(
                    compiled_path,
                    host_inputs,
                    devices,
                    pallas=pallas,
                    relu=discriminator is SeqaxNumericalDiscriminator.RELU_SILU,
                )
                validate_activation_mutant_stablehlo(
                    runtime_hlo,
                    expected_count=scenario.parameters.layers,
                    expected_sha256=_activation_mutant_stablehlo_sha256(
                        contract,
                        discriminator,
                        label,
                    ),
                    relu=discriminator is SeqaxNumericalDiscriminator.RELU_SILU,
                )
                runtime_hlo_path = path.with_name(f"{label}_runtime_stablehlo.txt")
                runtime_output_path = path.with_name(f"{label}_runtime_output.npy")
                _write_text(runtime_hlo_path, canonical_seqax_stablehlo(runtime_hlo))
                _save_array(runtime_output_path, runtime_output)
                causal_failure = _mutation_failure(
                    runtime_output,
                    clause=SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY,
                    contract=contract,
                    scenario=scenario,
                    seed=seed,
                    inputs=host_inputs,
                    rms_inputs=rms_inputs,
                    rms_mean_square=rms_mean_square,
                    rms_inverse=rms_inverse,
                    normalized_float32=normalized_float32,
                    normalized_inputs=normalized_inputs,
                    gate_float32=gate_float32,
                    gates=gates,
                    silus=silus,
                    up_float32=up_float32,
                    up=up,
                    hidden=hidden,
                    down_float32=down_float32,
                    down_bfloat16=down_bfloat16,
                )
                if np.array_equal(runtime_output, baseline):
                    raise ValueError(
                        f"SEQAX_BF16_ACTIVATION_MUTANT_NOT_CAUSAL name={discriminator} path={label}"
                    )
                failure += f"; {label}_causal {causal_failure}"
                paths.extend((runtime_hlo_path, runtime_output_path))
        record(discriminator, tuple(paths), failure)

    rms_bypass_discriminator = SeqaxNumericalDiscriminator.BYPASS_RMS_NORM_CHECKPOINT
    rms_bypass = canonical_seqax_stablehlo(
        _bypass_rms_norm_checkpoint(compiled.instrumented_pallas.stablehlo)
    )
    rms_bypass_path = root / "discriminators" / rms_bypass_discriminator / "mutant_stablehlo.txt"
    _write_text(rms_bypass_path, rms_bypass)
    try:
        _validate_strict_silu_stablehlo(
            rms_bypass,
            expected_count=scenario.parameters.layers,
            instrumented=True,
        )
    except ValueError as error:
        rms_bypass_failure = f"{type(error).__name__}: {error}"
    else:
        raise ValueError(f"SEQAX_BF16_HLO_DISCRIMINATOR_ACCEPTED name={rms_bypass_discriminator}")
    record(rms_bypass_discriminator, (rms_bypass_path,), rms_bypass_failure)

    for discriminator, mutant in _checkpoint_mutants(
        host_inputs,
        rms_inputs,
        rms_mean_square,
        rms_inverse,
        normalized_float32,
        normalized_inputs,
        gate_float32,
        gates,
        up_float32,
        up,
    ).items():
        path = root / "discriminators" / discriminator / "mutant_checkpoint.npy"
        _save_array(path, _checkpoint_mutant_storage(discriminator, mutant[0], scenario))
        assessment = _assess_checkpoint_mutation(
            baseline,
            contract=contract,
            scenario=scenario,
            seed=seed,
            inputs=host_inputs,
            rms_inputs=rms_inputs,
            rms_mean_square=rms_mean_square,
            rms_inverse=rms_inverse,
            normalized_float32=normalized_float32,
            normalized_inputs=normalized_inputs,
            gate_float32=gate_float32,
            gates=gates,
            silus=silus,
            up_float32=up_float32,
            up=up,
            hidden=hidden,
            down_float32=down_float32,
            down_bfloat16=down_bfloat16,
            **_checkpoint_mutation_arguments(discriminator, mutant),
        )
        if not _checkpoint_mutation_rejected(discriminator, assessment):
            raise ValueError(f"SEQAX_BF16_CHECKPOINT_DISCRIMINATOR_ACCEPTED name={discriminator}")
        record(
            discriminator,
            (path,),
            f"{seqax_discriminator_clause(discriminator).value}: rejected "
            f"assessment={assessment.model_dump(mode='json')}",
        )

    corrupted_down = tuple(value.copy() for value in down_float32)
    corrupted_down[0].reshape(-1)[0] += np.float32(1e-3)
    down_path = (
        root
        / "discriminators"
        / SeqaxNumericalDiscriminator.CORRUPT_DOWN_CHECKPOINT
        / "mutant_down_float32.npy"
    )
    _save_array(down_path, corrupted_down[0])
    down_assessment = assess_seqax_bf16_forward(
        baseline,
        baseline,
        seed=seed,
        inputs=host_inputs,
        pallas_rms_input_checkpoints=rms_inputs,
        control_rms_input_checkpoints=rms_inputs,
        pallas_rms_mean_square_checkpoints=rms_mean_square,
        control_rms_mean_square_checkpoints=rms_mean_square,
        pallas_rms_inverse_checkpoints=rms_inverse,
        control_rms_inverse_checkpoints=rms_inverse,
        pallas_normalized_float32_checkpoints=normalized_float32,
        control_normalized_float32_checkpoints=normalized_float32,
        pallas_normalized_input_checkpoints=normalized_inputs,
        control_normalized_input_checkpoints=normalized_inputs,
        pallas_gate_float32_checkpoints=gate_float32,
        control_gate_float32_checkpoints=gate_float32,
        pallas_gate_checkpoints=gates,
        control_gate_checkpoints=gates,
        pallas_silu_checkpoints=silus,
        control_silu_checkpoints=silus,
        pallas_up_float32_checkpoints=up_float32,
        control_up_float32_checkpoints=up_float32,
        pallas_up_checkpoints=up,
        control_up_checkpoints=up,
        pallas_hidden_checkpoints=hidden,
        control_hidden_checkpoints=hidden,
        pallas_down_float32_checkpoints=corrupted_down,
        control_down_float32_checkpoints=down_float32,
        pallas_down_bfloat16_checkpoints=down_bfloat16,
        control_down_bfloat16_checkpoints=down_bfloat16,
        policy=contract.policy,
        scenario=scenario,
    )
    if down_assessment.checkpoint_values_consistent:
        raise ValueError("SEQAX_BF16_DOWN_CHECKPOINT_DISCRIMINATOR_ACCEPTED")
    record(
        SeqaxNumericalDiscriminator.CORRUPT_DOWN_CHECKPOINT,
        (down_path,),
        "down_projection_oracle: rejected "
        f"ratio={down_assessment.pallas_down_float32_max_bound_ratio}",
    )

    collective = _drop_reduction_collective(canonical_text(compiled.physical))
    collective_path = (
        root
        / "discriminators"
        / SeqaxNumericalDiscriminator.DROP_REDUCTION_COLLECTIVE
        / "mutant_physical.xdsl"
    )
    _write_text(collective_path, collective)
    try:
        _parse_physical(collective).verify()
    except VerifyException as error:
        collective_failure = f"{type(error).__name__}: {error}"
    else:
        raise ValueError("SEQAX_BF16_COLLECTIVE_DISCRIMINATOR_ACCEPTED")
    record(
        SeqaxNumericalDiscriminator.DROP_REDUCTION_COLLECTIVE,
        (collective_path,),
        collective_failure,
    )

    input_mutations = {
        SeqaxNumericalDiscriminator.DROP_EMBEDDING_SHARD: (SeqaxInputMutation.DROP_EMBEDDING_SHARD),
        SeqaxNumericalDiscriminator.ROLL_MODEL_SHARD: SeqaxInputMutation.ROLL_MODEL_SHARD,
        SeqaxNumericalDiscriminator.OMIT_MLP_TERM: SeqaxInputMutation.OMIT_MLP_TERM,
        SeqaxNumericalDiscriminator.SWAP_GATE_UP: SeqaxInputMutation.SWAP_GATE_UP,
    }
    for discriminator, mutation in input_mutations.items():
        mutant_inputs = mutate_seqax_forward_inputs(host_inputs, mutation)
        resident = _resident_inputs(mutant_inputs, compiled.pallas.plan, compiled.pallas.mesh)
        output = _execute(compiled.pallas.executable, resident)
        mutation_root = root / "discriminators" / discriminator
        paths = []
        for index, value in enumerate(mutant_inputs):
            path = mutation_root / "inputs" / f"{index:02d}.npy"
            _save_array(path, value)
            paths.append(path)
        output_path = mutation_root / "runtime_output.npy"
        _save_array(output_path, output)
        paths.append(output_path)
        failure = _mutation_failure(
            output,
            clause=SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY,
            contract=contract,
            scenario=scenario,
            seed=seed,
            inputs=host_inputs,
            rms_inputs=rms_inputs,
            rms_mean_square=rms_mean_square,
            rms_inverse=rms_inverse,
            normalized_float32=normalized_float32,
            normalized_inputs=normalized_inputs,
            gate_float32=gate_float32,
            gates=gates,
            silus=silus,
            up_float32=up_float32,
            up=up,
            hidden=hidden,
            down_float32=down_float32,
            down_bfloat16=down_bfloat16,
        )
        record(discriminator, tuple(paths), failure)

    output_mutations: dict[SeqaxNumericalDiscriminator, np.ndarray] = {}
    localized = baseline.copy()
    localized.reshape(-1)[0] += np.float32(1.0)
    output_mutations[SeqaxNumericalDiscriminator.LOCALIZED_SPIKE] = localized
    output_mutations[SeqaxNumericalDiscriminator.DISTRIBUTED_DRIFT] = baseline + np.float32(0.5)
    nonfinite = baseline.copy()
    nonfinite.reshape(-1)[0] = np.nan
    output_mutations[SeqaxNumericalDiscriminator.NONFINITE_OUTPUT] = nonfinite
    output_mutations[SeqaxNumericalDiscriminator.DTYPE_OUTPUT] = baseline.astype(np.float64)
    output_mutations[SeqaxNumericalDiscriminator.SHAPE_OUTPUT] = baseline[..., :-1]
    for discriminator, output in output_mutations.items():
        path = root / "discriminators" / discriminator / "mutant_output.npy"
        _save_array(path, output)
        failure = _mutation_failure(
            output,
            clause=seqax_discriminator_clause(discriminator),
            contract=contract,
            scenario=scenario,
            seed=seed,
            inputs=host_inputs,
            rms_inputs=rms_inputs,
            rms_mean_square=rms_mean_square,
            rms_inverse=rms_inverse,
            normalized_float32=normalized_float32,
            normalized_inputs=normalized_inputs,
            gate_float32=gate_float32,
            gates=gates,
            silus=silus,
            up_float32=up_float32,
            up=up,
            hidden=hidden,
            down_float32=down_float32,
            down_bfloat16=down_bfloat16,
        )
        record(discriminator, (path,), failure)

    result = tuple(observations)
    if tuple(value.discriminator for value in result) != contract.required_discriminators:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_ORDER_MISMATCH")
    _write_json(
        root / "discriminators" / "observations.json",
        [value.model_dump(mode="json") for value in result],
    )
    return result


def _runtime(contract: SeqaxBf16ValidationContract) -> SeqaxBf16Runtime:
    runtime = _runtime_identity()
    expected = contract.runtime
    if (
        not runtime.python.startswith(expected.python_major_minor + ".")
        or runtime.jax != expected.jax
        or runtime.jaxlib != expected.jaxlib
        or runtime.libtpu != expected.libtpu
        or runtime.xla != expected.libtpu_init_args
        or ml_dtypes.__version__ != expected.ml_dtypes
        or platform.machine() != expected.cpu_machine
        or platform.system() != expected.cpu_system
    ):
        raise ValueError(
            "SEQAX_BF16_RUNTIME_MISMATCH "
            f"runtime={runtime} ml_dtypes={ml_dtypes.__version__} expected={expected}"
        )
    return SeqaxBf16Runtime(
        runtime=runtime,
        ml_dtypes=ml_dtypes.__version__,
        cpu_machine=platform.machine(),
        cpu_system=platform.system(),
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
    if value.endswith(("/physical.xdsl", "mutant_physical.xdsl")):
        return ArtifactRole.PHYSICAL_IR
    if value.endswith("/lowered_pallas.py"):
        return ArtifactRole.PALLAS_SOURCE
    if value.endswith("/lowered_control.py"):
        return ArtifactRole.JAX_SOURCE
    if value.endswith(("/plan.json", "_manifest.json")):
        return ArtifactRole.PLAN_MANIFEST
    if value.endswith("stablehlo.txt"):
        return ArtifactRole.STABLEHLO
    if value.endswith("compiler_hlo.txt"):
        return ArtifactRole.COMPILER_HLO
    if "/inputs/" in value:
        return ArtifactRole.CORRECTNESS_INPUT
    if value.endswith("cpu_reference.npy"):
        return ArtifactRole.ORACLE_OUTPUT
    if value.endswith(".npy"):
        return ArtifactRole.CORRECTNESS_OUTPUT
    return ArtifactRole.SEARCH_EVIDENCE


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    return tuple(
        ArtifactReference(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            role=_artifact_role(path.relative_to(root)),
        )
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.name != "receipt.json"
    )


def _build_receipt(
    root: Path,
    contract: SeqaxBf16ValidationContract,
    run_id: str,
) -> SeqaxBf16ValidationReceipt:
    path = root / "receipt.json"
    if path.exists():
        raise ValueError("SEQAX_BF16_RECEIPT_EXISTS")
    receipt = SeqaxBf16ValidationReceipt(
        schema_version="seqax-bf16-forward-validation-receipt-v5",
        contract_id=contract.contract_id,
        run_id=run_id,
        status="passed",
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(root / "ledger.sqlite"),
        artifacts=_artifact_manifest(root),
    )
    _write_json_atomic(path, receipt.model_dump(mode="json"))
    return receipt


def _execute_seqax_bf16_validation(
    root: Path,
    contract: SeqaxBf16ValidationContract,
    runtime: SeqaxBf16Runtime,
    devices: tuple[Any, ...],
    run_id: str,
    source_commit: str,
) -> SeqaxBf16ValidationResult:
    repository_root = Path(__file__).resolve().parents[2]
    _write_json(
        root / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    _source_state(repository_root, root)
    source_state = json.loads((root / "source_state.json").read_text())
    if source_state["git_commit"] != source_commit:
        raise ValueError("SEQAX_BF16_SOURCE_CHANGED_DURING_RUN")
    ledger_path = root / "ledger.sqlite"
    with ExperimentLedger(ledger_path) as ledger:
        _transition_or_replay(
            ledger,
            run_id,
            RunState.CREATED,
            {
                "schema": SEQAX_BF16_RUN_SCHEMA,
                "contract_id": contract.contract_id,
                "source_commit": source_state["git_commit"],
            },
        )

    compiled = tuple(_prepare_scenario(root, scenario, devices) for scenario in contract.scenarios)
    with ExperimentLedger(ledger_path) as ledger:
        _transition_or_replay(
            ledger,
            run_id,
            RunState.VERIFIED,
            {
                "distributed_schedules": {
                    value.scenario.name: value.record.distributed_schedule_sha256
                    for value in compiled
                }
            },
        )
        _transition_or_replay(
            ledger,
            run_id,
            RunState.LOWERED,
            {
                "physical_schedules": {
                    value.scenario.name: value.record.physical_schedule_sha256 for value in compiled
                },
                "pallas_sources": {
                    value.scenario.name: value.record.pallas_source_sha256 for value in compiled
                },
            },
        )
        _transition_or_replay(
            ledger,
            run_id,
            RunState.COMPILED,
            {"plans": [value.record.model_dump(mode="json") for value in compiled]},
        )

    observations = tuple(
        _run_seed(root, value, seed, contract)
        for value in compiled
        for seed in value.scenario.seeds
    )
    with ExperimentLedger(ledger_path) as ledger:
        _transition_or_replay(
            ledger,
            run_id,
            RunState.CORRECT,
            {
                "observations": len(observations),
                "pallas_outputs": [value.pallas_output_sha256 for value in observations],
                "control_outputs": [value.control_output_sha256 for value in observations],
            },
        )

    discriminators = _run_discriminators(root, contract, compiled[0], devices)
    result = SeqaxBf16ValidationResult(
        schema_version=SEQAX_BF16_RUN_SCHEMA,
        contract_id=contract.contract_id,
        run_id=run_id,
        runtime=runtime,
        devices=_device_inventory(devices),
        source_state_sha256=_sha256(root / "source_state.json"),
        source_manifest=_source_manifest(),
        plans=tuple(value.record for value in compiled),
        observations=observations,
        discriminators=discriminators,
        passed=True,
        claim_scope="declared-surface-dual-jax-cpu-bf16-numerical-agreement-v1",
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    with ExperimentLedger(ledger_path) as ledger:
        _transition_or_replay(
            ledger,
            run_id,
            RunState.VALIDATED,
            {
                "discriminators": [
                    {
                        "name": value.discriminator.value,
                        "artifacts": list(value.artifact_sha256),
                    }
                    for value in discriminators
                ],
                "result_sha256": _sha256(root / "result.json"),
            },
        )
        already_accepted = ledger.current_state(run_id) is RunState.ACCEPTED
    _close_ledger(ledger_path)
    _validate(
        root,
        contract,
        require_accepted=already_accepted,
        require_receipt=False,
    )
    with ExperimentLedger(ledger_path) as ledger:
        _transition_or_replay(
            ledger,
            run_id,
            RunState.ACCEPTED,
            {"result_sha256": _sha256(root / "result.json"), "passed": True},
        )
    _close_ledger(ledger_path)
    _build_receipt(root, contract, run_id)
    return validate_seqax_bf16_validation(root, contract)


def run_seqax_bf16_validation(
    root: Path,
    contract: SeqaxBf16ValidationContract,
) -> SeqaxBf16ValidationResult:
    canonical_contract = default_seqax_bf16_validation_contract()
    if contract != canonical_contract:
        raise ValueError("SEQAX_BF16_EXTERNAL_CONTRACT_MISMATCH")
    if root.is_symlink():
        raise ValueError(f"SEQAX_BF16_ROOT_INVALID path={root}")
    root = root.resolve()
    _require_safe_root(root)
    with _exclusive_run_lock(root):
        repository_root = Path(__file__).resolve().parents[2]
        _require_clean_repository(repository_root)
        if _sha256(repository_root / "uv.lock") != contract.runtime.uv_lock_sha256:
            raise ValueError("SEQAX_BF16_LOCK_MISMATCH")
        if (root / "receipt.json").exists() or (root / "receipt.json").is_symlink():
            return validate_seqax_bf16_validation(root, contract)
        _require_compilation_source_root(repository_root, contract)
        if contract.hlo_identity_status != "pinned":
            raise ValueError("SEQAX_BF16_HLO_IDENTITIES_PENDING")
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        run_id = semantic_sha256(
            SEQAX_BF16_RUN_SCHEMA,
            contract.contract_id,
            source_commit,
        )
        identity = SeqaxBf16RunIdentity(
            schema_version=SEQAX_BF16_RUN_SCHEMA,
            contract_id=contract.contract_id,
            run_id=run_id,
            source_commit=source_commit,
        )
        runtime = _runtime(contract)
        devices = tuple(jax.devices())
        _validate_devices(devices, contract)
        _prepare_output_root(root, identity, contract)
        _write_json_atomic(root / "run_identity.json", identity.model_dump(mode="json"))
        try:
            return _execute_seqax_bf16_validation(
                root,
                contract,
                runtime,
                devices,
                run_id,
                source_commit,
            )
        except Exception as error:
            _record_failure(root, run_id, error)
            raise


def _require_compilation_source_root(
    repository_root: Path,
    contract: SeqaxBf16ValidationContract,
) -> None:
    observed = repository_root.resolve()
    expected = Path(contract.compilation_source_root)
    if observed != expected:
        raise ValueError(
            f"SEQAX_BF16_COMPILATION_SOURCE_ROOT_MISMATCH observed={observed} expected={expected}"
        )


def _canonical_plans(
    scenario: SeqaxBf16NumericalScenario,
) -> tuple[Any, Any, SeqaxPallasPlan, JaxDistributedMeshPlan]:
    distributed = seqax_forward_schedule(
        **scenario.parameters.model_dump(),
        numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
    )
    physical = lower_seqax_forward_to_physical(distributed).module
    return (
        distributed,
        physical,
        lower_seqax_physical_to_pallas(distributed, physical),
        lower_distributed_program_to_jax_mesh(distributed),
    )


def _expected_files(
    root: Path,
    contract: SeqaxBf16ValidationContract,
    *,
    receipt_present: bool,
) -> set[Path]:
    expected = {
        root / "run_identity.json",
        root / "contract.json",
        root / "source_state.json",
        root / "source_diff.patch",
        root / "ledger.sqlite",
        root / "result.json",
        root / "discriminators" / "observations.json",
    }
    plan_files = (
        "distributed.xdsl",
        "physical.xdsl",
        "lowered_pallas.py",
        "lowered_control.py",
        "pallas_stablehlo.txt",
        "pallas_compiler_hlo.txt",
        "control_stablehlo.txt",
        "control_compiler_hlo.txt",
        "instrumented_pallas_stablehlo.txt",
        "instrumented_pallas_compiler_hlo.txt",
        "instrumented_control_stablehlo.txt",
        "instrumented_control_compiler_hlo.txt",
        "pallas_manifest.json",
        "control_manifest.json",
        "plan.json",
    )
    for scenario in contract.scenarios:
        expected.update(_plan_root(root, scenario) / name for name in plan_files)
        for seed in scenario.seeds:
            seed_root = _seed_root(root, scenario.name, seed)
            expected.update(seed_root / "inputs" / f"{index:02d}.npy" for index in range(13))
            expected.update(
                seed_root / name
                for name in (
                    "cpu_reference.npy",
                    "pallas_output.npy",
                    "control_output.npy",
                    "instrumented_pallas_output.npy",
                    "instrumented_control_output.npy",
                    "observation.json",
                )
            )
            for layer in range(scenario.parameters.layers):
                expected.update(
                    seed_root / "checkpoints" / f"{path}_{kind}_{layer:02d}.npy"
                    for path in ("pallas", "control")
                    for kind in (
                        "rms_input",
                        "rms_mean_square",
                        "rms_inverse",
                        "normalized_float32",
                        "normalized_input",
                        "gate_float32",
                        "gate",
                        "silu",
                        "up_float32",
                        "up",
                        "hidden",
                        "down_float32",
                        "down_bfloat16",
                    )
                )
    for discriminator in contract.required_discriminators:
        discriminator_root = root / "discriminators" / discriminator
        if discriminator in {
            SeqaxNumericalDiscriminator.REMOVE_INPUT_BARRIER,
            SeqaxNumericalDiscriminator.REMOVE_OUTPUT_BARRIER,
            SeqaxNumericalDiscriminator.REMOVE_HIDDEN_BARRIER,
            SeqaxNumericalDiscriminator.BYPASS_RMS_NORM_CHECKPOINT,
        }:
            expected.add(discriminator_root / "mutant_stablehlo.txt")
        elif discriminator in {
            SeqaxNumericalDiscriminator.IDENTITY_SILU,
            SeqaxNumericalDiscriminator.RELU_SILU,
        }:
            expected.update(
                discriminator_root / name
                for name in (
                    "mutant_stablehlo.txt",
                    "pallas_runtime_stablehlo.txt",
                    "pallas_runtime_output.npy",
                    "control_runtime_stablehlo.txt",
                    "control_runtime_output.npy",
                )
            )
        elif discriminator is SeqaxNumericalDiscriminator.CORRUPT_DOWN_CHECKPOINT:
            expected.add(discriminator_root / "mutant_down_float32.npy")
        elif discriminator in {
            SeqaxNumericalDiscriminator.WRONG_RMS_SCALE_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_RMS_MEAN_SQUARE_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_RMS_INV_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_FLOAT32_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT,
            SeqaxNumericalDiscriminator.WRONG_GATE_WEIGHT_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_GATE_FLOAT32_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_GATE_BFLOAT16_CHECKPOINT,
            SeqaxNumericalDiscriminator.WRONG_UP_WEIGHT_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_UP_FLOAT32_CHECKPOINT,
            SeqaxNumericalDiscriminator.CORRUPT_UP_BFLOAT16_CHECKPOINT,
        }:
            expected.add(discriminator_root / "mutant_checkpoint.npy")
        elif discriminator is SeqaxNumericalDiscriminator.DROP_REDUCTION_COLLECTIVE:
            expected.add(discriminator_root / "mutant_physical.xdsl")
        elif discriminator in {
            SeqaxNumericalDiscriminator.DROP_EMBEDDING_SHARD,
            SeqaxNumericalDiscriminator.ROLL_MODEL_SHARD,
            SeqaxNumericalDiscriminator.OMIT_MLP_TERM,
            SeqaxNumericalDiscriminator.SWAP_GATE_UP,
        }:
            expected.update(
                discriminator_root / "inputs" / f"{index:02d}.npy" for index in range(13)
            )
            expected.add(discriminator_root / "runtime_output.npy")
        else:
            expected.add(discriminator_root / "mutant_output.npy")
    if receipt_present:
        expected.add(root / "receipt.json")
    return {path.resolve() for path in expected}


def _validate_closed_world(root: Path, expected: set[Path]) -> None:
    _preflight_existing_root(root)
    observed = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if observed != expected:
        raise ValueError(
            "SEQAX_BF16_CLOSED_WORLD_MISMATCH "
            f"missing={sorted(map(str, expected - observed))} "
            f"extra={sorted(map(str, observed - expected))}"
        )


def _validate_source(
    root: Path,
    contract: SeqaxBf16ValidationContract,
    result: SeqaxBf16ValidationResult,
) -> str:
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state_path = root / "source_state.json"
    state = json.loads(state_path.read_text())
    if (
        result.source_state_sha256 != _sha256(state_path)
        or state.get("git_commit") != commit
        or state.get("git_dirty") is not False
        or state.get("git_status") != []
        or state.get("uv_lock_sha256") != contract.runtime.uv_lock_sha256
        or _sha256(repository_root / "uv.lock") != contract.runtime.uv_lock_sha256
        or (root / "source_diff.patch").read_bytes() != b""
        or result.source_manifest != _source_manifest()
    ):
        raise ValueError("SEQAX_BF16_SOURCE_STATE_MISMATCH")
    for source in result.source_manifest:
        blob = subprocess.run(
            ["git", "show", f"{commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(f"SEQAX_BF16_SOURCE_BLOB_MISMATCH path={source.path}")
    return commit


def _validate_plan(
    root: Path,
    scenario: SeqaxBf16NumericalScenario,
    record: SeqaxBf16PlanRecord,
) -> None:
    distributed, physical, pallas_plan, control_plan = _canonical_plans(scenario)
    plan_root = _plan_root(root, scenario)
    if (
        (plan_root / "distributed.xdsl").read_text() != canonical_text(distributed)
        or (plan_root / "physical.xdsl").read_text() != canonical_text(physical)
        or (plan_root / "lowered_pallas.py").read_text() != pallas_plan.render_executable_source()
        or (plan_root / "lowered_control.py").read_text() != control_plan.render_executable_source()
        or json.loads((plan_root / "pallas_manifest.json").read_text()) != pallas_plan.manifest()
        or json.loads((plan_root / "control_manifest.json").read_text()) != control_plan.manifest()
    ):
        raise ValueError(f"SEQAX_BF16_PLAN_REPLAY_MISMATCH scenario={scenario.name}")
    pallas_stablehlo = (plan_root / "pallas_stablehlo.txt").read_text()
    control_stablehlo = (plan_root / "control_stablehlo.txt").read_text()
    pallas_compiler_hlo = (plan_root / "pallas_compiler_hlo.txt").read_text()
    validate_strict_silu_stablehlo(
        pallas_stablehlo,
        expected_count=scenario.parameters.layers,
        expected_sha256=scenario.pallas_stablehlo_sha256,
    )
    validate_strict_silu_stablehlo(
        control_stablehlo,
        expected_count=scenario.parameters.layers,
        expected_sha256=scenario.control_stablehlo_sha256,
    )
    validate_instrumented_strict_silu_stablehlo(
        (plan_root / "instrumented_pallas_stablehlo.txt").read_text(),
        expected_count=scenario.parameters.layers,
        expected_sha256=scenario.instrumented_pallas_stablehlo_sha256,
    )
    validate_instrumented_strict_silu_stablehlo(
        (plan_root / "instrumented_control_stablehlo.txt").read_text(),
        expected_count=scenario.parameters.layers,
        expected_sha256=scenario.instrumented_control_stablehlo_sha256,
    )
    all_gathers, reduce_scatters = _physical_collective_counts(physical)
    _validate_compiled_program(
        pallas_stablehlo,
        pallas_compiler_hlo,
        pallas_region_count=pallas_plan.pallas_region_count,
        pallas_vector_region_count=pallas_plan.pallas_vector_region_count,
        all_gather_count=all_gathers,
        reduce_scatter_count=reduce_scatters,
    )
    expected = SeqaxBf16PlanRecord(
        scenario=scenario.name,
        distributed_schedule_sha256=pallas_plan.distributed_schedule_sha256,
        physical_schedule_sha256=pallas_plan.physical_schedule_sha256,
        pallas_source_sha256=pallas_plan.source_sha256(),
        control_source_sha256=control_plan.source_sha256(),
        pallas_stablehlo_sha256=_sha256(plan_root / "pallas_stablehlo.txt"),
        pallas_compiler_hlo_sha256=_sha256(plan_root / "pallas_compiler_hlo.txt"),
        control_stablehlo_sha256=_sha256(plan_root / "control_stablehlo.txt"),
        control_compiler_hlo_sha256=_sha256(plan_root / "control_compiler_hlo.txt"),
        instrumented_pallas_stablehlo_sha256=_sha256(
            plan_root / "instrumented_pallas_stablehlo.txt"
        ),
        instrumented_pallas_compiler_hlo_sha256=_sha256(
            plan_root / "instrumented_pallas_compiler_hlo.txt"
        ),
        instrumented_control_stablehlo_sha256=_sha256(
            plan_root / "instrumented_control_stablehlo.txt"
        ),
        instrumented_control_compiler_hlo_sha256=_sha256(
            plan_root / "instrumented_control_compiler_hlo.txt"
        ),
        pallas_region_count=pallas_plan.pallas_region_count,
        all_gather_count=all_gathers,
        reduce_scatter_count=reduce_scatters,
        strict_silu_count=scenario.parameters.layers,
        strict_hidden_count=scenario.parameters.layers,
    )
    saved = SeqaxBf16PlanRecord.model_validate_json((plan_root / "plan.json").read_text())
    if record != expected or saved != expected:
        raise ValueError(f"SEQAX_BF16_PLAN_IDENTITY_MISMATCH scenario={scenario.name}")


def _validate_seed(
    root: Path,
    contract: SeqaxBf16ValidationContract,
    scenario: SeqaxBf16NumericalScenario,
    seed: int,
) -> SeqaxBf16SeedObservation:
    seed_root = _seed_root(root, scenario.name, seed)
    expected_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=seed, **scenario.parameters.model_dump())
    )
    saved_inputs = tuple(
        _load_array(seed_root / "inputs" / f"{index:02d}.npy") for index in range(13)
    )
    if any(
        saved.shape != expected.shape
        or saved.dtype != expected.dtype
        or not np.array_equal(saved, expected)
        for saved, expected in zip(saved_inputs, expected_inputs, strict=True)
    ):
        raise ValueError(f"SEQAX_BF16_INPUT_REPLAY_MISMATCH scenario={scenario.name} seed={seed}")
    fresh_cpu = np.asarray(
        seqax_forward_canonical_reference(
            expected_inputs,
            quantization_decimals=contract.policy.cpu_reference_quantization_decimals,
            **scenario.parameters.model_dump(),
        )
    )
    saved_cpu = _load_array(seed_root / "cpu_reference.npy")
    cpu_replay = assess_seqax_cpu_reference_replay(
        saved_cpu,
        fresh_cpu,
        policy=contract.policy,
        scenario=scenario,
    )
    if not cpu_replay.within_bounds:
        raise ValueError(f"SEQAX_BF16_CPU_REPLAY_MISMATCH scenario={scenario.name} seed={seed}")
    pallas = _load_array(seed_root / "pallas_output.npy")
    control = _load_array(seed_root / "control_output.npy")
    instrumented_pallas = _load_array(seed_root / "instrumented_pallas_output.npy")
    instrumented_control = _load_array(seed_root / "instrumented_control_output.npy")
    (
        pallas_rms_inputs,
        pallas_rms_mean_square,
        pallas_rms_inverse,
        pallas_normalized_float32,
        pallas_normalized_inputs,
        pallas_gate_float32,
        pallas_gates,
        pallas_silus,
        pallas_up_float32,
        pallas_up,
        pallas_hidden,
        pallas_down_float32,
        pallas_down_bfloat16,
    ) = _load_saved_checkpoints(seed_root, scenario, "pallas")
    (
        control_rms_inputs,
        control_rms_mean_square,
        control_rms_inverse,
        control_normalized_float32,
        control_normalized_inputs,
        control_gate_float32,
        control_gates,
        control_silus,
        control_up_float32,
        control_up,
        control_hidden,
        control_down_float32,
        control_down_bfloat16,
    ) = _load_saved_checkpoints(seed_root, scenario, "control")
    evidence = {
        "seed": seed,
        "inputs": saved_inputs,
        "pallas_rms_input_checkpoints": pallas_rms_inputs,
        "control_rms_input_checkpoints": control_rms_inputs,
        "pallas_rms_mean_square_checkpoints": pallas_rms_mean_square,
        "control_rms_mean_square_checkpoints": control_rms_mean_square,
        "pallas_rms_inverse_checkpoints": pallas_rms_inverse,
        "control_rms_inverse_checkpoints": control_rms_inverse,
        "pallas_normalized_float32_checkpoints": pallas_normalized_float32,
        "control_normalized_float32_checkpoints": control_normalized_float32,
        "pallas_normalized_input_checkpoints": pallas_normalized_inputs,
        "control_normalized_input_checkpoints": control_normalized_inputs,
        "pallas_gate_float32_checkpoints": pallas_gate_float32,
        "control_gate_float32_checkpoints": control_gate_float32,
        "pallas_gate_checkpoints": pallas_gates,
        "control_gate_checkpoints": control_gates,
        "pallas_silu_checkpoints": pallas_silus,
        "control_silu_checkpoints": control_silus,
        "pallas_up_float32_checkpoints": pallas_up_float32,
        "control_up_float32_checkpoints": control_up_float32,
        "pallas_up_checkpoints": pallas_up,
        "control_up_checkpoints": control_up,
        "pallas_hidden_checkpoints": pallas_hidden,
        "control_hidden_checkpoints": control_hidden,
        "pallas_down_float32_checkpoints": pallas_down_float32,
        "control_down_float32_checkpoints": control_down_float32,
        "pallas_down_bfloat16_checkpoints": pallas_down_bfloat16,
        "control_down_bfloat16_checkpoints": control_down_bfloat16,
        "policy": contract.policy,
        "scenario": scenario,
    }
    fresh_normal_assessment = assess_seqax_bf16_outputs(
        pallas,
        control,
        seed=seed,
        inputs=saved_inputs,
        policy=contract.policy,
        scenario=scenario,
    )
    fresh_instrumented_assessment = assess_seqax_bf16_forward(
        instrumented_pallas,
        instrumented_control,
        **evidence,
    )
    saved_normal_assessment = _assess_output_arrays(
        pallas,
        control,
        saved_cpu,
        policy=contract.policy,
        scenario=scenario,
    )
    saved_instrumented_output_assessment = _assess_output_arrays(
        instrumented_pallas,
        instrumented_control,
        saved_cpu,
        policy=contract.policy,
        scenario=scenario,
    )
    saved_instrumented_assessment = fresh_instrumented_assessment.model_copy(
        update=saved_instrumented_output_assessment.model_dump()
    )
    expected = SeqaxBf16SeedObservation(
        scenario=scenario.name,
        seed=seed,
        input_sha256=arrays_sha256(saved_inputs),
        cpu_reference_sha256=array_sha256(saved_cpu),
        pallas_output_sha256=array_sha256(pallas),
        control_output_sha256=array_sha256(control),
        instrumented_pallas_output_sha256=array_sha256(instrumented_pallas),
        instrumented_control_output_sha256=array_sha256(instrumented_control),
        pallas_rms_input_sha256=_checkpoint_hashes(pallas_rms_inputs),
        control_rms_input_sha256=_checkpoint_hashes(control_rms_inputs),
        pallas_rms_mean_square_sha256=_checkpoint_hashes(pallas_rms_mean_square),
        control_rms_mean_square_sha256=_checkpoint_hashes(control_rms_mean_square),
        pallas_rms_inverse_sha256=_checkpoint_hashes(pallas_rms_inverse),
        control_rms_inverse_sha256=_checkpoint_hashes(control_rms_inverse),
        pallas_normalized_float32_sha256=_checkpoint_hashes(pallas_normalized_float32),
        control_normalized_float32_sha256=_checkpoint_hashes(control_normalized_float32),
        pallas_normalized_input_sha256=_checkpoint_hashes(pallas_normalized_inputs),
        control_normalized_input_sha256=_checkpoint_hashes(control_normalized_inputs),
        pallas_gate_float32_sha256=_checkpoint_hashes(pallas_gate_float32),
        control_gate_float32_sha256=_checkpoint_hashes(control_gate_float32),
        pallas_gate_sha256=_checkpoint_hashes(pallas_gates),
        control_gate_sha256=_checkpoint_hashes(control_gates),
        pallas_silu_sha256=_checkpoint_hashes(pallas_silus),
        control_silu_sha256=_checkpoint_hashes(control_silus),
        pallas_up_float32_sha256=_checkpoint_hashes(pallas_up_float32),
        control_up_float32_sha256=_checkpoint_hashes(control_up_float32),
        pallas_up_sha256=_checkpoint_hashes(pallas_up),
        control_up_sha256=_checkpoint_hashes(control_up),
        pallas_hidden_sha256=_checkpoint_hashes(pallas_hidden),
        control_hidden_sha256=_checkpoint_hashes(control_hidden),
        pallas_down_float32_sha256=_checkpoint_hashes(pallas_down_float32),
        control_down_float32_sha256=_checkpoint_hashes(control_down_float32),
        pallas_down_bfloat16_sha256=_checkpoint_hashes(pallas_down_bfloat16),
        control_down_bfloat16_sha256=_checkpoint_hashes(control_down_bfloat16),
        normal_assessment=saved_normal_assessment,
        instrumented_assessment=saved_instrumented_assessment,
        instrumentation_difference=_instrumentation_difference(
            pallas,
            control,
            instrumented_pallas,
            instrumented_control,
            contract,
        ),
    )
    saved = SeqaxBf16SeedObservation.model_validate_json(
        (seed_root / "observation.json").read_text()
    )
    if (
        expected != saved
        or not fresh_normal_assessment.final_outputs_satisfy_policy
        or not fresh_instrumented_assessment.final_outputs_satisfy_policy
        or not fresh_instrumented_assessment.checkpoint_values_consistent
        or not saved_normal_assessment.final_outputs_satisfy_policy
        or not saved_instrumented_assessment.final_outputs_satisfy_policy
        or not saved_instrumented_assessment.checkpoint_values_consistent
    ):
        raise ValueError(
            f"SEQAX_BF16_OBSERVATION_REPLAY_MISMATCH scenario={scenario.name} seed={seed}"
        )
    return expected


def _discriminator_observation(
    root: Path,
    discriminator: SeqaxNumericalDiscriminator,
    paths: tuple[Path, ...],
    failure: str,
) -> SeqaxBf16DiscriminatorObservation:
    artifact_paths, artifact_sha256 = _artifact_identities(root, paths)
    return SeqaxBf16DiscriminatorObservation(
        discriminator=discriminator,
        clause=seqax_discriminator_clause(discriminator),
        artifact_paths=artifact_paths,
        artifact_sha256=artifact_sha256,
        rejected=True,
        failure=failure,
    )


def _validate_discriminators(
    root: Path,
    contract: SeqaxBf16ValidationContract,
) -> tuple[SeqaxBf16DiscriminatorObservation, ...]:
    scenario = contract.scenarios[0]
    seed = scenario.seeds[0]
    seed_root = _seed_root(root, scenario.name, seed)
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=seed, **scenario.parameters.model_dump())
    )
    baseline = _load_array(seed_root / "pallas_output.npy")
    (
        rms_inputs,
        rms_mean_square,
        rms_inverse,
        normalized_float32,
        normalized_inputs,
        gate_float32,
        gates,
        silus,
        up_float32,
        up,
        hidden,
        down_float32,
        down_bfloat16,
    ) = _load_saved_checkpoints(seed_root, scenario, "pallas")
    saved_cpu = _load_array(seed_root / "cpu_reference.npy")
    fresh_cpu = np.asarray(
        seqax_forward_canonical_reference(
            host_inputs,
            quantization_decimals=contract.policy.cpu_reference_quantization_decimals,
            **scenario.parameters.model_dump(),
        )
    )
    if not assess_seqax_cpu_reference_replay(
        saved_cpu,
        fresh_cpu,
        policy=contract.policy,
        scenario=scenario,
    ).within_bounds:
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_CPU_REPLAY_MISMATCH")

    def recorded_mutation_failure(
        output: np.ndarray,
        clause: SeqaxDiscriminatorClause,
    ) -> str:
        arguments = {
            "clause": clause,
            "contract": contract,
            "scenario": scenario,
            "seed": seed,
            "inputs": host_inputs,
            "rms_inputs": rms_inputs,
            "rms_mean_square": rms_mean_square,
            "rms_inverse": rms_inverse,
            "normalized_float32": normalized_float32,
            "normalized_inputs": normalized_inputs,
            "gate_float32": gate_float32,
            "gates": gates,
            "silus": silus,
            "up_float32": up_float32,
            "up": up,
            "hidden": hidden,
            "down_float32": down_float32,
            "down_bfloat16": down_bfloat16,
        }
        saved_failure = _mutation_failure(
            output,
            cpu_reference=saved_cpu,
            **arguments,
        )
        _mutation_failure(
            output,
            cpu_reference=fresh_cpu,
            **arguments,
        )
        return saved_failure

    def checkpoint_mutation_assessments(
        **mutation: tuple[np.ndarray, ...],
    ) -> tuple[SeqaxBf16NumericalAssessment, SeqaxBf16NumericalAssessment]:
        arguments = {
            "contract": contract,
            "scenario": scenario,
            "seed": seed,
            "inputs": host_inputs,
            "rms_inputs": rms_inputs,
            "rms_mean_square": rms_mean_square,
            "rms_inverse": rms_inverse,
            "normalized_float32": normalized_float32,
            "normalized_inputs": normalized_inputs,
            "gate_float32": gate_float32,
            "gates": gates,
            "silus": silus,
            "up_float32": up_float32,
            "up": up,
            "hidden": hidden,
            "down_float32": down_float32,
            "down_bfloat16": down_bfloat16,
            **mutation,
        }
        return (
            _assess_checkpoint_mutation(
                baseline,
                cpu_reference=saved_cpu,
                **arguments,
            ),
            _assess_checkpoint_mutation(
                baseline,
                cpu_reference=fresh_cpu,
                **arguments,
            ),
        )

    pallas_hlo = canonical_seqax_stablehlo(
        (_plan_root(root, scenario) / "pallas_stablehlo.txt").read_text()
    )
    expected = []

    hlo_mutants = {
        SeqaxNumericalDiscriminator.REMOVE_INPUT_BARRIER: _remove_strict_barrier(
            pallas_hlo, input_barrier=True
        ),
        SeqaxNumericalDiscriminator.REMOVE_OUTPUT_BARRIER: _remove_strict_barrier(
            pallas_hlo, input_barrier=False
        ),
        SeqaxNumericalDiscriminator.REMOVE_HIDDEN_BARRIER: _remove_hidden_barrier(pallas_hlo),
        SeqaxNumericalDiscriminator.IDENTITY_SILU: _replace_silu_body(pallas_hlo, relu=False),
        SeqaxNumericalDiscriminator.RELU_SILU: _replace_silu_body(pallas_hlo, relu=True),
    }
    hlo_mutants = {
        discriminator: canonical_seqax_stablehlo(mutant)
        for discriminator, mutant in hlo_mutants.items()
    }
    for discriminator, mutant in hlo_mutants.items():
        path = root / "discriminators" / discriminator / "mutant_stablehlo.txt"
        if path.read_text() != mutant:
            raise ValueError(f"SEQAX_BF16_HLO_MUTANT_REPLAY_MISMATCH name={discriminator}")
        try:
            _validate_strict_silu_stablehlo(
                mutant,
                expected_count=scenario.parameters.layers,
                instrumented=False,
            )
        except ValueError as error:
            failure = f"{type(error).__name__}: {error}"
        else:
            raise ValueError(f"SEQAX_BF16_HLO_DISCRIMINATOR_ACCEPTED name={discriminator}")
        paths = [path]
        if discriminator in {
            SeqaxNumericalDiscriminator.IDENTITY_SILU,
            SeqaxNumericalDiscriminator.RELU_SILU,
        }:
            for label in ("pallas", "control"):
                runtime_hlo_path = path.with_name(f"{label}_runtime_stablehlo.txt")
                runtime_output_path = path.with_name(f"{label}_runtime_output.npy")
                runtime_hlo = runtime_hlo_path.read_text()
                runtime_output = _load_array(runtime_output_path)
                validate_activation_mutant_stablehlo(
                    runtime_hlo,
                    expected_count=scenario.parameters.layers,
                    expected_sha256=_activation_mutant_stablehlo_sha256(
                        contract,
                        discriminator,
                        label,
                    ),
                    relu=discriminator is SeqaxNumericalDiscriminator.RELU_SILU,
                )
                causal_failure = recorded_mutation_failure(
                    runtime_output,
                    SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY,
                )
                if np.array_equal(runtime_output, baseline):
                    raise ValueError(
                        f"SEQAX_BF16_ACTIVATION_MUTANT_NOT_CAUSAL name={discriminator} path={label}"
                    )
                failure += f"; {label}_causal {causal_failure}"
                paths.extend((runtime_hlo_path, runtime_output_path))
        expected.append(_discriminator_observation(root, discriminator, tuple(paths), failure))

    rms_bypass_discriminator = SeqaxNumericalDiscriminator.BYPASS_RMS_NORM_CHECKPOINT
    instrumented_pallas_hlo = canonical_seqax_stablehlo(
        (_plan_root(root, scenario) / "instrumented_pallas_stablehlo.txt").read_text()
    )
    rms_bypass = canonical_seqax_stablehlo(_bypass_rms_norm_checkpoint(instrumented_pallas_hlo))
    rms_bypass_path = root / "discriminators" / rms_bypass_discriminator / "mutant_stablehlo.txt"
    if rms_bypass_path.read_text() != rms_bypass:
        raise ValueError(f"SEQAX_BF16_HLO_MUTANT_REPLAY_MISMATCH name={rms_bypass_discriminator}")
    try:
        _validate_strict_silu_stablehlo(
            rms_bypass,
            expected_count=scenario.parameters.layers,
            instrumented=True,
        )
    except ValueError as error:
        rms_bypass_failure = f"{type(error).__name__}: {error}"
    else:
        raise ValueError(f"SEQAX_BF16_HLO_DISCRIMINATOR_ACCEPTED name={rms_bypass_discriminator}")
    expected.append(
        _discriminator_observation(
            root,
            rms_bypass_discriminator,
            (rms_bypass_path,),
            rms_bypass_failure,
        )
    )

    for discriminator, mutant in _checkpoint_mutants(
        host_inputs,
        rms_inputs,
        rms_mean_square,
        rms_inverse,
        normalized_float32,
        normalized_inputs,
        gate_float32,
        gates,
        up_float32,
        up,
    ).items():
        path = root / "discriminators" / discriminator / "mutant_checkpoint.npy"
        saved_storage = _load_array(path)
        expected_storage = _checkpoint_mutant_storage(discriminator, mutant[0], scenario)
        if (
            saved_storage.shape != expected_storage.shape
            or saved_storage.dtype != expected_storage.dtype
            or not np.array_equal(saved_storage, expected_storage)
        ):
            raise ValueError(f"SEQAX_BF16_CHECKPOINT_MUTANT_REPLAY_MISMATCH name={discriminator}")
        replayed_mutant = (
            _decode_checkpoint_mutant(discriminator, saved_storage, scenario),
            *mutant[1:],
        )
        assessment, fresh_assessment = checkpoint_mutation_assessments(
            **_checkpoint_mutation_arguments(discriminator, replayed_mutant),
        )
        if not all(
            _checkpoint_mutation_rejected(discriminator, value)
            for value in (assessment, fresh_assessment)
        ):
            raise ValueError(f"SEQAX_BF16_CHECKPOINT_DISCRIMINATOR_ACCEPTED name={discriminator}")
        failure = (
            f"{seqax_discriminator_clause(discriminator).value}: rejected "
            f"assessment={assessment.model_dump(mode='json')}"
        )
        expected.append(_discriminator_observation(root, discriminator, (path,), failure))

    corrupted_down = tuple(value.copy() for value in down_float32)
    corrupted_down[0].reshape(-1)[0] += np.float32(1e-3)
    down_path = (
        root
        / "discriminators"
        / SeqaxNumericalDiscriminator.CORRUPT_DOWN_CHECKPOINT
        / "mutant_down_float32.npy"
    )
    saved_corrupted_down = _load_array(down_path)
    if not np.array_equal(saved_corrupted_down, corrupted_down[0]):
        raise ValueError("SEQAX_BF16_DOWN_CHECKPOINT_MUTANT_REPLAY_MISMATCH")
    down_assessment = assess_seqax_bf16_forward(
        baseline,
        baseline,
        seed=seed,
        inputs=host_inputs,
        pallas_rms_input_checkpoints=rms_inputs,
        control_rms_input_checkpoints=rms_inputs,
        pallas_rms_mean_square_checkpoints=rms_mean_square,
        control_rms_mean_square_checkpoints=rms_mean_square,
        pallas_rms_inverse_checkpoints=rms_inverse,
        control_rms_inverse_checkpoints=rms_inverse,
        pallas_normalized_float32_checkpoints=normalized_float32,
        control_normalized_float32_checkpoints=normalized_float32,
        pallas_normalized_input_checkpoints=normalized_inputs,
        control_normalized_input_checkpoints=normalized_inputs,
        pallas_gate_float32_checkpoints=gate_float32,
        control_gate_float32_checkpoints=gate_float32,
        pallas_gate_checkpoints=gates,
        control_gate_checkpoints=gates,
        pallas_silu_checkpoints=silus,
        control_silu_checkpoints=silus,
        pallas_up_float32_checkpoints=up_float32,
        control_up_float32_checkpoints=up_float32,
        pallas_up_checkpoints=up,
        control_up_checkpoints=up,
        pallas_hidden_checkpoints=hidden,
        control_hidden_checkpoints=hidden,
        pallas_down_float32_checkpoints=corrupted_down,
        control_down_float32_checkpoints=down_float32,
        pallas_down_bfloat16_checkpoints=down_bfloat16,
        control_down_bfloat16_checkpoints=down_bfloat16,
        policy=contract.policy,
        scenario=scenario,
    )
    if down_assessment.checkpoint_values_consistent:
        raise ValueError("SEQAX_BF16_DOWN_CHECKPOINT_DISCRIMINATOR_ACCEPTED")
    expected.append(
        _discriminator_observation(
            root,
            SeqaxNumericalDiscriminator.CORRUPT_DOWN_CHECKPOINT,
            (down_path,),
            "down_projection_oracle: rejected "
            f"ratio={down_assessment.pallas_down_float32_max_bound_ratio}",
        )
    )

    _distributed, physical, _pallas, _control = _canonical_plans(scenario)
    collective = _drop_reduction_collective(canonical_text(physical))
    collective_path = (
        root
        / "discriminators"
        / SeqaxNumericalDiscriminator.DROP_REDUCTION_COLLECTIVE
        / "mutant_physical.xdsl"
    )
    if collective_path.read_text() != collective:
        raise ValueError("SEQAX_BF16_COLLECTIVE_MUTANT_REPLAY_MISMATCH")
    try:
        _parse_physical(collective).verify()
    except VerifyException as error:
        failure = f"{type(error).__name__}: {error}"
    else:
        raise ValueError("SEQAX_BF16_COLLECTIVE_DISCRIMINATOR_ACCEPTED")
    expected.append(
        _discriminator_observation(
            root,
            SeqaxNumericalDiscriminator.DROP_REDUCTION_COLLECTIVE,
            (collective_path,),
            failure,
        )
    )

    input_mutations = {
        SeqaxNumericalDiscriminator.DROP_EMBEDDING_SHARD: (SeqaxInputMutation.DROP_EMBEDDING_SHARD),
        SeqaxNumericalDiscriminator.ROLL_MODEL_SHARD: SeqaxInputMutation.ROLL_MODEL_SHARD,
        SeqaxNumericalDiscriminator.OMIT_MLP_TERM: SeqaxInputMutation.OMIT_MLP_TERM,
        SeqaxNumericalDiscriminator.SWAP_GATE_UP: SeqaxInputMutation.SWAP_GATE_UP,
    }
    for discriminator, mutation in input_mutations.items():
        mutation_root = root / "discriminators" / discriminator
        regenerated = mutate_seqax_forward_inputs(host_inputs, mutation)
        saved_inputs = tuple(
            _load_array(mutation_root / "inputs" / f"{index:02d}.npy") for index in range(13)
        )
        if any(
            saved.shape != fresh.shape
            or saved.dtype != fresh.dtype
            or not np.array_equal(saved, fresh)
            for saved, fresh in zip(saved_inputs, regenerated, strict=True)
        ):
            raise ValueError(f"SEQAX_BF16_INPUT_MUTANT_REPLAY_MISMATCH name={discriminator}")
        output_path = mutation_root / "runtime_output.npy"
        output = _load_array(output_path)
        failure = recorded_mutation_failure(
            output,
            SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY,
        )
        paths = tuple(
            [mutation_root / "inputs" / f"{index:02d}.npy" for index in range(13)] + [output_path]
        )
        expected.append(_discriminator_observation(root, discriminator, paths, failure))

    output_mutations: dict[SeqaxNumericalDiscriminator, np.ndarray] = {}
    localized = baseline.copy()
    localized.reshape(-1)[0] += np.float32(1.0)
    output_mutations[SeqaxNumericalDiscriminator.LOCALIZED_SPIKE] = localized
    output_mutations[SeqaxNumericalDiscriminator.DISTRIBUTED_DRIFT] = baseline + np.float32(0.5)
    nonfinite = baseline.copy()
    nonfinite.reshape(-1)[0] = np.nan
    output_mutations[SeqaxNumericalDiscriminator.NONFINITE_OUTPUT] = nonfinite
    output_mutations[SeqaxNumericalDiscriminator.DTYPE_OUTPUT] = baseline.astype(np.float64)
    output_mutations[SeqaxNumericalDiscriminator.SHAPE_OUTPUT] = baseline[..., :-1]
    for discriminator, regenerated in output_mutations.items():
        path = root / "discriminators" / discriminator / "mutant_output.npy"
        saved = _load_array(path)
        if (
            saved.shape != regenerated.shape
            or saved.dtype != regenerated.dtype
            or not np.array_equal(saved, regenerated, equal_nan=True)
        ):
            raise ValueError(f"SEQAX_BF16_OUTPUT_MUTANT_REPLAY_MISMATCH name={discriminator}")
        failure = recorded_mutation_failure(
            saved,
            seqax_discriminator_clause(discriminator),
        )
        expected.append(_discriminator_observation(root, discriminator, (path,), failure))

    expected_tuple = tuple(expected)
    saved_tuple = tuple(
        SeqaxBf16DiscriminatorObservation.model_validate_json(json.dumps(value))
        for value in json.loads((root / "discriminators" / "observations.json").read_text())
    )
    if (
        tuple(value.discriminator for value in expected_tuple) != contract.required_discriminators
        or saved_tuple != expected_tuple
    ):
        raise ValueError("SEQAX_BF16_DISCRIMINATOR_REPLAY_MISMATCH")
    return expected_tuple


def _ledger_payloads(
    contract: SeqaxBf16ValidationContract,
    result: SeqaxBf16ValidationResult,
    source_commit: str,
    result_sha256: str,
    *,
    require_accepted: bool,
) -> tuple[tuple[RunState, dict[str, object]], ...]:
    payloads: tuple[tuple[RunState, dict[str, object]], ...] = (
        (
            RunState.CREATED,
            {
                "schema": SEQAX_BF16_RUN_SCHEMA,
                "contract_id": contract.contract_id,
                "source_commit": source_commit,
            },
        ),
        (
            RunState.VERIFIED,
            {
                "distributed_schedules": {
                    value.scenario: value.distributed_schedule_sha256 for value in result.plans
                }
            },
        ),
        (
            RunState.LOWERED,
            {
                "physical_schedules": {
                    value.scenario: value.physical_schedule_sha256 for value in result.plans
                },
                "pallas_sources": {
                    value.scenario: value.pallas_source_sha256 for value in result.plans
                },
            },
        ),
        (
            RunState.COMPILED,
            {"plans": [value.model_dump(mode="json") for value in result.plans]},
        ),
        (
            RunState.CORRECT,
            {
                "observations": len(result.observations),
                "pallas_outputs": [value.pallas_output_sha256 for value in result.observations],
                "control_outputs": [value.control_output_sha256 for value in result.observations],
            },
        ),
        (
            RunState.VALIDATED,
            {
                "discriminators": [
                    {
                        "name": value.discriminator.value,
                        "artifacts": list(value.artifact_sha256),
                    }
                    for value in result.discriminators
                ],
                "result_sha256": result_sha256,
            },
        ),
    )
    if require_accepted:
        payloads += (
            (
                RunState.ACCEPTED,
                {"result_sha256": result_sha256, "passed": True},
            ),
        )
    return payloads


def _validate(
    root: Path,
    trusted_contract: SeqaxBf16ValidationContract,
    *,
    require_accepted: bool,
    require_receipt: bool | None = None,
) -> SeqaxBf16ValidationResult:
    if require_receipt is None:
        require_receipt = require_accepted
    if require_receipt and not require_accepted:
        raise ValueError("SEQAX_BF16_RECEIPT_REQUIRES_ACCEPTED_LEDGER")
    _validate_closed_world(
        root,
        _expected_files(root, trusted_contract, receipt_present=require_receipt),
    )
    saved_contract = SeqaxBf16ValidationContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != trusted_contract:
        raise ValueError("SEQAX_BF16_CONTRACT_MISMATCH")
    result = SeqaxBf16ValidationResult.model_validate_json((root / "result.json").read_text())
    source_commit = _validate_source(root, trusted_contract, result)
    expected_run_id = semantic_sha256(
        SEQAX_BF16_RUN_SCHEMA,
        trusted_contract.contract_id,
        source_commit,
    )
    identity = SeqaxBf16RunIdentity.model_validate_json((root / "run_identity.json").read_text())
    if identity != SeqaxBf16RunIdentity(
        schema_version=SEQAX_BF16_RUN_SCHEMA,
        contract_id=trusted_contract.contract_id,
        run_id=expected_run_id,
        source_commit=source_commit,
    ):
        raise ValueError("SEQAX_BF16_RUN_IDENTITY_MISMATCH")
    expected_runtime = trusted_contract.runtime
    if (
        result.schema_version != SEQAX_BF16_RUN_SCHEMA
        or result.contract_id != trusted_contract.contract_id
        or result.run_id != expected_run_id
        or not result.runtime.runtime.python.startswith(expected_runtime.python_major_minor + ".")
        or result.runtime.runtime.jax != expected_runtime.jax
        or result.runtime.runtime.jaxlib != expected_runtime.jaxlib
        or result.runtime.runtime.libtpu != expected_runtime.libtpu
        or result.runtime.runtime.xla != expected_runtime.libtpu_init_args
        or result.runtime.ml_dtypes != expected_runtime.ml_dtypes
        or result.runtime.cpu_machine != expected_runtime.cpu_machine
        or result.runtime.cpu_system != expected_runtime.cpu_system
        or not result.passed
        or result.claim_scope != "declared-surface-dual-jax-cpu-bf16-numerical-agreement-v1"
    ):
        raise ValueError("SEQAX_BF16_RESULT_IDENTITY_MISMATCH")
    if (
        tuple(value.id for value in result.devices) != tuple(range(trusted_contract.device_count))
        or len({value.process_index for value in result.devices}) != 1
        or any(value.platform != trusted_contract.backend for value in result.devices)
        or any(
            value.device_kind not in {trusted_contract.device_kind, "TPU v7x"}
            for value in result.devices
        )
    ):
        raise ValueError("SEQAX_BF16_DEVICE_INVENTORY_MISMATCH")
    if tuple(value.scenario for value in result.plans) != tuple(
        scenario.name for scenario in trusted_contract.scenarios
    ):
        raise ValueError("SEQAX_BF16_PLAN_ORDER_MISMATCH")
    for scenario, record in zip(trusted_contract.scenarios, result.plans, strict=True):
        _validate_plan(root, scenario, record)
    observations = tuple(
        _validate_seed(root, trusted_contract, scenario, seed)
        for scenario in trusted_contract.scenarios
        for seed in scenario.seeds
    )
    if result.observations != observations:
        raise ValueError("SEQAX_BF16_RESULT_OBSERVATIONS_MISMATCH")
    discriminators = _validate_discriminators(root, trusted_contract)
    if result.discriminators != discriminators:
        raise ValueError("SEQAX_BF16_RESULT_DISCRIMINATORS_MISMATCH")

    result_sha256 = _sha256(root / "result.json")
    payloads = _ledger_payloads(
        trusted_contract,
        result,
        source_commit,
        result_sha256,
        require_accepted=require_accepted,
    )
    history = read_ledger_history(root / "ledger.sqlite", result.run_id)
    if tuple(value.state for value in history) != tuple(state for state, _payload in payloads):
        raise ValueError("SEQAX_BF16_LEDGER_STATE_MISMATCH")
    if tuple(value.payload_sha256 for value in history) != tuple(
        ExperimentLedger.payload_sha256(payload) for _state, payload in payloads
    ):
        raise ValueError("SEQAX_BF16_LEDGER_PAYLOAD_MISMATCH")
    if require_receipt:
        receipt = SeqaxBf16ValidationReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        if (
            receipt.contract_id != trusted_contract.contract_id
            or receipt.run_id != result.run_id
            or receipt.result_sha256 != result_sha256
            or receipt.ledger_sha256 != _sha256(root / "ledger.sqlite")
            or receipt.artifacts != _artifact_manifest(root)
        ):
            raise ValueError("SEQAX_BF16_RECEIPT_MISMATCH")
    return result


def validate_seqax_bf16_validation(
    root: Path,
    trusted_contract: SeqaxBf16ValidationContract,
) -> SeqaxBf16ValidationResult:
    canonical = default_seqax_bf16_validation_contract()
    if trusted_contract != canonical:
        raise ValueError("SEQAX_BF16_EXTERNAL_CONTRACT_MISMATCH")
    if trusted_contract.hlo_identity_status != "pinned":
        raise ValueError("SEQAX_BF16_HLO_IDENTITIES_PENDING")
    if root.is_symlink():
        raise ValueError(f"SEQAX_BF16_ROOT_INVALID path={root}")
    root = root.resolve()
    _preflight_existing_root(root)
    return _validate(root, trusted_contract, require_accepted=True)
