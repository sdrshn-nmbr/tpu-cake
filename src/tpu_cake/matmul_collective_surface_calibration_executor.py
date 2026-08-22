from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import model_identity_sha256
from tpu_cake.ledger import EvidenceRun, RunState, payload_sha256
from tpu_cake.matmul_collective_surface_calibration_archive import (
    copy_parent_archive,
    validate_and_extract_parent_archive,
)
from tpu_cake.matmul_collective_surface_calibration_evidence import (
    MatmulCollectiveSurfaceCalibrationEvidence,
    validate_surface_calibration_evidence,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    MatmulCollectiveSurfaceCalibrationProtocol,
    load_matmul_collective_surface_calibration_protocol,
)
from tpu_cake.matmul_collective_surface_calibration_seal import (
    MatmulCollectiveSurfaceCalibrationSealedEvidence,
    derive_surface_calibration_seal,
    validate_surface_calibration_seal,
)
from tpu_cake.matmul_collective_surface_calibration_worker import (
    CALIBRATION_EXECUTABLE_DEPENDENCIES,
    CALIBRATION_EXECUTOR_SOURCE_PATH,
    CALIBRATION_VERIFIER_SOURCE_PATH,
    CALIBRATION_WORKER_SOURCE_PATH,
    SurfaceCalibrationAttemptClaim,
    SurfaceCalibrationExecutionAuthority,
    SurfaceCalibrationRunIdentity,
    SurfaceCalibrationWorkerRequest,
    SurfaceCalibrationWorkerResult,
    _file_sha256,
    _json_bytes,
    _write_bytes_exclusive,
    _write_model_exclusive,
    validate_calibration_execution_authority,
)
from tpu_cake.matmul_collective_surface_correctness_evidence import (
    MatmulCollectiveSurfaceCorrectnessEvidence,
    SurfaceCompileContinuityEvidence,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    default_matmul_collective_surface_design_contract,
)
from tpu_cake.matmul_collective_surface_runner import (
    SurfacePhase,
    SurfacePhaseLedger,
    _source_subprocess_environment,
    record_surface_phase,
)

CALIBRATION_MANIFEST_SCHEMA = "matmul-collective-surface-calibration-manifest-v1"
CALIBRATION_RECEIPT_SCHEMA = "matmul-collective-surface-calibration-receipt-v1"


class SurfaceCalibrationManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def path_is_canonical(self) -> SurfaceCalibrationManifestEntry:
        parts = self.path.split("/")
        if self.path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_MANIFEST_PATH_INVALID")
        return self


class SurfaceCalibrationPhaseReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-receipt-v1"] = (
        CALIBRATION_RECEIPT_SCHEMA
    )
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_claim_path: str
    attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent_receipt_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_seal_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_ledger_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_sequence: Literal[4] = 4
    previous_phase_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @computed_field
    @property
    def receipt_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCalibrationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-manifest-v1"] = (
        CALIBRATION_MANIFEST_SCHEMA
    )
    identity: SurfaceCalibrationRunIdentity
    evidence_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_seal_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_ledger_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[SurfaceCalibrationManifestEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def artifact_inventory_is_canonical(self) -> SurfaceCalibrationManifest:
        paths = tuple(value.path for value in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_MANIFEST_ORDER_INVALID")
        return self


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_exclusive_durable(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(mode=mode, exist_ok=False)
    _fsync_directory(path.parent)


def _reject_links_in_path(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.absolute().parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PATH_SYMLINK path={current}")


def _require_safe_new_root(root: Path) -> None:
    if not root.is_absolute():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROOT_NOT_ABSOLUTE")
    resolved = root.resolve(strict=False)
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/"), Path.home().resolve(), repository_root)
    if (
        any(resolved == value or resolved in value.parents for value in protected)
        or repository_root in resolved.parents
    ):
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_UNSAFE_ROOT path={resolved}")
    _reject_links_in_path(resolved.parent)
    if resolved.exists() or resolved.is_symlink():
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ATTEMPT_EXISTS path={resolved}")
    if not resolved.parent.is_dir():
        raise ValueError(
            f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_INVALID path={resolved.parent}"
        )


def _preflight_existing_root(root: Path) -> None:
    status = root.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid() or status.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROOT_AUTHORITY_INVALID")
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARTIFACT_SYMLINK")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARTIFACT_HARDLINK")


def _canonical_design(path: Path) -> MatmulCollectiveSurfaceDesignContract:
    design = MatmulCollectiveSurfaceDesignContract.model_validate_json(path.read_text())
    if design != default_matmul_collective_surface_design_contract():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_NONCANONICAL_DESIGN")
    return design


def _canonical_protocol(
    path: Path,
    design: MatmulCollectiveSurfaceDesignContract,
) -> MatmulCollectiveSurfaceCalibrationProtocol:
    return load_matmul_collective_surface_calibration_protocol(path, design)


def _canonical_subprocess_environment(
    design: MatmulCollectiveSurfaceDesignContract,
    *,
    compilation_cache_path: Path | None = None,
) -> dict[str, str]:
    environment = _source_subprocess_environment()
    environment.update(design.compiler_environment)
    environment["PYTHONSAFEPATH"] = "1"
    if compilation_cache_path is not None:
        environment["JAX_COMPILATION_CACHE_DIR"] = str(compilation_cache_path)
    return environment


def _read_committed_source_blobs(
    repository_root: Path,
    source_commit: str,
) -> dict[str, bytes]:
    blobs = {}
    for path in (*CALIBRATION_EXECUTABLE_DEPENDENCIES, "uv.lock"):
        git_path = f"src/{path}" if path.startswith("tpu_cake/") else path
        try:
            blobs[path] = subprocess.run(
                ["/usr/bin/git", "show", f"{source_commit}:{git_path}"],
                cwd=repository_root,
                env=_source_subprocess_environment(),
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_BLOB_UNAVAILABLE path={path}"
            ) from error
    return blobs


def _probe_execution_authority(
    protocol_path: Path,
    design_path: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCalibrationExecutionAuthority, dict[str, bytes]]:
    with tempfile.TemporaryDirectory(prefix="tpu-cake-calibration-authority-") as directory:
        output = Path(directory) / "authority.json"
        subprocess.run(
            [
                sys.executable,
                "-P",
                "-m",
                "tpu_cake.matmul_collective_surface_calibration_worker",
                "--probe-output",
                str(output),
                "--protocol",
                str(protocol_path),
                "--design",
                str(design_path),
            ],
            cwd=Path(design.compilation_source_root),
            env=_canonical_subprocess_environment(design),
            check=True,
        )
        authority = SurfaceCalibrationExecutionAuthority.model_validate_json(output.read_text())
    source_blobs = _read_committed_source_blobs(
        Path(design.compilation_source_root), authority.source.source_commit
    )
    validate_calibration_execution_authority(authority, protocol, design, source_blobs)
    running = Path(__file__)
    expected = source_blobs[CALIBRATION_EXECUTOR_SOURCE_PATH.removeprefix("src/")]
    if (
        running.is_symlink()
        or running.resolve()
        != (Path(design.compilation_source_root) / CALIBRATION_EXECUTOR_SOURCE_PATH).resolve()
        or running.read_bytes() != expected
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_RUNNING_EXECUTOR_MISMATCH")
    return authority, source_blobs


def _write_source_bundle(
    root: Path,
    authority: SurfaceCalibrationExecutionAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    for path, blob in source_blobs.items():
        _write_bytes_exclusive(root / "source" / "committed" / path, blob)
    aliases = {
        "executor.py": CALIBRATION_EXECUTOR_SOURCE_PATH,
        "worker.py": CALIBRATION_WORKER_SOURCE_PATH,
        "verifier.py": CALIBRATION_VERIFIER_SOURCE_PATH,
        "archive.py": "src/tpu_cake/matmul_collective_surface_calibration_archive.py",
        "evidence.py": "src/tpu_cake/matmul_collective_surface_calibration_evidence.py",
        "protocol.py": "src/tpu_cake/matmul_collective_surface_calibration_protocol.py",
        "seal.py": "src/tpu_cake/matmul_collective_surface_calibration_seal.py",
    }
    for destination, source in aliases.items():
        _write_bytes_exclusive(
            root / "source" / destination,
            source_blobs[source.removeprefix("src/")],
        )
    if authority.source.authority_sha256 != model_identity_sha256(authority.source):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_BUNDLE_INVALID")


def _parent_root(
    root: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> Path:
    return root / "parent" / protocol.correctness_parent.archive_root_name


def _stage_and_verify_parent(
    root: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    *,
    zstd_path: Path,
) -> Path:
    parent = protocol.correctness_parent
    parent_directory = root / "parent"
    _mkdir_exclusive_durable(parent_directory)
    staged = parent_directory / parent.archive_filename
    copy_parent_archive(
        Path(parent.archive_path),
        staged,
        expected_sha256=parent.archive_sha256,
        expected_size_bytes=parent.archive_size_bytes,
    )
    extraction_directory = root / "parent-extraction"
    validate_and_extract_parent_archive(
        staged,
        extraction_directory,
        expected_root_name=parent.archive_root_name,
        maximum_members=parent.archive_maximum_members,
        maximum_member_size_bytes=parent.archive_maximum_member_size_bytes,
        maximum_total_size_bytes=parent.archive_maximum_total_size_bytes,
        zstd_path=zstd_path,
    )
    extracted = _parent_root(root, protocol)
    os.rename(extraction_directory / parent.archive_root_name, extracted)
    extraction_directory.rmdir()
    _fsync_directory(parent_directory)
    _fsync_directory(root)
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            str(extracted / "source/verifier.py"),
            "--root",
            str(extracted),
            "--protocol",
            str(
                extracted
                / "source/committed/contracts/matmul-collective-surface-correctness-v1.json"
            ),
            "--design",
            str(extracted / "source/committed/contracts/matmul-collective-surface-design-v1.json"),
        ],
        cwd="/",
        env={
            "HOME": "/nonexistent",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    replay = json.loads(completed.stdout)
    expected = {
        "attempt_id": parent.attempt_id,
        "protocol_id": parent.protocol_id,
        "source_authority_sha256": parent.source_authority_sha256,
        "execution_authority_sha256": parent.execution_authority_sha256,
        "evidence_sha256": parent.evidence_sha256,
        "ledger_sha256": parent.ledger_file_sha256,
        "phase_ledger_sha256": parent.phase_ledger_sha256,
        "receipt_sha256": parent.receipt_sha256,
        "case_count": parent.case_count,
        "execution_count": parent.execution_count,
        "split": parent.split,
    }
    if replay != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_REPLAY_MISMATCH")
    return extracted


def _claim_attempt(
    root: Path,
    attempt_id: str,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    source_commit: str,
) -> tuple[Path, SurfaceCalibrationAttemptClaim]:
    registry = Path(protocol.attempt_registry_root)
    _reject_links_in_path(registry.parent)
    if not registry.exists():
        _mkdir_exclusive_durable(registry)
    status = registry.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid() or status.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ATTEMPT_REGISTRY_INVALID")
    claim = SurfaceCalibrationAttemptClaim(
        attempt_id=attempt_id,
        protocol_id=protocol.protocol_id,
        permanent_claim_key=protocol.permanent_claim_key,
        correctness_parent_receipt_sha256=protocol.correctness_parent.receipt_sha256,
        source_commit=source_commit,
        output_root=str(root),
    )
    path = registry / f"{protocol.permanent_claim_key}.json"
    try:
        _write_model_exclusive(path, claim)
    except FileExistsError as error:
        raise ValueError(
            "MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ATTEMPT_PERMANENTLY_CLAIMED"
        ) from error
    return path, claim


@contextmanager
def _exclusive_claim_lock(claim_key: str) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-calibration-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    status = lock_root.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid() or status.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_LOCK_ROOT_INVALID")
    descriptor = os.open(
        lock_root / f"{claim_key}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ATTEMPT_LOCKED") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _launch_worker(
    root: Path,
    request_path: Path,
    design: MatmulCollectiveSurfaceDesignContract,
    cache_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "tpu_cake.matmul_collective_surface_calibration_worker",
            "--root",
            str(root),
            "--request",
            str(request_path),
        ],
        cwd=Path(design.compilation_source_root),
        env=_canonical_subprocess_environment(design, compilation_cache_path=cache_path),
        check=True,
    )


def _validate_worker_result(
    root: Path,
    request: SurfaceCalibrationWorkerRequest,
    authority: SurfaceCalibrationExecutionAuthority,
) -> SurfaceCalibrationWorkerResult:
    result = SurfaceCalibrationWorkerResult.model_validate_json(
        (root / "worker-result.json").read_text()
    )
    if (
        result.attempt_id != request.attempt_id
        or result.invocation_nonce != request.invocation_nonce
        or result.execution_authority_sha256 != authority.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_WORKER_RESULT_MISMATCH")
    validate_surface_calibration_evidence(
        result.evidence,
        request.protocol,
        request.design,
        expected_protocol_file_sha256=authority.protocol_file_sha256,
        expected_design_file_sha256=authority.design_file_sha256,
        expected_execution_authority_sha256=authority.authority_sha256,
        expected_invocation_nonce=request.invocation_nonce,
        expected_worker_pid=result.worker_pid,
    )
    return result


def _extend_phase_ledger(
    parent_root: Path,
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    seal: MatmulCollectiveSurfaceCalibrationSealedEvidence,
) -> SurfacePhaseLedger:
    ledger = SurfacePhaseLedger.model_validate_json((parent_root / "phase_ledger.json").read_text())
    ledger = record_surface_phase(ledger, SurfacePhase.CALIBRATION, evidence.evidence_sha256)
    return record_surface_phase(ledger, SurfacePhase.CALIBRATION_SEALED, seal.seal_sha256)


def _manifest_entries(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset({"manifest.json"}),
) -> tuple[SurfaceCalibrationManifestEntry, ...]:
    _preflight_existing_root(root)
    paths = tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.relative_to(root).as_posix() not in excluded
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    return tuple(
        SurfaceCalibrationManifestEntry(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
        )
        for path in paths
    )


def _artifact_set_sha256(root: Path) -> str:
    entries = _manifest_entries(
        root,
        excluded=frozenset({"ledger.sqlite", "receipt.json", "manifest.json"}),
    )
    return payload_sha256({value.path: [value.size_bytes, value.sha256] for value in entries})


def _continuity_payloads(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    return _schedule_payload(evidence.continuity), {
        f"{value.scenario_name}:{value.strategy.value}": value.compile_record_sha256
        for value in evidence.continuity
    }


def _schedule_payload(
    continuity: tuple[SurfaceCompileContinuityEvidence, ...],
) -> dict[str, list[str]]:
    return {
        f"{value.scenario_name}:{value.strategy.value}": [
            value.observed_distributed_schedule_sha256,
            value.observed_physical_schedule_sha256,
            value.observed_pallas_source_sha256,
        ]
        for value in continuity
    }


def _parent_schedule_payload(
    parent_root: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> dict[str, list[str]]:
    parent = MatmulCollectiveSurfaceCorrectnessEvidence.model_validate_json(
        (parent_root / "evidence.json").read_text()
    )
    continuity = tuple(
        value for value in parent.continuity if value.scenario_name in protocol.scenarios
    )
    expected = tuple(
        (scenario, strategy) for scenario in protocol.scenarios for strategy in protocol.strategies
    )
    if tuple((value.scenario_name, value.strategy) for value in continuity) != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_CONTINUITY_MISMATCH")
    return _schedule_payload(continuity)


def _run_archived_independent_verifier(
    root: Path,
    protocol_path: Path,
    design_path: Path,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            str(root / "source/verifier.py"),
            "--root",
            str(root),
            "--protocol",
            str(protocol_path),
            "--design",
            str(design_path),
        ],
        cwd="/",
        env={
            "HOME": "/nonexistent",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_REPLAY_INVALID")
    return value


def execute_surface_calibration(
    root: Path,
    protocol_path: Path,
    design_path: Path,
    attempt_id: str,
    *,
    zstd_path: Path = Path("/usr/bin/zstd"),
) -> SurfaceCalibrationManifest:
    design = _canonical_design(design_path)
    protocol = _canonical_protocol(protocol_path, design)
    root = root.resolve(strict=False)
    if len(attempt_id) != 64 or any(value not in "0123456789abcdef" for value in attempt_id):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ATTEMPT_ID_INVALID")
    with _exclusive_claim_lock(protocol.permanent_claim_key):
        _require_safe_new_root(root)
        authority, source_blobs = _probe_execution_authority(
            protocol_path, design_path, protocol, design
        )
        _mkdir_exclusive_durable(root)
        try:
            _write_model_exclusive(root / "protocol.json", protocol)
            _write_model_exclusive(root / "design.json", design)
            _write_model_exclusive(root / "execution_authority.json", authority)
            _write_source_bundle(root, authority, source_blobs)
            parent_root = _stage_and_verify_parent(root, protocol, zstd_path=zstd_path)
            claim_path, claim = _claim_attempt(
                root, attempt_id, protocol, authority.source.source_commit
            )
            claim_sha256 = _file_sha256(claim_path)
            _write_model_exclusive(root / "attempt_claim.json", claim)
            with tempfile.TemporaryDirectory(
                prefix="tpu-cake-calibration-cache-"
            ) as cache_directory:
                cache_path = Path(cache_directory).resolve(strict=True)
                identity = SurfaceCalibrationRunIdentity(
                    attempt_id=attempt_id,
                    protocol_id=protocol.protocol_id,
                    execution_authority_sha256=authority.authority_sha256,
                    source_authority_sha256=authority.source.authority_sha256,
                    attempt_claim_path=str(claim_path),
                    attempt_claim_sha256=claim_sha256,
                    output_root=str(root),
                    parent_correctness_root=str(parent_root),
                    compilation_cache_path=str(cache_path),
                )
                _write_model_exclusive(root / "run_identity.json", identity)
                request = SurfaceCalibrationWorkerRequest(
                    attempt_id=attempt_id,
                    invocation_nonce=hashlib.sha256(os.urandom(32)).hexdigest(),
                    output_root=str(root),
                    parent_correctness_root=str(parent_root),
                    compilation_cache_path=str(cache_path),
                    protocol_file_sha256=authority.protocol_file_sha256,
                    design_file_sha256=authority.design_file_sha256,
                    execution_authority_sha256=authority.authority_sha256,
                    source_commit=authority.source.source_commit,
                    source_authority_sha256=authority.source.authority_sha256,
                    protocol=protocol,
                    design=design,
                )
                _write_model_exclusive(root / "worker-request.json", request)
                _write_bytes_exclusive(
                    root / "STARTED.json",
                    _json_bytes(
                        {
                            "attempt_id": attempt_id,
                            "invocation_nonce": request.invocation_nonce,
                            "protocol_id": protocol.protocol_id,
                            "state": "started",
                        }
                    ),
                )
                run = EvidenceRun(root / "ledger.sqlite", attempt_id)
                run.create(
                    {
                        "protocol_id": protocol.protocol_id,
                        "execution_authority_sha256": authority.authority_sha256,
                        "attempt_claim_path": str(claim_path),
                        "attempt_claim_sha256": claim_sha256,
                    }
                )
                run.transition(
                    RunState.VERIFIED,
                    {
                        "source_authority_sha256": authority.source.authority_sha256,
                        "parent_receipt_sha256": protocol.correctness_parent.receipt_sha256,
                        "parent_archive_sha256": protocol.correctness_parent.archive_sha256,
                        "devices": [value.model_dump(mode="json") for value in authority.devices],
                    },
                )
                parent_schedules = _parent_schedule_payload(parent_root, protocol)
                run.transition(
                    RunState.LOWERED,
                    {"continuity_schedule_set_sha256": payload_sha256(parent_schedules)},
                )
                _launch_worker(root, root / "worker-request.json", design, cache_path)
                result = _validate_worker_result(root, request, authority)
            evidence = result.evidence
            _write_model_exclusive(root / "evidence.json", evidence)
            seal = derive_surface_calibration_seal(evidence, protocol, design)
            validate_surface_calibration_seal(seal, evidence, protocol, design)
            _write_model_exclusive(root / "calibration-seal.json", seal)
            phase_ledger = _extend_phase_ledger(parent_root, evidence, seal)
            _write_model_exclusive(root / "phase_ledger.json", phase_ledger)
            schedules, compiled = _continuity_payloads(evidence)
            if schedules != parent_schedules:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_FRESH_LOWERING_MISMATCH")
            run.transition(
                RunState.COMPILED,
                {"fresh_compile_set_sha256": payload_sha256(compiled)},
            )
            run.transition(RunState.CORRECT, {"evidence_sha256": evidence.evidence_sha256})
            artifact_set_sha256 = _artifact_set_sha256(root)
            run.transition(
                RunState.VALIDATED,
                {
                    "calibration_seal_sha256": seal.seal_sha256,
                    "phase_ledger_sha256": model_identity_sha256(phase_ledger),
                    "artifact_set_sha256": artifact_set_sha256,
                },
            )
            run.transition(
                RunState.ACCEPTED,
                {
                    "producer_validation": "schema-fit-bootstrap-and-artifact-replay-v1",
                    "holdout_authorization": seal.holdout_authorization,
                },
            )
            run.seal("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_LEDGER_SIDECARS")
            receipt = SurfaceCalibrationPhaseReceipt(
                attempt_id=attempt_id,
                protocol_id=protocol.protocol_id,
                attempt_claim_path=str(claim_path),
                attempt_claim_sha256=claim_sha256,
                correctness_parent_receipt_file_sha256=(
                    protocol.correctness_parent.receipt_file_sha256
                ),
                correctness_parent_receipt_sha256=protocol.correctness_parent.receipt_sha256,
                evidence_file_sha256=_file_sha256(root / "evidence.json"),
                evidence_sha256=evidence.evidence_sha256,
                calibration_seal_file_sha256=_file_sha256(root / "calibration-seal.json"),
                calibration_seal_sha256=seal.seal_sha256,
                ledger_snapshot_sha256=_file_sha256(root / "ledger.sqlite"),
                phase_ledger_file_sha256=_file_sha256(root / "phase_ledger.json"),
                phase_ledger_sha256=model_identity_sha256(phase_ledger),
                previous_phase_receipt_sha256=protocol.correctness_parent.receipt_sha256,
                artifact_set_sha256=artifact_set_sha256,
            )
            _write_model_exclusive(root / "receipt.json", receipt)
            manifest = SurfaceCalibrationManifest(
                identity=identity,
                evidence_file_sha256=_file_sha256(root / "evidence.json"),
                evidence_sha256=evidence.evidence_sha256,
                calibration_seal_file_sha256=_file_sha256(root / "calibration-seal.json"),
                calibration_seal_sha256=seal.seal_sha256,
                ledger_snapshot_sha256=_file_sha256(root / "ledger.sqlite"),
                phase_ledger_file_sha256=_file_sha256(root / "phase_ledger.json"),
                phase_ledger_sha256=model_identity_sha256(phase_ledger),
                receipt_file_sha256=_file_sha256(root / "receipt.json"),
                receipt_sha256=receipt.receipt_sha256,
                artifacts=_manifest_entries(root),
            )
            _write_model_exclusive(root / "manifest.json", manifest)
            replay = _run_archived_independent_verifier(
                root,
                root / "source/committed/contracts/matmul-collective-surface-calibration-v1.json",
                root / "source/committed/contracts/matmul-collective-surface-design-v1.json",
            )
            expected_replay = {
                "attempt_id": attempt_id,
                "protocol_id": protocol.protocol_id,
                "source_authority_sha256": authority.source.authority_sha256,
                "execution_authority_sha256": authority.authority_sha256,
                "correctness_parent_receipt_sha256": (protocol.correctness_parent.receipt_sha256),
                "evidence_sha256": evidence.evidence_sha256,
                "seal_sha256": seal.seal_sha256,
                "ledger_sha256": _file_sha256(root / "ledger.sqlite"),
                "phase_ledger_sha256": model_identity_sha256(phase_ledger),
                "receipt_sha256": receipt.receipt_sha256,
                "sample_count": len(evidence.samples),
                "holdout_authorization": seal.holdout_authorization,
            }
            if replay != expected_replay:
                raise ValueError(
                    "MATMUL_COLLECTIVE_SURFACE_CALIBRATION_INDEPENDENT_REPLAY_MISMATCH"
                )
            return manifest
        except Exception as error:
            failure = root / "failure.json"
            if not failure.exists():
                _write_bytes_exclusive(
                    failure,
                    _json_bytes({"error_type": type(error).__name__, "error": str(error)}),
                )
            raise


def main() -> None:
    if not sys.flags.safe_path:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SAFE_PATH_REQUIRED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--zstd", type=Path, default=Path("/usr/bin/zstd"))
    args = parser.parse_args()
    manifest = execute_surface_calibration(
        args.root,
        args.protocol,
        args.design,
        args.attempt_id,
        zstd_path=args.zstd,
    )
    print(
        json.dumps(
            {
                "attempt_id": manifest.identity.attempt_id,
                "protocol_id": manifest.identity.protocol_id,
                "evidence_sha256": manifest.evidence_sha256,
                "calibration_seal_sha256": manifest.calibration_seal_sha256,
                "receipt_sha256": manifest.receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
