from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256
from tpu_cake.ledger import EvidenceRun, RunState, payload_sha256
from tpu_cake.matmul_collective_surface_correctness_evidence import (
    MatmulCollectiveSurfaceCorrectnessEvidence,
    validate_surface_correctness_evidence,
)
from tpu_cake.matmul_collective_surface_correctness_protocol import (
    MatmulCollectiveSurfaceCorrectnessProtocol,
    default_matmul_collective_surface_correctness_protocol,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceSplit,
    default_matmul_collective_surface_design_contract,
)
from tpu_cake.matmul_collective_surface_runner import (
    SURFACE_EXECUTABLE_DEPENDENCIES,
    SurfacePhase,
    SurfacePhaseLedger,
    _source_subprocess_environment,
    record_surface_phase,
)
from tpu_cake.runner import _runtime_identity

CORRECTNESS_EXECUTOR_SCHEMA = "matmul-collective-surface-correctness-executor-v1"
CORRECTNESS_MANIFEST_SCHEMA = "matmul-collective-surface-correctness-manifest-v1"
CORRECTNESS_RECEIPT_SCHEMA = "matmul-collective-surface-correctness-receipt-v1"
CORRECTNESS_EXECUTOR_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_correctness_executor.py"
CORRECTNESS_WORKER_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_correctness_worker.py"
CORRECTNESS_VERIFIER_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_correctness_verifier.py"
CORRECTNESS_GENERATOR_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_correctness.py"
CORRECTNESS_ORACLE_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_correctness_oracle.py"
CORRECTNESS_PROTOCOL_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_correctness_protocol.py"
CORRECTNESS_EVIDENCE_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_correctness_evidence.py"
CORRECTNESS_EXECUTABLE_DEPENDENCIES = tuple(
    sorted(
        {
            *SURFACE_EXECUTABLE_DEPENDENCIES,
            "contracts/matmul-collective-surface-correctness-v1.json",
            "contracts/matmul-collective-surface-design-v1.json",
            CORRECTNESS_EXECUTOR_SOURCE_PATH.removeprefix("src/"),
            CORRECTNESS_WORKER_SOURCE_PATH.removeprefix("src/"),
            CORRECTNESS_VERIFIER_SOURCE_PATH.removeprefix("src/"),
            CORRECTNESS_GENERATOR_SOURCE_PATH.removeprefix("src/"),
            CORRECTNESS_ORACLE_SOURCE_PATH.removeprefix("src/"),
            CORRECTNESS_PROTOCOL_SOURCE_PATH.removeprefix("src/"),
            CORRECTNESS_EVIDENCE_SOURCE_PATH.removeprefix("src/"),
        }
    )
)
_FINAL_STATES = (
    RunState.CREATED,
    RunState.VERIFIED,
    RunState.LOWERED,
    RunState.COMPILED,
    RunState.CORRECT,
    RunState.VALIDATED,
    RunState.ACCEPTED,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_model_exclusive(path: Path, value: BaseModel) -> None:
    _write_bytes_exclusive(
        path,
        _json_bytes(value.model_dump(mode="json", exclude_computed_fields=True)),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_exclusive_durable(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(mode=mode, exist_ok=False)
    _fsync_directory(path.parent)


def _reject_archive_links(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_ROOT_INVALID")
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_LINK_INVALID")


def _verify_parent_compile_snapshot(
    root: Path,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    _reject_archive_links(root)
    parent = protocol.parent_compile
    if (
        _file_sha256(root / "manifest.json") != parent.manifest_file_sha256
        or _file_sha256(root / "compile_report.json") != parent.compile_report_file_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_FILE_HASH_MISMATCH")
    verifier = root / "source/verifier.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--root",
            str(root),
            "--contract",
            str(root / "contract.json"),
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
    try:
        verification = json.loads(completed.stdout)
        authority = json.loads((root / "execution_authority.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
    except (json.JSONDecodeError, KeyError) as error:
        raise ValueError(
            "MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_VERIFICATION_INVALID"
        ) from error
    expected = {
        "attempt_id": parent.attempt_id,
        "design_id": parent.design_id,
        "source_authority_sha256": parent.source_authority_sha256,
        "execution_authority_sha256": parent.execution_authority_sha256,
        "compile_report_sha256": parent.compile_report_sha256,
        "ledger_sha256": parent.compile_ledger_sha256,
    }
    if (
        any(verification.get(key) != value for key, value in expected.items())
        or authority.get("source", {}).get("source_commit") != parent.source_commit
        or manifest.get("identity", {}).get("attempt_id") != parent.attempt_id
        or manifest.get("report_sha256") != parent.compile_report_sha256
        or manifest.get("ledger_sha256") != parent.compile_ledger_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_AUTHORITY_MISMATCH")
    return verification


def _copy_parent_compile_snapshot(
    destination_root: Path,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
) -> Path:
    source = Path(protocol.parent_compile.archive_path)
    _verify_parent_compile_snapshot(source, protocol)
    destination = destination_root / "parent_compile"
    shutil.copytree(source, destination, symlinks=False)
    _verify_parent_compile_snapshot(destination, protocol)
    manifest = json.loads((destination / "manifest.json").read_text())
    identity = manifest["identity"]
    claim_source = Path(identity["attempt_claim_path"])
    if (
        claim_source.is_symlink()
        or not claim_source.is_file()
        or claim_source.stat().st_nlink != 1
        or _file_sha256(claim_source) != identity["attempt_claim_sha256"]
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_CLAIM_INVALID")
    claim_destination = destination_root / "parent_compile_claim.json"
    _write_bytes_exclusive(claim_destination, claim_source.read_bytes())
    return destination


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_METADATA_REDIRECT code={code}")


def _metadata(path: str) -> str:
    request = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
    ).open(request, timeout=5) as response:
        if response.headers.get("Metadata-Flavor") != "Google":
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_METADATA_HEADER_MISSING")
        payload = response.read(4097)
    if len(payload) > 4096:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_METADATA_TOO_LARGE")
    return payload.decode().strip()


def _compiler_environment(design: MatmulCollectiveSurfaceDesignContract) -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in design.compiler_environment}
    if observed != design.compiler_environment:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_DECLARED_ENVIRONMENT_MISMATCH")
    forbidden = {
        key: value
        for key, value in os.environ.items()
        if (key == "TPU_LIBRARY_PATH" or key.startswith(("JAX_", "XLA_", "PJRT_", "LIBTPU_")))
        and key not in design.compiler_environment
        and key != "JAX_COMPILATION_CACHE_DIR"
    }
    if forbidden:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_UNDECLARED_ENVIRONMENT")
    return dict(design.compiler_environment)


def _canonical_subprocess_environment(
    design: MatmulCollectiveSurfaceDesignContract,
    *,
    compilation_cache_dir: Path | None = None,
) -> dict[str, str]:
    environment = _source_subprocess_environment()
    environment.update(design.compiler_environment)
    if compilation_cache_dir is not None:
        environment["JAX_COMPILATION_CACHE_DIR"] = str(compilation_cache_dir)
    return environment


class SurfaceCorrectnessSourceAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch: Literal["main"] = "main"
    origin_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_url: Literal["https://github.com/sdrshn-nmbr/tpu-cake.git"] = (
        "https://github.com/sdrshn-nmbr/tpu-cake.git"
    )
    source_root: Literal["/home/sudarshan/tpu-cake-main"] = "/home/sudarshan/tpu-cake-main"
    runtime: dict[str, str]
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies: tuple[SourceFileContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def main_is_exact(self) -> SurfaceCorrectnessSourceAuthority:
        if not (self.source_commit == self.origin_main_commit == self.remote_main_commit):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_MAIN_MISMATCH")
        paths = tuple(value.path for value in self.dependencies)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_INVENTORY_INVALID")
        return self

    @computed_field
    @property
    def authority_sha256(self) -> str:
        return model_identity_sha256(self)


def _git_blob_path(path: str) -> str:
    return f"src/{path}" if path.startswith("tpu_cake/") else path


def _read_committed_correctness_blobs(
    repository_root: Path,
    source_commit: str,
) -> dict[str, bytes]:
    values = {}
    for path in (*CORRECTNESS_EXECUTABLE_DEPENDENCIES, "uv.lock"):
        try:
            values[path] = subprocess.run(
                ["/usr/bin/git", "show", f"{source_commit}:{_git_blob_path(path)}"],
                cwd=repository_root,
                env=_source_subprocess_environment(),
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_BLOB_UNAVAILABLE path={path}"
            ) from error
    return values


def _correctness_runtime_identity() -> dict[str, str]:
    runtime = {
        key: value
        for key, value in _runtime_identity().model_dump(mode="python").items()
        if value is not None
    }
    runtime.update(
        {
            "numpy": importlib.metadata.version("numpy"),
            "ml_dtypes": importlib.metadata.version("ml_dtypes"),
        }
    )
    return runtime


def validate_correctness_source_authority(
    authority: SurfaceCorrectnessSourceAuthority,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    source_blobs: dict[str, bytes],
) -> None:
    authority = SurfaceCorrectnessSourceAuthority.model_validate(
        authority.model_dump(mode="python", exclude_computed_fields=True)
    )
    paths = tuple(value.path for value in authority.dependencies)
    if (
        paths != CORRECTNESS_EXECUTABLE_DEPENDENCIES
        or tuple(source_blobs) != (*paths, "uv.lock")
        or authority.runtime != _correctness_runtime_identity()
        or authority.runtime.get("numpy") != protocol.numpy_version
        or authority.runtime.get("ml_dtypes") != protocol.ml_dtypes_version
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_AUTHORITY_MISMATCH")
    expected_hashes = {value.path: value.sha256 for value in authority.dependencies}
    expected_hashes["uv.lock"] = authority.uv_lock_sha256
    if any(_sha256_bytes(source_blobs[path]) != expected_hashes[path] for path in source_blobs):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_HASH_MISMATCH")
    committed = _read_committed_correctness_blobs(
        Path(authority.source_root), authority.source_commit
    )
    if committed != source_blobs:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_COMMIT_MISMATCH")


def capture_correctness_source_authority(
    repository_root: Path,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
) -> tuple[SurfaceCorrectnessSourceAuthority, dict[str, bytes]]:
    repository_root = repository_root.resolve(strict=True)
    if repository_root != Path("/home/sudarshan/tpu-cake-main"):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_ROOT_MISMATCH")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository_root,
            env=_source_subprocess_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_DIRTY")
    source_commit = git("rev-parse", "HEAD")
    remote = subprocess.run(
        [
            "/usr/bin/git",
            "ls-remote",
            "https://github.com/sdrshn-nmbr/tpu-cake.git",
            "refs/heads/main",
        ],
        cwd="/",
        env=_source_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(remote) != 2 or remote[1] != "refs/heads/main":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_REMOTE_MAIN_INVALID")
    blobs = _read_committed_correctness_blobs(repository_root, source_commit)
    authority = SurfaceCorrectnessSourceAuthority(
        source_commit=source_commit,
        origin_main_commit=git("rev-parse", "origin/main"),
        remote_main_commit=remote[0],
        runtime=_correctness_runtime_identity(),
        uv_lock_sha256=_sha256_bytes(blobs["uv.lock"]),
        dependencies=tuple(
            SourceFileContract(path=path, sha256=_sha256_bytes(blobs[path]))
            for path in CORRECTNESS_EXECUTABLE_DEPENDENCIES
        ),
    )
    validate_correctness_source_authority(authority, protocol, blobs)
    return authority, blobs


def _source_component_hashes(source_blobs: Mapping[str, bytes]) -> dict[str, str]:
    return {
        "executor_source_sha256": _sha256_bytes(
            source_blobs[CORRECTNESS_EXECUTOR_SOURCE_PATH.removeprefix("src/")]
        ),
        "worker_source_sha256": _sha256_bytes(
            source_blobs[CORRECTNESS_WORKER_SOURCE_PATH.removeprefix("src/")]
        ),
        "verifier_source_sha256": _sha256_bytes(
            source_blobs[CORRECTNESS_VERIFIER_SOURCE_PATH.removeprefix("src/")]
        ),
        "generator_source_sha256": _sha256_bytes(
            source_blobs[CORRECTNESS_GENERATOR_SOURCE_PATH.removeprefix("src/")]
        ),
        "oracle_source_sha256": _sha256_bytes(
            source_blobs[CORRECTNESS_ORACLE_SOURCE_PATH.removeprefix("src/")]
        ),
        "protocol_source_sha256": _sha256_bytes(
            source_blobs[CORRECTNESS_PROTOCOL_SOURCE_PATH.removeprefix("src/")]
        ),
        "evidence_source_sha256": _sha256_bytes(
            source_blobs[CORRECTNESS_EVIDENCE_SOURCE_PATH.removeprefix("src/")]
        ),
    }


class SurfaceCorrectnessDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0, le=7)
    process_index: Literal[0] = 0
    platform: Literal["tpu"] = "tpu"
    device_kind: Literal["TPU7x"] = "TPU7x"


class SurfaceCorrectnessExecutionAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-correctness-executor-v1"] = (
        CORRECTNESS_EXECUTOR_SCHEMA
    )
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: SurfaceCorrectnessSourceAuthority
    executor_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project: Literal["astral-medley-465922-b2"] = "astral-medley-465922-b2"
    zone: Literal["us-central1-c"] = "us-central1-c"
    hostname: Literal["tpu-cake-v7x-rsag-wx7r"] = "tpu-cake-v7x-rsag-wx7r"
    numeric_project_id: Literal["541760035156"] = "541760035156"
    instance_id: Literal["5064039476077763048"] = "5064039476077763048"
    instance_hostname: Literal[
        "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
    ] = "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
    machine_type: Literal["tpu7x-standard-4t"] = "tpu7x-standard-4t"
    cpu_platform: Literal["Intel Emerald Rapids"] = "Intel Emerald Rapids"
    compiler_environment: dict[str, str]
    devices: tuple[SurfaceCorrectnessDevice, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def device_inventory_is_exact(self) -> SurfaceCorrectnessExecutionAuthority:
        if tuple(value.id for value in self.devices) != tuple(range(8)):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_DEVICE_INVENTORY_MISMATCH")
        return self

    @computed_field
    @property
    def authority_sha256(self) -> str:
        return model_identity_sha256(self)


def validate_correctness_execution_authority(
    authority: SurfaceCorrectnessExecutionAuthority,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
    source_blobs: dict[str, bytes],
) -> None:
    authority = SurfaceCorrectnessExecutionAuthority.model_validate(
        authority.model_dump(mode="python", exclude_computed_fields=True)
    )
    validate_correctness_source_authority(authority.source, protocol, source_blobs)
    component_hashes = _source_component_hashes(source_blobs)
    protocol_blob = source_blobs["contracts/matmul-collective-surface-correctness-v1.json"]
    expected_runtime = {
        **design.runtime,
        "numpy": protocol.numpy_version,
        "ml_dtypes": protocol.ml_dtypes_version,
    }
    if (
        authority.protocol_id != protocol.protocol_id
        or authority.protocol_file_sha256 != _sha256_bytes(protocol_blob)
        or any(getattr(authority, key) != value for key, value in component_hashes.items())
        or authority.project != design.project
        or authority.zone != design.zone
        or authority.hostname != design.hostname
        or authority.numeric_project_id != design.numeric_project_id
        or authority.instance_id != design.instance_id
        or authority.instance_hostname != design.instance_hostname
        or authority.machine_type != design.machine_type
        or authority.cpu_platform != design.cpu_platform
        or authority.source.runtime != expected_runtime
        or authority.compiler_environment != design.compiler_environment
        or tuple(value.id for value in authority.devices) != design.device_ids
        or any(
            value.process_index != design.device_process_index
            or value.platform != design.backend
            or value.device_kind != design.device_kind
            for value in authority.devices
        )
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_EXECUTION_AUTHORITY_MISMATCH")


def _probe_execution_authority(
    protocol_path: Path,
    design_path: Path,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCorrectnessExecutionAuthority, dict[str, bytes]]:
    with tempfile.TemporaryDirectory(prefix="tpu-cake-correctness-authority-") as directory:
        output = Path(directory) / "authority.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tpu_cake.matmul_collective_surface_correctness_worker",
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
        authority = SurfaceCorrectnessExecutionAuthority.model_validate_json(output.read_text())
    source_blobs = _read_committed_correctness_blobs(
        Path(design.compilation_source_root), authority.source.source_commit
    )
    validate_correctness_execution_authority(authority, protocol, design, source_blobs)
    running = Path(__file__)
    expected = source_blobs[CORRECTNESS_EXECUTOR_SOURCE_PATH.removeprefix("src/")]
    if (
        running.is_symlink()
        or running.resolve()
        != (Path(design.compilation_source_root) / CORRECTNESS_EXECUTOR_SOURCE_PATH).resolve()
        or running.read_bytes() != expected
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RUNNING_EXECUTOR_SOURCE_MISMATCH")
    return authority, source_blobs


class SurfaceCorrectnessWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: MatmulCollectiveSurfaceSplit
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compilation_cache_schema: Literal["isolated-empty-temporary-directory-v1"] = (
        "isolated-empty-temporary-directory-v1"
    )
    parent_snapshot_path: str
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol
    design: MatmulCollectiveSurfaceDesignContract


class SurfaceCorrectnessWorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: MatmulCollectiveSurfaceSplit
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: MatmulCollectiveSurfaceCorrectnessEvidence


class SurfaceCorrectnessRunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: MatmulCollectiveSurfaceSplit
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_claim_path: str
    attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: str


class SurfaceCorrectnessManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def path_is_canonical(self) -> SurfaceCorrectnessManifestEntry:
        parts = self.path.split("/")
        if self.path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_MANIFEST_PATH_INVALID")
        return self


class SurfaceCorrectnessPhaseReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-correctness-receipt-v1"] = (
        CORRECTNESS_RECEIPT_SCHEMA
    )
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_sequence: Literal[1, 6]
    phase: Literal["correctness", "holdout_correctness"]
    split: MatmulCollectiveSurfaceSplit
    parent_compile_manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_phase_receipt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_claim_path: str
    attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def phase_is_canonical(self) -> SurfaceCorrectnessPhaseReceipt:
        calibration = self.split is MatmulCollectiveSurfaceSplit.CALIBRATION
        if calibration != (self.phase_sequence == 1 and self.phase == "correctness"):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RECEIPT_PHASE_MISMATCH")
        if calibration != (self.previous_phase_receipt_sha256 is None):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RECEIPT_CHAIN_MISMATCH")
        return self

    @computed_field
    @property
    def receipt_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCorrectnessManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-correctness-manifest-v1"] = (
        CORRECTNESS_MANIFEST_SCHEMA
    )
    identity: SurfaceCorrectnessRunIdentity
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[SurfaceCorrectnessManifestEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def artifact_inventory_is_canonical(self) -> SurfaceCorrectnessManifest:
        paths = tuple(value.path for value in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_MANIFEST_ORDER_INVALID")
        return self


def _canonical_protocol(path: Path) -> MatmulCollectiveSurfaceCorrectnessProtocol:
    protocol = MatmulCollectiveSurfaceCorrectnessProtocol.model_validate_json(path.read_text())
    if protocol != default_matmul_collective_surface_correctness_protocol():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_NONCANONICAL_PROTOCOL")
    return protocol


def _canonical_design(path: Path) -> MatmulCollectiveSurfaceDesignContract:
    design = MatmulCollectiveSurfaceDesignContract.model_validate_json(path.read_text())
    if design != default_matmul_collective_surface_design_contract():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_NONCANONICAL_DESIGN")
    return design


def _reject_links_in_path(root: Path) -> None:
    current = Path(root.anchor)
    for part in root.absolute().parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PATH_SYMLINK path={current}")


def _require_safe_new_root(root: Path) -> None:
    if not root.is_absolute():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ROOT_NOT_ABSOLUTE")
    root = root.resolve(strict=False)
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/"), Path.home().resolve(), repository_root)
    if (
        any(root == value or root in value.parents for value in protected)
        or repository_root in root.parents
    ):
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_UNSAFE_ROOT path={root}")
    _reject_links_in_path(root.parent)
    if root.exists() or root.is_symlink():
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ATTEMPT_EXISTS path={root}")
    if not root.parent.is_dir():
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_INVALID path={root.parent}")


def _preflight_existing_root(root: Path) -> None:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ROOT_AUTHORITY_INVALID")
    for path in root.rglob("*"):
        path_info = path.lstat()
        if stat.S_ISLNK(path_info.st_mode):
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ARTIFACT_SYMLINK path={path}")
        if stat.S_ISREG(path_info.st_mode) and path_info.st_nlink != 1:
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ARTIFACT_HARDLINK path={path}")


def _attempt_claim_path(
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    split: MatmulCollectiveSurfaceSplit,
) -> Path:
    key = hashlib.sha256(f"{protocol.protocol_id}:{split.value}".encode()).hexdigest()
    return Path(protocol.attempt_registry_root) / f"{key}.json"


def _attempt_claim_payload(
    root: Path,
    attempt_id: str,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    split: MatmulCollectiveSurfaceSplit,
    source_commit: str,
) -> dict[str, str]:
    return {
        "schema_version": CORRECTNESS_EXECUTOR_SCHEMA,
        "attempt_id": attempt_id,
        "protocol_id": protocol.protocol_id,
        "split": split.value,
        "source_commit": source_commit,
        "output_root": str(root),
        "state": "claimed",
    }


def _claim_attempt(
    root: Path,
    attempt_id: str,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    split: MatmulCollectiveSurfaceSplit,
    source_commit: str,
) -> Path:
    registry = Path(protocol.attempt_registry_root)
    _reject_links_in_path(registry.parent)
    if not registry.exists():
        _mkdir_exclusive_durable(registry)
    info = registry.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ATTEMPT_REGISTRY_INVALID")
    claim = _attempt_claim_path(protocol, split)
    try:
        _write_bytes_exclusive(
            claim,
            _json_bytes(_attempt_claim_payload(root, attempt_id, protocol, split, source_commit)),
        )
    except FileExistsError as error:
        raise ValueError(
            "MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ATTEMPT_PERMANENTLY_CLAIMED"
        ) from error
    return claim


@contextmanager
def _exclusive_attempt_lock(attempt_id: str) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-correctness-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    info = lock_root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_LOCK_ROOT_INVALID")
    descriptor = os.open(
        lock_root / f"{attempt_id}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ATTEMPT_LOCKED") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_source_bundle(
    root: Path,
    authority: SurfaceCorrectnessExecutionAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    for path, blob in source_blobs.items():
        _write_bytes_exclusive(root / "source" / "committed" / path, blob)
    component_paths = {
        "executor.py": CORRECTNESS_EXECUTOR_SOURCE_PATH,
        "worker.py": CORRECTNESS_WORKER_SOURCE_PATH,
        "verifier.py": CORRECTNESS_VERIFIER_SOURCE_PATH,
        "generator.py": CORRECTNESS_GENERATOR_SOURCE_PATH,
        "oracle.py": CORRECTNESS_ORACLE_SOURCE_PATH,
        "protocol.py": CORRECTNESS_PROTOCOL_SOURCE_PATH,
        "evidence.py": CORRECTNESS_EVIDENCE_SOURCE_PATH,
    }
    for destination, source in component_paths.items():
        blob = source_blobs[source.removeprefix("src/")]
        _write_bytes_exclusive(root / "source" / destination, blob)
    if authority.source.authority_sha256 != model_identity_sha256(authority.source):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SOURCE_BUNDLE_INVALID")


def _launch_worker(
    root: Path,
    request_path: Path,
    design: MatmulCollectiveSurfaceDesignContract,
) -> None:
    with tempfile.TemporaryDirectory(prefix="tpu-cake-correctness-cache-") as cache_root:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tpu_cake.matmul_collective_surface_correctness_worker",
                "--root",
                str(root),
                "--request",
                str(request_path),
            ],
            cwd=Path(design.compilation_source_root),
            env=_canonical_subprocess_environment(
                design,
                compilation_cache_dir=Path(cache_root),
            ),
            check=True,
        )


def _validate_worker_result(
    root: Path,
    request: SurfaceCorrectnessWorkerRequest,
    identity: SurfaceCorrectnessRunIdentity,
    authority: SurfaceCorrectnessExecutionAuthority,
) -> SurfaceCorrectnessWorkerResult:
    started = json.loads((root / "STARTED.json").read_text())
    if started != {
        "attempt_id": identity.attempt_id,
        "invocation_nonce": request.invocation_nonce,
        "split": request.split.value,
        "state": "started",
    }:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_START_CLAIM_MISMATCH")
    result = SurfaceCorrectnessWorkerResult.model_validate_json(
        (root / "worker-result.json").read_text()
    )
    if (
        result.attempt_id != identity.attempt_id
        or result.split is not request.split
        or result.invocation_nonce != request.invocation_nonce
        or result.execution_authority_sha256 != authority.authority_sha256
        or result.evidence.correctness_execution_authority_sha256 != authority.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_WORKER_RESULT_MISMATCH")
    validate_surface_correctness_evidence(
        result.evidence,
        request.protocol,
        request.design,
        expected_protocol_file_sha256=authority.protocol_file_sha256,
        expected_execution_authority_sha256=authority.authority_sha256,
        expected_invocation_nonce=request.invocation_nonce,
        expected_worker_pid=result.worker_pid,
    )
    return result


def _manifest_entries(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset({"manifest.json"}),
) -> tuple[SurfaceCorrectnessManifestEntry, ...]:
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
        SurfaceCorrectnessManifestEntry(
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
    evidence: MatmulCollectiveSurfaceCorrectnessEvidence,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    schedules = {
        f"{value.scenario_name}:{value.strategy.value}": [
            value.observed_distributed_schedule_sha256,
            value.observed_physical_schedule_sha256,
            value.observed_pallas_source_sha256,
        ]
        for value in evidence.continuity
    }
    compiled = {
        f"{value.scenario_name}:{value.strategy.value}": value.compile_record_sha256
        for value in evidence.continuity
    }
    return schedules, compiled


def _run_archived_independent_verifier(
    root: Path,
    identity: SurfaceCorrectnessRunIdentity,
    authority: SurfaceCorrectnessExecutionAuthority,
    evidence: MatmulCollectiveSurfaceCorrectnessEvidence,
    phase_ledger: SurfacePhaseLedger,
    receipt: SurfaceCorrectnessPhaseReceipt,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "source/verifier.py"),
            "--root",
            str(root),
            "--protocol",
            str(root / "source/committed/contracts/matmul-collective-surface-correctness-v1.json"),
            "--design",
            str(root / "source/committed/contracts/matmul-collective-surface-design-v1.json"),
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
    verification = json.loads(completed.stdout)
    expected = {
        "attempt_id": identity.attempt_id,
        "protocol_id": identity.protocol_id,
        "split": identity.split.value,
        "source_authority_sha256": authority.source.authority_sha256,
        "execution_authority_sha256": authority.authority_sha256,
        "evidence_sha256": evidence.evidence_sha256,
        "ledger_sha256": _file_sha256(root / "ledger.sqlite"),
        "phase_ledger_sha256": model_identity_sha256(phase_ledger),
        "receipt_sha256": receipt.receipt_sha256,
        "case_count": len(evidence.cases),
        "execution_count": sum(len(case.executions) for case in evidence.cases),
    }
    if verification != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_INDEPENDENT_REPLAY_MISMATCH")
    return verification


def execute_surface_correctness(
    root: Path,
    protocol_path: Path,
    design_path: Path,
    attempt_id: str,
    split: MatmulCollectiveSurfaceSplit = MatmulCollectiveSurfaceSplit.CALIBRATION,
) -> SurfaceCorrectnessManifest:
    protocol = _canonical_protocol(protocol_path)
    design = _canonical_design(design_path)
    root = root.resolve(strict=False)
    if split is not MatmulCollectiveSurfaceSplit.CALIBRATION:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_HOLDOUT_NOT_AUTHORIZED")
    if len(attempt_id) != 64 or any(value not in "0123456789abcdef" for value in attempt_id):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ATTEMPT_ID_INVALID")
    with _exclusive_attempt_lock(attempt_id):
        _require_safe_new_root(root)
        authority, source_blobs = _probe_execution_authority(
            protocol_path, design_path, protocol, design
        )
        claim = _claim_attempt(
            root,
            attempt_id,
            protocol,
            split,
            authority.source.source_commit,
        )
        claim_sha256 = _file_sha256(claim)
        _mkdir_exclusive_durable(root)
        try:
            identity = SurfaceCorrectnessRunIdentity(
                attempt_id=attempt_id,
                protocol_id=protocol.protocol_id,
                split=split,
                execution_authority_sha256=authority.authority_sha256,
                attempt_claim_path=str(claim),
                attempt_claim_sha256=claim_sha256,
                output_root=str(root),
            )
            _write_model_exclusive(root / "protocol.json", protocol)
            _write_model_exclusive(root / "design.json", design)
            _write_model_exclusive(root / "execution_authority.json", authority)
            _write_model_exclusive(root / "run_identity.json", identity)
            _write_bytes_exclusive(root / "attempt_claim.json", claim.read_bytes())
            _write_source_bundle(root, authority, source_blobs)
            parent_snapshot = _copy_parent_compile_snapshot(root, protocol)
            ledger = EvidenceRun(root / "ledger.sqlite", attempt_id)
            created_payload = {
                "protocol_id": protocol.protocol_id,
                "split": split.value,
                "execution_authority_sha256": authority.authority_sha256,
                "attempt_claim_path": str(claim),
                "attempt_claim_sha256": claim_sha256,
            }
            ledger.create(created_payload)
            component_hashes = _source_component_hashes(source_blobs)
            verified_payload = {
                "source_authority_sha256": authority.source.authority_sha256,
                "parent_compile_manifest_file_sha256": (
                    protocol.parent_compile.manifest_file_sha256
                ),
                "parent_compile_report_sha256": protocol.parent_compile.compile_report_sha256,
                **{
                    key: component_hashes[key]
                    for key in (
                        "executor_source_sha256",
                        "worker_source_sha256",
                        "verifier_source_sha256",
                    )
                },
                "devices": [value.model_dump(mode="json") for value in authority.devices],
            }
            ledger.transition(RunState.VERIFIED, verified_payload)
            request = SurfaceCorrectnessWorkerRequest(
                attempt_id=attempt_id,
                split=split,
                invocation_nonce=hashlib.sha256(os.urandom(32)).hexdigest(),
                execution_authority_sha256=authority.authority_sha256,
                parent_snapshot_path=str(parent_snapshot),
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
                        "split": split.value,
                        "state": "started",
                    }
                ),
            )
            _launch_worker(root, root / "worker-request.json", design)
            result = _validate_worker_result(root, request, identity, authority)
            evidence = result.evidence
            _write_model_exclusive(root / "evidence.json", evidence)
            schedules, compiled = _continuity_payloads(evidence)
            ledger.transition(
                RunState.LOWERED,
                {"continuity_schedule_set_sha256": payload_sha256(schedules)},
            )
            ledger.transition(
                RunState.COMPILED,
                {"fresh_compile_set_sha256": payload_sha256(compiled)},
            )
            ledger.transition(RunState.CORRECT, {"evidence_sha256": evidence.evidence_sha256})
            phase_ledger = SurfacePhaseLedger(attempt_id=attempt_id)
            phase_ledger = record_surface_phase(
                phase_ledger,
                SurfacePhase.COMPILE,
                protocol.parent_compile.compile_report_sha256,
            )
            phase_ledger = record_surface_phase(
                phase_ledger,
                SurfacePhase.CORRECTNESS,
                evidence.evidence_sha256,
            )
            _write_model_exclusive(root / "phase_ledger.json", phase_ledger)
            artifact_set_sha256 = _artifact_set_sha256(root)
            ledger.transition(
                RunState.VALIDATED,
                {
                    "phase_ledger_sha256": model_identity_sha256(phase_ledger),
                    "artifact_set_sha256": artifact_set_sha256,
                },
            )
            ledger.transition(
                RunState.ACCEPTED,
                {
                    "evidence_sha256": evidence.evidence_sha256,
                    "validation": "producer-schema-and-artifact-replay-v1",
                },
            )
            ledger.seal("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_LEDGER_SIDECARS")
            receipt = SurfaceCorrectnessPhaseReceipt(
                attempt_id=attempt_id,
                protocol_id=protocol.protocol_id,
                phase_sequence=1,
                phase="correctness",
                split=split,
                parent_compile_manifest_file_sha256=(protocol.parent_compile.manifest_file_sha256),
                evidence_file_sha256=_file_sha256(root / "evidence.json"),
                evidence_sha256=evidence.evidence_sha256,
                artifact_set_sha256=artifact_set_sha256,
                ledger_snapshot_sha256=_file_sha256(root / "ledger.sqlite"),
                attempt_claim_path=str(claim),
                attempt_claim_sha256=claim_sha256,
            )
            _write_model_exclusive(root / "receipt.json", receipt)
            manifest = SurfaceCorrectnessManifest(
                identity=identity,
                evidence_sha256=evidence.evidence_sha256,
                ledger_snapshot_sha256=_file_sha256(root / "ledger.sqlite"),
                receipt_sha256=receipt.receipt_sha256,
                artifacts=_manifest_entries(root),
            )
            _write_model_exclusive(root / "manifest.json", manifest)
            _run_archived_independent_verifier(
                root,
                identity,
                authority,
                evidence,
                phase_ledger,
                receipt,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    manifest = execute_surface_correctness(
        args.root,
        args.protocol,
        args.design,
        args.attempt_id,
    )
    print(manifest.model_dump_json())


if __name__ == "__main__":
    main()
