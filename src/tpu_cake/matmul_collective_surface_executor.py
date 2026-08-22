from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
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

from tpu_cake.compiler_analysis import (
    CompilerExecutableAnalysis,
    validate_compiler_analysis,
)
from tpu_cake.identity import model_identity_sha256
from tpu_cake.ledger import EvidenceRun, RunState, payload_sha256, read_ledger_history
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    default_matmul_collective_surface_design_contract,
    derive_matmul_collective_surface_design_report,
)
from tpu_cake.matmul_collective_surface_runner import (
    _SYSTEM_GIT,
    SURFACE_EXECUTABLE_DEPENDENCIES,
    CompileCaptureRecord,
    MatmulCollectiveSurfaceCompileReport,
    MatmulCollectiveSurfaceSourceAuthority,
    SurfaceCompileStatus,
    _read_committed_source_blobs,
    _semantic_compiler_hlo,
    _source_subprocess_environment,
    _text_sha256,
    _validate_compiler_hlo_static_abi,
    _validate_stablehlo_static_abi,
    derive_surface_input_identities,
    validate_compile_capture_report,
    validate_surface_source_authority,
)
from tpu_cake.receipt import _validate_matmul_compiler_strategy
from tpu_cake.runner import MatmulCollectiveStrategy

EXECUTOR_SCHEMA = "matmul-collective-surface-compile-executor-v1"
EXECUTOR_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_executor.py"
WORKER_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_compile_worker.py"
VERIFIER_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_compile_verifier.py"
_FINAL_STATES = (RunState.CREATED, RunState.VERIFIED, RunState.LOWERED, RunState.COMPILED)


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
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_exclusive_durable(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(mode=mode, exist_ok=False)
    _fsync_directory(path.parent)


def _write_json_exclusive(path: Path, value: object) -> None:
    _write_bytes_exclusive(path, _json_bytes(value))


def _write_model_exclusive(path: Path, value: BaseModel) -> None:
    _write_json_exclusive(
        path,
        value.model_dump(mode="json", exclude_computed_fields=True),
    )


class SurfaceCompileDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0)
    process_index: int = Field(ge=0)
    platform: str
    device_kind: str


class SurfaceCompileExecutionAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-compile-executor-v1"] = EXECUTOR_SCHEMA
    source: MatmulCollectiveSurfaceSourceAuthority
    executor_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project: str
    zone: str
    hostname: str
    numeric_project_id: str
    instance_id: str
    instance_hostname: str
    machine_type: str
    cpu_platform: str
    backend: str
    runtime: dict[str, str | None]
    compiler_environment: dict[str, str]
    devices: tuple[SurfaceCompileDevice, ...] = Field(min_length=8, max_length=8)

    @computed_field
    @property
    def authority_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCompileRunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_output_root: str
    attempt_claim_path: str
    attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SurfaceCompileWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    repetition: int = Field(ge=1, le=2)
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compilation_cache_schema: Literal["isolated-empty-temporary-directory-v1"] = (
        "isolated-empty-temporary-directory-v1"
    )
    contract: MatmulCollectiveSurfaceDesignContract


class SurfaceCompileAbstractInputABI(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lhs_shape: tuple[int, int]
    lhs_dtype: Literal["bfloat16"]
    lhs_sharding: Literal["PartitionSpec(None, 't')"]
    rhs_shape: tuple[int, int]
    rhs_dtype: Literal["bfloat16"]
    rhs_sharding: Literal["PartitionSpec('t', None)"]
    output_shape: tuple[int, int]
    output_dtype: Literal["float32"]
    output_sharding: Literal["PartitionSpec(None, 't')"]
    schema_version: Literal["global-shape-dtype-named-sharding-v1"] = (
        "global-shape-dtype-named-sharding-v1"
    )


class SurfaceCompileCaptureEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capture: CompileCaptureRecord
    abstract_input_abi: SurfaceCompileAbstractInputABI
    stablehlo_path: str
    compiler_hlo_path: str
    compiler_analysis_path: str
    compiler_analysis: CompilerExecutableAnalysis


class SurfaceCompileWorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    repetition: int = Field(ge=1, le=2)
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captures: tuple[SurfaceCompileCaptureEnvelope, ...] = Field(min_length=40, max_length=40)


class SurfaceCompileManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SurfaceCompileManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-compile-executor-v1"] = EXECUTOR_SCHEMA
    identity: SurfaceCompileRunIdentity
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[SurfaceCompileManifestEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def paths_are_canonical(self) -> SurfaceCompileManifest:
        paths = tuple(value.path for value in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_MANIFEST_ORDER")
        return self


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
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COMPILE_METADATA_REDIRECT code={code}")


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
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_METADATA_HEADER_MISSING")
        payload = response.read(4097)
    if len(payload) > 4096:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_METADATA_TOO_LARGE")
    return payload.decode().strip()


def _compiler_environment(contract: MatmulCollectiveSurfaceDesignContract) -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in contract.compiler_environment}
    if observed != contract.compiler_environment:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_DECLARED_ENVIRONMENT_MISMATCH")
    forbidden = {
        key: value
        for key, value in os.environ.items()
        if (key == "TPU_LIBRARY_PATH" or key.startswith(("JAX_", "XLA_", "PJRT_", "LIBTPU_")))
        and key not in contract.compiler_environment
        and key != "JAX_COMPILATION_CACHE_DIR"
    }
    if forbidden:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_UNDECLARED_ENVIRONMENT")
    return dict(contract.compiler_environment)


def _canonical_subprocess_environment(
    contract: MatmulCollectiveSurfaceDesignContract,
    *,
    compilation_cache_dir: Path | None = None,
) -> dict[str, str]:
    environment = _source_subprocess_environment()
    environment.update(contract.compiler_environment)
    if compilation_cache_dir is not None:
        environment["JAX_COMPILATION_CACHE_DIR"] = str(compilation_cache_dir)
    return environment


def _executor_source_sha256(repository_root: Path, commit: str) -> str:
    return _sha256_bytes(_executor_source_blob(repository_root, commit))


def _executor_source_blob(repository_root: Path, commit: str) -> bytes:
    try:
        return subprocess.run(
            [_SYSTEM_GIT, "show", f"{commit}:{EXECUTOR_SOURCE_PATH}"],
            cwd=repository_root,
            env=_source_subprocess_environment(),
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_EXECUTOR_SOURCE_UNAVAILABLE") from error


def _worker_source_sha256(repository_root: Path, commit: str) -> str:
    return _sha256_bytes(_worker_source_blob(repository_root, commit))


def _worker_source_blob(repository_root: Path, commit: str) -> bytes:
    try:
        return subprocess.run(
            [_SYSTEM_GIT, "show", f"{commit}:{WORKER_SOURCE_PATH}"],
            cwd=repository_root,
            env=_source_subprocess_environment(),
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_SOURCE_UNAVAILABLE") from error


def _verifier_source_sha256(repository_root: Path, commit: str) -> str:
    return _sha256_bytes(_verifier_source_blob(repository_root, commit))


def _verifier_source_blob(repository_root: Path, commit: str) -> bytes:
    try:
        return subprocess.run(
            [_SYSTEM_GIT, "show", f"{commit}:{VERIFIER_SOURCE_PATH}"],
            cwd=repository_root,
            env=_source_subprocess_environment(),
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_VERIFIER_SOURCE_UNAVAILABLE") from error


def _probe_execution_authority(
    contract_path: Path,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCompileExecutionAuthority, dict[str, bytes]]:
    with tempfile.TemporaryDirectory(prefix="tpu-cake-surface-authority-") as directory:
        output = Path(directory) / "authority.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tpu_cake.matmul_collective_surface_compile_worker",
                "probe",
                "--contract",
                str(contract_path),
                "--output",
                str(output),
            ],
            cwd=Path(contract.compilation_source_root),
            env=_canonical_subprocess_environment(contract),
            check=True,
        )
        authority = SurfaceCompileExecutionAuthority.model_validate_json(output.read_text())
    source_blobs = _read_committed_source_blobs(
        Path(contract.compilation_source_root), authority.source.source_commit
    )
    validate_execution_authority(
        authority,
        contract,
        source_blobs,
    )
    repository_root = Path(contract.compilation_source_root)
    if Path(__file__).resolve() != (
        repository_root / EXECUTOR_SOURCE_PATH
    ).resolve() or _file_sha256(Path(__file__)) != _executor_source_sha256(
        repository_root, authority.source.source_commit
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_RUNNING_EXECUTOR_SOURCE_MISMATCH")
    return authority, source_blobs


def validate_execution_authority(
    authority: SurfaceCompileExecutionAuthority,
    contract: MatmulCollectiveSurfaceDesignContract,
    source_blobs: Mapping[str, bytes],
) -> None:
    validate_surface_source_authority(authority.source, contract, source_blobs)
    repository_root = Path(contract.compilation_source_root)
    if authority.executor_source_sha256 != _executor_source_sha256(
        repository_root, authority.source.source_commit
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_EXECUTOR_SOURCE_MISMATCH")
    if authority.worker_source_sha256 != _worker_source_sha256(
        repository_root, authority.source.source_commit
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_SOURCE_MISMATCH")
    if authority.verifier_source_sha256 != _verifier_source_sha256(
        repository_root, authority.source.source_commit
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_VERIFIER_SOURCE_MISMATCH")
    expected_devices = tuple(authority.devices)
    if (
        authority.project != contract.project
        or authority.zone != contract.zone
        or authority.hostname != contract.hostname
        or authority.numeric_project_id != contract.numeric_project_id
        or authority.instance_id != contract.instance_id
        or authority.instance_hostname != contract.instance_hostname
        or authority.machine_type != contract.machine_type
        or authority.cpu_platform != contract.cpu_platform
        or authority.backend != contract.backend
        or authority.runtime != contract.runtime
        or authority.compiler_environment != contract.compiler_environment
        or len(expected_devices) != contract.device_count
        or tuple(value.id for value in expected_devices) != contract.device_ids
        or any(
            value.process_index != contract.device_process_index
            or value.platform != contract.backend
            or value.device_kind != contract.device_kind
            for value in expected_devices
        )
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_EXECUTION_AUTHORITY_MISMATCH")


def _canonical_contract(path: Path) -> MatmulCollectiveSurfaceDesignContract:
    contract = MatmulCollectiveSurfaceDesignContract.model_validate_json(path.read_text())
    if contract != default_matmul_collective_surface_design_contract():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_NONCANONICAL_CONTRACT")
    return contract


def _write_source_bundle(
    root: Path,
    authority: SurfaceCompileExecutionAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    for path, blob in source_blobs.items():
        _write_bytes_exclusive(root / "source" / "committed" / path, blob)
    executor_blob = _executor_source_blob(
        Path(authority.source.compilation_source_root),
        authority.source.source_commit,
    )
    _write_bytes_exclusive(root / "source" / "executor.py", executor_blob)
    worker_blob = _worker_source_blob(
        Path(authority.source.compilation_source_root),
        authority.source.source_commit,
    )
    _write_bytes_exclusive(root / "source" / "worker.py", worker_blob)
    verifier_blob = _verifier_source_blob(
        Path(authority.source.compilation_source_root),
        authority.source.source_commit,
    )
    _write_bytes_exclusive(root / "source" / "verifier.py", verifier_blob)


def _validate_source_bundle_offline(
    root: Path,
    authority: SurfaceCompileExecutionAuthority,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> dict[str, bytes]:
    source = authority.source
    if (
        source.branch != contract.source_branch
        or source.source_commit != source.origin_main_commit
        or source.source_commit != source.remote_main_commit
        or source.remote_url != contract.source_remote_url
        or source.compilation_source_root != contract.compilation_source_root
        or source.runtime != contract.runtime
        or authority.runtime != contract.runtime
        or authority.compiler_environment != contract.compiler_environment
        or authority.project != contract.project
        or authority.zone != contract.zone
        or authority.hostname != contract.hostname
        or authority.numeric_project_id != contract.numeric_project_id
        or authority.instance_id != contract.instance_id
        or authority.instance_hostname != contract.instance_hostname
        or authority.machine_type != contract.machine_type
        or authority.cpu_platform != contract.cpu_platform
        or authority.backend != contract.backend
        or len(authority.devices) != contract.device_count
        or tuple(value.id for value in authority.devices) != contract.device_ids
        or any(
            value.process_index != contract.device_process_index
            or value.platform != contract.backend
            or value.device_kind != contract.device_kind
            for value in authority.devices
        )
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_SOURCE_AUTHORITY_MISMATCH")
    dependency_paths = tuple(value.path for value in source.dependencies)
    if dependency_paths != SURFACE_EXECUTABLE_DEPENDENCIES:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_DEPENDENCY_CLOSURE_MISMATCH")
    expected_paths = (*SURFACE_EXECUTABLE_DEPENDENCIES, "uv.lock")
    bundle_root = root / "source" / "committed"
    observed_paths = tuple(
        sorted(
            path.relative_to(bundle_root).as_posix()
            for path in bundle_root.rglob("*")
            if path.is_file()
        )
    )
    if observed_paths != tuple(sorted(expected_paths)):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_SOURCE_INVENTORY_MISMATCH")
    blobs = {path: (bundle_root / path).read_bytes() for path in expected_paths}
    expected_hashes = {value.path: value.sha256 for value in source.dependencies}
    expected_hashes["uv.lock"] = source.uv_lock_sha256
    if any(_sha256_bytes(blobs[path]) != expected_hashes[path] for path in expected_paths):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_SOURCE_HASH_MISMATCH")
    if _file_sha256(root / "source" / "executor.py") != authority.executor_source_sha256:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_EXECUTOR_HASH_MISMATCH")
    if _file_sha256(root / "source" / "worker.py") != authority.worker_source_sha256:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_WORKER_HASH_MISMATCH")
    if _file_sha256(root / "source" / "verifier.py") != authority.verifier_source_sha256:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_VERIFIER_HASH_MISMATCH")
    return blobs


def _validate_compile_report_offline(
    report: MatmulCollectiveSurfaceCompileReport,
    contract: MatmulCollectiveSurfaceDesignContract,
    authority: SurfaceCompileExecutionAuthority,
) -> None:
    if (
        report.design_id != contract.design_id
        or report.source_authority_sha256 != authority.source.authority_sha256
        or report.execution_authority_sha256 != authority.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_REPORT_AUTHORITY_MISMATCH")
    expected = tuple(
        (scenario.name, strategy, repetition)
        for scenario in contract.scenarios
        for strategy in contract.strategies
        for repetition in (1, 2)
    )
    if (
        tuple((value.scenario_name, value.strategy, value.repetition) for value in report.captures)
        != expected
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_INVENTORY_MISMATCH")
    design = derive_matmul_collective_surface_design_report(contract)
    arms = {(value.scenario_name, value.strategy): value for value in design.arms}
    inputs = {
        value.scenario_name: value.input_contract_sha256
        for value in derive_surface_input_identities(contract)
    }
    if len(inputs) != len(contract.scenarios) or len(set(inputs.values())) != len(inputs):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_INPUT_IDENTITY_COLLISION")
    semantic_hashes: dict[tuple[str, MatmulCollectiveStrategy], set[tuple[str, str]]] = {}
    for capture in report.captures:
        arm = arms[(capture.scenario_name, capture.strategy)]
        if (
            capture.status is SurfaceCompileStatus.FAILED
            or capture.input_contract_sha256 != inputs[capture.scenario_name]
            or capture.distributed_schedule_sha256 != arm.distributed_schedule_sha256
            or capture.physical_schedule_sha256 != arm.physical_schedule_sha256
            or capture.pallas_source_sha256 != arm.pallas_source_sha256
            or capture.stablehlo_sha256 != _text_sha256(capture.stablehlo)
            or capture.semantic_stablehlo_sha256
            != _text_sha256(capture.stablehlo.rstrip("\n") + "\n")
            or capture.compiler_hlo_sha256 != _text_sha256(capture.compiler_hlo)
            or capture.semantic_compiler_hlo_sha256
            != _text_sha256(_semantic_compiler_hlo(capture.compiler_hlo))
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_CAPTURE_MISMATCH")
        _validate_stablehlo_static_abi(capture, contract)
        _validate_compiler_hlo_static_abi(capture, contract)
        _validate_matmul_compiler_strategy(
            capture.stablehlo,
            capture.compiler_hlo,
            capture.strategy,
        )
        semantic_hashes.setdefault((capture.scenario_name, capture.strategy), set()).add(
            (capture.semantic_stablehlo_sha256, capture.semantic_compiler_hlo_sha256)
        )
    if any(len(values) != 1 for values in semantic_hashes.values()):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_OFFLINE_UNSTABLE_COMPILER_HASH")


def _reject_links(root: Path) -> None:
    current = Path(root.anchor)
    for part in root.absolute().parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COMPILE_PATH_SYMLINK path={current}")


def _require_safe_new_root(root: Path) -> None:
    if not root.is_absolute():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ROOT_NOT_ABSOLUTE")
    root = root.resolve(strict=False)
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/"), Path.home().resolve(), repository_root)
    if (
        any(root == value or root in value.parents for value in protected)
        or repository_root in root.parents
    ):
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COMPILE_UNSAFE_ROOT path={root}")
    _reject_links(root.parent)
    if root.exists() or root.is_symlink():
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_EXISTS path={root}")
    if not root.parent.is_dir():
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COMPILE_PARENT_INVALID path={root.parent}")


def _preflight_existing_root(root: Path) -> None:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ROOT_AUTHORITY_INVALID")
    for path in root.rglob("*"):
        path_info = path.lstat()
        if stat.S_ISLNK(path_info.st_mode):
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COMPILE_ARTIFACT_SYMLINK path={path}")
        if stat.S_ISREG(path_info.st_mode) and path_info.st_nlink != 1:
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COMPILE_ARTIFACT_HARDLINK path={path}")


def _claim_attempt(
    root: Path,
    attempt_id: str,
    contract: MatmulCollectiveSurfaceDesignContract,
    source_commit: str,
) -> Path:
    registry_root = _attempt_registry_root(contract)
    _reject_links(registry_root.parent)
    if not registry_root.exists():
        _mkdir_exclusive_durable(registry_root)
    info = registry_root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_REGISTRY_INVALID")
    claim_path = _attempt_claim_path(contract, source_commit)
    payload = _attempt_claim_payload(root, attempt_id, contract, source_commit)
    try:
        _write_json_exclusive(claim_path, payload)
    except FileExistsError as error:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_PERMANENTLY_CLAIMED") from error
    return claim_path


def _attempt_registry_root(contract: MatmulCollectiveSurfaceDesignContract) -> Path:
    return Path(contract.attempt_registry_root)


def _attempt_claim_path(
    contract: MatmulCollectiveSurfaceDesignContract,
    source_commit: str,
) -> Path:
    key = hashlib.sha256(f"{contract.design_id}:{source_commit}".encode()).hexdigest()
    return _attempt_registry_root(contract) / f"{key}.json"


def _attempt_claim_payload(
    root: Path,
    attempt_id: str,
    contract: MatmulCollectiveSurfaceDesignContract,
    source_commit: str,
) -> dict[str, str]:
    return {
        "schema_version": EXECUTOR_SCHEMA,
        "attempt_id": attempt_id,
        "design_id": contract.design_id,
        "source_commit": source_commit,
        "output_root": str(root),
        "state": "claimed",
    }


def _validate_attempt_claim(
    root: Path,
    attempt_id: str,
    contract: MatmulCollectiveSurfaceDesignContract,
    source_commit: str,
) -> None:
    claim_path = _attempt_claim_path(contract, source_commit)
    if claim_path.is_symlink() or not claim_path.is_file() or claim_path.stat().st_nlink != 1:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_CLAIM_MISSING")
    if json.loads(claim_path.read_text()) != _attempt_claim_payload(
        root, attempt_id, contract, source_commit
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_CLAIM_MISMATCH")


@contextmanager
def _exclusive_attempt_lock(attempt_id: str) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-surface-compile-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    info = lock_root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LOCK_ROOT_INVALID")
    descriptor = os.open(
        lock_root / f"{attempt_id}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_LOCKED") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _expected_capture_paths(
    repetition: int,
    scenario_name: str,
    strategy: MatmulCollectiveStrategy,
) -> tuple[str, str]:
    base = f"repetition-{repetition}/arms/{scenario_name}/{strategy.value}"
    return f"{base}/stablehlo.txt", f"{base}/compiler_hlo.txt"


def _analysis_path(
    repetition: int,
    scenario_name: str,
    strategy: MatmulCollectiveStrategy,
) -> str:
    base = f"repetition-{repetition}/arms/{scenario_name}/{strategy.value}"
    return f"{base}/compiler_analysis.json"


def _launch_worker(root: Path, request_path: Path, repetition: int) -> None:
    with tempfile.TemporaryDirectory(prefix=f"tpu-cake-surface-cache-{repetition}-") as cache_root:
        request = SurfaceCompileWorkerRequest.model_validate_json(request_path.read_text())
        environment = _canonical_subprocess_environment(
            request.contract,
            compilation_cache_dir=Path(cache_root),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tpu_cake.matmul_collective_surface_compile_worker",
                "worker",
                "--root",
                str(root),
                "--request",
                str(request_path),
            ],
            cwd=Path(request.contract.compilation_source_root),
            env=environment,
            check=True,
        )


def _validate_worker_results(
    root: Path,
    identity: SurfaceCompileRunIdentity,
    authority: SurfaceCompileExecutionAuthority,
    contract: MatmulCollectiveSurfaceDesignContract,
    requests: tuple[SurfaceCompileWorkerRequest, ...],
) -> tuple[SurfaceCompileWorkerResult, SurfaceCompileWorkerResult]:
    results = tuple(
        SurfaceCompileWorkerResult.model_validate_json(
            (root / f"repetition-{repetition}/result.json").read_text()
        )
        for repetition in (1, 2)
    )
    if len({value.worker_pid for value in results}) != 2:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_PID_REUSED")
    if any(
        request.contract != contract
        or request.attempt_id != identity.attempt_id
        or request.authority_sha256 != authority.authority_sha256
        for request in requests
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_REQUEST_MISMATCH")
    expected_inventory = tuple(
        (scenario.name, strategy)
        for scenario in contract.scenarios
        for strategy in contract.strategies
    )
    for result, request in zip(results, requests, strict=True):
        started = json.loads((root / f"repetition-{request.repetition}/STARTED.json").read_text())
        if started != {
            "attempt_id": request.attempt_id,
            "invocation_nonce": request.invocation_nonce,
            "repetition": request.repetition,
            "state": "started",
        }:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_START_CLAIM_MISMATCH")
        observed = tuple(
            (value.capture.scenario_name, value.capture.strategy) for value in result.captures
        )
        if (
            result.attempt_id != identity.attempt_id
            or result.repetition != request.repetition
            or result.invocation_nonce != request.invocation_nonce
            or result.authority_sha256 != authority.authority_sha256
            or observed != expected_inventory
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_WORKER_RESULT_MISMATCH")
        for envelope in result.captures:
            capture = envelope.capture
            scenario = next(
                value for value in contract.scenarios if value.name == capture.scenario_name
            )
            expected_paths = _expected_capture_paths(
                result.repetition, capture.scenario_name, capture.strategy
            )
            expected_analysis_path = _analysis_path(
                result.repetition, capture.scenario_name, capture.strategy
            )
            if (envelope.stablehlo_path, envelope.compiler_hlo_path) != expected_paths:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_RAW_PATH_MISMATCH")
            if envelope.compiler_analysis_path != expected_analysis_path:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ANALYSIS_PATH_MISMATCH")
            stablehlo_path, compiler_hlo_path = (root / value for value in expected_paths)
            if (
                stablehlo_path.read_text() != capture.stablehlo
                or compiler_hlo_path.read_text() != capture.compiler_hlo
            ):
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_RAW_HLO_MISMATCH")
            analysis = validate_compiler_analysis(
                root / expected_analysis_path,
                stablehlo_path=stablehlo_path,
                compiler_hlo_path=compiler_hlo_path,
            )
            expected_abi = SurfaceCompileAbstractInputABI(
                lhs_shape=(scenario.m, scenario.k),
                lhs_dtype="bfloat16",
                lhs_sharding="PartitionSpec(None, 't')",
                rhs_shape=(scenario.k, scenario.n),
                rhs_dtype="bfloat16",
                rhs_sharding="PartitionSpec('t', None)",
                output_shape=(scenario.m, scenario.n),
                output_dtype="float32",
                output_sharding="PartitionSpec(None, 't')",
            )
            if (
                envelope.compiler_analysis != analysis
                or analysis.analysis_schema != contract.compiler_analysis_schema
                or envelope.abstract_input_abi != expected_abi
                or envelope.abstract_input_abi.schema_version != contract.compile_input_abi_schema
            ):
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ABI_ANALYSIS_MISMATCH")
    return results  # type: ignore[return-value]


def _canonical_captures(
    contract: MatmulCollectiveSurfaceDesignContract,
    results: tuple[SurfaceCompileWorkerResult, SurfaceCompileWorkerResult],
) -> tuple[CompileCaptureRecord, ...]:
    by_key = {
        (
            envelope.capture.scenario_name,
            envelope.capture.strategy,
            result.repetition,
        ): envelope.capture
        for result in results
        for envelope in result.captures
    }
    return tuple(
        by_key[(scenario.name, strategy, repetition)]
        for scenario in contract.scenarios
        for strategy in contract.strategies
        for repetition in (1, 2)
    )


def _manifest_entries(root: Path) -> tuple[SurfaceCompileManifestEntry, ...]:
    _preflight_existing_root(root)
    paths = tuple(
        sorted(path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json")
    )
    return tuple(
        SurfaceCompileManifestEntry(
            path=path.relative_to(root).as_posix(),
            size=path.stat().st_size,
            sha256=_file_sha256(path),
        )
        for path in paths
    )


def _validate_manifest(root: Path, manifest: SurfaceCompileManifest) -> None:
    expected = _manifest_entries(root)
    if manifest.artifacts != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_MANIFEST_MISMATCH")
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if observed != {"manifest.json", *(value.path for value in expected)}:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_CLOSED_WORLD_MISMATCH")


def _validate_ledger_closed_world(path: Path, run_id: str) -> None:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_INTEGRITY_MISMATCH")
        objects = tuple(
            connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_autoindex_%' ORDER BY type, name"
            )
        )
        if objects != (("table", "events"), ("table", "sqlite_sequence")):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_SCHEMA_MISMATCH")
        columns = tuple(
            (value[1], value[2], value[3], value[5])
            for value in connection.execute("PRAGMA table_info(events)")
        )
        if columns != (
            ("sequence", "INTEGER", 0, 1),
            ("run_id", "TEXT", 1, 0),
            ("state", "TEXT", 1, 0),
            ("timestamp_ns", "INTEGER", 1, 0),
            ("payload_sha256", "TEXT", 1, 0),
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_COLUMNS_MISMATCH")
        indexes = tuple(
            (value[1], value[2], value[3], value[4])
            for value in connection.execute("PRAGMA index_list(events)")
        )
        if indexes != (("sqlite_autoindex_events_1", 1, "u", 0),):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_INDEX_MISMATCH")
        rows = tuple(connection.execute("SELECT sequence, run_id FROM events ORDER BY sequence"))
        if rows != tuple((index, run_id) for index in range(1, 5)):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_RUN_INVENTORY_MISMATCH")
        if tuple(connection.execute("SELECT name, seq FROM sqlite_sequence")) != (("events", 4),):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_SEQUENCE_MISMATCH")


def compile_surface(
    root: Path,
    contract_path: Path,
    attempt_id: str,
) -> SurfaceCompileManifest:
    contract = _canonical_contract(contract_path)
    if len(attempt_id) != 64 or any(value not in "0123456789abcdef" for value in attempt_id):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_ID_INVALID")
    with _exclusive_attempt_lock(attempt_id):
        _require_safe_new_root(root)
        authority, source_blobs = _probe_execution_authority(contract_path, contract)
        claim_path = _claim_attempt(root, attempt_id, contract, authority.source.source_commit)
        claim_sha256 = _file_sha256(claim_path)
        _mkdir_exclusive_durable(root)
        try:
            identity = SurfaceCompileRunIdentity(
                attempt_id=attempt_id,
                design_id=contract.design_id,
                execution_authority_sha256=authority.authority_sha256,
                producer_output_root=str(root),
                attempt_claim_path=str(claim_path),
                attempt_claim_sha256=claim_sha256,
            )
            _write_json_exclusive(root / "attempt.json", {"attempt_id": attempt_id})
            _write_model_exclusive(root / "contract.json", contract)
            _write_model_exclusive(root / "run_identity.json", identity)
            _write_model_exclusive(root / "execution_authority.json", authority)
            _write_source_bundle(root, authority, source_blobs)
            ledger = EvidenceRun(root / "ledger.sqlite", attempt_id)
            ledger.create(
                {
                    "design_id": contract.design_id,
                    "execution_authority_sha256": authority.authority_sha256,
                    "attempt_claim_path": str(claim_path),
                    "attempt_claim_sha256": claim_sha256,
                }
            )
            ledger.transition(
                RunState.VERIFIED,
                {
                    "source_authority_sha256": authority.source.authority_sha256,
                    "executor_source_sha256": authority.executor_source_sha256,
                    "worker_source_sha256": authority.worker_source_sha256,
                    "verifier_source_sha256": authority.verifier_source_sha256,
                    "devices": [value.model_dump(mode="json") for value in authority.devices],
                },
            )
            requests = tuple(
                SurfaceCompileWorkerRequest(
                    attempt_id=attempt_id,
                    repetition=repetition,
                    invocation_nonce=hashlib.sha256(os.urandom(32)).hexdigest(),
                    authority_sha256=authority.authority_sha256,
                    contract=contract,
                )
                for repetition in (1, 2)
            )
            for request in requests:
                repetition_root = root / f"repetition-{request.repetition}"
                _mkdir_exclusive_durable(repetition_root)
                request_path = repetition_root / "request.json"
                _write_model_exclusive(request_path, request)
                _write_json_exclusive(
                    repetition_root / "STARTED.json",
                    {
                        "attempt_id": attempt_id,
                        "invocation_nonce": request.invocation_nonce,
                        "repetition": request.repetition,
                        "state": "started",
                    },
                )
                _launch_worker(root, request_path, request.repetition)
            results = _validate_worker_results(root, identity, authority, contract, requests)
            captures = _canonical_captures(contract, results)
            report = MatmulCollectiveSurfaceCompileReport(
                design_id=contract.design_id,
                source_authority_sha256=authority.source.authority_sha256,
                execution_authority_sha256=authority.authority_sha256,
                captures=captures,
            )
            validate_compile_capture_report(
                report,
                contract,
                authority.source,
                source_blobs,
                authority.authority_sha256,
            )
            ledger.transition(
                RunState.LOWERED,
                {
                    "arm_identities_sha256": payload_sha256(
                        {
                            f"{value.scenario_name}:{value.strategy.value}": (
                                value.physical_schedule_sha256,
                                value.pallas_source_sha256,
                            )
                            for value in captures
                            if value.repetition == 1
                        }
                    )
                },
            )
            _write_model_exclusive(root / "compile_report.json", report)
            ledger.transition(
                RunState.COMPILED,
                {"compile_report_sha256": report.report_sha256},
            )
            ledger.seal("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_SIDECARS")
            manifest = SurfaceCompileManifest(
                identity=identity,
                report_sha256=report.report_sha256,
                ledger_sha256=_file_sha256(root / "ledger.sqlite"),
                artifacts=_manifest_entries(root),
            )
            _write_model_exclusive(root / "manifest.json", manifest)
            _validate_manifest(root, manifest)
            return manifest
        except Exception as error:
            failure = root / "failure.json"
            if not failure.exists():
                _write_json_exclusive(
                    failure,
                    {"error_type": type(error).__name__, "error": str(error)},
                )
            raise


def verify_surface_compile(root: Path, contract_path: Path) -> SurfaceCompileManifest:
    contract = _canonical_contract(contract_path)
    _reject_links(root)
    if not root.is_dir() or (root / "failure.json").exists():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ATTEMPT_INCOMPLETE")
    _preflight_existing_root(root)
    recorded_contract = MatmulCollectiveSurfaceDesignContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if recorded_contract != contract:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_CONTRACT_MISMATCH")
    identity = SurfaceCompileRunIdentity.model_validate_json(
        (root / "run_identity.json").read_text()
    )
    authority = SurfaceCompileExecutionAuthority.model_validate_json(
        (root / "execution_authority.json").read_text()
    )
    if (
        identity.design_id != contract.design_id
        or identity.execution_authority_sha256 != authority.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_IDENTITY_MISMATCH")
    _validate_source_bundle_offline(root, authority, contract)
    expected_claim_path = _attempt_claim_path(contract, authority.source.source_commit)
    expected_claim_sha256 = _sha256_bytes(
        _json_bytes(
            _attempt_claim_payload(
                Path(identity.producer_output_root),
                identity.attempt_id,
                contract,
                authority.source.source_commit,
            )
        )
    )
    if (
        identity.attempt_claim_path != str(expected_claim_path)
        or identity.attempt_claim_sha256 != expected_claim_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_CLAIM_IDENTITY_MISMATCH")
    requests = tuple(
        SurfaceCompileWorkerRequest.model_validate_json(
            (root / f"repetition-{repetition}/request.json").read_text()
        )
        for repetition in (1, 2)
    )
    results = _validate_worker_results(root, identity, authority, contract, requests)
    report = MatmulCollectiveSurfaceCompileReport.model_validate_json(
        (root / "compile_report.json").read_text()
    )
    expected_report = MatmulCollectiveSurfaceCompileReport(
        design_id=contract.design_id,
        source_authority_sha256=authority.source.authority_sha256,
        execution_authority_sha256=authority.authority_sha256,
        captures=_canonical_captures(contract, results),
    )
    if report != expected_report:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_REPORT_REPLAY_MISMATCH")
    _validate_compile_report_offline(report, contract, authority)
    _validate_ledger_closed_world(root / "ledger.sqlite", identity.attempt_id)
    history = read_ledger_history(root / "ledger.sqlite", identity.attempt_id)
    if tuple(value.state for value in history) != _FINAL_STATES:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_STATE_MISMATCH")
    expected_payloads = (
        {
            "design_id": contract.design_id,
            "execution_authority_sha256": authority.authority_sha256,
            "attempt_claim_path": identity.attempt_claim_path,
            "attempt_claim_sha256": identity.attempt_claim_sha256,
        },
        {
            "source_authority_sha256": authority.source.authority_sha256,
            "executor_source_sha256": authority.executor_source_sha256,
            "worker_source_sha256": authority.worker_source_sha256,
            "verifier_source_sha256": authority.verifier_source_sha256,
            "devices": [value.model_dump(mode="json") for value in authority.devices],
        },
        {
            "arm_identities_sha256": payload_sha256(
                {
                    f"{value.scenario_name}:{value.strategy.value}": (
                        value.physical_schedule_sha256,
                        value.pallas_source_sha256,
                    )
                    for value in report.captures
                    if value.repetition == 1
                }
            )
        },
        {"compile_report_sha256": report.report_sha256},
    )
    if tuple(value.payload_sha256 for value in history) != tuple(
        payload_sha256(value) for value in expected_payloads
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LEDGER_PAYLOAD_MISMATCH")
    manifest = SurfaceCompileManifest.model_validate_json((root / "manifest.json").read_text())
    if (
        manifest.identity != identity
        or manifest.report_sha256 != report.report_sha256
        or manifest.ledger_sha256 != _file_sha256(root / "ledger.sqlite")
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_MANIFEST_AUTHORITY_MISMATCH")
    _validate_manifest(root, manifest)
    return manifest


def verify_surface_compile_live(root: Path, contract_path: Path) -> SurfaceCompileManifest:
    manifest = verify_surface_compile(root, contract_path)
    contract = _canonical_contract(contract_path)
    authority = SurfaceCompileExecutionAuthority.model_validate_json(
        (root / "execution_authority.json").read_text()
    )
    observed_authority, _ = _probe_execution_authority(contract_path, contract)
    if observed_authority != authority:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LIVE_REPLAY_AUTHORITY_MISMATCH")
    _validate_attempt_claim(
        root,
        manifest.identity.attempt_id,
        contract,
        authority.source.source_commit,
    )
    if _file_sha256(Path(manifest.identity.attempt_claim_path)) != (
        manifest.identity.attempt_claim_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_LIVE_CLAIM_HASH_MISMATCH")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    compile_command = commands.add_parser("compile")
    compile_command.add_argument("--root", required=True, type=Path)
    compile_command.add_argument("--contract", required=True, type=Path)
    compile_command.add_argument("--attempt-id", required=True)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--root", required=True, type=Path)
    verify_command.add_argument("--contract", required=True, type=Path)
    verify_live_command = commands.add_parser("verify-live")
    verify_live_command.add_argument("--root", required=True, type=Path)
    verify_live_command.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "compile":
        manifest = compile_surface(args.root, args.contract, args.attempt_id)
        print(manifest.model_dump_json())
    elif args.command == "verify":
        manifest = verify_surface_compile(args.root, args.contract)
        print(manifest.model_dump_json())
    elif args.command == "verify-live":
        manifest = verify_surface_compile_live(args.root, args.contract)
        print(manifest.model_dump_json())


if __name__ == "__main__":
    main()
