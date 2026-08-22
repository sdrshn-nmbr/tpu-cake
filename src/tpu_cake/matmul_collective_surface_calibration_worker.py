from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import SourceFileContract
from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import array_sha256, model_identity_sha256
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.matmul_collective_surface_calibration_evidence import (
    MatmulCollectiveSurfaceCalibrationEvidence,
    SurfaceCalibrationCallSample,
    SurfaceCalibrationOutputGate,
    SurfaceCalibrationResidentPair,
    SurfaceCalibrationTimingInput,
    SurfaceCalibrationWarmupExecution,
    validate_surface_calibration_evidence,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    MatmulCollectiveSurfaceCalibrationProtocol,
)
from tpu_cake.matmul_collective_surface_correctness import (
    correctness_sentinel_coordinates,
    make_correctness_operand_shard,
)
from tpu_cake.matmul_collective_surface_correctness_evidence import (
    MatmulCollectiveSurfaceCorrectnessEvidence,
    SurfaceCompileContinuityEvidence,
    SurfaceCorrectnessCaseEvidence,
    SurfaceCorrectnessInputCase,
    SurfaceCorrectnessSentinel,
    SurfaceCorrectnessShardIdentity,
    SurfaceCorrectnessSlice,
)
from tpu_cake.matmul_collective_surface_correctness_executor import (
    SurfaceCorrectnessManifest,
    SurfaceCorrectnessPhaseReceipt,
)
from tpu_cake.matmul_collective_surface_correctness_oracle import make_correctness_oracle
from tpu_cake.matmul_collective_surface_correctness_worker import (
    _canonical_slice,
    _device_sentinel_hex,
    _error_metrics,
    _save_array_exclusive,
    _shard_index,
    _validate_output,
    _verify_resident_sentinels,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceScenario,
)
from tpu_cake.matmul_collective_surface_runner import (
    SURFACE_EXECUTABLE_DEPENDENCIES,
    _source_subprocess_environment,
    derive_surface_input_identities,
    make_compile_capture_record,
)
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.runner import MatmulCollectiveStrategy, _runtime_identity
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule

CALIBRATION_WORKER_SCHEMA = "matmul-collective-surface-calibration-worker-v1"
CALIBRATION_EXECUTION_AUTHORITY_SCHEMA = (
    "matmul-collective-surface-calibration-execution-authority-v1"
)
CALIBRATION_ATTEMPT_CLAIM_SCHEMA = "matmul-collective-surface-calibration-attempt-claim-v1"
CALIBRATION_EXECUTOR_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_calibration_executor.py"
CALIBRATION_WORKER_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_calibration_worker.py"
CALIBRATION_VERIFIER_SOURCE_PATH = "src/tpu_cake/matmul_collective_surface_calibration_verifier.py"
CALIBRATION_EXECUTABLE_DEPENDENCIES = tuple(
    sorted(
        {
            *SURFACE_EXECUTABLE_DEPENDENCIES,
            "contracts/matmul-collective-surface-calibration-v1.json",
            "contracts/matmul-collective-surface-correctness-v1.json",
            "contracts/matmul-collective-surface-design-v1.json",
            "tpu_cake/matmul_collective_surface_calibration_archive.py",
            "tpu_cake/matmul_collective_surface_calibration_evidence.py",
            "tpu_cake/matmul_collective_surface_calibration_protocol.py",
            "tpu_cake/matmul_collective_surface_calibration_seal.py",
            "tpu_cake/matmul_collective_surface_calibration_executor.py",
            "tpu_cake/matmul_collective_surface_calibration_worker.py",
            "tpu_cake/matmul_collective_surface_calibration_verifier.py",
            "tpu_cake/matmul_collective_surface_correctness.py",
            "tpu_cake/matmul_collective_surface_correctness_evidence.py",
            "tpu_cake/matmul_collective_surface_correctness_executor.py",
            "tpu_cake/matmul_collective_surface_correctness_oracle.py",
            "tpu_cake/matmul_collective_surface_correctness_protocol.py",
            "tpu_cake/matmul_collective_surface_correctness_worker.py",
        }
    )
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _write_model_exclusive(path: Path, value: BaseModel) -> None:
    _write_bytes_exclusive(
        path,
        _json_bytes(value.model_dump(mode="json", exclude_computed_fields=True)),
    )


class SurfaceCalibrationSourceAuthority(BaseModel):
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
    def main_and_inventory_are_exact(self) -> SurfaceCalibrationSourceAuthority:
        paths = tuple(value.path for value in self.dependencies)
        if not (
            self.source_commit == self.origin_main_commit == self.remote_main_commit
            and paths == CALIBRATION_EXECUTABLE_DEPENDENCIES
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_AUTHORITY_INVALID")
        return self

    @computed_field
    @property
    def authority_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCalibrationDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0, le=7)
    process_index: Literal[0] = 0
    platform: Literal["tpu"] = "tpu"
    device_kind: Literal["TPU7x"] = "TPU7x"


class SurfaceCalibrationExecutionAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-execution-authority-v1"] = (
        CALIBRATION_EXECUTION_AUTHORITY_SCHEMA
    )
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: SurfaceCalibrationSourceAuthority
    executor_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    devices: tuple[SurfaceCalibrationDevice, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def device_inventory_is_exact(self) -> SurfaceCalibrationExecutionAuthority:
        if tuple(value.id for value in self.devices) != tuple(range(8)):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_DEVICE_INVENTORY_MISMATCH")
        return self

    @computed_field
    @property
    def authority_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCalibrationAttemptClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-attempt-claim-v1"] = (
        CALIBRATION_ATTEMPT_CLAIM_SCHEMA
    )
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    permanent_claim_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    output_root: str
    state: Literal["claimed"] = "claimed"


class SurfaceCalibrationRunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_claim_path: str
    attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: str
    parent_correctness_root: str
    compilation_cache_path: str


class SurfaceCalibrationWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-worker-v1"] = (
        CALIBRATION_WORKER_SCHEMA
    )
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: str
    parent_correctness_root: str
    compilation_cache_path: str
    protocol_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol: MatmulCollectiveSurfaceCalibrationProtocol
    design: MatmulCollectiveSurfaceDesignContract

    @model_validator(mode="after")
    def contract_binding_is_exact(self) -> SurfaceCalibrationWorkerRequest:
        if self.protocol.design_id != self.design.design_id:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_REQUEST_DESIGN_MISMATCH")
        return self


class SurfaceCalibrationWorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: MatmulCollectiveSurfaceCalibrationEvidence


@dataclass(frozen=True)
class _HostShardCapture:
    global_slice: tuple[slice, slice]
    local_shape: tuple[int, int]
    host_callback_payload_nbytes: int
    host_callback_payload_sha256: str
    sentinel_coordinates: tuple[tuple[int, int], ...]
    expected_sentinel_hex: tuple[str, ...]


@dataclass(frozen=True)
class _CompiledArm:
    strategy: MatmulCollectiveStrategy
    executable: Any
    mesh: Mesh
    continuity: SurfaceCompileContinuityEvidence


@dataclass(frozen=True)
class _ResidentScenario:
    scenario: MatmulCollectiveSurfaceScenario
    lhs: jax.Array
    rhs: jax.Array
    lhs_shards: tuple[SurfaceCorrectnessShardIdentity, ...]
    rhs_shards: tuple[SurfaceCorrectnessShardIdentity, ...]
    oracle: np.ndarray
    timing_input: SurfaceCalibrationTimingInput
    pair: SurfaceCalibrationResidentPair


def _git_blob_path(path: str) -> str:
    return f"src/{path}" if path.startswith("tpu_cake/") else path


def _read_committed_calibration_blobs(
    repository_root: Path,
    source_commit: str,
) -> dict[str, bytes]:
    values = {}
    for path in (*CALIBRATION_EXECUTABLE_DEPENDENCIES, "uv.lock"):
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
                f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_COMMIT_UNAVAILABLE path={path}"
            ) from error
    return values


def _calibration_runtime_identity() -> dict[str, str]:
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


def validate_calibration_source_authority(
    authority: SurfaceCalibrationSourceAuthority,
    design: MatmulCollectiveSurfaceDesignContract,
    source_blobs: dict[str, bytes],
) -> None:
    authority = SurfaceCalibrationSourceAuthority.model_validate(
        authority.model_dump(mode="python", exclude_computed_fields=True)
    )
    paths = tuple(value.path for value in authority.dependencies)
    if (
        paths != CALIBRATION_EXECUTABLE_DEPENDENCIES
        or tuple(source_blobs) != (*paths, "uv.lock")
        or authority.source_root != design.compilation_source_root
        or authority.remote_url != design.source_remote_url
        or authority.runtime != _calibration_runtime_identity()
        or {key: authority.runtime.get(key) for key in design.runtime} != design.runtime
        or set(authority.runtime) != {*design.runtime, "numpy", "ml_dtypes"}
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_AUTHORITY_MISMATCH")
    expected = {value.path: value.sha256 for value in authority.dependencies}
    expected["uv.lock"] = authority.uv_lock_sha256
    if any(_sha256_bytes(source_blobs[path]) != expected[path] for path in source_blobs):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_HASH_MISMATCH")
    if (
        _read_committed_calibration_blobs(Path(authority.source_root), authority.source_commit)
        != source_blobs
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_COMMIT_MISMATCH")


def _validate_source_checkout(branch: str, status: str) -> None:
    if branch != "main":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_BRANCH_MISMATCH")
    if status:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_DIRTY")


def capture_calibration_source_authority(
    repository_root: Path,
    design: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCalibrationSourceAuthority, dict[str, bytes]]:
    repository_root = repository_root.resolve(strict=True)
    if repository_root != Path(design.compilation_source_root):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SOURCE_ROOT_MISMATCH")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository_root,
            env=_source_subprocess_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    _validate_source_checkout(
        git("branch", "--show-current"),
        git("status", "--porcelain=v1"),
    )
    source_commit = git("rev-parse", "HEAD")
    remote = subprocess.run(
        ["/usr/bin/git", "ls-remote", design.source_remote_url, "refs/heads/main"],
        cwd="/",
        env=_source_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(remote) != 2 or remote[1] != "refs/heads/main":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_REMOTE_MAIN_INVALID")
    blobs = _read_committed_calibration_blobs(repository_root, source_commit)
    authority = SurfaceCalibrationSourceAuthority(
        source_commit=source_commit,
        origin_main_commit=git("rev-parse", "origin/main"),
        remote_main_commit=remote[0],
        runtime=_calibration_runtime_identity(),
        uv_lock_sha256=_sha256_bytes(blobs["uv.lock"]),
        dependencies=tuple(
            SourceFileContract(path=path, sha256=_sha256_bytes(blobs[path]))
            for path in CALIBRATION_EXECUTABLE_DEPENDENCIES
        ),
    )
    validate_calibration_source_authority(authority, design, blobs)
    return authority, blobs


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
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_METADATA_REDIRECT code={code}")


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
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_METADATA_HEADER_MISSING")
        payload = response.read(4097)
    if len(payload) > 4096:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_METADATA_TOO_LARGE")
    return payload.decode().strip()


def _compiler_environment(design: MatmulCollectiveSurfaceDesignContract) -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in design.compiler_environment}
    forbidden = {
        key: value
        for key, value in os.environ.items()
        if (key == "TPU_LIBRARY_PATH" or key.startswith(("JAX_", "XLA_", "PJRT_", "LIBTPU_")))
        and key not in design.compiler_environment
        and key != "JAX_COMPILATION_CACHE_DIR"
    }
    if observed != design.compiler_environment or forbidden:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ENVIRONMENT_MISMATCH")
    return dict(design.compiler_environment)


def validate_calibration_execution_authority(
    authority: SurfaceCalibrationExecutionAuthority,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
    source_blobs: dict[str, bytes],
) -> None:
    authority = SurfaceCalibrationExecutionAuthority.model_validate(
        authority.model_dump(mode="python", exclude_computed_fields=True)
    )
    validate_calibration_source_authority(authority.source, design, source_blobs)
    if (
        authority.protocol_id != protocol.protocol_id
        or authority.protocol_file_sha256
        != _sha256_bytes(source_blobs["contracts/matmul-collective-surface-calibration-v1.json"])
        or authority.design_id != design.design_id
        or authority.design_file_sha256
        != _sha256_bytes(source_blobs["contracts/matmul-collective-surface-design-v1.json"])
        or authority.executor_source_sha256
        != _sha256_bytes(source_blobs[CALIBRATION_EXECUTOR_SOURCE_PATH.removeprefix("src/")])
        or authority.worker_source_sha256
        != _sha256_bytes(source_blobs[CALIBRATION_WORKER_SOURCE_PATH.removeprefix("src/")])
        or authority.verifier_source_sha256
        != _sha256_bytes(source_blobs[CALIBRATION_VERIFIER_SOURCE_PATH.removeprefix("src/")])
        or authority.project != design.project
        or authority.zone != design.zone
        or authority.hostname != design.hostname
        or authority.numeric_project_id != design.numeric_project_id
        or authority.instance_id != design.instance_id
        or authority.instance_hostname != design.instance_hostname
        or authority.machine_type != design.machine_type
        or authority.cpu_platform != design.cpu_platform
        or authority.compiler_environment != design.compiler_environment
        or tuple(value.id for value in authority.devices) != design.device_ids
        or any(
            value.process_index != design.device_process_index
            or value.platform != design.backend
            or value.device_kind != design.device_kind
            for value in authority.devices
        )
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_EXECUTION_AUTHORITY_MISMATCH")


def _validate_loaded_tpu_cake_sources(
    repository_root: Path,
    source_blobs: dict[str, bytes],
) -> None:
    expected = {path: source_blobs[path] for path in CALIBRATION_EXECUTABLE_DEPENDENCIES}
    source_root = (repository_root / "src").resolve()
    observed: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if name != "tpu_cake" and not name.startswith("tpu_cake."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_LOADED_SOURCE_MISSING")
        path = Path(module_file)
        try:
            relative = path.resolve().relative_to(source_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "MATMUL_COLLECTIVE_SURFACE_CALIBRATION_LOADED_SOURCE_OUTSIDE_ROOT"
            ) from error
        if path.is_symlink() or relative not in expected or path.read_bytes() != expected[relative]:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_LOADED_SOURCE_MISMATCH")
        observed.add(relative)
    worker_relative = CALIBRATION_WORKER_SOURCE_PATH.removeprefix("src/")
    running = Path(__file__)
    if (
        running.is_symlink()
        or running.resolve() != (repository_root / CALIBRATION_WORKER_SOURCE_PATH).resolve()
        or running.read_bytes() != expected[worker_relative]
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_RUNNING_WORKER_MISMATCH")
    if {"tpu_cake/__init__.py", worker_relative} - (observed | {worker_relative}):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_LOADED_SOURCE_INCOMPLETE")


def capture_calibration_execution_authority(
    repository_root: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCalibrationExecutionAuthority, dict[str, bytes]]:
    source, source_blobs = capture_calibration_source_authority(repository_root, design)
    authority = SurfaceCalibrationExecutionAuthority(
        protocol_id=protocol.protocol_id,
        protocol_file_sha256=_sha256_bytes(
            source_blobs["contracts/matmul-collective-surface-calibration-v1.json"]
        ),
        design_id=design.design_id,
        design_file_sha256=_sha256_bytes(
            source_blobs["contracts/matmul-collective-surface-design-v1.json"]
        ),
        source=source,
        executor_source_sha256=_sha256_bytes(
            source_blobs[CALIBRATION_EXECUTOR_SOURCE_PATH.removeprefix("src/")]
        ),
        worker_source_sha256=_sha256_bytes(
            source_blobs[CALIBRATION_WORKER_SOURCE_PATH.removeprefix("src/")]
        ),
        verifier_source_sha256=_sha256_bytes(
            source_blobs[CALIBRATION_VERIFIER_SOURCE_PATH.removeprefix("src/")]
        ),
        project=_metadata("project/project-id"),
        zone=_metadata("instance/zone").rsplit("/", maxsplit=1)[-1],
        hostname=_metadata("instance/name"),
        numeric_project_id=_metadata("project/numeric-project-id"),
        instance_id=_metadata("instance/id"),
        instance_hostname=_metadata("instance/hostname"),
        machine_type=_metadata("instance/machine-type").rsplit("/", maxsplit=1)[-1],
        cpu_platform=_metadata("instance/cpu-platform"),
        compiler_environment=_compiler_environment(design),
        devices=tuple(
            SurfaceCalibrationDevice(
                id=int(device.id),
                process_index=int(device.process_index),
                platform=str(device.platform),
                device_kind=str(device.device_kind),
            )
            for device in jax.devices()
        ),
    )
    validate_calibration_execution_authority(authority, protocol, design, source_blobs)
    _validate_loaded_tpu_cake_sources(repository_root, source_blobs)
    return authority, source_blobs


class _OperandCallback:
    def __init__(
        self,
        *,
        protocol_id: str,
        scenario: MatmulCollectiveSurfaceScenario,
        role: str,
    ) -> None:
        self.protocol_id = protocol_id
        self.scenario = scenario
        self.role = role
        self.global_shape = (scenario.m, scenario.k) if role == "lhs" else (scenario.k, scenario.n)
        self.captures: dict[tuple[tuple[int, int], tuple[int, int]], _HostShardCapture] = {}

    def __call__(self, index: Any) -> np.ndarray:
        global_slice = _canonical_slice(index, self.global_shape)
        local_k = self.scenario.k // 8
        k_slice = global_slice[1] if self.role == "lhs" else global_slice[0]
        full_slice = global_slice[0] if self.role == "lhs" else global_slice[1]
        full_size = self.scenario.m if self.role == "lhs" else self.scenario.n
        if (
            full_slice.start != 0
            or full_slice.stop != full_size
            or k_slice.stop - k_slice.start != local_k
            or k_slice.start % local_k
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CALLBACK_SHARDING_MISMATCH")
        shard_index = k_slice.start // local_k
        value = make_correctness_operand_shard(
            "signed-periodic",
            self.role,
            m=self.scenario.m,
            k=self.scenario.k,
            n=self.scenario.n,
            k_start=k_slice.start,
            k_stop=k_slice.stop,
        )
        expected_shape = tuple(item.stop - item.start for item in global_slice)
        if value.shape != expected_shape or value.dtype != np.dtype(ml_dtypes.bfloat16):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CALLBACK_PAYLOAD_INVALID")
        coordinates = correctness_sentinel_coordinates(
            "signed-periodic",
            self.role,
            protocol_id=self.protocol_id,
            scenario_name=self.scenario.name,
            m=self.scenario.m,
            k=self.scenario.k,
            n=self.scenario.n,
            device_id=shard_index,
        )
        local_coordinates = tuple(
            (first - global_slice[0].start, second - global_slice[1].start)
            for first, second in coordinates
        )
        expected = np.asarray(
            value[
                np.asarray(tuple(item[0] for item in local_coordinates)),
                np.asarray(tuple(item[1] for item in local_coordinates)),
            ]
        )
        key = tuple((item.start, item.stop) for item in global_slice)
        if key in self.captures:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CALLBACK_SHARD_REPEATED")
        self.captures[key] = _HostShardCapture(
            global_slice=global_slice,
            local_shape=expected_shape,
            host_callback_payload_nbytes=value.nbytes,
            host_callback_payload_sha256=_sha256_bytes(
                np.ascontiguousarray(value).tobytes(order="C")
            ),
            sentinel_coordinates=coordinates,
            expected_sentinel_hex=tuple(
                expected[index : index + 1].tobytes().hex() for index in range(expected.size)
            ),
        )
        return value


def _materialize_timing_operand(
    protocol_id: str,
    scenario: MatmulCollectiveSurfaceScenario,
    role: str,
    mesh: Mesh,
) -> tuple[jax.Array, tuple[SurfaceCorrectnessShardIdentity, ...]]:
    spec = PartitionSpec(None, "t") if role == "lhs" else PartitionSpec("t", None)
    callback = _OperandCallback(protocol_id=protocol_id, scenario=scenario, role=role)
    value = jax.make_array_from_callback(
        callback.global_shape,
        NamedSharding(mesh, spec),
        callback,
    )
    if len(callback.captures) != 8:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CALLBACK_INVENTORY_MISMATCH")
    identities = []
    addressable = tuple(value.addressable_shards)
    if len(addressable) != 8:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SHARD_INVENTORY_MISMATCH")
    for shard in addressable:
        device_id = int(shard.device.id)
        global_slice = _shard_index(shard.index, callback.global_shape)
        key = tuple((item.start, item.stop) for item in global_slice)
        capture = callback.captures.get(key)
        if capture is None or int(shard.device.process_index) != 0:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_DEVICE_SHARD_MISMATCH")
        local_k = scenario.k // 8
        k_slice = global_slice[1] if role == "lhs" else global_slice[0]
        shard_index = k_slice.start // local_k
        if shard_index != device_id:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_DEVICE_SLICE_MISMATCH")
        local_coordinates = tuple(
            (first - global_slice[0].start, second - global_slice[1].start)
            for first, second in capture.sentinel_coordinates
        )
        observed = _device_sentinel_hex(shard.data, local_coordinates)
        identities.append(
            SurfaceCorrectnessShardIdentity(
                role=role,
                shard_index=shard_index,
                device_id=device_id,
                process_index=0,
                global_shape=callback.global_shape,
                sharding=(
                    "PartitionSpec(None, 't')" if role == "lhs" else "PartitionSpec('t', None)"
                ),
                global_slice=tuple(
                    SurfaceCorrectnessSlice(start=item.start, stop=item.stop)
                    for item in global_slice
                ),
                local_shape=capture.local_shape,
                host_callback_payload_nbytes=capture.host_callback_payload_nbytes,
                host_callback_payload_sha256=capture.host_callback_payload_sha256,
                sentinels=tuple(
                    SurfaceCorrectnessSentinel(
                        ordinal=ordinal,
                        global_coordinate=coordinate,
                        local_coordinate=local_coordinate,
                        expected_bfloat16_hex=expected,
                        observed_bfloat16_hex=actual,
                    )
                    for ordinal, (coordinate, local_coordinate, expected, actual) in enumerate(
                        zip(
                            capture.sentinel_coordinates,
                            local_coordinates,
                            capture.expected_sentinel_hex,
                            observed,
                            strict=True,
                        )
                    )
                ),
            )
        )
    identities.sort(key=lambda item: item.device_id)
    return value, tuple(identities)


def _compile_arm(
    root: Path,
    request: SurfaceCalibrationWorkerRequest,
    parent: SurfaceCompileContinuityEvidence,
    scenario: MatmulCollectiveSurfaceScenario,
    strategy: MatmulCollectiveStrategy,
) -> _CompiledArm:
    distributed = distributed_matmul_schedule(
        mesh_size=request.design.mesh_size,
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
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_MESH_AXIS_MISMATCH")
    abstract_inputs = (
        jax.ShapeDtypeStruct(
            (scenario.m, scenario.k),
            jnp.bfloat16,
            sharding=NamedSharding(mesh, PartitionSpec(None, "t")),
        ),
        jax.ShapeDtypeStruct(
            (scenario.k, scenario.n),
            jnp.bfloat16,
            sharding=NamedSharding(mesh, PartitionSpec("t", None)),
        ),
    )
    lowered = executable.lower(*abstract_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiled = lowered.compile()
    compiler_hlo = compiled.as_text()
    if not isinstance(compiler_hlo, str) or not compiler_hlo:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_COMPILER_HLO_UNAVAILABLE")
    input_identity = next(
        value
        for value in derive_surface_input_identities(request.design)
        if value.scenario_name == scenario.name
    )
    fresh = make_compile_capture_record(
        scenario_name=scenario.name,
        strategy=strategy,
        repetition=1,
        input_contract_sha256=input_identity.input_contract_sha256,
        distributed_schedule_sha256=schedule_sha256(distributed),
        physical_schedule_sha256=plan.schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
    )
    stablehlo_path = f"continuity/{scenario.name}/{strategy.value}/stablehlo.txt"
    compiler_hlo_path = f"continuity/{scenario.name}/{strategy.value}/compiler_hlo.txt"
    _write_bytes_exclusive(root / stablehlo_path, fresh.stablehlo.encode())
    _write_bytes_exclusive(root / compiler_hlo_path, fresh.compiler_hlo.encode())
    continuity = SurfaceCompileContinuityEvidence(
        scenario_name=scenario.name,
        strategy=strategy,
        stablehlo_path=stablehlo_path,
        stablehlo_file_sha256=_file_sha256(root / stablehlo_path),
        compiler_hlo_path=compiler_hlo_path,
        compiler_hlo_file_sha256=_file_sha256(root / compiler_hlo_path),
        parent_distributed_schedule_sha256=parent.observed_distributed_schedule_sha256,
        observed_distributed_schedule_sha256=fresh.distributed_schedule_sha256,
        parent_physical_schedule_sha256=parent.observed_physical_schedule_sha256,
        observed_physical_schedule_sha256=fresh.physical_schedule_sha256,
        parent_pallas_source_sha256=parent.observed_pallas_source_sha256,
        observed_pallas_source_sha256=fresh.pallas_source_sha256,
        parent_semantic_stablehlo_sha256=parent.observed_semantic_stablehlo_sha256,
        observed_semantic_stablehlo_sha256=fresh.semantic_stablehlo_sha256,
        parent_semantic_compiler_hlo_sha256=parent.observed_semantic_compiler_hlo_sha256,
        observed_semantic_compiler_hlo_sha256=fresh.semantic_compiler_hlo_sha256,
    )
    return _CompiledArm(strategy=strategy, executable=compiled, mesh=mesh, continuity=continuity)


def _validate_all_meshes(
    compiled: dict[tuple[str, MatmulCollectiveStrategy], _CompiledArm],
) -> None:
    if len(compiled) != 32:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_COMPILED_INVENTORY_MISMATCH")
    first = next(iter(compiled.values())).mesh
    expected = tuple(int(value.id) for value in first.devices.flat)
    if expected != tuple(range(8)) or any(
        tuple(int(value.id) for value in arm.mesh.devices.flat) != expected
        or arm.mesh.axis_names != first.axis_names
        for arm in compiled.values()
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_COMPILED_MESH_MISMATCH")


def _compile_all_arms(
    root: Path,
    request: SurfaceCalibrationWorkerRequest,
    parent: MatmulCollectiveSurfaceCorrectnessEvidence,
) -> tuple[
    dict[tuple[str, MatmulCollectiveStrategy], _CompiledArm],
    tuple[SurfaceCompileContinuityEvidence, ...],
]:
    scenarios = {value.name: value for value in request.design.calibration_scenarios}
    compiled = {}
    continuity = []
    for scenario_name in request.protocol.scenarios:
        scenario = scenarios[scenario_name]
        for strategy in request.protocol.strategies:
            arm = _compile_arm(
                root,
                request,
                _parent_continuity(parent, scenario_name, strategy),
                scenario,
                strategy,
            )
            compiled[(scenario_name, strategy)] = arm
            continuity.append(arm.continuity)
    _validate_all_meshes(compiled)
    return compiled, tuple(continuity)


def _load_parent_correctness(
    root: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> MatmulCollectiveSurfaceCorrectnessEvidence:
    root = root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_ROOT_INVALID")
    for path in root.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_LINK_INVALID")
    required_hashes = _parent_required_file_hashes(protocol)
    if any(_file_sha256(root / name) != expected for name, expected in required_hashes.items()):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_FILE_HASH_MISMATCH")
    evidence = MatmulCollectiveSurfaceCorrectnessEvidence.model_validate_json(
        (root / "evidence.json").read_text()
    )
    receipt = SurfaceCorrectnessPhaseReceipt.model_validate_json(
        (root / "receipt.json").read_text()
    )
    manifest = SurfaceCorrectnessManifest.model_validate_json((root / "manifest.json").read_text())
    identity = json.loads((root / "run_identity.json").read_text())
    _validate_parent_models(evidence, receipt, manifest, identity, protocol)
    return evidence


def _parent_required_file_hashes(
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> dict[str, str]:
    parent = protocol.correctness_parent
    return {
        "manifest.json": parent.manifest_file_sha256,
        "evidence.json": parent.evidence_file_sha256,
        "receipt.json": parent.receipt_file_sha256,
        "phase_ledger.json": parent.phase_ledger_file_sha256,
        "ledger.sqlite": parent.ledger_file_sha256,
        "run_identity.json": parent.run_identity_file_sha256,
        "attempt_claim.json": parent.attempt_claim_file_sha256,
        "execution_authority.json": parent.execution_authority_file_sha256,
    }


def _validate_parent_models(
    evidence: MatmulCollectiveSurfaceCorrectnessEvidence,
    receipt: SurfaceCorrectnessPhaseReceipt,
    manifest: SurfaceCorrectnessManifest,
    identity: dict[str, Any],
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> None:
    parent = protocol.correctness_parent
    if (
        evidence.evidence_sha256 != parent.evidence_sha256
        or evidence.protocol_id != parent.protocol_id
        or evidence.split.value != "calibration"
        or receipt.receipt_sha256 != parent.receipt_sha256
        or receipt.attempt_id != parent.attempt_id
        or receipt.evidence_sha256 != parent.evidence_sha256
        or manifest.evidence_sha256 != parent.evidence_sha256
        or identity.get("attempt_id") != parent.attempt_id
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_AUTHORITY_MISMATCH")


def _parent_case(
    evidence: MatmulCollectiveSurfaceCorrectnessEvidence,
    scenario_name: str,
    strategies: tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy],
) -> tuple[SurfaceCorrectnessCaseEvidence, dict[MatmulCollectiveStrategy, str]]:
    matches = tuple(
        value
        for value in evidence.cases
        if value.input.scenario_name == scenario_name and value.input.pattern == "signed-periodic"
    )
    if len(matches) != 1:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_CASE_INVENTORY_MISMATCH")
    case = matches[0]
    hashes: dict[MatmulCollectiveStrategy, str] = {}
    for strategy in strategies:
        outputs = tuple(
            value.output.array_sha256 for value in case.executions if value.strategy is strategy
        )
        if len(outputs) != 2 or len(set(outputs)) != 1:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_OUTPUT_MISMATCH")
        hashes[strategy] = outputs[0]
    if any(value != case.oracle.array_sha256 for value in hashes.values()):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_ORACLE_MISMATCH")
    return case, hashes


def _parent_continuity(
    evidence: MatmulCollectiveSurfaceCorrectnessEvidence,
    scenario_name: str,
    strategy: MatmulCollectiveStrategy,
) -> SurfaceCompileContinuityEvidence:
    values = tuple(
        value
        for value in evidence.continuity
        if value.scenario_name == scenario_name and value.strategy is strategy
    )
    if len(values) != 1:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_CONTINUITY_MISMATCH")
    return values[0]


def _materialize_all_inputs(
    root: Path,
    request: SurfaceCalibrationWorkerRequest,
    parent: MatmulCollectiveSurfaceCorrectnessEvidence,
    compiled: dict[tuple[str, MatmulCollectiveStrategy], _CompiledArm],
) -> tuple[dict[str, _ResidentScenario], tuple[SurfaceCalibrationTimingInput, ...]]:
    scenarios = {value.name: value for value in request.design.calibration_scenarios}
    residents = {}
    inputs = []
    first_strategy = request.protocol.strategies[0]
    for scenario_name in request.protocol.scenarios:
        scenario = scenarios[scenario_name]
        parent_case, parent_hashes = _parent_case(
            parent, scenario_name, request.protocol.strategies
        )
        mesh = compiled[(scenario_name, first_strategy)].mesh
        lhs, lhs_shards = _materialize_timing_operand(
            request.protocol.correctness_parent.protocol_id, scenario, "lhs", mesh
        )
        rhs, rhs_shards = _materialize_timing_operand(
            request.protocol.correctness_parent.protocol_id, scenario, "rhs", mesh
        )
        regenerated_input = SurfaceCorrectnessInputCase(
            scenario_name=scenario_name,
            pattern="signed-periodic",
            protocol_id=request.protocol.correctness_parent.protocol_id,
            pattern_contract_sha256=parent_case.input.pattern_contract_sha256,
            lhs_shards=lhs_shards,
            rhs_shards=rhs_shards,
        )
        if regenerated_input != parent_case.input:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PARENT_INPUT_MISMATCH")
        oracle_value = make_correctness_oracle(
            "signed-periodic", m=scenario.m, k=scenario.k, n=scenario.n
        )
        if array_sha256(oracle_value) != parent_case.oracle.array_sha256:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_REGENERATED_ORACLE_MISMATCH")
        oracle = _save_array_exclusive(root, f"oracles/{scenario_name}.npy", oracle_value)
        timing_input = SurfaceCalibrationTimingInput(
            scenario_name=scenario_name,
            parent_case_sha256=parent_case.case_sha256,
            parent_xla_array_sha256=parent_hashes[request.protocol.strategies[0]],
            parent_pallas_array_sha256=parent_hashes[request.protocol.strategies[1]],
            input=regenerated_input,
            oracle=oracle,
        )
        pair = SurfaceCalibrationResidentPair(
            scenario_name=scenario_name,
            xla_compile_record_sha256=compiled[
                (scenario_name, request.protocol.strategies[0])
            ].continuity.compile_record_sha256,
            pallas_compile_record_sha256=compiled[
                (scenario_name, request.protocol.strategies[1])
            ].continuity.compile_record_sha256,
            invocation_nonce=request.invocation_nonce,
            worker_pid=os.getpid(),
        )
        residents[scenario_name] = _ResidentScenario(
            scenario=scenario,
            lhs=lhs,
            rhs=rhs,
            lhs_shards=lhs_shards,
            rhs_shards=rhs_shards,
            oracle=oracle_value,
            timing_input=timing_input,
            pair=pair,
        )
        inputs.append(timing_input)
    if len(residents) != 16:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_RESIDENT_INVENTORY_MISMATCH")
    return residents, tuple(inputs)


def _run_and_wait(arm: _CompiledArm, resident: _ResidentScenario) -> tuple[jax.Array, int, int]:
    start = time.perf_counter_ns()
    output = arm.executable(resident.lhs, resident.rhs)
    output.block_until_ready()
    stop = time.perf_counter_ns()
    if stop <= start:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CLOCK_NONPOSITIVE")
    return output, start, stop


def _verify_all_resident_sentinels(residents: dict[str, _ResidentScenario]) -> None:
    for resident in residents.values():
        _verify_resident_sentinels(resident.lhs, resident.lhs_shards)
        _verify_resident_sentinels(resident.rhs, resident.rhs_shards)


def _run_output_gates(
    root: Path,
    request: SurfaceCalibrationWorkerRequest,
    compiled: dict[tuple[str, MatmulCollectiveStrategy], _CompiledArm],
    residents: dict[str, _ResidentScenario],
    phase: Literal["before_timing", "after_timing"],
) -> tuple[SurfaceCalibrationOutputGate, ...]:
    gates = []
    for scenario_name in request.protocol.scenarios:
        resident = residents[scenario_name]
        for strategy in request.protocol.strategies:
            output, start, _ = _run_and_wait(compiled[(scenario_name, strategy)], resident)
            _validate_output(output, resident.scenario)
            candidate = np.ascontiguousarray(np.asarray(jax.device_get(output)), dtype=np.float32)
            saved = _save_array_exclusive(
                root,
                f"outputs/{scenario_name}/{strategy.value}-{phase}.npy",
                candidate,
            )
            expected_parent = (
                resident.timing_input.parent_xla_array_sha256
                if strategy is request.protocol.strategies[0]
                else resident.timing_input.parent_pallas_array_sha256
            )
            if saved.array_sha256 != expected_parent:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_OUTPUT_PARENT_MISMATCH")
            mismatches, maximum_absolute, maximum_normalized = _error_metrics(
                candidate,
                resident.oracle,
                absolute_tolerance=request.protocol.absolute_tolerance,
                relative_tolerance=request.protocol.relative_tolerance,
            )
            stop = time.perf_counter_ns()
            if stop <= start:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_GATE_CLOCK_NONPOSITIVE")
            gates.append(
                SurfaceCalibrationOutputGate(
                    scenario_name=scenario_name,
                    strategy=strategy,
                    phase=phase,
                    resident_pair_sha256=resident.pair.resident_pair_sha256,
                    invocation_nonce=request.invocation_nonce,
                    worker_pid=os.getpid(),
                    start_ns=start,
                    stop_ns=stop,
                    oracle_array_sha256=resident.timing_input.oracle.array_sha256,
                    output=saved,
                    mismatched_element_count=mismatches,
                    maximum_absolute_error=maximum_absolute,
                    maximum_normalized_error=maximum_normalized,
                )
            )
    _verify_all_resident_sentinels(residents)
    return tuple(gates)


def _run_warmups(
    request: SurfaceCalibrationWorkerRequest,
    compiled: dict[tuple[str, MatmulCollectiveStrategy], _CompiledArm],
    residents: dict[str, _ResidentScenario],
) -> tuple[SurfaceCalibrationWarmupExecution, ...]:
    warmups = []
    repetitions: dict[tuple[str, MatmulCollectiveStrategy], int] = {}
    for scenario_index, scenario_name in enumerate(request.protocol.scenarios):
        resident = residents[scenario_name]
        for strategy in request.protocol.warmup_strategy_order(scenario_index):
            key = (scenario_name, strategy)
            repetitions[key] = repetitions.get(key, 0) + 1
            _, start, stop = _run_and_wait(compiled[key], resident)
            warmups.append(
                SurfaceCalibrationWarmupExecution(
                    sequence=len(warmups) + 1,
                    scenario_name=scenario_name,
                    scenario_position=scenario_index + 1,
                    strategy=strategy,
                    strategy_repetition=repetitions[key],
                    resident_pair_sha256=resident.pair.resident_pair_sha256,
                    invocation_nonce=request.invocation_nonce,
                    worker_pid=os.getpid(),
                    start_ns=start,
                    stop_ns=stop,
                )
            )
    if len(warmups) != 320:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_WARMUP_COUNT_MISMATCH")
    _verify_all_resident_sentinels(residents)
    return tuple(warmups)


def _collect_samples(
    request: SurfaceCalibrationWorkerRequest,
    compiled: dict[tuple[str, MatmulCollectiveStrategy], _CompiledArm],
    residents: dict[str, _ResidentScenario],
) -> tuple[SurfaceCalibrationCallSample, ...]:
    samples = []
    for round_index in range(request.protocol.paired_rounds):
        for scenario_position, scenario_name in enumerate(
            request.protocol.scenario_order(round_index), start=1
        ):
            resident = residents[scenario_name]
            for arm_position, strategy in enumerate(
                request.protocol.strategy_order(round_index), start=1
            ):
                for call_index in range(request.protocol.calls_per_position):
                    _, start, stop = _run_and_wait(compiled[(scenario_name, strategy)], resident)
                    samples.append(
                        SurfaceCalibrationCallSample(
                            sequence=len(samples) + 1,
                            round_index=round_index,
                            scenario_name=scenario_name,
                            scenario_position=scenario_position,
                            strategy=strategy,
                            arm_position=arm_position,
                            call_index=call_index,
                            resident_pair_sha256=resident.pair.resident_pair_sha256,
                            invocation_nonce=request.invocation_nonce,
                            worker_pid=os.getpid(),
                            start_ns=start,
                            stop_ns=stop,
                            duration_ns=stop - start,
                        )
                    )
    if len(samples) != 2560:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SAMPLE_COUNT_MISMATCH")
    _verify_all_resident_sentinels(residents)
    return tuple(samples)


def execute_calibration_worker(
    root: Path,
    request: SurfaceCalibrationWorkerRequest,
    parent: MatmulCollectiveSurfaceCorrectnessEvidence,
) -> MatmulCollectiveSurfaceCalibrationEvidence:
    request = SurfaceCalibrationWorkerRequest.model_validate(
        request.model_dump(mode="python", exclude_computed_fields=True)
    )
    parent = MatmulCollectiveSurfaceCorrectnessEvidence.model_validate(
        parent.model_dump(mode="python", exclude_computed_fields=True)
    )
    if tuple(request.protocol.scenarios) != tuple(
        value.name for value in request.design.calibration_scenarios
    ) or any("holdout" in value for value in request.protocol.scenarios):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_HOLDOUT_NOT_AUTHORIZED")
    compiled, continuity = _compile_all_arms(root, request, parent)
    residents, inputs = _materialize_all_inputs(root, request, parent, compiled)
    _verify_all_resident_sentinels(residents)
    before = _run_output_gates(root, request, compiled, residents, "before_timing")
    warmups = _run_warmups(request, compiled, residents)
    samples = _collect_samples(request, compiled, residents)
    after = _run_output_gates(root, request, compiled, residents, "after_timing")
    evidence = MatmulCollectiveSurfaceCalibrationEvidence(
        protocol_id=request.protocol.protocol_id,
        protocol_file_sha256=request.protocol_file_sha256,
        design_id=request.design.design_id,
        design_file_sha256=request.design_file_sha256,
        correctness_parent_attempt_id=request.protocol.correctness_parent.attempt_id,
        correctness_parent_evidence_sha256=request.protocol.correctness_parent.evidence_sha256,
        correctness_parent_receipt_sha256=request.protocol.correctness_parent.receipt_sha256,
        calibration_execution_authority_sha256=request.execution_authority_sha256,
        invocation_nonce=request.invocation_nonce,
        worker_pid=os.getpid(),
        continuity=continuity,
        inputs=inputs,
        resident_pairs=tuple(residents[name].pair for name in request.protocol.scenarios),
        output_gates=(*before, *after),
        warmups=warmups,
        samples=samples,
    )
    validate_surface_calibration_evidence(
        evidence,
        request.protocol,
        request.design,
        expected_protocol_file_sha256=request.protocol_file_sha256,
        expected_design_file_sha256=request.design_file_sha256,
        expected_execution_authority_sha256=request.execution_authority_sha256,
        expected_invocation_nonce=request.invocation_nonce,
        expected_worker_pid=os.getpid(),
    )
    return evidence


def _validate_empty_compilation_cache(request: SurfaceCalibrationWorkerRequest) -> Path:
    raw = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    path = Path(request.compilation_cache_path)
    if (
        raw != request.compilation_cache_path
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or any(path.iterdir())
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CACHE_AUTHORITY_MISMATCH")
    return path


def _validate_control_file(path: Path) -> None:
    info = path.lstat()
    if (
        path.is_symlink()
        or not path.is_file()
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_mode & 0o077
    ):
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CONTROL_FILE_INVALID path={path}")


def _validate_worker_authorization(
    root: Path,
    request_path: Path,
    request: SurfaceCalibrationWorkerRequest,
) -> tuple[SurfaceCalibrationRunIdentity, SurfaceCalibrationExecutionAuthority]:
    if not root.is_absolute():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROOT_NOT_ABSOLUTE")
    if root.is_symlink():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROOT_SYMLINK")
    root = root.resolve(strict=True)
    info = root.stat()
    if root.is_symlink() or not root.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROOT_AUTHORITY_INVALID")
    _validate_control_file(request_path)
    if request_path.resolve(strict=True) != (root / "worker-request.json").resolve(strict=True):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_REQUEST_PATH_MISMATCH")
    for path in (
        root / "run_identity.json",
        root / "execution_authority.json",
        root / "STARTED.json",
        root / "attempt_claim.json",
    ):
        _validate_control_file(path)
    identity = SurfaceCalibrationRunIdentity.model_validate_json(
        (root / "run_identity.json").read_text()
    )
    authority = SurfaceCalibrationExecutionAuthority.model_validate_json(
        (root / "execution_authority.json").read_text()
    )
    started = {
        "attempt_id": request.attempt_id,
        "invocation_nonce": request.invocation_nonce,
        "protocol_id": request.protocol.protocol_id,
        "state": "started",
    }
    parent_root = _expected_parent_correctness_root(root, request.protocol).resolve(strict=True)
    cache_path = Path(request.compilation_cache_path).resolve(strict=True)
    if (
        request.output_root != str(root)
        or Path(request.parent_correctness_root).resolve(strict=True) != parent_root
        or identity.attempt_id != request.attempt_id
        or identity.protocol_id != request.protocol.protocol_id
        or identity.execution_authority_sha256 != request.execution_authority_sha256
        or identity.source_authority_sha256 != request.source_authority_sha256
        or identity.output_root != str(root)
        or Path(identity.parent_correctness_root).resolve(strict=True) != parent_root
        or Path(identity.compilation_cache_path).resolve(strict=True) != cache_path
        or request.source_commit != authority.source.source_commit
        or request.source_authority_sha256 != authority.source.authority_sha256
        or request.execution_authority_sha256 != authority.authority_sha256
        or request.protocol_file_sha256 != authority.protocol_file_sha256
        or request.design_file_sha256 != authority.design_file_sha256
        or json.loads((root / "STARTED.json").read_text()) != started
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_WORKER_AUTHORIZATION_MISMATCH")
    claim_path = (
        Path(request.protocol.attempt_registry_root)
        / f"{request.protocol.permanent_claim_key}.json"
    )
    archived_claim = root / "attempt_claim.json"
    expected_claim = SurfaceCalibrationAttemptClaim(
        attempt_id=request.attempt_id,
        protocol_id=request.protocol.protocol_id,
        permanent_claim_key=request.protocol.permanent_claim_key,
        correctness_parent_receipt_sha256=request.protocol.correctness_parent.receipt_sha256,
        source_commit=request.source_commit,
        output_root=str(root),
    )
    expected_bytes = _json_bytes(
        expected_claim.model_dump(mode="json", exclude_computed_fields=True)
    )
    if (
        Path(identity.attempt_claim_path) != claim_path
        or claim_path.is_symlink()
        or not claim_path.is_file()
        or claim_path.stat().st_nlink != 1
        or archived_claim.is_symlink()
        or not archived_claim.is_file()
        or archived_claim.stat().st_nlink != 1
        or claim_path.read_bytes() != expected_bytes
        or archived_claim.read_bytes() != expected_bytes
        or identity.attempt_claim_sha256 != _sha256_bytes(expected_bytes)
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_WORKER_CLAIM_MISMATCH")
    _validate_control_file(claim_path)
    return identity, authority


def _expected_parent_correctness_root(
    root: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> Path:
    return root / "parent" / protocol.correctness_parent.archive_root_name


def main() -> None:
    if not sys.flags.safe_path:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SAFE_PATH_REQUIRED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()
    if args.probe_output is not None:
        if (
            args.protocol is None
            or args.design is None
            or args.root is not None
            or args.request is not None
        ):
            parser.error("authority probe requires --probe-output, --protocol, and --design only")
        protocol = MatmulCollectiveSurfaceCalibrationProtocol.model_validate_json(
            args.protocol.read_text()
        )
        design = MatmulCollectiveSurfaceDesignContract.model_validate_json(args.design.read_text())
        authority, _ = capture_calibration_execution_authority(
            Path(design.compilation_source_root), protocol, design
        )
        _write_model_exclusive(args.probe_output, authority)
        return
    if (
        args.root is None
        or args.request is None
        or args.protocol is not None
        or args.design is not None
    ):
        parser.error("worker execution requires --root and --request only")
    request = SurfaceCalibrationWorkerRequest.model_validate_json(args.request.read_text())
    _, recorded_authority = _validate_worker_authorization(args.root, args.request, request)
    _validate_empty_compilation_cache(request)
    observed_authority, source_blobs = capture_calibration_execution_authority(
        Path(request.design.compilation_source_root), request.protocol, request.design
    )
    if observed_authority != recorded_authority:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_WORKER_AUTHORITY_CHANGED")
    parent = _load_parent_correctness(Path(request.parent_correctness_root), request.protocol)
    evidence = execute_calibration_worker(args.root, request, parent)
    _validate_loaded_tpu_cake_sources(Path(request.design.compilation_source_root), source_blobs)
    _, final_authority = _validate_worker_authorization(args.root, args.request, request)
    if final_authority != recorded_authority:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_AUTHORITY_MUTATED")
    result = SurfaceCalibrationWorkerResult(
        attempt_id=request.attempt_id,
        invocation_nonce=request.invocation_nonce,
        worker_pid=os.getpid(),
        execution_authority_sha256=request.execution_authority_sha256,
        evidence=evidence,
    )
    _write_model_exclusive(args.root / "worker-result.json", result)


if __name__ == "__main__":
    main()
