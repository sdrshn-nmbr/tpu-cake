from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.rpa_lowering import ShardedFusedRpaPlan, lower_inkling_sharded_rpa_to_pallas
from tpu_cake.rpa_surface import (
    InklingShardedRpaCorrectnessObservation,
    InklingShardedRpaDevice,
    InklingShardedRpaSurfaceContract,
    InklingShardedRpaSurfaceReceipt,
    InklingShardedRpaSurfaceResult,
    InklingShardedRpaTimingRound,
    default_inkling_sharded_rpa_surface_contract,
)
from tpu_cake.runner import _runtime_identity
from tpu_cake.workloads.inkling_rpa import (
    inkling_sharded_fused_rpa_inputs,
    inkling_sharded_fused_rpa_reference,
    inkling_sharded_fused_rpa_schedule,
)

_RECEIPT_SCHEMA = "inkling-sharded-rpa-surface-receipt-v1"


@dataclass(frozen=True)
class _CompiledSurface:
    plan: ShardedFusedRpaPlan
    mesh: Any
    executable: Callable[..., tuple[Any, Any]]
    stablehlo: str
    compiler_hlo: str


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_computed_fields=True)
    if isinstance(value, dict):
        return {key: _json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_payload(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_payload(value), indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _write_json_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(_json_payload(value), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
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


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise OSError(errno.ENOTSUP, "exclusive directory publication is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _save_bf16(path: Path, value: np.ndarray) -> None:
    logical = np.asarray(value)
    if str(logical.dtype) != "bfloat16":
        raise ValueError(f"INKLING_SHARDED_RPA_BF16_SAVE_DTYPE path={path} dtype={logical.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, logical.view(np.uint16), allow_pickle=False)


def _load_bf16(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"INKLING_SHARDED_RPA_ARRAY_INVALID path={path}")
    storage = np.load(path, allow_pickle=False)
    if storage.dtype != np.uint16:
        raise ValueError(f"INKLING_SHARDED_RPA_ARRAY_STORAGE_DTYPE path={path}")
    return storage.view(np.dtype(jax.numpy.bfloat16))


def _source_manifest() -> tuple[SourceFileContract, ...]:
    repository = _repository_root()
    paths = (
        "src/tpu_cake/canonical.py",
        "src/tpu_cake/contracts.py",
        "src/tpu_cake/dialects/tpu_schedule.py",
        "src/tpu_cake/frontend.py",
        "src/tpu_cake/identity.py",
        "src/tpu_cake/ledger.py",
        "src/tpu_cake/rpa_device_main.py",
        "src/tpu_cake/rpa_lowering.py",
        "src/tpu_cake/rpa_surface.py",
        "src/tpu_cake/rpa_surface_runner.py",
        "src/tpu_cake/runner.py",
        "src/tpu_cake/source.py",
        "src/tpu_cake/workloads/inkling_rpa.py",
    )
    return tuple(SourceFileContract(path=path, sha256=_sha256(repository / path)) for path in paths)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> None:
    status = _git_output(repository, "status", "--porcelain=v1").splitlines()
    if status:
        raise ValueError(f"INKLING_SHARDED_RPA_SOURCE_DIRTY status={status}")


def _require_compilation_root(
    repository: Path,
    contract: InklingShardedRpaSurfaceContract,
) -> None:
    if repository.resolve() != Path(contract.compilation_source_root):
        raise ValueError("INKLING_SHARDED_RPA_COMPILATION_ROOT_MISMATCH")


def _require_backend_source(contract: InklingShardedRpaSurfaceContract) -> None:
    backend_python_path = Path(contract.backend_python_path).resolve()
    resolved_sys_path = {Path(value).resolve() for value in sys.path if value}
    if backend_python_path not in resolved_sys_path or not backend_python_path.is_dir():
        raise ValueError("INKLING_SHARDED_RPA_BACKEND_PYTHON_PATH_MISMATCH")
    backend_repository = backend_python_path.parents[2]
    if (
        _git_output(backend_repository, "rev-parse", "HEAD")
        != contract.plan.backend_repository_revision
    ):
        raise ValueError("INKLING_SHARDED_RPA_BACKEND_REVISION_MISMATCH")
    if _git_output(backend_repository, "status", "--porcelain=v1").splitlines():
        raise ValueError("INKLING_SHARDED_RPA_BACKEND_SOURCE_DIRTY")


def _require_backend_runtime(contract: InklingShardedRpaSurfaceContract) -> None:
    observed = tuple(
        f"{name}=={importlib.metadata.version(name)}"
        for name in ("fastapi", "orjson", "psutil", "pyzmq")
    )
    if observed != contract.backend_import_packages:
        raise ValueError(
            "INKLING_SHARDED_RPA_BACKEND_RUNTIME_MISMATCH "
            f"expected={contract.backend_import_packages} observed={observed}"
        )


def _source_state(repository: Path) -> dict[str, Any]:
    commit = _git_output(repository, "rev-parse", "HEAD")
    uv_lock_sha256 = _sha256(repository / "uv.lock")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "git_status": [],
        "source_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "uv_lock_sha256": uv_lock_sha256,
    }


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"INKLING_SHARDED_RPA_PATH_SYMLINK path={current}")


def _require_safe_output_root(root: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    resolved_parent = root.parent.resolve()
    protected = (Path("/").resolve(), Path.home().resolve(), repository)
    if any(root.resolve() == value or root.resolve() in value.parents for value in protected):
        raise ValueError(f"INKLING_SHARDED_RPA_UNSAFE_ROOT path={root}")
    if repository in root.resolve().parents:
        raise ValueError(f"INKLING_SHARDED_RPA_ROOT_INSIDE_REPOSITORY path={root}")
    _reject_symlink_components(root.parent)
    if not resolved_parent.is_dir():
        raise ValueError(f"INKLING_SHARDED_RPA_OUTPUT_PARENT_INVALID path={root.parent}")


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-sharded-rpa-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    info = lock_root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError(f"INKLING_SHARDED_RPA_LOCK_ROOT_INVALID path={lock_root}")
    name = hashlib.sha256(str(root.resolve()).encode()).hexdigest()
    descriptor = os.open(
        lock_root / f"{name}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        lock_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.getuid()
            or lock_info.st_nlink != 1
            or lock_info.st_mode & 0o077
        ):
            raise ValueError("INKLING_SHARDED_RPA_LOCK_FILE_INVALID")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"INKLING_SHARDED_RPA_RUN_LOCKED path={root}") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _device_inventory(devices: tuple[Any, ...]) -> tuple[InklingShardedRpaDevice, ...]:
    observed = tuple(
        InklingShardedRpaDevice(
            id=int(device.id),
            process_index=int(device.process_index),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        for device in devices
    )
    expected = tuple(
        InklingShardedRpaDevice(
            id=index,
            process_index=0,
            platform="tpu",
            device_kind="TPU7x",
        )
        for index in range(8)
    )
    if observed != expected:
        raise ValueError(f"INKLING_SHARDED_RPA_DEVICE_INVENTORY observed={observed}")
    return observed


def _compile_surface(
    contract: InklingShardedRpaSurfaceContract,
    kernel: Callable[..., tuple[Any, Any]],
    devices: tuple[Any, ...],
    host_inputs: tuple[np.ndarray, ...],
) -> _CompiledSurface:
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    if contract.plan != default_inkling_sharded_rpa_surface_contract().plan:
        raise ValueError("INKLING_SHARDED_RPA_PLAN_CONTRACT_MISMATCH")
    mesh, executable = plan.build_executable(
        kernel,
        backend_manifest=plan.backend_manifest,
        devices=devices,
    )
    placed = plan.place_inputs(host_inputs, mesh=mesh)
    lowered = executable.lower(*placed)
    stablehlo = str(lowered.compiler_ir("stablehlo")).rstrip("\n") + "\n"
    compiled = lowered.compile()
    compiler_hlo = compiled.as_text().rstrip("\n") + "\n"
    if hashlib.sha256(stablehlo.encode()).hexdigest() != contract.plan.stablehlo_sha256:
        raise ValueError("INKLING_SHARDED_RPA_STABLEHLO_IDENTITY_MISMATCH")
    if "tpu_custom_call" not in compiler_hlo or "RPAd" not in compiler_hlo:
        raise ValueError("INKLING_SHARDED_RPA_COMPILER_HLO_MARKERS_MISSING")
    return _CompiledSurface(
        plan=plan,
        mesh=mesh,
        executable=compiled,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
    )


def _place_inputs(
    compiled: _CompiledSurface,
    host_inputs: tuple[np.ndarray, ...],
) -> tuple[Any, ...]:
    return compiled.plan.place_inputs(host_inputs, mesh=compiled.mesh)


def _execute(
    executable: Callable[..., tuple[Any, Any]], inputs: tuple[Any, ...]
) -> tuple[np.ndarray, np.ndarray]:
    outputs = executable(*inputs)
    jax.block_until_ready(outputs)
    return tuple(np.asarray(value) for value in outputs)


def _execute_timed(executable: Callable[..., tuple[Any, Any]], inputs: tuple[Any, ...]) -> None:
    jax.block_until_ready(executable(*inputs))


def _errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            "INKLING_SHARDED_RPA_OUTPUT_ABI_MISMATCH "
            f"actual={actual.shape}/{actual.dtype} expected={expected.shape}/{expected.dtype}"
        )
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = max((abs(float(value)) for value in difference.ravel()), default=0.0)
    numerator_squared = math.fsum(float(value) ** 2 for value in difference.ravel())
    denominator_squared = math.fsum(
        float(value) ** 2 for value in expected.astype(np.float64).ravel()
    )
    relative_l2 = math.sqrt(numerator_squared / max(denominator_squared, 1e-60))
    return maximum, round(relative_l2, 15)


def _validate_output_abi(
    contract: InklingShardedRpaSurfaceContract,
    output: np.ndarray,
    cache: np.ndarray,
) -> None:
    for name, value, shape, dtype in zip(
        ("output", "cache"),
        (output, cache),
        contract.plan.global_output_shapes,
        contract.plan.output_dtypes,
        strict=True,
    ):
        if value.shape != shape or str(value.dtype) != dtype:
            raise ValueError(
                f"INKLING_SHARDED_RPA_{name.upper()}_ABI_MISMATCH "
                f"actual={value.shape}/{value.dtype} expected={shape}/{dtype}"
            )


def _correctness_observation(
    root: Path,
    contract: InklingShardedRpaSurfaceContract,
    compiled: _CompiledSurface,
    seed: int,
) -> InklingShardedRpaCorrectnessObservation:
    host_inputs = inkling_sharded_fused_rpa_inputs(seed)
    oracle_output, oracle_cache = inkling_sharded_fused_rpa_reference(host_inputs)
    placed = _place_inputs(compiled, host_inputs)
    executions = tuple(
        _execute(compiled.executable, placed) for _ in range(contract.repeat_executions)
    )
    (first_output, first_cache), (second_output, second_cache) = executions
    _validate_output_abi(contract, first_output, first_cache)
    _validate_output_abi(contract, second_output, second_cache)
    _validate_output_abi(contract, oracle_output, oracle_cache)
    maximum, relative_l2 = _errors(first_output, oracle_output)
    repeated_output_exact = np.array_equal(first_output, second_output)
    repeated_cache_exact = np.array_equal(first_cache, second_cache)
    cache_exact = np.array_equal(first_cache, oracle_cache)
    passed = (
        repeated_output_exact
        and repeated_cache_exact
        and cache_exact
        and maximum <= contract.output_maximum_absolute_error
        and relative_l2 <= contract.output_relative_l2_error
    )
    observation = InklingShardedRpaCorrectnessObservation(
        seed=seed,
        input_sha256=arrays_sha256(host_inputs),
        output_sha256=array_sha256(first_output),
        repeat_output_sha256=array_sha256(second_output),
        oracle_output_sha256=array_sha256(oracle_output),
        cache_sha256=array_sha256(first_cache),
        repeat_cache_sha256=array_sha256(second_cache),
        oracle_cache_sha256=array_sha256(oracle_cache),
        repeated_output_exact=repeated_output_exact,
        repeated_cache_exact=repeated_cache_exact,
        maximum_absolute_error=maximum,
        relative_l2_error=relative_l2,
        passed=passed,
    )
    seed_root = root / "correctness" / f"seed-{seed}"
    _save_bf16(seed_root / "output.npy", first_output)
    _save_bf16(seed_root / "repeat_output.npy", second_output)
    _save_bf16(seed_root / "cache.npy", first_cache)
    _save_bf16(seed_root / "repeat_cache.npy", second_cache)
    _save_bf16(seed_root / "oracle_output.npy", oracle_output)
    _write_json(seed_root / "observation.json", observation)
    if not passed:
        raise ValueError(f"INKLING_SHARDED_RPA_CORRECTNESS_FAILED observation={observation}")
    return observation


def _timing_rounds(
    contract: InklingShardedRpaSurfaceContract,
    compiled: _CompiledSurface,
    placed: tuple[Any, ...],
) -> tuple[InklingShardedRpaTimingRound, ...]:
    for _ in range(contract.warmup_iterations):
        _execute_timed(compiled.executable, placed)
    rounds = []
    for round_index in range(contract.timing_rounds):
        samples = []
        for _ in range(contract.samples_per_round):
            started = time.perf_counter_ns()
            _execute_timed(compiled.executable, placed)
            samples.append(time.perf_counter_ns() - started)
        rounds.append(
            InklingShardedRpaTimingRound(
                round_index=round_index,
                samples_ns=tuple(samples),
                median_ns=float(statistics.median(samples)),
            )
        )
    return tuple(rounds)


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    if relative == "contract.json":
        return ArtifactRole.SEARCH_CONTRACT
    if relative == "run_identity.json":
        return ArtifactRole.INVOCATION
    if relative == "source_state.json":
        return ArtifactRole.SOURCE_STATE
    if relative == "source_diff.patch":
        return ArtifactRole.SOURCE_DIFF
    if relative == "source_manifest.json":
        return ArtifactRole.BACKEND_MANIFEST
    if relative == "physical.xdsl":
        return ArtifactRole.PHYSICAL_IR
    if relative == "plan.json":
        return ArtifactRole.PLAN_MANIFEST
    if relative == "stablehlo.txt":
        return ArtifactRole.STABLEHLO
    if relative == "compiler_hlo.txt":
        return ArtifactRole.COMPILER_HLO
    if relative == "ledger.sqlite":
        return ArtifactRole.EXECUTION_LEDGER
    if relative == "correctness.json" or relative.endswith("/observation.json"):
        return ArtifactRole.PROFILE_ASSESSMENT
    if relative.endswith("/oracle_output.npy"):
        return ArtifactRole.ORACLE_OUTPUT
    if path.name in {"output.npy", "repeat_output.npy", "cache.npy", "repeat_cache.npy"}:
        return ArtifactRole.CORRECTNESS_OUTPUT
    if relative == "rounds.json":
        return ArtifactRole.TIMING_SAMPLES
    if relative in {
        "timing/pre_output.npy",
        "timing/pre_cache.npy",
        "timing/post_output.npy",
        "timing/post_cache.npy",
    }:
        return ArtifactRole.CORRECTNESS_OUTPUT
    if relative == "result.json":
        return ArtifactRole.SEARCH_RESULT
    raise ValueError(f"INKLING_SHARDED_RPA_UNKNOWN_ARTIFACT path={relative}")


def _artifacts(root: Path) -> tuple[ArtifactReference, ...]:
    return tuple(
        ArtifactReference(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            role=_artifact_role(path.relative_to(root)),
        )
        for path in sorted(
            (value for value in root.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(root).as_posix(),
        )
        if path.name != "receipt.json"
    )


def _ledger_payload(
    contract: InklingShardedRpaSurfaceContract,
    run_id: str,
    source_state: dict[str, Any],
    root: Path,
    state: RunState,
) -> dict[str, Any]:
    if state is RunState.CREATED:
        return {"surface_id": contract.surface_id, "run_id": run_id}
    if state is RunState.VERIFIED:
        return {
            "schedule_sha256": contract.plan.schedule_sha256,
            "execution_sha256": contract.plan.execution_sha256,
            "source_commit": source_state["git_commit"],
        }
    if state is RunState.LOWERED:
        return {
            "physical_sha256": _sha256(root / "physical.xdsl"),
            "plan_sha256": _sha256(root / "plan.json"),
        }
    if state is RunState.COMPILED:
        return {
            "stablehlo_sha256": _sha256(root / "stablehlo.txt"),
            "compiler_hlo_sha256": _sha256(root / "compiler_hlo.txt"),
        }
    if state is RunState.CORRECT:
        return {"correctness_sha256": _sha256(root / "correctness.json")}
    if state is RunState.TIMED:
        return {"rounds_sha256": _sha256(root / "rounds.json")}
    if state is RunState.ACCEPTED:
        return {"result_sha256": _sha256(root / "result.json")}
    raise ValueError(f"INKLING_SHARDED_RPA_LEDGER_STATE_UNSUPPORTED state={state}")


def _validate_source(root: Path, result: InklingShardedRpaSurfaceResult) -> None:
    repository = _repository_root()
    state = json.loads((root / "source_state.json").read_text())
    manifest = tuple(
        SourceFileContract.model_validate(value)
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    if (
        state["git_dirty"] is not False
        or state["git_status"] != []
        or state["source_diff_sha256"] != hashlib.sha256(b"").hexdigest()
        or (root / "source_diff.patch").read_bytes() != b""
        or state["git_commit"] != result.source_commit
        or state["uv_lock_sha256"] != result.uv_lock_sha256
        or _sha256(root / "source_state.json") != result.source_state_sha256
        or _sha256(root / "source_manifest.json") != result.source_manifest_sha256
        or manifest != result.source_manifest
        or manifest != _source_manifest()
    ):
        raise ValueError("INKLING_SHARDED_RPA_SOURCE_EVIDENCE_MISMATCH")
    if _git_output(repository, "rev-parse", "HEAD") != result.source_commit:
        raise ValueError("INKLING_SHARDED_RPA_VERIFIER_COMMIT_MISMATCH")
    if _sha256(repository / "uv.lock") != result.uv_lock_sha256:
        raise ValueError("INKLING_SHARDED_RPA_VERIFIER_LOCK_MISMATCH")
    for source in manifest:
        live = repository / source.path
        blob = subprocess.run(
            ["git", "show", f"{result.source_commit}:{source.path}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        if _sha256(live) != source.sha256 or hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(f"INKLING_SHARDED_RPA_SOURCE_BLOB_MISMATCH path={source.path}")


def _validate_correctness(
    root: Path,
    contract: InklingShardedRpaSurfaceContract,
    saved: tuple[InklingShardedRpaCorrectnessObservation, ...],
) -> None:
    replayed = []
    for seed in contract.correctness_seeds:
        inputs = inkling_sharded_fused_rpa_inputs(seed)
        oracle_output, oracle_cache = inkling_sharded_fused_rpa_reference(inputs)
        seed_root = root / "correctness" / f"seed-{seed}"
        output = _load_bf16(seed_root / "output.npy")
        repeat_output = _load_bf16(seed_root / "repeat_output.npy")
        cache = _load_bf16(seed_root / "cache.npy")
        repeat_cache = _load_bf16(seed_root / "repeat_cache.npy")
        saved_oracle = _load_bf16(seed_root / "oracle_output.npy")
        if not np.array_equal(saved_oracle, oracle_output):
            raise ValueError(f"INKLING_SHARDED_RPA_ORACLE_REPLAY_MISMATCH seed={seed}")
        _validate_output_abi(contract, output, cache)
        _validate_output_abi(contract, repeat_output, repeat_cache)
        maximum, relative_l2 = _errors(output, oracle_output)
        observation = InklingShardedRpaCorrectnessObservation.model_validate_json(
            (seed_root / "observation.json").read_text()
        )
        expected = observation.model_copy(
            update={
                "input_sha256": arrays_sha256(inputs),
                "output_sha256": array_sha256(output),
                "repeat_output_sha256": array_sha256(repeat_output),
                "oracle_output_sha256": array_sha256(oracle_output),
                "cache_sha256": array_sha256(cache),
                "repeat_cache_sha256": array_sha256(repeat_cache),
                "oracle_cache_sha256": array_sha256(oracle_cache),
                "repeated_output_exact": np.array_equal(output, repeat_output),
                "repeated_cache_exact": np.array_equal(cache, repeat_cache),
                "maximum_absolute_error": maximum,
                "relative_l2_error": relative_l2,
                "passed": (
                    np.array_equal(output, repeat_output)
                    and np.array_equal(cache, repeat_cache)
                    and np.array_equal(cache, oracle_cache)
                    and maximum <= contract.output_maximum_absolute_error
                    and relative_l2 <= contract.output_relative_l2_error
                ),
            }
        )
        if observation != expected or not observation.passed:
            raise ValueError(f"INKLING_SHARDED_RPA_CORRECTNESS_REPLAY_MISMATCH seed={seed}")
        replayed.append(observation)
    if tuple(replayed) != saved:
        raise ValueError("INKLING_SHARDED_RPA_CORRECTNESS_INVENTORY_MISMATCH")


def _validate_plan_artifacts(
    root: Path,
    contract: InklingShardedRpaSurfaceContract,
    result: InklingShardedRpaSurfaceResult,
) -> None:
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    if (
        canonical_text(inkling_sharded_fused_rpa_schedule()) != (root / "physical.xdsl").read_text()
        or json.loads((root / "plan.json").read_text()) != contract.plan.model_dump(mode="json")
        or plan.schedule_sha256 != contract.plan.schedule_sha256
        or plan.source_sha256() != contract.plan.execution_sha256
        or _sha256(root / "stablehlo.txt") != contract.plan.stablehlo_sha256
        or _sha256(root / "compiler_hlo.txt") != result.compiler_hlo_sha256
    ):
        raise ValueError("INKLING_SHARDED_RPA_PLAN_REPLAY_MISMATCH")
    compiler_hlo = (root / "compiler_hlo.txt").read_text()
    if "tpu_custom_call" not in compiler_hlo or "RPAd" not in compiler_hlo:
        raise ValueError("INKLING_SHARDED_RPA_COMPILER_HLO_REPLAY_MISMATCH")


def validate_inkling_sharded_rpa_surface(
    root: Path,
    expected_contract: InklingShardedRpaSurfaceContract,
    *,
    require_receipt: bool = True,
    require_accepted_ledger: bool = True,
) -> InklingShardedRpaSurfaceResult:
    if root.is_symlink():
        raise ValueError(f"INKLING_SHARDED_RPA_ROOT_INVALID path={root}")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"INKLING_SHARDED_RPA_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError(f"INKLING_SHARDED_RPA_LINK_INVALID path={path}")
    contract = InklingShardedRpaSurfaceContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if contract != expected_contract or contract != default_inkling_sharded_rpa_surface_contract():
        raise ValueError("INKLING_SHARDED_RPA_EXTERNAL_CONTRACT_MISMATCH")
    result = InklingShardedRpaSurfaceResult.model_validate_json((root / "result.json").read_text())
    identity = json.loads((root / "run_identity.json").read_text())
    expected_run_id = semantic_sha256(
        "inkling-sharded-rpa-surface-run-v1",
        contract.surface_id,
        result.source_commit,
        contract.plan.schedule_sha256,
        contract.plan.execution_sha256,
    )
    if identity != {
        "schema": contract.surface_schema,
        "surface_id": contract.surface_id,
        "run_id": expected_run_id,
        "source_commit": result.source_commit,
    } or (result.surface_id, result.run_id) != (contract.surface_id, expected_run_id):
        raise ValueError("INKLING_SHARDED_RPA_RUN_IDENTITY_MISMATCH")
    if (
        result.runtime != contract.runtime
        or result.plan != contract.plan
        or result.claim_scope != contract.claim_scope
        or tuple(value.seed for value in result.correctness) != contract.correctness_seeds
        or len(result.rounds) != contract.timing_rounds
        or any(len(value.samples_ns) != contract.samples_per_round for value in result.rounds)
        or result.devices
        != tuple(
            InklingShardedRpaDevice(
                id=index,
                process_index=0,
                platform="tpu",
                device_kind="TPU7x",
            )
            for index in range(8)
        )
    ):
        raise ValueError("INKLING_SHARDED_RPA_RESULT_CONTRACT_MISMATCH")
    _validate_source(root, result)
    _validate_plan_artifacts(root, contract, result)
    _validate_correctness(root, contract, result.correctness)
    timing_inputs = inkling_sharded_fused_rpa_inputs(contract.timing_seed)
    if result.timing_input_sha256 != arrays_sha256(timing_inputs):
        raise ValueError("INKLING_SHARDED_RPA_TIMING_INPUT_MISMATCH")
    timing_observation = next(
        value for value in result.correctness if value.seed == contract.timing_seed
    )
    pre_timing = (
        _load_bf16(root / "timing" / "pre_output.npy"),
        _load_bf16(root / "timing" / "pre_cache.npy"),
    )
    post_timing = (
        _load_bf16(root / "timing" / "post_output.npy"),
        _load_bf16(root / "timing" / "post_cache.npy"),
    )
    _validate_output_abi(contract, *pre_timing)
    _validate_output_abi(contract, *post_timing)
    expected_timing_output = (
        timing_observation.output_sha256,
        timing_observation.cache_sha256,
    )
    if (
        tuple(array_sha256(value) for value in pre_timing) != expected_timing_output
        or tuple(array_sha256(value) for value in post_timing) != expected_timing_output
        or not all(
            np.array_equal(before, after)
            for before, after in zip(pre_timing, post_timing, strict=True)
        )
        or result.pre_timing_output_sha256 != expected_timing_output
        or result.post_timing_output_sha256 != expected_timing_output
    ):
        raise ValueError("INKLING_SHARDED_RPA_TIMING_OUTPUT_MISMATCH")
    saved_rounds = tuple(
        InklingShardedRpaTimingRound.model_validate_json(json.dumps(value))
        for value in json.loads((root / "rounds.json").read_text())
    )
    if saved_rounds != result.rounds:
        raise ValueError("INKLING_SHARDED_RPA_TIMING_REPLAY_MISMATCH")
    source_state = json.loads((root / "source_state.json").read_text())
    expected_states = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.TIMED,
        *((RunState.ACCEPTED,) if require_accepted_ledger else ()),
    )
    history = read_ledger_history(root / "ledger.sqlite", expected_run_id)
    if tuple(value.state for value in history) != expected_states or tuple(
        value.payload_sha256 for value in history
    ) != tuple(
        ExperimentLedger.payload_sha256(
            _ledger_payload(contract, expected_run_id, source_state, root, state)
        )
        for state in expected_states
    ):
        raise ValueError("INKLING_SHARDED_RPA_LEDGER_REPLAY_MISMATCH")
    if require_receipt:
        receipt = InklingShardedRpaSurfaceReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        observed_paths = tuple(
            path.relative_to(root).as_posix()
            for path in sorted(
                (value for value in root.rglob("*") if value.is_file()),
                key=lambda value: value.relative_to(root).as_posix(),
            )
            if path.name != "receipt.json"
        )
        if (
            receipt.surface_id != contract.surface_id
            or receipt.run_id != result.run_id
            or receipt.result_sha256 != _sha256(root / "result.json")
            or receipt.claim_scope != contract.claim_scope
            or tuple(value.path for value in receipt.artifacts) != observed_paths
        ):
            raise ValueError("INKLING_SHARDED_RPA_RECEIPT_IDENTITY_MISMATCH")
        for artifact in receipt.artifacts:
            path = root / artifact.path
            if (
                path.stat().st_size != artifact.size_bytes
                or _sha256(path) != artifact.sha256
                or _artifact_role(Path(artifact.path)) is not artifact.role
            ):
                raise ValueError(f"INKLING_SHARDED_RPA_RECEIPT_ARTIFACT_MISMATCH path={path}")
    elif (root / "receipt.json").exists():
        raise ValueError("INKLING_SHARDED_RPA_UNEXPECTED_RECEIPT")
    return result


def _run_staged(
    root: Path,
    contract: InklingShardedRpaSurfaceContract,
    kernel: Callable[..., tuple[Any, Any]],
    devices: tuple[Any, ...],
    source_state: dict[str, Any],
    runtime: RuntimeIdentity,
) -> InklingShardedRpaSurfaceResult:
    source_manifest = _source_manifest()
    run_id = semantic_sha256(
        "inkling-sharded-rpa-surface-run-v1",
        contract.surface_id,
        source_state["git_commit"],
        contract.plan.schedule_sha256,
        contract.plan.execution_sha256,
    )
    _write_json(
        root / "run_identity.json",
        {
            "schema": contract.surface_schema,
            "surface_id": contract.surface_id,
            "run_id": run_id,
            "source_commit": source_state["git_commit"],
        },
    )
    _write_json(root / "contract.json", contract)
    _write_json(root / "source_state.json", source_state)
    _write_text(root / "source_diff.patch", "")
    _write_json(root / "source_manifest.json", source_manifest)
    _write_text(root / "physical.xdsl", canonical_text(inkling_sharded_fused_rpa_schedule()))
    _write_json(root / "plan.json", contract.plan)
    ledger = ExperimentLedger(root / "ledger.sqlite")
    ledger.create(run_id, _ledger_payload(contract, run_id, source_state, root, RunState.CREATED))
    ledger.transition(
        run_id,
        RunState.VERIFIED,
        _ledger_payload(contract, run_id, source_state, root, RunState.VERIFIED),
    )
    ledger.transition(
        run_id,
        RunState.LOWERED,
        _ledger_payload(contract, run_id, source_state, root, RunState.LOWERED),
    )

    first_inputs = inkling_sharded_fused_rpa_inputs(contract.correctness_seeds[0])
    compiled = _compile_surface(contract, kernel, devices, first_inputs)
    _write_text(root / "stablehlo.txt", compiled.stablehlo)
    _write_text(root / "compiler_hlo.txt", compiled.compiler_hlo)
    ledger.transition(
        run_id,
        RunState.COMPILED,
        _ledger_payload(contract, run_id, source_state, root, RunState.COMPILED),
    )

    correctness = tuple(
        _correctness_observation(root, contract, compiled, seed)
        for seed in contract.correctness_seeds
    )
    _write_json(root / "correctness.json", correctness)
    ledger.transition(
        run_id,
        RunState.CORRECT,
        _ledger_payload(contract, run_id, source_state, root, RunState.CORRECT),
    )

    timing_inputs = inkling_sharded_fused_rpa_inputs(contract.timing_seed)
    timing_placed = _place_inputs(compiled, timing_inputs)
    pre_timing = _execute(compiled.executable, timing_placed)
    rounds = _timing_rounds(contract, compiled, timing_placed)
    post_timing = _execute(compiled.executable, timing_placed)
    _validate_output_abi(contract, *pre_timing)
    _validate_output_abi(contract, *post_timing)
    _save_bf16(root / "timing" / "pre_output.npy", pre_timing[0])
    _save_bf16(root / "timing" / "pre_cache.npy", pre_timing[1])
    _save_bf16(root / "timing" / "post_output.npy", post_timing[0])
    _save_bf16(root / "timing" / "post_cache.npy", post_timing[1])
    _write_json(root / "rounds.json", rounds)
    ledger.transition(
        run_id,
        RunState.TIMED,
        _ledger_payload(contract, run_id, source_state, root, RunState.TIMED),
    )
    medians = tuple(value.median_ns for value in rounds)
    ordered = sorted(medians)
    result = InklingShardedRpaSurfaceResult(
        surface_id=contract.surface_id,
        run_id=run_id,
        source_commit=source_state["git_commit"],
        uv_lock_sha256=source_state["uv_lock_sha256"],
        source_state_sha256=_sha256(root / "source_state.json"),
        source_manifest_sha256=_sha256(root / "source_manifest.json"),
        source_manifest=source_manifest,
        runtime=runtime,
        devices=_device_inventory(devices),
        plan=contract.plan,
        compiler_hlo_sha256=_sha256(root / "compiler_hlo.txt"),
        correctness=correctness,
        timing_input_sha256=arrays_sha256(timing_inputs),
        pre_timing_output_sha256=tuple(array_sha256(value) for value in pre_timing),
        rounds=rounds,
        post_timing_output_sha256=tuple(array_sha256(value) for value in post_timing),
        median_round_duration_ns=float(statistics.median(medians)),
        p90_round_duration_ns=float(
            ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
        ),
        coefficient_of_variation=statistics.pstdev(medians) / statistics.mean(medians),
        accepted=True,
        claim_scope=contract.claim_scope,
    )
    _write_json(root / "result.json", result)
    ledger.close()
    validate_inkling_sharded_rpa_surface(
        root,
        contract,
        require_receipt=False,
        require_accepted_ledger=False,
    )
    with ExperimentLedger(root / "ledger.sqlite") as accepted_ledger:
        accepted_ledger.transition(
            run_id,
            RunState.ACCEPTED,
            _ledger_payload(contract, run_id, source_state, root, RunState.ACCEPTED),
        )
    artifacts = _artifacts(root)
    receipt = InklingShardedRpaSurfaceReceipt(
        receipt_schema=_RECEIPT_SCHEMA,
        surface_id=contract.surface_id,
        run_id=run_id,
        result_sha256=_sha256(root / "result.json"),
        artifact_count=len(artifacts),
        artifacts=artifacts,
        accepted=True,
        claim_scope=contract.claim_scope,
    )
    _write_json_atomic(root / "receipt.json", receipt)
    validate_inkling_sharded_rpa_surface(root, contract)
    return result


def run_inkling_sharded_rpa_surface(
    output_root: Path,
    contract: InklingShardedRpaSurfaceContract,
    kernel: Callable[..., tuple[Any, Any]],
) -> InklingShardedRpaSurfaceResult:
    output_root = output_root.absolute()
    repository = _repository_root()
    canonical_contract = default_inkling_sharded_rpa_surface_contract()
    if contract != canonical_contract:
        raise ValueError("INKLING_SHARDED_RPA_EXTERNAL_CONTRACT_MISMATCH")
    _require_compilation_root(repository, contract)
    _require_safe_output_root(output_root)
    _require_clean_repository(repository)
    runtime = _runtime_identity()
    if runtime != contract.runtime:
        raise ValueError("INKLING_SHARDED_RPA_RUNTIME_MISMATCH")
    _require_backend_source(contract)
    _require_backend_runtime(contract)
    with _exclusive_lock(output_root):
        if output_root.exists():
            return validate_inkling_sharded_rpa_surface(output_root, contract)
        source_state = _source_state(repository)
        devices = tuple(jax.devices())
        _device_inventory(devices)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.staging-",
                dir=output_root.parent,
            )
        )
        try:
            _run_staged(staging, contract, kernel, devices, source_state, runtime)
            _rename_directory_noreplace(staging, output_root)
            directory = os.open(output_root.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return validate_inkling_sharded_rpa_surface(output_root, contract)
        except Exception as error:
            if staging.exists():
                _write_json(
                    staging / "failure.json",
                    {"error_type": type(error).__name__, "message": str(error)},
                )
                failure = output_root.with_name(f"{output_root.name}.failed-{time.time_ns()}")
                try:
                    _rename_directory_noreplace(staging, failure)
                except OSError:
                    pass
            raise
