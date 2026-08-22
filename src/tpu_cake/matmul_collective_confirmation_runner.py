from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import sqlite3
import stat
import statistics
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_cake.artifacts import file_sha256
from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    KernelExperiment,
    RunReceipt,
    SourceFileContract,
)
from tpu_cake.identity import array_sha256, semantic_sha256, workload_rng
from tpu_cake.ledger import (
    EvidenceRun,
    ExperimentLedger,
    RunState,
    payload_sha256,
    read_ledger_history,
    seal_ledger,
)
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.matmul_collective_confirmation import (
    MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA,
    MatmulCollectiveConfirmationContract,
    MatmulCollectiveConfirmationReceipt,
    MatmulCollectiveConfirmationResult,
    MatmulCollectiveConfirmationRunIdentity,
    MatmulCollectiveCorrectnessObservation,
    MatmulCollectiveDevice,
    MatmulCollectiveHost,
    MatmulCollectivePlan,
    MatmulCollectiveTimingOutput,
    MatmulCollectiveTimingRound,
    collective_confirmation_orders,
    collective_confirmation_statistics,
    default_matmul_collective_confirmation_contract,
)
from tpu_cake.pallas_lowering import PallasMatmulPlan, lower_physical_matmul_to_pallas
from tpu_cake.receipt import _validate_matmul_compiler_strategy, validate_receipt
from tpu_cake.runner import MatmulCollectiveStrategy, _runtime_identity, _source_state
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


@dataclass(frozen=True)
class MatmulCollectivePlanSource:
    strategy: MatmulCollectiveStrategy
    plan: PallasMatmulPlan
    physical_text: str


@dataclass(frozen=True)
class PreparedMatmulCollective:
    source: MatmulCollectivePlanSource
    executable: Any
    mesh: Mesh


@dataclass(frozen=True)
class CompiledMatmulCollective:
    prepared: PreparedMatmulCollective
    executable: Any
    stablehlo: str
    compiler_hlo: str

    @property
    def strategy(self) -> MatmulCollectiveStrategy:
        return self.prepared.source.strategy


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
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


def _write_json_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_PATH_SYMLINK path={current}")


def _require_safe_root(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == value or root in value.parents for value in protected) or (
        repository_root in root.parents
    ):
        raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_UNSAFE_ROOT path={root}")


def _preflight_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_HARDLINK path={path}")


@contextmanager
def _exclusive_run_lock(run_id: str) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-matmul-confirmation-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    root_stat = lock_root.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.getuid()
        or root_stat.st_mode & 0o077
    ):
        raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_LOCK_ROOT_INVALID path={lock_root}")
    descriptor = os.open(
        lock_root / f"{run_id}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_LOCK_FILE_INVALID run_id={run_id}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_RUN_LOCKED run_id={run_id}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_source(
    repository_root: Path,
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[str, dict[str, str]]:
    status = _git(repository_root, "status", "--porcelain=v1")
    branch = _git(repository_root, "branch", "--show-current")
    commit = _git(repository_root, "rev-parse", "HEAD")
    origin_main = _git(repository_root, "rev-parse", "origin/main")
    remote_url = _git(repository_root, "remote", "get-url", "origin")
    server_main_record = _git(
        repository_root,
        "ls-remote",
        "origin",
        "refs/heads/main",
    )
    server_main_fields = server_main_record.split()
    if len(server_main_fields) != 2 or server_main_fields[1] != "refs/heads/main":
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_SERVER_MAIN_UNAVAILABLE")
    server_main = server_main_fields[0]
    if status:
        raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_SOURCE_DIRTY status={status.splitlines()}")
    if branch != contract.source_branch:
        raise ValueError(
            "MATMUL_COLLECTIVE_CONFIRMATION_BRANCH_MISMATCH "
            f"expected={contract.source_branch} observed={branch}"
        )
    if contract.require_origin_main and commit != origin_main:
        raise ValueError(
            "MATMUL_COLLECTIVE_CONFIRMATION_ORIGIN_MAIN_MISMATCH "
            f"head={commit} origin_main={origin_main}"
        )
    if remote_url != contract.source_remote_url or server_main != commit:
        raise ValueError(
            "MATMUL_COLLECTIVE_CONFIRMATION_SERVER_MAIN_MISMATCH "
            f"head={commit} server_main={server_main} remote_url={remote_url}"
        )
    if repository_root.resolve() != Path(contract.compilation_source_root):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_COMPILATION_ROOT_MISMATCH")
    return commit, {
        "remote_url": remote_url,
        "server_main_commit": server_main,
    }


def _source_manifest() -> tuple[SourceFileContract, ...]:
    repository_root = Path(__file__).resolve().parents[2]
    tracked = {
        Path(value)
        for value in _git(repository_root, "ls-files", "src/tpu_cake").splitlines()
        if value.endswith(".py")
    }
    tracked.update(
        {
            Path("src/tpu_cake/matmul_collective_confirmation.py"),
            Path("src/tpu_cake/matmul_collective_confirmation_runner.py"),
        }
    )
    names = tuple(sorted(tracked))
    return tuple(
        SourceFileContract(
            path=name.relative_to("src").as_posix(),
            sha256=_sha256(repository_root / name),
        )
        for name in names
    )


def _device_inventory(devices: tuple[Any, ...]) -> tuple[MatmulCollectiveDevice, ...]:
    return tuple(
        MatmulCollectiveDevice(
            id=int(device.id),
            process_index=int(device.process_index),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        for device in devices
    )


class _RejectMetadataRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise ValueError(
            f"MATMUL_COLLECTIVE_CONFIRMATION_METADATA_REDIRECT code={code} url={newurl}"
        )


def _metadata(path: str) -> str:
    request = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    opener = urllib.request.build_opener(_RejectMetadataRedirects())
    with opener.open(request, timeout=5) as response:
        if response.headers.get("Metadata-Flavor") != "Google":
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_METADATA_HEADER_MISSING")
        payload = response.read(4097)
        if len(payload) > 4096:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_METADATA_RESPONSE_TOO_LARGE")
        return payload.decode().strip()


def _host_identity() -> MatmulCollectiveHost:
    zone_resource = _metadata("instance/zone")
    machine_type_resource = _metadata("instance/machine-type")
    return MatmulCollectiveHost(
        project=_metadata("project/project-id"),
        numeric_project_id=_metadata("project/numeric-project-id"),
        zone=zone_resource.rsplit("/", maxsplit=1)[-1],
        hostname=_metadata("instance/name"),
        instance_hostname=_metadata("instance/hostname"),
        machine_type=machine_type_resource.rsplit("/", maxsplit=1)[-1],
        instance_id=_metadata("instance/id"),
        cpu_platform=_metadata("instance/cpu-platform"),
        zone_resource=zone_resource,
        machine_type_resource=machine_type_resource,
    )


def _host_matches_contract(
    host: MatmulCollectiveHost,
    contract: MatmulCollectiveConfirmationContract,
) -> bool:
    return host == MatmulCollectiveHost(
        project=contract.project,
        numeric_project_id=contract.numeric_project_id,
        zone=contract.zone,
        hostname=contract.hostname,
        instance_hostname=contract.instance_hostname,
        machine_type=contract.machine_type,
        instance_id=contract.instance_id,
        cpu_platform=contract.cpu_platform,
        zone_resource=f"projects/{contract.numeric_project_id}/zones/{contract.zone}",
        machine_type_resource=(
            f"projects/{contract.numeric_project_id}/machineTypes/{contract.machine_type}"
        ),
    )


def _validate_devices(
    devices: tuple[Any, ...],
    contract: MatmulCollectiveConfirmationContract,
) -> None:
    if (
        jax.default_backend() != contract.backend
        or len(devices) != contract.device_count
        or any(device.platform != contract.backend for device in devices)
        or any(device.device_kind != contract.device_kind for device in devices)
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_DEVICE_MISMATCH")


def _validate_diagnostics(
    diagnostic_root: Path,
    diagnostic_archive: Path,
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[bytes, ...]:
    if _sha256(diagnostic_archive) != contract.diagnostic_archive_sha256:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_DIAGNOSTIC_ARCHIVE_MISMATCH")
    receipts = []
    for authority in contract.diagnostics:
        bundle = diagnostic_root / (
            "xla"
            if authority.strategy is MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
            else "pallas"
        )
        receipt_path = bundle / "receipt.json"
        if _sha256(receipt_path) != authority.receipt_sha256:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_DIAGNOSTIC_RECEIPT_MISMATCH")
        receipt = RunReceipt.model_validate_json(receipt_path.read_text())
        experiment = KernelExperiment.model_validate_json(
            (bundle / "timing" / "experiment.json").read_text()
        )
        validate_receipt(receipt, experiment, root=bundle)
        artifacts = {artifact.path: artifact for artifact in receipt.artifacts}
        source_state_path = bundle / "timing" / "source_state.json"
        source_state_artifact = artifacts.get("timing/source_state.json")
        source_state = json.loads(source_state_path.read_text())
        expected = {
            "timing/result.json": authority.timing_result_sha256,
            "profile_assessment.json": authority.profile_assessment_sha256,
            "timing/physical.xdsl": authority.schedule_sha256,
            "timing/lowered_pallas.py": authority.pallas_source_sha256,
            "timing/stablehlo.txt": authority.stablehlo_sha256,
            "timing/compiler_hlo.txt": authority.compiler_hlo_sha256,
        }
        artifact_mismatch = any(
            artifacts.get(path) is None or artifacts[path].sha256 != digest
            for path, digest in expected.items()
        )
        if (
            receipt.experiment_id != authority.experiment_id
            or receipt.schedule_sha256 != authority.schedule_sha256
            or source_state_artifact is None
            or source_state_artifact.sha256 != _sha256(source_state_path)
            or source_state.get("git_commit") != authority.source_commit
            or artifact_mismatch
        ):
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_DIAGNOSTIC_IDENTITY_MISMATCH")
        receipts.append(receipt_path.read_bytes())
    return tuple(receipts)


def _plan_sources(
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[str, tuple[MatmulCollectivePlanSource, ...]]:
    values = contract.parameters
    distributed = distributed_matmul_schedule(
        mesh_size=int(values["mesh_size"]),
        m=int(values["m"]),
        k=int(values["k"]),
        n=int(values["n"]),
    )
    distributed.verify()
    distributed_text = canonical_text(distributed)
    sources = []
    for strategy in (contract.baseline, contract.candidate):
        physical = lower_distributed_matmul(
            distributed,
            tile=MatmulTile(int(values["tile_m"]), int(values["tile_n"])),
            collective_implementation=strategy.lowering_implementation(),
        )
        plan = lower_physical_matmul_to_pallas(physical)
        authority = next(value for value in contract.diagnostics if value.strategy is strategy)
        if (
            plan.tile_k != int(values["tile_k"])
            or plan.schedule_sha256 != authority.schedule_sha256
            or plan.source_sha256() != authority.pallas_source_sha256
        ):
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_PLAN_MISMATCH strategy={strategy}"
            )
        sources.append(
            MatmulCollectivePlanSource(
                strategy=strategy,
                plan=plan,
                physical_text=canonical_text(physical),
            )
        )
    return distributed_text, tuple(sources)


def _prepare_plans(
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[str, tuple[PreparedMatmulCollective, ...]]:
    distributed_text, sources = _plan_sources(contract)
    prepared = []
    for source in sources:
        executable, mesh = source.plan.build(interpret=False)
        prepared.append(
            PreparedMatmulCollective(
                source=source,
                executable=executable,
                mesh=mesh,
            )
        )
    return distributed_text, tuple(prepared)


def _quantized_inputs(
    contract: MatmulCollectiveConfirmationContract,
    *,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = contract.parameters
    workload_identity = semantic_sha256(
        "distributed-matmul-workload",
        str(values["mesh_size"]),
        str(values["m"]),
        str(values["k"]),
        str(values["n"]),
    )
    generator = (
        workload_rng(workload_identity, "device-run", "attempt-0", "inputs")
        if seed is None
        else workload_rng(
            workload_identity,
            "collective-confirmation",
            f"seed-{seed}",
            "inputs",
        )
    )
    lhs = generator.normal(size=(int(values["m"]), int(values["k"]))).astype(np.float32)
    rhs = generator.normal(size=(int(values["k"]), int(values["n"]))).astype(np.float32)
    lhs = lhs.astype(ml_dtypes.bfloat16).astype(np.float32)
    rhs = rhs.astype(ml_dtypes.bfloat16).astype(np.float32)
    return lhs, rhs


def _resident_inputs(
    lhs: np.ndarray,
    rhs: np.ndarray,
    plan: PallasMatmulPlan,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array]:
    return (
        jax.device_put(
            jnp.asarray(lhs, dtype=jnp.bfloat16),
            NamedSharding(mesh, PartitionSpec(None, plan.mesh_axis)),
        ),
        jax.device_put(
            jnp.asarray(rhs, dtype=jnp.bfloat16),
            NamedSharding(mesh, PartitionSpec(plan.mesh_axis, None)),
        ),
    )


def _compile(
    root: Path,
    contract: MatmulCollectiveConfirmationContract,
    prepared: tuple[PreparedMatmulCollective, ...],
    resident: tuple[jax.Array, jax.Array],
) -> tuple[CompiledMatmulCollective, ...]:
    compiled = []
    for value in prepared:
        lowered = value.executable.lower(*resident)
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo")).rstrip("\n") + "\n"
        executable = lowered.compile()
        compiler_hlo = executable.as_text()
        if not isinstance(compiler_hlo, str) or not compiler_hlo:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_COMPILER_HLO_UNAVAILABLE")
        compiler_hlo = compiler_hlo.rstrip("\n") + "\n"
        authority = next(
            item
            for item in contract.diagnostics
            if item.strategy is value.source.strategy
        )
        if (
            _text_sha256(stablehlo) != authority.stablehlo_sha256
            or _text_sha256(compiler_hlo) != authority.compiler_hlo_sha256
        ):
            raise ValueError(
                "MATMUL_COLLECTIVE_CONFIRMATION_HLO_MISMATCH "
                f"strategy={value.source.strategy}"
            )
        _validate_matmul_compiler_strategy(
            stablehlo,
            compiler_hlo,
            value.source.strategy,
        )
        strategy_root = root / "plans" / value.source.strategy
        _write_text(strategy_root / "physical.xdsl", value.source.physical_text)
        _write_text(
            strategy_root / "lowered_pallas.py",
            value.source.plan.render_executable_source(),
        )
        _write_json(strategy_root / "plan_manifest.json", value.source.plan.manifest())
        _write_text(strategy_root / "stablehlo.txt", stablehlo)
        _write_text(strategy_root / "compiler_hlo.txt", compiler_hlo)
        compiled.append(
            CompiledMatmulCollective(
                prepared=value,
                executable=executable,
                stablehlo=stablehlo,
                compiler_hlo=compiler_hlo,
            )
        )
    return tuple(compiled)


def _validate_output(
    output: jax.Array,
    contract: MatmulCollectiveConfirmationContract,
) -> None:
    values = contract.parameters
    expected_shape = (int(values["m"]), int(values["n"]))
    sharding = getattr(output, "sharding", None)
    if (
        tuple(output.shape) != expected_shape
        or output.dtype != jnp.float32
        or not isinstance(sharding, NamedSharding)
        or sharding.spec != PartitionSpec(None, "t")
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_OUTPUT_ABI_MISMATCH")


def _execute(
    executable: Any,
    resident: tuple[jax.Array, jax.Array],
    contract: MatmulCollectiveConfirmationContract,
) -> np.ndarray:
    output = executable(*resident)
    output.block_until_ready()
    _validate_output(output, contract)
    return np.asarray(output)


def _assessment(
    output: np.ndarray,
    oracle: np.ndarray,
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[bool, float, float]:
    values = contract.parameters
    expected_shape = (int(values["m"]), int(values["n"]))
    if (
        output.shape != expected_shape
        or oracle.shape != expected_shape
        or output.dtype != np.dtype(np.float32)
        or oracle.dtype != np.dtype(np.float32)
        or not output.flags.c_contiguous
        or not oracle.flags.c_contiguous
    ):
        return False, float("inf"), float("inf")
    absolute = np.abs(output - oracle)
    denominator = np.maximum(np.abs(oracle), np.finfo(np.float32).tiny)
    return (
        bool(
            np.allclose(
                output,
                oracle,
                atol=contract.absolute_tolerance,
                rtol=contract.relative_tolerance,
            )
        ),
        float(absolute.max()),
        float((absolute / denominator).max()),
    )


def _correctness(
    root: Path,
    contract: MatmulCollectiveConfirmationContract,
    compiled: Mapping[MatmulCollectiveStrategy, CompiledMatmulCollective],
) -> tuple[
    tuple[tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy], ...],
    tuple[MatmulCollectiveCorrectnessObservation, ...],
]:
    by_strategy: dict[MatmulCollectiveStrategy, list[MatmulCollectiveCorrectnessObservation]] = {
        contract.baseline: [],
        contract.candidate: [],
    }
    orders = []
    mesh = compiled[contract.baseline].prepared.mesh
    plan = compiled[contract.baseline].prepared.source.plan
    for index, seed in enumerate(contract.correctness_seeds):
        lhs, rhs = _quantized_inputs(contract, seed=seed)
        lhs_sha256 = array_sha256(lhs)
        rhs_sha256 = array_sha256(rhs)
        oracle = lhs @ rhs
        seed_root = root / "correctness" / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        np.save(seed_root / "oracle.npy", oracle, allow_pickle=False)
        resident = _resident_inputs(lhs, rhs, plan, mesh)
        order = (
            (contract.baseline, contract.candidate)
            if index % 2 == 0
            else (contract.candidate, contract.baseline)
        )
        orders.append(order)
        for strategy in order:
            output = _execute(compiled[strategy].executable, resident, contract)
            passed, maximum_absolute_error, maximum_relative_error = _assessment(
                output,
                oracle,
                contract,
            )
            if not passed:
                raise ValueError(
                    "MATMUL_COLLECTIVE_CONFIRMATION_CORRECTNESS_FAILED "
                    f"strategy={strategy} seed={seed} "
                    f"maximum_absolute_error={maximum_absolute_error} "
                    f"maximum_relative_error={maximum_relative_error}"
                )
            np.save(seed_root / f"{strategy}.npy", output, allow_pickle=False)
            by_strategy[strategy].append(
                MatmulCollectiveCorrectnessObservation(
                    strategy=strategy,
                    seed=seed,
                    lhs_sha256=lhs_sha256,
                    rhs_sha256=rhs_sha256,
                    oracle_sha256=array_sha256(oracle),
                    output_sha256=array_sha256(output),
                    maximum_absolute_error=maximum_absolute_error,
                    maximum_relative_error=maximum_relative_error,
                    passed=True,
                )
            )
        del resident, lhs, rhs, oracle, output
        gc.collect()
    observations = tuple(
        observation
        for strategy in (contract.baseline, contract.candidate)
        for observation in by_strategy[strategy]
    )
    return tuple(orders), observations


def _timing_observations(
    contract: MatmulCollectiveConfirmationContract,
    compiled: Mapping[MatmulCollectiveStrategy, CompiledMatmulCollective],
    resident: tuple[jax.Array, jax.Array],
) -> tuple[MatmulCollectiveTimingRound, ...]:
    observations = []
    for round_index, order in enumerate(collective_confirmation_orders(contract)):
        for position, strategy in enumerate(order):
            samples = []
            for _ in range(contract.calls_per_position):
                started = time.perf_counter_ns()
                output = compiled[strategy].executable(*resident)
                output.block_until_ready()
                samples.append(time.perf_counter_ns() - started)
            observations.append(
                MatmulCollectiveTimingRound(
                    round_index=round_index,
                    position=position,
                    strategy=strategy,
                    samples_ns=tuple(samples),
                    median_ns=float(statistics.median(samples)),
                )
            )
    return tuple(observations)


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    fixed = {
        "run_identity.json": ArtifactRole.EXPERIMENT,
        "contract.json": ArtifactRole.EXPERIMENT,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_authority.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "source_manifest.json": ArtifactRole.BACKEND_MANIFEST,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "correctness.json": ArtifactRole.SEARCH_EVIDENCE,
        "timing_input.json": ArtifactRole.CORRECTNESS_INPUT,
        "timing_started.json": ArtifactRole.INVOCATION,
        "rounds.json": ArtifactRole.TIMING_SAMPLES,
        "result.json": ArtifactRole.SEARCH_RESULT,
    }
    if relative in fixed:
        return fixed[relative]
    if relative.startswith("diagnostics/") and path.name.endswith("receipt.json"):
        return ArtifactRole.SEARCH_EVIDENCE
    plan_roles = {
        "physical.xdsl": ArtifactRole.PHYSICAL_IR,
        "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
        "stablehlo.txt": ArtifactRole.STABLEHLO,
        "compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "pre_timing_output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "post_timing_output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
    }
    if relative.startswith("plans/") and path.name in plan_roles:
        return plan_roles[path.name]
    if relative.startswith("correctness/") and path.suffix == ".npy":
        return (
            ArtifactRole.ORACLE_OUTPUT
            if path.name == "oracle.npy"
            else ArtifactRole.CORRECTNESS_OUTPUT
        )
    raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_ARTIFACT_UNRECOGNIZED path={relative}")


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
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_CLOSED_WORLD_MISMATCH")
    for artifact in artifacts:
        path = root / artifact.path
        if (
            path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != artifact.size_bytes
            or _sha256(path) != artifact.sha256
            or _artifact_role(Path(artifact.path)) is not artifact.role
        ):
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_ARTIFACT_MISMATCH path={artifact.path}"
            )


def _expected_files(
    root: Path,
    contract: MatmulCollectiveConfirmationContract,
    *,
    receipt_present: bool,
) -> set[Path]:
    names = {
        "run_identity.json",
        "contract.json",
        "source_state.json",
        "source_authority.json",
        "source_diff.patch",
        "source_manifest.json",
        "ledger.sqlite",
        "distributed.xdsl",
        "correctness.json",
        "timing_input.json",
        "timing_started.json",
        "rounds.json",
        "result.json",
    }
    expected = {root / name for name in names}
    expected.update(
        root / "diagnostics" / f"{strategy}-receipt.json"
        for strategy in (contract.baseline, contract.candidate)
    )
    plan_names = (
        "physical.xdsl",
        "lowered_pallas.py",
        "plan_manifest.json",
        "stablehlo.txt",
        "compiler_hlo.txt",
        "pre_timing_output.npy",
        "post_timing_output.npy",
    )
    expected.update(
        root / "plans" / strategy / name
        for strategy in (contract.baseline, contract.candidate)
        for name in plan_names
    )
    expected.update(
        root / "correctness" / f"seed-{seed}" / name
        for seed in contract.correctness_seeds
        for name in (
            "oracle.npy",
            f"{contract.baseline}.npy",
            f"{contract.candidate}.npy",
        )
    )
    if receipt_present:
        expected.add(root / "receipt.json")
    return {path.resolve() for path in expected}


def _record_state(
    ledger_path: Path,
    run_id: str,
    state: RunState,
    payload: Mapping[str, object],
) -> None:
    EvidenceRun(ledger_path, run_id).record(
        state,
        payload,
        conflict_error="MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_CONFLICT state={state}",
    )


def _attempt_claim_path(
    contract: MatmulCollectiveConfirmationContract,
    run_id: str,
) -> Path:
    return Path(contract.attempt_registry_root) / f"{run_id}.json"


def _timing_attempt_payload(
    root: Path,
    identity: MatmulCollectiveConfirmationRunIdentity,
    contract: MatmulCollectiveConfirmationContract,
    state: str,
) -> dict[str, object]:
    return {
        "schema": MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA,
        "confirmation_id": identity.confirmation_id,
        "run_id": identity.run_id,
        "source_commit": identity.source_commit,
        "output_root": str(root),
        "paired_rounds": contract.paired_rounds,
        "calls_per_position": contract.calls_per_position,
        "state": state,
    }


def _claim_timing_attempt(
    root: Path,
    identity: MatmulCollectiveConfirmationRunIdentity,
    contract: MatmulCollectiveConfirmationContract,
) -> str:
    registry = Path(contract.attempt_registry_root)
    _reject_symlink_components(registry)
    registry.mkdir(parents=True, mode=0o700, exist_ok=True)
    registry_stat = registry.lstat()
    if (
        not stat.S_ISDIR(registry_stat.st_mode)
        or registry_stat.st_uid != os.getuid()
        or registry_stat.st_mode & 0o077
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_ATTEMPT_REGISTRY_INVALID")
    reserved = _timing_attempt_payload(root, identity, contract, "reserved")
    started = _timing_attempt_payload(root, identity, contract, "started")
    claim_path = _attempt_claim_path(contract, identity.run_id)
    try:
        _write_json_exclusive(claim_path, reserved)
    except FileExistsError as error:
        if claim_path.is_symlink() or json.loads(claim_path.read_text()) != reserved:
            raise ValueError(
                "MATMUL_COLLECTIVE_CONFIRMATION_TIMING_ATTEMPT_ALREADY_CLAIMED "
                f"path={claim_path}"
            ) from error
    _write_json_atomic(root / "timing_started.json", started)
    _write_json_atomic(claim_path, started)
    return _sha256(root / "timing_started.json")


def _ledger_payloads(
    contract: MatmulCollectiveConfirmationContract,
    result: MatmulCollectiveConfirmationResult,
    root: Path,
    *,
    include_accepted: bool,
) -> tuple[tuple[RunState, dict[str, object]], ...]:
    payloads: tuple[tuple[RunState, dict[str, object]], ...] = (
        (
            RunState.CREATED,
            {
                "confirmation_id": contract.confirmation_id,
                "source_commit": result.source_commit,
                "diagnostic_archive_sha256": contract.diagnostic_archive_sha256,
                "diagnostic_receipt_sha256": list(result.diagnostic_receipt_sha256),
                "devices": [value.model_dump(mode="json") for value in result.devices],
            },
        ),
        (
            RunState.VERIFIED,
            {
                "source_manifest_sha256": result.source_manifest_sha256,
                "source_authority_sha256": result.source_authority_sha256,
                "timing_lhs_sha256": result.timing_lhs_sha256,
                "timing_rhs_sha256": result.timing_rhs_sha256,
            },
        ),
        (
            RunState.LOWERED,
            {
                "plans": {
                    value.strategy: {
                        "schedule_sha256": value.schedule_sha256,
                        "pallas_source_sha256": value.pallas_source_sha256,
                    }
                    for value in result.plans
                }
            },
        ),
        (
            RunState.COMPILED,
            {
                "compiler": {
                    value.strategy: {
                        "stablehlo_sha256": value.stablehlo_sha256,
                        "compiler_hlo_sha256": value.compiler_hlo_sha256,
                    }
                    for value in result.plans
                }
            },
        ),
        (
            RunState.CORRECT,
            {
                "correctness_sha256": _sha256(root / "correctness.json"),
                "pre_timing_outputs": {
                    value.strategy: value.sha256 for value in result.pre_timing_outputs
                },
            },
        ),
        (
            RunState.TIMED,
            {
                "rounds_sha256": _sha256(root / "rounds.json"),
                "statistics": result.statistics.model_dump(mode="json"),
                "post_timing_outputs": {
                    value.strategy: value.sha256 for value in result.post_timing_outputs
                },
                "timing_attempt_sha256": result.timing_attempt_sha256,
            },
        ),
    )
    if include_accepted:
        payloads += (
            (
                RunState.ACCEPTED,
                {
                    "result_sha256": _sha256(root / "result.json"),
                    "candidate_promoted": result.statistics.candidate_promoted,
                    "selected_strategy": result.statistics.selected_strategy,
                },
            ),
        )
    return payloads


def _validate_source(
    root: Path,
    identity: MatmulCollectiveConfirmationRunIdentity,
    result: MatmulCollectiveConfirmationResult,
    contract: MatmulCollectiveConfirmationContract,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    state = json.loads((root / "source_state.json").read_text())
    authority = json.loads((root / "source_authority.json").read_text())
    manifest = tuple(
        SourceFileContract.model_validate(value)
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    commit = _git(repository_root, "rev-parse", "HEAD")
    if (
        _git(repository_root, "status", "--porcelain=v1")
        or commit != identity.source_commit
        or result.source_commit != identity.source_commit
        or state.get("git_commit") != commit
        or state.get("git_dirty") is not False
        or state.get("git_status") != []
        or (root / "source_diff.patch").read_bytes()
        or result.source_state_sha256 != _sha256(root / "source_state.json")
        or result.source_authority_sha256 != _sha256(root / "source_authority.json")
        or authority
        != {
            "remote_url": contract.source_remote_url,
            "server_main_commit": identity.source_commit,
        }
        or result.source_manifest_sha256 != _sha256(root / "source_manifest.json")
        or result.source_manifest != manifest
        or manifest != _source_manifest()
        or state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_SOURCE_MISMATCH")
    for source in manifest:
        blob = subprocess.run(
            ["git", "show", f"{commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_SOURCE_BLOB_MISMATCH path={source.path}"
            )


def _replay_plans(
    root: Path,
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[MatmulCollectivePlan, ...]:
    distributed_text, sources = _plan_sources(contract)
    if (root / "distributed.xdsl").read_text() != distributed_text:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_DISTRIBUTED_REPLAY_MISMATCH")
    records = []
    for value in sources:
        strategy_root = root / "plans" / value.strategy
        stablehlo = (strategy_root / "stablehlo.txt").read_text()
        compiler_hlo = (strategy_root / "compiler_hlo.txt").read_text()
        authority = next(
            item for item in contract.diagnostics if item.strategy is value.strategy
        )
        if (
            (strategy_root / "physical.xdsl").read_text() != value.physical_text
            or (strategy_root / "lowered_pallas.py").read_text()
            != value.plan.render_executable_source()
            or json.loads((strategy_root / "plan_manifest.json").read_text())
            != value.plan.manifest()
            or _sha256(strategy_root / "physical.xdsl") != authority.schedule_sha256
            or _sha256(strategy_root / "lowered_pallas.py")
            != authority.pallas_source_sha256
            or _sha256(strategy_root / "stablehlo.txt") != authority.stablehlo_sha256
            or _sha256(strategy_root / "compiler_hlo.txt") != authority.compiler_hlo_sha256
        ):
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_PLAN_REPLAY_MISMATCH strategy={value.strategy}"
            )
        _validate_matmul_compiler_strategy(stablehlo, compiler_hlo, value.strategy)
        records.append(
            MatmulCollectivePlan(
                strategy=value.strategy,
                schedule_sha256=authority.schedule_sha256,
                pallas_source_sha256=authority.pallas_source_sha256,
                stablehlo_sha256=authority.stablehlo_sha256,
                compiler_hlo_sha256=authority.compiler_hlo_sha256,
            )
        )
    return tuple(records)


def _replay_correctness(
    root: Path,
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[MatmulCollectiveCorrectnessObservation, ...]:
    by_strategy: dict[MatmulCollectiveStrategy, list[MatmulCollectiveCorrectnessObservation]] = {
        contract.baseline: [],
        contract.candidate: [],
    }
    for seed in contract.correctness_seeds:
        lhs, rhs = _quantized_inputs(contract, seed=seed)
        oracle = lhs @ rhs
        seed_root = root / "correctness" / f"seed-{seed}"
        saved_oracle = np.load(seed_root / "oracle.npy", allow_pickle=False)
        if not np.array_equal(saved_oracle, oracle):
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_ORACLE_REPLAY_FAILED seed={seed}"
            )
        for strategy in (contract.baseline, contract.candidate):
            output = np.load(seed_root / f"{strategy}.npy", allow_pickle=False)
            passed, maximum_absolute_error, maximum_relative_error = _assessment(
                output,
                oracle,
                contract,
            )
            if not passed:
                raise ValueError(
                    "MATMUL_COLLECTIVE_CONFIRMATION_CORRECTNESS_REPLAY_FAILED "
                    f"strategy={strategy} seed={seed}"
                )
            by_strategy[strategy].append(
                MatmulCollectiveCorrectnessObservation(
                    strategy=strategy,
                    seed=seed,
                    lhs_sha256=array_sha256(lhs),
                    rhs_sha256=array_sha256(rhs),
                    oracle_sha256=array_sha256(oracle),
                    output_sha256=array_sha256(output),
                    maximum_absolute_error=maximum_absolute_error,
                    maximum_relative_error=maximum_relative_error,
                    passed=True,
                )
            )
        del lhs, rhs, oracle, saved_oracle, output
        gc.collect()
    return tuple(
        observation
        for strategy in (contract.baseline, contract.candidate)
        for observation in by_strategy[strategy]
    )


def _validate_ledger_database(path: Path, run_id: str, expected_rows: int) -> None:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_INTEGRITY_FAILED")
        objects = connection.execute(
            "SELECT type, name, tbl_name FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        if objects != [
            ("index", "sqlite_autoindex_events_1", "events"),
            ("table", "events", "events"),
            ("table", "sqlite_sequence", "sqlite_sequence"),
        ]:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_SCHEMA_MISMATCH")
        columns = connection.execute("PRAGMA table_info(events)").fetchall()
        if columns != [
            (0, "sequence", "INTEGER", 0, None, 1),
            (1, "run_id", "TEXT", 1, None, 0),
            (2, "state", "TEXT", 1, None, 0),
            (3, "timestamp_ns", "INTEGER", 1, None, 0),
            (4, "payload_sha256", "TEXT", 1, None, 0),
        ]:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_COLUMNS_MISMATCH")
        indexes = connection.execute("PRAGMA index_list(events)").fetchall()
        if indexes != [(0, "sqlite_autoindex_events_1", 1, "u", 0)]:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_INDEX_MISMATCH")
        index_columns = connection.execute(
            "PRAGMA index_info(sqlite_autoindex_events_1)"
        ).fetchall()
        if index_columns != [(0, 1, "run_id"), (1, 2, "state")]:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_INDEX_MISMATCH")
        rows = connection.execute(
            "SELECT sequence, run_id, timestamp_ns FROM events ORDER BY sequence"
        ).fetchall()
        sequence = connection.execute(
            "SELECT name, seq FROM sqlite_sequence ORDER BY name"
        ).fetchall()
    if (
        len(rows) != expected_rows
        or [value[0] for value in rows] != list(range(1, expected_rows + 1))
        or {value[1] for value in rows} != {run_id}
        or any(value[2] <= 0 for value in rows)
        or sequence != [("events", expected_rows)]
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_SCOPE_MISMATCH")


def _validate(
    root: Path,
    contract: MatmulCollectiveConfirmationContract,
    *,
    require_accepted: bool,
    require_receipt: bool,
) -> MatmulCollectiveConfirmationResult:
    _preflight_root(root)
    saved_contract = MatmulCollectiveConfirmationContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != contract:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_CONTRACT_MISMATCH")
    identity = MatmulCollectiveConfirmationRunIdentity.model_validate_json(
        (root / "run_identity.json").read_text()
    )
    result = MatmulCollectiveConfirmationResult.model_validate_json(
        (root / "result.json").read_text()
    )
    if (
        identity.confirmation_id != contract.confirmation_id
        or result.confirmation_id != contract.confirmation_id
        or result.run_id != identity.run_id
        or result.source_commit != identity.source_commit
        or not Path(result.producer_output_root).is_absolute()
        or result.diagnostic_archive_sha256 != contract.diagnostic_archive_sha256
        or result.diagnostic_receipt_sha256
        != tuple(value.receipt_sha256 for value in contract.diagnostics)
        or result.runtime != contract.runtime
        or not _host_matches_contract(result.host, contract)
        or result.xla_flags != contract.xla_flags
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_RESULT_IDENTITY_MISMATCH")
    if identity.run_id != semantic_sha256(
        MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA,
        contract.confirmation_id,
        identity.source_commit,
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_RUN_ID_MISMATCH")
    expected_devices = tuple(
        MatmulCollectiveDevice(
            id=index,
            process_index=0,
            platform=contract.backend,
            device_kind=contract.device_kind,
        )
        for index in range(contract.device_count)
    )
    if result.devices != expected_devices:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_DEVICE_INVENTORY_MISMATCH")
    _validate_source(root, identity, result, contract)
    for authority in contract.diagnostics:
        receipt_path = root / "diagnostics" / f"{authority.strategy}-receipt.json"
        if _sha256(receipt_path) != authority.receipt_sha256:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_DIAGNOSTIC_COPY_MISMATCH")
    plans = _replay_plans(root, contract)
    if result.plans != plans:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_PLAN_IDENTITY_MISMATCH")
    correctness = _replay_correctness(root, contract)
    correctness_file = tuple(
        MatmulCollectiveCorrectnessObservation.model_validate(value)
        for value in json.loads((root / "correctness.json").read_text())
    )
    expected_correctness_orders = tuple(
        (contract.baseline, contract.candidate)
        if index % 2 == 0
        else (contract.candidate, contract.baseline)
        for index in range(len(contract.correctness_seeds))
    )
    if (
        result.correctness_execution_orders != expected_correctness_orders
        or result.correctness != correctness
        or correctness_file != correctness
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_CORRECTNESS_MISMATCH")
    lhs, rhs = _quantized_inputs(contract, seed=None)
    if (
        array_sha256(lhs) != contract.timing_lhs_sha256
        or array_sha256(rhs) != contract.timing_rhs_sha256
        or result.timing_lhs_sha256 != contract.timing_lhs_sha256
        or result.timing_rhs_sha256 != contract.timing_rhs_sha256
        or json.loads((root / "timing_input.json").read_text())
        != {
            "schema": contract.timing_input_schema,
            "lhs_sha256": contract.timing_lhs_sha256,
            "rhs_sha256": contract.timing_rhs_sha256,
        }
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_INPUT_MISMATCH")
    oracle = lhs @ rhs
    pre_outputs = []
    post_outputs = []
    for strategy in (contract.baseline, contract.candidate):
        strategy_root = root / "plans" / strategy
        pre = np.load(strategy_root / "pre_timing_output.npy", allow_pickle=False)
        post = np.load(strategy_root / "post_timing_output.npy", allow_pickle=False)
        pre_passed, _, _ = _assessment(pre, oracle, contract)
        post_passed, _, _ = _assessment(post, oracle, contract)
        if not pre_passed or not post_passed or array_sha256(pre) != array_sha256(post):
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_TIMING_OUTPUT_MISMATCH strategy={strategy}"
            )
        pre_outputs.append(
            MatmulCollectiveTimingOutput(strategy=strategy, sha256=array_sha256(pre))
        )
        post_outputs.append(
            MatmulCollectiveTimingOutput(strategy=strategy, sha256=array_sha256(post))
        )
    rounds = tuple(
        MatmulCollectiveTimingRound.model_validate(value)
        for value in json.loads((root / "rounds.json").read_text())
    )
    statistics_record = collective_confirmation_statistics(contract, rounds)
    timing_attempt = json.loads((root / "timing_started.json").read_text())
    if (
        result.timing_attempt_sha256 != _sha256(root / "timing_started.json")
        or timing_attempt
        != _timing_attempt_payload(
            Path(result.producer_output_root),
            identity,
            contract,
            "started",
        )
        or result.pre_timing_outputs != tuple(pre_outputs)
        or result.post_timing_outputs != tuple(post_outputs)
        or result.execution_orders != collective_confirmation_orders(contract)
        or result.rounds != rounds
        or result.statistics != statistics_record
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_STATISTICS_REPLAY_MISMATCH")
    history = read_ledger_history(root / "ledger.sqlite", result.run_id)
    payloads = _ledger_payloads(
        contract,
        result,
        root,
        include_accepted=require_accepted,
    )
    _validate_ledger_database(root / "ledger.sqlite", result.run_id, len(payloads))
    if tuple(value.state for value in history) != tuple(state for state, _ in payloads):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_STATE_MISMATCH")
    if tuple(value.payload_sha256 for value in history) != tuple(
        payload_sha256(payload) for _, payload in payloads
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_PAYLOAD_MISMATCH")
    expected_files = _expected_files(root, contract, receipt_present=require_receipt)
    observed_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_CLOSED_WORLD_MISMATCH")
    if require_receipt:
        receipt = MatmulCollectiveConfirmationReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        _validate_manifest(root, receipt.artifacts)
        expected_receipt = MatmulCollectiveConfirmationReceipt(
            confirmation_id=contract.confirmation_id,
            run_id=result.run_id,
            result_sha256=_sha256(root / "result.json"),
            ledger_sha256=_sha256(root / "ledger.sqlite"),
            artifacts=_artifact_manifest(root),
        )
        if receipt != expected_receipt:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_RECEIPT_MISMATCH")
    return result


def _prepare_output_root(
    root: Path,
    identity: MatmulCollectiveConfirmationRunIdentity,
    contract: MatmulCollectiveConfirmationContract,
) -> RunState | None:
    claim_path = _attempt_claim_path(contract, identity.run_id)
    reserved_claim = _timing_attempt_payload(root, identity, contract, "reserved")
    started_claim = _timing_attempt_payload(root, identity, contract, "started")
    claimed = claim_path.exists() or claim_path.is_symlink()
    claim = (
        json.loads(claim_path.read_text())
        if claimed and claim_path.is_file() and not claim_path.is_symlink()
        else None
    )
    if not root.exists():
        if claimed and claim != reserved_claim:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_ATTEMPT_NOT_RETRYABLE")
        root.mkdir(parents=True, exist_ok=False)
        return None
    _preflight_root(root)
    if not any(root.iterdir()):
        if claimed and claim != reserved_claim:
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_ATTEMPT_NOT_RETRYABLE")
        return None
    identity_path = root / "run_identity.json"
    if not identity_path.is_file() or (
        MatmulCollectiveConfirmationRunIdentity.model_validate_json(identity_path.read_text())
        != identity
    ):
        raise ValueError(f"MATMUL_COLLECTIVE_CONFIRMATION_ROOT_NOT_OWNED path={root}")
    marker_path = root / "timing_started.json"
    if marker_path.is_file():
        if not claim_path.is_file() or claim_path.is_symlink():
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_CLAIM_MISSING")
        if (
            json.loads(marker_path.read_text()) != started_claim
            or json.loads(claim_path.read_text()) != started_claim
        ):
            raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_CLAIM_MISMATCH")
        ledger_path = root / "ledger.sqlite"
        states = (
            tuple(value.state for value in read_ledger_history(ledger_path, identity.run_id))
            if ledger_path.is_file()
            else ()
        )
        if states and states[-1] in {RunState.TIMED, RunState.ACCEPTED}:
            return states[-1]
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_ATTEMPT_NOT_RETRYABLE")
    if claimed and claim != reserved_claim:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_ATTEMPT_NOT_RETRYABLE")
    while True:
        archived = root.with_name(f"{root.name}.incomplete-{time.time_ns()}")
        if not archived.exists() and not archived.is_symlink():
            break
    root.rename(archived)
    root.mkdir(parents=True, exist_ok=False)
    print(f"MATMUL_COLLECTIVE_CONFIRMATION_ARCHIVED source={root} archive={archived}")
    return None


def _record_failure(root: Path, run_id: str, error: Exception) -> None:
    if not (root / "run_identity.json").is_file():
        return
    ledger_path = root / "ledger.sqlite"
    if ledger_path.is_file():
        with ExperimentLedger(ledger_path) as ledger:
            state = ledger.current_state(run_id)
        if state in {RunState.TIMED, RunState.ACCEPTED}:
            return
    _write_json_atomic(
        root / "failure.json",
        {
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    if not ledger_path.is_file():
        return
    with ExperimentLedger(ledger_path) as ledger:
        state = ledger.current_state(run_id)
        if state not in {None, RunState.TIMED, RunState.ACCEPTED, RunState.REJECTED}:
            ledger.transition(
                run_id,
                RunState.REJECTED,
                {"error_type": type(error).__name__, "error": str(error)},
            )
    seal_ledger(ledger_path, "MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_SIDECARS")


def _publish_receipt(
    root: Path,
    contract: MatmulCollectiveConfirmationContract,
    result: MatmulCollectiveConfirmationResult,
) -> MatmulCollectiveConfirmationResult:
    receipt = MatmulCollectiveConfirmationReceipt(
        confirmation_id=contract.confirmation_id,
        run_id=result.run_id,
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
    contract: MatmulCollectiveConfirmationContract,
    identity: MatmulCollectiveConfirmationRunIdentity,
    source_authority: Mapping[str, str],
    host: MatmulCollectiveHost,
    devices: tuple[Any, ...],
    diagnostic_receipts: tuple[bytes, ...],
) -> MatmulCollectiveConfirmationResult:
    repository_root = Path(__file__).resolve().parents[2]
    _write_json(
        root / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    _source_state(repository_root, root)
    _write_json(root / "source_authority.json", source_authority)
    manifest = _source_manifest()
    _write_json(
        root / "source_manifest.json",
        [value.model_dump(mode="json") for value in manifest],
    )
    for authority, receipt in zip(contract.diagnostics, diagnostic_receipts, strict=True):
        path = root / "diagnostics" / f"{authority.strategy}-receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(receipt)
    device_inventory = _device_inventory(devices)
    ledger_path = root / "ledger.sqlite"
    _record_state(
        ledger_path,
        identity.run_id,
        RunState.CREATED,
        {
            "confirmation_id": contract.confirmation_id,
            "source_commit": identity.source_commit,
            "diagnostic_archive_sha256": contract.diagnostic_archive_sha256,
            "diagnostic_receipt_sha256": [
                value.receipt_sha256 for value in contract.diagnostics
            ],
            "devices": [value.model_dump(mode="json") for value in device_inventory],
        },
    )
    distributed_text, prepared = _prepare_plans(contract)
    _write_text(root / "distributed.xdsl", distributed_text)
    lhs, rhs = _quantized_inputs(contract, seed=None)
    lhs_sha256 = array_sha256(lhs)
    rhs_sha256 = array_sha256(rhs)
    if lhs_sha256 != contract.timing_lhs_sha256 or rhs_sha256 != contract.timing_rhs_sha256:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_TIMING_INPUT_IDENTITY_MISMATCH")
    _write_json(
        root / "timing_input.json",
        {
            "schema": contract.timing_input_schema,
            "lhs_sha256": lhs_sha256,
            "rhs_sha256": rhs_sha256,
        },
    )
    _record_state(
        ledger_path,
        identity.run_id,
        RunState.VERIFIED,
        {
            "source_manifest_sha256": _sha256(root / "source_manifest.json"),
            "source_authority_sha256": _sha256(root / "source_authority.json"),
            "timing_lhs_sha256": lhs_sha256,
            "timing_rhs_sha256": rhs_sha256,
        },
    )
    plan_records = tuple(
        MatmulCollectivePlan(
            strategy=value.source.strategy,
            schedule_sha256=value.source.plan.schedule_sha256,
            pallas_source_sha256=value.source.plan.source_sha256(),
            stablehlo_sha256=next(
                item.stablehlo_sha256
                for item in contract.diagnostics
                if item.strategy is value.source.strategy
            ),
            compiler_hlo_sha256=next(
                item.compiler_hlo_sha256
                for item in contract.diagnostics
                if item.strategy is value.source.strategy
            ),
        )
        for value in prepared
    )
    _record_state(
        ledger_path,
        identity.run_id,
        RunState.LOWERED,
        {
            "plans": {
                value.strategy: {
                    "schedule_sha256": value.schedule_sha256,
                    "pallas_source_sha256": value.pallas_source_sha256,
                }
                for value in plan_records
            }
        },
    )
    resident = _resident_inputs(
        lhs,
        rhs,
        prepared[0].source.plan,
        prepared[0].mesh,
    )
    compiled_values = _compile(root, contract, prepared, resident)
    compiled = {value.strategy: value for value in compiled_values}
    _record_state(
        ledger_path,
        identity.run_id,
        RunState.COMPILED,
        {
            "compiler": {
                value.strategy: {
                    "stablehlo_sha256": value.stablehlo_sha256,
                    "compiler_hlo_sha256": value.compiler_hlo_sha256,
                }
                for value in plan_records
            }
        },
    )
    correctness_orders, correctness = _correctness(root, contract, compiled)
    _write_json(
        root / "correctness.json",
        [value.model_dump(mode="json") for value in correctness],
    )
    timing_oracle = lhs @ rhs
    pre_outputs = []
    for strategy in (contract.baseline, contract.candidate):
        output = _execute(compiled[strategy].executable, resident, contract)
        passed, _, _ = _assessment(output, timing_oracle, contract)
        if not passed:
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_PRE_TIMING_FAILED strategy={strategy}"
            )
        np.save(
            root / "plans" / strategy / "pre_timing_output.npy",
            output,
            allow_pickle=False,
        )
        pre_outputs.append(
            MatmulCollectiveTimingOutput(strategy=strategy, sha256=array_sha256(output))
        )
    _record_state(
        ledger_path,
        identity.run_id,
        RunState.CORRECT,
        {
            "correctness_sha256": _sha256(root / "correctness.json"),
            "pre_timing_outputs": {
                value.strategy: value.sha256 for value in pre_outputs
            },
        },
    )
    for warmup_index in range(contract.warmup_iterations):
        order = (
            (contract.baseline, contract.candidate)
            if warmup_index % 2 == 0
            else (contract.candidate, contract.baseline)
        )
        for strategy in order:
            compiled[strategy].executable(*resident).block_until_ready()
    timing_attempt_sha256 = _claim_timing_attempt(root, identity, contract)
    rounds = _timing_observations(contract, compiled, resident)
    statistics_record = collective_confirmation_statistics(contract, rounds)
    _write_json(root / "rounds.json", [value.model_dump(mode="json") for value in rounds])
    post_outputs = []
    for strategy in (contract.baseline, contract.candidate):
        output = _execute(compiled[strategy].executable, resident, contract)
        passed, _, _ = _assessment(output, timing_oracle, contract)
        pre_sha256 = next(value.sha256 for value in pre_outputs if value.strategy is strategy)
        if not passed or array_sha256(output) != pre_sha256:
            raise ValueError(
                f"MATMUL_COLLECTIVE_CONFIRMATION_POST_TIMING_FAILED strategy={strategy}"
            )
        np.save(
            root / "plans" / strategy / "post_timing_output.npy",
            output,
            allow_pickle=False,
        )
        post_outputs.append(
            MatmulCollectiveTimingOutput(strategy=strategy, sha256=array_sha256(output))
        )
    result = MatmulCollectiveConfirmationResult(
        confirmation_id=contract.confirmation_id,
        run_id=identity.run_id,
        source_commit=identity.source_commit,
        producer_output_root=str(root),
        diagnostic_archive_sha256=contract.diagnostic_archive_sha256,
        diagnostic_receipt_sha256=tuple(
            value.receipt_sha256 for value in contract.diagnostics
        ),
        runtime=contract.runtime,
        host=host,
        xla_flags=os.environ.get("XLA_FLAGS"),
        devices=device_inventory,
        source_state_sha256=_sha256(root / "source_state.json"),
        source_authority_sha256=_sha256(root / "source_authority.json"),
        source_manifest_sha256=_sha256(root / "source_manifest.json"),
        source_manifest=manifest,
        plans=plan_records,
        correctness_execution_orders=correctness_orders,
        correctness=correctness,
        timing_lhs_sha256=lhs_sha256,
        timing_rhs_sha256=rhs_sha256,
        pre_timing_outputs=tuple(pre_outputs),
        execution_orders=collective_confirmation_orders(contract),
        rounds=rounds,
        post_timing_outputs=tuple(post_outputs),
        timing_attempt_sha256=timing_attempt_sha256,
        statistics=statistics_record,
        claim_scope=(
            "mesh8-m1024-k65536-n1024-bf16-f32-"
            "tile-mn128-k8192-standalone-matmul-collective"
        ),
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    _record_state(
        ledger_path,
        identity.run_id,
        RunState.TIMED,
        _ledger_payloads(contract, result, root, include_accepted=False)[-1][1],
    )
    seal_ledger(ledger_path, "MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_SIDECARS")
    _validate(
        root,
        contract,
        require_accepted=False,
        require_receipt=False,
    )
    _record_state(
        ledger_path,
        identity.run_id,
        RunState.ACCEPTED,
        {
            "result_sha256": _sha256(root / "result.json"),
            "candidate_promoted": statistics_record.candidate_promoted,
            "selected_strategy": statistics_record.selected_strategy,
        },
    )
    seal_ledger(ledger_path, "MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_SIDECARS")
    return _publish_receipt(root, contract, result)


def run_matmul_collective_confirmation(
    root: Path,
    diagnostic_root: Path,
    diagnostic_archive: Path,
    contract: MatmulCollectiveConfirmationContract,
) -> MatmulCollectiveConfirmationResult:
    canonical = default_matmul_collective_confirmation_contract(contract.runtime)
    if contract != canonical:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_EXTERNAL_CONTRACT_MISMATCH")
    _reject_symlink_components(root)
    _reject_symlink_components(diagnostic_root)
    _reject_symlink_components(diagnostic_archive)
    root = root.resolve()
    _require_safe_root(root)
    repository_root = Path(__file__).resolve().parents[2]
    source_commit, source_authority = _require_source(repository_root, contract)
    runtime = _runtime_identity()
    host = _host_identity()
    if (
        runtime != contract.runtime
        or not _host_matches_contract(host, contract)
        or os.environ.get("XLA_FLAGS") != contract.xla_flags
    ):
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_RUNTIME_MISMATCH")
    devices = tuple(jax.devices())
    _validate_devices(devices, contract)
    diagnostic_receipts = _validate_diagnostics(
        diagnostic_root.resolve(),
        diagnostic_archive.resolve(),
        contract,
    )
    identity = MatmulCollectiveConfirmationRunIdentity(
        confirmation_id=contract.confirmation_id,
        run_id=semantic_sha256(
            MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA,
            contract.confirmation_id,
            source_commit,
        ),
        source_commit=source_commit,
    )
    with _exclusive_run_lock(identity.run_id):
        if (root / "receipt.json").is_file():
            return validate_matmul_collective_confirmation(root, contract)
        state = _prepare_output_root(root, identity, contract)
        _write_json_atomic(root / "run_identity.json", identity.model_dump(mode="json"))
        try:
            if state in {RunState.TIMED, RunState.ACCEPTED}:
                result = _validate(
                    root,
                    contract,
                    require_accepted=state is RunState.ACCEPTED,
                    require_receipt=False,
                )
                if state is RunState.TIMED:
                    _record_state(
                        root / "ledger.sqlite",
                        result.run_id,
                        RunState.ACCEPTED,
                        {
                            "result_sha256": _sha256(root / "result.json"),
                            "candidate_promoted": result.statistics.candidate_promoted,
                            "selected_strategy": result.statistics.selected_strategy,
                        },
                    )
                    seal_ledger(
                        root / "ledger.sqlite",
                        "MATMUL_COLLECTIVE_CONFIRMATION_LEDGER_SIDECARS",
                    )
                return _publish_receipt(root, contract, result)
            return _execute_confirmation(
                root,
                contract,
                identity,
                source_authority,
                host,
                devices,
                diagnostic_receipts,
            )
        except Exception as error:
            _record_failure(root, identity.run_id, error)
            raise


def validate_matmul_collective_confirmation(
    root: Path,
    trusted_contract: MatmulCollectiveConfirmationContract,
) -> MatmulCollectiveConfirmationResult:
    canonical = default_matmul_collective_confirmation_contract(trusted_contract.runtime)
    if trusted_contract != canonical:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_EXTERNAL_CONTRACT_MISMATCH")
    _reject_symlink_components(root)
    root = root.resolve()
    if _runtime_identity().jax != trusted_contract.runtime.jax:
        raise ValueError("MATMUL_COLLECTIVE_CONFIRMATION_VERIFIER_RUNTIME_MISMATCH")
    return _validate(
        root,
        trusted_contract,
        require_accepted=True,
        require_receipt=True,
    )
