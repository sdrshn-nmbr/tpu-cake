from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tpu_cake.artifacts import build_artifact_manifest, file_sha256
from tpu_cake.contracts import ArtifactRole, SourceFileContract
from tpu_cake.ledger import EvidenceRun, RunState, finalize_ledger
from tpu_cake.seqax_silu_fusion_correctness import (
    SeqaxSiluFusionCorrectnessArchiveSeal,
    SeqaxSiluFusionCorrectnessAttemptClaim,
    SeqaxSiluFusionCorrectnessContract,
    SeqaxSiluFusionCorrectnessFailure,
    SeqaxSiluFusionCorrectnessFailureArchiveSeal,
    SeqaxSiluFusionCorrectnessFailureReceipt,
    SeqaxSiluFusionCorrectnessFailureReplaySeal,
    SeqaxSiluFusionCorrectnessReceipt,
    SeqaxSiluFusionCorrectnessReplaySeal,
    SeqaxSiluFusionCorrectnessSourceAuthority,
    SeqaxSiluFusionCorrectnessWorkerRequest,
    SeqaxSiluFusionCorrectnessWorkerResult,
    default_seqax_silu_fusion_correctness_contract,
)

_EVIDENCE_ROOT = Path("/home/sudarshan/tpu-cake-evidence")
_CONTRACT_PATH = Path("contracts/seqax-silu-fusion-correctness-v1.json")
_DESIGN_PATH = Path("contracts/seqax-silu-fusion-design-v1.json")
_PAIR_PATH = Path("contracts/seqax-silu-fusion-compiler-pair-v1.json")
_SCHEMA_SOURCE = Path("src/tpu_cake/seqax_silu_fusion_correctness.py")
_RUNNER_SOURCE = Path("src/tpu_cake/seqax_silu_fusion_correctness_runner.py")
_WORKER_SOURCE = Path("src/tpu_cake/seqax_silu_fusion_correctness_worker.py")
_VERIFIER_SOURCE = Path("src/tpu_cake/seqax_silu_fusion_correctness_verifier.py")
_VERIFIER_MODULE = "tpu_cake.seqax_silu_fusion_correctness_verifier"


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


def _write_json_exclusive(path: Path, value: object) -> None:
    _write_bytes_exclusive(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_blob(repository_root: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", "show", f"{commit}:{path}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout


def _source_paths(repository_root: Path, commit: str) -> tuple[str, ...]:
    paths = _git(
        repository_root,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        "src/tpu_cake",
        _CONTRACT_PATH.as_posix(),
        _DESIGN_PATH.as_posix(),
        _PAIR_PATH.as_posix(),
        "pyproject.toml",
        "uv.lock",
    ).splitlines()
    required = {
        _CONTRACT_PATH.as_posix(),
        _DESIGN_PATH.as_posix(),
        _PAIR_PATH.as_posix(),
        _SCHEMA_SOURCE.as_posix(),
        _RUNNER_SOURCE.as_posix(),
        _WORKER_SOURCE.as_posix(),
        _VERIFIER_SOURCE.as_posix(),
        "src/tpu_cake/cli.py",
        "pyproject.toml",
        "uv.lock",
    }
    if not required.issubset(paths):
        raise ValueError(
            "SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_DEPENDENCY_MISSING "
            f"missing={sorted(required - set(paths))}"
        )
    return tuple(sorted(paths))


def _source_authority(
    repository_root: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
) -> tuple[SeqaxSiluFusionCorrectnessSourceAuthority, dict[str, bytes]]:
    repository_root = repository_root.resolve(strict=True)
    if repository_root != Path(contract.compilation_source_root):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_ROOT_MISMATCH")
    branch = _git(repository_root, "branch", "--show-current")
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != "main" or status:
        raise ValueError(
            "SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_CHECKOUT_INVALID "
            f"branch={branch!r} status={status.splitlines()}"
        )
    remote = subprocess.run(
        ["/usr/bin/git", "ls-remote", contract.source_remote_url, "refs/heads/main"],
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(remote) != 2 or remote[1] != "refs/heads/main":
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_REMOTE_MAIN_INVALID")
    commit = _git(repository_root, "rev-parse", "HEAD")
    paths = _source_paths(repository_root, commit)
    blobs = {path: _git_blob(repository_root, commit, path) for path in paths}
    manifest = tuple(
        SourceFileContract(path=path, sha256=hashlib.sha256(blobs[path]).hexdigest())
        for path in paths
    )
    source = SeqaxSiluFusionCorrectnessSourceAuthority(
        source_commit=commit,
        source_tree=_git(repository_root, "rev-parse", f"{commit}^{{tree}}"),
        branch="main",
        origin_main_commit=_git(repository_root, "rev-parse", "origin/main"),
        remote_main_commit=remote[0],
        remote_url=contract.source_remote_url,
        source_root=contract.compilation_source_root,
        uv_lock_sha256=hashlib.sha256(blobs["uv.lock"]).hexdigest(),
        cli_sha256=hashlib.sha256(blobs["src/tpu_cake/cli.py"]).hexdigest(),
        correctness_contract_sha256=hashlib.sha256(blobs[_CONTRACT_PATH.as_posix()]).hexdigest(),
        compiler_design_sha256=hashlib.sha256(blobs[_DESIGN_PATH.as_posix()]).hexdigest(),
        compiler_pair_sha256=hashlib.sha256(blobs[_PAIR_PATH.as_posix()]).hexdigest(),
        correctness_schema_source_sha256=hashlib.sha256(
            blobs[_SCHEMA_SOURCE.as_posix()]
        ).hexdigest(),
        runner_source_sha256=hashlib.sha256(blobs[_RUNNER_SOURCE.as_posix()]).hexdigest(),
        worker_source_sha256=hashlib.sha256(blobs[_WORKER_SOURCE.as_posix()]).hexdigest(),
        verifier_source_sha256=hashlib.sha256(blobs[_VERIFIER_SOURCE.as_posix()]).hexdigest(),
        source_manifest=manifest,
        runtime=contract.runtime,
    )
    return source, blobs


def _reject_links_in_path(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.absolute().parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"SEQAX_SILU_FUSION_CORRECTNESS_PATH_SYMLINK path={current}")


def _require_safe_new_root(
    root: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
) -> Path:
    if not root.is_absolute() or root != root.resolve(strict=False):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ROOT_NONCANONICAL")
    evidence_root = _EVIDENCE_ROOT.resolve(strict=True)
    if root.parent != evidence_root:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ROOT_OUTSIDE_EVIDENCE")
    pattern = re.compile(
        rf"seqax-silu-fusion-correctness-{contract.contract_id[:7]}-[0-9a-f]{{8}}$"
    )
    if pattern.fullmatch(root.name) is None:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ROOT_NAME_INVALID")
    _reject_links_in_path(root.parent)
    if root.exists() or root.is_symlink():
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ROOT_EXISTS")
    return root


@contextmanager
def _claim_lock(contract: SeqaxSiluFusionCorrectnessContract) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-silu-correctness-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    info = lock_root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_LOCK_ROOT_INVALID")
    descriptor = os.open(
        lock_root / f"{contract.correctness_claim_key}-{contract.contract_id}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CLAIM_LOCKED") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _prepare_claim(
    root: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
    source: SeqaxSiluFusionCorrectnessSourceAuthority,
) -> tuple[Path, SeqaxSiluFusionCorrectnessAttemptClaim]:
    registry = Path(contract.correctness_claim_registry_root)
    _reject_links_in_path(registry.parent)
    if not registry.exists():
        registry.mkdir(mode=0o700, exist_ok=False)
        _fsync_directory(registry.parent)
    info = registry.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CLAIM_REGISTRY_INVALID")
    claim = SeqaxSiluFusionCorrectnessAttemptClaim(
        contract_id=contract.contract_id,
        invocation_id=os.urandom(16).hex(),
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root=str(root),
    )
    claim_path = registry / (f"{contract.correctness_claim_key}-{contract.contract_id}.json")
    if claim_path.exists() or claim_path.is_symlink():
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PERMANENTLY_CLAIMED")
    return claim_path, claim


def _reserve_claim(
    claim_path: Path,
    claim: SeqaxSiluFusionCorrectnessAttemptClaim,
) -> None:
    try:
        _write_json_exclusive(
            claim_path,
            claim.model_dump(mode="json", exclude_computed_fields=True),
        )
    except FileExistsError as error:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PERMANENTLY_CLAIMED") from error


def _write_source_bundle(root: Path, blobs: dict[str, bytes]) -> None:
    for path, blob in blobs.items():
        _write_bytes_exclusive(root / "source" / "committed" / path, blob)


def _artifact_role(path: Path) -> ArtifactRole:
    value = path.as_posix()
    fixed = {
        "attempt_claim.json": ArtifactRole.INVOCATION,
        "compiler_pair.json": ArtifactRole.COMPILER_ANALYSIS,
        "contract.json": ArtifactRole.EXPERIMENT,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "source.json": ArtifactRole.SOURCE_STATE,
        "source_manifest.json": ArtifactRole.SOURCE_STATE,
        "worker_request.json": ArtifactRole.INVOCATION,
        "worker-result.json": ArtifactRole.CORRECTNESS_OUTPUT,
        "worker-failure.json": ArtifactRole.CORRECTNESS_OUTPUT,
    }
    if value in fixed:
        return fixed[value]
    if value.startswith("source/committed/"):
        return ArtifactRole.SOURCE_STATE
    plan_match = re.fullmatch(r"plans/(separate|silu_multiply)/([^/]+)", value)
    if plan_match is not None:
        roles = {
            "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
            "physical.xdsl": ArtifactRole.PHYSICAL_IR,
            "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
            "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
            "uninstrumented_stablehlo.txt": ArtifactRole.STABLEHLO,
            "uninstrumented_pre_optimization_hlo.txt": ArtifactRole.COMPILER_HLO,
            "uninstrumented_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
            "instrumented_stablehlo.txt": ArtifactRole.STABLEHLO,
            "instrumented_pre_optimization_hlo.txt": ArtifactRole.COMPILER_HLO,
            "instrumented_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
            "compiler_candidate.json": ArtifactRole.COMPILER_ANALYSIS,
            "record.json": ArtifactRole.COMPILER_ANALYSIS,
        }
        role = roles.get(plan_match.group(2))
        if role is not None:
            return role
    if re.fullmatch(r"seeds/seed-[0-9]+/inputs/[0-9]{2}\.npy", value):
        return ArtifactRole.CORRECTNESS_INPUT
    if re.fullmatch(r"seeds/seed-[0-9]+/cpu_reference\.npy", value):
        return ArtifactRole.ORACLE_OUTPUT
    correctness_patterns = (
        r"seeds/seed-[0-9]+/observation\.json",
        r"seeds/seed-[0-9]+/boundary/(strict|mutant)_hidden\.npy",
        r"seeds/seed-[0-9]+/(separate|silu_multiply)/(un)?instrumented_output\.npy",
        (
            r"seeds/seed-[0-9]+/(separate|silu_multiply)/"
            r"(un)?instrumented_assessment\.json"
        ),
        r"seeds/seed-[0-9]+/(separate|silu_multiply)/checkpoint_assessment\.json",
        (
            r"seeds/seed-[0-9]+/(separate|silu_multiply)/checkpoints/"
            r"(rms_input|rms_mean_square|rms_inverse|normalized_float32|"
            r"normalized_bfloat16|gate_float32|gate_bfloat16|silu_bfloat16|"
            r"up_float32|up_bfloat16|hidden_bfloat16|down_float32|down_bfloat16)\.npy"
        ),
    )
    if any(re.fullmatch(pattern, value) for pattern in correctness_patterns):
        return ArtifactRole.CORRECTNESS_OUTPUT
    raise ValueError(f"SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_ROLE_UNKNOWN path={value}")


def _worker_environment(
    root: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
) -> dict[str, str]:
    environment = dict(contract.worker_environment)
    environment.update(contract.compiler_environment)
    environment["PYTHONPATH"] = str(root / "source" / "committed" / "src")
    return environment


def _launch_worker(
    root: Path,
    request_path: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
) -> subprocess.CompletedProcess[str]:
    bundle = root / "source" / "committed"
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-m",
            "tpu_cake.seqax_silu_fusion_correctness_worker",
            "--root",
            str(root),
            "--request",
            str(request_path),
        ],
        cwd=bundle,
        env=_worker_environment(root, contract),
        check=False,
        capture_output=True,
        text=True,
    )


def _verify(
    root: Path,
    *,
    final: bool,
    relocated: bool = False,
) -> dict[str, str]:
    bundle = root / "source" / "committed"
    environment = {
        "HOME": "/nonexistent",
        "JAX_PLATFORMS": "cpu",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(bundle / "src"),
        "PYTHONSAFEPATH": "1",
    }
    arguments = [
        sys.executable,
        "-B",
        "-P",
        "-m",
        _VERIFIER_MODULE,
        "--root",
        str(root),
    ]
    if final:
        arguments.append("--final")
    if relocated:
        arguments.append("--relocated")
    completed = subprocess.run(
        arguments,
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SEQAX_SILU_FUSION_CORRECTNESS_VERIFIER_FAILED "
            f"returncode={completed.returncode} stdout={completed.stdout!r} "
            f"stderr={completed.stderr!r}"
        )
    return json.loads(completed.stdout)


def _verify_failure(root: Path, *, relocated: bool = False) -> dict[str, str]:
    bundle = root / "source" / "committed"
    environment = {
        "HOME": "/nonexistent",
        "JAX_PLATFORMS": "cpu",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(bundle / "src"),
        "PYTHONSAFEPATH": "1",
    }
    arguments = [
        sys.executable,
        "-B",
        "-P",
        "-m",
        _VERIFIER_MODULE,
        "--root",
        str(root),
        "--failed",
    ]
    if relocated:
        arguments.append("--relocated")
    completed = subprocess.run(
        arguments,
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_VERIFIER_FAILED "
            f"returncode={completed.returncode} stdout={completed.stdout!r} "
            f"stderr={completed.stderr!r}"
        )
    return json.loads(completed.stdout)


def _registry_path(
    contract: SeqaxSiluFusionCorrectnessContract,
    suffix: str,
) -> Path:
    return Path(contract.correctness_claim_registry_root) / (
        f"{contract.correctness_claim_key}-{contract.contract_id}.{suffix}.json"
    )


def _seal_replay(
    contract: SeqaxSiluFusionCorrectnessContract,
    claim: SeqaxSiluFusionCorrectnessAttemptClaim,
    receipt: SeqaxSiluFusionCorrectnessReceipt,
) -> SeqaxSiluFusionCorrectnessReplaySeal:
    result = receipt.result
    seal = SeqaxSiluFusionCorrectnessReplaySeal(
        contract_id=contract.contract_id,
        claim_id=claim.claim_id,
        result_id=result.result_id,
        receipt_id=receipt.receipt_id,
        source_commit=result.source.source_commit,
        source_tree=result.source.source_tree,
        output_root=claim.output_root,
        independent_replay_performed=True,
        retry_authorized=False,
        resume_authorized=False,
    )
    _write_json_exclusive(
        _registry_path(contract, "replay"),
        seal.model_dump(mode="json", exclude_computed_fields=True),
    )
    return seal


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _create_archive(
    root: Path,
    *,
    label: str | None = None,
) -> tuple[Path, str, int]:
    zstd = _require_zstd()
    suffix = f".{label}" if label is not None else ""
    archive = root.parent / f"{root.name}{suffix}.tar.zst"
    if archive.exists() or archive.is_symlink():
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ARCHIVE_EXISTS")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{root.name}-",
        suffix=".tar",
        dir=root.parent,
    )
    os.close(descriptor)
    temporary_tar = Path(temporary_name)
    try:
        members = (root, *sorted(root.rglob("*"), key=lambda value: value.as_posix()))
        with tarfile.open(temporary_tar, "w", format=tarfile.GNU_FORMAT) as archive_file:
            for path in members:
                relative = path.relative_to(root.parent)
                info = _normalized_tar_info(
                    archive_file.gettarinfo(str(path), arcname=relative.as_posix())
                )
                if info.isfile():
                    with path.open("rb") as stream:
                        archive_file.addfile(info, stream)
                else:
                    archive_file.addfile(info)
        subprocess.run(
            [zstd, "-19", "--threads=1", "--no-progress", str(temporary_tar), "-o", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        temporary_tar.unlink(missing_ok=True)
    return archive, file_sha256(archive), len(members)


def _require_zstd() -> str:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ZSTD_MISSING")
    return zstd


def _require_compiler_evidence_ready(
    contract: SeqaxSiluFusionCorrectnessContract,
) -> None:
    if contract.compiler_evidence_status != "verified":
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_COMPILER_EVIDENCE_PENDING")


def _verify_extracted_archive(
    archive: Path,
    expected_root_name: str,
) -> dict[str, str]:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ZSTD_MISSING")
    with tempfile.TemporaryDirectory(prefix="tpu-cake-silu-correctness-replay-") as temporary:
        temporary_root = Path(temporary)
        expanded = temporary_root / "archive.tar"
        with expanded.open("xb") as stream:
            subprocess.run(
                [zstd, "--decompress", "--stdout", str(archive)],
                check=True,
                stdout=stream,
                stderr=subprocess.PIPE,
            )
        extraction = temporary_root / "extracted"
        extraction.mkdir(mode=0o700)
        with tarfile.open(expanded, "r:") as archive_file:
            members = archive_file.getmembers()
            roots = {Path(member.name).parts[0] for member in members if member.name}
            if roots != {expected_root_name} or any(
                member.issym() or member.islnk() for member in members
            ):
                raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ARCHIVE_LAYOUT_INVALID")
            archive_file.extractall(extraction, filter="data")
        return _verify(
            extraction / expected_root_name,
            final=True,
            relocated=True,
        )


def _verify_extracted_failure_archive(
    archive: Path,
    expected_root_name: str,
) -> dict[str, str]:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ZSTD_MISSING")
    with tempfile.TemporaryDirectory(
        prefix="tpu-cake-silu-correctness-failure-replay-"
    ) as temporary:
        temporary_root = Path(temporary)
        expanded = temporary_root / "archive.tar"
        with expanded.open("xb") as stream:
            subprocess.run(
                [zstd, "--decompress", "--stdout", str(archive)],
                check=True,
                stdout=stream,
                stderr=subprocess.PIPE,
            )
        extraction = temporary_root / "extracted"
        extraction.mkdir(mode=0o700)
        with tarfile.open(expanded, "r:") as archive_file:
            members = archive_file.getmembers()
            roots = {Path(member.name).parts[0] for member in members if member.name}
            if roots != {expected_root_name} or any(
                member.issym() or member.islnk() for member in members
            ):
                raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARCHIVE_LAYOUT_INVALID")
            archive_file.extractall(extraction, filter="data")
        return _verify_failure(
            extraction / expected_root_name,
            relocated=True,
        )


def _seal_archive(
    contract: SeqaxSiluFusionCorrectnessContract,
    claim: SeqaxSiluFusionCorrectnessAttemptClaim,
    receipt: SeqaxSiluFusionCorrectnessReceipt,
    replay: SeqaxSiluFusionCorrectnessReplaySeal,
    archive: Path,
    archive_sha256: str,
    member_count: int,
) -> SeqaxSiluFusionCorrectnessArchiveSeal:
    seal = SeqaxSiluFusionCorrectnessArchiveSeal(
        contract_id=contract.contract_id,
        claim_id=claim.claim_id,
        result_id=receipt.result.result_id,
        receipt_id=receipt.receipt_id,
        replay_seal_id=replay.replay_seal_id,
        archive_path=str(archive),
        archive_sha256=archive_sha256,
        archive_member_count=member_count,
        single_root=True,
        extracted_replay_performed=True,
    )
    _write_json_exclusive(
        _registry_path(contract, "archive"),
        seal.model_dump(mode="json", exclude_computed_fields=True),
    )
    return seal


def _record_failure(
    root: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
    claim: SeqaxSiluFusionCorrectnessAttemptClaim,
    source: SeqaxSiluFusionCorrectnessSourceAuthority,
    *,
    phase: str,
    completed: subprocess.CompletedProcess[str] | None,
    error: BaseException | None,
) -> tuple[
    SeqaxSiluFusionCorrectnessFailureReceipt,
    SeqaxSiluFusionCorrectnessFailureReplaySeal,
    SeqaxSiluFusionCorrectnessFailureArchiveSeal,
]:
    state = EvidenceRun(root / "ledger.sqlite", claim.claim_id).current_state()
    if state is None:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_LEDGER_EMPTY")
    execution_status = (
        "executed"
        if state in {RunState.CORRECT, RunState.VALIDATED, RunState.ACCEPTED}
        else "may-have-executed"
        if state is RunState.COMPILED
        else "not-executed"
    )
    failure = SeqaxSiluFusionCorrectnessFailure(
        phase=phase,
        returncode=None if completed is None else completed.returncode,
        stdout="" if completed is None else completed.stdout,
        stderr="" if completed is None else completed.stderr,
        error_type=None if error is None else type(error).__name__,
        error_message=None if error is None else str(error) or repr(error),
        final_ledger_state=state,
        model_outputs_execution_status=execution_status,
        timing_collected=False,
        profile_collected=False,
        retry_authorized=False,
        resume_authorized=False,
    )
    _write_json_exclusive(
        root / "worker-failure.json",
        failure.model_dump(mode="json", exclude_computed_fields=True),
    )
    finalize_ledger(root / "ledger.sqlite")
    incomplete_receipt_path = root / "receipt.json"
    incomplete_receipt = (
        SeqaxSiluFusionCorrectnessReceipt.model_validate_json(incomplete_receipt_path.read_text())
        if incomplete_receipt_path.exists()
        else None
    )
    artifacts = build_artifact_manifest(
        root,
        role_for_path=_artifact_role,
        excluded_paths=("failure-receipt.json", "receipt.json"),
    )
    receipt = SeqaxSiluFusionCorrectnessFailureReceipt(
        claim=claim,
        source=source,
        failure=failure,
        incomplete_success_receipt=incomplete_receipt,
        artifacts=artifacts,
        independent_replay_required=True,
        independent_replay_performed_at_receipt_creation=False,
        archive_required=True,
        retry_authorized=False,
        resume_authorized=False,
    )
    _write_json_exclusive(
        root / "failure-receipt.json",
        receipt.model_dump(mode="json", exclude_computed_fields=True),
    )
    replayed = _verify_failure(root)
    if replayed != {"failure_receipt_id": receipt.failure_receipt_id}:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_REPLAY_MISMATCH")
    replay = SeqaxSiluFusionCorrectnessFailureReplaySeal(
        contract_id=contract.contract_id,
        claim_id=claim.claim_id,
        failure_receipt_id=receipt.failure_receipt_id,
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root=str(root),
        independent_replay_performed=True,
        retry_authorized=False,
        resume_authorized=False,
    )
    _write_json_exclusive(
        _registry_path(contract, "failure-replay"),
        replay.model_dump(mode="json", exclude_computed_fields=True),
    )
    archive, archive_sha256, member_count = _create_archive(root, label="failure")
    extracted = _verify_extracted_failure_archive(archive, root.name)
    if extracted != {"failure_receipt_id": receipt.failure_receipt_id}:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARCHIVE_REPLAY_MISMATCH")
    archive_seal = SeqaxSiluFusionCorrectnessFailureArchiveSeal(
        contract_id=contract.contract_id,
        claim_id=claim.claim_id,
        failure_receipt_id=receipt.failure_receipt_id,
        failure_replay_seal_id=replay.failure_replay_seal_id,
        archive_path=str(archive),
        archive_sha256=archive_sha256,
        archive_member_count=member_count,
        single_root=True,
        extracted_replay_performed=True,
    )
    _write_json_exclusive(
        _registry_path(contract, "failure-archive"),
        archive_seal.model_dump(mode="json", exclude_computed_fields=True),
    )
    return receipt, replay, archive_seal


def run_correctness(
    root: Path,
    contract_path: Path,
) -> tuple[
    SeqaxSiluFusionCorrectnessReceipt,
    SeqaxSiluFusionCorrectnessReplaySeal,
    SeqaxSiluFusionCorrectnessArchiveSeal,
]:
    repository_root = Path(__file__).resolve().parents[2]
    canonical_contract_path = repository_root / _CONTRACT_PATH
    if contract_path.resolve(strict=True) != canonical_contract_path:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CONTRACT_PATH_MISMATCH")
    contract = SeqaxSiluFusionCorrectnessContract.model_validate_json(contract_path.read_text())
    if contract != default_seqax_silu_fusion_correctness_contract(contract.runtime):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CONTRACT_NONCANONICAL")
    _require_compiler_evidence_ready(contract)
    root = _require_safe_new_root(root, contract)
    source, blobs = _source_authority(repository_root, contract)
    _require_zstd()
    with _claim_lock(contract):
        claim_path, claim = _prepare_claim(root, contract, source)
        root.mkdir(mode=0o700, exist_ok=False)
        _fsync_directory(root.parent)
        _write_json_exclusive(
            root / "attempt_claim.json",
            claim.model_dump(mode="json", exclude_computed_fields=True),
        )
        _write_bytes_exclusive(root / "contract.json", blobs[_CONTRACT_PATH.as_posix()])
        _write_bytes_exclusive(root / "compiler_pair.json", blobs[_PAIR_PATH.as_posix()])
        _write_json_exclusive(
            root / "source.json",
            source.model_dump(mode="json", exclude_computed_fields=True),
        )
        _write_json_exclusive(
            root / "source_manifest.json",
            [value.model_dump(mode="json") for value in source.source_manifest],
        )
        _write_source_bundle(root, blobs)
        request = SeqaxSiluFusionCorrectnessWorkerRequest(
            contract=contract,
            claim=claim,
            source=source,
        )
        request_path = root / "worker_request.json"
        _write_json_exclusive(
            request_path,
            request.model_dump(mode="json", exclude_computed_fields=True),
        )
        EvidenceRun(root / "ledger.sqlite", claim.claim_id).create(
            {
                "claim_id": claim.claim_id,
                "claim_path": str(claim_path),
                "contract_id": contract.contract_id,
            }
        )
        _reserve_claim(claim_path, claim)
    try:
        completed = _launch_worker(root, request_path, contract)
    except BaseException as error:
        failure_receipt, failure_replay, failure_archive = _record_failure(
            root,
            contract,
            claim,
            source,
            phase="worker-launch",
            completed=None,
            error=error,
        )
        message = (
            "SEQAX_SILU_FUSION_CORRECTNESS_ORCHESTRATOR_FAILED "
            f"failure_receipt_id={failure_receipt.failure_receipt_id} "
            f"failure_replay_seal_id={failure_replay.failure_replay_seal_id} "
            f"failure_archive_seal_id={failure_archive.failure_archive_seal_id} "
            f"root={root} phase=worker-launch error={error}"
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        raise RuntimeError(message) from error
    if completed.returncode != 0:
        failure_receipt, failure_replay, failure_archive = _record_failure(
            root,
            contract,
            claim,
            source,
            phase="worker-execution",
            completed=completed,
            error=None,
        )
        raise RuntimeError(
            "SEQAX_SILU_FUSION_CORRECTNESS_WORKER_FAILED "
            f"failure_receipt_id={failure_receipt.failure_receipt_id} "
            f"failure_replay_seal_id={failure_replay.failure_replay_seal_id} "
            f"failure_archive_seal_id={failure_archive.failure_archive_seal_id} "
            f"root={root} stderr={completed.stderr.strip()}"
        )
    phase = "worker-result"
    try:
        worker_result = SeqaxSiluFusionCorrectnessWorkerResult.model_validate_json(
            (root / "worker-result.json").read_text()
        )
        if (
            worker_result.result.contract_id != contract.contract_id
            or worker_result.result.claim_id != claim.claim_id
            or worker_result.result.source != source
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_WORKER_RESULT_MISMATCH")
        finalize_ledger(root / "ledger.sqlite")
        phase = "preliminary-replay"
        preliminary = _verify(root, final=False)
        if preliminary != {"result_id": worker_result.result.result_id}:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PRELIMINARY_REPLAY_MISMATCH")
        phase = "acceptance"
        run = EvidenceRun(root / "ledger.sqlite", claim.claim_id)
        run.transition(
            RunState.VALIDATED,
            {"independent_result_id": worker_result.result.result_id},
        )
        run.transition(
            RunState.ACCEPTED,
            {"contract_id": contract.contract_id, "timing_authorized": False},
        )
        finalize_ledger(root / "ledger.sqlite")
        artifacts = build_artifact_manifest(root, role_for_path=_artifact_role)
        receipt = SeqaxSiluFusionCorrectnessReceipt(
            result=worker_result.result,
            final_ledger_state=RunState.ACCEPTED,
            artifacts=artifacts,
            independent_replay_performed=True,
            archive_required=True,
            retry_authorized=False,
            resume_authorized=False,
            timing_authorized=False,
            profile_authorized=False,
        )
        _write_json_exclusive(
            root / "receipt.json",
            receipt.model_dump(mode="json", exclude_computed_fields=True),
        )
        phase = "final-replay"
        final_replay = _verify(root, final=True)
        expected_final = {
            "result_id": receipt.result.result_id,
            "receipt_id": receipt.receipt_id,
        }
        if final_replay != expected_final:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FINAL_REPLAY_MISMATCH")
        replay = _seal_replay(contract, claim, receipt)
        phase = "archive"
        archive, archive_sha256, member_count = _create_archive(root)
        extracted = _verify_extracted_archive(archive, root.name)
        if extracted != expected_final:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ARCHIVE_REPLAY_MISMATCH")
        archive_seal = _seal_archive(
            contract,
            claim,
            receipt,
            replay,
            archive,
            archive_sha256,
            member_count,
        )
        return receipt, replay, archive_seal
    except BaseException as error:
        failure_receipt, failure_replay, failure_archive = _record_failure(
            root,
            contract,
            claim,
            source,
            phase=phase,
            completed=completed,
            error=error,
        )
        message = (
            "SEQAX_SILU_FUSION_CORRECTNESS_ORCHESTRATOR_FAILED "
            f"failure_receipt_id={failure_receipt.failure_receipt_id} "
            f"failure_replay_seal_id={failure_replay.failure_replay_seal_id} "
            f"failure_archive_seal_id={failure_archive.failure_archive_seal_id} "
            f"root={root} phase={phase} error={error}"
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        raise RuntimeError(message) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    arguments = parser.parse_args()
    receipt, replay, archive = run_correctness(arguments.root, arguments.contract)
    print(
        json.dumps(
            {
                "archive_seal_id": archive.archive_seal_id,
                "archive_sha256": archive.archive_sha256,
                "receipt_id": receipt.receipt_id,
                "replay_seal_id": replay.replay_seal_id,
                "result_id": receipt.result.result_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
