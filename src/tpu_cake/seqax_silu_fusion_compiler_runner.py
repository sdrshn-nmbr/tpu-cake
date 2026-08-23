from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tpu_cake.artifacts import build_artifact_manifest
from tpu_cake.contracts import ArtifactRole, SourceFileContract
from tpu_cake.ledger import EvidenceRun, RunState
from tpu_cake.seqax_silu_fusion import (
    SeqaxSiluFusionDesignContract,
    default_seqax_silu_fusion_design_contract,
)
from tpu_cake.seqax_silu_fusion_compiler import (
    SeqaxSiluFusionCompilerAttemptClaim,
    SeqaxSiluFusionCompilerFailure,
    SeqaxSiluFusionCompilerFailureReceipt,
    SeqaxSiluFusionCompilerFailureReplaySeal,
    SeqaxSiluFusionCompilerReceipt,
    SeqaxSiluFusionCompilerReplaySeal,
    SeqaxSiluFusionCompilerSourceAuthority,
    SeqaxSiluFusionCompilerWorkerRequest,
    SeqaxSiluFusionCompilerWorkerResult,
)

_EVIDENCE_ROOT = Path("/home/sudarshan/tpu-cake-evidence")
_DESIGN_RELATIVE_PATH = Path("contracts/seqax-silu-fusion-design-v1.json")
_RUNNER_RELATIVE_PATH = Path("src/tpu_cake/seqax_silu_fusion_compiler_runner.py")
_WORKER_RELATIVE_PATH = Path("src/tpu_cake/seqax_silu_fusion_compiler_worker.py")
_COMPILER_RELATIVE_PATH = Path("src/tpu_cake/seqax_silu_fusion_compiler.py")
_PAIR_RELATIVE_PATH = Path("src/tpu_cake/seqax_silu_fusion_compiler_pair.py")
_VERIFIER_MODULE = "tpu_cake.seqax_silu_fusion_compiler_verifier"


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
        _DESIGN_RELATIVE_PATH.as_posix(),
        "pyproject.toml",
        "uv.lock",
    ).splitlines()
    required = {
        _DESIGN_RELATIVE_PATH.as_posix(),
        _RUNNER_RELATIVE_PATH.as_posix(),
        _WORKER_RELATIVE_PATH.as_posix(),
        _COMPILER_RELATIVE_PATH.as_posix(),
        _PAIR_RELATIVE_PATH.as_posix(),
        "src/tpu_cake/seqax_silu_fusion_compiler_verifier.py",
        "src/tpu_cake/cli.py",
        "pyproject.toml",
        "uv.lock",
    }
    if not required.issubset(paths):
        raise ValueError(
            f"SEQAX_SILU_FUSION_SOURCE_DEPENDENCY_MISSING missing={sorted(required - set(paths))}"
        )
    return tuple(sorted(paths))


def _source_authority(
    repository_root: Path,
    design: SeqaxSiluFusionDesignContract,
) -> tuple[SeqaxSiluFusionCompilerSourceAuthority, dict[str, bytes]]:
    repository_root = repository_root.resolve(strict=True)
    if repository_root != Path(design.compilation_source_root):
        raise ValueError("SEQAX_SILU_FUSION_SOURCE_ROOT_MISMATCH")
    branch = _git(repository_root, "branch", "--show-current")
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != "main" or status:
        raise ValueError(
            f"SEQAX_SILU_FUSION_SOURCE_CHECKOUT_INVALID branch={branch!r} "
            f"status={status.splitlines()}"
        )
    remote = subprocess.run(
        ["/usr/bin/git", "ls-remote", design.source_remote_url, "refs/heads/main"],
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(remote) != 2 or remote[1] != "refs/heads/main":
        raise ValueError("SEQAX_SILU_FUSION_REMOTE_MAIN_INVALID")
    source_commit = _git(repository_root, "rev-parse", "HEAD")
    paths = _source_paths(repository_root, source_commit)
    blobs = {path: _git_blob(repository_root, source_commit, path) for path in paths}
    manifest = tuple(
        SourceFileContract(path=path, sha256=hashlib.sha256(blobs[path]).hexdigest())
        for path in paths
    )
    authority = SeqaxSiluFusionCompilerSourceAuthority(
        source_commit=source_commit,
        source_tree=_git(repository_root, "rev-parse", f"{source_commit}^{{tree}}"),
        branch="main",
        origin_main_commit=_git(repository_root, "rev-parse", "origin/main"),
        remote_main_commit=remote[0],
        remote_url=design.source_remote_url,
        source_root=design.compilation_source_root,
        uv_lock_sha256=hashlib.sha256(blobs["uv.lock"]).hexdigest(),
        cli_sha256=hashlib.sha256(blobs["src/tpu_cake/cli.py"]).hexdigest(),
        design_file_sha256=hashlib.sha256(blobs[_DESIGN_RELATIVE_PATH.as_posix()]).hexdigest(),
        runner_source_sha256=hashlib.sha256(blobs[_RUNNER_RELATIVE_PATH.as_posix()]).hexdigest(),
        worker_source_sha256=hashlib.sha256(blobs[_WORKER_RELATIVE_PATH.as_posix()]).hexdigest(),
        compiler_source_sha256=hashlib.sha256(
            blobs[_COMPILER_RELATIVE_PATH.as_posix()]
        ).hexdigest(),
        pair_source_sha256=hashlib.sha256(blobs[_PAIR_RELATIVE_PATH.as_posix()]).hexdigest(),
        source_manifest=manifest,
        runtime=design.runtime,
    )
    return authority, blobs


def _reject_links_in_path(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.absolute().parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"SEQAX_SILU_FUSION_PATH_SYMLINK path={current}")


def _require_safe_new_root(
    root: Path,
    design: SeqaxSiluFusionDesignContract,
    ordinal: int,
) -> Path:
    if not root.is_absolute():
        raise ValueError("SEQAX_SILU_FUSION_ROOT_NOT_ABSOLUTE")
    resolved = root.resolve(strict=False)
    if root != resolved:
        raise ValueError("SEQAX_SILU_FUSION_ROOT_NOT_CANONICAL")
    evidence_root = _EVIDENCE_ROOT.resolve(strict=True)
    if resolved.parent != evidence_root:
        raise ValueError("SEQAX_SILU_FUSION_ROOT_NOT_DIRECT_EVIDENCE_CHILD")
    pattern = re.compile(
        rf"seqax-silu-fusion-compiler-{design.design_id[:7]}-{ordinal}-[0-9a-f]{{8}}$"
    )
    if pattern.fullmatch(resolved.name) is None:
        raise ValueError("SEQAX_SILU_FUSION_ROOT_NAME_INVALID")
    _reject_links_in_path(resolved.parent)
    if resolved.exists() or resolved.is_symlink():
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_ROOT_EXISTS")
    return resolved


def _require_existing_capture_root(
    root: Path,
    design: SeqaxSiluFusionDesignContract,
    ordinal: int,
) -> Path:
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise ValueError("SEQAX_SILU_FUSION_PRIOR_ROOT_NONCANONICAL")
    if root.parent != _EVIDENCE_ROOT.resolve(strict=True):
        raise ValueError("SEQAX_SILU_FUSION_PRIOR_ROOT_OUTSIDE_EVIDENCE")
    pattern = re.compile(
        rf"seqax-silu-fusion-compiler-{design.design_id[:7]}-{ordinal}-[0-9a-f]{{8}}$"
    )
    if pattern.fullmatch(root.name) is None or root.is_symlink() or not root.is_dir():
        raise ValueError("SEQAX_SILU_FUSION_PRIOR_ROOT_INVALID")
    return root


@contextmanager
def _claim_lock(design: SeqaxSiluFusionDesignContract, ordinal: int) -> Iterator[None]:
    lock_root = Path(tempfile.gettempdir()) / f"tpu-cake-silu-fusion-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    status = lock_root.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid() or status.st_mode & 0o077:
        raise ValueError("SEQAX_SILU_FUSION_LOCK_ROOT_INVALID")
    descriptor = os.open(
        lock_root / f"{design.compiler_claim_key}-{design.design_id}-{ordinal}.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("SEQAX_SILU_FUSION_CAPTURE_LOCKED") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _claim_registry(design: SeqaxSiluFusionDesignContract) -> Path:
    registry = Path(design.compiler_claim_registry_root)
    _reject_links_in_path(registry.parent)
    if not registry.exists():
        registry.mkdir(mode=0o700, exist_ok=False)
        _fsync_directory(registry.parent)
    status = registry.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid() or status.st_mode & 0o077:
        raise ValueError("SEQAX_SILU_FUSION_CLAIM_REGISTRY_INVALID")
    return registry


def _claim_path(design: SeqaxSiluFusionDesignContract, ordinal: int) -> Path:
    return _claim_registry(design) / (
        f"{design.compiler_claim_key}-{design.design_id}-{ordinal}.json"
    )


def _require_claim_slot_unused(
    design: SeqaxSiluFusionDesignContract,
    ordinal: int,
) -> Path:
    claim_path = _claim_path(design, ordinal)
    if claim_path.exists():
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_PERMANENTLY_CLAIMED")
    if _replay_seal_path(design, ordinal).exists():
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_REPLAY_SEAL_EXISTS")
    if _failure_replay_seal_path(design, ordinal).exists():
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_FAILURE_REPLAY_SEAL_EXISTS")
    return claim_path


def _prepare_claim(
    root: Path,
    design: SeqaxSiluFusionDesignContract,
    ordinal: int,
    source: SeqaxSiluFusionCompilerSourceAuthority,
) -> tuple[Path, SeqaxSiluFusionCompilerAttemptClaim]:
    claim_path = _require_claim_slot_unused(design, ordinal)
    invocation_id = os.urandom(16).hex()
    claim = SeqaxSiluFusionCompilerAttemptClaim(
        design_id=design.design_id,
        capture_ordinal=ordinal,
        invocation_id=invocation_id,
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root=str(root),
    )
    return claim_path, claim


def _reserve_claim(
    claim_path: Path,
    claim: SeqaxSiluFusionCompilerAttemptClaim,
) -> None:
    try:
        _write_json_exclusive(
            claim_path,
            claim.model_dump(mode="json", exclude_computed_fields=True),
        )
    except FileExistsError as error:
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_PERMANENTLY_CLAIMED") from error


def _replay_seal_path(design: SeqaxSiluFusionDesignContract, ordinal: int) -> Path:
    return Path(design.compiler_claim_registry_root) / (
        f"{design.compiler_claim_key}-{design.design_id}-{ordinal}.replay.json"
    )


def _failure_replay_seal_path(
    design: SeqaxSiluFusionDesignContract,
    ordinal: int,
) -> Path:
    return Path(design.compiler_claim_registry_root) / (
        f"{design.compiler_claim_key}-{design.design_id}-{ordinal}.failure-replay.json"
    )


def _private_registry_file(registry: Path, name: str) -> Path:
    if registry.is_symlink() or not registry.is_dir():
        raise ValueError("SEQAX_SILU_FUSION_CLAIM_REGISTRY_INVALID")
    registry_info = registry.lstat()
    if (
        not stat.S_ISDIR(registry_info.st_mode)
        or registry_info.st_uid != os.getuid()
        or registry_info.st_mode & 0o077
    ):
        raise ValueError("SEQAX_SILU_FUSION_CLAIM_REGISTRY_INVALID")
    path = registry / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("SEQAX_SILU_FUSION_REGISTRY_FILE_INVALID")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_mode & 0o077
    ):
        raise ValueError("SEQAX_SILU_FUSION_REGISTRY_FILE_INVALID")
    return path


def _require_prior_replay_seal(
    design: SeqaxSiluFusionDesignContract,
    source: SeqaxSiluFusionCompilerSourceAuthority,
) -> None:
    registry = Path(design.compiler_claim_registry_root)
    claim_name = f"{design.compiler_claim_key}-{design.design_id}-0.json"
    seal_name = f"{design.compiler_claim_key}-{design.design_id}-0.replay.json"
    if not (registry / claim_name).exists() or not (registry / seal_name).exists():
        raise ValueError("SEQAX_SILU_FUSION_PRIOR_REPLAY_SEAL_MISSING")
    claim_path = _private_registry_file(registry, claim_name)
    seal_path = _private_registry_file(registry, seal_name)
    claim = SeqaxSiluFusionCompilerAttemptClaim.model_validate_json(claim_path.read_text())
    seal = SeqaxSiluFusionCompilerReplaySeal.model_validate_json(seal_path.read_text())
    prior_root = _require_existing_capture_root(Path(claim.output_root), design, 0)
    receipt_path = prior_root / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("SEQAX_SILU_FUSION_PRIOR_RECEIPT_MISSING")
    receipt = SeqaxSiluFusionCompilerReceipt.model_validate_json(receipt_path.read_text())
    capture = receipt.capture
    if (
        claim.design_id != design.design_id
        or claim.capture_ordinal != 0
        or claim.source_commit != source.source_commit
        or claim.source_tree != source.source_tree
        or seal.design_id != design.design_id
        or seal.capture_ordinal != 0
        or seal.claim_id != claim.claim_id
        or seal.capture_id != capture.capture_id
        or seal.receipt_id != receipt.receipt_id
        or seal.semantic_pair_id != capture.semantic_pair_id
        or seal.source_commit != source.source_commit
        or seal.source_tree != source.source_tree
        or seal.output_root != claim.output_root
    ):
        raise ValueError("SEQAX_SILU_FUSION_PRIOR_REPLAY_SEAL_MISMATCH")
    replay = _independent_verify(prior_root)
    expected_replay = {
        "capture_id": capture.capture_id,
        "receipt_id": receipt.receipt_id,
        "semantic_pair_id": capture.semantic_pair_id,
    }
    if replay != expected_replay:
        raise ValueError(
            "SEQAX_SILU_FUSION_PRIOR_INDEPENDENT_REPLAY_MISMATCH "
            f"expected={expected_replay} observed={replay}"
        )


def _seal_replay(
    design: SeqaxSiluFusionDesignContract,
    claim: SeqaxSiluFusionCompilerAttemptClaim,
    receipt: SeqaxSiluFusionCompilerReceipt,
) -> SeqaxSiluFusionCompilerReplaySeal:
    capture = receipt.capture
    seal = SeqaxSiluFusionCompilerReplaySeal(
        design_id=design.design_id,
        capture_ordinal=claim.capture_ordinal,
        claim_id=claim.claim_id,
        capture_id=capture.capture_id,
        receipt_id=receipt.receipt_id,
        semantic_pair_id=capture.semantic_pair_id,
        source_commit=capture.source.source_commit,
        source_tree=capture.source.source_tree,
        output_root=claim.output_root,
        independent_replay_performed=True,
    )
    _write_json_exclusive(
        _replay_seal_path(design, claim.capture_ordinal),
        seal.model_dump(mode="json", exclude_computed_fields=True),
    )
    return seal


def _write_source_bundle(root: Path, blobs: dict[str, bytes]) -> None:
    for path, blob in blobs.items():
        _write_bytes_exclusive(root / "source" / "committed" / path, blob)


def _subprocess_environment(design: SeqaxSiluFusionDesignContract) -> dict[str, str]:
    environment = dict(design.worker_environment)
    environment.update(design.compiler_environment)
    return environment


def _artifact_role(path: Path) -> ArtifactRole:
    value = path.as_posix()
    fixed = {
        "attempt_claim.json": ArtifactRole.INVOCATION,
        "contract.json": ArtifactRole.EXPERIMENT,
        "source.json": ArtifactRole.SOURCE_STATE,
        "source_manifest.json": ArtifactRole.SOURCE_STATE,
        "worker_request.json": ArtifactRole.INVOCATION,
        "controller-failure.json": ArtifactRole.COMPILER_ANALYSIS,
        "worker-failure.json": ArtifactRole.COMPILER_ANALYSIS,
        "worker-result.json": ArtifactRole.COMPILER_ANALYSIS,
        "receipt.json": ArtifactRole.COMPILER_ANALYSIS,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
    }
    if value in fixed:
        return fixed[value]
    if value.startswith("source/committed/"):
        return ArtifactRole.SOURCE_STATE
    candidate_roles = {
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "physical.xdsl": ArtifactRole.PHYSICAL_IR,
        "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
        "physical_resources.json": ArtifactRole.COST_MODEL,
        "stablehlo.txt": ArtifactRole.STABLEHLO,
        "pre_optimization_hlo.txt": ArtifactRole.COMPILER_HLO,
        "compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "compiler_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
        "reachable_collectives.json": ArtifactRole.COMPILER_ANALYSIS,
        "buffer_assignment.pb": ArtifactRole.COMPILER_ANALYSIS,
        "fusion_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
    }
    match = re.fullmatch(r"candidates/(separate|silu_multiply)/([^/]+)", value)
    if match is not None and match.group(2) in candidate_roles:
        return candidate_roles[match.group(2)]
    raise ValueError(f"SEQAX_SILU_FUSION_ARTIFACT_ROLE_UNKNOWN path={value}")


def _launch_worker(
    root: Path,
    request_path: Path,
    design: SeqaxSiluFusionDesignContract,
) -> subprocess.CompletedProcess[str]:
    bundle = root / "source" / "committed"
    environment = _subprocess_environment(design)
    environment["PYTHONPATH"] = str(bundle / "src")
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-m",
            "tpu_cake.seqax_silu_fusion_compiler_worker",
            "--root",
            str(root),
            "--request",
            str(request_path),
        ],
        cwd=bundle,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _independent_verify(root: Path) -> dict[str, object]:
    bundled = root / "source" / "committed"
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(bundled / "src"),
        "PYTHONSAFEPATH": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-m",
            _VERIFIER_MODULE,
            "--root",
            str(root),
            "--design",
            str(bundled / _DESIGN_RELATIVE_PATH),
        ],
        cwd="/",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _independent_verify_failure(
    root: Path,
    *,
    allow_missing_seal: bool,
) -> dict[str, object]:
    bundled = root / "source" / "committed"
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(bundled / "src"),
        "PYTHONSAFEPATH": "1",
    }
    arguments = [
        sys.executable,
        "-B",
        "-P",
        "-m",
        _VERIFIER_MODULE,
        "--failed-root",
        str(root),
        "--design",
        str(bundled / _DESIGN_RELATIVE_PATH),
    ]
    if allow_missing_seal:
        arguments.append("--allow-missing-failure-seal")
    completed = subprocess.run(
        arguments,
        cwd="/",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _record_failure(
    root: Path,
    design: SeqaxSiluFusionDesignContract,
    claim: SeqaxSiluFusionCompilerAttemptClaim,
    source: SeqaxSiluFusionCompilerSourceAuthority,
    *,
    stage: str,
    error: BaseException,
    completed: subprocess.CompletedProcess[str] | None,
) -> tuple[SeqaxSiluFusionCompilerFailureReceipt, SeqaxSiluFusionCompilerFailureReplaySeal]:
    failure = SeqaxSiluFusionCompilerFailure(
        stage=stage,
        error_type=type(error).__name__,
        error_message=str(error) or repr(error),
        returncode=None if completed is None else completed.returncode,
        stdout="" if completed is None else completed.stdout,
        stderr="" if completed is None else completed.stderr,
        worker_process_started=completed is not None,
        model_outputs_executed=False,
        correctness_outputs_collected=False,
        timing_collected=False,
        profile_collected=False,
    )
    failure_payload = failure.model_dump(mode="json", exclude_computed_fields=True)
    _write_json_exclusive(root / "controller-failure.json", failure_payload)
    _write_json_exclusive(
        root / "worker-failure.json",
        failure_payload,
    )
    state = EvidenceRun(root / "ledger.sqlite", claim.claim_id).current_state()
    if state not in {RunState.CREATED, RunState.VERIFIED, RunState.LOWERED, RunState.COMPILED}:
        raise ValueError(f"SEQAX_SILU_FUSION_FAILURE_LEDGER_STATE_INVALID state={state}")
    incomplete_receipt = None
    if (root / "receipt.json").exists():
        incomplete_receipt = SeqaxSiluFusionCompilerReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
    artifacts = build_artifact_manifest(
        root,
        role_for_path=_artifact_role,
        excluded_paths=(),
    )
    receipt = SeqaxSiluFusionCompilerFailureReceipt(
        claim=claim,
        source=source,
        final_ledger_state=state,
        failure=failure,
        incomplete_success_receipt_id=(
            None if incomplete_receipt is None else incomplete_receipt.receipt_id
        ),
        artifacts=artifacts,
        independent_replay_required=True,
        independent_replay_performed_at_receipt_creation=False,
        retry_authorized=False,
        ordinal_one_launched=claim.capture_ordinal == 1,
    )
    _write_json_exclusive(
        root / "failure-receipt.json",
        receipt.model_dump(mode="json", exclude_computed_fields=True),
    )
    replay = _independent_verify_failure(root, allow_missing_seal=True)
    expected_replay = {"failure_receipt_id": receipt.failure_receipt_id}
    if replay != expected_replay:
        raise ValueError(
            "SEQAX_SILU_FUSION_FAILURE_REPLAY_MISMATCH "
            f"expected={expected_replay} observed={replay}"
        )
    replay_seal = SeqaxSiluFusionCompilerFailureReplaySeal(
        design_id=design.design_id,
        capture_ordinal=claim.capture_ordinal,
        claim_id=claim.claim_id,
        failure_receipt_id=receipt.failure_receipt_id,
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root=str(root),
        independent_replay_performed=True,
        retry_authorized=False,
        ordinal_one_launched=claim.capture_ordinal == 1,
    )
    _write_json_exclusive(
        _failure_replay_seal_path(design, claim.capture_ordinal),
        replay_seal.model_dump(mode="json", exclude_computed_fields=True),
    )
    sealed_replay = _independent_verify_failure(root, allow_missing_seal=False)
    expected_sealed_replay = {
        "failure_receipt_id": receipt.failure_receipt_id,
        "failure_replay_seal_id": replay_seal.failure_replay_seal_id,
    }
    if sealed_replay != expected_sealed_replay:
        raise ValueError(
            "SEQAX_SILU_FUSION_FAILURE_SEAL_REPLAY_MISMATCH "
            f"expected={expected_sealed_replay} observed={sealed_replay}"
        )
    return receipt, replay_seal


def run_capture(
    root: Path,
    design_path: Path,
    ordinal: int,
) -> SeqaxSiluFusionCompilerReceipt:
    repository_root = Path(__file__).resolve().parents[2]
    canonical_design_path = repository_root / _DESIGN_RELATIVE_PATH
    if design_path.resolve(strict=True) != canonical_design_path:
        raise ValueError("SEQAX_SILU_FUSION_DESIGN_PATH_MISMATCH")
    design = SeqaxSiluFusionDesignContract.model_validate_json(design_path.read_text())
    if design != default_seqax_silu_fusion_design_contract(design.runtime):
        raise ValueError("SEQAX_SILU_FUSION_DESIGN_NONCANONICAL")
    if ordinal not in design.capture_ordinals:
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_ORDINAL_INVALID")
    root = _require_safe_new_root(root, design, ordinal)
    source, blobs = _source_authority(repository_root, design)
    if ordinal == 1:
        _require_prior_replay_seal(design, source)
    claim_path, claim = _prepare_claim(root, design, ordinal, source)
    request = SeqaxSiluFusionCompilerWorkerRequest(claim=claim, design=design, source=source)
    request_payload = request.wire_payload()
    if (
        SeqaxSiluFusionCompilerWorkerRequest.model_validate_json(json.dumps(request_payload))
        != request
    ):
        raise ValueError("SEQAX_SILU_FUSION_WORKER_REQUEST_ROUNDTRIP_MISMATCH")
    root.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(root.parent)
    _write_json_exclusive(
        root / "attempt_claim.json",
        claim.model_dump(mode="json", exclude_computed_fields=True),
    )
    _write_bytes_exclusive(root / "contract.json", blobs[_DESIGN_RELATIVE_PATH.as_posix()])
    _write_json_exclusive(
        root / "source.json",
        source.model_dump(mode="json", exclude_computed_fields=True),
    )
    _write_json_exclusive(
        root / "source_manifest.json",
        [value.model_dump(mode="json") for value in source.source_manifest],
    )
    _write_source_bundle(root, blobs)
    request_path = root / "worker_request.json"
    _write_json_exclusive(request_path, request_payload)
    EvidenceRun(root / "ledger.sqlite", claim.claim_id).create(
        {
            "claim_id": claim.claim_id,
            "claim_path": str(claim_path),
            "design_id": design.design_id,
            "capture_ordinal": ordinal,
        }
    )
    stage = "claim-reservation"
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        with _claim_lock(design, ordinal):
            if _require_claim_slot_unused(design, ordinal) != claim_path:
                raise ValueError("SEQAX_SILU_FUSION_CLAIM_PATH_MISMATCH")
            _reserve_claim(claim_path, claim)
        stage = "worker-launch"
        completed = _launch_worker(root, request_path, design)
        if completed.returncode != 0:
            stage = "worker-process"
            raise RuntimeError(
                "SEQAX_SILU_FUSION_WORKER_FAILED "
                f"returncode={completed.returncode} stderr={completed.stderr.strip()}"
            )
        stage = "worker-result"
        result = SeqaxSiluFusionCompilerWorkerResult.model_validate_json(
            (root / "worker-result.json").read_text()
        )
        if (
            result.capture.design_id != design.design_id
            or result.capture.capture_ordinal != ordinal
            or result.capture.invocation_id != claim.invocation_id
            or result.capture.claim_id != claim.claim_id
            or result.capture.source != source
        ):
            raise ValueError("SEQAX_SILU_FUSION_WORKER_RESULT_MISMATCH")
        stage = "receipt"
        artifacts = build_artifact_manifest(root, role_for_path=_artifact_role)
        receipt = SeqaxSiluFusionCompilerReceipt(
            capture=result.capture,
            final_ledger_state=RunState.COMPILED,
            artifacts=artifacts,
        )
        _write_json_exclusive(
            root / "receipt.json",
            receipt.model_dump(mode="json", exclude_computed_fields=True),
        )
        stage = "independent-replay"
        replay = _independent_verify(root)
        expected_replay = {
            "capture_id": receipt.capture.capture_id,
            "receipt_id": receipt.receipt_id,
            "semantic_pair_id": receipt.capture.semantic_pair_id,
        }
        if replay != expected_replay:
            raise ValueError(
                "SEQAX_SILU_FUSION_INDEPENDENT_REPLAY_MISMATCH "
                f"expected={expected_replay} observed={replay}"
            )
        stage = "replay-seal"
        _seal_replay(design, claim, receipt)
    except BaseException as error:
        try:
            reserved_claim = SeqaxSiluFusionCompilerAttemptClaim.model_validate_json(
                claim_path.read_text()
            )
        except (OSError, ValueError):
            reserved_claim = None
        if reserved_claim != claim:
            raise
        failure_receipt, failure_replay_seal = _record_failure(
            root,
            design,
            claim,
            source,
            stage=stage,
            error=error,
            completed=completed,
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(
            "SEQAX_SILU_FUSION_CAPTURE_FAILED "
            f"failure_receipt_id={failure_receipt.failure_receipt_id} "
            f"failure_replay_seal_id={failure_replay_seal.failure_replay_seal_id} "
            f"stage={stage} error={str(error) or repr(error)}"
        ) from error
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    arguments = parser.parse_args()
    receipt = run_capture(arguments.root, arguments.design, arguments.ordinal)
    print(
        json.dumps(
            {
                "capture_id": receipt.capture.capture_id,
                "receipt_id": receipt.receipt_id,
                "semantic_pair_id": receipt.capture.semantic_pair_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
