from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import jax
import numpy as np

from tpu_cake.artifacts import (
    build_artifact_manifest,
    file_sha256,
    save_array,
    validate_artifact_manifest,
    write_json,
    write_text,
)
from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import validate_compiler_analysis, write_compiler_analysis
from tpu_cake.contracts import ArtifactReference, ArtifactRole, SourceFileContract
from tpu_cake.identity import array_sha256, arrays_sha256, json_sha256, semantic_sha256
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_large_residual import default_seqax_large_residual_contract
from tpu_cake.seqax_large_residual_qualification import (
    SeqaxLargeResidualCrossCandidateObservation,
    SeqaxLargeResidualDeviceMemory,
    SeqaxLargeResidualHost,
    SeqaxLargeResidualQualificationClaim,
    SeqaxLargeResidualQualificationContract,
    SeqaxLargeResidualQualificationFailure,
    SeqaxLargeResidualQualificationObservation,
    SeqaxLargeResidualQualificationReceipt,
    SeqaxLargeResidualQualificationResult,
    analyze_large_residual_boundary,
    default_seqax_large_residual_qualification_contract,
)
from tpu_cake.seqax_large_residual_runner import SeqaxLargeResidualCompilerCaptureRecord
from tpu_cake.seqax_numerical import assess_seqax_bf16_final_outputs
from tpu_cake.seqax_pallas_search_runner import _validate_output_abi
from tpu_cake.seqax_residual_profile_runner import (
    _compile,
    _device_inventory,
    _execute,
    _json_sha256,
    _prepare_candidates,
    _resident_inputs,
    _text_sha256,
    _validate_devices,
)
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)

_CONTRACT_PATH = Path("contracts/seqax-large-residual-qualification-v1.json")
_COMPILER_CAPTURE_PATH = Path("contracts/seqax-large-residual-compiler-captures-v1.json")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "artifacts.py",
        package / "canonical.py",
        package / "compiler_analysis.py",
        package / "contracts.py",
        package / "cost_model.py",
        package / "dtensor_interpreter.py",
        package / "identity.py",
        package / "jax_lowering.py",
        package / "lowering.py",
        package / "physical_cost_model.py",
        package / "physical_geometry.py",
        package / "runner.py",
        package / "seqax_large_residual.py",
        package / "seqax_large_residual_runner.py",
        package / "seqax_large_residual_qualification.py",
        package / "seqax_large_residual_qualification_runner.py",
        package / "seqax_numerical.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_pallas_search_runner.py",
        package / "seqax_physical_execution.py",
        package / "seqax_physical_lowering.py",
        package / "seqax_residual_profile.py",
        package / "seqax_residual_profile_runner.py",
        package / "stablehlo.py",
        package / "dialects" / "distributed_tensor.py",
        package / "dialects" / "tpu_schedule.py",
        package / "workloads" / "seqax_forward.py",
        package / "workloads" / "seqax_oracle.py",
    )
    return tuple(
        SourceFileContract(
            path=path.relative_to(package.parent).as_posix(),
            sha256=file_sha256(path),
        )
        for path in paths
    )


def _repository_state(repository_root: Path) -> tuple[str, str, list[str], str]:
    source_commit = subprocess.run(
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
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return source_commit, file_sha256(repository_root / "uv.lock"), status, diff


def _require_safe_new_root(root: Path, repository_root: Path) -> None:
    absolute = root.absolute()
    protected = (Path("/"), Path.home().resolve(), repository_root.resolve())
    if root.exists() or root.is_symlink():
        raise ValueError(f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_ROOT_EXISTS path={root}")
    if any(absolute == value or absolute in value.parents for value in protected) or (
        repository_root.resolve() in absolute.parents
    ):
        raise ValueError(f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_UNSAFE_ROOT path={root}")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_PATH_SYMLINK path={current}")


def _preflight(
    root: Path,
    contract: SeqaxLargeResidualQualificationContract,
) -> tuple[Path, str, str, str, tuple[Any, ...], SeqaxLargeResidualHost]:
    repository_root = Path(__file__).resolve().parents[2]
    if repository_root.resolve() != Path(contract.compilation_source_root):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_COMPILATION_ROOT_MISMATCH")
    _require_safe_new_root(root, repository_root)
    source_commit, uv_lock_sha256, status, diff = _repository_state(repository_root)
    if status:
        raise ValueError(f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_SOURCE_DIRTY status={status}")
    if _runtime_identity() != contract.runtime:
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_RUNTIME_MISMATCH")
    devices = tuple(jax.devices())
    _validate_devices(devices, default_seqax_large_residual_contract(contract.runtime))
    if tuple(value.id for value in _device_inventory(devices)) != tuple(range(8)):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_DEVICE_INVENTORY_MISMATCH")
    host = _host_identity()
    if host != _expected_host(contract):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_HOST_MISMATCH")
    return repository_root, source_commit, uv_lock_sha256, diff, devices, host


def _claim_attempt(
    root: Path,
    contract: SeqaxLargeResidualQualificationContract,
    source_commit: str,
    uv_lock_sha256: str,
) -> tuple[SeqaxLargeResidualQualificationClaim, Path]:
    attempt_id = semantic_sha256(contract.qualification_id, source_commit, uv_lock_sha256)
    claim = SeqaxLargeResidualQualificationClaim(
        qualification_id=contract.qualification_id,
        attempt_id=attempt_id,
        source_commit=source_commit,
        uv_lock_sha256=uv_lock_sha256,
    )
    claim_path = Path(contract.attempt_registry_root) / f"{contract.qualification_id}.json"
    payload = claim.model_dump(mode="json")
    _write_exclusive_json(claim_path, payload)
    root.mkdir(parents=True, exist_ok=False)
    try:
        _write_exclusive_json(root / "attempt_claim.json", payload)
    except Exception as error:
        _record_failure(root, attempt_id, error)
        raise
    return claim, claim_path


def _record_failure(root: Path, attempt_id: str, error: Exception) -> None:
    if not (root / "attempt_claim.json").is_file():
        return
    failure = SeqaxLargeResidualQualificationFailure(
        attempt_id=attempt_id,
        error_type=type(error).__name__,
        error=str(error),
    )
    try:
        _write_exclusive_json(root / "failure.json", failure.model_dump(mode="json"))
    except FileExistsError:
        return


def _device_memory(devices: tuple[Any, ...]) -> tuple[SeqaxLargeResidualDeviceMemory, ...]:
    observations = []
    for device in devices:
        stats = device.memory_stats()
        if stats is None:
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_MEMORY_STATS_UNAVAILABLE")
        try:
            observations.append(
                SeqaxLargeResidualDeviceMemory(
                    device_id=device.id,
                    bytes_limit=int(stats["bytes_limit"]),
                    bytes_in_use=int(stats["bytes_in_use"]),
                    peak_bytes_in_use=int(stats["peak_bytes_in_use"]),
                    largest_alloc_size=int(stats["largest_alloc_size"]),
                )
            )
        except KeyError as error:
            raise ValueError(
                f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_MEMORY_FIELD_MISSING field={error.args[0]}"
            ) from error
    return tuple(observations)


def _load_compiler_capture(
    repository_root: Path,
    contract: SeqaxLargeResidualQualificationContract,
) -> SeqaxLargeResidualCompilerCaptureRecord:
    record = SeqaxLargeResidualCompilerCaptureRecord.model_validate_json(
        (repository_root / _COMPILER_CAPTURE_PATH).read_text()
    )
    if (
        record.record_id != contract.compiler_capture_record_id
        or record.source_commit != contract.compiler_capture_source_commit
        or record.capture.uv_lock_sha256 != contract.compiler_capture_uv_lock_sha256
        or record.capture_invocation_ids != contract.compiler_capture_invocation_ids
        or record.capture_log_sha256 != contract.compiler_capture_log_sha256
    ):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_CAPTURE_PROVENANCE_MISMATCH")
    return record


def _validate_compiled_against_capture(
    compiled: tuple[Any, ...],
    capture: SeqaxLargeResidualCompilerCaptureRecord,
) -> None:
    for observed, expected in zip(compiled, capture.capture.candidates, strict=True):
        values = (
            observed.prepared.expected.candidate,
            observed.prepared.plan.distributed_schedule_sha256,
            observed.prepared.plan.physical_schedule_sha256,
            observed.prepared.plan.source_sha256(),
            _json_sha256(observed.prepared.plan.manifest()),
            _text_sha256(observed.pallas_stablehlo),
            _text_sha256(observed.pallas_compiler_hlo),
            _text_sha256(observed.control_stablehlo),
            _text_sha256(observed.control_compiler_hlo),
            observed.pallas_compiler_analysis.collectives,
            observed.control_compiler_analysis.collectives,
        )
        required = (
            expected.candidate,
            expected.distributed_schedule_sha256,
            expected.physical_schedule_sha256,
            expected.pallas_source_sha256,
            expected.pallas_manifest_sha256,
            expected.pallas_stablehlo_sha256,
            expected.pallas_compiler_hlo_sha256,
            expected.control_stablehlo_sha256,
            expected.control_compiler_hlo_sha256,
            expected.pallas_compiler_collectives,
            expected.control_compiler_collectives,
        )
        if values != required:
            raise ValueError(
                "SEQAX_LARGE_RESIDUAL_QUALIFICATION_COMPILER_CAPTURE_MISMATCH "
                f"candidate={expected.candidate}"
            )


def _write_compiler_artifacts(root: Path, compiled: tuple[Any, ...]) -> None:
    for value in compiled:
        candidate_root = root / "candidates" / value.prepared.expected.candidate
        write_text(candidate_root / "distributed.xdsl", canonical_text(value.prepared.distributed))
        write_text(candidate_root / "physical.xdsl", canonical_text(value.prepared.physical))
        write_text(
            candidate_root / "lowered_pallas.py", value.prepared.plan.render_executable_source()
        )
        write_json(candidate_root / "plan_manifest.json", value.prepared.plan.manifest())
        write_text(candidate_root / "pallas_stablehlo.txt", value.pallas_stablehlo)
        write_text(candidate_root / "pallas_compiler_hlo.txt", value.pallas_compiler_hlo)
        write_text(candidate_root / "control_stablehlo.txt", value.control_stablehlo)
        write_text(candidate_root / "control_compiler_hlo.txt", value.control_compiler_hlo)
        write_compiler_analysis(
            candidate_root / "pallas_compiler_analysis.json",
            value.pallas_compiler_analysis,
        )
        write_compiler_analysis(
            candidate_root / "control_compiler_analysis.json",
            value.control_compiler_analysis,
        )


def _save_inputs_once(root: Path, seed: int, inputs: tuple[np.ndarray, ...]) -> None:
    for index, value in enumerate(inputs):
        save_array(root / "inputs" / str(seed) / f"{index:02d}.npy", value)


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    fixed = {
        "attempt_claim.json": ArtifactRole.EXPERIMENT,
        "contract.json": ArtifactRole.EXPERIMENT,
        "compiler_capture.json": ArtifactRole.EXPERIMENT,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "source_manifest.json": ArtifactRole.BACKEND_MANIFEST,
        "host.json": ArtifactRole.PREFLIGHT_RESULT,
        "memory_before_residency.json": ArtifactRole.SEARCH_EVIDENCE,
        "memory_after_residency.json": ArtifactRole.SEARCH_EVIDENCE,
        "result.json": ArtifactRole.SEARCH_RESULT,
    }
    if relative in fixed:
        return fixed[relative]
    if relative.startswith("inputs/") and path.suffix == ".npy":
        return ArtifactRole.CORRECTNESS_INPUT
    if relative.startswith("oracles/") and path.suffix == ".npy":
        return ArtifactRole.ORACLE_OUTPUT
    if relative.startswith("outputs/") and path.suffix == ".npy":
        return ArtifactRole.CORRECTNESS_OUTPUT
    if relative.startswith("candidates/"):
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
        }
        if path.name in roles:
            return roles[path.name]
    raise ValueError(f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_ARTIFACT_UNRECOGNIZED path={relative}")


def _manifest(root: Path) -> tuple[ArtifactReference, ...]:
    return build_artifact_manifest(
        root,
        role_for_path=_artifact_role,
        excluded_paths=("manifest.json", "receipt.json"),
    )


def _validate_claim(
    root: Path,
    contract: SeqaxLargeResidualQualificationContract,
) -> SeqaxLargeResidualQualificationClaim:
    copied = root / "attempt_claim.json"
    claim_path = Path(contract.attempt_registry_root) / f"{contract.qualification_id}.json"
    if (
        copied.is_symlink()
        or claim_path.is_symlink()
        or not copied.is_file()
        or not claim_path.is_file()
        or copied.stat().st_nlink != 1
        or claim_path.stat().st_nlink != 1
        or copied.read_bytes() != claim_path.read_bytes()
    ):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_CLAIM_MISMATCH")
    return SeqaxLargeResidualQualificationClaim.model_validate_json(copied.read_text())


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
            f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_METADATA_REDIRECT code={code} url={newurl}"
        )


def _metadata(path: str) -> str:
    request = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectMetadataRedirects(),
    )
    with opener.open(request, timeout=5) as response:
        if response.headers.get("Metadata-Flavor") != "Google":
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_METADATA_HEADER_MISSING")
        payload = response.read(4097)
    if len(payload) > 4096:
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_METADATA_RESPONSE_TOO_LARGE")
    return payload.decode().strip()


def _host_identity() -> SeqaxLargeResidualHost:
    zone_resource = _metadata("instance/zone")
    machine_type_resource = _metadata("instance/machine-type")
    return SeqaxLargeResidualHost(
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


def _expected_host(
    contract: SeqaxLargeResidualQualificationContract,
) -> SeqaxLargeResidualHost:
    return SeqaxLargeResidualHost(
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


def verify_seqax_large_residual_qualification(
    root: Path,
    contract: SeqaxLargeResidualQualificationContract,
    *,
    replay_cpu_oracle: bool = False,
) -> SeqaxLargeResidualQualificationResult:
    claim = _validate_claim(root, contract)
    saved_contract = SeqaxLargeResidualQualificationContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != contract or claim.qualification_id != contract.qualification_id:
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_IDENTITY_MISMATCH")
    result = SeqaxLargeResidualQualificationResult.model_validate_json(
        (root / "result.json").read_text()
    )
    if (
        result.qualification_id != contract.qualification_id
        or result.attempt_id != claim.attempt_id
    ):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_RESULT_IDENTITY_MISMATCH")
    artifacts = tuple(
        ArtifactReference.model_validate(value)
        for value in json.loads((root / "manifest.json").read_text())
    )
    validate_artifact_manifest(
        root,
        artifacts,
        role_for_path=_artifact_role,
        duplicate_error="SEQAX_LARGE_RESIDUAL_QUALIFICATION_MANIFEST_DUPLICATE",
        closed_world_error="SEQAX_LARGE_RESIDUAL_QUALIFICATION_MANIFEST_CLOSED_WORLD_MISMATCH",
        mismatch_error=lambda path: (
            f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_MANIFEST_ARTIFACT_MISMATCH path={path}"
        ),
        symlink_error="SEQAX_LARGE_RESIDUAL_QUALIFICATION_MANIFEST_SYMLINK",
        excluded_paths=("manifest.json", "receipt.json"),
    )
    source_state = json.loads((root / "source_state.json").read_text())
    source_manifest_payload = json.loads((root / "source_manifest.json").read_text())
    source_manifest = tuple(
        SourceFileContract.model_validate(value) for value in source_manifest_payload
    )
    if (
        result.source_commit != claim.source_commit
        or source_state["git_commit"] != claim.source_commit
        or source_state["uv_lock_sha256"] != claim.uv_lock_sha256
        or source_state["git_dirty"] is not False
        or source_state["git_status"] != []
        or (root / "source_diff.patch").read_bytes() != b""
        or source_state["source_diff_sha256"] != hashlib.sha256(b"").hexdigest()
        or result.source_manifest != source_manifest
        or result.source_manifest_sha256 != json_sha256(source_manifest_payload)
    ):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_SOURCE_STATE_MISMATCH")
    host = SeqaxLargeResidualHost.model_validate_json((root / "host.json").read_text())
    if host != result.host or host != _expected_host(contract):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_HOST_REPLAY_MISMATCH")
    repository_root = Path(__file__).resolve().parents[2]
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != claim.source_commit:
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_REPLAY_SOURCE_COMMIT_MISMATCH")
    for source in source_manifest:
        path = repository_root / "src" / source.path
        if file_sha256(path) != source.sha256:
            raise ValueError(
                f"SEQAX_LARGE_RESIDUAL_QUALIFICATION_SOURCE_BLOB_MISMATCH path={source.path}"
            )
    capture = SeqaxLargeResidualCompilerCaptureRecord.model_validate_json(
        (root / "compiler_capture.json").read_text()
    )
    if capture.record_id != contract.compiler_capture_record_id:
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_CAPTURE_RECORD_MISMATCH")
    prepared = _prepare_candidates(default_seqax_large_residual_contract(contract.runtime))
    for value, expected in zip(prepared, capture.capture.candidates, strict=True):
        candidate_root = root / "candidates" / value.expected.candidate
        pallas_analysis = validate_compiler_analysis(
            candidate_root / "pallas_compiler_analysis.json",
            stablehlo_path=candidate_root / "pallas_stablehlo.txt",
            compiler_hlo_path=candidate_root / "pallas_compiler_hlo.txt",
        )
        control_analysis = validate_compiler_analysis(
            candidate_root / "control_compiler_analysis.json",
            stablehlo_path=candidate_root / "control_stablehlo.txt",
            compiler_hlo_path=candidate_root / "control_compiler_hlo.txt",
        )
        if (
            (candidate_root / "distributed.xdsl").read_text() != canonical_text(value.distributed)
            or (candidate_root / "physical.xdsl").read_text() != canonical_text(value.physical)
            or (candidate_root / "lowered_pallas.py").read_text()
            != value.plan.render_executable_source()
            or json.loads((candidate_root / "plan_manifest.json").read_text())
            != value.plan.manifest()
            or file_sha256(candidate_root / "pallas_stablehlo.txt")
            != expected.pallas_stablehlo_sha256
            or file_sha256(candidate_root / "pallas_compiler_hlo.txt")
            != expected.pallas_compiler_hlo_sha256
            or file_sha256(candidate_root / "control_stablehlo.txt")
            != expected.control_stablehlo_sha256
            or file_sha256(candidate_root / "control_compiler_hlo.txt")
            != expected.control_compiler_hlo_sha256
            or pallas_analysis.collectives != expected.pallas_compiler_collectives
            or control_analysis.collectives != expected.control_compiler_collectives
        ):
            raise ValueError(
                "SEQAX_LARGE_RESIDUAL_QUALIFICATION_PLAN_REPLAY_MISMATCH "
                f"candidate={value.expected.candidate}"
            )
    standard_hlo = (
        root / "candidates" / SeqaxResidualNormStrategy.STANDARD / "pallas_compiler_hlo.txt"
    ).read_text()
    if analyze_large_residual_boundary(standard_hlo) != result.standard_boundary:
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_BOUNDARY_REPLAY_MISMATCH")
    before = tuple(
        SeqaxLargeResidualDeviceMemory.model_validate(value)
        for value in json.loads((root / "memory_before_residency.json").read_text())
    )
    after = tuple(
        SeqaxLargeResidualDeviceMemory.model_validate(value)
        for value in json.loads((root / "memory_after_residency.json").read_text())
    )
    if before != result.memory_before_residency or after != result.memory_after_residency:
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_MEMORY_REPLAY_MISMATCH")
    output_shape = (
        contract.parameters["batch"],
        contract.parameters["sequence"],
        contract.parameters["vocabulary"],
    )
    for seed_index, seed in enumerate(contract.correctness_seeds):
        inputs = tuple(
            np.load(root / "inputs" / str(seed) / f"{index:02d}.npy", allow_pickle=False)
            for index in range(13)
        )
        regenerated = tuple(
            np.asarray(value) for value in seqax_forward_inputs(seed=seed, **contract.parameters)
        )
        if (
            any(
                not np.array_equal(saved, fresh)
                for saved, fresh in zip(inputs, regenerated, strict=True)
            )
            or arrays_sha256(inputs) != result.shared_input_sha256[seed_index]
        ):
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_INPUT_REPLAY_MISMATCH")
        cpu = np.load(root / "oracles" / str(seed) / "cpu.npy", allow_pickle=False)
        if array_sha256(cpu) != result.cpu_output_sha256[seed_index]:
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_ORACLE_HASH_MISMATCH")
        if replay_cpu_oracle:
            fresh_cpu = seqax_forward_canonical_reference(
                inputs,
                quantization_decimals=contract.numerical_policy.cpu_reference_quantization_decimals,
                **contract.parameters,
            )
            if not np.array_equal(cpu, fresh_cpu):
                raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_ORACLE_REPLAY_MISMATCH")
        pair = tuple(value for value in result.observations if value.seed == seed)
        replayed_outputs = {}
        for observation in pair:
            candidate_root = root / "outputs" / str(seed) / observation.candidate
            pallas = tuple(
                np.load(candidate_root / f"pallas-{index}.npy", allow_pickle=False)
                for index in range(contract.repeat_executions)
            )
            control = tuple(
                np.load(candidate_root / f"control-{index}.npy", allow_pickle=False)
                for index in range(contract.repeat_executions)
            )
            assessment = assess_seqax_bf16_final_outputs(
                pallas[0],
                control[0],
                cpu,
                policy=contract.numerical_policy,
                expected_shape=output_shape,
                layers=contract.parameters["layers"],
            )
            if (
                tuple(array_sha256(value) for value in pallas) != observation.pallas_repeat_sha256
                or tuple(array_sha256(value) for value in control)
                != observation.control_repeat_sha256
                or any(not np.array_equal(value, pallas[0]) for value in pallas[1:])
                or any(not np.array_equal(value, control[0]) for value in control[1:])
                or assessment != observation.assessment
            ):
                raise ValueError(
                    "SEQAX_LARGE_RESIDUAL_QUALIFICATION_OUTPUT_REPLAY_MISMATCH "
                    f"candidate={observation.candidate} seed={seed}"
                )
            replayed_outputs[observation.candidate] = pallas[0]
        cross = assess_seqax_bf16_final_outputs(
            replayed_outputs[SeqaxResidualNormStrategy.STANDARD],
            replayed_outputs[SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE],
            cpu,
            policy=contract.numerical_policy,
            expected_shape=output_shape,
            layers=contract.parameters["layers"],
        )
        if cross != result.cross_candidate[seed_index].standard_as_pallas_residual_as_control:
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_CROSS_REPLAY_MISMATCH")
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = SeqaxLargeResidualQualificationReceipt.model_validate_json(
            receipt_path.read_text()
        )
        if (
            receipt.qualification_id != contract.qualification_id
            or receipt.attempt_id != claim.attempt_id
            or receipt.result_sha256 != file_sha256(root / "result.json")
            or receipt.manifest_sha256 != file_sha256(root / "manifest.json")
            or receipt.artifacts != artifacts
        ):
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_RECEIPT_MISMATCH")
    return result


def run_seqax_large_residual_qualification(root: Path) -> SeqaxLargeResidualQualificationReceipt:
    runtime = _runtime_identity()
    repository_root = Path(__file__).resolve().parents[2]
    contract = SeqaxLargeResidualQualificationContract.model_validate_json(
        (repository_root / _CONTRACT_PATH).read_text()
    )
    if contract != default_seqax_large_residual_qualification_contract(runtime):
        raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_CONTRACT_MISMATCH")
    repository_root, source_commit, uv_lock_sha256, diff, devices, host = _preflight(
        root,
        contract,
    )
    claim: SeqaxLargeResidualQualificationClaim | None = None
    try:
        claim, _claim_path = _claim_attempt(
            root,
            contract,
            source_commit,
            uv_lock_sha256,
        )
        source_manifest = _source_manifest()
        source_manifest_payload = tuple(value.model_dump(mode="json") for value in source_manifest)
        write_json(
            root / "contract.json",
            contract.model_dump(mode="json", exclude_computed_fields=True),
        )
        write_json(
            root / "source_state.json",
            {
                "git_commit": source_commit,
                "git_dirty": False,
                "git_status": [],
                "source_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "uv_lock_sha256": uv_lock_sha256,
                "python_executable": os.path.realpath(os.sys.executable),
            },
        )
        write_text(root / "source_diff.patch", diff)
        write_json(root / "source_manifest.json", source_manifest_payload)
        write_json(root / "host.json", host.model_dump(mode="json"))
        capture = _load_compiler_capture(repository_root, contract)
        write_json(
            root / "compiler_capture.json",
            capture.model_dump(mode="json", exclude_computed_fields=True),
        )

        host_inputs_by_seed = tuple(
            tuple(
                np.asarray(value)
                for value in seqax_forward_inputs(seed=seed, **contract.parameters)
            )
            for seed in contract.correctness_seeds
        )
        for seed, host_inputs in zip(
            contract.correctness_seeds,
            host_inputs_by_seed,
            strict=True,
        ):
            _save_inputs_once(root, seed, host_inputs)

        prepared = _prepare_candidates(default_seqax_large_residual_contract(runtime))
        compiled = tuple(_compile(value, host_inputs_by_seed[0], devices) for value in prepared)
        _validate_compiled_against_capture(compiled, capture)
        standard_boundary = analyze_large_residual_boundary(compiled[0].pallas_compiler_hlo)
        if standard_boundary.chain_count != contract.expected_standard_boundary_chains:
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_BOUNDARY_COUNT_MISMATCH")
        _write_compiler_artifacts(root, compiled)

        memory_before = _device_memory(devices)
        write_json(
            root / "memory_before_residency.json",
            [value.model_dump(mode="json") for value in memory_before],
        )
        if any(
            value.plan.input_contracts != prepared[0].plan.input_contracts for value in prepared[1:]
        ):
            raise ValueError("SEQAX_LARGE_RESIDUAL_QUALIFICATION_INPUT_ABI_MISMATCH")
        shared_resident_inputs = tuple(
            _resident_inputs(inputs, prepared[0], compiled[0].mesh)
            for inputs in host_inputs_by_seed
        )
        jax.block_until_ready(shared_resident_inputs)
        memory_after = _device_memory(devices)
        write_json(
            root / "memory_after_residency.json",
            [value.model_dump(mode="json") for value in memory_after],
        )

        observations = []
        cross_candidate = []
        cpu_hashes = []
        output_shape = (
            contract.parameters["batch"],
            contract.parameters["sequence"],
            contract.parameters["vocabulary"],
        )
        for seed_index, seed in enumerate(contract.correctness_seeds):
            host_inputs = host_inputs_by_seed[seed_index]
            resident_inputs = shared_resident_inputs[seed_index]
            cpu = seqax_forward_canonical_reference(
                host_inputs,
                quantization_decimals=contract.numerical_policy.cpu_reference_quantization_decimals,
                **contract.parameters,
            )
            save_array(root / "oracles" / str(seed) / "cpu.npy", cpu)
            cpu_hash = array_sha256(cpu)
            cpu_hashes.append(cpu_hash)
            seed_outputs: dict[SeqaxResidualNormStrategy, np.ndarray] = {}
            for value in compiled:
                candidate = value.prepared.expected.candidate
                pallas_repeats = tuple(
                    _execute(value.pallas_executable, resident_inputs)
                    for _ in range(contract.repeat_executions)
                )
                control_repeats = tuple(
                    _execute(value.control_executable, resident_inputs)
                    for _ in range(contract.repeat_executions)
                )
                for output in (*pallas_repeats, *control_repeats):
                    _validate_output_abi(output, value.prepared.plan.output_contracts[0], candidate)
                for repeat_index, output in enumerate(pallas_repeats):
                    save_array(
                        root / "outputs" / str(seed) / candidate / f"pallas-{repeat_index}.npy",
                        output,
                    )
                for repeat_index, output in enumerate(control_repeats):
                    save_array(
                        root / "outputs" / str(seed) / candidate / f"control-{repeat_index}.npy",
                        output,
                    )
                assessment = assess_seqax_bf16_final_outputs(
                    pallas_repeats[0],
                    control_repeats[0],
                    cpu,
                    policy=contract.numerical_policy,
                    expected_shape=output_shape,
                    layers=contract.parameters["layers"],
                )
                observation = SeqaxLargeResidualQualificationObservation(
                    candidate=candidate,
                    seed=seed,
                    input_sha256=arrays_sha256(host_inputs),
                    cpu_output_sha256=cpu_hash,
                    control_output_sha256=array_sha256(control_repeats[0]),
                    pallas_output_sha256=array_sha256(pallas_repeats[0]),
                    pallas_repeat_sha256=tuple(array_sha256(value) for value in pallas_repeats),
                    control_repeat_sha256=tuple(array_sha256(value) for value in control_repeats),
                    assessment=assessment,
                )
                observations.append(observation)
                seed_outputs[candidate] = pallas_repeats[0]
            standard = seed_outputs[SeqaxResidualNormStrategy.STANDARD]
            residual = seed_outputs[SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE]
            cross_candidate.append(
                SeqaxLargeResidualCrossCandidateObservation(
                    seed=seed,
                    standard_output_sha256=array_sha256(standard),
                    residual_all_reduce_output_sha256=array_sha256(residual),
                    standard_as_pallas_residual_as_control=assess_seqax_bf16_final_outputs(
                        standard,
                        residual,
                        cpu,
                        policy=contract.numerical_policy,
                        expected_shape=output_shape,
                        layers=contract.parameters["layers"],
                    ),
                )
            )

        result = SeqaxLargeResidualQualificationResult(
            qualification_id=contract.qualification_id,
            attempt_id=claim.attempt_id,
            source_commit=source_commit,
            source_manifest=source_manifest,
            source_manifest_sha256=json_sha256(source_manifest_payload),
            compiler_capture_record_id=capture.record_id,
            host=host,
            standard_boundary=standard_boundary,
            shared_input_sha256=tuple(arrays_sha256(value) for value in host_inputs_by_seed),
            cpu_output_sha256=tuple(cpu_hashes),
            observations=tuple(observations),
            cross_candidate=tuple(cross_candidate),
            memory_before_residency=memory_before,
            memory_after_residency=memory_after,
            producer_passed=True,
            independent_replay_performed=False,
        )
        write_json(root / "result.json", result.model_dump(mode="json"))
        artifacts = _manifest(root)
        write_json(root / "manifest.json", [value.model_dump(mode="json") for value in artifacts])
        verify_seqax_large_residual_qualification(root, contract)
        receipt = SeqaxLargeResidualQualificationReceipt(
            qualification_id=contract.qualification_id,
            attempt_id=claim.attempt_id,
            result_sha256=file_sha256(root / "result.json"),
            manifest_sha256=file_sha256(root / "manifest.json"),
            artifacts=artifacts,
        )
        _write_exclusive_json(root / "receipt.json", receipt.model_dump(mode="json"))
        return receipt
    except Exception as error:
        if claim is not None:
            _record_failure(root, claim.attempt_id, error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = run_seqax_large_residual_qualification(args.output)
    print(receipt.model_dump_json())


if __name__ == "__main__":
    main()
