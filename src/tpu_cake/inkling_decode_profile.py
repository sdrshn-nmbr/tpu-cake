from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import jax
import psutil
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from xprof import profile_data

from tpu_cake.contracts import ProfileExpectation, RuntimeIdentity, SourceFileContract
from tpu_cake.evidence import (
    CaptureAssessment,
    CaptureEvidence,
    Finding,
    FindingSeverity,
    ProgramEvidence,
)
from tpu_cake.xprof_evidence import assess_evidence, collect_capture

INKLING_DECODE_PROFILE_SCHEMA = "inkling-whole-decode-profile-v1"


class HloIdentityStatus(StrEnum):
    PENDING = "pending"
    PINNED = "pinned"


class InklingDecodePromptContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_case_ids: tuple[str, ...]

    @model_validator(mode="after")
    def cases_are_unique(self) -> InklingDecodePromptContract:
        if not self.selected_case_ids:
            raise ValueError("prompt selection cannot be empty")
        if len(self.selected_case_ids) != len(set(self.selected_case_ids)):
            raise ValueError("prompt case IDs must be unique")
        return self


class InklingDecodeServerContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dtype: Literal["bfloat16"]
    quantization: None = None
    context_length: int = Field(gt=0)
    max_running_requests: int = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
    max_prefill_tokens: int = Field(gt=0)
    chunked_prefill_size: int = Field(gt=0)
    page_size: int = Field(gt=0)
    tp_size: int = Field(gt=0)
    ep_size: int = Field(gt=0)
    attention_backend: str
    moe_backend: str
    disable_overlap_schedule: bool
    disable_radix_cache: bool
    speculative_algorithm: None = None
    version: str


class InklingDecodeProgramContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name_prefix: str = Field(min_length=1)
    hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    module_events_per_tpu_core: dict[str, int]


class InklingDecodeProfileContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["inkling-whole-decode-profile-v1"] = INKLING_DECODE_PROFILE_SCHEMA
    name: str = Field(min_length=1)
    hlo_identity_status: HloIdentityStatus
    inkling_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    inkling_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inkling_source_root: str = Field(min_length=1)
    capture_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    capture_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_source_manifest: tuple[SourceFileContract, ...]
    server_command_fragments: tuple[str, ...]
    runtime: RuntimeIdentity
    xplane_device_type: Literal["TPU v7x"]
    xplane_tpu_version: str = Field(min_length=1)
    device_count: Literal[8]
    process_count: Literal[1]
    concurrency: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    profile_stop_after_minimum_completion_tokens: int = Field(gt=1)
    host_tracer_level: Literal[0]
    python_tracer_level: Literal[0]
    prompts: InklingDecodePromptContract
    server: InklingDecodeServerContract
    profile: ProfileExpectation
    main_program_prefix: str = Field(min_length=1)
    programs: tuple[InklingDecodeProgramContract, ...]

    @computed_field
    @property
    def contract_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"contract_id"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @model_validator(mode="after")
    def protocol_is_complete(self) -> InklingDecodeProfileContract:
        if self.profile_stop_after_minimum_completion_tokens > self.output_tokens:
            raise ValueError("profile stop completion must not exceed output tokens")
        if len(self.prompts.selected_case_ids) != self.concurrency:
            raise ValueError("prompt selection must match concurrency")
        if self.server.tp_size != self.device_count:
            raise ValueError("server tensor parallelism must match physical device count")
        prefixes = [program.name_prefix for program in self.programs]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("program prefixes must be unique")
        if self.main_program_prefix not in prefixes:
            raise ValueError("main program must be part of the required program inventory")
        zero = "0" * 64
        zero_commit = "0" * 40
        hashes = [program.hlo_sha256 for program in self.programs]
        expected_planes = {f"/device:TPU:{index}" for index in range(self.server.tp_size)}
        if self.hlo_identity_status is HloIdentityStatus.PENDING:
            if (
                self.capture_git_commit != zero_commit
                or any(digest != zero for digest in hashes)
                or any(program.module_events_per_tpu_core for program in self.programs)
            ):
                raise ValueError("pending HLO and module-event identities must be placeholders")
        elif (
            self.capture_git_commit == zero_commit
            or any(digest == zero for digest in hashes)
            or any(
                set(program.module_events_per_tpu_core) != expected_planes
                or any(count <= 0 for count in program.module_events_per_tpu_core.values())
                for program in self.programs
            )
        ):
            raise ValueError("pinned HLO and module-event identities must be complete")
        paths = [source.path for source in self.capture_source_manifest]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("capture source manifest must be nonempty and unique")
        if not self.server_command_fragments:
            raise ValueError("server command fragments cannot be empty")
        return self


class InklingDecodePromptEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    native_input_tokens: int = Field(gt=0)
    input_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    standard_input_tokens: int | None = Field(default=None, gt=0)
    decoded_prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class InklingDecodeRequestEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rid: tuple[str, ...]
    sampling_params: dict[str, Any]
    stream: bool
    return_routed_experts: bool
    prompt_cases: tuple[InklingDecodePromptEvidence, ...]


class InklingDecodeChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    completion_tokens: int = Field(gt=0)
    request_state_slot: int = Field(ge=0)
    recurrent_state_slot: int = Field(ge=0)
    server_batch_size: int = Field(gt=0)


class InklingDecodeChunkResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    meta_info: InklingDecodeChunkMetadata
    output_ids: tuple[int, ...]
    text: str


class InklingDecodeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    elapsed_ms: float = Field(gt=0)
    response: InklingDecodeChunkResponse


class InklingDecodeServerProcess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pid: int = Field(gt=0)
    create_time: float = Field(gt=0)
    cwd: str
    cmdline: tuple[str, ...]


class InklingDecodeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    captured_at_utc: str
    command: tuple[str, ...]
    tpu_cake_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tpu_cake_git_status_porcelain: str
    tpu_cake_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_runtime: RuntimeIdentity
    capture_source_manifest: tuple[SourceFileContract, ...]
    inkling_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    inkling_git_status_porcelain: str
    inkling_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hostname: str
    python: str
    prompt_cases_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    profile_directory_name: str
    capture_xplane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    xplane_process_id: int = Field(gt=0)
    xplane_profile_start_time_ns: int = Field(gt=0)
    xplane_profile_stop_time_ns: int = Field(gt=0)
    profile_start_requested_at_ns: int = Field(gt=0)
    profile_stop_completed_at_ns: int = Field(gt=0)
    server_process: InklingDecodeServerProcess
    server_info: dict[str, Any]


class InklingDecodeProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: InklingDecodeRequestEvidence
    provenance: InklingDecodeProvenance
    profile_start_response: dict[str, Any]
    profile_stop_response: dict[str, Any]
    profile_start_condition: Literal["every request emitted at least one token"]
    profile_stop_after_minimum_completion_tokens: int = Field(gt=1)
    host_tracer_level: Literal[0]
    python_tracer_level: Literal[0]
    elapsed_ms: float = Field(gt=0)
    chunks_by_request: tuple[tuple[InklingDecodeChunk, ...], ...]


class InklingDecodeProfileAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture: CaptureAssessment
    required_programs: tuple[ProgramEvidence, ...]
    module_events_per_tpu_core: dict[str, dict[str, int]]
    xplane_runtime: dict[str, str]
    findings: tuple[Finding, ...]

    @computed_field
    @property
    def accepted(self) -> bool:
        return self.capture.accepted and not self.findings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity() -> RuntimeIdentity:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return RuntimeIdentity(
        python=platform.python_version(),
        jax=jax.__version__,
        jaxlib=version("jaxlib"),
        libtpu=version("libtpu"),
        xla=os.environ.get("LIBTPU_INIT_ARGS"),
    )


def _text(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _require_new_output(path: Path) -> None:
    absolute = path.absolute()
    if absolute.exists() or absolute.is_symlink():
        raise ValueError(f"INKLING_PROFILE_OUTPUT_EXISTS path={absolute}")
    for ancestor in absolute.parents:
        if ancestor.is_symlink():
            raise ValueError(f"INKLING_PROFILE_OUTPUT_ANCESTOR_IS_SYMLINK path={ancestor}")


def _require_new_profile_directory(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("INKLING_PROFILE_DIRECTORY_MUST_BE_ABSOLUTE")
    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_outside_repositories(path: Path, repositories: tuple[Path, ...]) -> None:
    target = path.resolve(strict=False)
    for repository in repositories:
        root = repository.resolve()
        if target == root or target.is_relative_to(root):
            raise ValueError(
                f"INKLING_PROFILE_OUTPUT_INSIDE_SOURCE_REPOSITORY path={target} root={root}"
            )


def _source_manifest(
    repo: Path, contract: InklingDecodeProfileContract
) -> tuple[SourceFileContract, ...]:
    observed: list[SourceFileContract] = []
    for expected in contract.capture_source_manifest:
        path = repo / expected.path
        if not path.is_file() or _sha256(path) != expected.sha256:
            raise ValueError(f"INKLING_PROFILE_CAPTURE_SOURCE_MISMATCH path={expected.path}")
        committed = subprocess.run(
            ["git", "show", f"HEAD:{expected.path}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != expected.sha256:
            raise ValueError(f"INKLING_PROFILE_CAPTURE_GIT_BLOB_MISMATCH path={expected.path}")
        observed.append(expected)
    return tuple(observed)


def _server_process(
    url: str, inkling_repo: Path, fragments: tuple[str, ...]
) -> InklingDecodeServerProcess:
    parsed = urlparse(url)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port is None:
        raise ValueError("INKLING_PROFILE_SERVER_MUST_BE_LOCAL")
    listeners = [
        connection
        for connection in psutil.net_connections(kind="tcp")
        if connection.status == psutil.CONN_LISTEN
        and connection.laddr
        and connection.laddr.port == parsed.port
        and connection.pid is not None
    ]
    pids = {connection.pid for connection in listeners}
    if len(pids) != 1:
        raise ValueError(f"INKLING_PROFILE_SERVER_LISTENER_MISMATCH pids={sorted(pids)}")
    process = psutil.Process(pids.pop())
    evidence = InklingDecodeServerProcess(
        pid=process.pid,
        create_time=process.create_time(),
        cwd=str(Path(process.cwd()).resolve()),
        cmdline=tuple(process.cmdline()),
    )
    if Path(evidence.cwd) != inkling_repo.resolve():
        raise ValueError(
            f"INKLING_PROFILE_SERVER_CWD_MISMATCH expected={inkling_repo.resolve()} "
            f"observed={evidence.cwd}"
        )
    command = " ".join(evidence.cmdline)
    missing = [fragment for fragment in fragments if fragment not in command]
    if missing:
        raise ValueError(f"INKLING_PROFILE_SERVER_COMMAND_MISMATCH missing={missing}")
    return evidence


def _wait_for_xplane(profile_directory: Path) -> Path:
    for _ in range(60):
        matches = tuple(profile_directory.rglob("*.xplane.pb"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("INKLING_PROFILE_XPLANE_INVENTORY_MISMATCH")
        time.sleep(1)
    raise TimeoutError("INKLING_PROFILE_XPLANE_NOT_PUBLISHED")


def _module_event_counts(xplane: Path) -> dict[str, dict[str, int]]:
    profile = profile_data.ProfileData.from_file(xplane)
    try:
        counts: dict[str, dict[str, int]] = {}
        for plane in profile.planes:
            if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
                continue
            module_lines = [line for line in plane.lines if line.name == "XLA Modules"]
            if len(module_lines) != 1:
                raise ValueError(f"INKLING_PROFILE_XLA_MODULE_LINE_MISMATCH plane={plane.name}")
            counts[plane.name] = dict(Counter(event.name for event in module_lines[0].events))
        return counts
    finally:
        profile.close()


def _xplane_identity(xplane: Path) -> tuple[int, int, int]:
    profile = profile_data.ProfileData.from_file(xplane)
    try:
        metadata = [plane for plane in profile.planes if plane.name == "/host:metadata"]
        if len(metadata) != 1:
            raise ValueError("INKLING_PROFILE_HOST_METADATA_INVENTORY_MISMATCH")
        stats = dict(metadata[0].stats)
        try:
            process_id = int(stats["process_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("INKLING_PROFILE_PROCESS_ID_MISSING") from error
        if process_id <= 0:
            raise ValueError("INKLING_PROFILE_PROCESS_ID_INVALID")
        environments = [plane for plane in profile.planes if plane.name == "Task Environment"]
        if len(environments) != 1:
            raise ValueError("INKLING_PROFILE_TASK_ENVIRONMENT_INVENTORY_MISMATCH")
        environment = dict(environments[0].stats)
        try:
            start_time = int(environment["profile_start_time"])
            stop_time = int(environment["profile_stop_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("INKLING_PROFILE_TIME_RANGE_MISSING") from error
        if not 0 < start_time < stop_time:
            raise ValueError("INKLING_PROFILE_TIME_RANGE_INVALID")
        return process_id, start_time, stop_time
    finally:
        profile.close()


def _xplane_runtime(xplane: Path) -> dict[str, str]:
    profile = profile_data.ProfileData.from_file(xplane)
    try:
        metadata = [plane for plane in profile.planes if plane.name == "/host:metadata"]
        if len(metadata) != 1:
            raise ValueError("INKLING_PROFILE_HOST_METADATA_INVENTORY_MISMATCH")
        stats = dict(metadata[0].stats)
        required = ("jax_version", "jaxlib_version", "tpu_version")
        if any(not stats.get(name) for name in required):
            raise ValueError("INKLING_PROFILE_RUNTIME_METADATA_MISSING")
        device_types = {
            dict(plane.stats).get("device_type_string")
            for plane in profile.planes
            if plane.name.startswith("/device:TPU:") and "SparseCore" not in plane.name
        }
        process_ids = {
            dict(plane.stats).get("process_id")
            for plane in profile.planes
            if plane.name.startswith("/device:TPU:") and "SparseCore" not in plane.name
        }
        if len(device_types) != 1 or None in device_types:
            raise ValueError("INKLING_PROFILE_DEVICE_TYPE_MISMATCH")
        if len(process_ids) != 1 or None in process_ids:
            raise ValueError("INKLING_PROFILE_PROCESS_INVENTORY_MISMATCH")
        return {
            "jax": stats["jax_version"],
            "jaxlib": stats["jaxlib_version"],
            "tpu_runtime": stats["tpu_version"],
            "device_type": device_types.pop(),
            "process_id": process_ids.pop(),
        }
    finally:
        profile.close()


def _write_json_new(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _post_json(url: str, endpoint: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        content = response.read()
        result: dict[str, Any] = {"status_code": response.status}
        if content:
            try:
                result["body"] = json.loads(content)
            except json.JSONDecodeError:
                result["body"] = content.decode()
        return result


def _get_json(url: str, endpoint: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{url.rstrip('/')}/{endpoint}", timeout=30) as response:
        return json.loads(response.read())


def _load_prompt_cases(path: Path, case_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise TypeError("INKLING_PROFILE_PROMPT_CORPUS_INVALID")
    cases = {case.get("id"): case for case in payload if isinstance(case, dict)}
    if len(cases) != len(payload):
        raise ValueError("INKLING_PROFILE_PROMPT_IDS_NOT_UNIQUE")
    try:
        selected = tuple(cases[case_id] for case_id in case_ids)
    except KeyError as error:
        raise ValueError(f"INKLING_PROFILE_PROMPT_MISSING id={error.args[0]}") from error
    if any(
        not isinstance(case.get("input_ids"), list)
        or not case["input_ids"]
        or any(type(token_id) is not int for token_id in case["input_ids"])
        for case in selected
    ):
        raise ValueError("INKLING_PROFILE_PROMPT_INPUT_IDS_INVALID")
    return selected


def _prompt_evidence(case: dict[str, Any]) -> dict[str, Any]:
    input_ids = case["input_ids"]
    return {
        "id": case["id"],
        "native_input_tokens": len(input_ids),
        "input_ids_sha256": hashlib.sha256(
            json.dumps(input_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "standard_input_tokens": case.get("standard_input_tokens"),
        "decoded_prompt_sha256": case.get("decoded_text_sha256"),
    }


def capture_inkling_decode_profile_request(
    *,
    url: str,
    output_path: Path,
    profile_root: Path,
    prompt_cases_path: Path,
    inkling_repo: Path,
    contract: InklingDecodeProfileContract,
) -> InklingDecodeProfileRequest:
    if contract.hlo_identity_status is not HloIdentityStatus.PENDING:
        raise ValueError("INKLING_PROFILE_CAPTURE_REQUIRES_PENDING_HLO_IDENTITIES")
    _require_new_output(output_path)
    repo = Path(__file__).resolve().parents[2]
    if inkling_repo.resolve() != Path(contract.inkling_source_root):
        raise ValueError(
            "INKLING_PROFILE_SERVER_SOURCE_ROOT_MISMATCH "
            f"expected={contract.inkling_source_root} observed={inkling_repo.resolve()}"
        )
    session_id = uuid.uuid4().hex
    profile_directory = profile_root / session_id
    source_manifest = _source_manifest(repo, contract)
    capture_runtime = _runtime_identity()
    preflight = (
        (capture_runtime, contract.runtime, "RUNTIME"),
        (_sha256(repo / "uv.lock"), contract.capture_uv_lock_sha256, "CAPTURE_LOCK"),
        (
            _text(["git", "rev-parse", "HEAD"], cwd=inkling_repo),
            contract.inkling_git_commit,
            "INKLING_COMMIT",
        ),
        (_sha256(inkling_repo / "uv.lock"), contract.inkling_uv_lock_sha256, "INKLING_LOCK"),
    )
    for observed, expected, label in preflight:
        if observed != expected:
            raise ValueError(
                f"INKLING_PROFILE_{label}_MISMATCH expected={expected} observed={observed}"
            )
    if _text(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError("INKLING_PROFILE_CAPTURE_SOURCE_DIRTY")
    if _text(["git", "status", "--porcelain"], cwd=inkling_repo):
        raise ValueError("INKLING_PROFILE_SERVER_SOURCE_DIRTY")
    _require_outside_repositories(output_path, (repo, inkling_repo))
    _require_outside_repositories(profile_directory, (repo, inkling_repo))
    _require_new_profile_directory(profile_directory)
    if _sha256(prompt_cases_path) != contract.prompts.corpus_sha256:
        raise ValueError("INKLING_PROFILE_PROMPT_CORPUS_MISMATCH")
    server_process = _server_process(url, inkling_repo, contract.server_command_fragments)
    server_info = _get_json(url, "get_server_info")
    for field, expected in contract.server.model_dump(mode="python").items():
        if server_info.get(field) != expected:
            raise ValueError(
                "INKLING_PROFILE_SERVER_CONFIGURATION_MISMATCH "
                f"field={field} expected={expected!r} observed={server_info.get(field)!r}"
            )
    cases = _load_prompt_cases(prompt_cases_path, contract.prompts.selected_case_ids)
    request_body = {
        "rid": [f"{session_id}:{index}" for index in range(contract.concurrency)],
        "input_ids": [case["input_ids"] for case in cases],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": contract.output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_routed_experts": False,
    }
    first_seen: set[int] = set()
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(contract.concurrency)]
    profile_started = False
    profile_stopped = False
    start_response: dict[str, Any] | None = None
    stop_response: dict[str, Any] | None = None
    profile_start_requested_at_ns: int | None = None
    profile_stop_completed_at_ns: int | None = None
    start_ns = time.perf_counter_ns()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw_line in response:
                line = raw_line.strip()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                item = json.loads(payload)
                index = item.pop("index")
                chunks[index].append(
                    {
                        "elapsed_ms": (time.perf_counter_ns() - start_ns) / 1e6,
                        "response": item,
                    }
                )
                first_seen.add(index)
                if not profile_started and len(first_seen) == contract.concurrency:
                    profile_start_requested_at_ns = time.time_ns()
                    start_response = _post_json(
                        url,
                        "start_profile",
                        {
                            "output_dir": str(profile_directory),
                            "host_tracer_level": contract.host_tracer_level,
                            "python_tracer_level": contract.python_tracer_level,
                        },
                    )
                    profile_started = True
                if (
                    profile_started
                    and not profile_stopped
                    and min(map(len, chunks))
                    >= contract.profile_stop_after_minimum_completion_tokens
                ):
                    stop_response = _post_json(url, "stop_profile")
                    profile_stop_completed_at_ns = time.time_ns()
                    profile_stopped = True
    finally:
        if profile_started and not profile_stopped:
            try:
                stop_response = _post_json(url, "stop_profile")
                profile_stop_completed_at_ns = time.time_ns()
            except OSError:
                pass
    if not profile_started:
        raise RuntimeError("INKLING_PROFILE_NEVER_STARTED")
    if (
        not profile_stopped
        or start_response is None
        or stop_response is None
        or profile_start_requested_at_ns is None
        or profile_stop_completed_at_ns is None
    ):
        raise RuntimeError("INKLING_PROFILE_NEVER_STOPPED")

    final_server_info = _get_json(url, "get_server_info")
    if final_server_info != server_info:
        raise RuntimeError("INKLING_PROFILE_SERVER_CONFIGURATION_CHANGED_DURING_CAPTURE")
    final_server_process = _server_process(url, inkling_repo, contract.server_command_fragments)
    if final_server_process != server_process:
        raise RuntimeError("INKLING_PROFILE_SERVER_PROCESS_CHANGED_DURING_CAPTURE")
    xplane = _wait_for_xplane(profile_directory)
    xplane_process_id, xplane_start_time, xplane_stop_time = _xplane_identity(xplane)
    xplane_runtime = _xplane_runtime(xplane)
    expected_xplane_runtime = {
        "jax": contract.runtime.jax,
        "jaxlib": contract.runtime.jaxlib,
        "tpu_runtime": contract.xplane_tpu_version,
        "device_type": contract.xplane_device_type,
        "process_id": str(xplane_process_id),
    }
    if xplane_runtime != expected_xplane_runtime:
        raise RuntimeError(
            "INKLING_PROFILE_XPLANE_RUNTIME_MISMATCH "
            f"expected={expected_xplane_runtime} observed={xplane_runtime}"
        )
    if len(_module_event_counts(xplane)) != contract.device_count:
        raise RuntimeError("INKLING_PROFILE_XPLANE_DEVICE_COUNT_MISMATCH")
    if not (
        profile_start_requested_at_ns <= xplane_start_time
        and xplane_stop_time <= profile_stop_completed_at_ns
    ):
        raise RuntimeError("INKLING_PROFILE_XPLANE_TIME_RANGE_MISMATCH")
    process = psutil.Process(server_process.pid)
    process_tree = {process.pid, *(child.pid for child in process.children(recursive=True))}
    if xplane_process_id not in process_tree:
        raise RuntimeError(
            "INKLING_PROFILE_XPLANE_SERVER_PROCESS_MISMATCH "
            f"xplane_process_id={xplane_process_id} server_process_tree={sorted(process_tree)}"
        )
    record = InklingDecodeProfileRequest.model_validate(
        {
            "request": {key: value for key, value in request_body.items() if key != "input_ids"}
            | {"prompt_cases": [_prompt_evidence(case) for case in cases]},
            "provenance": {
                "captured_at_utc": datetime.fromtimestamp(xplane_stop_time / 1e9, UTC).isoformat(),
                "command": sys.argv,
                "tpu_cake_git_commit": _text(["git", "rev-parse", "HEAD"], cwd=repo),
                "tpu_cake_git_status_porcelain": _text(["git", "status", "--porcelain"], cwd=repo),
                "tpu_cake_uv_lock_sha256": _sha256(repo / "uv.lock"),
                "capture_runtime": capture_runtime,
                "capture_source_manifest": source_manifest,
                "inkling_git_commit": _text(["git", "rev-parse", "HEAD"], cwd=inkling_repo),
                "inkling_git_status_porcelain": _text(
                    ["git", "status", "--porcelain"], cwd=inkling_repo
                ),
                "inkling_uv_lock_sha256": _sha256(inkling_repo / "uv.lock"),
                "hostname": _text(["hostname"], cwd=repo),
                "python": sys.version.split()[0],
                "prompt_cases_file_sha256": _sha256(prompt_cases_path),
                "profile_session_id": session_id,
                "profile_directory_name": profile_directory.name,
                "capture_xplane_sha256": _sha256(xplane),
                "xplane_process_id": xplane_process_id,
                "xplane_profile_start_time_ns": xplane_start_time,
                "xplane_profile_stop_time_ns": xplane_stop_time,
                "profile_start_requested_at_ns": profile_start_requested_at_ns,
                "profile_stop_completed_at_ns": profile_stop_completed_at_ns,
                "server_process": server_process,
                "server_info": final_server_info,
            },
            "profile_start_response": start_response,
            "profile_stop_response": stop_response,
            "profile_start_condition": "every request emitted at least one token",
            "profile_stop_after_minimum_completion_tokens": (
                contract.profile_stop_after_minimum_completion_tokens
            ),
            "host_tracer_level": contract.host_tracer_level,
            "python_tracer_level": contract.python_tracer_level,
            "elapsed_ms": (time.perf_counter_ns() - start_ns) / 1e6,
            "chunks_by_request": chunks,
        }
    )
    _write_json_new(output_path, record)
    return record


def _finding(code: str, message: str, *evidence: str) -> Finding:
    return Finding(
        code=code,
        severity=FindingSeverity.ERROR,
        message=message,
        evidence=evidence,
    )


def _matching_programs(capture: CaptureEvidence, prefix: str) -> tuple[ProgramEvidence, ...]:
    return tuple(
        program
        for program in capture.programs
        if program.name.startswith(f"{prefix}(") and program.name.endswith(")")
    )


def _validate_chunks(
    record: InklingDecodeProfileRequest,
    contract: InklingDecodeProfileContract,
) -> list[Finding]:
    findings: list[Finding] = []
    if len(record.chunks_by_request) != contract.concurrency:
        return [_finding("CHUNK_INVENTORY_MISMATCH", "chunk inventory does not match concurrency")]
    request_slots: list[int] = []
    recurrent_slots: list[int] = []
    for index, chunks in enumerate(record.chunks_by_request):
        if len(chunks) != contract.output_tokens:
            findings.append(
                _finding(
                    "OUTPUT_TOKEN_COUNT_MISMATCH",
                    "request does not contain the exact output-token sequence",
                    f"request_index={index}",
                    f"observed={len(chunks)}",
                )
            )
            continue
        expected_id = record.request.rid[index]
        expected_output: tuple[int, ...] = ()
        request_state_slots: set[int] = set()
        recurrent_state_slots: set[int] = set()
        for completion, chunk in enumerate(chunks, start=1):
            metadata = chunk.response.meta_info
            output_ids = chunk.response.output_ids
            if metadata.id != expected_id:
                findings.append(
                    _finding(
                        "CHUNK_REQUEST_ID_MISMATCH",
                        "streamed chunk is assigned to the wrong request",
                        f"request_index={index}",
                        f"completion={completion}",
                    )
                )
            if metadata.completion_tokens != completion or len(output_ids) != completion:
                findings.append(
                    _finding(
                        "CHUNK_COMPLETION_SEQUENCE_MISMATCH",
                        "streamed completion counters are not exact and contiguous",
                        f"request_index={index}",
                        f"completion={completion}",
                    )
                )
            if metadata.server_batch_size != contract.concurrency:
                findings.append(
                    _finding(
                        "SERVER_BATCH_SIZE_MISMATCH",
                        "streamed response was not produced by the full declared batch",
                        f"request_index={index}",
                        f"completion={completion}",
                        f"observed={metadata.server_batch_size}",
                    )
                )
            if output_ids[:-1] != expected_output:
                findings.append(
                    _finding(
                        "CHUNK_OUTPUT_PREFIX_MISMATCH",
                        "streamed output token IDs are not cumulative",
                        f"request_index={index}",
                        f"completion={completion}",
                    )
                )
            expected_output = output_ids
            request_state_slots.add(metadata.request_state_slot)
            recurrent_state_slots.add(metadata.recurrent_state_slot)
        if len(request_state_slots) != 1 or len(recurrent_state_slots) != 1:
            findings.append(
                _finding(
                    "CHUNK_SLOT_CHANGED",
                    "request or recurrent state slot changed during the response",
                    f"request_index={index}",
                )
            )
        else:
            request_slots.extend(request_state_slots)
            recurrent_slots.extend(recurrent_state_slots)
    if (
        len(request_slots) != contract.concurrency
        or len(set(request_slots)) != contract.concurrency
    ):
        findings.append(
            _finding(
                "REQUEST_STATE_SLOT_INVENTORY_MISMATCH",
                "requests do not occupy distinct request-state slots for the complete response",
            )
        )
    if (
        len(recurrent_slots) != contract.concurrency
        or len(set(recurrent_slots)) != contract.concurrency
    ):
        findings.append(
            _finding(
                "RECURRENT_STATE_SLOT_INVENTORY_MISMATCH",
                "requests do not occupy distinct recurrent-state slots for the complete response",
            )
        )
    return findings


def _validate_request(
    record: InklingDecodeProfileRequest,
    prompt_cases_path: Path,
    contract: InklingDecodeProfileContract,
) -> list[Finding]:
    findings: list[Finding] = []
    expected_cases = tuple(
        InklingDecodePromptEvidence.model_validate(_prompt_evidence(case))
        for case in _load_prompt_cases(prompt_cases_path, contract.prompts.selected_case_ids)
    )
    request = record.request
    provenance = record.provenance
    expected_sampling = {
        "temperature": 0,
        "max_new_tokens": contract.output_tokens,
        "ignore_eos": True,
    }
    checks = (
        (len(request.rid) == contract.concurrency, "REQUEST_CONCURRENCY_MISMATCH"),
        (len(set(request.rid)) == contract.concurrency, "REQUEST_IDS_NOT_UNIQUE"),
        (
            request.rid
            == tuple(
                f"{provenance.profile_session_id}:{index}" for index in range(contract.concurrency)
            ),
            "REQUEST_SESSION_ID_MISMATCH",
        ),
        (request.sampling_params == expected_sampling, "SAMPLING_PROTOCOL_MISMATCH"),
        (request.stream is True, "REQUEST_NOT_STREAMING"),
        (request.return_routed_experts is False, "ROUTED_EXPERT_OUTPUT_ENABLED"),
        (request.prompt_cases == expected_cases, "PROMPT_CASES_MISMATCH"),
        (
            record.profile_stop_after_minimum_completion_tokens
            == contract.profile_stop_after_minimum_completion_tokens,
            "PROFILE_STOP_COMPLETION_MISMATCH",
        ),
        (record.host_tracer_level == contract.host_tracer_level, "HOST_TRACER_MISMATCH"),
        (record.python_tracer_level == contract.python_tracer_level, "PYTHON_TRACER_MISMATCH"),
        (provenance.inkling_git_commit == contract.inkling_git_commit, "SOURCE_COMMIT_MISMATCH"),
        (provenance.inkling_git_status_porcelain == "", "SOURCE_NOT_CLEAN"),
        (
            provenance.inkling_uv_lock_sha256 == contract.inkling_uv_lock_sha256,
            "INKLING_LOCK_MISMATCH",
        ),
        (
            provenance.capture_source_manifest == contract.capture_source_manifest,
            "CAPTURE_SOURCE_MISMATCH",
        ),
        (provenance.tpu_cake_git_status_porcelain == "", "CAPTURE_SOURCE_NOT_CLEAN"),
        (
            contract.hlo_identity_status is HloIdentityStatus.PENDING
            or provenance.tpu_cake_git_commit == contract.capture_git_commit,
            "CAPTURE_SOURCE_COMMIT_MISMATCH",
        ),
        (
            provenance.tpu_cake_uv_lock_sha256 == contract.capture_uv_lock_sha256,
            "CAPTURE_LOCK_MISMATCH",
        ),
        (provenance.capture_runtime == contract.runtime, "CAPTURE_RUNTIME_MISMATCH"),
        (
            provenance.prompt_cases_file_sha256 == contract.prompts.corpus_sha256,
            "PROMPT_CORPUS_MISMATCH",
        ),
        (
            provenance.profile_directory_name == provenance.profile_session_id,
            "PROFILE_SESSION_DIRECTORY_MISMATCH",
        ),
        (
            Path(provenance.server_process.cwd) == Path(contract.inkling_source_root),
            "SERVER_PROCESS_CWD_MISMATCH",
        ),
        (_sha256(prompt_cases_path) == contract.prompts.corpus_sha256, "PROMPT_FILE_MISMATCH"),
        (
            record.profile_start_response.get("status_code") == 200,
            "PROFILE_START_FAILED",
        ),
        (record.profile_stop_response.get("status_code") == 200, "PROFILE_STOP_FAILED"),
    )
    findings.extend(
        _finding(code, code.replace("_", " ").lower()) for passed, code in checks if not passed
    )
    findings.extend(_validate_chunks(record, contract))
    command = " ".join(provenance.server_process.cmdline)
    missing_fragments = [
        fragment for fragment in contract.server_command_fragments if fragment not in command
    ]
    if missing_fragments:
        findings.append(
            _finding(
                "SERVER_PROCESS_COMMAND_MISMATCH",
                "server process command does not match the external contract",
                *missing_fragments,
            )
        )
    server_info = provenance.server_info
    for field, expected in contract.server.model_dump(mode="python").items():
        if server_info.get(field) != expected:
            findings.append(
                _finding(
                    "SERVER_CONFIGURATION_MISMATCH",
                    "server configuration does not match the external contract",
                    f"field={field}",
                    f"expected={expected!r}",
                    f"observed={server_info.get(field)!r}",
                )
            )
    return findings


def assess_inkling_decode_profile(
    *,
    capture: CaptureEvidence,
    module_events_per_tpu_core: dict[str, dict[str, int]],
    xplane_process_id: int,
    xplane_profile_start_time_ns: int,
    xplane_profile_stop_time_ns: int,
    xplane_runtime: dict[str, str],
    capture_root_name: str,
    request_path: Path,
    prompt_cases_path: Path,
    contract: InklingDecodeProfileContract,
) -> InklingDecodeProfileAssessment:
    record = InklingDecodeProfileRequest.model_validate_json(request_path.read_text())
    generic = assess_evidence(capture, contract.profile)
    findings = _validate_request(record, prompt_cases_path, contract)
    expected_xplane_runtime = {
        "jax": contract.runtime.jax,
        "jaxlib": contract.runtime.jaxlib,
        "tpu_runtime": contract.xplane_tpu_version,
        "device_type": contract.xplane_device_type,
        "process_id": str(xplane_process_id),
    }
    if xplane_runtime != expected_xplane_runtime:
        findings.append(
            _finding(
                "XPLANE_RUNTIME_MISMATCH",
                "raw XPlane runtime does not match the external contract",
            )
        )
    observed_device_types = {
        plane.device_type
        for plane in capture.planes
        if plane.name.startswith("/device:TPU:") and "SparseCore" not in plane.name
    }
    if observed_device_types != {contract.xplane_device_type}:
        findings.append(
            _finding(
                "XPLANE_DEVICE_TYPE_MISMATCH",
                "raw XPlane device type does not match the external contract",
            )
        )
    if record.provenance.capture_xplane_sha256 != capture.xplane.sha256:
        findings.append(
            _finding(
                "REQUEST_XPLANE_MISMATCH",
                "request evidence does not name the inspected raw XPlane",
            )
        )
    if record.provenance.xplane_process_id != xplane_process_id:
        findings.append(
            _finding(
                "XPLANE_PROCESS_ID_MISMATCH",
                "raw XPlane process does not match the captured server process evidence",
            )
        )
    if (
        record.provenance.xplane_profile_start_time_ns != xplane_profile_start_time_ns
        or record.provenance.xplane_profile_stop_time_ns != xplane_profile_stop_time_ns
        or record.provenance.captured_at_utc
        != datetime.fromtimestamp(xplane_profile_stop_time_ns / 1e9, UTC).isoformat()
        or not (
            record.provenance.profile_start_requested_at_ns <= xplane_profile_start_time_ns
            and xplane_profile_stop_time_ns <= record.provenance.profile_stop_completed_at_ns
        )
    ):
        findings.append(
            _finding(
                "XPLANE_TIME_RANGE_MISMATCH",
                "request timing does not match the raw XPlane profile interval",
            )
        )
    if capture_root_name != record.provenance.profile_session_id:
        findings.append(
            _finding(
                "PROFILE_SESSION_ROOT_MISMATCH",
                "capture root does not match the request profile session",
                f"expected={record.provenance.profile_session_id}",
                f"observed={capture_root_name}",
            )
        )
    required: list[ProgramEvidence] = []
    required_program_names: dict[str, str] = {}
    for expected in contract.programs:
        matching = _matching_programs(capture, expected.name_prefix)
        if len(matching) != 1:
            findings.append(
                _finding(
                    "TIMED_PROGRAM_INVENTORY_MISMATCH",
                    "required timed program must appear exactly once",
                    f"name_prefix={expected.name_prefix}",
                    f"observed={len(matching)}",
                )
            )
            continue
        program = matching[0]
        required.append(program)
        required_program_names[expected.name_prefix] = program.name
        if program.program_id not in capture.timed_program_ids or program.timed_self_us <= 0:
            findings.append(
                _finding(
                    "REQUIRED_PROGRAM_NOT_TIMED",
                    "required program has no positive XProf self time",
                    f"name_prefix={expected.name_prefix}",
                )
            )
        if program.hlo is None:
            findings.append(
                _finding(
                    "REQUIRED_PROGRAM_HLO_MISSING",
                    "required program has no captured HLO",
                    f"name_prefix={expected.name_prefix}",
                )
            )
        elif (
            contract.hlo_identity_status is HloIdentityStatus.PINNED
            and program.hlo.sha256 != expected.hlo_sha256
        ):
            findings.append(
                _finding(
                    "REQUIRED_PROGRAM_HLO_MISMATCH",
                    "required program HLO does not match the external contract",
                    f"name_prefix={expected.name_prefix}",
                    f"expected={expected.hlo_sha256}",
                    f"observed={program.hlo.sha256}",
                )
            )
    required_program_ids = {program.program_id for program in required}
    if required_program_ids != capture.timed_program_ids:
        findings.append(
            _finding(
                "TIMED_PROGRAM_SET_MISMATCH",
                "timed program inventory must exactly match the external contract",
                f"expected={sorted(required_program_ids)}",
                f"observed={sorted(capture.timed_program_ids)}",
            )
        )
    main = _matching_programs(capture, contract.main_program_prefix)
    if len(main) == 1:
        for marker in contract.profile.required_timed_hlo_markers:
            if main[0].marker_counts.get(marker, 0) <= 0:
                findings.append(
                    _finding(
                        "MAIN_PROGRAM_MARKER_MISSING",
                        "required marker is absent from the main decode program",
                        f"marker={marker}",
                    )
                )
    if contract.hlo_identity_status is HloIdentityStatus.PENDING:
        findings.append(
            _finding(
                "HLO_IDENTITIES_PENDING",
                "capture is descriptive until exact program HLO identities are pinned",
            )
        )
    expected_planes = {f"/device:TPU:{index}" for index in range(contract.server.tp_size)}
    if set(module_events_per_tpu_core) != expected_planes:
        findings.append(
            _finding(
                "MODULE_EVENT_TPU_CORE_INVENTORY_MISMATCH",
                "raw XPlane does not contain the exact TPU core inventory",
            )
        )
    for plane, event_counts in sorted(module_events_per_tpu_core.items()):
        matched_events: set[str] = set()
        for expected in contract.programs:
            names = [
                name
                for name in event_counts
                if name.startswith(f"{expected.name_prefix}(") and name.endswith(")")
            ]
            if len(names) != 1:
                findings.append(
                    _finding(
                        "MODULE_EVENT_PROGRAM_INVENTORY_MISMATCH",
                        "required raw module event must have one exact concrete name per TPU core",
                        f"plane={plane}",
                        f"name_prefix={expected.name_prefix}",
                        f"observed={len(names)}",
                    )
                )
                continue
            name = names[0]
            matched_events.add(name)
            if name != required_program_names.get(expected.name_prefix):
                findings.append(
                    _finding(
                        "MODULE_EVENT_PROGRAM_ID_MISMATCH",
                        "raw module event name does not match the captured HLO program identity",
                        f"plane={plane}",
                        f"expected={required_program_names.get(expected.name_prefix)}",
                        f"observed={name}",
                    )
                )
            expected_count = expected.module_events_per_tpu_core.get(plane)
            if (
                contract.hlo_identity_status is HloIdentityStatus.PINNED
                and event_counts[name] != expected_count
            ):
                findings.append(
                    _finding(
                        "MODULE_EVENT_COUNT_MISMATCH",
                        "raw module event count does not match the external contract",
                        f"plane={plane}",
                        f"name_prefix={expected.name_prefix}",
                        f"expected={expected_count}",
                        f"observed={event_counts[name]}",
                    )
                )
        if matched_events != set(event_counts):
            findings.append(
                _finding(
                    "MODULE_EVENT_SET_MISMATCH",
                    "raw module event inventory contains an uncontracted program",
                    f"plane={plane}",
                    f"unmatched={sorted(set(event_counts) - matched_events)}",
                )
            )
    return InklingDecodeProfileAssessment(
        contract_id=contract.contract_id,
        request_sha256=_sha256(request_path),
        prompt_corpus_sha256=_sha256(prompt_cases_path),
        capture=generic,
        required_programs=tuple(required),
        module_events_per_tpu_core=module_events_per_tpu_core,
        xplane_runtime=xplane_runtime,
        findings=tuple(findings),
    )


def inspect_inkling_decode_profile(
    *,
    capture_root: Path,
    request_path: Path,
    prompt_cases_path: Path,
    contract: InklingDecodeProfileContract,
) -> InklingDecodeProfileAssessment:
    xplanes = tuple(capture_root.resolve().rglob("*.xplane.pb"))
    if len(xplanes) != 1:
        raise ValueError(f"INKLING_PROFILE_XPLANE_INVENTORY_MISMATCH observed={len(xplanes)}")
    xplane = xplanes[0]
    process_id, profile_start_time, profile_stop_time = _xplane_identity(xplane)
    runtime = _xplane_runtime(xplane)
    return assess_inkling_decode_profile(
        capture=collect_capture(capture_root, contract.profile),
        module_events_per_tpu_core=_module_event_counts(xplane),
        xplane_process_id=process_id,
        xplane_profile_start_time_ns=profile_start_time,
        xplane_profile_stop_time_ns=profile_stop_time,
        xplane_runtime=runtime,
        capture_root_name=capture_root.resolve().name,
        request_path=request_path,
        prompt_cases_path=prompt_cases_path,
        contract=contract,
    )


def validate_inkling_decode_profile(
    *,
    capture_root: Path,
    request_path: Path,
    prompt_cases_path: Path,
    contract: InklingDecodeProfileContract,
) -> InklingDecodeProfileAssessment:
    if contract.hlo_identity_status is not HloIdentityStatus.PINNED:
        raise ValueError("INKLING_DECODE_PROFILE_HLO_IDENTITIES_PENDING")
    repo = Path(__file__).resolve().parents[2]
    if _text(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError("INKLING_DECODE_PROFILE_SOURCE_DIRTY")
    if _sha256(repo / "uv.lock") != contract.capture_uv_lock_sha256:
        raise ValueError("INKLING_DECODE_PROFILE_LOCK_MISMATCH")
    _source_manifest(repo, contract)
    assessment = inspect_inkling_decode_profile(
        capture_root=capture_root,
        request_path=request_path,
        prompt_cases_path=prompt_cases_path,
        contract=contract,
    )
    if not assessment.accepted:
        codes = sorted(
            {finding.code for finding in (*assessment.capture.findings, *assessment.findings)}
        )
        raise ValueError(f"INKLING_DECODE_PROFILE_REJECTED findings={codes}")
    return assessment


def write_inkling_decode_profile_assessment(
    path: Path,
    assessment: InklingDecodeProfileAssessment,
) -> None:
    _require_new_output(path)
    _write_json_new(path, assessment)
