from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.artifacts import file_sha256
from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256
from tpu_cake.inkling_decode_profile import (
    InklingDecodeProfileContract,
    _declared_server_configuration,
    _get_json,
    _server_process,
)

ROUTE_CORPUS_SCHEMA = "inkling-gmm-route-corpus-v1"
ROUTE_CAPTURE_SCHEMA = "inkling-gmm-route-capture-v1"


class InklingGmmRouteCorpusContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["inkling-gmm-route-corpus-v1"] = ROUTE_CORPUS_SCHEMA
    name: str = Field(min_length=1)
    profile_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    inkling_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    inkling_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inkling_source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_server_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_server_command_fragments: tuple[str, ...] = Field(min_length=1)
    required_server_environment: dict[str, str]
    prompt_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    concurrency: int = Field(gt=0)
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=1)
    selected_completion_steps: tuple[int, ...] = Field(min_length=1)
    num_layers: int = Field(gt=0)
    first_dense_layers: int = Field(ge=0)
    num_experts_per_token: int = Field(gt=0)
    num_routed_experts: int = Field(gt=0)
    expected_server_batch_size: int = Field(gt=0)
    route_row_count_rule: Literal["prompt_tokens+completion_tokens-1"] = (
        "prompt_tokens+completion_tokens-1"
    )

    @computed_field
    @property
    def contract_id(self) -> str:
        return model_identity_sha256(self, exclude={"contract_id"})

    @model_validator(mode="after")
    def workload_is_complete(self) -> InklingGmmRouteCorpusContract:
        if self.concurrency != self.expected_server_batch_size:
            raise ValueError("route corpus must preserve the complete server batch")
        if self.first_dense_layers >= self.num_layers:
            raise ValueError("route corpus must contain at least one MoE layer")
        if self.num_experts_per_token > self.num_routed_experts:
            raise ValueError("top-k cannot exceed the routed expert count")
        expected = tuple(range(self.selected_completion_steps[0], self.output_tokens + 1))
        if self.selected_completion_steps != expected or expected[0] <= 1:
            raise ValueError("selected decode steps must be contiguous and exclude prefill")
        paths = tuple(item.path for item in self.inkling_source_manifest)
        if len(paths) != len(set(paths)):
            raise ValueError("Inkling source manifest paths must be unique")
        return self


class RouteChunkEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    completion_tokens: int = Field(gt=0)
    prompt_tokens: int = Field(gt=0)
    server_batch_size: int = Field(gt=0)
    request_state_slot: int = Field(ge=0)
    recurrent_state_slot: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    finish_reason_type: str | None
    routed_experts_present: bool


class RouteRequestEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_index: int = Field(ge=0)
    request_id: str = Field(min_length=1)
    prompt_case_id: str = Field(min_length=1)
    input_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: tuple[RouteChunkEvidence, ...] = Field(min_length=1)
    final_routed_experts_base64: str = Field(min_length=1)


class InklingGmmRouteCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["inkling-gmm-route-capture-v1"] = ROUTE_CAPTURE_SCHEMA
    capture_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tpu_cake_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tpu_cake_git_status_porcelain: Literal[""] = ""
    server_process_id: int = Field(gt=0)
    server_command: tuple[str, ...] = Field(min_length=1)
    server_environment: dict[str, str]
    server_configuration: dict[str, Any]
    cohort_assumption: Literal["single-localhost-batch-request-with-idle-boundaries"] = (
        "single-localhost-batch-request-with-idle-boundaries"
    )
    server_idle_before: Literal[True] = True
    server_idle_after: Literal[True] = True
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_sse_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_sse_bytes: int = Field(gt=0)
    requests: tuple[RouteRequestEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identity_is_valid(self) -> InklingGmmRouteCapture:
        if tuple(request.request_index for request in self.requests) != tuple(
            range(len(self.requests))
        ):
            raise ValueError("route capture request order mismatch")
        for values, label in (
            (tuple(request.request_id for request in self.requests), "request IDs"),
            (tuple(request.prompt_case_id for request in self.requests), "prompt case IDs"),
            (tuple(request.input_ids_sha256 for request in self.requests), "input hashes"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"route capture {label} must be unique")
        if (
            self.capture_id != "0" * 64
            and model_identity_sha256(self, exclude={"capture_id"}) != self.capture_id
        ):
            raise ValueError("route capture identity mismatch")
        return self


class RouteGroupSizes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    completion_step: int = Field(gt=1)
    layer_index: int = Field(ge=0)
    group_sizes: tuple[int, ...]

    @model_validator(mode="after")
    def counts_are_valid(self) -> RouteGroupSizes:
        if any(value < 0 for value in self.group_sizes):
            raise ValueError("route group sizes cannot be negative")
        return self


class InklingGmmRouteCorpusReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["inkling-gmm-route-corpus-v1"] = ROUTE_CORPUS_SCHEMA
    report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    concurrency: int = Field(gt=0)
    selected_completion_steps: tuple[int, ...] = Field(min_length=1)
    first_moe_layer: int = Field(ge=0)
    num_layers: int = Field(gt=0)
    num_experts_per_token: int = Field(gt=0)
    num_routed_experts: int = Field(gt=0)
    request_state_slots: tuple[int, ...]
    recurrent_state_slots: tuple[int, ...]
    group_sizes: tuple[RouteGroupSizes, ...] = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_scope: Literal["operational-assumption-no-emitted-step-id"] = (
        "operational-assumption-no-emitted-step-id"
    )

    @model_validator(mode="after")
    def identity_is_valid(self) -> InklingGmmRouteCorpusReport:
        expected_inventory = tuple(
            (completion, layer)
            for completion in self.selected_completion_steps
            for layer in range(self.first_moe_layer, self.num_layers)
        )
        observed_inventory = tuple(
            (group.completion_step, group.layer_index) for group in self.group_sizes
        )
        expected_routes = self.concurrency * self.num_experts_per_token
        if observed_inventory != expected_inventory:
            raise ValueError("route corpus group inventory mismatch")
        if any(
            len(group.group_sizes) != self.num_routed_experts
            or sum(group.group_sizes) != expected_routes
            for group in self.group_sizes
        ):
            raise ValueError("route corpus group sizes mismatch")
        if (
            len(self.request_state_slots) != self.concurrency
            or len(set(self.request_state_slots)) != self.concurrency
            or len(self.recurrent_state_slots) != self.concurrency
            or len(set(self.recurrent_state_slots)) != self.concurrency
        ):
            raise ValueError("route corpus slot inventory mismatch")
        if (
            _json_sha256([group.model_dump(mode="json") for group in self.group_sizes])
            != self.corpus_sha256
        ):
            raise ValueError("route corpus hash mismatch")
        if (
            self.report_id != "0" * 64
            and model_identity_sha256(self, exclude={"report_id"}) != self.report_id
        ):
            raise ValueError("route corpus report identity mismatch")
        return self


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_prompt_cases(path: Path, selected_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise TypeError("INKLING_ROUTE_PROMPT_CORPUS_INVALID")
    cases = {case.get("id"): case for case in payload if isinstance(case, dict)}
    if len(cases) != len(payload):
        raise ValueError("INKLING_ROUTE_PROMPT_IDS_NOT_UNIQUE")
    try:
        selected = tuple(cases[case_id] for case_id in selected_ids)
    except KeyError as error:
        raise ValueError(f"INKLING_ROUTE_PROMPT_MISSING id={error.args[0]}") from error
    if any(
        not isinstance(case.get("input_ids"), list)
        or not case["input_ids"]
        or any(type(token_id) is not int for token_id in case["input_ids"])
        for case in selected
    ):
        raise ValueError("INKLING_ROUTE_PROMPT_INPUT_IDS_INVALID")
    return selected


def _decode_routes(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value.encode(), validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("INKLING_ROUTE_BASE64_INVALID") from error
    if base64.b64encode(decoded).decode() != value:
        raise ValueError("INKLING_ROUTE_BASE64_NONCANONICAL")
    return decoded


def _source_sha256() -> str:
    return file_sha256(Path(__file__))


def _text(command: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _process_environment(process_id: int, names: set[str]) -> dict[str, str]:
    entries = Path(f"/proc/{process_id}/environ").read_bytes().split(b"\0")
    environment = {}
    for entry in entries:
        if not entry or b"=" not in entry:
            continue
        name, value = entry.split(b"=", 1)
        decoded_name = name.decode()
        if decoded_name in names:
            environment[decoded_name] = value.decode()
    return environment


def _server_is_idle(url: str) -> bool:
    states = _get_json(url, "get_server_info")["internal_states"]
    return bool(states) and all(
        int(state["running_batch_size"]) == 0
        and int(state["waiting_queue_size"]) == 0
        and int(state["req_to_token_pool_used"]) == 0
        for state in states
    )


def _wait_for_server_idle(url: str, *, timeout_seconds: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _server_is_idle(url):
            return
        time.sleep(0.25)
    raise TimeoutError("INKLING_ROUTE_SERVER_IDLE_TIMEOUT")


def capture_routes(
    *,
    url: str,
    output_path: Path,
    prompt_cases_path: Path,
    model_config_path: Path,
    inkling_repo: Path,
    contract: InklingGmmRouteCorpusContract,
    profile_contract: InklingDecodeProfileContract,
) -> InklingGmmRouteCapture:
    raw_sse_path = output_path.with_name(f"{output_path.name}.sse")
    for artifact in (output_path, raw_sse_path):
        if artifact.exists():
            raise FileExistsError(f"INKLING_ROUTE_CAPTURE_OUTPUT_EXISTS path={artifact}")
    tpu_cake_repo = Path(__file__).resolve().parents[2]
    tpu_cake_status = _text(["git", "status", "--porcelain"], cwd=tpu_cake_repo)
    if tpu_cake_status:
        raise ValueError("INKLING_ROUTE_CAPTURE_SOURCE_DIRTY")
    if file_sha256(tpu_cake_repo / "uv.lock") != contract.capture_uv_lock_sha256:
        raise ValueError("INKLING_ROUTE_CAPTURE_LOCK_MISMATCH")
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("INKLING_ROUTE_CAPTURE_MUST_USE_LOCALHOST")
    if profile_contract.contract_id != contract.profile_contract_id:
        raise ValueError("INKLING_ROUTE_PROFILE_CONTRACT_MISMATCH")
    profile_values = (
        profile_contract.inkling_git_commit,
        profile_contract.inkling_uv_lock_sha256,
        profile_contract.server.revision,
        profile_contract.prompts.corpus_sha256,
        profile_contract.concurrency,
    )
    contract_values = (
        contract.inkling_git_commit,
        contract.inkling_uv_lock_sha256,
        contract.model_revision,
        contract.prompt_corpus_sha256,
        contract.concurrency,
    )
    if profile_values != contract_values:
        raise ValueError("INKLING_ROUTE_PROFILE_DECLARATION_MISMATCH")
    if (
        _json_sha256(profile_contract.server.model_dump(mode="json"))
        != contract.profile_server_configuration_sha256
    ):
        raise ValueError("INKLING_ROUTE_PROFILE_SERVER_HASH_MISMATCH")
    if _text(["git", "rev-parse", "HEAD"], cwd=inkling_repo) != contract.inkling_git_commit:
        raise ValueError("INKLING_ROUTE_SERVER_REVISION_MISMATCH")
    if _text(["git", "status", "--porcelain"], cwd=inkling_repo):
        raise ValueError("INKLING_ROUTE_SERVER_SOURCE_DIRTY")
    if file_sha256(inkling_repo / "uv.lock") != contract.inkling_uv_lock_sha256:
        raise ValueError("INKLING_ROUTE_SERVER_LOCK_MISMATCH")
    for source in contract.inkling_source_manifest:
        if file_sha256(inkling_repo / source.path) != source.sha256:
            raise ValueError(f"INKLING_ROUTE_SERVER_SOURCE_MISMATCH path={source.path}")
    if file_sha256(prompt_cases_path) != contract.prompt_corpus_sha256:
        raise ValueError("INKLING_ROUTE_PROMPT_CORPUS_MISMATCH")
    if file_sha256(model_config_path) != contract.model_config_sha256:
        raise ValueError("INKLING_ROUTE_MODEL_CONFIG_MISMATCH")
    if _source_sha256() != contract.producer_source_sha256:
        raise ValueError("INKLING_ROUTE_PRODUCER_SOURCE_MISMATCH")
    model_config = json.loads(model_config_path.read_text())["text_config"]
    model_shape = {
        "num_hidden_layers": contract.num_layers,
        "dense_mlp_idx": contract.first_dense_layers,
        "num_experts_per_tok": contract.num_experts_per_token,
        "n_routed_experts": contract.num_routed_experts,
    }
    if {name: model_config.get(name) for name in model_shape} != model_shape:
        raise ValueError("INKLING_ROUTE_MODEL_SHAPE_MISMATCH")
    server_process = _server_process(url, inkling_repo, profile_contract.server_command_fragments)
    if "--enable-return-routed-experts" not in server_process.cmdline:
        raise ValueError("INKLING_ROUTE_SERVER_CAPTURE_DISABLED")
    server_environment = _process_environment(
        server_process.pid, set(contract.required_server_environment)
    )
    if server_environment != contract.required_server_environment:
        raise ValueError("INKLING_ROUTE_SERVER_ENVIRONMENT_MISMATCH")
    command = " ".join(server_process.cmdline)
    if any(fragment not in command for fragment in contract.required_server_command_fragments):
        raise ValueError("INKLING_ROUTE_SERVER_COMMAND_MISMATCH")
    server_configuration = _declared_server_configuration(
        _get_json(url, "get_server_info"), profile_contract.server
    )
    if server_configuration != profile_contract.server.model_dump(mode="python"):
        raise ValueError("INKLING_ROUTE_SERVER_CONFIGURATION_MISMATCH")
    if not _server_is_idle(url):
        raise ValueError("INKLING_ROUTE_SERVER_NOT_IDLE_BEFORE_CAPTURE")
    cases = _load_prompt_cases(prompt_cases_path, profile_contract.prompts.selected_case_ids)
    if len(cases) != contract.concurrency or any(
        len(case["input_ids"]) != contract.prompt_tokens for case in cases
    ):
        raise ValueError("INKLING_ROUTE_PROMPT_SHAPE_MISMATCH")
    prompt_manifest = [
        {"id": case["id"], "input_ids_sha256": _json_sha256(case["input_ids"])} for case in cases
    ]
    if _json_sha256(prompt_manifest) != contract.prompt_case_manifest_sha256:
        raise ValueError("INKLING_ROUTE_PROMPT_MANIFEST_MISMATCH")

    session_id = uuid.uuid4().hex
    request_body = {
        "rid": [f"{session_id}:{index}" for index in range(contract.concurrency)],
        "input_ids": [case["input_ids"] for case in cases],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": contract.output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_routed_experts": True,
    }
    final_routes: list[bytes | None] = [None] * contract.concurrency
    chunks: list[list[RouteChunkEvidence]] = [[] for _ in range(contract.concurrency)]
    request = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with (
        raw_sse_path.open("xb") as raw_sse,
        urllib.request.urlopen(request, timeout=3600) as response,
    ):
        for raw_line in response:
            raw_sse.write(raw_line)
            line = raw_line.strip()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            item = json.loads(payload)
            index = item.pop("index")
            if type(index) is not int or not 0 <= index < contract.concurrency:
                raise ValueError(f"INKLING_ROUTE_RESPONSE_INDEX_INVALID index={index!r}")
            meta = item["meta_info"]
            if meta["id"] != request_body["rid"][index]:
                raise ValueError(f"INKLING_ROUTE_RESPONSE_ID_MISMATCH index={index}")
            encoded_routes = meta["routed_experts"]
            if encoded_routes is not None:
                if final_routes[index] is not None:
                    raise ValueError(f"INKLING_ROUTE_MULTIPLE_PAYLOADS request={index}")
                final_routes[index] = _decode_routes(encoded_routes)
            finish_reason = meta["finish_reason"]
            chunks[index].append(
                RouteChunkEvidence(
                    completion_tokens=meta["completion_tokens"],
                    prompt_tokens=meta["prompt_tokens"],
                    server_batch_size=meta["server_batch_size"],
                    request_state_slot=meta["request_state_slot"],
                    recurrent_state_slot=meta["recurrent_state_slot"],
                    cached_tokens=meta["cached_tokens"],
                    finish_reason_type=(
                        finish_reason.get("type") if isinstance(finish_reason, dict) else None
                    ),
                    routed_experts_present=encoded_routes is not None,
                )
            )
    requests = tuple(
        RouteRequestEvidence(
            request_index=index,
            request_id=request_body["rid"][index],
            prompt_case_id=cases[index]["id"],
            input_ids_sha256=_json_sha256(cases[index]["input_ids"]),
            chunks=tuple(chunks[index]),
            final_routed_experts_base64=base64.b64encode(final_routes[index] or b"").decode(),
        )
        for index in range(contract.concurrency)
    )
    _wait_for_server_idle(url)
    final_server_process = _server_process(
        url, inkling_repo, profile_contract.server_command_fragments
    )
    if final_server_process != server_process:
        raise RuntimeError("INKLING_ROUTE_SERVER_PROCESS_CHANGED")
    if (
        _declared_server_configuration(_get_json(url, "get_server_info"), profile_contract.server)
        != server_configuration
    ):
        raise RuntimeError("INKLING_ROUTE_SERVER_CONFIGURATION_CHANGED")
    provisional = InklingGmmRouteCapture(
        capture_id="0" * 64,
        contract_id=contract.contract_id,
        profile_contract_id=profile_contract.contract_id,
        prompt_corpus_sha256=contract.prompt_corpus_sha256,
        producer_source_sha256=_source_sha256(),
        capture_uv_lock_sha256=file_sha256(tpu_cake_repo / "uv.lock"),
        tpu_cake_git_commit=_text(["git", "rev-parse", "HEAD"], cwd=tpu_cake_repo),
        tpu_cake_git_status_porcelain="",
        server_process_id=server_process.pid,
        server_command=server_process.cmdline,
        server_environment=server_environment,
        server_configuration=server_configuration,
        server_idle_before=True,
        server_idle_after=True,
        request_sha256=_json_sha256(request_body),
        raw_sse_sha256=file_sha256(raw_sse_path),
        raw_sse_bytes=raw_sse_path.stat().st_size,
        requests=requests,
    )
    capture = provisional.model_copy(
        update={"capture_id": model_identity_sha256(provisional, exclude={"capture_id"})}
    )
    InklingGmmRouteCapture.model_validate(capture)
    output_path.write_text(
        json.dumps(capture.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return capture


def _validated_route_arrays(
    capture: InklingGmmRouteCapture,
    contract: InklingGmmRouteCorpusContract,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[int, ...]]:
    if capture.contract_id != contract.contract_id:
        raise ValueError("INKLING_ROUTE_CAPTURE_CONTRACT_MISMATCH")
    if (
        capture.profile_contract_id != contract.profile_contract_id
        or capture.prompt_corpus_sha256 != contract.prompt_corpus_sha256
        or capture.producer_source_sha256 != contract.producer_source_sha256
        or capture.capture_uv_lock_sha256 != contract.capture_uv_lock_sha256
        or capture.server_environment != contract.required_server_environment
        or _json_sha256(capture.server_configuration)
        != contract.profile_server_configuration_sha256
        or any(
            fragment not in " ".join(capture.server_command)
            for fragment in contract.required_server_command_fragments
        )
    ):
        raise ValueError("INKLING_ROUTE_CAPTURE_PROVENANCE_MISMATCH")
    if len(capture.requests) != contract.concurrency:
        raise ValueError("INKLING_ROUTE_REQUEST_COUNT_MISMATCH")
    request_ids = tuple(request.request_id for request in capture.requests)
    case_ids = tuple(request.prompt_case_id for request in capture.requests)
    input_hashes = tuple(request.input_ids_sha256 for request in capture.requests)
    if (
        len(set(request_ids)) != contract.concurrency
        or len(set(case_ids)) != contract.concurrency
        or len(set(input_hashes)) != contract.concurrency
        or _json_sha256(
            [
                {"id": request.prompt_case_id, "input_ids_sha256": request.input_ids_sha256}
                for request in capture.requests
            ]
        )
        != contract.prompt_case_manifest_sha256
    ):
        raise ValueError("INKLING_ROUTE_REQUEST_MANIFEST_MISMATCH")
    arrays: list[np.ndarray] = []
    request_slots: list[int] = []
    recurrent_slots: list[int] = []
    bytes_per_row = contract.num_layers * contract.num_experts_per_token * 4
    for expected_index, request in enumerate(capture.requests):
        if request.request_index != expected_index:
            raise ValueError("INKLING_ROUTE_REQUEST_ORDER_MISMATCH")
        expected_completions = tuple(range(1, contract.output_tokens + 1))
        if tuple(chunk.completion_tokens for chunk in request.chunks) != expected_completions:
            raise ValueError(f"INKLING_ROUTE_COMPLETION_SEQUENCE_MISMATCH request={expected_index}")
        final = _decode_routes(request.final_routed_experts_base64)
        expected_rows = contract.prompt_tokens + contract.output_tokens - 1
        expected_bytes = expected_rows * bytes_per_row
        if len(final) != expected_bytes:
            raise ValueError("INKLING_ROUTE_FINAL_PAYLOAD_SIZE_MISMATCH")
        route_array = np.frombuffer(final, dtype="<i4").reshape(
            expected_rows, contract.num_layers, contract.num_experts_per_token
        )
        for chunk in request.chunks:
            if chunk.prompt_tokens != contract.prompt_tokens:
                raise ValueError("INKLING_ROUTE_PROMPT_TOKEN_MISMATCH")
            if chunk.server_batch_size != contract.expected_server_batch_size:
                raise ValueError("INKLING_ROUTE_SERVER_BATCH_MISMATCH")
            if chunk.cached_tokens != 0:
                raise ValueError("INKLING_ROUTE_CACHE_REUSE_DETECTED")
        if (
            any(chunk.routed_experts_present for chunk in request.chunks[:-1])
            or not request.chunks[-1].routed_experts_present
        ):
            raise ValueError("INKLING_ROUTE_PAYLOAD_NOT_FINAL_ONLY")
        if (
            any(chunk.finish_reason_type is not None for chunk in request.chunks[:-1])
            or request.chunks[-1].finish_reason_type != "length"
        ):
            raise ValueError("INKLING_ROUTE_FINISH_REASON_MISMATCH")
        slots = {chunk.request_state_slot for chunk in request.chunks}
        recurrent = {chunk.recurrent_state_slot for chunk in request.chunks}
        if len(slots) != 1 or len(recurrent) != 1:
            raise ValueError("INKLING_ROUTE_REQUEST_SLOT_CHANGED")
        request_slots.append(slots.pop())
        recurrent_slots.append(recurrent.pop())
        arrays.append(route_array)
    if (
        len(set(request_slots)) != contract.concurrency
        or len(set(recurrent_slots)) != contract.concurrency
    ):
        raise ValueError("INKLING_ROUTE_REQUEST_SLOTS_NOT_UNIQUE")
    return tuple(arrays), tuple(sorted(request_slots)), tuple(sorted(recurrent_slots))


def derive_route_corpus(
    *,
    capture: InklingGmmRouteCapture,
    capture_sha256: str,
    raw_sse_sha256: str,
    raw_sse_bytes: int,
    contract: InklingGmmRouteCorpusContract,
) -> InklingGmmRouteCorpusReport:
    if raw_sse_sha256 != capture.raw_sse_sha256 or raw_sse_bytes != capture.raw_sse_bytes:
        raise ValueError("INKLING_ROUTE_RAW_SSE_MISMATCH")
    arrays, request_slots, recurrent_slots = _validated_route_arrays(capture, contract)
    groups: list[RouteGroupSizes] = []
    for completion in contract.selected_completion_steps:
        row_index = contract.prompt_tokens + completion - 2
        step_routes = np.stack([request[row_index] for request in arrays], axis=0)
        dense = step_routes[:, : contract.first_dense_layers, :]
        if dense.size and not np.all(dense == -1):
            raise ValueError("INKLING_ROUTE_DENSE_LAYER_SENTINEL_MISMATCH")
        moe = step_routes[:, contract.first_dense_layers :, :]
        if np.any(moe < 0) or np.any(moe >= contract.num_routed_experts):
            raise ValueError("INKLING_ROUTE_EXPERT_ID_OUT_OF_RANGE")
        if np.any(np.diff(np.sort(moe, axis=-1), axis=-1) == 0):
            raise ValueError("INKLING_ROUTE_DUPLICATE_TOPK_EXPERT")
        for layer_offset in range(moe.shape[1]):
            counts = np.bincount(
                moe[:, layer_offset, :].reshape(-1), minlength=contract.num_routed_experts
            )
            groups.append(
                RouteGroupSizes(
                    completion_step=completion,
                    layer_index=contract.first_dense_layers + layer_offset,
                    group_sizes=tuple(int(value) for value in counts),
                )
            )
    expected_tokens = contract.concurrency * contract.num_experts_per_token
    if any(
        len(group.group_sizes) != contract.num_routed_experts
        or sum(group.group_sizes) != expected_tokens
        for group in groups
    ):
        raise ValueError("INKLING_ROUTE_GROUP_SIZE_TOTAL_MISMATCH")
    corpus_sha256 = _json_sha256([group.model_dump(mode="json") for group in groups])
    provisional = InklingGmmRouteCorpusReport(
        report_id="0" * 64,
        contract_id=contract.contract_id,
        capture_id=capture.capture_id,
        capture_sha256=capture_sha256,
        producer_source_sha256=capture.producer_source_sha256,
        verifier_source_sha256=contract.verifier_source_sha256,
        concurrency=contract.concurrency,
        selected_completion_steps=contract.selected_completion_steps,
        first_moe_layer=contract.first_dense_layers,
        num_layers=contract.num_layers,
        num_experts_per_token=contract.num_experts_per_token,
        num_routed_experts=contract.num_routed_experts,
        request_state_slots=request_slots,
        recurrent_state_slots=recurrent_slots,
        group_sizes=tuple(groups),
        corpus_sha256=corpus_sha256,
    )
    report = provisional.model_copy(
        update={"report_id": model_identity_sha256(provisional, exclude={"report_id"})}
    )
    InklingGmmRouteCorpusReport.model_validate(report)
    return report


def write_report(path: Path, report: InklingGmmRouteCorpusReport) -> None:
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--url", default="http://127.0.0.1:30000")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--prompt-cases", type=Path, required=True)
    capture.add_argument("--model-config", type=Path, required=True)
    capture.add_argument("--inkling-repo", type=Path, required=True)
    capture.add_argument("--contract", type=Path, required=True)
    capture.add_argument("--profile-contract", type=Path, required=True)
    derive = commands.add_parser("derive")
    derive.add_argument("capture", type=Path)
    derive.add_argument("--raw-sse", type=Path, required=True)
    derive.add_argument("--contract", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("report", type=Path)
    verify.add_argument("--capture", type=Path, required=True)
    verify.add_argument("--raw-sse", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    contract = InklingGmmRouteCorpusContract.model_validate_json(args.contract.read_text())
    if args.command == "capture":
        profile = InklingDecodeProfileContract.model_validate_json(
            args.profile_contract.read_text()
        )
        capture_routes(
            url=args.url,
            output_path=args.output,
            prompt_cases_path=args.prompt_cases,
            model_config_path=args.model_config,
            inkling_repo=args.inkling_repo,
            contract=contract,
            profile_contract=profile,
        )
        return
    capture = InklingGmmRouteCapture.model_validate_json(args.capture.read_text())
    report = derive_route_corpus(
        capture=capture,
        capture_sha256=file_sha256(args.capture),
        raw_sse_sha256=file_sha256(args.raw_sse),
        raw_sse_bytes=args.raw_sse.stat().st_size,
        contract=contract,
    )
    if args.command == "derive":
        write_report(args.output, report)
        print(f"INKLING_GMM_ROUTE_CORPUS_WRITTEN report_id={report.report_id}")
        return
    expected = InklingGmmRouteCorpusReport.model_validate_json(args.report.read_text())
    if capture.producer_source_sha256 != _source_sha256():
        raise ValueError("INKLING_ROUTE_REPLAY_SOURCE_MISMATCH")
    if report != expected:
        raise ValueError("INKLING_ROUTE_CORPUS_REPLAY_MISMATCH")
    print(f"INKLING_GMM_ROUTE_CORPUS_REPLAYED report_id={report.report_id}")


if __name__ == "__main__":
    main()
