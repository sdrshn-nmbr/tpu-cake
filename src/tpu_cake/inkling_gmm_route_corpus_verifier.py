from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from tpu_cake.artifacts import file_sha256
from tpu_cake.identity import json_sha256

CONTRACT_KEYS = {
    "schema_version",
    "name",
    "profile_contract_id",
    "inkling_git_commit",
    "inkling_uv_lock_sha256",
    "inkling_source_manifest",
    "model_revision",
    "model_config_sha256",
    "model_weight_manifest_sha256",
    "capture_source_manifest",
    "producer_source_sha256",
    "verifier_source_sha256",
    "capture_uv_lock_sha256",
    "profile_server_configuration_sha256",
    "prompt_case_manifest_sha256",
    "required_server_command_fragments",
    "required_server_environment",
    "forbidden_server_environment_names",
    "prompt_corpus_sha256",
    "concurrency",
    "prompt_tokens",
    "output_tokens",
    "selected_completion_steps",
    "num_layers",
    "first_dense_layers",
    "num_experts_per_token",
    "num_routed_experts",
    "expected_server_batch_size",
    "route_row_count_rule",
}
CAPTURE_KEYS = {
    "schema_version",
    "capture_id",
    "contract_id",
    "profile_contract_id",
    "prompt_corpus_sha256",
    "producer_source_sha256",
    "capture_uv_lock_sha256",
    "observed_tpu_cake_git_commit",
    "tpu_cake_git_status_porcelain",
    "server_process_id",
    "server_command",
    "server_environment",
    "server_configuration",
    "cohort_assumption",
    "server_idle_before",
    "server_idle_after",
    "server_launch_receipt_id",
    "server_launch_receipt_sha256",
    "server_launch_nonce",
    "request_sha256",
    "request_bytes",
    "model_weight_manifest_sha256",
    "model_weight_manifest",
    "raw_sse_sha256",
    "raw_sse_bytes",
    "requests",
}
REQUEST_EVIDENCE_KEYS = {
    "request_index",
    "request_id",
    "prompt_case_id",
    "input_ids_sha256",
    "chunks",
    "final_routed_experts_base64",
}
CHUNK_KEYS = {
    "completion_tokens",
    "prompt_tokens",
    "server_batch_size",
    "request_state_slot",
    "recurrent_state_slot",
    "cached_tokens",
    "finish_reason_type",
    "routed_experts_present",
}
REPORT_KEYS = {
    "schema_version",
    "report_id",
    "contract_id",
    "capture_id",
    "capture_sha256",
    "producer_source_sha256",
    "verifier_source_sha256",
    "server_launch_receipt_id",
    "server_launch_receipt_sha256",
    "request_sha256",
    "model_weight_manifest_sha256",
    "concurrency",
    "selected_completion_steps",
    "first_moe_layer",
    "num_layers",
    "num_experts_per_token",
    "num_routed_experts",
    "request_state_slots",
    "recurrent_state_slots",
    "group_sizes",
    "corpus_sha256",
    "cohort_scope",
}
LAUNCH_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "contract_id",
    "launch_nonce",
    "observed_tpu_cake_git_commit",
    "capture_source_manifest_sha256",
    "inkling_git_commit",
    "inkling_uv_lock_sha256",
    "inkling_source_manifest_sha256",
    "model_revision",
    "model_config_sha256",
    "model_weight_manifest_sha256",
    "model_weight_manifest",
    "server_command",
    "server_environment",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"INKLING_ROUTE_VERIFY_OBJECT_REQUIRED path={path}")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], error: str) -> None:
    if set(value) != expected:
        raise ValueError(error)


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _validate_contract(contract: dict[str, Any]) -> None:
    _exact_keys(contract, CONTRACT_KEYS, "INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
    hashes = (
        "profile_contract_id",
        "inkling_uv_lock_sha256",
        "model_config_sha256",
        "model_weight_manifest_sha256",
        "producer_source_sha256",
        "verifier_source_sha256",
        "capture_uv_lock_sha256",
        "profile_server_configuration_sha256",
        "prompt_case_manifest_sha256",
        "prompt_corpus_sha256",
    )
    integers = (
        "concurrency",
        "prompt_tokens",
        "output_tokens",
        "num_layers",
        "num_experts_per_token",
        "num_routed_experts",
        "expected_server_batch_size",
    )
    sources = contract["inkling_source_manifest"]
    capture_sources = contract["capture_source_manifest"]
    fragments = contract["required_server_command_fragments"]
    environment = contract["required_server_environment"]
    forbidden_environment = contract["forbidden_server_environment_names"]
    selected = contract["selected_completion_steps"]
    if (
        contract["schema_version"] != "inkling-gmm-route-corpus-v1"
        or not isinstance(contract["name"], str)
        or not contract["name"]
        or any(not _is_hex(contract[name], 64) for name in hashes)
        or not _is_hex(contract["inkling_git_commit"], 40)
        or not _is_hex(contract["model_revision"], 40)
        or any(not _positive_int(contract[name]) for name in integers)
        or type(contract["first_dense_layers"]) is not int
        or contract["first_dense_layers"] < 0
        or not isinstance(sources, list)
        or not sources
        or not isinstance(capture_sources, list)
        or not capture_sources
        or not isinstance(fragments, list)
        or not fragments
        or any(not isinstance(fragment, str) or not fragment for fragment in fragments)
        or not isinstance(environment, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items())
        or not isinstance(forbidden_environment, list)
        or not forbidden_environment
        or any(not isinstance(name, str) or not name for name in forbidden_environment)
        or len(forbidden_environment) != len(set(forbidden_environment))
        or bool(set(forbidden_environment) & set(environment))
        or not isinstance(selected, list)
        or not selected
        or any(type(step) is not int for step in selected)
        or selected != list(range(selected[0], contract["output_tokens"] + 1))
        or selected[0] <= 1
        or contract["concurrency"] != contract["expected_server_batch_size"]
        or contract["first_dense_layers"] >= contract["num_layers"]
        or contract["num_experts_per_token"] > contract["num_routed_experts"]
        or contract["route_row_count_rule"] != "prompt_tokens+completion_tokens-1"
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
    paths = []
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
        _exact_keys(source, {"path", "sha256"}, "INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
        if (
            not isinstance(source["path"], str)
            or not source["path"]
            or not _is_hex(source["sha256"], 64)
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
        paths.append(source["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
    capture_paths = []
    for source in capture_sources:
        if not isinstance(source, dict):
            raise TypeError("INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
        _exact_keys(source, {"path", "sha256"}, "INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
        if (
            not isinstance(source["path"], str)
            or not source["path"]
            or not _is_hex(source["sha256"], 64)
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")
        capture_paths.append(source["path"])
    if len(capture_paths) != len(set(capture_paths)):
        raise ValueError("INKLING_ROUTE_VERIFY_CONTRACT_SCHEMA")


def _validate_capture(capture: dict[str, Any], contract: dict[str, Any]) -> None:
    _exact_keys(capture, CAPTURE_KEYS, "INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
    hash_fields = (
        "capture_id",
        "contract_id",
        "profile_contract_id",
        "prompt_corpus_sha256",
        "producer_source_sha256",
        "capture_uv_lock_sha256",
        "request_sha256",
        "model_weight_manifest_sha256",
        "raw_sse_sha256",
    )
    if (
        capture["schema_version"] != "inkling-gmm-route-capture-v1"
        or any(not _is_hex(capture[name], 64) for name in hash_fields)
        or not _is_hex(capture["observed_tpu_cake_git_commit"], 40)
        or capture["tpu_cake_git_status_porcelain"] != ""
        or capture["cohort_assumption"] != "single-localhost-batch-request-with-idle-boundaries"
        or capture["server_idle_before"] is not True
        or capture["server_idle_after"] is not True
        or not _is_hex(capture["server_launch_receipt_id"], 64)
        or not _is_hex(capture["server_launch_receipt_sha256"], 64)
        or not _is_hex(capture["server_launch_nonce"], 32)
        or not _positive_int(capture["server_process_id"])
        or not _positive_int(capture["request_bytes"])
        or not _positive_int(capture["raw_sse_bytes"])
        or not isinstance(capture["server_command"], list)
        or not capture["server_command"]
        or any(not isinstance(value, str) for value in capture["server_command"])
        or not isinstance(capture["server_environment"], dict)
        or not isinstance(capture["server_configuration"], dict)
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
    manifest = capture["model_weight_manifest"]
    if not isinstance(manifest, list) or len(manifest) < 2:
        raise ValueError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
    paths = []
    for item in manifest:
        if not isinstance(item, dict):
            raise TypeError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
        _exact_keys(item, {"path", "sha256", "bytes"}, "INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
        if (
            not isinstance(item["path"], str)
            or not item["path"]
            or not _is_hex(item["sha256"], 64)
            or not _positive_int(item["bytes"])
        ):
            raise TypeError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
        paths.append(item["path"])
    if (
        len(paths) != len(set(paths))
        or json_sha256(manifest) != capture["model_weight_manifest_sha256"]
        or capture["model_weight_manifest_sha256"] != contract["model_weight_manifest_sha256"]
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
    requests = capture["requests"]
    if not isinstance(requests, list) or len(requests) != contract["concurrency"]:
        raise ValueError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
    for request in requests:
        if not isinstance(request, dict):
            raise TypeError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
        _exact_keys(request, REQUEST_EVIDENCE_KEYS, "INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
        if (
            type(request["request_index"]) is not int
            or request["request_index"] < 0
            or not isinstance(request["request_id"], str)
            or not request["request_id"]
            or not isinstance(request["prompt_case_id"], str)
            or not request["prompt_case_id"]
            or not _is_hex(request["input_ids_sha256"], 64)
            or not isinstance(request["final_routed_experts_base64"], str)
            or not request["final_routed_experts_base64"]
            or not isinstance(request["chunks"], list)
            or len(request["chunks"]) != contract["output_tokens"]
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
        for chunk in request["chunks"]:
            if not isinstance(chunk, dict):
                raise TypeError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
            _exact_keys(chunk, CHUNK_KEYS, "INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")
            if (
                any(
                    type(chunk[name]) is not int
                    for name in (
                        "completion_tokens",
                        "prompt_tokens",
                        "server_batch_size",
                        "request_state_slot",
                        "recurrent_state_slot",
                        "cached_tokens",
                    )
                )
                or chunk["completion_tokens"] <= 0
                or chunk["prompt_tokens"] <= 0
                or chunk["server_batch_size"] <= 0
                or chunk["request_state_slot"] < 0
                or chunk["recurrent_state_slot"] < 0
                or chunk["cached_tokens"] < 0
                or chunk["finish_reason_type"] not in {None, "length"}
                or type(chunk["routed_experts_present"]) is not bool
            ):
                raise ValueError("INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA")


def _validate_report(report: dict[str, Any]) -> None:
    _exact_keys(report, REPORT_KEYS, "INKLING_ROUTE_VERIFY_REPORT_SCHEMA")
    hashes = (
        "report_id",
        "contract_id",
        "capture_id",
        "capture_sha256",
        "producer_source_sha256",
        "verifier_source_sha256",
        "server_launch_receipt_id",
        "server_launch_receipt_sha256",
        "request_sha256",
        "model_weight_manifest_sha256",
        "corpus_sha256",
    )
    integers = (
        "concurrency",
        "num_layers",
        "num_experts_per_token",
        "num_routed_experts",
    )
    if (
        report["schema_version"] != "inkling-gmm-route-corpus-v1"
        or any(not _is_hex(report[name], 64) for name in hashes)
        or any(not _positive_int(report[name]) for name in integers)
        or type(report["first_moe_layer"]) is not int
        or report["first_moe_layer"] < 0
        or report["cohort_scope"] != "operational-assumption-no-emitted-step-id"
        or any(
            not isinstance(report[name], list)
            or any(type(value) is not int for value in report[name])
            for name in (
                "selected_completion_steps",
                "request_state_slots",
                "recurrent_state_slots",
            )
        )
        or not isinstance(report["group_sizes"], list)
        or not report["group_sizes"]
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_REPORT_SCHEMA")
    for group in report["group_sizes"]:
        if not isinstance(group, dict):
            raise TypeError("INKLING_ROUTE_VERIFY_REPORT_SCHEMA")
        _exact_keys(
            group,
            {"completion_step", "layer_index", "group_sizes"},
            "INKLING_ROUTE_VERIFY_REPORT_SCHEMA",
        )
        if (
            type(group["completion_step"]) is not int
            or type(group["layer_index"]) is not int
            or not isinstance(group["group_sizes"], list)
            or any(type(value) is not int or value < 0 for value in group["group_sizes"])
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_REPORT_SCHEMA")


def _validate_launch_receipt(receipt: dict[str, Any], contract: dict[str, Any]) -> None:
    _exact_keys(receipt, LAUNCH_RECEIPT_KEYS, "INKLING_ROUTE_VERIFY_LAUNCH_SCHEMA")
    hash_fields = (
        "receipt_id",
        "contract_id",
        "capture_source_manifest_sha256",
        "inkling_uv_lock_sha256",
        "inkling_source_manifest_sha256",
        "model_config_sha256",
        "model_weight_manifest_sha256",
    )
    manifest = receipt["model_weight_manifest"]
    if (
        receipt["schema_version"] != "inkling-gmm-route-server-launch-v1"
        or any(not _is_hex(receipt[name], 64) for name in hash_fields)
        or not _is_hex(receipt["launch_nonce"], 32)
        or not _is_hex(receipt["observed_tpu_cake_git_commit"], 40)
        or not _is_hex(receipt["inkling_git_commit"], 40)
        or not _is_hex(receipt["model_revision"], 40)
        or not isinstance(receipt["server_command"], list)
        or not receipt["server_command"]
        or any(not isinstance(value, str) for value in receipt["server_command"])
        or not isinstance(receipt["server_environment"], dict)
        or not isinstance(manifest, list)
        or len(manifest) < 2
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_LAUNCH_SCHEMA")
    for item in manifest:
        if not isinstance(item, dict):
            raise TypeError("INKLING_ROUTE_VERIFY_LAUNCH_SCHEMA")
        _exact_keys(item, {"path", "sha256", "bytes"}, "INKLING_ROUTE_VERIFY_LAUNCH_SCHEMA")
        if (
            not isinstance(item["path"], str)
            or not item["path"]
            or not _is_hex(item["sha256"], 64)
            or not _positive_int(item["bytes"])
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_LAUNCH_SCHEMA")
    if (
        json_sha256(manifest) != receipt["model_weight_manifest_sha256"]
        or receipt["model_weight_manifest_sha256"] != contract["model_weight_manifest_sha256"]
        or receipt["contract_id"] != json_sha256(contract)
        or receipt["capture_source_manifest_sha256"]
        != json_sha256(contract["capture_source_manifest"])
        or receipt["inkling_source_manifest_sha256"]
        != json_sha256(contract["inkling_source_manifest"])
        or receipt["inkling_git_commit"] != contract["inkling_git_commit"]
        or receipt["inkling_uv_lock_sha256"] != contract["inkling_uv_lock_sha256"]
        or receipt["model_revision"] != contract["model_revision"]
        or receipt["model_config_sha256"] != contract["model_config_sha256"]
        or receipt["server_environment"] != contract["required_server_environment"]
        or any(
            fragment not in " ".join(receipt["server_command"])
            for fragment in contract["required_server_command_fragments"]
        )
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_LAUNCH_CONTENT_MISMATCH")
    _verify_self_hash(receipt, "receipt_id", "INKLING_ROUTE_VERIFY_LAUNCH_ID_MISMATCH")


def _module_command(command: list[str]) -> list[str]:
    try:
        module_index = command.index("-m")
    except ValueError as error:
        raise ValueError("INKLING_ROUTE_VERIFY_MODULE_COMMAND_MISSING") from error
    return command[module_index:]


def _decode(value: object) -> bytes:
    if not isinstance(value, str):
        raise TypeError("INKLING_ROUTE_VERIFY_BASE64_REQUIRED")
    try:
        decoded = base64.b64decode(value.encode(), validate=True)
    except ValueError as error:
        raise ValueError("INKLING_ROUTE_VERIFY_BASE64_INVALID") from error
    if base64.b64encode(decoded).decode() != value:
        raise ValueError("INKLING_ROUTE_VERIFY_BASE64_NONCANONICAL")
    return decoded


def _parse_sse(path: Path, concurrency: int) -> tuple[tuple[dict[str, Any], ...], ...]:
    events: list[list[dict[str, Any]]] = [[] for _ in range(concurrency)]
    done = False
    for raw_line in path.read_bytes().splitlines():
        if not raw_line:
            continue
        if not raw_line.startswith(b"data: "):
            raise ValueError("INKLING_ROUTE_VERIFY_SSE_LINE_INVALID")
        payload = raw_line[6:]
        if payload == b"[DONE]":
            if done:
                raise ValueError("INKLING_ROUTE_VERIFY_SSE_DONE_DUPLICATE")
            done = True
            continue
        if done:
            raise ValueError("INKLING_ROUTE_VERIFY_SSE_DATA_AFTER_DONE")
        item = json.loads(payload)
        index = item.pop("index")
        if type(index) is not int or not 0 <= index < concurrency:
            raise ValueError(f"INKLING_ROUTE_VERIFY_INDEX_INVALID index={index!r}")
        events[index].append(item)
    if not done or any(not request_events for request_events in events):
        raise ValueError("INKLING_ROUTE_VERIFY_SSE_INVENTORY_MISMATCH")
    return tuple(tuple(request_events) for request_events in events)


def _verify_self_hash(payload: dict[str, Any], field: str, error: str) -> None:
    identity = dict(payload)
    observed = identity.pop(field)
    if json_sha256(identity) != observed:
        raise ValueError(error)


def _chunk_evidence(meta: dict[str, Any]) -> dict[str, Any]:
    finish = meta["finish_reason"]
    return {
        "completion_tokens": meta["completion_tokens"],
        "prompt_tokens": meta["prompt_tokens"],
        "server_batch_size": meta["server_batch_size"],
        "request_state_slot": meta["request_state_slot"],
        "recurrent_state_slot": meta["recurrent_state_slot"],
        "cached_tokens": meta["cached_tokens"],
        "finish_reason_type": finish.get("type") if isinstance(finish, dict) else None,
        "routed_experts_present": meta["routed_experts"] is not None,
    }


def _verify_request(
    path: Path,
    capture: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    raw = path.read_bytes()
    if file_sha256(path) != capture["request_sha256"] or len(raw) != capture["request_bytes"]:
        raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_ARTIFACT_MISMATCH")
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_INVALID") from error
    if not isinstance(request, dict):
        raise TypeError("INKLING_ROUTE_VERIFY_REQUEST_INVALID")
    _exact_keys(
        request,
        {"rid", "input_ids", "sampling_params", "stream", "return_routed_experts"},
        "INKLING_ROUTE_VERIFY_REQUEST_INVALID",
    )
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if raw != canonical:
        raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_NONCANONICAL")
    sampling = request["sampling_params"]
    if not isinstance(sampling, dict):
        raise TypeError("INKLING_ROUTE_VERIFY_REQUEST_INVALID")
    _exact_keys(
        sampling,
        {"temperature", "max_new_tokens", "ignore_eos"},
        "INKLING_ROUTE_VERIFY_REQUEST_INVALID",
    )
    request_ids = request["rid"]
    input_ids = request["input_ids"]
    if (
        request["stream"] is not True
        or request["return_routed_experts"] is not True
        or sampling
        != {
            "temperature": 0,
            "max_new_tokens": contract["output_tokens"],
            "ignore_eos": True,
        }
        or not isinstance(request_ids, list)
        or not isinstance(input_ids, list)
        or len(request_ids) != contract["concurrency"]
        or len(input_ids) != contract["concurrency"]
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_INVALID")
    captured_requests = capture["requests"]
    if request_ids != [request["request_id"] for request in captured_requests]:
        raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_ID_MISMATCH")
    session_ids = set()
    for index, (request_id, tokens, evidence) in enumerate(
        zip(request_ids, input_ids, captured_requests, strict=True)
    ):
        if not isinstance(request_id, str) or not re.fullmatch(
            rf"[0-9a-f]{{32}}:{index}", request_id
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_ID_MISMATCH")
        session_ids.add(request_id.split(":", 1)[0])
        if (
            not isinstance(tokens, list)
            or len(tokens) != contract["prompt_tokens"]
            or any(type(token) is not int or token < 0 for token in tokens)
            or json_sha256(tokens) != evidence["input_ids_sha256"]
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_INPUT_MISMATCH")
    if len(session_ids) != 1:
        raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_ID_MISMATCH")


def verify_independently(
    *,
    raw_sse_path: Path,
    request_path: Path,
    launch_receipt_path: Path,
    capture_path: Path,
    report_path: Path,
    contract_path: Path,
) -> str:
    contract = _load(contract_path)
    capture = _load(capture_path)
    report = _load(report_path)
    launch_receipt = _load(launch_receipt_path)
    _validate_contract(contract)
    _validate_capture(capture, contract)
    _validate_report(report)
    _validate_launch_receipt(launch_receipt, contract)
    repository = Path(__file__).resolve().parents[2]
    if any(
        file_sha256(repository / source["path"]) != source["sha256"]
        for source in contract["capture_source_manifest"]
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_CAPTURE_SOURCE_MISMATCH")
    if contract["verifier_source_sha256"] != file_sha256(Path(__file__)) or contract[
        "producer_source_sha256"
    ] != file_sha256(Path(__file__).with_name("inkling_gmm_route_corpus.py")):
        raise ValueError("INKLING_ROUTE_VERIFY_SOURCE_MISMATCH")
    _verify_self_hash(capture, "capture_id", "INKLING_ROUTE_VERIFY_CAPTURE_ID_MISMATCH")
    _verify_self_hash(report, "report_id", "INKLING_ROUTE_VERIFY_REPORT_ID_MISMATCH")
    contract_id = json_sha256(contract)
    if capture["contract_id"] != contract_id or report["contract_id"] != contract_id:
        raise ValueError("INKLING_ROUTE_VERIFY_CONTRACT_MISMATCH")
    if (
        capture["raw_sse_sha256"] != file_sha256(raw_sse_path)
        or capture["raw_sse_bytes"] != raw_sse_path.stat().st_size
        or report["capture_sha256"] != file_sha256(capture_path)
        or report["capture_id"] != capture["capture_id"]
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_ARTIFACT_MISMATCH")
    if (
        capture["server_launch_receipt_sha256"] != file_sha256(launch_receipt_path)
        or capture["server_launch_receipt_id"] != launch_receipt["receipt_id"]
        or capture["server_launch_nonce"] != launch_receipt["launch_nonce"]
        or capture["observed_tpu_cake_git_commit"] != launch_receipt["observed_tpu_cake_git_commit"]
        or _module_command(capture["server_command"])
        != _module_command(launch_receipt["server_command"])
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_LAUNCH_ARTIFACT_MISMATCH")
    _verify_request(request_path, capture, contract)
    if (
        capture["profile_contract_id"] != contract["profile_contract_id"]
        or capture["prompt_corpus_sha256"] != contract["prompt_corpus_sha256"]
        or capture["producer_source_sha256"] != contract["producer_source_sha256"]
        or capture["capture_uv_lock_sha256"] != contract["capture_uv_lock_sha256"]
        or capture["server_environment"] != contract["required_server_environment"]
        or json_sha256(capture["server_configuration"])
        != contract["profile_server_configuration_sha256"]
        or any(
            fragment not in " ".join(capture["server_command"])
            for fragment in contract["required_server_command_fragments"]
        )
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_PROVENANCE_MISMATCH")

    concurrency = contract["concurrency"]
    prompt_tokens = contract["prompt_tokens"]
    output_tokens = contract["output_tokens"]
    num_layers = contract["num_layers"]
    first_moe_layer = contract["first_dense_layers"]
    topk = contract["num_experts_per_token"]
    num_experts = contract["num_routed_experts"]
    expected_completions = list(range(1, output_tokens + 1))
    events = _parse_sse(raw_sse_path, concurrency)
    requests = capture["requests"]
    if len(requests) != concurrency:
        raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_COUNT_MISMATCH")
    if (
        json_sha256(
            [
                {"id": request["prompt_case_id"], "input_ids_sha256": request["input_ids_sha256"]}
                for request in requests
            ]
        )
        != contract["prompt_case_manifest_sha256"]
    ):
        raise ValueError("INKLING_ROUTE_VERIFY_PROMPT_MANIFEST_MISMATCH")

    arrays = []
    request_slots = []
    recurrent_slots = []
    seen_request_ids = set()
    for index, (request, request_events) in enumerate(zip(requests, events, strict=True)):
        if request["request_index"] != index or request["request_id"] in seen_request_ids:
            raise ValueError("INKLING_ROUTE_VERIFY_REQUEST_IDENTITY_MISMATCH")
        seen_request_ids.add(request["request_id"])
        metas = [item["meta_info"] for item in request_events]
        if any(meta["id"] != request["request_id"] for meta in metas):
            raise ValueError("INKLING_ROUTE_VERIFY_RESPONSE_ID_MISMATCH")
        if [_chunk_evidence(meta) for meta in metas] != request["chunks"]:
            raise ValueError("INKLING_ROUTE_VERIFY_CHUNK_EVIDENCE_MISMATCH")
        if [meta["completion_tokens"] for meta in metas] != expected_completions:
            raise ValueError("INKLING_ROUTE_VERIFY_COMPLETION_MISMATCH")
        final_finish = metas[-1]["finish_reason"]
        if (
            any(meta["finish_reason"] is not None for meta in metas[:-1])
            or not isinstance(final_finish, dict)
            or final_finish.get("type") != "length"
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_FINISH_REASON_MISMATCH")
        if any(
            meta["prompt_tokens"] != prompt_tokens
            or (meta["completion_tokens"] > 1 and meta["server_batch_size"] != concurrency)
            or meta["cached_tokens"] != 0
            for meta in metas
        ):
            raise ValueError("INKLING_ROUTE_VERIFY_COHORT_MISMATCH")
        if any(meta["routed_experts"] is not None for meta in metas[:-1]):
            raise ValueError("INKLING_ROUTE_VERIFY_EARLY_PAYLOAD")
        encoded = metas[-1]["routed_experts"]
        if encoded != request["final_routed_experts_base64"]:
            raise ValueError("INKLING_ROUTE_VERIFY_FINAL_PAYLOAD_MISMATCH")
        raw_routes = _decode(encoded)
        rows = prompt_tokens + output_tokens - 1
        if len(raw_routes) != rows * num_layers * topk * 4:
            raise ValueError("INKLING_ROUTE_VERIFY_PAYLOAD_SIZE_MISMATCH")
        arrays.append(np.frombuffer(raw_routes, dtype="<i4").reshape(rows, num_layers, topk))
        slots = {meta["request_state_slot"] for meta in metas}
        recurrent = {meta["recurrent_state_slot"] for meta in metas}
        if len(slots) != 1 or len(recurrent) != 1:
            raise ValueError("INKLING_ROUTE_VERIFY_SLOT_CHANGED")
        request_slots.append(slots.pop())
        recurrent_slots.append(recurrent.pop())
    if len(set(request_slots)) != concurrency or len(set(recurrent_slots)) != concurrency:
        raise ValueError("INKLING_ROUTE_VERIFY_SLOT_INVENTORY_MISMATCH")

    groups = []
    for completion in contract["selected_completion_steps"]:
        row = prompt_tokens + completion - 2
        step = np.stack([routes[row] for routes in arrays])
        if not np.all(step[:, :first_moe_layer] == -1):
            raise ValueError("INKLING_ROUTE_VERIFY_DENSE_SENTINEL_MISMATCH")
        moe = step[:, first_moe_layer:]
        if np.any(moe < 0) or np.any(moe >= num_experts):
            raise ValueError("INKLING_ROUTE_VERIFY_EXPERT_RANGE_MISMATCH")
        if np.any(np.diff(np.sort(moe, axis=-1), axis=-1) == 0):
            raise ValueError("INKLING_ROUTE_VERIFY_DUPLICATE_TOPK")
        for layer_offset in range(moe.shape[1]):
            counts = np.bincount(moe[:, layer_offset].reshape(-1), minlength=num_experts)
            groups.append(
                {
                    "completion_step": completion,
                    "layer_index": first_moe_layer + layer_offset,
                    "group_sizes": [int(value) for value in counts],
                }
            )
    expected_report_fields = {
        "producer_source_sha256": contract["producer_source_sha256"],
        "verifier_source_sha256": contract["verifier_source_sha256"],
        "server_launch_receipt_id": capture["server_launch_receipt_id"],
        "server_launch_receipt_sha256": capture["server_launch_receipt_sha256"],
        "request_sha256": capture["request_sha256"],
        "model_weight_manifest_sha256": contract["model_weight_manifest_sha256"],
        "concurrency": concurrency,
        "selected_completion_steps": contract["selected_completion_steps"],
        "first_moe_layer": first_moe_layer,
        "num_layers": num_layers,
        "num_experts_per_token": topk,
        "num_routed_experts": num_experts,
        "request_state_slots": sorted(request_slots),
        "recurrent_state_slots": sorted(recurrent_slots),
        "group_sizes": groups,
        "corpus_sha256": json_sha256(groups),
        "cohort_scope": "operational-assumption-no-emitted-step-id",
    }
    if any(report.get(name) != value for name, value in expected_report_fields.items()):
        raise ValueError("INKLING_ROUTE_VERIFY_REPORT_CONTENT_MISMATCH")
    return report["report_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--raw-sse", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    report_id = verify_independently(
        raw_sse_path=args.raw_sse,
        request_path=args.request,
        launch_receipt_path=args.launch_receipt,
        capture_path=args.capture,
        report_path=args.report,
        contract_path=args.contract,
    )
    print(f"INKLING_GMM_ROUTE_CORPUS_INDEPENDENTLY_REPLAYED report_id={report_id}")


if __name__ == "__main__":
    main()
