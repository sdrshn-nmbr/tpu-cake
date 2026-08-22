from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from tpu_cake.artifacts import file_sha256
from tpu_cake.identity import json_sha256


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"INKLING_ROUTE_VERIFY_OBJECT_REQUIRED path={path}")
    return value


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
    done = 0
    for raw_line in path.read_bytes().splitlines():
        if not raw_line.startswith(b"data: "):
            continue
        payload = raw_line[6:]
        if payload == b"[DONE]":
            done += 1
            continue
        item = json.loads(payload)
        index = item.pop("index")
        if type(index) is not int or not 0 <= index < concurrency:
            raise ValueError(f"INKLING_ROUTE_VERIFY_INDEX_INVALID index={index!r}")
        events[index].append(item)
    if done != 1 or any(not request_events for request_events in events):
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


def verify_independently(
    *,
    raw_sse_path: Path,
    capture_path: Path,
    report_path: Path,
    contract_path: Path,
) -> str:
    contract = _load(contract_path)
    capture = _load(capture_path)
    report = _load(report_path)
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
            or meta["server_batch_size"] != concurrency
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
    parser.add_argument("--raw-sse", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    report_id = verify_independently(
        raw_sse_path=args.raw_sse,
        capture_path=args.capture,
        report_path=args.report,
        contract_path=args.contract,
    )
    print(f"INKLING_GMM_ROUTE_CORPUS_INDEPENDENTLY_REPLAYED report_id={report_id}")


if __name__ == "__main__":
    main()
