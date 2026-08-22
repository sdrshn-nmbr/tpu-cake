import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tpu_cake import inkling_gmm_route_corpus as route_corpus_module
from tpu_cake import inkling_gmm_route_corpus_verifier as route_verifier
from tpu_cake.artifacts import file_sha256
from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256
from tpu_cake.inkling_decode_profile import InklingDecodeProfileContract
from tpu_cake.inkling_gmm_route_corpus import (
    InklingGmmRouteCapture,
    InklingGmmRouteCorpusContract,
    RouteChunkEvidence,
    RouteRequestEvidence,
    _json_sha256,
    _parser,
    derive_route_corpus,
)


def _contract() -> InklingGmmRouteCorpusContract:
    return InklingGmmRouteCorpusContract(
        name="test",
        profile_contract_id="a" * 64,
        inkling_git_commit="b" * 40,
        inkling_uv_lock_sha256="c" * 64,
        inkling_source_manifest=(SourceFileContract(path="a.py", sha256="d" * 64),),
        model_revision="e" * 40,
        model_config_sha256="f" * 64,
        producer_source_sha256="9" * 64,
        verifier_source_sha256="a" * 64,
        capture_uv_lock_sha256="c" * 64,
        profile_server_configuration_sha256=_json_sha256({"model": "test"}),
        prompt_case_manifest_sha256=_json_sha256(
            [{"id": f"case-{index}", "input_ids_sha256": str(index) * 64} for index in range(3)]
        ),
        required_server_command_fragments=("server", "--enable-return-routed-experts"),
        required_server_environment={"HF_HOME": "/tmp/huggingface"},
        prompt_corpus_sha256="1" * 64,
        concurrency=3,
        prompt_tokens=3,
        output_tokens=3,
        selected_completion_steps=(2, 3),
        num_layers=4,
        first_dense_layers=1,
        num_experts_per_token=2,
        num_routed_experts=8,
        expected_server_batch_size=3,
    )


def _capture(contract: InklingGmmRouteCorpusContract) -> InklingGmmRouteCapture:
    requests = []
    for request_index in range(contract.concurrency):
        rows = contract.prompt_tokens + contract.output_tokens - 1
        routes = np.full(
            (rows, contract.num_layers, contract.num_experts_per_token),
            -1,
            dtype="<i4",
        )
        for row in range(rows):
            for layer in range(contract.first_dense_layers, contract.num_layers):
                first = (request_index + row + layer) % contract.num_routed_experts
                routes[row, layer] = (first, (first + 1) % contract.num_routed_experts)
        final = routes.tobytes()
        chunks = []
        for completion in range(1, contract.output_tokens + 1):
            chunks.append(
                RouteChunkEvidence(
                    completion_tokens=completion,
                    prompt_tokens=contract.prompt_tokens,
                    server_batch_size=contract.concurrency,
                    request_state_slot=request_index,
                    recurrent_state_slot=request_index + 10,
                    cached_tokens=0,
                    finish_reason_type=("length" if completion == contract.output_tokens else None),
                    routed_experts_present=completion == contract.output_tokens,
                )
            )
        requests.append(
            RouteRequestEvidence(
                request_index=request_index,
                request_id=f"run:{request_index}",
                prompt_case_id=f"case-{request_index}",
                input_ids_sha256=str(request_index) * 64,
                chunks=tuple(chunks),
                final_routed_experts_base64=base64.b64encode(final).decode(),
            )
        )
    provisional = InklingGmmRouteCapture(
        capture_id="0" * 64,
        contract_id=contract.contract_id,
        profile_contract_id=contract.profile_contract_id,
        prompt_corpus_sha256=contract.prompt_corpus_sha256,
        producer_source_sha256=contract.producer_source_sha256,
        capture_uv_lock_sha256=contract.capture_uv_lock_sha256,
        tpu_cake_git_commit="4" * 40,
        server_process_id=1,
        server_command=("server", "--enable-return-routed-experts"),
        server_environment=contract.required_server_environment,
        server_configuration={"model": "test"},
        request_sha256="8" * 64,
        raw_sse_sha256="7" * 64,
        raw_sse_bytes=1,
        requests=tuple(requests),
    )
    return provisional.model_copy(
        update={"capture_id": model_identity_sha256(provisional, exclude={"capture_id"})}
    )


def _repair_capture(payload: dict[str, object]) -> InklingGmmRouteCapture:
    payload["capture_id"] = model_identity_sha256(
        InklingGmmRouteCapture.model_validate({**payload, "capture_id": "0" * 64}),
        exclude={"capture_id"},
    )
    return InklingGmmRouteCapture.model_validate(payload)


def _raw_sse(capture: InklingGmmRouteCapture) -> bytes:
    lines = []
    for chunk_index in range(len(capture.requests[0].chunks)):
        for request in capture.requests:
            chunk = request.chunks[chunk_index]
            item = {
                "index": request.request_index,
                "meta_info": {
                    "id": request.request_id,
                    "completion_tokens": chunk.completion_tokens,
                    "prompt_tokens": chunk.prompt_tokens,
                    "server_batch_size": chunk.server_batch_size,
                    "request_state_slot": chunk.request_state_slot,
                    "recurrent_state_slot": chunk.recurrent_state_slot,
                    "cached_tokens": chunk.cached_tokens,
                    "finish_reason": (
                        {"type": chunk.finish_reason_type}
                        if chunk.finish_reason_type is not None
                        else None
                    ),
                    "routed_experts": (
                        request.final_routed_experts_base64
                        if chunk.routed_experts_present
                        else None
                    ),
                },
            }
            lines.append(b"data: " + json.dumps(item, separators=(",", ":")).encode() + b"\n")
    lines.append(b"data: [DONE]\n")
    return b"".join(lines)


def test_route_corpus_reconstructs_each_full_batch_decode_layer() -> None:
    contract = _contract()
    report = derive_route_corpus(
        capture=_capture(contract),
        capture_sha256="7" * 64,
        raw_sse_sha256="7" * 64,
        raw_sse_bytes=1,
        contract=contract,
    )

    assert len(report.group_sizes) == 6
    assert {(group.completion_step, group.layer_index) for group in report.group_sizes} == {
        (step, layer) for step in (2, 3) for layer in (1, 2, 3)
    }
    assert all(len(group.group_sizes) == 8 for group in report.group_sizes)
    assert all(sum(group.group_sizes) == 6 for group in report.group_sizes)
    by_step_layer = {
        (group.completion_step, group.layer_index): group.group_sizes
        for group in report.group_sizes
    }
    assert by_step_layer[(2, 1)] == (0, 0, 0, 0, 1, 2, 2, 1)
    assert by_step_layer[(3, 1)] == (1, 0, 0, 0, 0, 1, 2, 2)
    assert report.request_state_slots == (0, 1, 2)
    assert report.recurrent_state_slots == (10, 11, 12)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("batch", "INKLING_ROUTE_SERVER_BATCH_MISMATCH"),
        ("payload", "INKLING_ROUTE_PAYLOAD_NOT_FINAL_ONLY"),
        ("cache", "INKLING_ROUTE_CACHE_REUSE_DETECTED"),
        ("completion", "INKLING_ROUTE_COMPLETION_SEQUENCE_MISMATCH"),
        ("size", "INKLING_ROUTE_FINAL_PAYLOAD_SIZE_MISMATCH"),
        ("base64", "INKLING_ROUTE_BASE64_INVALID"),
        ("slot", "INKLING_ROUTE_REQUEST_SLOT_CHANGED"),
        ("dense", "INKLING_ROUTE_DENSE_LAYER_SENTINEL_MISMATCH"),
        ("range", "INKLING_ROUTE_EXPERT_ID_OUT_OF_RANGE"),
        ("duplicate", "INKLING_ROUTE_DUPLICATE_TOPK_EXPERT"),
    ),
)
def test_route_corpus_rejects_adversarial_capture_mutations(mutation: str, error: str) -> None:
    contract = _contract()
    payload = _capture(contract).model_dump(mode="json")
    if mutation == "batch":
        payload["requests"][0]["chunks"][1]["server_batch_size"] = 2
    elif mutation == "payload":
        payload["requests"][0]["chunks"][1]["routed_experts_present"] = True
    elif mutation == "cache":
        payload["requests"][0]["chunks"][1]["cached_tokens"] = 1
    elif mutation == "completion":
        payload["requests"][0]["chunks"][1]["completion_tokens"] = 1
    elif mutation == "size":
        encoded = payload["requests"][0]["final_routed_experts_base64"]
        payload["requests"][0]["final_routed_experts_base64"] = base64.b64encode(
            base64.b64decode(encoded) + b"\0\0\0\0"
        ).decode()
    elif mutation == "base64":
        payload["requests"][0]["final_routed_experts_base64"] += "\n"
    elif mutation == "slot":
        payload["requests"][0]["chunks"][1]["request_state_slot"] = 9
    else:
        raw = bytearray(base64.b64decode(payload["requests"][0]["final_routed_experts_base64"]))
        values = np.frombuffer(raw, dtype="<i4")
        row = contract.prompt_tokens
        if mutation == "dense":
            values.reshape(-1, 4, 2)[row, 0, 0] = 0
        elif mutation == "range":
            values.reshape(-1, 4, 2)[row, 1, 0] = 8
        else:
            values.reshape(-1, 4, 2)[row, 1] = (2, 2)
        payload["requests"][0]["final_routed_experts_base64"] = base64.b64encode(raw).decode()
    capture = _repair_capture(payload)

    with pytest.raises(ValueError, match=error):
        derive_route_corpus(
            capture=capture,
            capture_sha256="7" * 64,
            raw_sse_sha256="7" * 64,
            raw_sse_bytes=1,
            contract=contract,
        )


@pytest.mark.parametrize(
    "mutation",
    ("profile", "prompt", "producer", "command", "configuration", "request_id", "input"),
)
def test_route_corpus_rejects_self_hashed_provenance_forgeries(mutation: str) -> None:
    contract = _contract()
    payload = _capture(contract).model_dump(mode="json")
    if mutation == "profile":
        payload["profile_contract_id"] = "f" * 64
    elif mutation == "prompt":
        payload["prompt_corpus_sha256"] = "f" * 64
    elif mutation == "producer":
        payload["producer_source_sha256"] = "f" * 64
    elif mutation == "command":
        payload["server_command"] = ["fabricated-server"]
    elif mutation == "configuration":
        payload["server_configuration"] = {"model": "fabricated"}
    elif mutation == "request_id":
        payload["requests"][1]["request_id"] = payload["requests"][0]["request_id"]
    else:
        payload["requests"][1]["input_ids_sha256"] = "f" * 64
    with pytest.raises(
        ValueError,
        match="(INKLING_ROUTE_(CAPTURE_PROVENANCE|REQUEST_MANIFEST)_MISMATCH|route capture)",
    ):
        capture = _repair_capture(payload)
        derive_route_corpus(
            capture=capture,
            capture_sha256="7" * 64,
            raw_sse_sha256="7" * 64,
            raw_sse_bytes=1,
            contract=contract,
        )


@pytest.mark.parametrize("mutation", ("inventory", "counts", "slots", "corpus"))
def test_route_report_rejects_self_hashed_structural_forgeries(mutation: str) -> None:
    contract = _contract()
    report = derive_route_corpus(
        capture=_capture(contract),
        capture_sha256="7" * 64,
        raw_sse_sha256="7" * 64,
        raw_sse_bytes=1,
        contract=contract,
    )
    payload = report.model_dump(mode="json")
    payload["report_id"] = "0" * 64
    if mutation == "inventory":
        payload["group_sizes"] = payload["group_sizes"][1:]
    elif mutation == "counts":
        payload["group_sizes"][0]["group_sizes"] = [999]
    elif mutation == "slots":
        payload["request_state_slots"] = []
    else:
        payload["corpus_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="route corpus"):
        type(report).model_validate(payload)


def test_independent_verifier_reconstructs_the_raw_sse(tmp_path: Path) -> None:
    contract = _contract().model_copy(
        update={
            "producer_source_sha256": file_sha256(Path(route_corpus_module.__file__)),
            "verifier_source_sha256": file_sha256(Path(route_verifier.__file__)),
        }
    )
    initial_capture = _capture(contract)
    raw = _raw_sse(initial_capture)
    capture_payload = initial_capture.model_dump(mode="json")
    capture_payload["raw_sse_sha256"] = hashlib.sha256(raw).hexdigest()
    capture_payload["raw_sse_bytes"] = len(raw)
    capture = _repair_capture(capture_payload)
    raw_path = tmp_path / "capture.sse"
    capture_path = tmp_path / "capture.json"
    contract_path = tmp_path / "contract.json"
    report_path = tmp_path / "report.json"
    raw_path.write_bytes(raw)
    capture_path.write_text(json.dumps(capture.model_dump(mode="json"), sort_keys=True))
    contract_path.write_text(
        json.dumps(contract.model_dump(mode="json", exclude_computed_fields=True), sort_keys=True)
    )
    report = derive_route_corpus(
        capture=capture,
        capture_sha256=file_sha256(capture_path),
        raw_sse_sha256=file_sha256(raw_path),
        raw_sse_bytes=raw_path.stat().st_size,
        contract=contract,
    )
    report_path.write_text(json.dumps(report.model_dump(mode="json"), sort_keys=True))

    assert (
        route_verifier.verify_independently(
            raw_sse_path=raw_path,
            capture_path=capture_path,
            report_path=report_path,
            contract_path=contract_path,
        )
        == report.report_id
    )
    raw_path.write_bytes(raw + b"data: {}\n")
    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_ARTIFACT_MISMATCH"):
        route_verifier.verify_independently(
            raw_sse_path=raw_path,
            capture_path=capture_path,
            report_path=report_path,
            contract_path=contract_path,
        )


def test_committed_route_contract_declares_the_profiled_shape() -> None:
    contract = InklingGmmRouteCorpusContract.model_validate_json(
        Path("contracts/inkling-gmm-route-corpus-v1.json").read_text()
    )
    profile = InklingDecodeProfileContract.model_validate_json(
        Path("contracts/inkling-whole-decode-profile-v1.json").read_text()
    )

    assert contract.profile_contract_id == profile.contract_id
    assert contract.inkling_git_commit == profile.inkling_git_commit
    assert contract.inkling_uv_lock_sha256 == profile.inkling_uv_lock_sha256
    assert contract.model_revision == profile.server.revision
    assert contract.prompt_corpus_sha256 == profile.prompts.corpus_sha256
    assert contract.producer_source_sha256 == file_sha256(Path(route_corpus_module.__file__))
    assert contract.verifier_source_sha256 == file_sha256(Path(route_verifier.__file__))
    assert contract.capture_uv_lock_sha256 == file_sha256(Path("uv.lock"))
    assert contract.concurrency == contract.expected_server_batch_size == 48
    assert contract.prompt_tokens == 1000
    assert contract.selected_completion_steps == tuple(range(2, 66))
    assert contract.num_layers == 42
    assert contract.first_dense_layers == 2
    assert contract.num_experts_per_token == 6
    assert contract.num_routed_experts == 256
    assert (
        _parser()
        .parse_args(
            [
                "verify",
                "report.json",
                "--capture",
                "capture.json",
                "--raw-sse",
                "capture.sse",
                "--contract",
                "contract.json",
            ]
        )
        .command
        == "verify"
    )
