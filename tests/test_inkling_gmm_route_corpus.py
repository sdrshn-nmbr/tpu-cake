import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tpu_cake import inkling_gmm_route_corpus as route_corpus_module
from tpu_cake import inkling_gmm_route_corpus_verifier as route_verifier
from tpu_cake.artifacts import file_sha256
from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import json_sha256, model_identity_sha256
from tpu_cake.inkling_decode_profile import InklingDecodeProfileContract
from tpu_cake.inkling_gmm_route_corpus import (
    InklingGmmRouteCapture,
    InklingGmmRouteCorpusContract,
    InklingGmmRouteCorpusReport,
    ModelArtifactEvidence,
    RouteChunkEvidence,
    RouteRequestEvidence,
    RouteServerLaunchReceipt,
    _consume_route_sse,
    _json_sha256,
    _manifest_sha256,
    _model_weight_manifest,
    _parser,
    _require_new_external_artifacts,
    _server_process_environment,
    derive_route_corpus,
    write_report,
)


def _contract() -> InklingGmmRouteCorpusContract:
    model_manifest = (
        ModelArtifactEvidence(path="model.safetensors.index.json", sha256="2" * 64, bytes=10),
        ModelArtifactEvidence(path="model-00001.safetensors", sha256="3" * 64, bytes=20),
    )
    return InklingGmmRouteCorpusContract(
        name="test",
        profile_contract_id="a" * 64,
        inkling_git_commit="b" * 40,
        inkling_uv_lock_sha256="c" * 64,
        inkling_source_manifest=(SourceFileContract(path="a.py", sha256="d" * 64),),
        model_revision="e" * 40,
        model_config_sha256="f" * 64,
        model_weight_manifest_sha256=_json_sha256(
            [item.model_dump(mode="json") for item in model_manifest]
        ),
        capture_source_manifest=(
            SourceFileContract(
                path="src/tpu_cake/inkling_gmm_route_corpus.py",
                sha256=file_sha256(Path(route_corpus_module.__file__)),
            ),
            SourceFileContract(
                path="src/tpu_cake/inkling_gmm_route_corpus_verifier.py",
                sha256=file_sha256(Path(route_verifier.__file__)),
            ),
        ),
        producer_source_sha256="9" * 64,
        verifier_source_sha256="a" * 64,
        capture_uv_lock_sha256="c" * 64,
        profile_server_configuration_sha256=_json_sha256({"model": "test"}),
        prompt_case_manifest_sha256=_json_sha256(
            [
                {
                    "id": f"case-{index}",
                    "input_ids_sha256": _json_sha256([index] * 3),
                }
                for index in range(3)
            ]
        ),
        required_server_command_fragments=("server", "--enable-return-routed-experts"),
        required_server_environment={"HF_HOME": "/tmp/huggingface"},
        forbidden_server_environment_names=("XLA_FLAGS", "PYTHONPATH"),
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
    model_manifest = (
        ModelArtifactEvidence(path="model.safetensors.index.json", sha256="2" * 64, bytes=10),
        ModelArtifactEvidence(path="model-00001.safetensors", sha256="3" * 64, bytes=20),
    )
    launch_receipt = _launch_receipt(contract)
    launch_receipt_bytes = _launch_receipt_bytes(launch_receipt)
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
                    server_batch_size=1 if completion == 1 else contract.concurrency,
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
                request_id=f"{'a' * 32}:{request_index}",
                prompt_case_id=f"case-{request_index}",
                input_ids_sha256=_json_sha256([request_index] * contract.prompt_tokens),
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
        observed_tpu_cake_git_commit="4" * 40,
        server_process_id=1,
        server_command=("python", "-m", "server", "--enable-return-routed-experts"),
        server_environment=contract.required_server_environment,
        server_configuration={"model": "test"},
        server_launch_receipt_id=launch_receipt.receipt_id,
        server_launch_receipt_sha256=hashlib.sha256(launch_receipt_bytes).hexdigest(),
        server_launch_nonce=launch_receipt.launch_nonce,
        request_sha256="8" * 64,
        request_bytes=1,
        model_weight_manifest_sha256=_json_sha256(
            [item.model_dump(mode="json") for item in model_manifest]
        ),
        model_weight_manifest=model_manifest,
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


def _request_bytes(
    capture: InklingGmmRouteCapture, contract: InklingGmmRouteCorpusContract
) -> bytes:
    request = {
        "rid": [item.request_id for item in capture.requests],
        "input_ids": [[index] * contract.prompt_tokens for index in range(contract.concurrency)],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": contract.output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_routed_experts": True,
    }
    return json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _launch_receipt(
    contract: InklingGmmRouteCorpusContract,
) -> RouteServerLaunchReceipt:
    model_manifest = (
        ModelArtifactEvidence(path="model.safetensors.index.json", sha256="2" * 64, bytes=10),
        ModelArtifactEvidence(path="model-00001.safetensors", sha256="3" * 64, bytes=20),
    )
    provisional = RouteServerLaunchReceipt(
        receipt_id="0" * 64,
        contract_id=contract.contract_id,
        launch_nonce="5" * 32,
        observed_tpu_cake_git_commit="4" * 40,
        capture_source_manifest_sha256=_json_sha256(
            [item.model_dump(mode="json") for item in contract.capture_source_manifest]
        ),
        inkling_git_commit=contract.inkling_git_commit,
        inkling_uv_lock_sha256=contract.inkling_uv_lock_sha256,
        inkling_source_manifest_sha256=_json_sha256(
            [item.model_dump(mode="json") for item in contract.inkling_source_manifest]
        ),
        model_revision=contract.model_revision,
        model_config_sha256=contract.model_config_sha256,
        model_weight_manifest_sha256=contract.model_weight_manifest_sha256,
        model_weight_manifest=model_manifest,
        server_command=("python", "-m", "server", "--enable-return-routed-experts"),
        server_environment=contract.required_server_environment,
    )
    return provisional.model_copy(
        update={"receipt_id": model_identity_sha256(provisional, exclude={"receipt_id"})}
    )


def _launch_receipt_bytes(receipt: RouteServerLaunchReceipt) -> bytes:
    return (json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()


def _independent_bundle(tmp_path: Path) -> dict[str, object]:
    contract = _contract().model_copy(
        update={
            "producer_source_sha256": file_sha256(Path(route_corpus_module.__file__)),
            "verifier_source_sha256": file_sha256(Path(route_verifier.__file__)),
        }
    )
    initial_capture = _capture(contract)
    raw = _raw_sse(initial_capture)
    request_bytes = _request_bytes(initial_capture, contract)
    capture_payload = initial_capture.model_dump(mode="json")
    capture_payload["raw_sse_sha256"] = hashlib.sha256(raw).hexdigest()
    capture_payload["raw_sse_bytes"] = len(raw)
    capture_payload["request_sha256"] = hashlib.sha256(request_bytes).hexdigest()
    capture_payload["request_bytes"] = len(request_bytes)
    capture = _repair_capture(capture_payload)
    paths = {
        "raw": tmp_path / "capture.sse",
        "request": tmp_path / "capture.request.json",
        "launch_receipt": tmp_path / "server-launch.json",
        "capture": tmp_path / "capture.json",
        "contract": tmp_path / "contract.json",
        "report": tmp_path / "report.json",
    }
    paths["raw"].write_bytes(raw)
    paths["request"].write_bytes(request_bytes)
    paths["launch_receipt"].write_bytes(_launch_receipt_bytes(_launch_receipt(contract)))
    paths["capture"].write_text(json.dumps(capture.model_dump(mode="json"), sort_keys=True))
    paths["contract"].write_text(
        json.dumps(contract.model_dump(mode="json", exclude_computed_fields=True), sort_keys=True)
    )
    report = derive_route_corpus(
        capture=capture,
        capture_sha256=file_sha256(paths["capture"]),
        raw_sse_sha256=file_sha256(paths["raw"]),
        raw_sse_bytes=paths["raw"].stat().st_size,
        contract=contract,
    )
    paths["report"].write_text(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    return {"contract_model": contract, "report_model": report, **paths}


def _independently_verify(bundle: dict[str, object]) -> str:
    return route_verifier.verify_independently(
        raw_sse_path=bundle["raw"],
        request_path=bundle["request"],
        launch_receipt_path=bundle["launch_receipt"],
        capture_path=bundle["capture"],
        report_path=bundle["report"],
        contract_path=bundle["contract"],
    )


def _repair_bundle_bindings(bundle: dict[str, object]) -> None:
    capture_path = bundle["capture"]
    report_path = bundle["report"]
    capture = json.loads(capture_path.read_text())
    capture["capture_id"] = json_sha256(
        {key: value for key, value in capture.items() if key != "capture_id"}
    )
    capture_path.write_text(json.dumps(capture, sort_keys=True))
    report = json.loads(report_path.read_text())
    report["capture_id"] = capture["capture_id"]
    report["capture_sha256"] = file_sha256(capture_path)
    report["request_sha256"] = capture["request_sha256"]
    report["report_id"] = json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    report_path.write_text(json.dumps(report, sort_keys=True))


def test_route_corpus_reconstructs_each_full_batch_decode_layer() -> None:
    contract = _contract()
    capture = _capture(contract)
    report = derive_route_corpus(
        capture=capture,
        capture_sha256="7" * 64,
        raw_sse_sha256="7" * 64,
        raw_sse_bytes=1,
        contract=contract,
    )

    assert all(request.chunks[0].server_batch_size == 1 for request in capture.requests)
    assert all(
        chunk.server_batch_size == contract.concurrency
        for request in capture.requests
        for chunk in request.chunks[1:]
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
    bundle = _independent_bundle(tmp_path)

    assert _independently_verify(bundle) == bundle["report_model"].report_id
    bundle["raw"].write_bytes(bundle["raw"].read_bytes() + b"data: {}\n")
    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_ARTIFACT_MISMATCH"):
        _independently_verify(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "forged-capture-v9"),
        ("tpu_cake_git_status_porcelain", " M source.py"),
        ("server_idle_before", False),
        ("server_idle_after", False),
        ("cohort_assumption", "known-mixed-cohort"),
        ("unknown_attestation", True),
    ),
)
def test_independent_verifier_rejects_capture_claim_forgeries(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = _independent_bundle(tmp_path)
    capture = json.loads(bundle["capture"].read_text())
    capture[field] = value
    bundle["capture"].write_text(json.dumps(capture, sort_keys=True))

    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA"):
        _independently_verify(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", "forged-report-v9"), ("independently_verified", True)),
)
def test_independent_verifier_rejects_report_claim_forgeries(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = _independent_bundle(tmp_path)
    report = json.loads(bundle["report"].read_text())
    report[field] = value
    bundle["report"].write_text(json.dumps(report, sort_keys=True))

    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_REPORT_SCHEMA"):
        _independently_verify(bundle)


def test_independent_verifier_reconstructs_exact_request(tmp_path: Path) -> None:
    bundle = _independent_bundle(tmp_path)
    request = json.loads(bundle["request"].read_text())
    request["sampling_params"]["max_new_tokens"] += 1
    raw = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    bundle["request"].write_bytes(raw)
    capture = json.loads(bundle["capture"].read_text())
    capture["request_sha256"] = file_sha256(bundle["request"])
    capture["request_bytes"] = len(raw)
    bundle["capture"].write_text(json.dumps(capture, sort_keys=True))
    _repair_bundle_bindings(bundle)

    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_REQUEST_INVALID"):
        _independently_verify(bundle)


def test_independent_verifier_requires_terminal_done(tmp_path: Path) -> None:
    bundle = _independent_bundle(tmp_path)
    lines = bundle["raw"].read_bytes().splitlines(keepends=True)
    bundle["raw"].write_bytes(lines[-1] + b"".join(lines[:-1]))
    capture = json.loads(bundle["capture"].read_text())
    capture["raw_sse_sha256"] = file_sha256(bundle["raw"])
    capture["raw_sse_bytes"] = bundle["raw"].stat().st_size
    bundle["capture"].write_text(json.dumps(capture, sort_keys=True))
    _repair_bundle_bindings(bundle)

    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_SSE_DATA_AFTER_DONE"):
        _independently_verify(bundle)


def test_independent_verifier_rejects_model_weight_manifest_forgery(tmp_path: Path) -> None:
    bundle = _independent_bundle(tmp_path)
    capture = json.loads(bundle["capture"].read_text())
    capture["model_weight_manifest"][1]["sha256"] = "f" * 64
    capture["model_weight_manifest_sha256"] = json_sha256(capture["model_weight_manifest"])
    bundle["capture"].write_text(json.dumps(capture, sort_keys=True))
    _repair_bundle_bindings(bundle)

    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_CAPTURE_SCHEMA"):
        _independently_verify(bundle)


def test_independent_verifier_rejects_forged_launch_command(tmp_path: Path) -> None:
    bundle = _independent_bundle(tmp_path)
    receipt = json.loads(bundle["launch_receipt"].read_text())
    receipt["server_command"] = ["fabricated-server"]
    receipt["receipt_id"] = json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_id"}
    )
    bundle["launch_receipt"].write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    capture = json.loads(bundle["capture"].read_text())
    capture["server_launch_receipt_id"] = receipt["receipt_id"]
    capture["server_launch_receipt_sha256"] = file_sha256(bundle["launch_receipt"])
    bundle["capture"].write_text(json.dumps(capture, sort_keys=True))
    report = json.loads(bundle["report"].read_text())
    report["server_launch_receipt_id"] = receipt["receipt_id"]
    report["server_launch_receipt_sha256"] = capture["server_launch_receipt_sha256"]
    bundle["report"].write_text(json.dumps(report, sort_keys=True))
    _repair_bundle_bindings(bundle)

    with pytest.raises(ValueError, match="INKLING_ROUTE_VERIFY_LAUNCH_CONTENT_MISMATCH"):
        _independently_verify(bundle)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing", "INKLING_ROUTE_SSE_DONE_MISSING"),
        ("duplicate", "INKLING_ROUTE_SSE_DONE_DUPLICATE"),
        ("before_data", "INKLING_ROUTE_SSE_DATA_AFTER_DONE"),
    ),
)
def test_capture_parser_requires_one_terminal_done(mutation: str, error: str) -> None:
    contract = _contract()
    capture = _capture(contract)
    lines = _raw_sse(capture).splitlines(keepends=True)
    if mutation == "missing":
        lines = lines[:-1]
    elif mutation == "duplicate":
        lines.append(lines[-1])
    else:
        lines = [lines[-1], *lines[:-1]]
    recorded = io.BytesIO()

    with pytest.raises(ValueError, match=error):
        _consume_route_sse(
            lines,
            recorded,
            request_ids=tuple(request.request_id for request in capture.requests),
            contract=contract,
        )
    assert recorded.getvalue().startswith(lines[0])


def test_model_manifest_changes_when_a_shard_changes(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-00001.safetensors",
                    "second": "model-00002.safetensors",
                }
            }
        )
    )
    (tmp_path / "model-00001.safetensors").write_bytes(b"first")
    second = tmp_path / "model-00002.safetensors"
    second.write_bytes(b"second")
    initial = _model_weight_manifest(tmp_path / "config.json")

    second.write_bytes(b"changed")
    changed = _model_weight_manifest(tmp_path / "config.json")

    assert _manifest_sha256(initial) != _manifest_sha256(changed)
    assert initial[2].sha256 != changed[2].sha256


def test_capture_artifacts_must_be_new_and_outside_sources(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    existing = tmp_path / "existing.json"
    existing.write_text("occupied")

    with pytest.raises(FileExistsError, match="OUTPUT_EXISTS"):
        _require_new_external_artifacts(
            (existing,),
            (repository,),
            exists_error="OUTPUT_EXISTS",
            inside_error="OUTPUT_INSIDE_SOURCE",
        )
    with pytest.raises(ValueError, match="OUTPUT_INSIDE_SOURCE"):
        _require_new_external_artifacts(
            (repository / "capture.json",),
            (repository,),
            exists_error="OUTPUT_EXISTS",
            inside_error="OUTPUT_INSIDE_SOURCE",
        )
    report = derive_route_corpus(
        capture=_capture(_contract()),
        capture_sha256="7" * 64,
        raw_sse_sha256="7" * 64,
        raw_sse_bytes=1,
        contract=_contract(),
    )
    with pytest.raises(FileExistsError):
        write_report(existing, report)
    assert existing.read_text() == "occupied"


def test_launch_environment_scrubs_execution_overrides() -> None:
    environment = _server_process_environment(
        _contract(),
        launch_nonce="5" * 32,
        launch_receipt_sha256="6" * 64,
        base_environment={
            "PATH": "/bin",
            "HF_HOME": "/wrong",
            "XLA_FLAGS": "--override",
            "PYTHONPATH": "/tmp/injected",
        },
    )

    assert environment["PATH"] == "/bin"
    assert environment["HF_HOME"] == "/tmp/huggingface"
    assert environment["TPU_CAKE_ROUTE_LAUNCH_NONCE"] == "5" * 32
    assert environment["TPU_CAKE_ROUTE_LAUNCH_RECEIPT_SHA256"] == "6" * 64
    assert "XLA_FLAGS" not in environment
    assert "PYTHONPATH" not in environment


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
    assert contract.model_weight_manifest_sha256 == (
        "8639eab79916427b771fb0a4890ecc9d8bf442baf3884e376e56cfc17308778e"
    )
    assert all(
        file_sha256(Path(source.path)) == source.sha256
        for source in contract.capture_source_manifest
    )
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
                "--request",
                "request.json",
                "--launch-receipt",
                "server-launch.json",
                "--contract",
                "contract.json",
            ]
        )
        .command
        == "verify"
    )


def test_committed_route_report_binds_the_documented_claim() -> None:
    report_path = Path("evidence/inkling/gmm-route-corpus-v1.json")
    report = InklingGmmRouteCorpusReport.model_validate_json(report_path.read_text())
    contract = InklingGmmRouteCorpusContract.model_validate_json(
        Path("contracts/inkling-gmm-route-corpus-v1.json").read_text()
    )

    assert report.contract_id == contract.contract_id
    assert report.report_id == "7d6b21dd9ef7879c5cb7050abf1f2327e504d1ae736913f2e502d6e575c225fa"
    assert (
        report.corpus_sha256 == "d3f7db0bdf366c12924e4b6b8e5f4d19a571b86cf2b222be17179a09f93044a4"
    )
    assert report.producer_source_sha256 == file_sha256(Path(route_corpus_module.__file__))
    assert report.verifier_source_sha256 == file_sha256(Path(route_verifier.__file__))
    assert report.selected_completion_steps == tuple(range(2, 66))
    assert (report.first_moe_layer, report.num_layers) == (2, 42)
    assert len(report.group_sizes) == 64 * 40
    assert all(len(group.group_sizes) == 256 for group in report.group_sizes)
    assert all(sum(group.group_sizes) == 288 for group in report.group_sizes)
    assert len(report.request_state_slots) == len(set(report.request_state_slots)) == 48
    assert len(report.recurrent_state_slots) == len(set(report.recurrent_state_slots)) == 48
    assert report.cohort_scope == "operational-assumption-no-emitted-step-id"
    assert file_sha256(report_path) == (
        "010870ad0aedc93708534bd1a3f4e2ffd69addacd69f5314992df1d5fae5ff63"
    )
    assert (
        "The committed replay is report "
        "`7d6b21dd9ef7879c5cb7050abf1f2327e504d1ae736913f2e502d6e575c225fa` "
        "with file SHA-256 "
        "`010870ad0aedc93708534bd1a3f4e2ffd69addacd69f5314992df1d5fae5ff63`."
        in Path("README.md").read_text()
    )
