import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpu_cake.contracts import ProfileExpectation, RuntimeIdentity, SourceFileContract
from tpu_cake.evidence import (
    ArtifactEvidence,
    CaptureEvidence,
    CounterEvidence,
    PlaneEvidence,
    ProgramEvidence,
)
from tpu_cake.inkling_decode_profile import (
    HloIdentityStatus,
    InklingDecodeProfileContract,
    InklingDecodeProgramContract,
    InklingDecodePromptContract,
    InklingDecodeServerContract,
    _require_new_output,
    _require_outside_repositories,
    assess_inkling_decode_profile,
    validate_inkling_decode_profile,
    write_inkling_decode_profile_assessment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = "1" * 32
XPLANE_START = 1_700_000_000_000_000_000
XPLANE_STOP = XPLANE_START + 1_000_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _artifact(path: str, digest: str = "a" * 64) -> ArtifactEvidence:
    return ArtifactEvidence(path=Path(path), size_bytes=1, sha256=digest)


def _server() -> InklingDecodeServerContract:
    return InklingDecodeServerContract(
        model_path="thinkingmachines/Inkling-Small",
        revision="8" * 40,
        dtype="bfloat16",
        context_length=4096,
        max_running_requests=2,
        max_total_tokens=32768,
        max_prefill_tokens=1024,
        chunked_prefill_size=1024,
        page_size=128,
        tp_size=8,
        ep_size=8,
        attention_backend="native",
        moe_backend="epmoe",
        disable_overlap_schedule=False,
        disable_radix_cache=True,
        version="0.0.0.dev0",
    )


def _prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "first",
                    "input_ids": [1, 2],
                    "standard_input_tokens": 2,
                    "decoded_text_sha256": "1" * 64,
                },
                {
                    "id": "second",
                    "input_ids": [3, 4, 5],
                    "standard_input_tokens": 3,
                    "decoded_text_sha256": "2" * 64,
                },
            ],
            separators=(",", ":"),
        )
    )
    return path


def _contract(prompt_path: Path, *, pinned: bool) -> InklingDecodeProfileContract:
    event_counts = {f"/device:TPU:{index}": 3 for index in range(8)} if pinned else {}
    return InklingDecodeProfileContract(
        name="test",
        hlo_identity_status=HloIdentityStatus.PINNED if pinned else HloIdentityStatus.PENDING,
        inkling_git_commit="9" * 40,
        inkling_uv_lock_sha256="b" * 64,
        inkling_source_root="/tmp/inkling",
        capture_git_commit=_head() if pinned else "0" * 40,
        capture_uv_lock_sha256="e" * 64,
        capture_source_manifest=(SourceFileContract(path="source.py", sha256="c" * 64),),
        server_command_fragments=("sgl_jax.launch_server", "Inkling-Small"),
        runtime=RuntimeIdentity(
            python="3.12.3",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla=" --xla_tpu_use_enhanced_launch_barrier=true",
        ),
        xplane_device_type="TPU v7x",
        xplane_tpu_version="runtime",
        device_count=8,
        process_count=1,
        concurrency=2,
        output_tokens=3,
        profile_stop_after_minimum_completion_tokens=3,
        host_tracer_level=0,
        python_tracer_level=0,
        prompts=InklingDecodePromptContract(
            corpus_sha256=_sha256(prompt_path), selected_case_ids=("first", "second")
        ),
        server=_server(),
        profile=ProfileExpectation(
            name="test",
            stage="steady_decode",
            minimum_tpu_device_planes=8,
            required_timed_hlo_markers=("ragged_paged_attention", "gmm_v2"),
        ),
        main_program_prefix="jit_jitted_run_model",
        programs=(
            InklingDecodeProgramContract(
                name_prefix="jit_jitted_run_model",
                hlo_sha256="a" * 64 if pinned else "0" * 64,
                module_events_per_tpu_core=event_counts,
            ),
            InklingDecodeProgramContract(
                name_prefix="jit_jitted_sampler",
                hlo_sha256="d" * 64 if pinned else "0" * 64,
                module_events_per_tpu_core=event_counts,
            ),
        ),
    )


def _capture() -> CaptureEvidence:
    return CaptureEvidence(
        xplane=_artifact("capture.xplane.pb"),
        hlo_stats=_artifact("hlo_stats.json"),
        planes=tuple(
            PlaneEvidence(
                name=f"/device:TPU:{index}",
                device_type="TPU v7x",
                line_count=1,
                event_count=1,
                tensor_core_event_count=1,
            )
            for index in range(8)
        ),
        counters=CounterEvidence(
            hbm_read_names=0,
            hbm_write_names=0,
            cycle_names=0,
            periodic_counter_names=(),
            periodic_samples_per_tpu_core={},
        ),
        programs=(
            ProgramEvidence(
                program_id="1",
                name="jit_jitted_run_model(1)",
                timed_self_us=100,
                hlo=_artifact("jit_jitted_run_model(1).hlo_proto.pb", "a" * 64),
                marker_counts={"ragged_paged_attention": 1, "gmm_v2": 1},
                forbidden_fragment_hits={},
            ),
            ProgramEvidence(
                program_id="2",
                name="jit_jitted_sampler(2)",
                timed_self_us=2,
                hlo=_artifact("jit_jitted_sampler(2).hlo_proto.pb", "d" * 64),
                marker_counts={"ragged_paged_attention": 0, "gmm_v2": 0},
                forbidden_fragment_hits={},
            ),
        ),
        timed_program_ids=frozenset({"1", "2"}),
    )


def _module_events() -> dict[str, dict[str, int]]:
    return {
        f"/device:TPU:{index}": {"jit_jitted_run_model(1)": 3, "jit_jitted_sampler(2)": 3}
        for index in range(8)
    }


def _chunks(request_id: str, request_slot: int) -> list[dict[str, object]]:
    output_ids: list[int] = []
    chunks: list[dict[str, object]] = []
    for completion in range(1, 4):
        output_ids.append(request_slot * 10 + completion)
        chunks.append(
            {
                "elapsed_ms": float(completion),
                "response": {
                    "meta_info": {
                        "id": request_id,
                        "completion_tokens": completion,
                        "request_state_slot": request_slot,
                        "recurrent_state_slot": request_slot + 10,
                        "server_batch_size": 2,
                    },
                    "output_ids": list(output_ids),
                    "text": "x" * completion,
                },
            }
        )
    return chunks


def _write_request(
    tmp_path: Path, prompt_path: Path, contract: InklingDecodeProfileContract
) -> Path:
    cases = []
    for case in json.loads(prompt_path.read_text()):
        cases.append(
            {
                "id": case["id"],
                "native_input_tokens": len(case["input_ids"]),
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(case["input_ids"], separators=(",", ":")).encode()
                ).hexdigest(),
                "standard_input_tokens": case["standard_input_tokens"],
                "decoded_prompt_sha256": case["decoded_text_sha256"],
            }
        )
    request_ids = [f"{SESSION_ID}:0", f"{SESSION_ID}:1"]
    path = tmp_path / "profile-request.json"
    path.write_text(
        json.dumps(
            {
                "request": {
                    "rid": request_ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 3, "ignore_eos": True},
                    "stream": True,
                    "return_routed_experts": False,
                    "prompt_cases": cases,
                },
                "provenance": {
                    "captured_at_utc": datetime.fromtimestamp(XPLANE_STOP / 1e9, UTC).isoformat(),
                    "command": ["capture"],
                    "tpu_cake_git_commit": _head(),
                    "tpu_cake_git_status_porcelain": "",
                    "tpu_cake_uv_lock_sha256": contract.capture_uv_lock_sha256,
                    "capture_runtime": contract.runtime.model_dump(mode="json"),
                    "capture_source_manifest": [
                        source.model_dump(mode="json")
                        for source in contract.capture_source_manifest
                    ],
                    "inkling_git_commit": contract.inkling_git_commit,
                    "inkling_git_status_porcelain": "",
                    "inkling_uv_lock_sha256": contract.inkling_uv_lock_sha256,
                    "hostname": "tpu",
                    "python": "3.12.3",
                    "prompt_cases_file_sha256": contract.prompts.corpus_sha256,
                    "profile_session_id": SESSION_ID,
                    "profile_directory_name": SESSION_ID,
                    "capture_xplane_sha256": "a" * 64,
                    "xplane_process_id": 123,
                    "xplane_profile_start_time_ns": XPLANE_START,
                    "xplane_profile_stop_time_ns": XPLANE_STOP,
                    "profile_start_requested_at_ns": XPLANE_START - 1,
                    "profile_stop_completed_at_ns": XPLANE_STOP + 1,
                    "server_process": {
                        "pid": 100,
                        "create_time": 1.0,
                        "cwd": contract.inkling_source_root,
                        "cmdline": [
                            "python",
                            "-m",
                            "sgl_jax.launch_server",
                            "--model-path",
                            "thinkingmachines/Inkling-Small",
                        ],
                    },
                    "server_info": contract.server.model_dump(mode="json"),
                },
                "profile_start_response": {"status_code": 200},
                "profile_stop_response": {"status_code": 200},
                "profile_start_condition": "every request emitted at least one token",
                "profile_stop_after_minimum_completion_tokens": 3,
                "host_tracer_level": 0,
                "python_tracer_level": 0,
                "elapsed_ms": 1,
                "chunks_by_request": [_chunks(request_ids[0], 0), _chunks(request_ids[1], 1)],
            }
        )
    )
    return path


def _assess(
    request: Path,
    prompts: Path,
    contract: InklingDecodeProfileContract,
    *,
    capture: CaptureEvidence | None = None,
    module_events: dict[str, dict[str, int]] | None = None,
    xplane_runtime: dict[str, str] | None = None,
):
    return assess_inkling_decode_profile(
        capture=_capture() if capture is None else capture,
        module_events_per_tpu_core=_module_events() if module_events is None else module_events,
        xplane_process_id=123,
        xplane_profile_start_time_ns=XPLANE_START,
        xplane_profile_stop_time_ns=XPLANE_STOP,
        xplane_runtime=xplane_runtime
        or {
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "tpu_runtime": "runtime",
            "device_type": "TPU v7x",
            "process_id": "123",
        },
        capture_root_name=SESSION_ID,
        request_path=request,
        prompt_cases_path=prompts,
        contract=contract,
    )


def test_pinned_whole_decode_profile_is_accepted(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    assert _assess(_write_request(tmp_path, prompts, contract), prompts, contract).accepted


def test_pending_hlo_identity_is_descriptive_only(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=False)
    assessment = _assess(_write_request(tmp_path, prompts, contract), prompts, contract)
    assert not assessment.accepted
    assert {finding.code for finding in assessment.findings} == {"HLO_IDENTITIES_PENDING"}


def test_empty_chunks_wrong_request_ids_and_timestamp_are_rejected(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    request = _write_request(tmp_path, prompts, contract)
    payload = json.loads(request.read_text())
    payload["request"]["rid"] = ["wrong-0", "wrong-1"]
    payload["provenance"]["captured_at_utc"] = "1999-01-01T00:00:00+00:00"
    payload["chunks_by_request"] = [[], []]
    request.write_text(json.dumps(payload))
    codes = {finding.code for finding in _assess(request, prompts, contract).findings}
    assert codes >= {
        "REQUEST_SESSION_ID_MISMATCH",
        "OUTPUT_TOKEN_COUNT_MISMATCH",
        "XPLANE_TIME_RANGE_MISMATCH",
    }


def test_wrong_server_process_and_xplane_are_rejected(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    request = _write_request(tmp_path, prompts, contract)
    payload = json.loads(request.read_text())
    payload["provenance"]["server_process"]["cwd"] = "/wrong"
    payload["provenance"]["server_process"]["cmdline"] = ["python", "wrong"]
    payload["provenance"]["capture_xplane_sha256"] = "f" * 64
    payload["provenance"]["tpu_cake_git_commit"] = "0" * 40
    request.write_text(json.dumps(payload))
    codes = {finding.code for finding in _assess(request, prompts, contract).findings}
    assert codes >= {
        "REQUEST_XPLANE_MISMATCH",
        "CAPTURE_SOURCE_COMMIT_MISMATCH",
        "SERVER_PROCESS_COMMAND_MISMATCH",
        "SERVER_PROCESS_CWD_MISMATCH",
    }


def test_serial_batch_and_shared_recurrent_slot_are_rejected(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    request = _write_request(tmp_path, prompts, contract)
    payload = json.loads(request.read_text())
    for chunks in payload["chunks_by_request"]:
        for chunk in chunks:
            metadata = chunk["response"]["meta_info"]
            metadata["server_batch_size"] = 1
            metadata["recurrent_state_slot"] = 0
    request.write_text(json.dumps(payload))
    codes = {finding.code for finding in _assess(request, prompts, contract).findings}
    assert codes >= {
        "SERVER_BATCH_SIZE_MISMATCH",
        "RECURRENT_STATE_SLOT_INVENTORY_MISMATCH",
    }


def test_raw_module_event_inventory_and_counts_are_exact(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    request = _write_request(tmp_path, prompts, contract)
    events = _module_events()
    events["/device:TPU:0"]["jit_jitted_run_model(1)"] = 2
    events["/device:TPU:0"]["jit_unexpected(9)"] = 1
    events["/device:TPU:1"] = {
        "jit_jitted_run_model(999)": 3,
        "jit_jitted_sampler(998)": 3,
    }
    codes = {
        finding.code
        for finding in _assess(request, prompts, contract, module_events=events).findings
    }
    assert codes >= {
        "MODULE_EVENT_COUNT_MISMATCH",
        "MODULE_EVENT_PROGRAM_ID_MISMATCH",
        "MODULE_EVENT_SET_MISMATCH",
    }


def test_wrong_runtime_and_tpu_generation_are_rejected(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    request = _write_request(tmp_path, prompts, contract)
    payload = json.loads(request.read_text())
    payload["provenance"]["capture_runtime"]["jax"] = "0.12.0"
    request.write_text(json.dumps(payload))
    capture = _capture().model_copy(
        update={
            "planes": tuple(
                plane.model_copy(update={"device_type": "TPU v6e"}) for plane in _capture().planes
            )
        }
    )
    codes = {
        finding.code
        for finding in _assess(
            request,
            prompts,
            contract,
            capture=capture,
            xplane_runtime={
                "jax": "0.12.0",
                "jaxlib": "0.12.0",
                "tpu_runtime": "wrong",
                "device_type": "TPU v6e",
                "process_id": "123",
            },
        ).findings
    }
    assert codes >= {
        "CAPTURE_RUNTIME_MISMATCH",
        "XPLANE_DEVICE_TYPE_MISMATCH",
        "XPLANE_RUNTIME_MISMATCH",
    }


def test_missing_sampler_wrong_server_and_extra_timed_program_are_rejected(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    request = _write_request(tmp_path, prompts, contract)
    payload = json.loads(request.read_text())
    payload["provenance"]["server_info"]["moe_backend"] = "wrong"
    request.write_text(json.dumps(payload))
    capture = _capture()
    extra = capture.programs[1].model_copy(update={"program_id": "3", "name": "jit_unexpected(3)"})
    capture = capture.model_copy(
        update={
            "programs": (capture.programs[0], extra),
            "timed_program_ids": frozenset({"1", "3"}),
        }
    )
    codes = {
        finding.code for finding in _assess(request, prompts, contract, capture=capture).findings
    }
    assert codes >= {"SERVER_CONFIGURATION_MISMATCH", "TIMED_PROGRAM_INVENTORY_MISMATCH"}


def test_main_markers_must_be_in_the_model_program(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    request = _write_request(tmp_path, prompts, contract)
    capture = _capture()
    main = capture.programs[0].model_copy(
        update={"marker_counts": {"ragged_paged_attention": 0, "gmm_v2": 0}}
    )
    assessment = _assess(
        request,
        prompts,
        contract,
        capture=capture.model_copy(update={"programs": (main, capture.programs[1])}),
    )
    assert "MAIN_PROGRAM_MARKER_MISSING" in {finding.code for finding in assessment.findings}


def test_tracked_whole_decode_contract_is_pending_and_source_bound() -> None:
    contract = InklingDecodeProfileContract.model_validate_json(
        (REPO_ROOT / "contracts" / "inkling-whole-decode-profile-v1.json").read_text()
    )
    assert contract.hlo_identity_status is HloIdentityStatus.PENDING
    assert contract.capture_uv_lock_sha256 == _sha256(REPO_ROOT / "uv.lock")
    assert len(contract.capture_source_manifest) >= 4
    for source in contract.capture_source_manifest:
        assert source.sha256 == _sha256(REPO_ROOT / source.path)
    assert len(contract.prompts.selected_case_ids) == contract.concurrency == 48
    assert contract.server.max_running_requests >= contract.concurrency
    assert contract.server.max_total_tokens >= contract.concurrency * (
        1_000 + contract.output_tokens
    )


def test_contract_rejects_insufficient_request_capacity(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=False)
    server = contract.server.model_copy(update={"max_running_requests": 1})
    with pytest.raises(ValueError, match="server request capacity must cover concurrency"):
        InklingDecodeProfileContract.model_validate(
            {
                **contract.model_dump(mode="python", exclude={"contract_id"}),
                "server": server,
            }
        )


def test_public_validation_rejects_pending_before_capture_reads(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=False)
    with pytest.raises(ValueError, match="^INKLING_DECODE_PROFILE_HLO_IDENTITIES_PENDING$"):
        validate_inkling_decode_profile(
            capture_root=tmp_path / "missing-capture",
            request_path=tmp_path / "missing-request.json",
            prompt_cases_path=tmp_path / "missing-prompts.json",
            contract=contract,
        )


def test_output_rejects_existing_path_and_every_symlink_ancestor(tmp_path) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("owned")
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    for path in (existing, alias / "nested" / "new.json"):
        with pytest.raises(ValueError):
            _require_new_output(path)


def test_capture_outputs_cannot_be_created_inside_source_repositories(tmp_path) -> None:
    with pytest.raises(ValueError, match="INKLING_PROFILE_OUTPUT_INSIDE_SOURCE_REPOSITORY"):
        _require_outside_repositories(REPO_ROOT / "runs" / "capture", (REPO_ROOT, tmp_path))


def test_assessment_writer_refuses_to_clobber(tmp_path) -> None:
    prompts = _prompt_file(tmp_path)
    contract = _contract(prompts, pinned=True)
    assessment = _assess(_write_request(tmp_path, prompts, contract), prompts, contract)
    output = tmp_path / "assessment.json"
    write_inkling_decode_profile_assessment(output, assessment)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="INKLING_PROFILE_OUTPUT_EXISTS"):
        write_inkling_decode_profile_assessment(output, assessment)
    assert output.read_bytes() == before
