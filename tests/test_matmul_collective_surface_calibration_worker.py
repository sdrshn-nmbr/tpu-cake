from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tpu_cake.matmul_collective_surface_calibration_worker as worker
from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import array_sha256
from tpu_cake.matmul_collective_surface_calibration_evidence import (
    SurfaceCalibrationResidentPair,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    default_matmul_collective_surface_calibration_protocol,
)
from tpu_cake.matmul_collective_surface_calibration_verifier import (
    _SOURCE_DEPENDENCIES,
)
from tpu_cake.matmul_collective_surface_correctness_evidence import (
    SurfaceCorrectnessSavedArray,
)
from tpu_cake.matmul_collective_surface_prediction import (
    default_matmul_collective_surface_design_contract,
)

PROTOCOL = default_matmul_collective_surface_calibration_protocol()
DESIGN = default_matmul_collective_surface_design_contract()


def _request(tmp_path: Path) -> worker.SurfaceCalibrationWorkerRequest:
    return worker.SurfaceCalibrationWorkerRequest(
        attempt_id="1" * 64,
        invocation_nonce="2" * 64,
        output_root=str(tmp_path),
        parent_correctness_root=str(
            tmp_path / "parent" / PROTOCOL.correctness_parent.archive_root_name
        ),
        compilation_cache_path=str(tmp_path / "cache"),
        protocol_file_sha256="3" * 64,
        design_file_sha256="4" * 64,
        execution_authority_sha256="5" * 64,
        source_commit="6" * 40,
        source_authority_sha256="7" * 64,
        protocol=PROTOCOL,
        design=DESIGN,
    )


class _ReadyOutput:
    def __init__(self, value: np.ndarray, events: list[str], name: str) -> None:
        self.value = value
        self.events = events
        self.name = name

    def block_until_ready(self) -> _ReadyOutput:
        self.events.append(f"ready:{self.name}")
        return self


def _timing_fixture(tmp_path: Path):
    request = _request(tmp_path)
    events: list[str] = []
    compiled = {}
    residents = {}
    for scenario_name in PROTOCOL.scenarios:
        pair = SurfaceCalibrationResidentPair(
            scenario_name=scenario_name,
            xla_compile_record_sha256="8" * 64,
            pallas_compile_record_sha256="9" * 64,
            invocation_nonce=request.invocation_nonce,
            worker_pid=os.getpid(),
        )
        residents[scenario_name] = SimpleNamespace(
            lhs=object(),
            rhs=object(),
            pair=pair,
        )
        for strategy in PROTOCOL.strategies:
            name = f"{scenario_name}:{strategy.value}"

            def executable(_lhs, _rhs, *, name=name):
                events.append(f"call:{name}")
                return _ReadyOutput(np.ones((1, 1), dtype=np.float32), events, name)

            compiled[(scenario_name, strategy)] = SimpleNamespace(executable=executable)
    return request, compiled, residents, events


def test_warmups_execute_exact_balanced_order_and_block_every_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, compiled, residents, events = _timing_fixture(tmp_path)
    clock = iter(range(1, 10_000))
    sentinel_checks = []
    monkeypatch.setattr(worker.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(
        worker,
        "_verify_all_resident_sentinels",
        lambda values: sentinel_checks.append(tuple(values)),
    )

    observed = worker._run_warmups(request, compiled, residents)

    assert len(observed) == 320
    assert tuple(value.sequence for value in observed) == tuple(range(1, 321))
    expected = []
    for scenario_index, scenario_name in enumerate(PROTOCOL.scenarios):
        expected.extend(
            (scenario_name, strategy) for strategy in PROTOCOL.warmup_strategy_order(scenario_index)
        )
    assert tuple((value.scenario_name, value.strategy) for value in observed) == tuple(expected)
    assert events[::2] == [f"call:{name}:{strategy.value}" for name, strategy in expected]
    assert all(value.startswith("ready:") for value in events[1::2])
    assert sentinel_checks == [PROTOCOL.scenarios]


def test_operand_callback_uses_parent_correctness_protocol_for_signed_periodic_sentinels() -> None:
    scenario = DESIGN.calibration_scenarios[0]
    callback = worker._OperandCallback(
        protocol_id=PROTOCOL.correctness_parent.protocol_id,
        scenario=scenario,
        role="lhs",
    )
    local_k = scenario.k // 8

    shard = callback((slice(0, scenario.m), slice(0, local_k)))
    capture = callback.captures[((0, scenario.m), (0, local_k))]

    assert str(shard.dtype) == "bfloat16"
    assert capture.host_callback_payload_nbytes == shard.nbytes
    assert len(capture.sentinel_coordinates) == 32
    assert len(capture.expected_sentinel_hex) == 32


def test_all_32_arms_compile_in_protocol_order_before_the_caller_can_materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    observed = []
    monkeypatch.setattr(worker, "_parent_continuity", lambda *_: object())
    monkeypatch.setattr(worker, "_validate_all_meshes", lambda compiled: None)

    def compile_arm(_root, _request, _parent, scenario, strategy):
        observed.append((scenario.name, strategy))
        return SimpleNamespace(continuity=(scenario.name, strategy))

    monkeypatch.setattr(worker, "_compile_arm", compile_arm)

    compiled, continuity = worker._compile_all_arms(tmp_path, request, SimpleNamespace())

    expected = [
        (scenario_name, strategy)
        for scenario_name in PROTOCOL.scenarios
        for strategy in PROTOCOL.strategies
    ]
    assert observed == expected
    assert list(compiled) == expected
    assert continuity == tuple(expected)


def test_samples_retain_all_2560_positive_durations_in_protocol_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, compiled, residents, events = _timing_fixture(tmp_path)
    clock = iter(range(10_000, 30_000))
    sentinel_checks = []
    monkeypatch.setattr(worker.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(
        worker,
        "_verify_all_resident_sentinels",
        lambda values: sentinel_checks.append(tuple(values)),
    )

    observed = worker._collect_samples(request, compiled, residents)

    assert len(observed) == 2560
    assert tuple(value.sequence for value in observed) == tuple(range(1, 2561))
    assert all(value.duration_ns == 1 for value in observed)
    expected = []
    for round_index in range(PROTOCOL.paired_rounds):
        for scenario_position, scenario_name in enumerate(
            PROTOCOL.scenario_order(round_index), start=1
        ):
            for arm_position, strategy in enumerate(PROTOCOL.strategy_order(round_index), start=1):
                for call_index in range(PROTOCOL.calls_per_position):
                    expected.append(
                        (
                            round_index,
                            scenario_name,
                            scenario_position,
                            strategy,
                            arm_position,
                            call_index,
                        )
                    )
    assert tuple(
        (
            value.round_index,
            value.scenario_name,
            value.scenario_position,
            value.strategy,
            value.arm_position,
            value.call_index,
        )
        for value in observed
    ) == tuple(expected)
    assert len(events) == 5120
    assert sentinel_checks == [PROTOCOL.scenarios]


def test_timing_rejects_a_nonpositive_clock_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, compiled, residents, _ = _timing_fixture(tmp_path)
    monkeypatch.setattr(worker.time, "perf_counter_ns", lambda: 100)

    with pytest.raises(ValueError, match="CLOCK_NONPOSITIVE"):
        worker._collect_samples(request, compiled, residents)


def test_output_gates_bind_full_candidate_to_parent_hash_and_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, compiled, residents, _ = _timing_fixture(tmp_path)
    candidate = np.ones((1, 1), dtype=np.float32)
    candidate_sha256 = array_sha256(candidate)
    scenario = DESIGN.calibration_scenarios[0]
    pair = residents[PROTOCOL.scenarios[0]].pair
    for scenario_name in PROTOCOL.scenarios:
        residents[scenario_name] = SimpleNamespace(
            lhs=object(),
            rhs=object(),
            pair=pair.model_copy(update={"scenario_name": scenario_name}),
            scenario=scenario.model_copy(update={"name": scenario_name}),
            oracle=candidate,
            timing_input=SimpleNamespace(
                parent_xla_array_sha256=candidate_sha256,
                parent_pallas_array_sha256=candidate_sha256,
                oracle=SimpleNamespace(array_sha256=candidate_sha256),
            ),
        )
    clock = iter(range(100, 1000))
    saved_paths = []
    monkeypatch.setattr(worker.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(worker, "_validate_output", lambda *_: None)
    monkeypatch.setattr(worker.jax, "device_get", lambda value: value.value)
    monkeypatch.setattr(worker, "_verify_all_resident_sentinels", lambda *_: None)

    def save(_root, path, value):
        saved_paths.append(path)
        return SurfaceCorrectnessSavedArray(
            path=path,
            file_sha256="a" * 64,
            array_sha256=array_sha256(value),
            shape=tuple(value.shape),
        )

    monkeypatch.setattr(worker, "_save_array_exclusive", save)

    gates = worker._run_output_gates(tmp_path, request, compiled, residents, "before_timing")

    assert len(gates) == 32
    assert saved_paths == [
        f"outputs/{scenario_name}/{strategy.value}-before_timing.npy"
        for scenario_name in PROTOCOL.scenarios
        for strategy in PROTOCOL.strategies
    ]
    assert all(value.mismatched_element_count == 0 for value in gates)


def test_output_gate_rejects_parent_candidate_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, compiled, residents, _ = _timing_fixture(tmp_path)
    candidate = np.ones((1, 1), dtype=np.float32)
    scenario = DESIGN.calibration_scenarios[0]
    for scenario_name in PROTOCOL.scenarios:
        residents[scenario_name] = SimpleNamespace(
            lhs=object(),
            rhs=object(),
            pair=residents[scenario_name].pair,
            scenario=scenario.model_copy(update={"name": scenario_name}),
            oracle=candidate,
            timing_input=SimpleNamespace(
                parent_xla_array_sha256="b" * 64,
                parent_pallas_array_sha256="b" * 64,
                oracle=SimpleNamespace(array_sha256=array_sha256(candidate)),
            ),
        )
    clock = iter(range(100, 1000))
    monkeypatch.setattr(worker.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(worker, "_validate_output", lambda *_: None)
    monkeypatch.setattr(worker.jax, "device_get", lambda value: value.value)
    monkeypatch.setattr(
        worker,
        "_save_array_exclusive",
        lambda _root, path, value: SurfaceCorrectnessSavedArray(
            path=path,
            file_sha256="c" * 64,
            array_sha256=array_sha256(value),
            shape=tuple(value.shape),
        ),
    )

    with pytest.raises(ValueError, match="OUTPUT_PARENT_MISMATCH"):
        worker._run_output_gates(tmp_path, request, compiled, residents, "before_timing")


def test_parent_case_requires_two_identical_outputs_per_strategy_and_oracle() -> None:
    array_hash = "a" * 64
    case = SimpleNamespace(
        input=SimpleNamespace(scenario_name="calibration-0", pattern="signed-periodic"),
        oracle=SimpleNamespace(array_sha256=array_hash),
        executions=tuple(
            SimpleNamespace(
                strategy=strategy,
                output=SimpleNamespace(array_sha256=array_hash),
            )
            for strategy in PROTOCOL.strategies
            for _ in range(2)
        ),
    )
    evidence = SimpleNamespace(cases=(case,))

    observed, hashes = worker._parent_case(evidence, "calibration-0", PROTOCOL.strategies)

    assert observed is case
    assert hashes == {strategy: array_hash for strategy in PROTOCOL.strategies}

    broken = SimpleNamespace(
        cases=(
            SimpleNamespace(
                input=case.input,
                oracle=case.oracle,
                executions=(*case.executions[:-1], case.executions[-1]),
            ),
        )
    )
    broken.cases[0].executions[-1].output.array_sha256 = "b" * 64
    with pytest.raises(ValueError, match="PARENT_OUTPUT_MISMATCH"):
        worker._parent_case(broken, "calibration-0", PROTOCOL.strategies)


def test_empty_cache_is_request_and_environment_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    cache = Path(request.compilation_cache_path)
    cache.mkdir()
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache))

    assert worker._validate_empty_compilation_cache(request) == cache

    (cache / "reused").write_text("compiled")
    with pytest.raises(ValueError, match="CACHE_AUTHORITY_MISMATCH"):
        worker._validate_empty_compilation_cache(request)


def test_worker_authorization_rejects_substituted_request_before_claim_inspection(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    canonical = tmp_path / "worker-request.json"
    substituted = tmp_path / "substituted.json"
    canonical.write_text("{}")
    substituted.write_text("{}")
    canonical.chmod(0o600)
    substituted.chmod(0o600)

    with pytest.raises(ValueError, match="REQUEST_PATH_MISMATCH"):
        worker._validate_worker_authorization(tmp_path, substituted, _request(tmp_path))


def test_control_files_must_be_private_owned_regular_single_links(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    control.write_text("{}")
    control.chmod(0o600)
    worker._validate_control_file(control)

    control.chmod(0o640)
    with pytest.raises(ValueError, match="CONTROL_FILE_INVALID"):
        worker._validate_control_file(control)


def test_parent_root_is_exactly_the_single_extracted_archive_tree(tmp_path: Path) -> None:
    expected = tmp_path / "parent" / PROTOCOL.correctness_parent.archive_root_name

    assert worker._expected_parent_correctness_root(tmp_path, PROTOCOL) == expected
    assert _request(tmp_path).parent_correctness_root == str(expected)


def test_worker_request_cannot_authorize_a_holdout_scenario(tmp_path: Path) -> None:
    payload = _request(tmp_path).model_dump(mode="python", exclude_computed_fields=True)
    payload["protocol"] = {
        **payload["protocol"],
        "scenarios": (*PROTOCOL.scenarios[:-1], "holdout-0"),
    }

    with pytest.raises(ValueError, match="PROTOCOL_MISMATCH"):
        worker.SurfaceCalibrationWorkerRequest.model_validate(payload)


def _source_authority(runtime: dict[str, str]) -> worker.SurfaceCalibrationSourceAuthority:
    return worker.SurfaceCalibrationSourceAuthority(
        source_commit="d" * 40,
        origin_main_commit="d" * 40,
        remote_main_commit="d" * 40,
        runtime=runtime,
        uv_lock_sha256="e" * 64,
        dependencies=tuple(
            SourceFileContract(path=path, sha256="f" * 64)
            for path in worker.CALIBRATION_EXECUTABLE_DEPENDENCIES
        ),
    )


def test_source_authority_accepts_design_runtime_plus_numpy_and_ml_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {**DESIGN.runtime, "numpy": "2.5.2", "ml_dtypes": "0.6.0"}
    authority = _source_authority(runtime)
    blobs = {
        **{path: b"source" for path in worker.CALIBRATION_EXECUTABLE_DEPENDENCIES},
        "uv.lock": b"lock",
    }
    authority = authority.model_copy(
        update={
            "uv_lock_sha256": hashlib.sha256(blobs["uv.lock"]).hexdigest(),
            "dependencies": tuple(
                SourceFileContract(path=path, sha256=hashlib.sha256(blob).hexdigest())
                for path, blob in blobs.items()
                if path != "uv.lock"
            ),
        }
    )
    monkeypatch.setattr(worker, "_calibration_runtime_identity", lambda: runtime)
    monkeypatch.setattr(worker, "_read_committed_calibration_blobs", lambda *_: blobs)

    worker.validate_calibration_source_authority(authority, DESIGN, blobs)

    broken = authority.model_copy(update={"runtime": dict(DESIGN.runtime)})
    monkeypatch.setattr(worker, "_calibration_runtime_identity", lambda: dict(DESIGN.runtime))
    with pytest.raises(ValueError, match="SOURCE_AUTHORITY_MISMATCH"):
        worker.validate_calibration_source_authority(broken, DESIGN, blobs)


def test_source_closure_includes_archive_executor_worker_and_verifier() -> None:
    assert {
        "tpu_cake/matmul_collective_surface_calibration_archive.py",
        "tpu_cake/matmul_collective_surface_calibration_executor.py",
        "tpu_cake/matmul_collective_surface_calibration_worker.py",
        "tpu_cake/matmul_collective_surface_calibration_verifier.py",
    } <= set(worker.CALIBRATION_EXECUTABLE_DEPENDENCIES)
    assert worker.CALIBRATION_EXECUTABLE_DEPENDENCIES == _SOURCE_DEPENDENCIES


def test_source_checkout_requires_main_and_no_untracked_files() -> None:
    worker._validate_source_checkout("main", "")
    with pytest.raises(ValueError, match="SOURCE_BRANCH_MISMATCH"):
        worker._validate_source_checkout("", "")
    with pytest.raises(ValueError, match="SOURCE_DIRTY"):
        worker._validate_source_checkout("main", "?? jax.py")


def test_parent_loader_requires_sqlite_ledger_and_computed_receipt_identity() -> None:
    required = worker._parent_required_file_hashes(PROTOCOL)

    assert required["ledger.sqlite"] == PROTOCOL.correctness_parent.ledger_file_sha256
    assert "ledger.json" not in required

    valid = SimpleNamespace(
        evidence_sha256=PROTOCOL.correctness_parent.evidence_sha256,
        protocol_id=PROTOCOL.correctness_parent.protocol_id,
        split=SimpleNamespace(value="calibration"),
    )
    manifest = SimpleNamespace(evidence_sha256=PROTOCOL.correctness_parent.evidence_sha256)
    receipt = SimpleNamespace(
        receipt_sha256=PROTOCOL.correctness_parent.receipt_sha256,
        attempt_id=PROTOCOL.correctness_parent.attempt_id,
        evidence_sha256=PROTOCOL.correctness_parent.evidence_sha256,
    )
    identity = {"attempt_id": PROTOCOL.correctness_parent.attempt_id}

    worker._validate_parent_models(valid, receipt, manifest, identity, PROTOCOL)

    with pytest.raises(ValueError, match="PARENT_AUTHORITY_MISMATCH"):
        worker._validate_parent_models(
            valid,
            SimpleNamespace(
                receipt_sha256="0" * 64,
                attempt_id=receipt.attempt_id,
                evidence_sha256=receipt.evidence_sha256,
            ),
            manifest,
            identity,
            PROTOCOL,
        )


def test_attempt_claim_payload_is_semantically_bound_and_canonical(tmp_path: Path) -> None:
    request = _request(tmp_path)
    claim = worker.SurfaceCalibrationAttemptClaim(
        attempt_id=request.attempt_id,
        protocol_id=request.protocol.protocol_id,
        permanent_claim_key=request.protocol.permanent_claim_key,
        correctness_parent_receipt_sha256=request.protocol.correctness_parent.receipt_sha256,
        source_commit=request.source_commit,
        output_root=request.output_root,
    )
    payload = worker._json_bytes(claim.model_dump(mode="json", exclude_computed_fields=True))

    assert json.loads(payload) == {
        "attempt_id": request.attempt_id,
        "correctness_parent_receipt_sha256": request.protocol.correctness_parent.receipt_sha256,
        "output_root": request.output_root,
        "permanent_claim_key": request.protocol.permanent_claim_key,
        "protocol_id": request.protocol.protocol_id,
        "schema_version": worker.CALIBRATION_ATTEMPT_CLAIM_SCHEMA,
        "source_commit": request.source_commit,
        "state": "claimed",
    }
