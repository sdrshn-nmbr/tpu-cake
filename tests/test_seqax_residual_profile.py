from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import tpu_cake.seqax_residual_profile as profile_model
import tpu_cake.seqax_residual_profile_runner as profile_runner
from tpu_cake.cli import _parser
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import array_sha256, arrays_sha256
from tpu_cake.runner import RunMode, _runtime_identity
from tpu_cake.seqax_numerical import (
    _assess_output_arrays,
    default_seqax_bf16_validation_contract,
)
from tpu_cake.seqax_pallas_diagnostic import SeqaxPallasDiagnosticAttribution
from tpu_cake.seqax_residual_profile import (
    SeqaxResidualCandidateResult,
    SeqaxResidualCorrectnessObservation,
    SeqaxResidualProfileCapture,
    SeqaxResidualProfileSummary,
    default_seqax_residual_profile_contract,
)
from tpu_cake.seqax_residual_profile_runner import (
    CompiledResidualProfile,
    _json_sha256,
    _prepare_candidates,
    _profile_summary,
    _require_safe_new_root,
    run_seqax_residual_profile,
    validate_seqax_residual_profile,
)
from tpu_cake.seqax_runner import expected_seqax_profiler_contract
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


def test_seqax_residual_profile_external_contract_matches_factory() -> None:
    saved = type(default_seqax_residual_profile_contract(_runtime_identity())).model_validate_json(
        Path("contracts/seqax-residual-all-reduce-profile-v1.json").read_text()
    )
    expected = default_seqax_residual_profile_contract(saved.runtime)

    assert saved == expected
    assert saved.hlo_identity_status == "pending"
    assert all(
        value == "0" * 64
        for candidate in saved.candidates
        for value in (
            candidate.pallas_stablehlo_sha256,
            candidate.pallas_compiler_hlo_sha256,
            candidate.control_stablehlo_sha256,
            candidate.control_compiler_hlo_sha256,
        )
    )


def test_seqax_residual_profile_pending_contract_refuses_before_writes(tmp_path: Path) -> None:
    contract = default_seqax_residual_profile_contract(_runtime_identity())
    root = tmp_path / "run"

    with pytest.raises(ValueError, match="HLO_IDENTITIES_PENDING"):
        run_seqax_residual_profile(root, contract)

    assert not root.exists()


def test_seqax_residual_profile_rejects_a_symlinked_output_ancestor(tmp_path: Path) -> None:
    contract = default_seqax_residual_profile_contract(_runtime_identity())
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="PATH_SYMLINK"):
        run_seqax_residual_profile(alias / "run", contract)

    assert tuple(outside.iterdir()) == ()


def test_seqax_residual_profile_plan_contract_binds_both_schedules() -> None:
    contract = default_seqax_residual_profile_contract(_runtime_identity())
    prepared = _prepare_candidates(contract)

    assert tuple(value.expected.candidate for value in prepared) == (
        SeqaxResidualNormStrategy.STANDARD,
        SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
    )
    assert tuple(value.plan.pallas_region_count for value in prepared) == (9, 9)
    assert tuple(value.expected.expected_all_reduces for value in prepared) == (0, 2)
    assert tuple(value.expected.expected_semantic_all_reduce_rows for value in prepared) == (5, 5)
    assert all(
        value.expected.pallas_manifest_sha256 == _json_sha256(value.plan.manifest())
        for value in prepared
    )


def test_seqax_residual_profile_input_identities_are_per_array() -> None:
    correctness_schema = SeqaxResidualCorrectnessObservation.model_json_schema()
    candidate_schema = SeqaxResidualCandidateResult.model_json_schema()

    assert correctness_schema["properties"]["input_sha256"]["type"] == "array"
    assert candidate_schema["properties"]["timing_input_sha256"]["type"] == "array"


def test_seqax_residual_profile_observed_inventory_distinguishes_sync_all_reduce() -> None:
    summary = SeqaxResidualProfileSummary(
        candidate=SeqaxResidualNormStrategy.STANDARD,
        mode=RunMode.TRACE,
        module_execution_count=50,
        module_median_duration_ns=100.0,
        module_p90_duration_ns=110.0,
        pallas_average_self_time_sum_ns_per_device=1.0,
        collective_completion_average_self_time_sum_ns_per_device=20.0,
        all_reduce_average_self_time_sum_ns_per_device=5.0,
        semantic_all_gather_rows=8,
        semantic_all_reduce_rows=5,
        semantic_reduce_scatter_rows=3,
        async_collective_completion_rows=11,
        static_all_gathers=17,
        static_all_reduces=0,
        static_reduce_scatters=3,
        pallas_regions=9,
        ring_equivalent_ici_bytes_per_device=34_048,
    )

    assert summary.semantic_all_reduce_rows == 5

    payload = summary.model_dump(mode="json")
    payload["async_collective_completion_rows"] = 16
    with pytest.raises(ValidationError, match="observed collective inventory mismatch"):
        SeqaxResidualProfileSummary.model_validate_json(json.dumps(payload))


def test_seqax_residual_profile_rejects_incomplete_all_reduce_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_seqax_residual_profile_contract(_runtime_identity())
    expected = contract.candidates[0]
    rows = [
        {"program_id": "7", "category": "all-gather"}
        for _ in range(expected.expected_semantic_all_gather_rows)
    ]
    rows += [
        {
            "program_id": "7",
            "category": "all-reduce",
            "occurrences": 1,
            "avg_self_time": 0.0,
        }
        for _ in range(expected.expected_semantic_all_reduce_rows)
    ]
    rows += [
        {"program_id": "7", "category": "reduce-scatter"}
        for _ in range(expected.expected_semantic_reduce_scatter_rows)
    ]
    rows += [
        {
            "program_id": "7",
            "category": "async-done",
            "hlo_op_name": "all-gather.call-done",
        }
        for _ in range(expected.expected_async_collective_completion_rows)
    ]
    monkeypatch.setattr(profile_runner, "_gviz_rows", lambda _path: tuple(rows))
    attribution = SeqaxPallasDiagnosticAttribution(
        program_id="7",
        module_execution_count=50,
        module_median_duration_ns=100.0,
        module_p90_duration_ns=110.0,
        pallas_average_self_time_sum_ns_per_device=1.0,
        collective_completion_average_self_time_sum_ns_per_device=2.0,
        cost_model_idealized_floor_ns=1.0,
        cost_model_materialized_hbm_floor_ns=1.0,
        module_to_idealized_floor_ratio=100.0,
        regions=(),
        categories=(),
        interpretation=(),
    )

    with pytest.raises(ValueError, match="ALL_REDUCE_EVIDENCE_MISMATCH"):
        _profile_summary(
            expected=expected,
            mode=RunMode.TRACE,
            attribution=attribution,
            hlo_stats=tmp_path / "hlo_stats.json",
        )


def test_seqax_residual_profile_rejects_output_inside_repository() -> None:
    root = Path(__file__).resolve().parents[1] / ".seqax-residual-profile-test-output"

    with pytest.raises(ValueError, match="UNSAFE_ROOT"):
        _require_safe_new_root(root)

    assert not root.exists()


def test_seqax_residual_profile_contract_rejects_a_caller_pinned_placeholder() -> None:
    contract = default_seqax_residual_profile_contract(_runtime_identity())
    payload = contract.model_dump(mode="json", exclude_computed_fields=True)
    payload["hlo_identity_status"] = "pinned"

    with pytest.raises(ValidationError, match="Pinned Seqax residual HLO identities"):
        type(contract).model_validate_json(json.dumps(payload))


def test_seqax_residual_profile_contract_is_canonical_json() -> None:
    expected = json.loads(Path("contracts/seqax-residual-all-reduce-profile-v1.json").read_text())
    saved = default_seqax_residual_profile_contract(
        RuntimeIdentity.model_validate(expected["runtime"])
    )

    assert saved.model_dump(mode="json", exclude_computed_fields=True) == expected


def test_seqax_residual_profile_commands_are_available() -> None:
    capture = _parser().parse_args(
        (
            "capture-seqax-residual-profile-hlo",
            "--contract",
            "contract.json",
        )
    )
    run = _parser().parse_args(
        (
            "run-seqax-residual-profile",
            "--contract",
            "contract.json",
            "--output-dir",
            "run",
        )
    )
    verify = _parser().parse_args(
        (
            "verify-seqax-residual-profile",
            "run",
            "--contract",
            "contract.json",
        )
    )

    assert capture.command == "capture-seqax-residual-profile-hlo"
    assert run.command == "run-seqax-residual-profile"
    assert verify.command == "verify-seqax-residual-profile"


def _repair_residual_receipt(root: Path, *relative_paths: str) -> None:
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in relative_paths:
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = next(value for value in receipt["artifacts"] if value["path"] == relative)
        artifact["size_bytes"] = path.stat().st_size
        artifact["sha256"] = digest
        if relative == "result.json":
            receipt["result_sha256"] = digest
        if relative == "ledger.sqlite":
            receipt["ledger_sha256"] = digest
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_seqax_residual_profile_runner_builds_and_replays_a_closed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pending = default_seqax_residual_profile_contract(_runtime_identity())
    hlo_text: dict[SeqaxResidualNormStrategy, tuple[str, str, str, str]] = {}
    candidates = []
    for candidate in pending.candidates:
        values = tuple(
            f"{candidate.candidate.value}-{name}\n"
            for name in (
                "pallas-stablehlo",
                "pallas-compiler-hlo",
                "control-stablehlo",
                "control-compiler-hlo",
            )
        )
        hlo_text[candidate.candidate] = values
        candidates.append(
            candidate.model_copy(
                update={
                    "pallas_stablehlo_sha256": hashlib.sha256(values[0].encode()).hexdigest(),
                    "pallas_compiler_hlo_sha256": hashlib.sha256(values[1].encode()).hexdigest(),
                    "control_stablehlo_sha256": hashlib.sha256(values[2].encode()).hexdigest(),
                    "control_compiler_hlo_sha256": hashlib.sha256(values[3].encode()).hexdigest(),
                }
            )
        )
    contract = pending.model_copy(
        update={"hlo_identity_status": "pinned", "candidates": tuple(candidates)}
    )
    numerical = default_seqax_bf16_validation_contract()
    scenario = next(
        value for value in numerical.scenarios if value.name == "calibration-m256-b2-s1-l1"
    )
    devices = tuple(
        SimpleNamespace(id=index, process_index=0, platform="tpu", device_kind="TPU7x")
        for index in range(8)
    )

    def source_state(_repository: Path, output: Path):
        (output / "source_diff.patch").write_bytes(b"")
        profile_runner._write_json(
            output / "source_state.json",
            {
                "git_commit": "0" * 40,
                "git_dirty": False,
                "git_status": [],
                "source_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "uv_lock_sha256": "0" * 64,
            },
        )
        return ()

    def validate_source(root: Path, result) -> None:
        if (
            result.source_state_sha256 != profile_runner._sha256(root / "source_state.json")
            or result.source_manifest_sha256
            != profile_runner._sha256(root / "source_manifest.json")
            or result.source_manifest != profile_runner._source_manifest()
            or (root / "source_diff.patch").read_bytes() != b""
        ):
            raise ValueError("SEQAX_RESIDUAL_PROFILE_SOURCE_MISMATCH")

    def compile_candidate(value, _inputs, _devices):
        pallas_stable, pallas_compiler, control_stable, control_compiler = hlo_text[
            value.expected.candidate
        ]
        return CompiledResidualProfile(
            prepared=value,
            pallas_executable=value.expected.candidate,
            control_executable=value.expected.candidate,
            mesh=None,
            pallas_stablehlo=pallas_stable,
            pallas_compiler_hlo=pallas_compiler,
            control_stablehlo=control_stable,
            control_compiler_hlo=control_compiler,
        )

    def correctness_observation(*, root, compiled, host_inputs, seed):
        output = np.zeros(scenario.output.shape, dtype=np.float32)
        assessment = _assess_output_arrays(
            output,
            output,
            output,
            policy=numerical.policy,
            scenario=scenario,
        )
        profile_runner._save_inputs(root, seed, host_inputs, scenario)
        seed_root = root / "correctness" / str(seed)
        profile_runner._save_array(seed_root / "cpu.npy", output)
        profile_runner._save_array(seed_root / "control.npy", output)
        profile_runner._save_array(seed_root / "pallas.npy", output)
        return SeqaxResidualCorrectnessObservation(
            candidate=compiled.prepared.expected.candidate,
            seed=seed,
            input_sha256=arrays_sha256(host_inputs),
            cpu_output_sha256=array_sha256(output),
            control_output_sha256=array_sha256(output),
            pallas_output_sha256=array_sha256(output),
            assessment=assessment,
        )

    def replay_correctness(*, root, prepared, saved) -> None:
        replayed = []
        for observation in saved:
            inputs = profile_runner._load_inputs(
                root / "candidates" / prepared.expected.candidate, observation.seed, scenario
            )
            expected_inputs = tuple(
                np.asarray(value)
                for value in seqax_forward_inputs(
                    seed=observation.seed,
                    **scenario.parameters.model_dump(),
                )
            )
            if any(
                not np.array_equal(actual, expected)
                for actual, expected in zip(inputs, expected_inputs, strict=True)
            ):
                raise ValueError("SEQAX_RESIDUAL_PROFILE_INPUT_REPLAY_MISMATCH")
            seed_root = (
                root
                / "candidates"
                / prepared.expected.candidate
                / "correctness"
                / str(observation.seed)
            )
            cpu = profile_runner._load_array(seed_root / "cpu.npy")
            control = profile_runner._load_array(seed_root / "control.npy")
            pallas = profile_runner._load_array(seed_root / "pallas.npy")
            replayed.append(
                SeqaxResidualCorrectnessObservation(
                    candidate=prepared.expected.candidate,
                    seed=observation.seed,
                    input_sha256=arrays_sha256(inputs),
                    cpu_output_sha256=array_sha256(cpu),
                    control_output_sha256=array_sha256(control),
                    pallas_output_sha256=array_sha256(pallas),
                    assessment=_assess_output_arrays(
                        pallas,
                        control,
                        cpu,
                        policy=numerical.policy,
                        scenario=scenario,
                    ),
                )
            )
        if tuple(replayed) != saved:
            raise ValueError("SEQAX_RESIDUAL_PROFILE_CORRECTNESS_REPLAY_MISMATCH")

    def capture_record(candidate_root, expected, mode, *, write):
        phase_root = candidate_root / mode.value
        config = expected_seqax_profiler_contract(mode)
        assessment = {"candidate": expected.candidate.value, "mode": mode.value, "accepted": True}
        attribution = {"candidate": expected.candidate.value, "mode": mode.value, "rows": 1}
        xplane = phase_root / "profile" / "plugins" / "profile" / "capture.xplane.pb"
        if write:
            profile_runner._write_json(phase_root / "profiler_config.json", config)
            profile_runner._write_json(phase_root / "profile_assessment.json", assessment)
            profile_runner._write_json(phase_root / "attribution.json", attribution)
            xplane.parent.mkdir(parents=True, exist_ok=True)
            xplane.write_bytes(f"{expected.candidate.value}:{mode.value}".encode())
        elif (
            json.loads((phase_root / "profiler_config.json").read_text()) != config
            or json.loads((phase_root / "profile_assessment.json").read_text()) != assessment
            or json.loads((phase_root / "attribution.json").read_text()) != attribution
            or xplane.read_bytes() != f"{expected.candidate.value}:{mode.value}".encode()
        ):
            raise ValueError("SEQAX_RESIDUAL_PROFILE_CAPTURE_REPLAY_MISMATCH")
        candidate_is_residual = expected.candidate is SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE
        median = 90.0 if candidate_is_residual else 100.0
        if mode is RunMode.COUNTERS:
            median += 5.0
        summary = SeqaxResidualProfileSummary(
            candidate=expected.candidate,
            mode=mode,
            module_execution_count=50,
            module_median_duration_ns=median,
            module_p90_duration_ns=median + 10.0,
            pallas_average_self_time_sum_ns_per_device=1.0,
            collective_completion_average_self_time_sum_ns_per_device=(
                10.0 if candidate_is_residual else 20.0
            ),
            all_reduce_average_self_time_sum_ns_per_device=(6.0 if candidate_is_residual else 5.0),
            semantic_all_gather_rows=expected.expected_semantic_all_gather_rows,
            semantic_all_reduce_rows=expected.expected_semantic_all_reduce_rows,
            semantic_reduce_scatter_rows=expected.expected_semantic_reduce_scatter_rows,
            async_collective_completion_rows=expected.expected_async_collective_completion_rows,
            static_all_gathers=expected.expected_all_gathers,
            static_all_reduces=expected.expected_all_reduces,
            static_reduce_scatters=expected.expected_reduce_scatters,
            pallas_regions=expected.expected_pallas_regions,
            ring_equivalent_ici_bytes_per_device=(
                expected.expected_ring_equivalent_ici_bytes_per_device
            ),
        )
        counter_names = ("COUNT_MXU_BUSY_TEST",) if mode is RunMode.COUNTERS else ()
        counter_samples = {str(index): 2 for index in (0, 2, 4, 6)} if counter_names else {}
        return SeqaxResidualProfileCapture(
            candidate=expected.candidate,
            mode=mode,
            step_event=(
                expected.trace_step_event if mode is RunMode.TRACE else expected.counter_step_event
            ),
            profiler_config_sha256=profile_runner._sha256(phase_root / "profiler_config.json"),
            xplane_sha256=profile_runner._sha256(xplane),
            assessment_sha256=profile_runner._sha256(phase_root / "profile_assessment.json"),
            attribution_sha256=profile_runner._sha256(phase_root / "attribution.json"),
            program_id="7",
            summary=summary,
            periodic_counter_names=counter_names,
            periodic_counter_samples_per_core=counter_samples,
            hbm_read_counter_names=1 if counter_names else 0,
            hbm_write_counter_names=1 if counter_names else 0,
            cycle_counter_names=1 if counter_names else 0,
        )

    def capture_phase(*, candidate_root, expected, mode, **_kwargs):
        return capture_record(candidate_root, expected, mode, write=True)

    def replay_profiles(*, root, prepared, **_kwargs):
        candidate_root = root / "candidates" / prepared.expected.candidate
        return tuple(
            capture_record(candidate_root, prepared.expected, mode, write=False)
            for mode in (RunMode.TRACE, RunMode.COUNTERS)
        )

    monkeypatch.setattr(
        profile_runner, "default_seqax_residual_profile_contract", lambda _runtime=None: contract
    )
    monkeypatch.setattr(profile_model, "_candidate_contracts", lambda: contract.candidates)
    monkeypatch.setattr(profile_runner, "_runtime_identity", lambda: contract.runtime)
    monkeypatch.setattr(profile_runner, "_validate_verifier_runtime", lambda: None)
    monkeypatch.setattr(profile_runner, "_require_compilation_root", lambda _root: None)
    monkeypatch.setattr(profile_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(profile_runner, "_source_state", source_state)
    monkeypatch.setattr(profile_runner, "_validate_source", validate_source)
    monkeypatch.setattr(profile_runner.jax, "devices", lambda: devices)
    monkeypatch.setattr(profile_runner, "_validate_devices", lambda _devices, _contract: None)
    monkeypatch.setattr(profile_runner, "_compile", compile_candidate)
    monkeypatch.setattr(
        profile_runner, "_validate_compiled_program", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(profile_runner, "_resident_inputs", lambda inputs, _prepared, _mesh: inputs)
    monkeypatch.setattr(
        profile_runner,
        "_execute",
        lambda _executable, _inputs: np.zeros(scenario.output.shape, dtype=np.float32),
    )
    monkeypatch.setattr(profile_runner, "_correctness_observation", correctness_observation)
    monkeypatch.setattr(profile_runner, "_replay_correctness", replay_correctness)
    monkeypatch.setattr(profile_runner, "_capture_candidate_phase", capture_phase)
    monkeypatch.setattr(profile_runner, "_replay_candidate_profiles", replay_profiles)

    root = tmp_path / "run"
    result = run_seqax_residual_profile(root, contract)
    assert not result.accepted
    assert (root / "receipt.json").is_file()
    assert validate_seqax_residual_profile(root, contract) == result

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert validate_seqax_residual_profile(relocated, contract) == result

    def copy(name: str) -> Path:
        destination = tmp_path / name
        shutil.copytree(root, destination)
        return destination

    changed_hlo = copy("changed-hlo")
    hlo_path = "candidates/standard/pallas_stablehlo.txt"
    (changed_hlo / hlo_path).write_text("forged\n")
    _repair_residual_receipt(changed_hlo, hlo_path)
    with pytest.raises(ValueError, match="PLAN_REPLAY_MISMATCH"):
        validate_seqax_residual_profile(changed_hlo, contract)

    changed_manifest = copy("changed-manifest")
    manifest_path = "candidates/standard/plan_manifest.json"
    (changed_manifest / manifest_path).write_text("{}\n")
    _repair_residual_receipt(changed_manifest, manifest_path)
    with pytest.raises(ValueError, match="PLAN_REPLAY_MISMATCH"):
        validate_seqax_residual_profile(changed_manifest, contract)

    changed_input = copy("changed-input")
    input_path = next(changed_input.glob("candidates/standard/correctness/*/inputs/00.npy"))
    storage = np.load(input_path, allow_pickle=False)
    storage.reshape(-1)[0] ^= np.array(1, dtype=storage.dtype)
    np.save(input_path, storage, allow_pickle=False)
    relative_input = input_path.relative_to(changed_input).as_posix()
    _repair_residual_receipt(changed_input, relative_input)
    with pytest.raises(ValueError, match="INPUT_REPLAY_MISMATCH"):
        validate_seqax_residual_profile(changed_input, contract)

    changed_output = copy("changed-output")
    output_path = f"candidates/standard/correctness/{contract.correctness_seeds[0]}/pallas.npy"
    output = np.load(changed_output / output_path, allow_pickle=False)
    output.reshape(-1)[0] += np.float32(1)
    np.save(changed_output / output_path, output, allow_pickle=False)
    _repair_residual_receipt(changed_output, output_path)
    with pytest.raises(ValueError, match="correctness policy failed|CORRECTNESS_REPLAY_MISMATCH"):
        validate_seqax_residual_profile(changed_output, contract)

    changed_profile = copy("changed-profile")
    profile_path = "candidates/standard/trace/profiler_config.json"
    (changed_profile / profile_path).write_text("{}\n")
    _repair_residual_receipt(changed_profile, profile_path)
    with pytest.raises(ValueError, match="CAPTURE_REPLAY_MISMATCH"):
        validate_seqax_residual_profile(changed_profile, contract)

    changed_xplane = copy("changed-xplane")
    xplane_path = "candidates/standard/trace/profile/plugins/profile/capture.xplane.pb"
    (changed_xplane / xplane_path).write_bytes(b"forged")
    _repair_residual_receipt(changed_xplane, xplane_path)
    with pytest.raises(ValueError, match="CAPTURE_REPLAY_MISMATCH"):
        validate_seqax_residual_profile(changed_xplane, contract)

    changed_attribution = copy("changed-attribution")
    attribution_path = "candidates/standard/trace/attribution.json"
    (changed_attribution / attribution_path).write_text("{}\n")
    _repair_residual_receipt(changed_attribution, attribution_path)
    with pytest.raises(ValueError, match="CAPTURE_REPLAY_MISMATCH"):
        validate_seqax_residual_profile(changed_attribution, contract)

    changed_run = copy("changed-run")
    result_payload = json.loads((changed_run / "result.json").read_text())
    result_payload["run_id"] = "0" * 64
    (changed_run / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_residual_receipt(changed_run, "result.json")
    with pytest.raises(ValueError, match="RUN_ID_MISMATCH"):
        validate_seqax_residual_profile(changed_run, contract)

    changed_device = copy("changed-device")
    result_payload = json.loads((changed_device / "result.json").read_text())
    result_payload["devices"][0]["process_index"] = 1
    (changed_device / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_residual_receipt(changed_device, "result.json")
    with pytest.raises(ValueError, match="RESULT_IDENTITY_MISMATCH"):
        validate_seqax_residual_profile(changed_device, contract)

    changed_ledger = copy("changed-ledger")
    with sqlite3.connect(changed_ledger / "ledger.sqlite") as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = 'countered'",
            ("0" * 64,),
        )
    _repair_residual_receipt(changed_ledger, "ledger.sqlite")
    with pytest.raises(ValueError, match="LEDGER_PAYLOAD_MISMATCH"):
        validate_seqax_residual_profile(changed_ledger, contract)

    extra_artifact = copy("extra-artifact")
    (extra_artifact / "extra.txt").write_text("not declared\n")
    with pytest.raises(ValueError, match="CLOSED_WORLD_MISMATCH"):
        validate_seqax_residual_profile(extra_artifact, contract)

    missing_artifact = copy("missing-artifact")
    (missing_artifact / xplane_path).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_seqax_residual_profile(missing_artifact, contract)
