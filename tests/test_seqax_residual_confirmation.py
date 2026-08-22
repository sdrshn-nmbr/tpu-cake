from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import tpu_cake.seqax_residual_confirmation_runner as confirmation_runner
from tpu_cake.cli import _parser
from tpu_cake.compiler_analysis import capture_compiler_analysis
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import array_sha256, arrays_sha256
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_numerical import (
    _assess_output_arrays,
    default_seqax_bf16_validation_contract,
)
from tpu_cake.seqax_residual_confirmation import (
    SOURCE_PROFILE_ARCHIVE_SHA256,
    SOURCE_PROFILE_RECEIPT_SHA256,
    SeqaxResidualConfirmationContract,
    SeqaxResidualTimingRound,
    confirmation_orders,
    confirmation_statistics,
    default_seqax_residual_confirmation_contract,
)
from tpu_cake.seqax_residual_confirmation_runner import (
    _require_safe_root,
    run_seqax_residual_confirmation,
    validate_seqax_residual_confirmation,
)
from tpu_cake.seqax_residual_profile import SeqaxResidualCorrectnessObservation
from tpu_cake.seqax_residual_profile_runner import (
    CompiledResidualProfile,
    _load_inputs,
    _save_inputs,
)
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


def _compiler_analysis(stablehlo: str, compiler_hlo: str):
    memory = SimpleNamespace(
        generated_code_size_in_bytes=100,
        argument_size_in_bytes=200,
        output_size_in_bytes=80,
        alias_size_in_bytes=0,
        temp_size_in_bytes=40,
        host_generated_code_size_in_bytes=0,
        host_argument_size_in_bytes=0,
        host_output_size_in_bytes=0,
        host_alias_size_in_bytes=0,
        host_temp_size_in_bytes=0,
        peak_memory_in_bytes=320,
        serialized_buffer_assignment_proto=b"residual-confirmation-fixture",
    )
    executable = SimpleNamespace(
        as_text=lambda: compiler_hlo,
        cost_analysis=lambda: {"bytes accessed": 512.0, "flops": 1024.0},
        memory_analysis=lambda: memory,
    )
    return capture_compiler_analysis(
        executable,
        stablehlo=stablehlo.rstrip("\n"),
        compiler_hlo=compiler_hlo.rstrip("\n"),
    )


def _rounds(
    contract: SeqaxResidualConfirmationContract,
    *,
    candidate_ns: int,
) -> tuple[SeqaxResidualTimingRound, ...]:
    observations = []
    for round_index, order in enumerate(confirmation_orders(contract)):
        for position, candidate in enumerate(order):
            duration = 100 if candidate is contract.baseline else candidate_ns
            samples = (duration,) * contract.measured_iterations
            observations.append(
                SeqaxResidualTimingRound(
                    round_index=round_index,
                    position=position,
                    candidate=candidate,
                    samples_ns=samples,
                    median_ns=float(duration),
                )
            )
    return tuple(observations)


def test_seqax_residual_confirmation_contract_is_canonical_json() -> None:
    payload = json.loads(
        Path("contracts/seqax-residual-all-reduce-confirmation-v1.json").read_text()
    )
    saved = SeqaxResidualConfirmationContract.model_validate_json(json.dumps(payload))
    expected = default_seqax_residual_confirmation_contract(
        RuntimeIdentity.model_validate(payload["runtime"])
    )

    assert saved == expected
    assert saved.source_profile_archive_sha256 == SOURCE_PROFILE_ARCHIVE_SHA256
    assert saved.source_profile_receipt_sha256 == SOURCE_PROFILE_RECEIPT_SHA256
    assert (
        saved.confirmation_id == "9cebd837c7ee307766867ef97f671d9d6807093c48c34ec7eda9d571912f5b77"
    )
    assert saved.analysis_index == 3
    assert not saved.allow_early_stopping
    assert not saved.allow_further_retry


def test_seqax_residual_confirmation_statistics_require_the_fixed_gate() -> None:
    contract = default_seqax_residual_confirmation_contract(_runtime_identity())

    confirmed = confirmation_statistics(contract, _rounds(contract, candidate_ns=90))
    retained = confirmation_statistics(contract, _rounds(contract, candidate_ns=98))

    assert confirmed.confirmed
    assert confirmed.improvement_confidence_interval == pytest.approx((0.1, 0.1))
    assert not retained.confirmed
    assert retained.improvement_confidence_interval == pytest.approx((0.02, 0.02))


def test_seqax_residual_confirmation_statistics_reject_reordered_evidence() -> None:
    contract = default_seqax_residual_confirmation_contract(_runtime_identity())
    rounds = list(_rounds(contract, candidate_ns=90))
    rounds[0], rounds[1] = rounds[1], rounds[0]

    with pytest.raises(ValueError, match="execution order mismatch"):
        confirmation_statistics(contract, tuple(rounds))


def test_seqax_residual_confirmation_contract_rejects_changed_profile_authority() -> None:
    contract = default_seqax_residual_confirmation_contract(_runtime_identity())
    payload = contract.model_dump(mode="json", exclude_computed_fields=True)
    payload["source_profile_receipt_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="profile provenance mismatch"):
        SeqaxResidualConfirmationContract.model_validate_json(json.dumps(payload))


def test_seqax_residual_confirmation_rejects_output_inside_repository() -> None:
    root = Path(__file__).resolve().parents[1] / ".seqax-residual-confirmation-output"

    with pytest.raises(ValueError, match="UNSAFE_ROOT"):
        _require_safe_root(root)

    assert not root.exists()


def test_seqax_residual_confirmation_commands_are_available() -> None:
    run = _parser().parse_args(
        (
            "run-seqax-residual-confirmation",
            "--contract",
            "contract.json",
            "--output-dir",
            "run",
        )
    )
    verify = _parser().parse_args(
        (
            "verify-seqax-residual-confirmation",
            "run",
            "--contract",
            "contract.json",
        )
    )

    assert run.command == "run-seqax-residual-confirmation"
    assert verify.command == "verify-seqax-residual-confirmation"


def test_seqax_residual_confirmation_atomic_markers_do_not_dirty_the_run_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated crash before atomic publication")

    monkeypatch.setattr(confirmation_runner.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        confirmation_runner._write_json_atomic(
            root / "run_identity.json",
            {"identity": "test"},
        )

    assert not any(root.iterdir())


def test_seqax_residual_confirmation_sibling_failure_stages_on_same_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[Path] = []

    def fail_open(path: Path, *_args: object) -> int:
        opened.append(Path(path))
        raise OSError("stop after resolving temporary path")

    monkeypatch.setattr(confirmation_runner.os, "open", fail_open)
    failure_path = tmp_path / "run.failure.json"
    with pytest.raises(OSError, match="temporary path"):
        confirmation_runner._write_sibling_json_atomic(failure_path, {"failure": "test"})

    assert opened[0].parent == failure_path.parent


def _repair_confirmation_receipt(root: Path, *relative_paths: str) -> None:
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


def test_seqax_residual_confirmation_runner_builds_and_replays_a_closed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_seqax_residual_confirmation_contract(_runtime_identity())
    numerical = default_seqax_bf16_validation_contract()
    scenario = next(
        value for value in numerical.scenarios if value.name == "calibration-m256-b2-s1-l1"
    )
    devices = tuple(
        SimpleNamespace(id=index, process_index=0, platform="tpu", device_kind="TPU7x")
        for index in range(8)
    )
    hlo = {
        candidate.candidate: (
            f"{candidate.candidate.value}-pallas-stable\n",
            f"{candidate.candidate.value}-pallas-compiler\n",
            f"{candidate.candidate.value}-control-stable\n",
            f"{candidate.candidate.value}-control-compiler\n",
        )
        for candidate in contract.plans
    }

    def source_state(repository: Path, output: Path):
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (output / "source_diff.patch").write_bytes(b"")
        confirmation_runner._write_json(
            output / "source_state.json",
            {
                "git_commit": commit,
                "git_dirty": False,
                "git_status": [],
                "source_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "uv_lock_sha256": "0" * 64,
            },
        )
        return ()

    def validate_source(root: Path, result) -> None:
        if (
            result.source_state_sha256 != confirmation_runner._sha256(root / "source_state.json")
            or result.source_manifest_sha256
            != confirmation_runner._sha256(root / "source_manifest.json")
            or result.source_manifest != confirmation_runner._source_manifest()
            or (root / "source_diff.patch").read_bytes() != b""
        ):
            raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_SOURCE_MISMATCH")

    def compile_candidate(value, _inputs, _devices):
        pallas_stable, pallas_compiler, control_stable, control_compiler = hlo[
            value.expected.candidate
        ]

        def executable(*_args):
            return (np.zeros(scenario.output.shape, dtype=np.float32),)

        return CompiledResidualProfile(
            prepared=value,
            pallas_executable=executable,
            control_executable=executable,
            mesh=None,
            pallas_stablehlo=pallas_stable,
            pallas_compiler_hlo=pallas_compiler,
            control_stablehlo=control_stable,
            control_compiler_hlo=control_compiler,
            pallas_compiler_analysis=_compiler_analysis(pallas_stable, pallas_compiler),
            control_compiler_analysis=_compiler_analysis(control_stable, control_compiler),
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
        _save_inputs(root, seed, host_inputs, scenario)
        seed_root = root / "correctness" / str(seed)
        for name in ("cpu.npy", "control.npy", "pallas.npy"):
            confirmation_runner._save_array(seed_root / name, output)
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
        candidate_root = root / "candidates" / prepared.expected.candidate
        for observation in saved:
            inputs = _load_inputs(candidate_root, observation.seed, scenario)
            expected = tuple(
                np.asarray(value)
                for value in seqax_forward_inputs(
                    seed=observation.seed,
                    **scenario.parameters.model_dump(),
                )
            )
            if any(
                not np.array_equal(actual, wanted)
                for actual, wanted in zip(inputs, expected, strict=True)
            ):
                raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_INPUT_REPLAY_MISMATCH")
            seed_root = candidate_root / "correctness" / str(observation.seed)
            cpu = confirmation_runner._load_array(seed_root / "cpu.npy")
            control = confirmation_runner._load_array(seed_root / "control.npy")
            pallas = confirmation_runner._load_array(seed_root / "pallas.npy")
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
            raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_CORRECTNESS_REPLAY_MISMATCH")

    def expected_plan_files(root: Path, prepared) -> None:
        candidate_root = root / "candidates" / prepared.expected.candidate
        expected_hlo = hlo[prepared.expected.candidate]
        observed_hlo = tuple(
            (candidate_root / name).read_text()
            for name in (
                "pallas_stablehlo.txt",
                "pallas_compiler_hlo.txt",
                "control_stablehlo.txt",
                "control_compiler_hlo.txt",
            )
        )
        if observed_hlo != expected_hlo:
            raise ValueError("SEQAX_RESIDUAL_CONFIRMATION_PLAN_REPLAY_MISMATCH")

    def timing_observations(contract, _compiled, _resident):
        return _rounds(contract, candidate_ns=90)

    monkeypatch.setattr(confirmation_runner, "_runtime_identity", lambda: contract.runtime)
    monkeypatch.setattr(confirmation_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(
        confirmation_runner,
        "_require_compilation_root",
        lambda _root, _contract: None,
    )
    monkeypatch.setattr(confirmation_runner, "_source_state", source_state)
    monkeypatch.setattr(confirmation_runner, "_validate_source", validate_source)
    monkeypatch.setattr(confirmation_runner, "_validate_verifier_runtime", lambda: None)
    monkeypatch.setattr(confirmation_runner.jax, "devices", lambda: devices)
    monkeypatch.setattr(confirmation_runner, "_validate_devices", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_compile", compile_candidate)
    monkeypatch.setattr(
        confirmation_runner,
        "_resident_inputs",
        lambda inputs, _prepared, _mesh: inputs,
    )
    monkeypatch.setattr(
        confirmation_runner,
        "_execute",
        lambda _executable, _inputs: np.zeros(scenario.output.shape, dtype=np.float32),
    )
    monkeypatch.setattr(confirmation_runner.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        confirmation_runner,
        "_correctness_observation",
        correctness_observation,
    )
    monkeypatch.setattr(confirmation_runner, "_replay_correctness", replay_correctness)
    monkeypatch.setattr(confirmation_runner, "_expected_plan_files", expected_plan_files)
    monkeypatch.setattr(confirmation_runner, "_validate_output_abi", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_timing_observations", timing_observations)

    root = tmp_path / "run"
    result = run_seqax_residual_confirmation(root, contract)
    assert result.winner is contract.candidate
    assert result.statistics.confirmed
    assert validate_seqax_residual_confirmation(root, contract) == result

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert validate_seqax_residual_confirmation(relocated, contract) == result

    def copy(name: str) -> Path:
        destination = tmp_path / name
        shutil.copytree(root, destination)
        return destination

    accepted_without_receipt = copy("accepted-without-receipt")
    (accepted_without_receipt / "receipt.json").unlink()
    (tmp_path / ".accepted-without-receipt-receipt.json.tmp-crash").write_text("partial")
    assert run_seqax_residual_confirmation(accepted_without_receipt, contract) == result
    assert (accepted_without_receipt / "receipt.json").is_file()

    timed_without_receipt = copy("timed-without-receipt")
    (timed_without_receipt / "receipt.json").unlink()
    with sqlite3.connect(timed_without_receipt / "ledger.sqlite") as connection:
        connection.execute("DELETE FROM events WHERE state = ?", ("accepted",))
    assert run_seqax_residual_confirmation(timed_without_receipt, contract) == result
    assert (timed_without_receipt / "receipt.json").is_file()

    post_receipt_failure = copy("post-receipt-failure")
    (post_receipt_failure / "receipt.json").unlink()
    original_validate = confirmation_runner._validate
    published_snapshot: dict[str, bytes] = {}

    def fail_after_receipt_publication(
        active_root: Path,
        active_contract: SeqaxResidualConfirmationContract,
        *,
        require_accepted: bool,
        require_receipt: bool,
    ):
        if require_receipt:
            published_snapshot.update(
                {
                    path.relative_to(active_root).as_posix(): path.read_bytes()
                    for path in active_root.rglob("*")
                    if path.is_file()
                }
            )
            raise RuntimeError("simulated post-receipt validation failure")
        return original_validate(
            active_root,
            active_contract,
            require_accepted=require_accepted,
            require_receipt=require_receipt,
        )

    monkeypatch.setattr(confirmation_runner, "_validate", fail_after_receipt_publication)
    with pytest.raises(RuntimeError, match="post-receipt validation failure"):
        run_seqax_residual_confirmation(post_receipt_failure, contract)
    observed_after_failure = {
        path.relative_to(post_receipt_failure).as_posix(): path.read_bytes()
        for path in post_receipt_failure.rglob("*")
        if path.is_file()
    }
    assert observed_after_failure == published_snapshot
    assert (tmp_path / "post-receipt-failure.failure.json").is_file()
    monkeypatch.setattr(confirmation_runner, "_validate", original_validate)
    assert run_seqax_residual_confirmation(post_receipt_failure, contract) == result

    changed_hlo = copy("changed-hlo")
    hlo_path = "candidates/standard/pallas_stablehlo.txt"
    (changed_hlo / hlo_path).write_text("forged\n")
    _repair_confirmation_receipt(changed_hlo, hlo_path)
    with pytest.raises(ValueError, match="PLAN_REPLAY_MISMATCH"):
        validate_seqax_residual_confirmation(changed_hlo, contract)

    changed_inventory = copy("changed-correctness-inventory")
    result_payload = json.loads((changed_inventory / "result.json").read_text())
    correctness_payload = json.loads((changed_inventory / "correctness.json").read_text())
    result_payload["correctness"] = [
        *result_payload["correctness"][1:],
        result_payload["correctness"][-1],
    ]
    correctness_payload = [*correctness_payload[1:], correctness_payload[-1]]
    (changed_inventory / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    (changed_inventory / "correctness.json").write_text(
        json.dumps(correctness_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_confirmation_receipt(
        changed_inventory,
        "result.json",
        "correctness.json",
    )
    with pytest.raises(ValidationError, match="correctness inventory mismatch"):
        validate_seqax_residual_confirmation(changed_inventory, contract)

    changed_input = copy("changed-input")
    input_path = next(changed_input.glob("candidates/standard/correctness/*/inputs/00.npy"))
    storage = np.load(input_path, allow_pickle=False)
    storage.reshape(-1)[0] ^= np.array(1, dtype=storage.dtype)
    np.save(input_path, storage, allow_pickle=False)
    relative_input = input_path.relative_to(changed_input).as_posix()
    _repair_confirmation_receipt(changed_input, relative_input)
    with pytest.raises(ValueError, match="INPUT_REPLAY_MISMATCH"):
        validate_seqax_residual_confirmation(changed_input, contract)

    changed_round = copy("changed-round")
    rounds = json.loads((changed_round / "rounds.json").read_text())
    rounds[0]["samples_ns"][0] += 1
    rounds[0]["median_ns"] = float(np.median(rounds[0]["samples_ns"]))
    (changed_round / "rounds.json").write_text(json.dumps(rounds, indent=2, sort_keys=True) + "\n")
    _repair_confirmation_receipt(changed_round, "rounds.json")
    with pytest.raises(ValueError, match="STATISTICS_REPLAY_MISMATCH"):
        validate_seqax_residual_confirmation(changed_round, contract)

    changed_output = copy("changed-output")
    output_path = "candidates/standard/post_timing_output.npy"
    output = np.load(changed_output / output_path, allow_pickle=False)
    output.reshape(-1)[0] += np.float32(1)
    np.save(changed_output / output_path, output, allow_pickle=False)
    _repair_confirmation_receipt(changed_output, output_path)
    with pytest.raises(ValueError, match="POST_TIMING_REPLAY_MISMATCH"):
        validate_seqax_residual_confirmation(changed_output, contract)

    changed_device = copy("changed-device")
    result_payload = json.loads((changed_device / "result.json").read_text())
    result_payload["devices"][0]["process_index"] = 1
    (changed_device / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_confirmation_receipt(changed_device, "result.json")
    with pytest.raises(ValueError, match="RESULT_IDENTITY_MISMATCH"):
        validate_seqax_residual_confirmation(changed_device, contract)

    changed_identity = copy("changed-run-identity")
    forged_commit = "1" * 40
    forged_run_id = confirmation_runner.semantic_sha256(
        confirmation_runner.SEQAX_RESIDUAL_CONFIRMATION_SCHEMA,
        contract.confirmation_id,
        forged_commit,
    )
    identity_payload = json.loads((changed_identity / "run_identity.json").read_text())
    identity_payload["source_commit"] = forged_commit
    identity_payload["run_id"] = forged_run_id
    (changed_identity / "run_identity.json").write_text(
        json.dumps(identity_payload, indent=2, sort_keys=True) + "\n"
    )
    result_payload = json.loads((changed_identity / "result.json").read_text())
    result_payload["run_id"] = forged_run_id
    (changed_identity / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    with sqlite3.connect(changed_identity / "ledger.sqlite") as connection:
        connection.execute("UPDATE events SET run_id = ?", (forged_run_id,))
    receipt_payload = json.loads((changed_identity / "receipt.json").read_text())
    receipt_payload["run_id"] = forged_run_id
    (changed_identity / "receipt.json").write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_confirmation_receipt(
        changed_identity,
        "run_identity.json",
        "result.json",
        "ledger.sqlite",
    )
    with pytest.raises(ValueError, match="RUN_IDENTITY_MISMATCH"):
        validate_seqax_residual_confirmation(changed_identity, contract)

    changed_identity_schema = copy("changed-run-identity-schema")
    identity_payload = json.loads((changed_identity_schema / "run_identity.json").read_text())
    identity_payload["schema_version"] = "forged-schema"
    (changed_identity_schema / "run_identity.json").write_text(
        json.dumps(identity_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_confirmation_receipt(changed_identity_schema, "run_identity.json")
    with pytest.raises(ValidationError, match="schema_version"):
        validate_seqax_residual_confirmation(changed_identity_schema, contract)

    changed_ledger = copy("changed-ledger")
    with sqlite3.connect(changed_ledger / "ledger.sqlite") as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = ?",
            ("0" * 64, "timed"),
        )
    _repair_confirmation_receipt(changed_ledger, "ledger.sqlite")
    with pytest.raises(ValueError, match="LEDGER_PAYLOAD_MISMATCH"):
        validate_seqax_residual_confirmation(changed_ledger, contract)

    changed_extra = copy("changed-extra")
    (changed_extra / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="CLOSED_WORLD_MISMATCH"):
        validate_seqax_residual_confirmation(changed_extra, contract)
