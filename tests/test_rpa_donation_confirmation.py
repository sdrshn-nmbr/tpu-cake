from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import bfloat16

import tpu_cake.rpa_donation_confirmation as donation_confirmation
import tpu_cake.rpa_donation_confirmation_runner as confirmation_runner
from tpu_cake.cli import _parser
from tpu_cake.contracts import SourceFileContract
from tpu_cake.rpa_donation_confirmation import (
    INKLING_RPA_DONATION_CORRECTNESS_SEEDS,
    INKLING_RPA_DONATION_TIMING_SEED,
    INKLING_RPA_INSPECTED_SURFACE_SEEDS,
    InklingRpaDonationArm,
    InklingRpaDonationConfirmationContract,
    InklingRpaDonationHloCapture,
    InklingRpaDonationState,
    InklingRpaDonationTimingRound,
    default_inkling_rpa_donation_confirmation_contract,
    donation_confirmation_orders,
    donation_confirmation_statistics,
)
from tpu_cake.rpa_donation_confirmation_runner import (
    _QUERY_CACHE_ALIAS,
    _CompiledArm,
    _timing_rounds,
    _validate_compiler_hlo_aliases,
    _validate_stablehlo_aliases,
    capture_inkling_rpa_donation_hlo_identities,
    run_inkling_rpa_donation_confirmation,
    validate_inkling_rpa_donation_confirmation,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rounds(
    candidate_ns: int,
    *,
    output_sha: str | None = None,
    cache_sha: str | None = None,
) -> tuple[InklingRpaDonationTimingRound, ...]:
    contract = default_inkling_rpa_donation_confirmation_contract()
    output_sha = output_sha or _digest("output")
    cache_sha = cache_sha or _digest("cache")
    rounds = []
    for round_index, order in enumerate(donation_confirmation_orders(contract)):
        for position, arm in enumerate(order):
            value = 100 if arm is InklingRpaDonationArm.NON_DONATING else candidate_ns
            rounds.append(
                InklingRpaDonationTimingRound(
                    round_index=round_index,
                    position=position,
                    arm=arm,
                    samples_ns=(value,) * 5,
                    median_ns=float(value),
                    terminal_output_sha256=output_sha,
                    terminal_cache_sha256=cache_sha,
                )
            )
    return tuple(rounds)


def _expected_states(
    output_sha: str | None = None,
    cache_sha: str | None = None,
) -> tuple[InklingRpaDonationState, InklingRpaDonationState]:
    output_sha = output_sha or _digest("output")
    cache_sha = cache_sha or _digest("cache")
    return tuple(
        InklingRpaDonationState(arm=arm, output_sha256=output_sha, cache_sha256=cache_sha)
        for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
    )


def test_donation_confirmation_contract_is_external_and_pinned() -> None:
    path = Path("contracts/inkling-rpa-donation-confirmation-v1.json")
    saved = InklingRpaDonationConfirmationContract.model_validate_json(path.read_text())
    canonical = default_inkling_rpa_donation_confirmation_contract()
    assert saved == canonical
    assert (
        saved.confirmation_id == "7dfee6b9b857bf5aef6a16745bf5a32a28a130a2131afd3395a35c9f4edecb5a"
    )
    assert saved.hlo_identity_status == "pinned"
    assert tuple(value.stablehlo_sha256 for value in saved.arms) == (
        "f5da8c8caa28f42ff79c9bb14cb5cd638d01de85eca4156d6116ec97d14f1c7e",
        "5b779f2014ab419c5dedbd40e2c8a428184f2eccad34db9e4ad7e322b2486b3a",
    )


def test_donation_confirmation_seeds_are_fresh() -> None:
    current = {*INKLING_RPA_DONATION_CORRECTNESS_SEEDS, INKLING_RPA_DONATION_TIMING_SEED}
    inspected = {
        *INKLING_RPA_INSPECTED_SURFACE_SEEDS,
        20260821,
        29101,
        39103,
        49109,
        59113,
        69119,
        79133,
        89137,
    }
    assert len(current) == 6
    assert current.isdisjoint(inspected)


def test_donation_confirmation_statistics_apply_exact_gate() -> None:
    contract = default_inkling_rpa_donation_confirmation_contract()
    confirmed = donation_confirmation_statistics(contract, _rounds(90), _expected_states())
    rejected = donation_confirmation_statistics(contract, _rounds(98), _expected_states())
    assert confirmed.confirmed
    assert confirmed.median_improvement == pytest.approx(0.1)
    assert confirmed.positive_rounds == 32
    assert confirmed.improvement_confidence_interval == pytest.approx((0.1, 0.1))
    assert not rejected.confirmed
    assert rejected.median_improvement == pytest.approx(0.02)


def test_donation_confirmation_statistics_reject_cross_arm_state_mismatch() -> None:
    contract = default_inkling_rpa_donation_confirmation_contract()
    rounds = list(_rounds(90))
    rounds[1] = rounds[1].model_copy(update={"terminal_cache_sha256": _digest("changed")})
    with pytest.raises(ValueError, match="terminal states differ"):
        donation_confirmation_statistics(contract, tuple(rounds), _expected_states())


def test_donation_confirmation_statistics_reject_round_state_drift() -> None:
    contract = default_inkling_rpa_donation_confirmation_contract()
    rounds = list(_rounds(90))
    for index in (14, 15):
        rounds[index] = rounds[index].model_copy(
            update={
                "terminal_output_sha256": _digest("drifted-output"),
                "terminal_cache_sha256": _digest("drifted-cache"),
            }
        )

    with pytest.raises(ValueError, match="does not match reference"):
        donation_confirmation_statistics(contract, tuple(rounds), _expected_states())


def test_hlo_capture_rejects_pinned_contract_before_repository_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pinned = default_inkling_rpa_donation_confirmation_contract()
    monkeypatch.setattr(
        "tpu_cake.rpa_donation_confirmation_runner._repository_root",
        lambda: pytest.fail("repository must not be read for pinned capture"),
    )
    with pytest.raises(ValueError, match="HLO_IDENTITIES_ALREADY_PINNED"):
        capture_inkling_rpa_donation_hlo_identities(tmp_path / "capture", pinned, lambda: None)


def test_hlo_alias_validation_is_exact() -> None:
    baseline_stablehlo = "module {\n  func.func public @main(%arg0: tensor<1xbf16>)\n}\n"
    candidate_stablehlo = (
        "module {\n  func.func public @main("
        "%arg0: tensor<1xbf16> {tf.aliasing_output = 0 : i32}, "
        "%arg1: tensor<1xbf16>, %arg2: tensor<1xbf16>, "
        "%arg3: tensor<1xbf16> {tf.aliasing_output = 1 : i32})\n}\n"
    )
    baseline_compiler = "HloModule baseline, entry_computation_layout={(bf16[])->bf16[]}\n"
    candidate_compiler = (
        "HloModule candidate, input_output_alias={ {0}: (0, {}, may-alias), "
        "{1}: (3, {}, may-alias) }, entry_computation_layout={(bf16[])->bf16[]}\n"
    )
    _validate_stablehlo_aliases(baseline_stablehlo, InklingRpaDonationArm.NON_DONATING)
    _validate_stablehlo_aliases(candidate_stablehlo, InklingRpaDonationArm.DONATING)
    _validate_compiler_hlo_aliases(baseline_compiler, InklingRpaDonationArm.NON_DONATING)
    _validate_compiler_hlo_aliases(candidate_compiler, InklingRpaDonationArm.DONATING)

    with pytest.raises(ValueError, match="CANDIDATE_COMPILER_ALIAS_MISSING"):
        _validate_compiler_hlo_aliases(
            baseline_compiler + f"// {_QUERY_CACHE_ALIAS}\n",
            InklingRpaDonationArm.DONATING,
        )
    with pytest.raises(ValueError, match="BASELINE_COMPILER_ALIAS_PRESENT"):
        _validate_compiler_hlo_aliases(
            "HloModule wrong, input_output_alias={ {0}: (0, {}, may-alias) }, "
            "entry_computation_layout={(bf16[])->bf16[]}\n",
            InklingRpaDonationArm.NON_DONATING,
        )
    with pytest.raises(ValueError, match="STABLEHLO_ALIAS_MISMATCH"):
        _validate_stablehlo_aliases(
            candidate_stablehlo.replace(
                ")\n}",
                ", %arg4: tensor<1xbf16> {tf.aliasing_output = 2 : i32})\n}",
            ),
            InklingRpaDonationArm.DONATING,
        )


def test_donation_confirmation_command_is_available() -> None:
    parsed = _parser().parse_args(
        (
            "verify-inkling-rpa-donation-confirmation",
            "run",
            "--contract",
            "contract.json",
        )
    )

    assert parsed.command == "verify-inkling-rpa-donation-confirmation"


def test_timing_inputs_are_resident_before_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_inkling_rpa_donation_confirmation_contract()
    events: list[str] = []

    class FakePlan:
        def place_inputs(self, inputs, *, mesh):
            del mesh
            events.append("place")
            return inputs

    def executable(*inputs):
        events.append("execute")
        return inputs[0], inputs[3]

    def block(value):
        events.append("block")
        return value

    monkeypatch.setattr(confirmation_runner.jax, "block_until_ready", block)
    monkeypatch.setattr(confirmation_runner.time, "perf_counter_ns", _counter())
    inputs = tuple(np.zeros((1,), dtype=np.uint16) for _ in range(11))
    compiled = tuple(
        _CompiledArm(
            arm=arm,
            plan=FakePlan(),
            mesh=None,
            executable=executable,
            stablehlo="",
            compiler_hlo="",
            evidence=InklingRpaDonationHloCapture(
                arm=arm,
                stablehlo_sha256=_digest(f"stable-{arm}"),
                compiler_hlo_sha256=_digest(f"compiler-{arm}"),
                compiler_hlo_alias_contract=(
                    "no-query-cache-alias"
                    if arm is InklingRpaDonationArm.NON_DONATING
                    else "query-output-aliases-arg0-cache-output-aliases-arg3"
                ),
            ),
        )
        for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
    )

    rounds = _timing_rounds(contract, compiled, inputs)

    assert len(rounds) == 64
    assert all(
        events[index + 1] == "block" for index, value in enumerate(events[:-1]) if value == "place"
    )


def _counter():
    value = 0

    def next_value() -> int:
        nonlocal value
        value += 100
        return value

    return next_value


def _repair_receipt(root: Path, *relative_paths: str) -> None:
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in relative_paths:
        path = root / relative
        artifact = next(value for value in receipt["artifacts"] if value["path"] == relative)
        artifact["size_bytes"] = path.stat().st_size
        artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative == "result.json":
            receipt["result_sha256"] = artifact["sha256"]
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_runner_builds_and_replays_a_closed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = default_inkling_rpa_donation_confirmation_contract()
    stablehlo = {
        InklingRpaDonationArm.NON_DONATING: "baseline-stablehlo\n",
        InklingRpaDonationArm.DONATING: "candidate-stablehlo\n",
    }
    compiler_hlo = {
        InklingRpaDonationArm.NON_DONATING: "baseline-compiler-hlo\n",
        InklingRpaDonationArm.DONATING: "candidate-compiler-hlo\n",
    }
    arm_payloads = []
    for arm in base.arms:
        arm_payloads.append(
            arm.model_copy(
                update={"stablehlo_sha256": hashlib.sha256(stablehlo[arm.arm].encode()).hexdigest()}
            )
        )
    contract = base.model_copy(update={"arms": tuple(arm_payloads)})
    devices = tuple(
        SimpleNamespace(id=index, process_index=0, platform="tpu", device_kind="TPU7x")
        for index in range(8)
    )
    inputs = tuple(np.full((1,), index, dtype=bfloat16) for index in range(11))

    class FakePlan:
        def place_inputs(self, values, *, mesh):
            del mesh
            return tuple(value.copy() for value in values)

    def compile_arm(active_contract, arm, _kernel, _plan, _devices):
        arm_contract = next(value for value in active_contract.arms if value.arm is arm)

        def executable(*values):
            return values[0].copy(), values[3].copy()

        return _CompiledArm(
            arm=arm,
            plan=FakePlan(),
            mesh=None,
            executable=executable,
            stablehlo=stablehlo[arm],
            compiler_hlo=compiler_hlo[arm],
            evidence=InklingRpaDonationHloCapture(
                arm=arm,
                stablehlo_sha256=arm_contract.stablehlo_sha256,
                compiler_hlo_sha256=hashlib.sha256(compiler_hlo[arm].encode()).hexdigest(),
                compiler_hlo_alias_contract=arm_contract.compiler_hlo_alias_contract,
            ),
        )

    manifest = (SourceFileContract(path="src/tpu_cake/cli.py", sha256="0" * 64),)
    monkeypatch.setattr(
        donation_confirmation,
        "default_inkling_rpa_donation_confirmation_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        confirmation_runner,
        "default_inkling_rpa_donation_confirmation_contract",
        lambda: contract,
    )
    monkeypatch.setattr(confirmation_runner, "_require_compilation_root", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_require_clean_repository", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_require_backend_source", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_require_backend_runtime", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_runtime_identity", lambda: contract.runtime)
    monkeypatch.setattr(confirmation_runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(confirmation_runner.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(confirmation_runner.jax, "devices", lambda: devices)
    monkeypatch.setattr(confirmation_runner, "_source_manifest", lambda: manifest)
    monkeypatch.setattr(confirmation_runner, "_validate_source", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_compile_arm", compile_arm)
    monkeypatch.setattr(
        confirmation_runner,
        "lower_inkling_sharded_rpa_to_pallas",
        lambda _schedule: object(),
    )
    monkeypatch.setattr(
        confirmation_runner,
        "inkling_sharded_fused_rpa_inputs",
        lambda _seed: tuple(value.copy() for value in inputs),
    )
    monkeypatch.setattr(
        confirmation_runner,
        "inkling_sharded_fused_rpa_reference",
        lambda values: (values[0].copy(), values[3].copy()),
    )
    monkeypatch.setattr(confirmation_runner, "_validate_output_abi", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_validate_stablehlo_aliases", lambda *_args: None)
    monkeypatch.setattr(confirmation_runner, "_validate_compiler_hlo_aliases", lambda *_args: None)
    monkeypatch.setattr(
        confirmation_runner,
        "_timing_rounds",
        lambda active_contract, _compiled, _inputs: _rounds(
            90,
            output_sha=confirmation_runner.array_sha256(inputs[0]),
            cache_sha=confirmation_runner.array_sha256(inputs[3]),
        ),
    )

    root = tmp_path / "run"
    result = run_inkling_rpa_donation_confirmation(root, contract, lambda: None)
    assert result.accepted
    assert result.winner is InklingRpaDonationArm.DONATING
    assert validate_inkling_rpa_donation_confirmation(root, contract) == result

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert validate_inkling_rpa_donation_confirmation(relocated, contract) == result

    original_validate = confirmation_runner.validate_inkling_rpa_donation_confirmation

    def fail_preaccept(
        active_root,
        active_contract,
        *,
        require_accepted=True,
        require_receipt=True,
    ):
        if not require_accepted and not require_receipt:
            raise RuntimeError("simulated preaccept replay failure")
        return original_validate(
            active_root,
            active_contract,
            require_accepted=require_accepted,
            require_receipt=require_receipt,
        )

    monkeypatch.setattr(
        confirmation_runner,
        "validate_inkling_rpa_donation_confirmation",
        fail_preaccept,
    )
    failed_root = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match="preaccept replay failure"):
        run_inkling_rpa_donation_confirmation(failed_root, contract, lambda: None)
    history = confirmation_runner.read_ledger_history(
        failed_root / "ledger.sqlite",
        result.run_id,
    )
    assert history[-1].state.value == "timed"
    snapshot = {
        path.relative_to(failed_root).as_posix(): path.read_bytes()
        for path in failed_root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="LEDGER_REPLAY_MISMATCH"):
        run_inkling_rpa_donation_confirmation(failed_root, contract, lambda: None)
    assert snapshot == {
        path.relative_to(failed_root).as_posix(): path.read_bytes()
        for path in failed_root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        confirmation_runner,
        "validate_inkling_rpa_donation_confirmation",
        original_validate,
    )

    changed_output = tmp_path / "changed-output"
    shutil.copytree(root, changed_output)
    relative = (
        f"correctness/seed-{contract.correctness_seeds[0]}/{contract.baseline.value}/output.npy"
    )
    output = np.load(changed_output / relative, allow_pickle=False)
    output.reshape(-1)[0] ^= np.uint16(1)
    np.save(changed_output / relative, output, allow_pickle=False)
    _repair_receipt(changed_output, relative)
    with pytest.raises(ValueError, match="CORRECTNESS_REPLAY_MISMATCH"):
        validate_inkling_rpa_donation_confirmation(changed_output, contract)

    changed_round = tmp_path / "changed-round"
    shutil.copytree(root, changed_round)
    rounds = json.loads((changed_round / "rounds.json").read_text())
    rounds[0]["samples_ns"][0] += 1
    rounds[0]["median_ns"] = float(np.median(rounds[0]["samples_ns"]))
    (changed_round / "rounds.json").write_text(json.dumps(rounds, indent=2, sort_keys=True) + "\n")
    _repair_receipt(changed_round, "rounds.json")
    with pytest.raises(ValueError, match="STATISTICS_REPLAY_MISMATCH"):
        validate_inkling_rpa_donation_confirmation(changed_round, contract)

    changed_hlo = tmp_path / "changed-hlo"
    shutil.copytree(root, changed_hlo)
    relative = "arms/non_donating/compiler_hlo.txt"
    (changed_hlo / relative).write_text("forged\n")
    _repair_receipt(changed_hlo, relative)
    with pytest.raises(ValueError, match="HLO_REPLAY_MISMATCH"):
        validate_inkling_rpa_donation_confirmation(changed_hlo, contract)

    changed_timing_state = tmp_path / "changed-timing-state"
    shutil.copytree(root, changed_timing_state)
    relative = "timing/non_donating/post_cache.npy"
    cache = np.load(changed_timing_state / relative, allow_pickle=False)
    cache.reshape(-1)[0] ^= np.uint16(1)
    np.save(changed_timing_state / relative, cache, allow_pickle=False)
    _repair_receipt(changed_timing_state, relative)
    with pytest.raises(ValueError, match="TIMING_STATE_MISMATCH"):
        validate_inkling_rpa_donation_confirmation(changed_timing_state, contract)

    changed_result = tmp_path / "changed-result"
    shutil.copytree(root, changed_result)
    result_payload = json.loads((changed_result / "result.json").read_text())
    result_payload["accepted"] = False
    result_payload["winner"] = None
    (changed_result / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_result, "result.json")
    with pytest.raises(ValueError, match="contradicts statistics"):
        validate_inkling_rpa_donation_confirmation(changed_result, contract)

    changed_ledger = tmp_path / "changed-ledger"
    shutil.copytree(root, changed_ledger)
    with sqlite3.connect(changed_ledger / "ledger.sqlite") as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = 'timed'",
            ("0" * 64,),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for suffix in ("-shm", "-wal"):
        (changed_ledger / f"ledger.sqlite{suffix}").unlink(missing_ok=True)
    _repair_receipt(changed_ledger, "ledger.sqlite")
    with pytest.raises(ValueError, match="LEDGER_REPLAY_MISMATCH"):
        validate_inkling_rpa_donation_confirmation(changed_ledger, contract)

    changed_identity = tmp_path / "changed-identity"
    shutil.copytree(root, changed_identity)
    identity = json.loads((changed_identity / "run_identity.json").read_text())
    identity["source_commit"] = "0" * 40
    (changed_identity / "run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_identity, "run_identity.json")
    with pytest.raises(ValueError, match="RUN_IDENTITY_MISMATCH"):
        validate_inkling_rpa_donation_confirmation(changed_identity, contract)

    extra = tmp_path / "extra"
    shutil.copytree(root, extra)
    (extra / "extra.txt").write_text("forged")
    with pytest.raises(ValueError, match="RECEIPT_IDENTITY_MISMATCH"):
        validate_inkling_rpa_donation_confirmation(extra, contract)
