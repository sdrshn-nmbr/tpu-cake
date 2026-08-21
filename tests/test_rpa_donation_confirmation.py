from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tpu_cake.rpa_donation_confirmation import (
    INKLING_RPA_DONATION_CORRECTNESS_SEEDS,
    INKLING_RPA_DONATION_TIMING_SEED,
    INKLING_RPA_INSPECTED_SURFACE_SEEDS,
    InklingRpaDonationArm,
    InklingRpaDonationConfirmationContract,
    InklingRpaDonationTimingRound,
    default_inkling_rpa_donation_confirmation_contract,
    donation_confirmation_orders,
    donation_confirmation_statistics,
)
from tpu_cake.rpa_donation_confirmation_runner import (
    _QUERY_CACHE_ALIAS,
    _validate_compiler_hlo_aliases,
    _validate_stablehlo_aliases,
    capture_inkling_rpa_donation_hlo_identities,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rounds(candidate_ns: int) -> tuple[InklingRpaDonationTimingRound, ...]:
    contract = default_inkling_rpa_donation_confirmation_contract()
    output_sha = _digest("output")
    cache_sha = _digest("cache")
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


def test_donation_confirmation_contract_is_external_and_pending() -> None:
    path = Path("contracts/inkling-rpa-donation-confirmation-v1.json")
    saved = InklingRpaDonationConfirmationContract.model_validate_json(path.read_text())
    canonical = default_inkling_rpa_donation_confirmation_contract()
    assert saved == canonical
    assert (
        saved.confirmation_id == "9feadd331f735242324483aa5256337ba1ac6450018aef434a7415dcc7636604"
    )
    assert saved.hlo_identity_status == "pending"
    assert tuple(value.stablehlo_sha256 for value in saved.arms) == ("0" * 64, "0" * 64)


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
    confirmed = donation_confirmation_statistics(contract, _rounds(90))
    rejected = donation_confirmation_statistics(contract, _rounds(98))
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
        donation_confirmation_statistics(contract, tuple(rounds))


def test_hlo_capture_rejects_pinned_contract_before_repository_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pending = default_inkling_rpa_donation_confirmation_contract()
    pinned = pending.model_copy(update={"hlo_identity_status": "pinned"})
    monkeypatch.setattr(
        "tpu_cake.rpa_donation_confirmation_runner.default_inkling_rpa_donation_confirmation_contract",
        lambda: pinned,
    )
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
