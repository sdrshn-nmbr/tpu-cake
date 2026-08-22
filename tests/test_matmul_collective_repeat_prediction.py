import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import tpu_cake.matmul_collective_repeat_prediction as prediction
from tpu_cake.matmul_collective_confirmation import MATMUL_COLLECTIVE_DIAGNOSTICS
from tpu_cake.matmul_collective_repeat_prediction import (
    MatmulCollectiveRepeatPredictionContract,
    _median,
    build_matmul_collective_repeat_prediction,
    default_matmul_collective_repeat_prediction_contract,
    validate_matmul_collective_repeat_prediction,
    write_matmul_collective_repeat_prediction,
)
from tpu_cake.runner import MatmulCollectiveStrategy


def _cost(schedule_sha256: str, idealized: str, serial: str) -> SimpleNamespace:
    return SimpleNamespace(
        schedule_sha256=schedule_sha256,
        metrics=(
            SimpleNamespace(
                name="idealized_time_floor",
                quantity=SimpleNamespace(value=Decimal(idealized)),
            ),
            SimpleNamespace(
                name="serial_resource_time",
                quantity=SimpleNamespace(value=Decimal(serial)),
            ),
        ),
    )


def _evidence() -> tuple[dict, SimpleNamespace]:
    baseline = MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
    candidate = MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING
    plans = tuple(
        SimpleNamespace(
            strategy=authority.strategy,
            schedule_sha256=authority.schedule_sha256,
            semantic_compiler_hlo_sha256=authority.semantic_compiler_hlo_sha256,
        )
        for authority in MATMUL_COLLECTIVE_DIAGNOSTICS
    )
    rounds = []
    for index in range(32):
        for position, (strategy, median) in enumerate(
            ((baseline, 90 + index % 2), (candidate, 100 + index % 2))
        ):
            rounds.append(
                SimpleNamespace(
                    round_index=index,
                    position=position,
                    strategy=strategy,
                    median_ns=float(median),
                    samples_ns=(median - 2, median - 1, median, median + 1, median + 2),
                )
            )
    statistics = SimpleNamespace(
        round_count=32,
        baseline=baseline,
        candidate=candidate,
        median_improvement=-0.11,
        improvement_confidence_interval=(-0.13, -0.09),
        minimum_practical_improvement=0.03,
        selected_strategy=baseline,
        decision="keep_baseline",
        ab_median_improvement=-0.12,
        ba_median_improvement=-0.10,
        position_order_effect=-0.02,
    )
    confirmation = SimpleNamespace(
        confirmation_id=prediction.MATMUL_COLLECTIVE_CONFIRMATION_ID,
        plans=plans,
        rounds=tuple(rounds),
        statistics=statistics,
    )
    diagnostics = {
        baseline: (
            SimpleNamespace(
                collective_strategy=baseline,
                schedule_sha256=plans[0].schedule_sha256,
                samples_ns=(99, 100),
                passed=True,
                measured_iterations=2,
                median_ns=99,
                p90_ns=100,
                coefficient_of_variation=0.005,
            ),
            _cost(plans[0].schedule_sha256, "10", "20"),
        ),
        candidate: (
            SimpleNamespace(
                collective_strategy=candidate,
                schedule_sha256=plans[1].schedule_sha256,
                samples_ns=(110, 111),
                passed=True,
                measured_iterations=2,
                median_ns=110,
                p90_ns=111,
                coefficient_of_variation=0.004,
            ),
            _cost(plans[1].schedule_sha256, "10", "21"),
        ),
    }
    return diagnostics, confirmation


def test_repeat_prediction_contract_file_is_canonical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "matmul-collective-repeat-prediction-v1.json"
    )
    saved = MatmulCollectiveRepeatPredictionContract.model_validate_json(path.read_text())

    assert saved == default_matmul_collective_repeat_prediction_contract()
    assert len(saved.contract_id) == 64


def test_repeat_prediction_preserves_half_nanosecond_sample_medians() -> None:
    diagnostics, confirmation = _evidence()

    report = build_matmul_collective_repeat_prediction(
        default_matmul_collective_repeat_prediction_contract(),
        diagnostics,
        confirmation,
    )

    by_strategy = {arm.strategy: arm for arm in report.arms}
    assert by_strategy[MatmulCollectiveStrategy.XLA_REDUCE_SCATTER].predicted_median_ns == Decimal(
        "99.5"
    )
    assert by_strategy[
        MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING
    ].predicted_median_ns == Decimal("110.5")
    assert report.strategy_ranking_agrees
    assert report.selected_strategy_agrees
    assert report.confirmation_decision == "keep_baseline"
    assert not report.prospective_validation
    assert not report.acceptance_bound_predeclared
    assert not report.supports_shape_extrapolation
    assert report.same_shape_holdout


def test_exact_median_rejects_empty_samples() -> None:
    assert _median((1, 2, 3, 4)) == Decimal("2.5")
    with pytest.raises(ValueError, match="EMPTY_SAMPLE"):
        _median(())


def test_repeat_prediction_validation_rejects_forged_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics, confirmation = _evidence()
    report = build_matmul_collective_repeat_prediction(
        default_matmul_collective_repeat_prediction_contract(),
        diagnostics,
        confirmation,
    )
    monkeypatch.setattr(
        prediction,
        "derive_matmul_collective_repeat_prediction",
        lambda *_args, **_kwargs: report,
    )
    forged = report.model_copy(update={"predicted_candidate_improvement": Decimal("0.5")})

    with pytest.raises(ValueError, match="REPORT_REPLAY_MISMATCH"):
        validate_matmul_collective_repeat_prediction(
            forged,
            diagnostic_root=Path("diagnostic"),
            diagnostic_archive=Path("diagnostic.tar.zst"),
            confirmation_root=Path("confirmation"),
            confirmation_archive=Path("confirmation.tar.zst"),
            confirmation_contract=SimpleNamespace(),
            contract=default_matmul_collective_repeat_prediction_contract(),
        )


def test_repeat_prediction_writer_rejects_evidence_overlap_and_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics, confirmation = _evidence()
    report = build_matmul_collective_repeat_prediction(
        default_matmul_collective_repeat_prediction_contract(),
        diagnostics,
        confirmation,
    )
    monkeypatch.setattr(
        prediction,
        "derive_matmul_collective_repeat_prediction",
        lambda *_args, **_kwargs: report,
    )
    diagnostic_root = tmp_path / "diagnostic"
    confirmation_root = tmp_path / "confirmation"
    diagnostic_root.mkdir()
    confirmation_root.mkdir()
    common = {
        "diagnostic_root": diagnostic_root,
        "diagnostic_archive": tmp_path / "diagnostic.tar.zst",
        "confirmation_root": confirmation_root,
        "confirmation_archive": tmp_path / "confirmation.tar.zst",
        "confirmation_contract": SimpleNamespace(),
        "contract": default_matmul_collective_repeat_prediction_contract(),
    }

    with pytest.raises(ValueError, match="OUTPUT_OVERLAPS_EVIDENCE"):
        write_matmul_collective_repeat_prediction(
            diagnostic_root / "report.json",
            **common,
        )
    output = tmp_path / "report.json"
    output.write_text(json.dumps({}))
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        write_matmul_collective_repeat_prediction(output, **common)


def test_repeat_prediction_cli_requires_both_evidence_archives() -> None:
    parser = prediction._parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--diagnostic-root",
            "diagnostic",
            "--diagnostic-archive",
            "diagnostic.tar.zst",
            "--confirmation-root",
            "confirmation",
            "--confirmation-archive",
            "confirmation.tar.zst",
            "--confirmation-contract",
            "confirmation-contract.json",
            "--contract",
            "prediction-contract.json",
            "--output",
            "report.json",
        ]
    )

    assert args.command == "evaluate"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate",
                "--diagnostic-root",
                "diagnostic",
                "--confirmation-root",
                "confirmation",
                "--confirmation-contract",
                "confirmation-contract.json",
                "--contract",
                "prediction-contract.json",
                "--output",
                "report.json",
            ]
        )
