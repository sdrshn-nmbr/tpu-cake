from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.artifacts import file_sha256, write_json
from tpu_cake.cost_model import CostModelReport
from tpu_cake.identity import model_identity_sha256
from tpu_cake.matmul_collective_confirmation import (
    MATMUL_COLLECTIVE_DIAGNOSTIC_ARCHIVE_SHA256,
    MATMUL_COLLECTIVE_DIAGNOSTICS,
    MatmulCollectiveConfirmationContract,
    MatmulCollectiveConfirmationResult,
)
from tpu_cake.matmul_collective_confirmation_runner import (
    _validate_diagnostics,
    validate_matmul_collective_confirmation,
)
from tpu_cake.runner import MatmulCollectiveStrategy, MatmulRunResult

MATMUL_COLLECTIVE_REPEAT_PREDICTION_SCHEMA = "matmul-collective-repeat-prediction-v1"
MATMUL_COLLECTIVE_CONFIRMATION_ID = (
    "7c33b7cf0d0a84439ddd5214ac10a73e1fe73dccde04804252148e7a76b50163"
)
MATMUL_COLLECTIVE_CONFIRMATION_RESULT_SHA256 = (
    "d02da084b88b216a5cb5f39f726fd2676f2635f02eb50c59453677dcf4ac019b"
)
MATMUL_COLLECTIVE_CONFIRMATION_RECEIPT_SHA256 = (
    "64630b3179073341228f7f8df5c9a0ece77f8adcaba196a5ded4fc6cdc900485"
)
MATMUL_COLLECTIVE_CONFIRMATION_ARCHIVE_SHA256 = (
    "93c7a0e99a2e3cf1517f3cacd4a3d9b56b98e753686b83f47895706c80f024e9"
)
MATMUL_COLLECTIVE_CONFIRMATION_CONTRACT_SHA256 = (
    "aab4e762692c4f619acc28b5b6f864c9b2d60596c0c1860bcf8648b6aed0ae1f"
)
MATMUL_COLLECTIVE_CONFIRMATION_ROUNDS_SHA256 = (
    "72b32e0331a049e9c2d4a039006133b3b82cd933566404a9934f7c7af39c6449"
)
MATMUL_COLLECTIVE_CONFIRMATION_TIMING_INPUT_SHA256 = (
    "e5c386b5b29f62dca7c880d1fc39ad7254cac56dd836a209e02433f9d15d9828"
)


class MatmulCollectiveCalibrationAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: MatmulCollectiveStrategy
    diagnostic_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timing_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


MATMUL_COLLECTIVE_CALIBRATION_AUTHORITIES = (
    MatmulCollectiveCalibrationAuthority(
        strategy=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
        diagnostic_receipt_sha256=MATMUL_COLLECTIVE_DIAGNOSTICS[0].receipt_sha256,
        timing_result_sha256=MATMUL_COLLECTIVE_DIAGNOSTICS[0].timing_result_sha256,
        cost_model_sha256=("c2450b4272eb52e5fb1e283f929637d738651f61f4c97272017ba04346af6816"),
    ),
    MatmulCollectiveCalibrationAuthority(
        strategy=MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING,
        diagnostic_receipt_sha256=MATMUL_COLLECTIVE_DIAGNOSTICS[1].receipt_sha256,
        timing_result_sha256=MATMUL_COLLECTIVE_DIAGNOSTICS[1].timing_result_sha256,
        cost_model_sha256=("25d9b12c744fb009a362cc12465bb2a22b1a6dc6f293d71139f3ef611a3a9525"),
    ),
)


class MatmulCollectiveRepeatPredictionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[MATMUL_COLLECTIVE_REPEAT_PREDICTION_SCHEMA] = (
        MATMUL_COLLECTIVE_REPEAT_PREDICTION_SCHEMA
    )
    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_rounds_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_timing_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration: tuple[MatmulCollectiveCalibrationAuthority, ...] = Field(
        min_length=2,
        max_length=2,
    )
    prediction_rule: str
    evaluation_rule: str
    prediction_rule_predeclared: bool
    acceptance_bound_predeclared: bool
    prospective_validation: bool
    same_shape_holdout: bool
    supports_shape_extrapolation: bool

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> MatmulCollectiveRepeatPredictionContract:
        if (
            self.confirmation_id,
            self.diagnostic_archive_sha256,
            self.confirmation_result_sha256,
            self.confirmation_receipt_sha256,
            self.confirmation_archive_sha256,
            self.confirmation_contract_sha256,
            self.confirmation_rounds_sha256,
            self.confirmation_timing_input_sha256,
            self.calibration,
            self.prediction_rule,
            self.evaluation_rule,
            self.prediction_rule_predeclared,
            self.acceptance_bound_predeclared,
            self.prospective_validation,
            self.same_shape_holdout,
            self.supports_shape_extrapolation,
        ) != (
            MATMUL_COLLECTIVE_CONFIRMATION_ID,
            MATMUL_COLLECTIVE_DIAGNOSTIC_ARCHIVE_SHA256,
            MATMUL_COLLECTIVE_CONFIRMATION_RESULT_SHA256,
            MATMUL_COLLECTIVE_CONFIRMATION_RECEIPT_SHA256,
            MATMUL_COLLECTIVE_CONFIRMATION_ARCHIVE_SHA256,
            MATMUL_COLLECTIVE_CONFIRMATION_CONTRACT_SHA256,
            MATMUL_COLLECTIVE_CONFIRMATION_ROUNDS_SHA256,
            MATMUL_COLLECTIVE_CONFIRMATION_TIMING_INPUT_SHA256,
            MATMUL_COLLECTIVE_CALIBRATION_AUTHORITIES,
            "predicted_median_ns(strategy)=median(diagnostic_timing_samples_ns(strategy))",
            "holdout_median_ns(strategy)=median(confirmation_round_medians_ns(strategy))",
            False,
            False,
            False,
            True,
            False,
        ):
            raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_PROTOCOL_MISMATCH")
        return self

    @computed_field
    @property
    def contract_id(self) -> str:
        return model_identity_sha256(self)


class MatmulCollectiveRepeatPredictionArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: MatmulCollectiveStrategy
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_sample_count: int = Field(gt=0)
    predicted_median_ns: Decimal = Field(gt=0)
    diagnostic_p90_ns: Decimal = Field(gt=0)
    diagnostic_coefficient_of_variation: Decimal = Field(ge=0)
    idealized_time_floor_ns: Decimal = Field(gt=0)
    serial_resource_scenario_ns: Decimal = Field(gt=0)
    unexplained_residual_over_idealized_ns: Decimal = Field(gt=0)
    unexplained_residual_over_serial_scenario_ns: Decimal = Field(gt=0)
    measured_to_idealized_ratio: Decimal = Field(gt=0)
    holdout_round_count: int = Field(gt=0)
    holdout_raw_call_count: int = Field(gt=0)
    holdout_raw_call_median_ns: Decimal = Field(gt=0)
    holdout_median_ns: Decimal = Field(gt=0)
    holdout_minimum_ns: Decimal = Field(gt=0)
    holdout_maximum_ns: Decimal = Field(gt=0)
    holdout_first_half_median_ns: Decimal = Field(gt=0)
    holdout_second_half_median_ns: Decimal = Field(gt=0)
    signed_relative_prediction_error: Decimal
    absolute_relative_prediction_error: Decimal = Field(ge=0)


class MatmulCollectiveRepeatPredictionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[MATMUL_COLLECTIVE_REPEAT_PREDICTION_SCHEMA] = (
        MATMUL_COLLECTIVE_REPEAT_PREDICTION_SCHEMA
    )
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_rounds_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_timing_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arms: tuple[MatmulCollectiveRepeatPredictionArm, ...] = Field(
        min_length=2,
        max_length=2,
    )
    predicted_candidate_improvement: Decimal
    observed_unpaired_candidate_improvement: Decimal
    observed_paired_candidate_improvement: Decimal
    observed_paired_improvement_confidence_interval: tuple[Decimal, Decimal]
    predicted_improvement_inside_observed_confidence_interval: bool
    strategy_ranking_agrees: bool
    selected_strategy_agrees: bool
    confirmation_decision: Literal["promote_candidate", "keep_baseline", "inconclusive"]
    confirmation_selected_strategy: MatmulCollectiveStrategy
    ab_median_improvement: Decimal
    ba_median_improvement: Decimal
    position_order_effect: Decimal
    prediction_rule_predeclared: bool
    acceptance_bound_predeclared: bool
    prospective_validation: bool
    same_shape_holdout: bool
    supports_shape_extrapolation: bool
    status: Literal["retrospective-same-shape-repeat-only"]
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def epistemic_scope_is_explicit(self) -> MatmulCollectiveRepeatPredictionReport:
        if (
            self.prediction_rule_predeclared
            or self.acceptance_bound_predeclared
            or self.prospective_validation
            or not self.same_shape_holdout
            or self.supports_shape_extrapolation
        ):
            raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_SCOPE_MISMATCH")
        return self


def default_matmul_collective_repeat_prediction_contract() -> (
    MatmulCollectiveRepeatPredictionContract
):
    return MatmulCollectiveRepeatPredictionContract(
        confirmation_id=MATMUL_COLLECTIVE_CONFIRMATION_ID,
        diagnostic_archive_sha256=MATMUL_COLLECTIVE_DIAGNOSTIC_ARCHIVE_SHA256,
        confirmation_result_sha256=MATMUL_COLLECTIVE_CONFIRMATION_RESULT_SHA256,
        confirmation_receipt_sha256=MATMUL_COLLECTIVE_CONFIRMATION_RECEIPT_SHA256,
        confirmation_archive_sha256=MATMUL_COLLECTIVE_CONFIRMATION_ARCHIVE_SHA256,
        confirmation_contract_sha256=MATMUL_COLLECTIVE_CONFIRMATION_CONTRACT_SHA256,
        confirmation_rounds_sha256=MATMUL_COLLECTIVE_CONFIRMATION_ROUNDS_SHA256,
        confirmation_timing_input_sha256=(MATMUL_COLLECTIVE_CONFIRMATION_TIMING_INPUT_SHA256),
        calibration=MATMUL_COLLECTIVE_CALIBRATION_AUTHORITIES,
        prediction_rule=(
            "predicted_median_ns(strategy)=median(diagnostic_timing_samples_ns(strategy))"
        ),
        evaluation_rule=(
            "holdout_median_ns(strategy)=median(confirmation_round_medians_ns(strategy))"
        ),
        prediction_rule_predeclared=False,
        acceptance_bound_predeclared=False,
        prospective_validation=False,
        same_shape_holdout=True,
        supports_shape_extrapolation=False,
    )


def _median(values: tuple[int | Decimal, ...]) -> Decimal:
    if not values:
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_EMPTY_SAMPLE")
    ordered = sorted(Decimal(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _cost_metric(report: CostModelReport, name: str) -> Decimal:
    matches = tuple(metric for metric in report.metrics if metric.name == name)
    if len(matches) != 1:
        raise ValueError(f"MATMUL_COLLECTIVE_REPEAT_PREDICTION_COST_METRIC_MISMATCH name={name}")
    return matches[0].quantity.value


def build_matmul_collective_repeat_prediction(
    contract: MatmulCollectiveRepeatPredictionContract,
    diagnostics: dict[
        MatmulCollectiveStrategy,
        tuple[MatmulRunResult, CostModelReport],
    ],
    confirmation: MatmulCollectiveConfirmationResult,
) -> MatmulCollectiveRepeatPredictionReport:
    if contract != default_matmul_collective_repeat_prediction_contract():
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_EXTERNAL_CONTRACT_MISMATCH")
    if confirmation.confirmation_id != contract.confirmation_id:
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_CONFIRMATION_MISMATCH")

    plans = {plan.strategy: plan for plan in confirmation.plans}
    arm_records = []
    for authority in contract.calibration:
        timing, cost = diagnostics[authority.strategy]
        plan = plans[authority.strategy]
        if (
            timing.collective_strategy is not authority.strategy
            or timing.schedule_sha256 != plan.schedule_sha256
            or cost.schedule_sha256 != plan.schedule_sha256
            or not timing.passed
            or not timing.samples_ns
            or timing.measured_iterations != len(timing.samples_ns)
            or timing.median_ns is None
            or timing.p90_ns is None
            or timing.coefficient_of_variation is None
        ):
            raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_DIAGNOSTIC_IDENTITY_MISMATCH")
        predicted = _median(timing.samples_ns)
        idealized = _cost_metric(cost, "idealized_time_floor")
        serial = _cost_metric(cost, "serial_resource_time")
        holdout = tuple(
            Decimal(str(round_.median_ns))
            for round_ in confirmation.rounds
            if round_.strategy is authority.strategy
        )
        if len(holdout) != confirmation.statistics.round_count or len(holdout) % 2:
            raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_HOLDOUT_COUNT_MISMATCH")
        observed = _median(holdout)
        raw_holdout = tuple(
            sample
            for round_ in confirmation.rounds
            if round_.strategy is authority.strategy
            for sample in round_.samples_ns
        )
        signed_error = (predicted - observed) / observed
        midpoint = len(holdout) // 2
        diagnostic = next(
            value for value in MATMUL_COLLECTIVE_DIAGNOSTICS if value.strategy is authority.strategy
        )
        arm_records.append(
            MatmulCollectiveRepeatPredictionArm(
                strategy=authority.strategy,
                schedule_sha256=plan.schedule_sha256,
                semantic_compiler_hlo_sha256=plan.semantic_compiler_hlo_sha256,
                diagnostic_sample_count=len(timing.samples_ns),
                predicted_median_ns=predicted,
                diagnostic_p90_ns=Decimal(timing.p90_ns or 0),
                diagnostic_coefficient_of_variation=Decimal(str(timing.coefficient_of_variation)),
                idealized_time_floor_ns=idealized,
                serial_resource_scenario_ns=serial,
                unexplained_residual_over_idealized_ns=predicted - idealized,
                unexplained_residual_over_serial_scenario_ns=predicted - serial,
                measured_to_idealized_ratio=predicted / idealized,
                holdout_round_count=len(holdout),
                holdout_raw_call_count=len(raw_holdout),
                holdout_raw_call_median_ns=_median(raw_holdout),
                holdout_median_ns=observed,
                holdout_minimum_ns=min(holdout),
                holdout_maximum_ns=max(holdout),
                holdout_first_half_median_ns=_median(holdout[:midpoint]),
                holdout_second_half_median_ns=_median(holdout[midpoint:]),
                signed_relative_prediction_error=signed_error,
                absolute_relative_prediction_error=abs(signed_error),
            )
        )
        if (
            plan.schedule_sha256 != diagnostic.schedule_sha256
            or plan.semantic_compiler_hlo_sha256 != diagnostic.semantic_compiler_hlo_sha256
        ):
            raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_COMPILER_IDENTITY_MISMATCH")

    arms = tuple(arm_records)
    by_strategy = {arm.strategy: arm for arm in arms}
    baseline = by_strategy[MatmulCollectiveStrategy.XLA_REDUCE_SCATTER]
    candidate = by_strategy[MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING]
    predicted_improvement = Decimal(1) - (
        candidate.predicted_median_ns / baseline.predicted_median_ns
    )
    observed_unpaired = Decimal(1) - (candidate.holdout_median_ns / baseline.holdout_median_ns)
    observed_paired = Decimal(str(confirmation.statistics.median_improvement))
    confidence_interval = tuple(
        Decimal(str(value)) for value in confirmation.statistics.improvement_confidence_interval
    )
    threshold = Decimal(str(confirmation.statistics.minimum_practical_improvement))
    predicted_selection = (
        confirmation.statistics.candidate
        if predicted_improvement > threshold
        else confirmation.statistics.baseline
    )
    return MatmulCollectiveRepeatPredictionReport(
        contract_id=contract.contract_id,
        confirmation_id=contract.confirmation_id,
        diagnostic_archive_sha256=contract.diagnostic_archive_sha256,
        confirmation_result_sha256=contract.confirmation_result_sha256,
        confirmation_receipt_sha256=contract.confirmation_receipt_sha256,
        confirmation_archive_sha256=contract.confirmation_archive_sha256,
        confirmation_contract_sha256=contract.confirmation_contract_sha256,
        confirmation_rounds_sha256=contract.confirmation_rounds_sha256,
        confirmation_timing_input_sha256=(contract.confirmation_timing_input_sha256),
        arms=arms,
        predicted_candidate_improvement=predicted_improvement,
        observed_unpaired_candidate_improvement=observed_unpaired,
        observed_paired_candidate_improvement=observed_paired,
        observed_paired_improvement_confidence_interval=confidence_interval,
        predicted_improvement_inside_observed_confidence_interval=(
            confidence_interval[0] <= predicted_improvement <= confidence_interval[1]
        ),
        strategy_ranking_agrees=((predicted_improvement > 0) == (observed_paired > 0)),
        selected_strategy_agrees=(predicted_selection is confirmation.statistics.selected_strategy),
        confirmation_decision=confirmation.statistics.decision,
        confirmation_selected_strategy=confirmation.statistics.selected_strategy,
        ab_median_improvement=Decimal(str(confirmation.statistics.ab_median_improvement)),
        ba_median_improvement=Decimal(str(confirmation.statistics.ba_median_improvement)),
        position_order_effect=Decimal(str(confirmation.statistics.position_order_effect)),
        prediction_rule_predeclared=contract.prediction_rule_predeclared,
        acceptance_bound_predeclared=contract.acceptance_bound_predeclared,
        prospective_validation=contract.prospective_validation,
        same_shape_holdout=contract.same_shape_holdout,
        supports_shape_extrapolation=contract.supports_shape_extrapolation,
        status="retrospective-same-shape-repeat-only",
        limitations=(
            "The prediction rule was recorded after the confirmation timing completed, so this is retrospective evaluation rather than prospective validation.",
            "One diagnostic timing point per strategy cannot identify causal compute, HBM, ICI, launch, synchronization, or compiler-scheduling coefficients.",
            "The confirmation repeats the exact diagnostic workload and compiler semantics; this report does not support model-shape, sequence-length, topology, runtime, or compiler extrapolation.",
            "The idealized roofline is a lower bound and the serial resource value is a calculated full-serialization scenario; neither is a latency prediction.",
            "No error acceptance bound is applied because none was declared before the holdout was observed.",
        ),
    )


def derive_matmul_collective_repeat_prediction(
    diagnostic_root: Path,
    diagnostic_archive: Path,
    confirmation_root: Path,
    confirmation_archive: Path,
    confirmation_contract: MatmulCollectiveConfirmationContract,
    contract: MatmulCollectiveRepeatPredictionContract,
) -> MatmulCollectiveRepeatPredictionReport:
    if contract != default_matmul_collective_repeat_prediction_contract():
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_EXTERNAL_CONTRACT_MISMATCH")
    _validate_diagnostics(diagnostic_root, diagnostic_archive, confirmation_contract)
    confirmation = validate_matmul_collective_confirmation(
        confirmation_root,
        confirmation_contract,
    )
    if (
        file_sha256(confirmation_archive) != contract.confirmation_archive_sha256
        or file_sha256(confirmation_root / "contract.json") != contract.confirmation_contract_sha256
        or file_sha256(confirmation_root / "rounds.json") != contract.confirmation_rounds_sha256
        or file_sha256(confirmation_root / "timing_input.json")
        != contract.confirmation_timing_input_sha256
        or file_sha256(confirmation_root / "result.json") != contract.confirmation_result_sha256
        or file_sha256(confirmation_root / "receipt.json") != contract.confirmation_receipt_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_CONFIRMATION_HASH_MISMATCH")

    diagnostics = {}
    for authority in contract.calibration:
        directory = (
            "xla" if authority.strategy is MatmulCollectiveStrategy.XLA_REDUCE_SCATTER else "pallas"
        )
        timing_root = diagnostic_root / directory / "timing"
        timing_path = timing_root / "result.json"
        cost_path = timing_root / "cost_model.json"
        if (
            file_sha256(timing_path) != authority.timing_result_sha256
            or file_sha256(cost_path) != authority.cost_model_sha256
        ):
            raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_CALIBRATION_HASH_MISMATCH")
        diagnostics[authority.strategy] = (
            MatmulRunResult.model_validate_json(timing_path.read_text()),
            CostModelReport.model_validate_json(cost_path.read_text()),
        )
    return build_matmul_collective_repeat_prediction(
        contract,
        diagnostics,
        confirmation,
    )


def validate_matmul_collective_repeat_prediction(
    report: MatmulCollectiveRepeatPredictionReport,
    *,
    diagnostic_root: Path,
    diagnostic_archive: Path,
    confirmation_root: Path,
    confirmation_archive: Path,
    confirmation_contract: MatmulCollectiveConfirmationContract,
    contract: MatmulCollectiveRepeatPredictionContract,
) -> None:
    expected = derive_matmul_collective_repeat_prediction(
        diagnostic_root,
        diagnostic_archive,
        confirmation_root,
        confirmation_archive,
        confirmation_contract,
        contract,
    )
    if report != expected:
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_REPORT_REPLAY_MISMATCH")


def write_matmul_collective_repeat_prediction(
    output: Path,
    *,
    diagnostic_root: Path,
    diagnostic_archive: Path,
    confirmation_root: Path,
    confirmation_archive: Path,
    confirmation_contract: MatmulCollectiveConfirmationContract,
    contract: MatmulCollectiveRepeatPredictionContract,
) -> MatmulCollectiveRepeatPredictionReport:
    roots = (diagnostic_root.resolve(), confirmation_root.resolve())
    resolved_output = output.resolve()
    if any(resolved_output == root or root in resolved_output.parents for root in roots):
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_OUTPUT_OVERLAPS_EVIDENCE")
    if output.exists() or output.is_symlink():
        raise ValueError("MATMUL_COLLECTIVE_REPEAT_PREDICTION_OUTPUT_EXISTS")
    report = derive_matmul_collective_repeat_prediction(
        diagnostic_root,
        diagnostic_archive,
        confirmation_root,
        confirmation_archive,
        confirmation_contract,
        contract,
    )
    write_json(output, report.model_dump(mode="json"))
    return report


def _add_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diagnostic-root", required=True, type=Path)
    parser.add_argument("--diagnostic-archive", required=True, type=Path)
    parser.add_argument("--confirmation-root", required=True, type=Path)
    parser.add_argument("--confirmation-archive", required=True, type=Path)
    parser.add_argument("--confirmation-contract", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    _add_evidence_arguments(evaluate)
    evaluate.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("report", type=Path)
    _add_evidence_arguments(verify)
    return parser


def main() -> None:
    args = _parser().parse_args()
    confirmation_contract = MatmulCollectiveConfirmationContract.model_validate_json(
        args.confirmation_contract.read_text()
    )
    contract = MatmulCollectiveRepeatPredictionContract.model_validate_json(
        args.contract.read_text()
    )
    common = {
        "diagnostic_root": args.diagnostic_root,
        "diagnostic_archive": args.diagnostic_archive,
        "confirmation_root": args.confirmation_root,
        "confirmation_archive": args.confirmation_archive,
        "confirmation_contract": confirmation_contract,
        "contract": contract,
    }
    if args.command == "evaluate":
        report = write_matmul_collective_repeat_prediction(args.output, **common)
        action = "EVALUATED"
    else:
        report = MatmulCollectiveRepeatPredictionReport.model_validate_json(args.report.read_text())
        validate_matmul_collective_repeat_prediction(report, **common)
        action = "REPLAYED"
    print(
        f"MATMUL_COLLECTIVE_REPEAT_PREDICTION_{action} "
        f"contract_id={report.contract_id} "
        f"ranking_agrees={report.strategy_ranking_agrees} "
        f"scope={report.status}"
    )


if __name__ == "__main__":
    main()
