from __future__ import annotations

import os
import time
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.artifacts import file_sha256 as _sha256
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import model_identity_sha256
from tpu_cake.metrics import MeasurementKind, Metric, Unit
from tpu_cake.runner import RunMode
from tpu_cake.seqax_cost_model import SeqaxCostModelReport
from tpu_cake.seqax_surface import seqax_forward_workload_surface
from tpu_cake.seqax_surface_profile import (
    SeqaxSurfaceProfileReceipt,
    validate_seqax_surface_profile_receipt,
)

SEQAX_COST_CALIBRATION_SCHEMA = "seqax-cost-calibration-v1"


class SeqaxCostCalibrationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^seqax-cost-calibration-v1$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_profile_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenarios: tuple[str, ...] = Field(min_length=3)
    primary_mode: RunMode
    replication_mode: RunMode
    response_metric: str = Field(min_length=1)
    baseline_metric: str = Field(min_length=1)
    materialization_metric: str = Field(min_length=1)
    predictor: str = Field(min_length=1)
    fit_formula: str = Field(min_length=1)
    maximum_in_surface_relative_error: Decimal = Field(gt=0, lt=1)
    maximum_cross_mode_relative_difference: Decimal = Field(gt=0, lt=1)
    decimal_places: int = Field(ge=6, le=18)
    runtime: RuntimeIdentity
    device_kind: str = Field(min_length=1)
    device_count: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    input_placement: str = Field(min_length=1)
    execution_scope: str = Field(min_length=1)
    measurement_scope: str = Field(min_length=1)
    acceptance_scope: str = Field(min_length=1)

    @model_validator(mode="after")
    def protocol_is_well_formed(self) -> SeqaxCostCalibrationContract:
        if (
            len(self.scenarios) != len(set(self.scenarios))
            or self.primary_mode is not RunMode.TRACE
            or self.replication_mode is not RunMode.COUNTERS
            or self.predictor != "declared_layer_count"
            or self.fit_formula
            != "measured_median_ns=idealized_floor_ns+fixed_residual_ns+layers*per_layer_residual_ns"
            or (self.device_kind, self.device_count) != ("TPU7x", 8)
            or self.measured_iterations != 50
            or self.input_placement != "resident-named-sharding-before-warmup"
            or self.execution_scope != "multi-device-local-shards"
            or self.measurement_scope != "profiled-device-module-self-time"
            or self.acceptance_scope != "descriptive-in-surface-fit-only"
        ):
            raise ValueError("SEQAX_COST_CALIBRATION_PROTOCOL_MISMATCH")
        return self

    @computed_field
    @property
    def contract_id(self) -> str:
        return model_identity_sha256(self)


class SeqaxCostCalibrationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: str = Field(min_length=1)
    layers: int = Field(gt=0)
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counter_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    idealized_floor_ns: Decimal = Field(gt=0)
    materialized_hbm_scenario_ns: Decimal = Field(gt=0)
    trace_median_ns: Decimal = Field(gt=0)
    trace_p90_ns: Decimal = Field(gt=0)
    counter_median_ns: Decimal = Field(gt=0)
    predicted_trace_median_ns: Decimal = Field(gt=0)
    residual_ns: Decimal
    absolute_relative_error: Decimal = Field(ge=0)
    cross_mode_relative_difference: Decimal = Field(ge=0)
    measured_to_idealized_ratio: Decimal = Field(gt=0)
    measured_to_materialized_ratio: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def values_are_finite(self) -> SeqaxCostCalibrationPoint:
        values = (
            self.idealized_floor_ns,
            self.materialized_hbm_scenario_ns,
            self.trace_median_ns,
            self.trace_p90_ns,
            self.counter_median_ns,
            self.predicted_trace_median_ns,
            self.residual_ns,
            self.absolute_relative_error,
            self.cross_mode_relative_difference,
            self.measured_to_idealized_ratio,
            self.measured_to_materialized_ratio,
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("SEQAX_COST_CALIBRATION_NONFINITE_POINT")
        if self.trace_p90_ns < self.trace_median_ns:
            raise ValueError("SEQAX_COST_CALIBRATION_P90_BELOW_MEDIAN")
        return self


class SeqaxCostCalibrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^seqax-cost-calibration-v1$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_profile_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeIdentity
    device_kind: str = Field(min_length=1)
    device_count: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    input_placement: str = Field(min_length=1)
    execution_scope: str = Field(min_length=1)
    measurement_scope: str = Field(min_length=1)
    fit_formula: str = Field(min_length=1)
    fixed_residual_ns: Decimal
    per_layer_residual_ns: Decimal
    points: tuple[SeqaxCostCalibrationPoint, ...] = Field(min_length=3)
    maximum_in_surface_relative_error: Decimal = Field(ge=0)
    maximum_cross_mode_relative_difference: Decimal = Field(ge=0)
    fit_within_declared_surface_error: bool
    cross_mode_consistent: bool
    predictive_validation: bool
    status: str = Field(pattern=r"^descriptive-in-surface-fit-only$")
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def report_is_consistent(self) -> SeqaxCostCalibrationReport:
        values = (
            self.fixed_residual_ns,
            self.per_layer_residual_ns,
            self.maximum_in_surface_relative_error,
            self.maximum_cross_mode_relative_difference,
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("SEQAX_COST_CALIBRATION_NONFINITE_REPORT")
        if self.predictive_validation:
            raise ValueError("SEQAX_COST_CALIBRATION_HAS_NO_HELD_OUT_PREDICTIVE_VALIDATION")
        return self


def default_seqax_cost_calibration_contract() -> SeqaxCostCalibrationContract:
    surface = seqax_forward_workload_surface()
    return SeqaxCostCalibrationContract(
        schema_version=SEQAX_COST_CALIBRATION_SCHEMA,
        surface_id=surface.surface_id,
        surface_profile_receipt_sha256=(
            "2b52109637c779b104b895f935652edab50a1de697477cba3b40ea225e3de636"
        ),
        surface_receipt_sha256=("61ca1f2b52975b384759fd9c1b68b5bda6bbd3be0e700a59486d596f4360c2c8"),
        scenarios=tuple(scenario.name for scenario in surface.scenarios),
        primary_mode=RunMode.TRACE,
        replication_mode=RunMode.COUNTERS,
        response_metric="median_compiled_forward_duration",
        baseline_metric="seqax_idealized_time_floor",
        materialization_metric="seqax_materialized_hbm_time",
        predictor="declared_layer_count",
        fit_formula=(
            "measured_median_ns=idealized_floor_ns+fixed_residual_ns+layers*per_layer_residual_ns"
        ),
        maximum_in_surface_relative_error=Decimal("0.02"),
        maximum_cross_mode_relative_difference=Decimal("0.10"),
        decimal_places=12,
        runtime=RuntimeIdentity(
            python="3.12.3",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla=" --xla_tpu_use_enhanced_launch_barrier=true",
        ),
        device_kind="TPU7x",
        device_count=8,
        measured_iterations=50,
        input_placement="resident-named-sharding-before-warmup",
        execution_scope="multi-device-local-shards",
        measurement_scope="profiled-device-module-self-time",
        acceptance_scope="descriptive-in-surface-fit-only",
    )


def _metric(metrics: tuple[Metric, ...], name: str, kind: MeasurementKind) -> Metric:
    matches = tuple(metric for metric in metrics if metric.name == name)
    if len(matches) != 1:
        raise ValueError(f"SEQAX_COST_CALIBRATION_METRIC_SET_MISMATCH name={name}")
    metric = matches[0]
    if metric.kind is not kind or metric.quantity.unit is not Unit.NANOSECOND:
        raise ValueError(f"SEQAX_COST_CALIBRATION_METRIC_CONTRACT_MISMATCH name={name}")
    if len(metric.sources) != 1:
        raise ValueError(f"SEQAX_COST_CALIBRATION_METRIC_SOURCE_MISMATCH name={name}")
    return metric


def _quantize(value: Decimal, decimal_places: int) -> Decimal:
    quantum = Decimal(1).scaleb(-decimal_places)
    return value.quantize(quantum, rounding=ROUND_HALF_EVEN)


def _fit_coefficients(
    layers: tuple[int, ...],
    measured: tuple[Decimal, ...],
    floors: tuple[Decimal, ...],
    *,
    decimal_places: int,
) -> tuple[Decimal, Decimal]:
    if not (len(layers) == len(measured) == len(floors)) or len(layers) < 3:
        raise ValueError("SEQAX_COST_CALIBRATION_POINT_COUNT_MISMATCH")
    if len(set(layers)) < 2:
        raise ValueError("SEQAX_COST_CALIBRATION_NEEDS_MULTIPLE_LAYER_COUNTS")
    with localcontext() as context:
        context.prec = 60
        count = Decimal(len(layers))
        x_values = tuple(Decimal(value) for value in layers)
        y_values = tuple(value - floor for value, floor in zip(measured, floors, strict=True))
        mean_x = sum(x_values) / count
        mean_y = sum(y_values) / count
        denominator = sum((value - mean_x) ** 2 for value in x_values)
        if denominator == 0:
            raise ValueError("SEQAX_COST_CALIBRATION_SINGULAR_FIT")
        slope = (
            sum(
                (x_value - mean_x) * (y_value - mean_y)
                for x_value, y_value in zip(x_values, y_values, strict=True)
            )
            / denominator
        )
        intercept = mean_y - slope * mean_x
    return _quantize(intercept, decimal_places), _quantize(slope, decimal_places)


def _relative_difference(left: Decimal, right: Decimal) -> Decimal:
    return abs(left - right) / max(abs(left), abs(right))


def derive_seqax_cost_calibration(
    profile_root: Path,
    contract: SeqaxCostCalibrationContract,
) -> SeqaxCostCalibrationReport:
    canonical_contract = default_seqax_cost_calibration_contract()
    if contract != canonical_contract:
        raise ValueError("SEQAX_COST_CALIBRATION_EXTERNAL_CONTRACT_MISMATCH")
    if profile_root.is_symlink() or not profile_root.is_dir():
        raise ValueError(f"SEQAX_COST_CALIBRATION_ROOT_INVALID path={profile_root}")
    profile_root = profile_root.resolve()
    receipt_path = profile_root / "receipt.json"
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_nlink != 1
        or _sha256(receipt_path) != contract.surface_profile_receipt_sha256
    ):
        raise ValueError("SEQAX_COST_CALIBRATION_PROFILE_RECEIPT_MISMATCH")
    receipt = SeqaxSurfaceProfileReceipt.model_validate_json(receipt_path.read_text())
    validate_seqax_surface_profile_receipt(receipt, root=profile_root)
    surface_receipt_path = profile_root / "surface" / "receipt.json"
    if (
        receipt.surface_id != contract.surface_id
        or receipt.surface_receipt_sha256 != contract.surface_receipt_sha256
        or _sha256(surface_receipt_path) != contract.surface_receipt_sha256
    ):
        raise ValueError("SEQAX_COST_CALIBRATION_SURFACE_IDENTITY_MISMATCH")

    surface = seqax_forward_workload_surface()
    scenarios = {scenario.name: scenario for scenario in surface.scenarios}
    if tuple(scenarios) != contract.scenarios:
        raise ValueError("SEQAX_COST_CALIBRATION_SCENARIO_SET_MISMATCH")
    result_by_key = {
        (result.invocation.scenario, result.invocation.mode): result for result in receipt.results
    }
    expected_result_keys = {
        (scenario, mode)
        for scenario in contract.scenarios
        for mode in (contract.primary_mode, contract.replication_mode)
    }
    if len(result_by_key) != len(receipt.results) or set(result_by_key) != expected_result_keys:
        raise ValueError("SEQAX_COST_CALIBRATION_RESULT_SET_MISMATCH")
    if any(
        result.invocation.runtime != contract.runtime
        or result.invocation.device_kind != contract.device_kind
        or result.invocation.device_count != contract.device_count
        or result.invocation.measured_iterations != contract.measured_iterations
        or result.invocation.input_placement != contract.input_placement
        or result.invocation.execution_scope != contract.execution_scope
        for result in receipt.results
    ):
        raise ValueError("SEQAX_COST_CALIBRATION_EXECUTION_IDENTITY_MISMATCH")
    raw_points: list[dict[str, object]] = []
    for name in contract.scenarios:
        scenario = scenarios[name]
        trace = result_by_key[(name, contract.primary_mode)]
        counters = result_by_key[(name, contract.replication_mode)]
        cost_path = profile_root / "surface" / "cost" / f"{name}.json"
        cost = SeqaxCostModelReport.model_validate_json(cost_path.read_text())
        if (
            cost.schedule_sha256 != trace.invocation.schedule_sha256
            or trace.invocation.schedule_sha256 != counters.invocation.schedule_sha256
        ):
            raise ValueError(f"SEQAX_COST_CALIBRATION_SCHEDULE_MISMATCH scenario={name}")
        idealized = _metric(
            cost.metrics,
            contract.baseline_metric,
            MeasurementKind.ESTIMATED,
        )
        materialized = _metric(
            cost.metrics,
            contract.materialization_metric,
            MeasurementKind.ESTIMATED,
        )
        trace_median = _metric(
            receipt.metrics,
            f"{name}_{contract.primary_mode.value}_{contract.response_metric}",
            MeasurementKind.DERIVED,
        )
        trace_p90 = _metric(
            receipt.metrics,
            f"{name}_{contract.primary_mode.value}_p90_compiled_forward_duration",
            MeasurementKind.DERIVED,
        )
        counter_median = _metric(
            receipt.metrics,
            f"{name}_{contract.replication_mode.value}_{contract.response_metric}",
            MeasurementKind.DERIVED,
        )
        if trace_p90.sources != trace_median.sources:
            raise ValueError(f"SEQAX_COST_CALIBRATION_TRACE_SOURCE_MISMATCH scenario={name}")
        raw_points.append(
            {
                "scenario": name,
                "layers": scenario.layers,
                "schedule_sha256": cost.schedule_sha256,
                "cost_report_sha256": _sha256(cost_path),
                "trace_source_sha256": trace_median.sources[0].artifact_sha256,
                "counter_source_sha256": counter_median.sources[0].artifact_sha256,
                "idealized_floor_ns": idealized.quantity.value,
                "materialized_hbm_scenario_ns": materialized.quantity.value,
                "trace_median_ns": trace_median.quantity.value,
                "trace_p90_ns": trace_p90.quantity.value,
                "counter_median_ns": counter_median.quantity.value,
            }
        )

    layers = tuple(int(point["layers"]) for point in raw_points)
    measured = tuple(Decimal(point["trace_median_ns"]) for point in raw_points)
    floors = tuple(Decimal(point["idealized_floor_ns"]) for point in raw_points)
    fixed, per_layer = _fit_coefficients(
        layers,
        measured,
        floors,
        decimal_places=contract.decimal_places,
    )
    points = []
    for point in raw_points:
        trace_median = Decimal(point["trace_median_ns"])
        counter_median = Decimal(point["counter_median_ns"])
        floor = Decimal(point["idealized_floor_ns"])
        materialized = Decimal(point["materialized_hbm_scenario_ns"])
        predicted = _quantize(
            floor + fixed + Decimal(point["layers"]) * per_layer,
            contract.decimal_places,
        )
        residual = _quantize(trace_median - predicted, contract.decimal_places)
        points.append(
            SeqaxCostCalibrationPoint(
                **point,
                predicted_trace_median_ns=predicted,
                residual_ns=residual,
                absolute_relative_error=_quantize(
                    abs(residual) / trace_median,
                    contract.decimal_places,
                ),
                cross_mode_relative_difference=_quantize(
                    _relative_difference(trace_median, counter_median),
                    contract.decimal_places,
                ),
                measured_to_idealized_ratio=_quantize(
                    trace_median / floor,
                    contract.decimal_places,
                ),
                measured_to_materialized_ratio=_quantize(
                    trace_median / materialized,
                    contract.decimal_places,
                ),
            )
        )
    maximum_error = max(point.absolute_relative_error for point in points)
    maximum_cross_mode = max(point.cross_mode_relative_difference for point in points)
    return SeqaxCostCalibrationReport(
        schema_version=SEQAX_COST_CALIBRATION_SCHEMA,
        contract_id=contract.contract_id,
        surface_id=contract.surface_id,
        surface_profile_receipt_sha256=contract.surface_profile_receipt_sha256,
        surface_receipt_sha256=contract.surface_receipt_sha256,
        runtime=contract.runtime,
        device_kind=contract.device_kind,
        device_count=contract.device_count,
        measured_iterations=contract.measured_iterations,
        input_placement=contract.input_placement,
        execution_scope=contract.execution_scope,
        measurement_scope=contract.measurement_scope,
        fit_formula=contract.fit_formula,
        fixed_residual_ns=fixed,
        per_layer_residual_ns=per_layer,
        points=tuple(points),
        maximum_in_surface_relative_error=maximum_error,
        maximum_cross_mode_relative_difference=maximum_cross_mode,
        fit_within_declared_surface_error=(
            maximum_error <= contract.maximum_in_surface_relative_error
        ),
        cross_mode_consistent=(
            maximum_cross_mode <= contract.maximum_cross_mode_relative_difference
        ),
        predictive_validation=False,
        status="descriptive-in-surface-fit-only",
        limitations=(
            "The coefficients are fitted to three profiler-instrumented device-module medians, not unprofiled serving latency.",
            "The coefficients are residual associations and have no causal attribution to launch, synchronization, communication, or compute.",
            "The report contains point estimates only; it does not estimate sampling uncertainty or coefficient confidence intervals.",
            "Only two distinct layer counts are present; model-size and sequence-length effects are not identified.",
            "The fit is descriptive on the calibration surface and has no held-out predictive validation.",
            "The advertised compute, HBM, and ICI terms remain lower-bound scenarios rather than measured resource times.",
        ),
    )


def validate_seqax_cost_calibration(
    report: SeqaxCostCalibrationReport,
    *,
    profile_root: Path,
    contract: SeqaxCostCalibrationContract,
) -> None:
    expected = derive_seqax_cost_calibration(profile_root, contract)
    if report != expected:
        raise ValueError("SEQAX_COST_CALIBRATION_REPORT_REPLAY_MISMATCH")


def write_seqax_cost_calibration(
    output: Path,
    *,
    profile_root: Path,
    contract: SeqaxCostCalibrationContract,
) -> SeqaxCostCalibrationReport:
    resolved_output = output.resolve()
    resolved_profile = profile_root.resolve()
    if resolved_output == resolved_profile or resolved_profile in resolved_output.parents:
        raise ValueError("SEQAX_COST_CALIBRATION_OUTPUT_OVERLAPS_PROFILE")
    if output.exists() or output.is_symlink():
        raise ValueError(f"SEQAX_COST_CALIBRATION_OUTPUT_EXISTS path={output}")
    report = derive_seqax_cost_calibration(profile_root, contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    payload = report.model_dump_json(indent=2) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return report
