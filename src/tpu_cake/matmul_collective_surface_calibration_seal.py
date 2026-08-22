from __future__ import annotations

import hashlib
import math
import struct
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import model_identity_sha256
from tpu_cake.matmul_collective_surface_calibration_evidence import (
    MatmulCollectiveSurfaceCalibrationEvidence,
    SurfaceCalibrationCallSample,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    MatmulCollectiveSurfaceCalibrationProtocol,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceArmPlan,
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceModel,
    SurfaceCalibrationObservation,
    _nonnegative_affine_fit,
    derive_matmul_collective_surface_design_report,
    fit_surface_model,
)
from tpu_cake.runner import MatmulCollectiveStrategy

MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SEALED_EVIDENCE_SCHEMA = (
    "matmul-collective-surface-calibration-sealed-evidence-v1"
)
_ARRAY_HASH_SCHEMA = "matmul-collective-surface-bootstrap-array-v1"


class SurfaceCalibrationArmObservation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    round_medians_ns: tuple[int, ...] = Field(min_length=16, max_length=16)
    median_ns: float = Field(gt=0)

    @model_validator(mode="after")
    def median_is_exact(self) -> SurfaceCalibrationArmObservation:
        if any(value <= 0 for value in self.round_medians_ns):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROUND_MEDIAN_INVALID")
        ordered = sorted(self.round_medians_ns)
        expected = (ordered[7] + ordered[8]) / 2
        if self.median_ns != expected:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARM_MEDIAN_MISMATCH")
        return self


class SurfaceHoldoutPredictionInterval(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    point_prediction_ns: float = Field(gt=0)
    lower_99pct_ns: float = Field(gt=0)
    upper_99pct_ns: float = Field(gt=0)
    relative_width: float = Field(ge=0)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> SurfaceHoldoutPredictionInterval:
        expected = (self.upper_99pct_ns - self.lower_99pct_ns) / self.point_prediction_ns
        if self.lower_99pct_ns > self.upper_99pct_ns or self.relative_width != expected:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_INTERVAL_INVALID")
        return self


class SurfaceHoldoutStrategyPrediction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    scenario_name: str
    point_improvement: float
    lower_99pct_improvement: float
    upper_99pct_improvement: float

    @model_validator(mode="after")
    def interval_is_ordered(self) -> SurfaceHoldoutStrategyPrediction:
        if self.lower_99pct_improvement > self.upper_99pct_improvement:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_IMPROVEMENT_INTERVAL_INVALID")
        return self


class MatmulCollectiveSurfaceCalibrationSealedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-sealed-evidence-v1"] = (
        MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SEALED_EVIDENCE_SCHEMA
    )
    seal_schema: Literal["matmul-collective-surface-calibration-seal-v1"] = (
        "matmul-collective-surface-calibration-seal-v1"
    )
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: tuple[SurfaceCalibrationArmObservation, ...] = Field(
        min_length=32,
        max_length=32,
    )
    model: MatmulCollectiveSurfaceModel
    bootstrap_sample_count: Literal[10000] = 10000
    bootstrap_array_hash_schema: Literal["matmul-collective-surface-bootstrap-array-v1"] = (
        _ARRAY_HASH_SCHEMA
    )
    bootstrap_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bootstrap_coefficient_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bootstrap_prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bootstrap_improvement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    holdout_predictions: tuple[SurfaceHoldoutPredictionInterval, ...] = Field(
        min_length=8,
        max_length=8,
    )
    strategy_predictions: tuple[SurfaceHoldoutStrategyPrediction, ...] = Field(
        min_length=4,
        max_length=4,
    )
    width_gate_passed: bool
    holdout_authorization: Literal[
        "pending_independent_replay",
        "denied_prediction_interval_width",
    ]

    @model_validator(mode="after")
    def authorization_matches_width_gate(
        self,
    ) -> MatmulCollectiveSurfaceCalibrationSealedEvidence:
        expected = (
            "pending_independent_replay"
            if self.width_gate_passed
            else "denied_prediction_interval_width"
        )
        if self.holdout_authorization != expected:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_AUTHORIZATION_MISMATCH")
        return self

    @computed_field
    @property
    def seal_sha256(self) -> str:
        return model_identity_sha256(self)


def derive_surface_calibration_observations(
    samples: tuple[SurfaceCalibrationCallSample, ...],
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCalibrationArmObservation, ...]:
    groups: dict[tuple[str, MatmulCollectiveStrategy, int], list[int]] = {}
    for sample in samples:
        groups.setdefault(
            (sample.scenario_name, sample.strategy, sample.round_index),
            [],
        ).append(sample.duration_ns)
    design_report = derive_matmul_collective_surface_design_report(design)
    observations = []
    for arm in design_report.calibration_arms:
        round_medians = []
        for round_index in range(protocol.paired_rounds):
            values = groups.get((arm.scenario_name, arm.strategy, round_index), [])
            if len(values) != protocol.calls_per_position:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SAMPLE_GROUP_INVALID")
            round_medians.append(sorted(values)[protocol.calls_per_position // 2])
        ordered = sorted(round_medians)
        observations.append(
            SurfaceCalibrationArmObservation(
                scenario_name=arm.scenario_name,
                strategy=arm.strategy,
                round_medians_ns=tuple(round_medians),
                median_ns=(ordered[7] + ordered[8]) / 2,
            )
        )
    if len(groups) != len(observations) * protocol.paired_rounds:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SAMPLE_GROUP_INVENTORY")
    return tuple(observations)


def derive_surface_calibration_seal(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> MatmulCollectiveSurfaceCalibrationSealedEvidence:
    if (
        evidence.protocol_id != protocol.protocol_id
        or evidence.design_id != design.design_id
        or evidence.correctness_parent_receipt_sha256 != protocol.correctness_parent.receipt_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SEAL_AUTHORITY_MISMATCH")
    observations = derive_surface_calibration_observations(evidence.samples, protocol, design)
    return seal_surface_calibration_observations(
        observations,
        protocol,
        design,
        calibration_evidence_sha256=evidence.evidence_sha256,
    )


def seal_surface_calibration_observations(
    observations: tuple[SurfaceCalibrationArmObservation, ...],
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
    *,
    calibration_evidence_sha256: str,
) -> MatmulCollectiveSurfaceCalibrationSealedEvidence:
    design_report = derive_matmul_collective_surface_design_report(design)
    expected = tuple(
        (value.scenario_name, value.strategy) for value in design_report.calibration_arms
    )
    if tuple((value.scenario_name, value.strategy) for value in observations) != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_OBSERVATION_INVENTORY")
    model = fit_surface_model(
        design,
        tuple(
            SurfaceCalibrationObservation(
                scenario_name=value.scenario_name,
                strategy=value.strategy,
                median_ns=value.median_ns,
            )
            for value in observations
        ),
    )
    calibration_matrix = np.asarray(
        [_feature_row(design, value) for value in design_report.calibration_arms],
        dtype=np.float64,
    )
    holdout_arms = tuple(
        value for value in design_report.arms if value not in design_report.calibration_arms
    )
    holdout_matrix = np.asarray(
        [_feature_row(design, value) for value in holdout_arms],
        dtype=np.float64,
    )
    round_medians = np.asarray(
        [value.round_medians_ns for value in observations],
        dtype=np.float64,
    )
    bootstrap_indices = np.asarray(
        [
            protocol.bootstrap_round_indices(replicate)
            for replicate in range(protocol.coefficient_bootstrap_samples)
        ],
        dtype=np.uint8,
    )
    bootstrap_measured = np.median(round_medians[:, bootstrap_indices], axis=2).T
    bootstrap_coefficients = np.empty((protocol.coefficient_bootstrap_samples, 6), dtype=np.float64)
    for index, measured in enumerate(bootstrap_measured):
        bootstrap_coefficients[index] = _nonnegative_affine_fit(calibration_matrix, measured)
    bootstrap_predictions = bootstrap_coefficients @ holdout_matrix.T
    if not np.all(np.isfinite(bootstrap_predictions)) or np.any(bootstrap_predictions <= 0):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_BOOTSTRAP_PREDICTION_INVALID")
    point_predictions = holdout_matrix @ np.asarray(model.coefficients, dtype=np.float64)
    lower, upper = np.quantile(
        bootstrap_predictions,
        (0.005, 0.995),
        axis=0,
        method="linear",
    )
    intervals = tuple(
        SurfaceHoldoutPredictionInterval(
            scenario_name=arm.scenario_name,
            strategy=arm.strategy,
            point_prediction_ns=float(point),
            lower_99pct_ns=float(low),
            upper_99pct_ns=float(high),
            relative_width=float((high - low) / point),
        )
        for arm, point, low, high in zip(
            holdout_arms,
            point_predictions,
            lower,
            upper,
            strict=True,
        )
    )
    point_pairs = point_predictions.reshape(4, 2)
    bootstrap_pairs = bootstrap_predictions.reshape(protocol.coefficient_bootstrap_samples, 4, 2)
    point_improvements = (point_pairs[:, 0] - point_pairs[:, 1]) / point_pairs[:, 0]
    bootstrap_improvements = (
        bootstrap_pairs[:, :, 0] - bootstrap_pairs[:, :, 1]
    ) / bootstrap_pairs[:, :, 0]
    improvement_lower, improvement_upper = np.quantile(
        bootstrap_improvements,
        (0.005, 0.995),
        axis=0,
        method="linear",
    )
    strategy_predictions = tuple(
        SurfaceHoldoutStrategyPrediction(
            scenario_name=design.holdout_scenarios[index].name,
            point_improvement=float(point_improvements[index]),
            lower_99pct_improvement=float(improvement_lower[index]),
            upper_99pct_improvement=float(improvement_upper[index]),
        )
        for index in range(4)
    )
    width_gate_passed = _prediction_width_gate(
        intervals,
        protocol.maximum_holdout_prediction_ci_relative_width,
    )
    return MatmulCollectiveSurfaceCalibrationSealedEvidence(
        protocol_id=protocol.protocol_id,
        design_id=design.design_id,
        correctness_parent_receipt_sha256=protocol.correctness_parent.receipt_sha256,
        calibration_evidence_sha256=calibration_evidence_sha256,
        observations=observations,
        model=model,
        bootstrap_sample_count=protocol.coefficient_bootstrap_samples,
        bootstrap_array_hash_schema=_ARRAY_HASH_SCHEMA,
        bootstrap_index_sha256=_array_sha256(bootstrap_indices, dtype=np.dtype("u1")),
        bootstrap_coefficient_sha256=_array_sha256(
            bootstrap_coefficients,
            dtype=np.dtype("<f8"),
        ),
        bootstrap_prediction_sha256=_array_sha256(
            bootstrap_predictions,
            dtype=np.dtype("<f8"),
        ),
        bootstrap_improvement_sha256=_array_sha256(
            bootstrap_improvements,
            dtype=np.dtype("<f8"),
        ),
        holdout_predictions=intervals,
        strategy_predictions=strategy_predictions,
        width_gate_passed=width_gate_passed,
        holdout_authorization=(
            "pending_independent_replay"
            if width_gate_passed
            else "denied_prediction_interval_width"
        ),
    )


def validate_surface_calibration_seal(
    seal: MatmulCollectiveSurfaceCalibrationSealedEvidence,
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> None:
    expected = derive_surface_calibration_seal(evidence, protocol, design)
    if seal != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SEAL_REPLAY_MISMATCH")


def _feature_row(
    contract: MatmulCollectiveSurfaceDesignContract,
    arm: MatmulCollectiveSurfaceArmPlan,
) -> tuple[float, float, float, float, float, float]:
    first = arm.strategy is contract.strategies[0]
    divisor = contract.feature_scale_divisor_ns
    compute = float(arm.compute_time_floor_ns) / divisor
    hbm = float(arm.hbm_time_floor_ns) / divisor
    ici = float(arm.ici_time_floor_ns) / divisor
    return (
        float(first),
        float(not first),
        compute,
        hbm,
        ici if first else 0.0,
        0.0 if first else ici,
    )


def _prediction_width_gate(
    intervals: tuple[SurfaceHoldoutPredictionInterval, ...],
    maximum_relative_width: float,
) -> bool:
    return len(intervals) == 8 and all(
        math.isfinite(value.relative_width) and value.relative_width <= maximum_relative_width
        for value in intervals
    )


def _array_sha256(value: np.ndarray, *, dtype: np.dtype) -> str:
    array = np.asarray(value, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(_ARRAY_HASH_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(struct.pack(">Q", array.ndim))
    for dimension in array.shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
