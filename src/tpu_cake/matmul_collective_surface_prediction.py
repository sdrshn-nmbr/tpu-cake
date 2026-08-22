from __future__ import annotations

import argparse
import itertools
import json
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.cost_model import estimate_distributed_matmul, tpu7x_tensorcore_rates
from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.metrics import MetricSource
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.runner import MatmulCollectiveStrategy
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule

MATMUL_COLLECTIVE_SURFACE_DESIGN_SCHEMA = "matmul-collective-surface-design-v1"
MATMUL_COLLECTIVE_SURFACE_RUNTIME = {
    "python": "3.12.3",
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "libtpu": "0.0.44.1",
    "xla": " --xla_tpu_use_enhanced_launch_barrier=true",
}


class MatmulCollectiveSurfaceSplit(StrEnum):
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"


class MatmulCollectiveSurfaceScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(pattern=r"^(calibration|holdout)-[0-9]+$")
    split: MatmulCollectiveSurfaceSplit
    m: int = Field(gt=0)
    k: int = Field(gt=0)
    n: int = Field(gt=0)
    tile_m: int = Field(gt=0)
    tile_n: int = Field(gt=0)

    @model_validator(mode="after")
    def is_executable_by_both_arms(self) -> MatmulCollectiveSurfaceScenario:
        expected_prefix = f"{self.split.value}-"
        if not self.name.startswith(expected_prefix):
            raise ValueError("Matmul collective surface split/name mismatch")
        if (
            self.m % 16
            or self.k % 1024
            or self.n % 1024
            or self.tile_m != min(128, self.m)
            or self.tile_n != 128
        ):
            raise ValueError("Matmul collective surface shape is not executable by both arms")
        return self


class MatmulCollectivePriorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    m: int = Field(gt=0)
    k: int = Field(gt=0)
    n: int = Field(gt=0)
    evidence_paths: tuple[str, ...] = Field(min_length=1)


class MatmulCollectiveSurfaceDesignContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = MATMUL_COLLECTIVE_SURFACE_DESIGN_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    source_branch: str
    require_origin_main: bool
    compilation_source_root: str
    project: str
    zone: str
    hostname: str
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    runtime: dict[str, str | None]
    mesh_size: int = Field(gt=0)
    input_dtype: str
    output_dtype: str
    strategies: tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy]
    scenarios: tuple[MatmulCollectiveSurfaceScenario, ...] = Field(min_length=20, max_length=20)
    prior_observation_inventory: tuple[MatmulCollectivePriorObservation, ...] = Field(min_length=1)
    physical_features: tuple[str, str, str]
    feature_transform: str
    feature_scale_divisor_ns: float = Field(gt=0)
    fit_rule: str
    prediction_target: str
    prediction_rule_predeclared: bool
    acceptance_bounds_predeclared: bool
    supports_shape_interpolation: bool
    supports_shape_extrapolation: bool
    maximum_condition_number: float = Field(gt=1)
    maximum_compute_hbm_correlation: float = Field(gt=0, lt=1)
    minimum_calibration_arms_per_limiting_resource: int = Field(ge=2)
    maximum_arm_relative_error: float = Field(gt=0, lt=1)
    arm_relative_error_rule: str
    maximum_median_relative_error: float = Field(gt=0, lt=1)
    median_relative_error_rule: str
    maximum_improvement_error_percentage_points: float = Field(gt=0, lt=100)
    improvement_error_rule: str
    strategy_ranking_indifference_band: float = Field(gt=0, lt=1)
    strategy_ranking_rule: str
    coefficient_bootstrap_samples: int = Field(ge=10_000)
    coefficient_bootstrap_seed: int = Field(ge=0)
    maximum_holdout_prediction_ci_relative_width: float = Field(gt=0, lt=1)
    prediction_ci_rule: str
    calibration_warmup_iterations: int = Field(gt=0)
    calibration_calls_per_position: int = Field(ge=3)
    calibration_paired_rounds: int = Field(ge=8)
    holdout_warmup_iterations: int = Field(gt=0)
    holdout_calls_per_position: int = Field(ge=3)
    holdout_paired_rounds: int = Field(ge=16)
    correctness_patterns: tuple[str, ...] = Field(min_length=5, max_length=5)
    timing_oracle_coordinate_count: int = Field(ge=64)
    profile_modes: tuple[str, str]
    profile_repetitions: int = Field(ge=5)
    profile_execution_separate_from_timing: bool
    trace_and_counter_fields_available_to_fit: bool
    scaling_book_authorities: tuple[str, ...] = Field(min_length=2)
    allow_early_stopping: bool
    allow_retry: bool
    allow_outlier_removal: bool
    candidates_resident_together: bool
    predictions_sealed_before_holdout: bool
    calibration_seal_schema: str
    one_shot_attempt_ledger: bool
    allow_calibration_refit_after_holdout: bool
    source_manifest_rule: str
    compiler_capture_repetitions: int = Field(ge=2)
    require_stable_compiler_semantic_hashes: bool
    compiler_failure_policy: str
    paired_order_rule: str

    @property
    def calibration_scenarios(self) -> tuple[MatmulCollectiveSurfaceScenario, ...]:
        return tuple(
            value
            for value in self.scenarios
            if value.split is MatmulCollectiveSurfaceSplit.CALIBRATION
        )

    @property
    def holdout_scenarios(self) -> tuple[MatmulCollectiveSurfaceScenario, ...]:
        return tuple(
            value for value in self.scenarios if value.split is MatmulCollectiveSurfaceSplit.HOLDOUT
        )

    @model_validator(mode="after")
    def is_canonical(self) -> MatmulCollectiveSurfaceDesignContract:
        prior_shapes = {(value.m, value.k, value.n) for value in self.prior_observation_inventory}
        holdout_shapes = {(value.m, value.k, value.n) for value in self.holdout_scenarios}
        if prior_shapes & holdout_shapes:
            raise ValueError("Matmul collective surface holdout was previously observed")
        expected = default_matmul_collective_surface_design_contract_payload()
        observed = self.model_dump(mode="json", exclude_computed_fields=True)
        if observed != expected:
            raise ValueError("Matmul collective surface scenario inventory or protocol mismatch")
        return self

    @computed_field
    @property
    def design_id(self) -> str:
        return model_identity_sha256(self)


class MatmulCollectiveSurfaceArmPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    split: MatmulCollectiveSurfaceSplit
    strategy: MatmulCollectiveStrategy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_bytes: int = Field(gt=0)
    output_local_shape: tuple[int, int]
    operations_per_device: int = Field(gt=0)
    hbm_bytes_per_device: int = Field(gt=0)
    ici_bytes_per_device: int = Field(gt=0)
    peak_live_vmem_bytes_per_device: int = Field(gt=0)
    collective_hbm_scratch_bytes_per_device: int = Field(ge=0)
    collective_vmem_scratch_bytes_per_device: int = Field(ge=0)
    collective_dma_semaphore_count: int = Field(ge=0)
    collective_capacity_semaphore_count: int = Field(ge=0)
    collective_startup_semaphore_count: int = Field(ge=0)
    arithmetic_intensity: Decimal = Field(gt=0)
    compute_time_floor_ns: Decimal = Field(gt=0)
    hbm_time_floor_ns: Decimal = Field(gt=0)
    ici_time_floor_ns: Decimal = Field(gt=0)
    idealized_time_floor_ns: Decimal = Field(gt=0)
    serial_resource_scenario_ns: Decimal = Field(gt=0)
    predicted_limiting_resource: str = Field(pattern=r"^(compute|hbm|ici)$")


class MatmulCollectiveSurfaceDesignReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = MATMUL_COLLECTIVE_SURFACE_DESIGN_SCHEMA
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    arms: tuple[MatmulCollectiveSurfaceArmPlan, ...] = Field(min_length=40, max_length=40)
    calibration_feature_rank: int = Field(ge=0)
    calibration_condition_number: float = Field(gt=0)
    calibration_compute_hbm_correlation: float = Field(ge=-1, le=1)
    holdouts_inside_calibration_hull: bool
    holdouts_inside_physical_feature_hull: bool
    physical_authority: str
    measured_performance_winner: None = None
    prospective_validation: bool = False

    @property
    def calibration_arms(self) -> tuple[MatmulCollectiveSurfaceArmPlan, ...]:
        return tuple(
            value for value in self.arms if value.split is MatmulCollectiveSurfaceSplit.CALIBRATION
        )


class SurfaceCalibrationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    median_ns: float = Field(gt=0)


class MatmulCollectiveSurfaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    coefficient_names: tuple[str, str, str, str, str, str]
    coefficients: tuple[float, float, float, float, float, float]
    calibration_predictions_ns: tuple[float, ...] = Field(min_length=32, max_length=32)
    calibration_relative_errors: tuple[float, ...] = Field(min_length=32, max_length=32)
    maximum_calibration_relative_error: float = Field(ge=0)
    median_calibration_relative_error: float = Field(ge=0)

    @model_validator(mode="after")
    def coefficients_are_nonnegative(self) -> MatmulCollectiveSurfaceModel:
        values = (*self.coefficients, *self.calibration_predictions_ns)
        if not all(np.isfinite(value) for value in values) or any(
            value < 0 for value in self.coefficients
        ):
            raise ValueError("Matmul collective surface model must be finite and nonnegative")
        return self


def default_matmul_collective_surface_design_contract_payload() -> dict[str, object]:
    scenarios = (
        ("calibration-0", "calibration", 16, 16384, 1024),
        ("calibration-1", "calibration", 16, 131072, 1024),
        ("calibration-2", "calibration", 128, 16384, 1024),
        ("calibration-3", "calibration", 128, 131072, 1024),
        ("calibration-4", "calibration", 512, 16384, 1024),
        ("calibration-5", "calibration", 512, 131072, 1024),
        ("calibration-6", "calibration", 768, 16384, 1024),
        ("calibration-7", "calibration", 768, 131072, 1024),
        ("calibration-8", "calibration", 16, 16384, 2048),
        ("calibration-9", "calibration", 16, 81920, 2048),
        ("calibration-10", "calibration", 128, 16384, 2048),
        ("calibration-11", "calibration", 128, 81920, 2048),
        ("calibration-12", "calibration", 512, 16384, 2048),
        ("calibration-13", "calibration", 512, 81920, 2048),
        ("calibration-14", "calibration", 768, 16384, 2048),
        ("calibration-15", "calibration", 768, 81920, 2048),
        ("holdout-0", "holdout", 32, 49152, 1024),
        ("holdout-1", "holdout", 96, 24576, 2048),
        ("holdout-2", "holdout", 640, 98304, 1024),
        ("holdout-3", "holdout", 640, 32768, 2048),
    )
    return {
        "schema_version": MATMUL_COLLECTIVE_SURFACE_DESIGN_SCHEMA,
        "identity_schema": SEMANTIC_IDENTITY_SCHEMA,
        "source_branch": "main",
        "require_origin_main": True,
        "compilation_source_root": "/home/sudarshan/tpu-cake-main",
        "project": "astral-medley-465922-b2",
        "zone": "us-central1-c",
        "hostname": "tpu-cake-v7x-rsag-wx7r",
        "backend": "tpu",
        "device_kind": "TPU7x",
        "device_count": 8,
        "runtime": MATMUL_COLLECTIVE_SURFACE_RUNTIME,
        "mesh_size": 8,
        "input_dtype": "bfloat16",
        "output_dtype": "float32",
        "strategies": [
            MatmulCollectiveStrategy.XLA_REDUCE_SCATTER.value,
            MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING.value,
        ],
        "scenarios": [
            {
                "name": name,
                "split": split,
                "m": m,
                "k": k,
                "n": n,
                "tile_m": min(128, m),
                "tile_n": 128,
            }
            for name, split, m, k, n in scenarios
        ],
        "prior_observation_inventory": [
            {
                "m": 1024,
                "k": 65536,
                "n": 1024,
                "evidence_paths": [
                    "contracts/matmul-collective-confirmation-v1.json",
                    "contracts/matmul-tile-search-tpu7x.json",
                ],
            }
        ],
        "physical_features": [
            "compute_time_floor_ns",
            "hbm_time_floor_ns",
            "ici_time_floor_ns",
        ],
        "feature_transform": "intercepts-plus-physical-floor-ns-divided-by-1000-v1",
        "feature_scale_divisor_ns": 1000.0,
        "fit_rule": "joint-nonnegative-affine-shared-compute-hbm-strategy-ici-v1",
        "prediction_target": "median-of-paired-round-medians-host-latency-ns",
        "prediction_rule_predeclared": True,
        "acceptance_bounds_predeclared": True,
        "supports_shape_interpolation": True,
        "supports_shape_extrapolation": False,
        "maximum_condition_number": 20.0,
        "maximum_compute_hbm_correlation": 0.90,
        "minimum_calibration_arms_per_limiting_resource": 4,
        "maximum_arm_relative_error": 0.15,
        "arm_relative_error_rule": "absolute-predicted-minus-measured-over-measured-per-holdout-arm-v1",
        "maximum_median_relative_error": 0.10,
        "median_relative_error_rule": "median-arm-relative-error-over-all-eight-holdout-arms-v1",
        "maximum_improvement_error_percentage_points": 5.0,
        "improvement_error_rule": "absolute-predicted-minus-observed-relative-strategy-improvement-times-100-per-holdout-shape-v1",
        "strategy_ranking_indifference_band": 0.01,
        "strategy_ranking_rule": "required-only-when-observed-paired-99pct-ci-excludes-indifference-band-v1",
        "coefficient_bootstrap_samples": 10000,
        "coefficient_bootstrap_seed": 17012026,
        "maximum_holdout_prediction_ci_relative_width": 0.20,
        "prediction_ci_rule": "99pct-percentile-bootstrap-calibration-round-resample-per-holdout-arm-v1",
        "calibration_warmup_iterations": 10,
        "calibration_calls_per_position": 5,
        "calibration_paired_rounds": 16,
        "holdout_warmup_iterations": 20,
        "holdout_calls_per_position": 5,
        "holdout_paired_rounds": 32,
        "correctness_patterns": [
            "constant",
            "one-hot-stripes",
            "signed-periodic",
            "block-diagonal",
            "low-rank",
        ],
        "timing_oracle_coordinate_count": 128,
        "profile_modes": ["trace", "counters"],
        "profile_repetitions": 20,
        "profile_execution_separate_from_timing": True,
        "trace_and_counter_fields_available_to_fit": False,
        "scaling_book_authorities": [
            "https://jax-ml.github.io/scaling-book/roofline/",
            "https://jax-ml.github.io/scaling-book/inference/#appendix-c-latency-bound-communications",
            "https://jax-ml.github.io/scaling-book/sharding/",
        ],
        "allow_early_stopping": False,
        "allow_retry": False,
        "allow_outlier_removal": False,
        "candidates_resident_together": True,
        "predictions_sealed_before_holdout": True,
        "calibration_seal_schema": "matmul-collective-surface-calibration-seal-v1",
        "one_shot_attempt_ledger": True,
        "allow_calibration_refit_after_holdout": False,
        "source_manifest_rule": "explicit-executable-dependency-closure-v1",
        "compiler_capture_repetitions": 2,
        "require_stable_compiler_semantic_hashes": True,
        "compiler_failure_policy": "fail-entire-one-shot-attempt-without-retry-v1",
        "paired_order_rule": "balanced-alternating-ab-ba-with-presealed-start-arm-v1",
    }


def default_matmul_collective_surface_design_contract() -> MatmulCollectiveSurfaceDesignContract:
    return MatmulCollectiveSurfaceDesignContract.model_validate_json(
        json.dumps(default_matmul_collective_surface_design_contract_payload())
    )


def _metric(report: object, name: str) -> Decimal:
    matches = tuple(value for value in report.metrics if value.name == name)  # type: ignore[attr-defined]
    if len(matches) != 1:
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_COST_METRIC_MISMATCH name={name}")
    return matches[0].quantity.value


def _physical_feature_row(
    arm: MatmulCollectiveSurfaceArmPlan,
    divisor: float,
) -> tuple[float, float, float]:
    return (
        float(arm.compute_time_floor_ns) / divisor,
        float(arm.hbm_time_floor_ns) / divisor,
        float(arm.ici_time_floor_ns) / divisor,
    )


def _joint_feature_row(
    contract: MatmulCollectiveSurfaceDesignContract,
    arm: MatmulCollectiveSurfaceArmPlan,
) -> tuple[float, float, float, float, float, float]:
    compute, hbm, ici = _physical_feature_row(arm, contract.feature_scale_divisor_ns)
    is_first = arm.strategy is contract.strategies[0]
    return (
        float(is_first),
        float(not is_first),
        compute,
        hbm,
        ici if is_first else 0.0,
        0.0 if is_first else ici,
    )


def _matrix_diagnostics(
    contract: MatmulCollectiveSurfaceDesignContract,
    arms: tuple[MatmulCollectiveSurfaceArmPlan, ...],
) -> tuple[int, float, float]:
    matrix = np.asarray([_joint_feature_row(contract, value) for value in arms], dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix))
    scales = np.maximum(np.linalg.norm(matrix, axis=0), 1.0)
    condition = float(np.linalg.cond(matrix / scales))
    physical = np.asarray(
        [_physical_feature_row(value, contract.feature_scale_divisor_ns) for value in arms],
        dtype=np.float64,
    )
    compute_std = float(np.std(physical[:, 0]))
    hbm_std = float(np.std(physical[:, 1]))
    compute_hbm_correlation = (
        1.0
        if compute_std == 0.0 or hbm_std == 0.0
        else float(np.corrcoef(physical[:, 0], physical[:, 1])[0, 1])
    )
    return rank, condition, compute_hbm_correlation


def _holdouts_inside_calibration_hull(
    contract: MatmulCollectiveSurfaceDesignContract,
) -> bool:
    calibration = contract.calibration_scenarios
    for holdout in contract.holdout_scenarios:
        plane = tuple(value for value in calibration if value.n == holdout.n)
        if not plane:
            return False
        min_m = min(value.m for value in plane)
        max_m = max(value.m for value in plane)
        min_k = min(value.k for value in plane)
        max_k = max(value.k for value in plane)
        corners = {
            (value.m, value.k)
            for value in plane
            if value.m in {min_m, max_m} and value.k in {min_k, max_k}
        }
        if len(corners) != 4 or not min_m < holdout.m < max_m or not min_k < holdout.k < max_k:
            return False
    return True


def _inside_convex_hull(point: np.ndarray, points: np.ndarray) -> bool:
    scales = np.maximum(np.max(np.abs(points), axis=0), 1.0)
    normalized_points = points / scales
    target = point / scales
    expected = np.concatenate((target, [1.0]))
    for count in range(1, min(points.shape[1] + 1, len(points)) + 1):
        for indices in itertools.combinations(range(len(points)), count):
            selected = normalized_points[list(indices)].T
            system = np.vstack((selected, np.ones(count)))
            weights, *_ = np.linalg.lstsq(system, expected, rcond=None)
            if np.all(weights >= -1e-10) and np.allclose(
                system @ weights,
                expected,
                rtol=1e-9,
                atol=1e-9,
            ):
                return True
    return False


def _holdouts_inside_physical_feature_hull(
    contract: MatmulCollectiveSurfaceDesignContract,
    arms: tuple[MatmulCollectiveSurfaceArmPlan, ...],
) -> bool:
    for strategy in contract.strategies:
        calibration = tuple(
            value
            for value in arms
            if value.strategy is strategy
            and value.split is MatmulCollectiveSurfaceSplit.CALIBRATION
        )
        holdouts = tuple(
            value
            for value in arms
            if value.strategy is strategy and value.split is MatmulCollectiveSurfaceSplit.HOLDOUT
        )
        points = np.asarray(
            [
                _physical_feature_row(value, contract.feature_scale_divisor_ns)
                for value in calibration
            ],
            dtype=np.float64,
        )
        if any(
            not _inside_convex_hull(
                np.asarray(
                    _physical_feature_row(value, contract.feature_scale_divisor_ns),
                    dtype=np.float64,
                ),
                points,
            )
            for value in holdouts
        ):
            return False
    return True


def derive_matmul_collective_surface_design_report(
    contract: MatmulCollectiveSurfaceDesignContract,
) -> MatmulCollectiveSurfaceDesignReport:
    if contract != default_matmul_collective_surface_design_contract():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_EXTERNAL_CONTRACT_MISMATCH")
    hardware = tpu7x_tensorcore_rates()
    source = MetricSource(
        artifact_sha256=contract.design_id,
        artifact_path="contracts/matmul-collective-surface-design-v1.json",
        tool="tpu-cake",
        field=MATMUL_COLLECTIVE_SURFACE_DESIGN_SCHEMA,
    )
    arms = []
    for scenario in contract.scenarios:
        distributed = distributed_matmul_schedule(
            mesh_size=contract.mesh_size,
            m=scenario.m,
            k=scenario.k,
            n=scenario.n,
        )
        distributed_hash = schedule_sha256(distributed)
        for strategy in contract.strategies:
            physical = lower_distributed_matmul(
                distributed,
                tile=MatmulTile(scenario.tile_m, scenario.tile_n),
                collective_implementation=strategy.lowering_implementation(),
            )
            plan = lower_physical_matmul_to_pallas(physical)
            cost = estimate_distributed_matmul(plan, hardware=hardware, source=source)
            arms.append(
                MatmulCollectiveSurfaceArmPlan(
                    scenario_name=scenario.name,
                    split=scenario.split,
                    strategy=strategy,
                    distributed_schedule_sha256=distributed_hash,
                    physical_schedule_sha256=plan.schedule_sha256,
                    pallas_source_sha256=plan.source_sha256(),
                    output_bytes=scenario.m * scenario.n * 4,
                    output_local_shape=plan.output_local_shape,
                    operations_per_device=cost.counts.operations_per_device,
                    hbm_bytes_per_device=(
                        cost.counts.hbm_read_bytes_per_device
                        + cost.counts.hbm_write_bytes_per_device
                    ),
                    ici_bytes_per_device=cost.counts.ici_bidirectional_bytes_per_device,
                    peak_live_vmem_bytes_per_device=(cost.counts.peak_live_vmem_bytes_per_device),
                    collective_hbm_scratch_bytes_per_device=(
                        cost.counts.collective_hbm_scratch_bytes_per_device
                    ),
                    collective_vmem_scratch_bytes_per_device=(
                        cost.counts.collective_vmem_scratch_bytes_per_device
                    ),
                    collective_dma_semaphore_count=(cost.counts.collective_dma_semaphore_count),
                    collective_capacity_semaphore_count=(
                        cost.counts.collective_capacity_semaphore_count
                    ),
                    collective_startup_semaphore_count=(
                        cost.counts.collective_startup_semaphore_count
                    ),
                    arithmetic_intensity=_metric(cost, "arithmetic_intensity"),
                    compute_time_floor_ns=_metric(cost, "compute_time_floor"),
                    hbm_time_floor_ns=_metric(cost, "hbm_time_floor"),
                    ici_time_floor_ns=_metric(cost, "ici_time_floor"),
                    idealized_time_floor_ns=_metric(cost, "idealized_time_floor"),
                    serial_resource_scenario_ns=_metric(cost, "serial_resource_time"),
                    predicted_limiting_resource=cost.predicted_limiting_resource,
                )
            )
    arm_tuple = tuple(arms)
    calibration = tuple(
        value for value in arm_tuple if value.split is MatmulCollectiveSurfaceSplit.CALIBRATION
    )
    rank, condition, correlation = _matrix_diagnostics(contract, calibration)
    if rank != 6 or condition >= contract.maximum_condition_number:
        raise ValueError(
            "MATMUL_COLLECTIVE_SURFACE_CALIBRATION_FEATURE_MATRIX_INVALID "
            f"rank={rank} condition={condition}"
        )
    if abs(correlation) >= contract.maximum_compute_hbm_correlation:
        raise ValueError(
            f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_FEATURE_CORRELATION compute_hbm={correlation}"
        )
    for strategy in contract.strategies:
        strategy_arms = tuple(value for value in calibration if value.strategy is strategy)
        for resource in ("compute", "hbm", "ici"):
            count = sum(value.predicted_limiting_resource == resource for value in strategy_arms)
            if count < contract.minimum_calibration_arms_per_limiting_resource:
                raise ValueError(
                    "MATMUL_COLLECTIVE_SURFACE_LIMITING_RESOURCE_COVERAGE "
                    f"strategy={strategy.value} resource={resource} count={count}"
                )
    inside_hull = _holdouts_inside_calibration_hull(contract)
    if not inside_hull:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_HOLDOUT_EXTRAPOLATION")
    inside_physical_hull = _holdouts_inside_physical_feature_hull(contract, arm_tuple)
    if not inside_physical_hull:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_PHYSICAL_FEATURE_EXTRAPOLATION")
    return MatmulCollectiveSurfaceDesignReport(
        design_id=contract.design_id,
        arms=arm_tuple,
        calibration_feature_rank=rank,
        calibration_condition_number=condition,
        calibration_compute_hbm_correlation=correlation,
        holdouts_inside_calibration_hull=inside_hull,
        holdouts_inside_physical_feature_hull=inside_physical_hull,
        physical_authority="canonical-xdsl-to-pallas-cost-replay",
    )


def _nonnegative_affine_fit(matrix: np.ndarray, observations: np.ndarray) -> np.ndarray:
    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    for count in range(1, matrix.shape[1] + 1):
        for active in itertools.combinations(range(matrix.shape[1]), count):
            candidate = np.zeros(matrix.shape[1], dtype=np.float64)
            solved, *_ = np.linalg.lstsq(matrix[:, active], observations, rcond=None)
            if np.any(solved < -1e-9):
                continue
            candidate[list(active)] = np.maximum(solved, 0.0)
            residual = observations - matrix @ candidate
            score = float(residual @ residual)
            record = (score, active, candidate)
            if best is None or (score, active) < (best[0], best[1]):
                best = record
    if best is None:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_NONNEGATIVE_FIT_FAILED")
    return best[2]


def fit_surface_model(
    contract: MatmulCollectiveSurfaceDesignContract,
    observations: tuple[SurfaceCalibrationObservation, ...],
) -> MatmulCollectiveSurfaceModel:
    report = derive_matmul_collective_surface_design_report(contract)
    expected = tuple((arm.scenario_name, arm.strategy) for arm in report.calibration_arms)
    observed = tuple((value.scenario_name, value.strategy) for value in observations)
    if observed != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_INVENTORY_MISMATCH")
    matrix = np.asarray(
        [_joint_feature_row(contract, value) for value in report.calibration_arms],
        dtype=np.float64,
    )
    rank, condition, _ = _matrix_diagnostics(contract, report.calibration_arms)
    if rank != 6:
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_FEATURE_RANK rank={rank}")
    if condition >= contract.maximum_condition_number:
        raise ValueError(
            f"MATMUL_COLLECTIVE_SURFACE_CALIBRATION_FEATURE_CONDITION condition={condition}"
        )
    measured = np.asarray([value.median_ns for value in observations], dtype=np.float64)
    coefficients = _nonnegative_affine_fit(matrix, measured)
    predicted = matrix @ coefficients
    relative = np.abs(predicted - measured) / measured
    return MatmulCollectiveSurfaceModel(
        coefficient_names=(
            "xla_intercept",
            "pallas_intercept",
            "shared_compute",
            "shared_hbm",
            "xla_ici",
            "pallas_ici",
        ),
        coefficients=tuple(float(value) for value in coefficients),
        calibration_predictions_ns=tuple(float(value) for value in predicted),
        calibration_relative_errors=tuple(float(value) for value in relative),
        maximum_calibration_relative_error=float(np.max(relative)),
        median_calibration_relative_error=float(np.median(relative)),
    )


def validate_matmul_collective_surface_design_report(
    report: MatmulCollectiveSurfaceDesignReport,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> None:
    expected = derive_matmul_collective_surface_design_report(contract)
    if report != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_DESIGN_REPORT_REPLAY_MISMATCH")


def write_matmul_collective_surface_design_report(
    output: Path,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> MatmulCollectiveSurfaceDesignReport:
    report = derive_matmul_collective_surface_design_report(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x") as stream:
            json.dump(report.model_dump(mode="json"), stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_DESIGN_OUTPUT_EXISTS") from None
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive")
    derive.add_argument("--contract", required=True, type=Path)
    derive.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--contract", required=True, type=Path)
    verify.add_argument("--report", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    contract = MatmulCollectiveSurfaceDesignContract.model_validate_json(args.contract.read_text())
    if args.command == "derive":
        report = write_matmul_collective_surface_design_report(args.output, contract)
        action = "DERIVED"
    else:
        report = MatmulCollectiveSurfaceDesignReport.model_validate_json(args.report.read_text())
        validate_matmul_collective_surface_design_report(report, contract)
        action = "REPLAYED"
    print(
        f"MATMUL_COLLECTIVE_SURFACE_DESIGN_{action} "
        f"design_id={report.design_id} arms={len(report.arms)}"
    )


if __name__ == "__main__":
    main()
