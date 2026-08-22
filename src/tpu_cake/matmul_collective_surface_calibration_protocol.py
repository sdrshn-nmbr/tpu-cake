from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import model_identity_sha256, semantic_seed, semantic_sha256
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    default_matmul_collective_surface_design_contract,
)
from tpu_cake.runner import MatmulCollectiveStrategy

MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PROTOCOL_SCHEMA = (
    "matmul-collective-surface-calibration-protocol-v1"
)
MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PROTOCOL_PATH = Path(
    "contracts/matmul-collective-surface-calibration-v1.json"
)


class SurfaceCalibrationCorrectnessParent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    archive_path: str
    archive_filename: str
    archive_root_name: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_size_bytes: int = Field(gt=0)
    archive_container_schema: Literal[
        "tar-zstd-single-root-no-links-no-devices-no-duplicates-v1"
    ] = "tar-zstd-single-root-no-links-no-devices-no-duplicates-v1"
    archive_maximum_members: Literal[2000] = 2000
    archive_maximum_member_size_bytes: Literal[1073741824] = 1_073_741_824
    archive_maximum_total_size_bytes: Literal[4294967296] = 4_294_967_296
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_contract_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_design_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archived_protocol_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archived_design_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_ledger_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_identity_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_claim_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_authority_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["calibration"] = "calibration"
    case_count: Literal[80] = 80
    execution_count: Literal[320] = 320
    independent_replay_required: Literal[True] = True


class MatmulCollectiveSurfaceCalibrationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-protocol-v1"] = (
        MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PROTOCOL_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent: SurfaceCalibrationCorrectnessParent
    parent_archive_staging_rule: Literal[
        "copy-exclusive-private-fsync-hash-before-inspection-v1"
    ] = "copy-exclusive-private-fsync-hash-before-inspection-v1"
    parent_archive_extraction_rule: Literal[
        "preflight-all-members-before-private-extraction-v1"
    ] = "preflight-all-members-before-private-extraction-v1"
    scenarios: tuple[str, ...] = Field(min_length=16, max_length=16)
    strategies: tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy]
    split: Literal["calibration"] = "calibration"
    timing_input_pattern: Literal["signed-periodic"] = "signed-periodic"
    timing_input_rule: Literal["exact-parent-correctness-input-shards-v1"] = (
        "exact-parent-correctness-input-shards-v1"
    )
    timing_oracle_rule: Literal["independently-regenerated-full-fp32-oracle-v1"] = (
        "independently-regenerated-full-fp32-oracle-v1"
    )
    untimed_output_gate: Literal["full-output-before-and-after-timing-v1"] = (
        "full-output-before-and-after-timing-v1"
    )
    output_parent_binding_rule: Literal["exact-parent-signed-periodic-strategy-array-sha256-v1"] = (
        "exact-parent-signed-periodic-strategy-array-sha256-v1"
    )
    absolute_tolerance: Literal[0.001] = 0.001
    relative_tolerance: Literal[0.001] = 0.001
    mismatch_rule: Literal["abs(candidate-oracle)>atol+rtol*abs(oracle)-v1"] = (
        "abs(candidate-oracle)>atol+rtol*abs(oracle)-v1"
    )
    warmup_iterations_per_strategy: Literal[10] = 10
    first_warmup_strategy: Literal[MatmulCollectiveStrategy.XLA_REDUCE_SCATTER] = (
        MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
    )
    warmup_order_rule: Literal[
        "all-resident-scenario-forward-explicit-first-balanced-alternating-v1"
    ] = "all-resident-scenario-forward-explicit-first-balanced-alternating-v1"
    calls_per_position: Literal[5] = 5
    paired_rounds: Literal[16] = 16
    scenario_order_rule: Literal["rounds-0-7-forward-rounds-8-15-reverse-v1"] = (
        "rounds-0-7-forward-rounds-8-15-reverse-v1"
    )
    first_timed_strategy: Literal[MatmulCollectiveStrategy.XLA_REDUCE_SCATTER] = (
        MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
    )
    paired_order_rule: Literal["explicit-first-balanced-alternating-ab-ba-v1"] = (
        "explicit-first-balanced-alternating-ab-ba-v1"
    )
    clock: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    synchronization_rule: Literal["each-call-output-block-until-ready-v1"] = (
        "each-call-output-block-until-ready-v1"
    )
    sample_retention_rule: Literal["retain-all-2560-positive-call-durations-v1"] = (
        "retain-all-2560-positive-call-durations-v1"
    )
    arm_estimator: Literal["median-of-16-paired-round-medians-v1"] = (
        "median-of-16-paired-round-medians-v1"
    )
    fit_rule: Literal["joint-nonnegative-affine-shared-compute-hbm-strategy-ici-v1"] = (
        "joint-nonnegative-affine-shared-compute-hbm-strategy-ici-v1"
    )
    coefficient_bootstrap_rule: Literal[
        "global-paired-round-index-resample-with-replacement-v1"
    ] = "global-paired-round-index-resample-with-replacement-v1"
    coefficient_bootstrap_index_rule: Literal[
        "semantic-seed-protocol-id-calibration-bootstrap-replicate-draw-round-index-v1-mod-16"
    ] = "semantic-seed-protocol-id-calibration-bootstrap-replicate-draw-round-index-v1-mod-16"
    coefficient_bootstrap_fit_rule: Literal[
        "rerun-exact-joint-nonnegative-fit-for-each-resample-v1"
    ] = "rerun-exact-joint-nonnegative-fit-for-each-resample-v1"
    coefficient_bootstrap_samples: Literal[10000] = 10000
    coefficient_bootstrap_seed: Literal[17012026] = 17012026
    prediction_interval_rule: Literal["two-sided-99pct-percentile-v1"] = (
        "two-sided-99pct-percentile-v1"
    )
    prediction_percentile_rule: Literal["numpy-2.5.2-quantile-linear-v1"] = (
        "numpy-2.5.2-quantile-linear-v1"
    )
    prediction_interval_relative_width_rule: Literal[
        "upper-minus-lower-over-point-prediction-v1"
    ] = "upper-minus-lower-over-point-prediction-v1"
    maximum_holdout_prediction_ci_relative_width: Literal[0.2] = 0.2
    holdout_authorization_rule: Literal[
        "seal-always-authorize-only-if-all-8-arm-widths-at-most-0.20-v1"
    ] = "seal-always-authorize-only-if-all-8-arm-widths-at-most-0.20-v1"
    strategy_improvement_rule: Literal["xla-median-minus-pallas-median-over-xla-median-v1"] = (
        "xla-median-minus-pallas-median-over-xla-median-v1"
    )
    predictions_sealed_before_holdout: Literal[True] = True
    calibration_seal_schema: Literal["matmul-collective-surface-calibration-seal-v1"] = (
        "matmul-collective-surface-calibration-seal-v1"
    )
    compilation_cache_rule: Literal["isolated-empty-temporary-directory-v1"] = (
        "isolated-empty-temporary-directory-v1"
    )
    compilation_excluded_from_timing: Literal[True] = True
    compile_continuity_rule: Literal[
        "parent-schedule-pallas-semantic-stablehlo-semantic-compilerhlo-v1"
    ] = "parent-schedule-pallas-semantic-stablehlo-semantic-compilerhlo-v1"
    candidates_resident_together: Literal[True] = True
    residency_scope: Literal[
        "all-32-executables-and-16-operand-pairs-before-first-output-gate-through-last-output-gate-v1"
    ] = "all-32-executables-and-16-operand-pairs-before-first-output-gate-through-last-output-gate-v1"
    allow_profile_data: Literal[False] = False
    allow_holdout_materialization: Literal[False] = False
    allow_early_stopping: Literal[False] = False
    allow_retry: Literal[False] = False
    allow_outlier_removal: Literal[False] = False
    allow_calibration_refit_after_seal: Literal[False] = False
    one_shot_attempt_ledger: Literal[True] = True
    permanent_claim_key_rule: Literal[
        "semantic-sha256-correctness-receipt-calibration-timing-v1"
    ] = "semantic-sha256-correctness-receipt-calibration-timing-v1"
    permanent_claim_point: Literal[
        "after-parent-independent-replay-before-compilation-input-materialization-or-warmup-v1"
    ] = "after-parent-independent-replay-before-compilation-input-materialization-or-warmup-v1"
    attempt_registry_root: str

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> MatmulCollectiveSurfaceCalibrationProtocol:
        expected = default_matmul_collective_surface_calibration_protocol_payload()
        if self.model_dump(mode="json", exclude_computed_fields=True) != expected:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PROTOCOL_MISMATCH")
        return self

    @computed_field
    @property
    def protocol_id(self) -> str:
        return model_identity_sha256(self)

    @computed_field
    @property
    def permanent_claim_key(self) -> str:
        return semantic_sha256(
            self.correctness_parent.receipt_sha256,
            "calibration-timing-v1",
        )

    def scenario_order(self, round_index: int) -> tuple[str, ...]:
        if not 0 <= round_index < self.paired_rounds:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROUND_INVALID")
        return self.scenarios if round_index < 8 else tuple(reversed(self.scenarios))

    def strategy_order(
        self,
        round_index: int,
    ) -> tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy]:
        if not 0 <= round_index < self.paired_rounds:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ROUND_INVALID")
        first_index = self.strategies.index(self.first_timed_strategy)
        starts_with_declared = round_index % 2 == 0
        order = self.strategies[first_index:] + self.strategies[:first_index]
        return order if starts_with_declared else tuple(reversed(order))

    def warmup_strategy_order(
        self,
        scenario_index: int,
    ) -> tuple[MatmulCollectiveStrategy, ...]:
        if not 0 <= scenario_index < len(self.scenarios):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SCENARIO_INVALID")
        first_index = self.strategies.index(self.first_warmup_strategy)
        pair = self.strategies[first_index:] + self.strategies[:first_index]
        if scenario_index % 2:
            pair = tuple(reversed(pair))
        return pair * self.warmup_iterations_per_strategy

    def bootstrap_round_indices(self, replicate_index: int) -> tuple[int, ...]:
        if not 0 <= replicate_index < self.coefficient_bootstrap_samples:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_BOOTSTRAP_REPLICATE_INVALID")
        return tuple(
            semantic_seed(
                self.protocol_id,
                "calibration-bootstrap",
                str(replicate_index),
                str(draw_index),
                "round-index-v1",
            )
            % self.paired_rounds
            for draw_index in range(self.paired_rounds)
        )


def default_matmul_collective_surface_calibration_protocol_payload() -> dict[str, object]:
    design = default_matmul_collective_surface_design_contract()
    return {
        "schema_version": MATMUL_COLLECTIVE_SURFACE_CALIBRATION_PROTOCOL_SCHEMA,
        "design_id": design.design_id,
        "correctness_parent": {
            "archive_path": (
                "/home/sudarshan/tpu-cake-evidence/"
                "matmul-collective-surface-correctness-e72af49-02c589e.tar.zst"
            ),
            "archive_filename": ("matmul-collective-surface-correctness-e72af49-02c589e.tar.zst"),
            "archive_root_name": "matmul-collective-surface-correctness-e72af49-02c589e",
            "archive_sha256": "c2f0c459cf17f592ccfc545ac49491c6357bffc56f4dbc085192a0ca0f1a6c6b",
            "archive_size_bytes": 4_356_899,
            "archive_container_schema": (
                "tar-zstd-single-root-no-links-no-devices-no-duplicates-v1"
            ),
            "archive_maximum_members": 2_000,
            "archive_maximum_member_size_bytes": 1_073_741_824,
            "archive_maximum_total_size_bytes": 4_294_967_296,
            "attempt_id": "02c589e35e63a1b5747d5d8bd45b16f9c74fffe7c92c99bcf0c6a36947b25e54",
            "protocol_id": "f5f81b36f9542334163bb86d893dc5fd4c1856c7ccc7d8023e29509a4876ccea",
            "source_commit": "e72af499e517c8c7e74ba0e84b6540479cf53ecf",
            "source_contract_file_sha256": (
                "29b7a4a8d1a2215c69dfd5976d714e0090d1c2dbfe642d4605fd2882ada69027"
            ),
            "source_design_file_sha256": (
                "9f4332de116319b0a5ad33703314ff2a7c293a8e9bda70c2c2596c424156f17f"
            ),
            "archived_protocol_file_sha256": (
                "b67027c1e8e3ba6a970464ad6b9598b24a3dc9897491b582b0034a814de003a2"
            ),
            "archived_design_file_sha256": (
                "dc0236f74b6d7322912290884fd1f00c7098c1f25c9bcecc367143183c2f34ff"
            ),
            "manifest_file_sha256": (
                "7e9426c55406cb9bb7ac626059a77fb077224ed01b5810f682153a5407e70e1a"
            ),
            "evidence_file_sha256": (
                "949e828c5519721c51fd9e4afdbb3dbbb9eba7e7e7b93fa87592885523f42fc9"
            ),
            "evidence_sha256": "92aaed8bd7059255102993020e0158c6fd59daaae703f2f7df49a8de95b8b6c3",
            "receipt_file_sha256": (
                "417eebdc9622b2f01917b621ff0d009727fd37dad70b5fb497661fdee655d32f"
            ),
            "receipt_sha256": "7a28ba5f08a349f37dcbe50924a25fa76eac4d77e40d468a4dd71ea201bf7db2",
            "artifact_set_sha256": (
                "30216ffd9dd599ea472714e5421e583b23e59871593c04820068f11f4ff55532"
            ),
            "phase_ledger_file_sha256": (
                "78d6a213896eb85c8370bdeed5d2733344ffe74c5d100930d406cf5c02e5341b"
            ),
            "phase_ledger_sha256": (
                "51e770fdca6712870c38a969d966098a7cf242bba2441a541cbb372377d12ac6"
            ),
            "ledger_file_sha256": (
                "6f6bdaf182d5135a0dd84e0dc03320d31cb373b9ced2e7bbe93fc2352b99cfd6"
            ),
            "run_identity_file_sha256": (
                "31c1346f52223191bacbb52c7065ae374e688638b1c468cfbd795f446fad2bc8"
            ),
            "attempt_claim_file_sha256": (
                "e688a865d313fdf68e88f33beba8fc12dc84b9b1fc07562ccfa7d7712e63373f"
            ),
            "execution_authority_file_sha256": (
                "9aa9aadf166906690018ca9fc305f4f25bc687beb8d5bd6aa08b8acb232233ba"
            ),
            "execution_authority_sha256": (
                "513af7b35b7cccba4c2b2a208003c9fe0a029db11eb408898c5e4f6ab62fdb88"
            ),
            "source_authority_sha256": (
                "91d5f3f59c02fb39863dc7465367bc75ec653a0c316339daee048b1a16aa9b9a"
            ),
            "verifier_source_sha256": (
                "f0115b1fc7561ad40dfc6e913b8fe86b91ecc8f6bb2788b233bd1609f99c19db"
            ),
            "split": "calibration",
            "case_count": 80,
            "execution_count": 320,
            "independent_replay_required": True,
        },
        "parent_archive_staging_rule": ("copy-exclusive-private-fsync-hash-before-inspection-v1"),
        "parent_archive_extraction_rule": ("preflight-all-members-before-private-extraction-v1"),
        "scenarios": [value.name for value in design.calibration_scenarios],
        "strategies": [value.value for value in design.strategies],
        "split": "calibration",
        "timing_input_pattern": "signed-periodic",
        "timing_input_rule": "exact-parent-correctness-input-shards-v1",
        "timing_oracle_rule": "independently-regenerated-full-fp32-oracle-v1",
        "untimed_output_gate": "full-output-before-and-after-timing-v1",
        "output_parent_binding_rule": ("exact-parent-signed-periodic-strategy-array-sha256-v1"),
        "absolute_tolerance": 0.001,
        "relative_tolerance": 0.001,
        "mismatch_rule": "abs(candidate-oracle)>atol+rtol*abs(oracle)-v1",
        "warmup_iterations_per_strategy": 10,
        "first_warmup_strategy": MatmulCollectiveStrategy.XLA_REDUCE_SCATTER.value,
        "warmup_order_rule": (
            "all-resident-scenario-forward-explicit-first-balanced-alternating-v1"
        ),
        "calls_per_position": 5,
        "paired_rounds": 16,
        "scenario_order_rule": "rounds-0-7-forward-rounds-8-15-reverse-v1",
        "first_timed_strategy": MatmulCollectiveStrategy.XLA_REDUCE_SCATTER.value,
        "paired_order_rule": "explicit-first-balanced-alternating-ab-ba-v1",
        "clock": "time.perf_counter_ns",
        "synchronization_rule": "each-call-output-block-until-ready-v1",
        "sample_retention_rule": "retain-all-2560-positive-call-durations-v1",
        "arm_estimator": "median-of-16-paired-round-medians-v1",
        "fit_rule": "joint-nonnegative-affine-shared-compute-hbm-strategy-ici-v1",
        "coefficient_bootstrap_rule": "global-paired-round-index-resample-with-replacement-v1",
        "coefficient_bootstrap_index_rule": (
            "semantic-seed-protocol-id-calibration-bootstrap-replicate-draw-round-index-v1-mod-16"
        ),
        "coefficient_bootstrap_fit_rule": (
            "rerun-exact-joint-nonnegative-fit-for-each-resample-v1"
        ),
        "coefficient_bootstrap_samples": 10_000,
        "coefficient_bootstrap_seed": 17_012_026,
        "prediction_interval_rule": "two-sided-99pct-percentile-v1",
        "prediction_percentile_rule": "numpy-2.5.2-quantile-linear-v1",
        "prediction_interval_relative_width_rule": ("upper-minus-lower-over-point-prediction-v1"),
        "maximum_holdout_prediction_ci_relative_width": 0.2,
        "holdout_authorization_rule": (
            "seal-always-authorize-only-if-all-8-arm-widths-at-most-0.20-v1"
        ),
        "strategy_improvement_rule": ("xla-median-minus-pallas-median-over-xla-median-v1"),
        "predictions_sealed_before_holdout": True,
        "calibration_seal_schema": "matmul-collective-surface-calibration-seal-v1",
        "compilation_cache_rule": "isolated-empty-temporary-directory-v1",
        "compilation_excluded_from_timing": True,
        "compile_continuity_rule": (
            "parent-schedule-pallas-semantic-stablehlo-semantic-compilerhlo-v1"
        ),
        "candidates_resident_together": True,
        "residency_scope": (
            "all-32-executables-and-16-operand-pairs-before-first-output-gate-"
            "through-last-output-gate-v1"
        ),
        "allow_profile_data": False,
        "allow_holdout_materialization": False,
        "allow_early_stopping": False,
        "allow_retry": False,
        "allow_outlier_removal": False,
        "allow_calibration_refit_after_seal": False,
        "one_shot_attempt_ledger": True,
        "permanent_claim_key_rule": ("semantic-sha256-correctness-receipt-calibration-timing-v1"),
        "permanent_claim_point": (
            "after-parent-independent-replay-before-compilation-input-materialization-or-warmup-v1"
        ),
        "attempt_registry_root": (
            "/home/sudarshan/tpu-cake-evidence/matmul-collective-surface-calibration-attempts-v1"
        ),
    }


def default_matmul_collective_surface_calibration_protocol() -> (
    MatmulCollectiveSurfaceCalibrationProtocol
):
    return MatmulCollectiveSurfaceCalibrationProtocol.model_validate_json(
        json.dumps(default_matmul_collective_surface_calibration_protocol_payload())
    )


def load_matmul_collective_surface_calibration_protocol(
    path: Path,
    design: MatmulCollectiveSurfaceDesignContract,
) -> MatmulCollectiveSurfaceCalibrationProtocol:
    protocol = MatmulCollectiveSurfaceCalibrationProtocol.model_validate_json(path.read_text())
    if design != default_matmul_collective_surface_design_contract():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_DESIGN_MISMATCH")
    if protocol.design_id != design.design_id:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_DESIGN_ID_MISMATCH")
    return protocol
