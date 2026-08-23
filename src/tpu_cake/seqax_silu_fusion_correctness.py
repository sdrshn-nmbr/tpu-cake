from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, json_sha256, model_identity_sha256
from tpu_cake.ledger import RunState
from tpu_cake.seqax_contract_types import (
    SeqaxFeedForwardFusion,
    SeqaxFeedForwardVectorExecution,
    SeqaxResidualNormStrategy,
)
from tpu_cake.seqax_silu_fusion import (
    SEQAX_SILU_FUSION_BOUNDARY_SEED,
    SEQAX_SILU_FUSION_COMPILATION_ROOT,
    SEQAX_SILU_FUSION_CORRECTNESS_SEEDS,
)

SEQAX_SILU_FUSION_CORRECTNESS_SCHEMA = "seqax-silu-fusion-correctness-v1"
SEQAX_SILU_FUSION_CORRECTNESS_CONTRACT_SHA256 = (
    "ffd334ec7ed3449266a5e4229eae1439ba6eb12afedf395819ca1e3ef542ffdd"
)
SEQAX_SILU_FUSION_CORRECTNESS_CLAIM_SCHEMA = "seqax-silu-fusion-correctness-claim-v1"
SEQAX_SILU_FUSION_CORRECTNESS_RESULT_SCHEMA = "seqax-silu-fusion-correctness-result-v1"
SEQAX_SILU_FUSION_CORRECTNESS_WORKER_SCHEMA = "seqax-silu-fusion-correctness-worker-v1"
SEQAX_SILU_FUSION_CORRECTNESS_RECEIPT_SCHEMA = "seqax-silu-fusion-correctness-receipt-v1"
SEQAX_SILU_FUSION_CORRECTNESS_REPLAY_SCHEMA = "seqax-silu-fusion-correctness-replay-v1"
SEQAX_SILU_FUSION_CORRECTNESS_ARCHIVE_SCHEMA = "seqax-silu-fusion-correctness-archive-v1"
SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_SCHEMA = "seqax-silu-fusion-correctness-failure-v1"
SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_RECEIPT_SCHEMA = (
    "seqax-silu-fusion-correctness-failure-receipt-v1"
)
SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_REPLAY_SCHEMA = (
    "seqax-silu-fusion-correctness-failure-replay-v1"
)
SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARCHIVE_SCHEMA = (
    "seqax-silu-fusion-correctness-failure-archive-v1"
)
SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_ROLES = frozenset(
    {
        ArtifactRole.EXPERIMENT,
        ArtifactRole.DISTRIBUTED_IR,
        ArtifactRole.PHYSICAL_IR,
        ArtifactRole.PALLAS_SOURCE,
        ArtifactRole.PLAN_MANIFEST,
        ArtifactRole.STABLEHLO,
        ArtifactRole.COMPILER_HLO,
        ArtifactRole.COMPILER_ANALYSIS,
        ArtifactRole.CORRECTNESS_INPUT,
        ArtifactRole.CORRECTNESS_OUTPUT,
        ArtifactRole.ORACLE_OUTPUT,
        ArtifactRole.EXECUTION_LEDGER,
        ArtifactRole.INVOCATION,
        ArtifactRole.SOURCE_STATE,
    }
)
SEQAX_SILU_FUSION_COMPILER_DESIGN_ID = (
    "ec6ca7194ce8430035e94210d9fb2a75c99bf5749c50ab579dedf4fd1914996b"
)
SEQAX_SILU_FUSION_COMPILER_DESIGN_SHA256 = (
    "e6c7fd18da0edba925b2bc6e0725c25b51f233f06013a9147c5eca0c977d0bff"
)
SEQAX_SILU_FUSION_COMPILER_PAIR_ID = (
    "afa1774255d9295ac30922b9bf851f85585fcb4c8267606085a3d851a2d89f9c"
)
SEQAX_SILU_FUSION_COMPILER_PAIR_SHA256 = (
    "06ed3cc20bc6642a91f8dbddf7ae9ed56c705f73a57efe15b60d135c25626807"
)
SEQAX_SILU_FUSION_COMPILER_CAPTURE_IDS = (
    "a2f76b116da7b2f618b0f48dc4648677720c07669e96dcbae530ab09b2c0b5cb",
    "2753d19222bfa3d2fb448a847c16de5c695170a6b973e74ed721ee385bd197b8",
)
SEQAX_SILU_FUSION_CANDIDATE_SEMANTIC_IDS = (
    "77f35a299a8da1296a95c9caaa29e1b0ef3e538e5a125780a25097aa6f0cde04",
    "9ba500002f0ae57e6e55aec1ec1b491e975f767489c89f2d7c50265bb999aeb0",
)
SEQAX_SILU_FUSION_CHECKPOINTS = (
    "rms_input",
    "rms_mean_square",
    "rms_inverse",
    "normalized_float32",
    "normalized_bfloat16",
    "gate_float32",
    "gate_bfloat16",
    "silu_bfloat16",
    "up_float32",
    "up_bfloat16",
    "hidden_bfloat16",
    "down_float32",
    "down_bfloat16",
)
SEQAX_SILU_FUSION_SEPARATE_CHECKPOINT_CAPTURE_MODES = ("executed-model-intermediate",) * 11 + (
    "instrumented-projection-only-all-reduce",
) * 2
SEQAX_SILU_FUSION_FUSED_CHECKPOINT_CAPTURE_MODES = (
    ("executed-model-intermediate",) * 7
    + ("reconstructed-strict-bf16-silu-from-gate",)
    + ("executed-model-intermediate",) * 3
    + ("instrumented-projection-only-all-reduce",) * 2
)


class SeqaxSiluFusionCorrectnessPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    input_generator: Literal["seqax-forward-inputs-v1"]
    cpu_oracle: Literal["seqax-forward-canonical-reference-v1"]
    cpu_reference: Literal["jax_cpu_reference_v1"]
    numerical_policy_schema: Literal["bf16-forward-numerical-v6"]
    numerical_semantics: Literal["typed_bf16_hidden_v2"]
    cpu_reference_quantization_decimals: Literal[6]
    unit_roundoff: Literal[0.00390625]
    cpu_relative_l2_units: Literal[3.0]
    cpu_row_scaled_max_units: Literal[8.0]
    cross_path_relative_l2_units: Literal[2.0]
    cross_path_row_scaled_max_units: Literal[2.0]
    depth_scaling: Literal["sqrt_layers"]
    row_scale_floor: Literal[1.0]
    metric_quantization_decimals: Literal[15]
    cpu_replay_rule: Literal["cpu_facing_numerical_bounds"]
    checkpoint_storage_dtype: Literal["uint16"]
    checkpoint_logical_dtype: Literal["bfloat16"]
    checkpoint_encoding: Literal["bf16-bit-pattern-v1"]
    mathematical_silu_max_ulp: Literal[1]
    rms_inverse_relative_error_units: Literal[4.0]
    require_float32_output: Literal[True]
    require_finite_output: Literal[True]
    require_each_candidate_uninstrumented_final_output_policy: Literal[True]
    require_each_candidate_instrumented_final_output_policy: Literal[True]
    require_each_candidate_uninstrumented_cpu_top1_match: Literal[True]
    require_each_candidate_instrumented_cpu_top1_match: Literal[True]
    require_each_candidate_checkpoint_values_consistent: Literal[True]
    require_exact_candidate_outputs: Literal[True]
    require_exact_candidate_checkpoints: Literal[True]
    require_exact_instrumentation_outputs: Literal[True]
    boundary_discriminator: Literal["omit-intermediate-bf16-silu-round"]
    boundary_mutant_must_differ: Literal[True]
    require_full_inputs_persisted: Literal[True]
    require_full_outputs_persisted: Literal[True]
    require_full_checkpoints_persisted: Literal[True]
    require_model_outputs_executed: Literal[True]
    timing_collected: Literal[False]
    profile_collected: Literal[False]


class SeqaxSiluFusionCorrectnessContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_SCHEMA
    )
    identity_schema: Literal["length-prefixed-v2"]
    claim_scope: Literal[
        "fixed-seqax-silu-fusion-full-output-and-strict-mlp-checkpoint-equivalence"
    ]
    compiler_evidence_status: Literal["pending-rebind", "verified"]
    compiler_design_path: Literal["contracts/seqax-silu-fusion-design-v1.json"]
    compiler_design_id: Literal[SEQAX_SILU_FUSION_COMPILER_DESIGN_ID]
    compiler_design_sha256: Literal[SEQAX_SILU_FUSION_COMPILER_DESIGN_SHA256]
    compiler_pair_record_path: Literal["contracts/seqax-silu-fusion-compiler-pair-v1.json"]
    compiler_pair_evidence_path: Literal[
        "/home/sudarshan/tpu-cake-evidence/seqax-silu-fusion-compiler-pair-ec6ca71-d4e91b70.json"
    ]
    compiler_pair_id: Literal[SEQAX_SILU_FUSION_COMPILER_PAIR_ID]
    compiler_pair_sha256: Literal[SEQAX_SILU_FUSION_COMPILER_PAIR_SHA256]
    compiler_capture_ids: tuple[str, str]
    candidate_semantic_ids: tuple[str, str]
    compilation_source_root: Literal[SEQAX_SILU_FUSION_COMPILATION_ROOT]
    source_remote_url: Literal["https://github.com/sdrshn-nmbr/tpu-cake.git"]
    source_branch: Literal["main"]
    correctness_claim_registry_root: Literal[
        "/home/sudarshan/tpu-cake-evidence/seqax-silu-fusion-correctness-claims"
    ]
    correctness_claim_key: Literal[SEQAX_SILU_FUSION_CORRECTNESS_SCHEMA]
    correctness_claim_identity_scope: Literal["contract-id"]
    correctness_claim_reservation: Literal["exclusive-create-only"]
    allow_retry: Literal[False]
    allow_resume: Literal[False]
    independent_replay_required: Literal[True]
    archive_required: Literal[True]
    timing_authorized: Literal[False]
    profile_authorized: Literal[False]
    uv_lock_sha256: Literal["03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9"]
    parameters: dict[str, int | str]
    residual_norm_strategy: Literal[SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE]
    vector_execution: Literal[SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL]
    candidates: tuple[
        Literal[SeqaxFeedForwardFusion.SEPARATE],
        Literal[SeqaxFeedForwardFusion.SILU_MULTIPLY],
    ]
    correctness_seeds: tuple[int, int, int, int, int]
    boundary_seed: int
    checkpoint_names: tuple[str, ...] = Field(min_length=13, max_length=13)
    checkpoint_capture_modes: dict[str, tuple[str, ...]]
    output_shape: tuple[Literal[256], Literal[1], Literal[16]]
    output_dtype: Literal["float32"]
    policy: SeqaxSiluFusionCorrectnessPolicy
    worker_environment: dict[str, str]
    compiler_environment: dict[str, str]
    project: Literal["astral-medley-465922-b2"]
    numeric_project_id: Literal["541760035156"]
    zone: Literal["us-central1-c"]
    hostname: Literal["tpu-cake-v7x-rsag-wx7r"]
    instance_hostname: Literal[
        "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
    ]
    machine_type: Literal["tpu7x-standard-4t"]
    instance_id: Literal["5064039476077763048"]
    cpu_platform: Literal["Intel Emerald Rapids"]
    runtime: RuntimeIdentity
    backend: Literal["tpu"]
    device_kind: Literal["TPU7x"]
    device_count: Literal[8]
    mesh: dict[str, int]

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxSiluFusionCorrectnessContract:
        expected = default_seqax_silu_fusion_correctness_contract(self.runtime)
        if self.model_dump(exclude_computed_fields=True) != expected.model_dump(
            exclude_computed_fields=True
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PROTOCOL_MISMATCH")
        return self

    @computed_field
    @property
    def contract_id(self) -> str:
        return model_identity_sha256(self)


class SeqaxSiluFusionCorrectnessSourceAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch: Literal["main"]
    origin_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_url: Literal["https://github.com/sdrshn-nmbr/tpu-cake.git"]
    source_root: Literal[SEQAX_SILU_FUSION_COMPILATION_ROOT]
    uv_lock_sha256: Literal["03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9"]
    cli_sha256: Literal["355040b20f7e48683811b009fc77f460652617fafcdc44c68a3d7309fd71f740"]
    correctness_contract_sha256: Literal[SEQAX_SILU_FUSION_CORRECTNESS_CONTRACT_SHA256]
    compiler_design_sha256: Literal[SEQAX_SILU_FUSION_COMPILER_DESIGN_SHA256]
    compiler_pair_sha256: Literal[SEQAX_SILU_FUSION_COMPILER_PAIR_SHA256]
    correctness_schema_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    runtime: RuntimeIdentity

    @model_validator(mode="after")
    def main_is_exact(self) -> SeqaxSiluFusionCorrectnessSourceAuthority:
        if not (self.source_commit == self.origin_main_commit == self.remote_main_commit):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_MAIN_MISMATCH")
        paths = tuple(value.path for value in self.source_manifest)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_MANIFEST_INVALID")
        return self

    @computed_field
    @property
    def source_authority_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessAttemptClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_CLAIM_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_CLAIM_SCHEMA
    )
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    output_root: str = Field(min_length=1)

    @computed_field
    @property
    def claim_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0, le=7)
    process_index: Literal[0]
    platform: Literal["tpu"]
    device_kind: Literal["TPU7x"]


class SeqaxSiluFusionCorrectnessHost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project: Literal["astral-medley-465922-b2"]
    numeric_project_id: Literal["541760035156"]
    zone: Literal["us-central1-c"]
    hostname: Literal["tpu-cake-v7x-rsag-wx7r"]
    instance_hostname: Literal[
        "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
    ]
    machine_type: Literal["tpu7x-standard-4t"]
    instance_id: Literal["5064039476077763048"]
    cpu_platform: Literal["Intel Emerald Rapids"]


class SeqaxSiluFusionCorrectnessPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxFeedForwardFusion
    candidate_semantic_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uninstrumented_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uninstrumented_pre_optimization_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uninstrumented_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_pre_optimization_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SeqaxSiluFusionFinalOutputMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cpu_relative_l2: float = Field(ge=0)
    cpu_row_scaled_max: float = Field(ge=0)
    cpu_top1_match: Literal[True]
    final_output_policy_passed: Literal[True]


class SeqaxSiluFusionCheckpointMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    rms_mean_square_max_bound_ratio: float = Field(ge=0)
    rms_inverse_relative_error_units: float = Field(ge=0)
    normalized_float32_max_bound_ratio: float = Field(ge=0)
    gate_float32_max_bound_ratio: float = Field(ge=0)
    silu_max_ulp_of_mathematical: int = Field(ge=0)
    up_float32_max_bound_ratio: float = Field(ge=0)
    hidden_matches_product: Literal[True]
    down_float32_max_bound_ratio: float = Field(ge=0)
    bfloat16_conversions_match: Literal[True]
    checkpoint_values_consistent: Literal[True]
    full_assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SeqaxSiluFusionCandidateCorrectness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxFeedForwardFusion
    uninstrumented_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: tuple[str, ...] = Field(min_length=13, max_length=13)
    checkpoint_capture_modes: tuple[str, ...] = Field(min_length=13, max_length=13)
    uninstrumented_metrics: SeqaxSiluFusionFinalOutputMetrics
    instrumented_metrics: SeqaxSiluFusionFinalOutputMetrics
    checkpoint_metrics: SeqaxSiluFusionCheckpointMetrics
    instrumentation_output_exact: Literal[True]

    @model_validator(mode="after")
    def hashes_are_valid(self) -> SeqaxSiluFusionCandidateCorrectness:
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.checkpoint_sha256):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CHECKPOINT_HASH_INVALID")
        expected_modes = (
            SEQAX_SILU_FUSION_SEPARATE_CHECKPOINT_CAPTURE_MODES
            if self.candidate is SeqaxFeedForwardFusion.SEPARATE
            else SEQAX_SILU_FUSION_FUSED_CHECKPOINT_CAPTURE_MODES
        )
        if self.checkpoint_capture_modes != expected_modes:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CHECKPOINT_CAPTURE_INVALID")
        return self


class SeqaxSiluFusionCorrectnessObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: int = Field(ge=0)
    input_sha256: tuple[str, ...] = Field(min_length=13, max_length=13)
    cpu_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[
        SeqaxSiluFusionCandidateCorrectness,
        SeqaxSiluFusionCandidateCorrectness,
    ]
    candidate_uninstrumented_outputs_exact: Literal[True]
    candidate_instrumented_outputs_exact: Literal[True]
    candidate_checkpoints_exact: Literal[True]
    boundary_case: bool
    boundary_strict_mutant_difference_count: int = Field(ge=0)
    boundary_mutant_rejected: bool

    @model_validator(mode="after")
    def observation_is_consistent(self) -> SeqaxSiluFusionCorrectnessObservation:
        if tuple(value.candidate for value in self.candidates) != (
            SeqaxFeedForwardFusion.SEPARATE,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CANDIDATE_ORDER_MISMATCH")
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.input_sha256):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_INPUT_HASH_INVALID")
        if self.boundary_case != (self.boundary_strict_mutant_difference_count > 0):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_BOUNDARY_DISCRIMINATOR_MISMATCH")
        if self.boundary_case != self.boundary_mutant_rejected:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_BOUNDARY_REJECTION_MISMATCH")
        return self


class SeqaxSiluFusionCorrectnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_RESULT_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_RESULT_SCHEMA
    )
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: SeqaxSiluFusionCorrectnessSourceAuthority
    host: SeqaxSiluFusionCorrectnessHost
    devices: tuple[SeqaxSiluFusionCorrectnessDevice, ...] = Field(min_length=8, max_length=8)
    plans: tuple[SeqaxSiluFusionCorrectnessPlan, SeqaxSiluFusionCorrectnessPlan]
    observations: tuple[SeqaxSiluFusionCorrectnessObservation, ...] = Field(
        min_length=6,
        max_length=6,
    )
    model_outputs_executed: Literal[True]
    full_inputs_persisted: Literal[True]
    full_outputs_persisted: Literal[True]
    full_checkpoints_persisted: Literal[True]
    producer_passed: Literal[True]
    timing_collected: Literal[False]
    profile_collected: Literal[False]
    worker_pid: int = Field(gt=0)
    worker_nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_import_root: str = Field(min_length=1)
    worker_environment: dict[str, str]
    compiler_environment: dict[str, str]

    @model_validator(mode="after")
    def result_is_complete(self) -> SeqaxSiluFusionCorrectnessResult:
        if tuple(value.id for value in self.devices) != tuple(range(8)):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_DEVICE_ORDER_MISMATCH")
        if tuple(value.candidate for value in self.plans) != (
            SeqaxFeedForwardFusion.SEPARATE,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PLAN_ORDER_MISMATCH")
        return self

    @computed_field
    @property
    def result_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_WORKER_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_WORKER_SCHEMA
    )
    contract: SeqaxSiluFusionCorrectnessContract
    claim: SeqaxSiluFusionCorrectnessAttemptClaim
    source: SeqaxSiluFusionCorrectnessSourceAuthority

    @model_validator(mode="after")
    def request_is_bound(self) -> SeqaxSiluFusionCorrectnessWorkerRequest:
        if (
            self.claim.contract_id != self.contract.contract_id
            or self.claim.source_commit != self.source.source_commit
            or self.claim.source_tree != self.source.source_tree
            or self.source.runtime != self.contract.runtime
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_WORKER_REQUEST_MISMATCH")
        return self


class SeqaxSiluFusionCorrectnessWorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    result: SeqaxSiluFusionCorrectnessResult

    @model_validator(mode="after")
    def result_matches_contract(self) -> SeqaxSiluFusionCorrectnessWorkerResult:
        contract = default_seqax_silu_fusion_correctness_contract(self.result.source.runtime)
        expected_seeds = (*contract.correctness_seeds, contract.boundary_seed)
        if (
            self.result.contract_id != contract.contract_id
            or self.result.source.runtime != contract.runtime
            or tuple(value.candidate_semantic_id for value in self.result.plans)
            != contract.candidate_semantic_ids
            or tuple(value.seed for value in self.result.observations) != expected_seeds
            or any(value.boundary_case for value in self.result.observations[:-1])
            or not self.result.observations[-1].boundary_case
            or self.result.worker_environment != contract.worker_environment
            or self.result.compiler_environment != contract.compiler_environment
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_WORKER_RESULT_MISMATCH")
        return self


class SeqaxSiluFusionCorrectnessReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_RECEIPT_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_RECEIPT_SCHEMA
    )
    result: SeqaxSiluFusionCorrectnessResult
    final_ledger_state: Literal[RunState.ACCEPTED]
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    independent_replay_performed: Literal[True]
    archive_required: Literal[True]
    retry_authorized: Literal[False]
    resume_authorized: Literal[False]
    timing_authorized: Literal[False]
    profile_authorized: Literal[False]

    @model_validator(mode="after")
    def receipt_is_closed(self) -> SeqaxSiluFusionCorrectnessReceipt:
        paths = tuple(value.path for value in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_MANIFEST_INVALID")
        if any(
            value.role not in SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_ROLES
            for value in self.artifacts
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_ROLE_INVALID")
        return self

    @computed_field
    @property
    def receipt_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessReplaySeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_REPLAY_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_REPLAY_SCHEMA
    )
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    output_root: str = Field(min_length=1)
    independent_replay_performed: Literal[True]
    retry_authorized: Literal[False]
    resume_authorized: Literal[False]

    @computed_field
    @property
    def replay_seal_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessArchiveSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_ARCHIVE_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_ARCHIVE_SCHEMA
    )
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_seal_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_path: str = Field(min_length=1)
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_member_count: int = Field(gt=0)
    single_root: Literal[True]
    extracted_replay_performed: Literal[True]

    @computed_field
    @property
    def archive_seal_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_SCHEMA
    )
    phase: Literal[
        "worker-launch",
        "worker-execution",
        "worker-result",
        "preliminary-replay",
        "acceptance",
        "final-replay",
        "archive",
    ]
    returncode: int | None
    stdout: str
    stderr: str
    error_type: str | None
    error_message: str | None
    final_ledger_state: RunState
    model_outputs_execution_status: Literal[
        "not-executed",
        "may-have-executed",
        "executed",
    ]
    timing_collected: Literal[False]
    profile_collected: Literal[False]
    retry_authorized: Literal[False]
    resume_authorized: Literal[False]

    @model_validator(mode="after")
    def failure_phase_is_consistent(self) -> SeqaxSiluFusionCorrectnessFailure:
        if self.phase == "worker-execution":
            if self.returncode is None or self.returncode == 0:
                raise ValueError("worker execution failure requires a nonzero return code")
        else:
            if not self.error_type or not self.error_message:
                raise ValueError("orchestrator failure requires an explicit exception")
            if self.phase == "worker-launch":
                if self.returncode is not None or self.final_ledger_state is not RunState.CREATED:
                    raise ValueError("worker launch failure state is invalid")
            elif self.returncode != 0:
                raise ValueError("post-worker failure requires a successful worker return")
        allowed_states = {
            "worker-result": {RunState.CORRECT},
            "preliminary-replay": {RunState.CORRECT},
            "acceptance": {RunState.CORRECT, RunState.VALIDATED, RunState.ACCEPTED},
            "final-replay": {RunState.ACCEPTED},
            "archive": {RunState.ACCEPTED},
        }
        if (
            self.phase in allowed_states
            and self.final_ledger_state not in allowed_states[self.phase]
        ):
            raise ValueError("orchestrator failure phase and ledger state disagree")
        return self


class SeqaxSiluFusionCorrectnessFailureReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_RECEIPT_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_RECEIPT_SCHEMA
    )
    claim: SeqaxSiluFusionCorrectnessAttemptClaim
    source: SeqaxSiluFusionCorrectnessSourceAuthority
    failure: SeqaxSiluFusionCorrectnessFailure
    incomplete_success_receipt: SeqaxSiluFusionCorrectnessReceipt | None
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    independent_replay_required: Literal[True]
    independent_replay_performed_at_receipt_creation: Literal[False]
    archive_required: Literal[True]
    retry_authorized: Literal[False]
    resume_authorized: Literal[False]

    @computed_field
    @property
    def failure_receipt_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessFailureReplaySeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_REPLAY_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_REPLAY_SCHEMA
    )
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_receipt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    output_root: str = Field(min_length=1)
    independent_replay_performed: Literal[True]
    retry_authorized: Literal[False]
    resume_authorized: Literal[False]

    @computed_field
    @property
    def failure_replay_seal_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


class SeqaxSiluFusionCorrectnessFailureArchiveSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARCHIVE_SCHEMA] = (
        SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARCHIVE_SCHEMA
    )
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_receipt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_replay_seal_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_path: str = Field(min_length=1)
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_member_count: int = Field(gt=0)
    single_root: Literal[True]
    extracted_replay_performed: Literal[True]

    @computed_field
    @property
    def failure_archive_seal_id(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_computed_fields=True))


def default_seqax_silu_fusion_correctness_contract(
    runtime: RuntimeIdentity,
) -> SeqaxSiluFusionCorrectnessContract:
    return SeqaxSiluFusionCorrectnessContract.model_construct(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        claim_scope=("fixed-seqax-silu-fusion-full-output-and-strict-mlp-checkpoint-equivalence"),
        compiler_evidence_status="verified",
        compiler_design_path="contracts/seqax-silu-fusion-design-v1.json",
        compiler_design_id=SEQAX_SILU_FUSION_COMPILER_DESIGN_ID,
        compiler_design_sha256=SEQAX_SILU_FUSION_COMPILER_DESIGN_SHA256,
        compiler_pair_record_path="contracts/seqax-silu-fusion-compiler-pair-v1.json",
        compiler_pair_evidence_path=(
            "/home/sudarshan/tpu-cake-evidence/"
            "seqax-silu-fusion-compiler-pair-ec6ca71-d4e91b70.json"
        ),
        compiler_pair_id=SEQAX_SILU_FUSION_COMPILER_PAIR_ID,
        compiler_pair_sha256=SEQAX_SILU_FUSION_COMPILER_PAIR_SHA256,
        compiler_capture_ids=SEQAX_SILU_FUSION_COMPILER_CAPTURE_IDS,
        candidate_semantic_ids=SEQAX_SILU_FUSION_CANDIDATE_SEMANTIC_IDS,
        compilation_source_root=SEQAX_SILU_FUSION_COMPILATION_ROOT,
        source_remote_url="https://github.com/sdrshn-nmbr/tpu-cake.git",
        source_branch="main",
        correctness_claim_registry_root=(
            "/home/sudarshan/tpu-cake-evidence/seqax-silu-fusion-correctness-claims"
        ),
        correctness_claim_key=SEQAX_SILU_FUSION_CORRECTNESS_SCHEMA,
        correctness_claim_identity_scope="contract-id",
        correctness_claim_reservation="exclusive-create-only",
        allow_retry=False,
        allow_resume=False,
        independent_replay_required=True,
        archive_required=True,
        timing_authorized=False,
        profile_authorized=False,
        uv_lock_sha256=("03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9"),
        parameters={
            "batch": 256,
            "data_mesh": 2,
            "feed_forward": 4096,
            "head": 8,
            "key_value_heads": 4,
            "layers": 1,
            "model": 32,
            "numerical_semantics": "typed_bf16_hidden_v2",
            "query_groups": 2,
            "rope_max_timescale": 256,
            "sequence": 1,
            "tensor_mesh": 4,
            "vocabulary": 16,
        },
        residual_norm_strategy=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        vector_execution=SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL,
        candidates=(
            SeqaxFeedForwardFusion.SEPARATE,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
        ),
        correctness_seeds=SEQAX_SILU_FUSION_CORRECTNESS_SEEDS,
        boundary_seed=SEQAX_SILU_FUSION_BOUNDARY_SEED,
        checkpoint_names=SEQAX_SILU_FUSION_CHECKPOINTS,
        checkpoint_capture_modes={
            SeqaxFeedForwardFusion.SEPARATE.value: (
                SEQAX_SILU_FUSION_SEPARATE_CHECKPOINT_CAPTURE_MODES
            ),
            SeqaxFeedForwardFusion.SILU_MULTIPLY.value: (
                SEQAX_SILU_FUSION_FUSED_CHECKPOINT_CAPTURE_MODES
            ),
        },
        output_shape=(256, 1, 16),
        output_dtype="float32",
        policy=SeqaxSiluFusionCorrectnessPolicy(
            input_generator="seqax-forward-inputs-v1",
            cpu_oracle="seqax-forward-canonical-reference-v1",
            cpu_reference="jax_cpu_reference_v1",
            numerical_policy_schema="bf16-forward-numerical-v6",
            numerical_semantics="typed_bf16_hidden_v2",
            cpu_reference_quantization_decimals=6,
            unit_roundoff=0.00390625,
            cpu_relative_l2_units=3.0,
            cpu_row_scaled_max_units=8.0,
            cross_path_relative_l2_units=2.0,
            cross_path_row_scaled_max_units=2.0,
            depth_scaling="sqrt_layers",
            row_scale_floor=1.0,
            metric_quantization_decimals=15,
            cpu_replay_rule="cpu_facing_numerical_bounds",
            checkpoint_storage_dtype="uint16",
            checkpoint_logical_dtype="bfloat16",
            checkpoint_encoding="bf16-bit-pattern-v1",
            mathematical_silu_max_ulp=1,
            rms_inverse_relative_error_units=4.0,
            require_float32_output=True,
            require_finite_output=True,
            require_each_candidate_uninstrumented_final_output_policy=True,
            require_each_candidate_instrumented_final_output_policy=True,
            require_each_candidate_uninstrumented_cpu_top1_match=True,
            require_each_candidate_instrumented_cpu_top1_match=True,
            require_each_candidate_checkpoint_values_consistent=True,
            require_exact_candidate_outputs=True,
            require_exact_candidate_checkpoints=True,
            require_exact_instrumentation_outputs=True,
            boundary_discriminator="omit-intermediate-bf16-silu-round",
            boundary_mutant_must_differ=True,
            require_full_inputs_persisted=True,
            require_full_outputs_persisted=True,
            require_full_checkpoints_persisted=True,
            require_model_outputs_executed=True,
            timing_collected=False,
            profile_collected=False,
        ),
        worker_environment={
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
        },
        compiler_environment={
            "LIBTPU_INIT_ARGS": " --xla_tpu_use_enhanced_launch_barrier=true",
            "TPU_LIBRARY_PATH": (
                "/home/sudarshan/tpu-cake-main/.venv/lib/python3.12/site-packages/libtpu/libtpu.so"
            ),
        },
        project="astral-medley-465922-b2",
        numeric_project_id="541760035156",
        zone="us-central1-c",
        hostname="tpu-cake-v7x-rsag-wx7r",
        instance_hostname=(
            "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
        ),
        machine_type="tpu7x-standard-4t",
        instance_id="5064039476077763048",
        cpu_platform="Intel Emerald Rapids",
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        mesh={"d": 2, "t": 4},
    )
