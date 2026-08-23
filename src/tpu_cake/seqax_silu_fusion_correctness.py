from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256
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
SEQAX_SILU_FUSION_COMPILER_DESIGN_ID = (
    "f77778c8c49b090c0c8717f81c32a6269e07b5b9d2fd505e36b243e5bbfc143a"
)
SEQAX_SILU_FUSION_COMPILER_DESIGN_SHA256 = (
    "bff54167ee3aa491903cf97bfdf5d4851415f5a0673c1bf2bc210851bc5f4dd8"
)
SEQAX_SILU_FUSION_COMPILER_PAIR_ID = (
    "63266abcb56a719a891b2d09a8e9b9b5b93143bf751fd0c69d16e3880c44b021"
)
SEQAX_SILU_FUSION_COMPILER_PAIR_SHA256 = (
    "4241e7035e3d90b6a12c476fbbbfba870ef76db9f6e2ff9222be9e120997dfb4"
)
SEQAX_SILU_FUSION_COMPILER_CAPTURE_IDS = (
    "e2b4fd712ff671e386707883f4d12bfc864e444487779d292a5d41752d6f8d49",
    "9bd531b3b5f377598a91abf26b8fc890980fd1042c66d15b129d0f7ad1f0633e",
)
SEQAX_SILU_FUSION_CANDIDATE_SEMANTIC_IDS = (
    "5851109941f77d2781e3a66efbb8052064cad1551be32054f2d8d3263697c6ef",
    "927de445547f6342fb558ba3c159f9dc99871b48604b4b5be9e729035781f9f9",
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
    compiler_design_path: Literal["contracts/seqax-silu-fusion-design-v1.json"]
    compiler_design_id: Literal[SEQAX_SILU_FUSION_COMPILER_DESIGN_ID]
    compiler_design_sha256: Literal[SEQAX_SILU_FUSION_COMPILER_DESIGN_SHA256]
    compiler_pair_record_path: Literal[
        "contracts/seqax-silu-fusion-compiler-pair-v1.json"
    ]
    compiler_pair_evidence_path: Literal[
        "/home/sudarshan/tpu-cake-evidence/seqax-silu-fusion-compiler-pair-f77778c-acde7301.json"
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
    uv_lock_sha256: Literal[
        "03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9"
    ]
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


def default_seqax_silu_fusion_correctness_contract(
    runtime: RuntimeIdentity,
) -> SeqaxSiluFusionCorrectnessContract:
    return SeqaxSiluFusionCorrectnessContract.model_construct(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        claim_scope=(
            "fixed-seqax-silu-fusion-full-output-and-strict-mlp-checkpoint-equivalence"
        ),
        compiler_design_path="contracts/seqax-silu-fusion-design-v1.json",
        compiler_design_id=SEQAX_SILU_FUSION_COMPILER_DESIGN_ID,
        compiler_design_sha256=SEQAX_SILU_FUSION_COMPILER_DESIGN_SHA256,
        compiler_pair_record_path="contracts/seqax-silu-fusion-compiler-pair-v1.json",
        compiler_pair_evidence_path=(
            "/home/sudarshan/tpu-cake-evidence/"
            "seqax-silu-fusion-compiler-pair-f77778c-acde7301.json"
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
        uv_lock_sha256=(
            "03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9"
        ),
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
                "/home/sudarshan/tpu-cake-main/.venv/lib/python3.12/site-packages/"
                "libtpu/libtpu.so"
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
