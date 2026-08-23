from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256, semantic_seed
from tpu_cake.seqax_contract_types import (
    SeqaxFeedForwardFusion,
    SeqaxFeedForwardVectorExecution,
    SeqaxResidualNormStrategy,
)

SEQAX_SILU_FUSION_DESIGN_SCHEMA = "seqax-silu-fusion-design-v1"
SEQAX_SILU_FUSION_COMPILATION_ROOT = "/home/sudarshan/tpu-cake-main"
SEQAX_SILU_FUSION_QUESTION = (
    "On one fixed synthetic Seqax TPU7x workload, does replacing two strict-BF16 "
    "full-local Pallas kernels for SiLU and multiply with one full-local Pallas kernel "
    "that preserves the same BF16 rounding boundary improve resident complete-forward "
    "latency by at least 3% with a 99% paired confidence interval?"
)
SEQAX_SILU_FUSION_SCOPE = (
    "fixed-batch256-sequence1-model32-feed-forward4096-layer1-typed-bf16-hidden-"
    "pallas-full-local-complete-forward"
)
SEQAX_SILU_FUSION_CORRECTNESS_SEEDS = tuple(
    semantic_seed(SEQAX_SILU_FUSION_DESIGN_SCHEMA, f"correctness:{index}") for index in range(5)
)
SEQAX_SILU_FUSION_TIMING_SEED = semantic_seed(
    SEQAX_SILU_FUSION_DESIGN_SCHEMA,
    "timing",
)
SEQAX_SILU_FUSION_BOUNDARY_SEED = semantic_seed(
    SEQAX_SILU_FUSION_DESIGN_SCHEMA,
    "boundary",
)


class SeqaxSiluFusionPlanContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxFeedForwardFusion
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_strict_vector_kernels: tuple[str, ...] = Field(min_length=1, max_length=2)
    expected_strict_vector_regions: int = Field(ge=1, le=2)
    expected_physical_vector_operations: int = Field(gt=0)
    expected_pallas_regions: int = Field(gt=0)
    expected_all_gathers: int = Field(ge=0)
    expected_all_reduces: int = Field(ge=0)
    expected_reduce_scatters: int = Field(ge=0)
    allocated_vmem_bytes_per_device: int = Field(gt=0)
    peak_live_vmem_bytes_per_device: int = Field(gt=0)
    ring_equivalent_ici_bytes_per_device: int = Field(gt=0)


class SeqaxSiluFusionDesignContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_DESIGN_SCHEMA] = SEQAX_SILU_FUSION_DESIGN_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    question: str
    claim_scope: str
    compilation_source_root: str
    source_remote_url: str
    source_branch: Literal["main"]
    worker_environment: dict[str, str]
    compiler_environment: dict[str, str]
    compile_input_mode: Literal["abstract-only"]
    compiler_capture_status: Literal["pending"]
    compile_capture_count: Literal[2]
    compile_capture_processes: Literal[2]
    compiler_pair_record_required: Literal[True]
    compiler_claim_registry_root: str
    compiler_claim_key: Literal["seqax-silu-fusion-design-v1"]
    compiler_claim_identity_scope: Literal["design-id-and-ordinal"]
    compiler_failure_evidence_policy: Literal["persist-raw-ir-and-analysis-before-semantic-gates"]
    compile_both_raw_candidates_before_semantic_gates: Literal[True]
    compiler_failure_receipt_required: Literal[True]
    compiler_failure_independent_replay_required: Literal[True]
    compiler_collective_gate_error_includes_observed_values: Literal[True]
    capture_ordinals: tuple[Literal[0], Literal[1]]
    ordinal_one_requires_ordinal_zero_replay_seal: Literal[True]
    allow_compile_retry: Literal[False]
    allow_compile_resume: Literal[False]
    model_outputs_executed_during_compile: Literal[False]
    correctness_policy_status: Literal["pending-dedicated-contract"]
    timing_authorized: Literal[False]
    scaling_book_sources: tuple[str, str]
    performance_hypothesis: str
    peak_memory_hypothesis: Literal["no-modeled-peak-vmem-improvement"]
    parameters: dict[str, int | str]
    residual_norm_strategy: Literal[SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE]
    vector_execution: Literal[SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL]
    baseline: Literal[SeqaxFeedForwardFusion.SEPARATE]
    candidate: Literal[SeqaxFeedForwardFusion.SILU_MULTIPLY]
    local_activation_shape: Literal["bf16[128,1,1024]"]
    local_activation_bytes_per_device: Literal[262144]
    activation_bytes_per_data_replica: Literal[1048576]
    eliminated_scheduled_intermediate_bytes_per_device: Literal[262144]
    expected_allocated_vmem_savings_bytes_per_device: Literal[262144]
    expected_peak_live_vmem_savings_bytes_per_device: Literal[0]
    baseline_compiler_requirement: Literal[
        "two-live-strict-vector-custom-calls-with-silu-output-feeding-multiply"
    ]
    candidate_compiler_requirement: Literal[
        "one-live-strict-fused-vector-custom-call-with-distinct-gate-up-and-live-down-use"
    ]
    gate_projection_pallas_region: Literal[5]
    up_projection_pallas_region: Literal[6]
    down_projection_pallas_region: Literal[7]
    compiler_names_are_evidence_not_semantic_identity: Literal[True]
    dead_custom_call_decoys_forbidden: Literal[True]
    identical_reachable_collectives_required: Literal[True]
    identical_non_mlp_physical_work_required: Literal[True]
    correctness_seeds: tuple[int, int, int, int, int]
    boundary_seed: int
    boundary_mutant_without_intermediate_bf16_round_must_fail: Literal[True]
    timing_seed: int
    warmup_iterations: Literal[5]
    measured_iterations_per_round: Literal[5]
    paired_rounds: Literal[32]
    bootstrap_samples: Literal[100000]
    confidence_level: Literal[0.99]
    minimum_practical_improvement: Literal[0.03]
    allow_early_stopping: Literal[False]
    allow_timing_retry: Literal[False]
    candidates_resident_together: Literal[True]
    timing_uses_uninstrumented_binaries: Literal[True]
    profile_only_after_decision: Literal[True]
    project: str
    numeric_project_id: str
    zone: str
    hostname: str
    instance_hostname: str
    machine_type: str
    instance_id: str
    cpu_platform: str
    runtime: RuntimeIdentity
    backend: Literal["tpu"]
    device_kind: Literal["TPU7x"]
    device_count: Literal[8]
    mesh: dict[str, int]
    candidates: tuple[SeqaxSiluFusionPlanContract, SeqaxSiluFusionPlanContract]

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxSiluFusionDesignContract:
        expected = default_seqax_silu_fusion_design_contract(self.runtime)
        if self.model_dump(exclude_computed_fields=True) != expected.model_dump(
            exclude_computed_fields=True
        ):
            raise ValueError("Seqax SiLU fusion design protocol mismatch")
        baseline, candidate = self.candidates
        if (
            baseline.allocated_vmem_bytes_per_device - candidate.allocated_vmem_bytes_per_device
            != self.expected_allocated_vmem_savings_bytes_per_device
            or baseline.peak_live_vmem_bytes_per_device - candidate.peak_live_vmem_bytes_per_device
            != self.expected_peak_live_vmem_savings_bytes_per_device
        ):
            raise ValueError("Seqax SiLU fusion resource delta mismatch")
        return self

    @computed_field
    @property
    def design_id(self) -> str:
        return model_identity_sha256(self)


def _parameters() -> dict[str, int | str]:
    return {
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
    }


def _plans() -> tuple[SeqaxSiluFusionPlanContract, SeqaxSiluFusionPlanContract]:
    common = {
        "expected_pallas_regions": 9,
        "expected_all_gathers": 15,
        "expected_all_reduces": 2,
        "expected_reduce_scatters": 1,
        "peak_live_vmem_bytes_per_device": 1_452_304,
        "ring_equivalent_ici_bytes_per_device": 323_744,
    }
    return (
        SeqaxSiluFusionPlanContract(
            candidate=SeqaxFeedForwardFusion.SEPARATE,
            distributed_schedule_sha256=(
                "700def05668c4c06bc85fc5ebf9cab378ba8c78f6eda59e5499ea3f9b30b42fd"
            ),
            physical_schedule_sha256=(
                "66893256ed29d9b19717d4da2fbda664dba8c42ef61885b6d8eb715893e593f6"
            ),
            pallas_source_sha256=(
                "994d344b9c0680d050ad709a1ed5b66f20ee54f51e4c9df01a8683b96e112127"
            ),
            pallas_manifest_sha256=(
                "df3ee329ff3842a3b0d8d069a9dd627c7988cde8e4fd336ba079e22363bfc80c"
            ),
            expected_strict_vector_kernels=(
                "seqax_strict_bf16_silu",
                "seqax_strict_bf16_multiply",
            ),
            expected_strict_vector_regions=2,
            expected_physical_vector_operations=43,
            allocated_vmem_bytes_per_device=3_041_840,
            **common,
        ),
        SeqaxSiluFusionPlanContract(
            candidate=SeqaxFeedForwardFusion.SILU_MULTIPLY,
            distributed_schedule_sha256=(
                "d6e5a96a416584c8afa5b625beaa38e276af42a4fae492e822885a6a216640bd"
            ),
            physical_schedule_sha256=(
                "f9b0aaaac12d3f5d32825610632c1f8f41bc4828542d86b9d2a361e473a09ef7"
            ),
            pallas_source_sha256=(
                "3ab75da76e74f4221922a1538ee15905fdde4e842633080869f7ef11e3c49db4"
            ),
            pallas_manifest_sha256=(
                "e0643c5e1b4baa4a0710b80ebb263249eba5fd08d67ab78b8e159c405ee0e51f"
            ),
            expected_strict_vector_kernels=("seqax_strict_bf16_silu_multiply",),
            expected_strict_vector_regions=1,
            expected_physical_vector_operations=42,
            allocated_vmem_bytes_per_device=2_779_696,
            **common,
        ),
    )


def default_seqax_silu_fusion_design_contract(
    runtime: RuntimeIdentity,
) -> SeqaxSiluFusionDesignContract:
    return SeqaxSiluFusionDesignContract.model_construct(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        question=SEQAX_SILU_FUSION_QUESTION,
        claim_scope=SEQAX_SILU_FUSION_SCOPE,
        compilation_source_root=SEQAX_SILU_FUSION_COMPILATION_ROOT,
        source_remote_url="https://github.com/sdrshn-nmbr/tpu-cake.git",
        source_branch="main",
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
            "TPU_LIBRARY_PATH": "/home/sudarshan/tpu-cake-main/.venv/lib/python3.12/site-packages/libtpu/libtpu.so",
        },
        compile_input_mode="abstract-only",
        compiler_capture_status="pending",
        compile_capture_count=2,
        compile_capture_processes=2,
        compiler_pair_record_required=True,
        compiler_claim_registry_root="/home/sudarshan/tpu-cake-evidence/seqax-silu-fusion-claims",
        compiler_claim_key="seqax-silu-fusion-design-v1",
        compiler_claim_identity_scope="design-id-and-ordinal",
        compiler_failure_evidence_policy=("persist-raw-ir-and-analysis-before-semantic-gates"),
        compile_both_raw_candidates_before_semantic_gates=True,
        compiler_failure_receipt_required=True,
        compiler_failure_independent_replay_required=True,
        compiler_collective_gate_error_includes_observed_values=True,
        capture_ordinals=(0, 1),
        ordinal_one_requires_ordinal_zero_replay_seal=True,
        allow_compile_retry=False,
        allow_compile_resume=False,
        model_outputs_executed_during_compile=False,
        correctness_policy_status="pending-dedicated-contract",
        timing_authorized=False,
        scaling_book_sources=(
            "https://jax-ml.github.io/scaling-book/roofline/",
            "https://jax-ml.github.io/scaling-book/tpus/",
        ),
        performance_hypothesis=(
            "remove one external low-arithmetic-intensity kernel and one scheduled BF16 "
            "intermediate while keeping Pallas ownership and strict BF16 semantics fixed"
        ),
        peak_memory_hypothesis="no-modeled-peak-vmem-improvement",
        parameters=_parameters(),
        residual_norm_strategy=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        vector_execution=SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL,
        baseline=SeqaxFeedForwardFusion.SEPARATE,
        candidate=SeqaxFeedForwardFusion.SILU_MULTIPLY,
        local_activation_shape="bf16[128,1,1024]",
        local_activation_bytes_per_device=262_144,
        activation_bytes_per_data_replica=1_048_576,
        eliminated_scheduled_intermediate_bytes_per_device=262_144,
        expected_allocated_vmem_savings_bytes_per_device=262_144,
        expected_peak_live_vmem_savings_bytes_per_device=0,
        baseline_compiler_requirement=(
            "two-live-strict-vector-custom-calls-with-silu-output-feeding-multiply"
        ),
        candidate_compiler_requirement=(
            "one-live-strict-fused-vector-custom-call-with-distinct-gate-up-and-live-down-use"
        ),
        gate_projection_pallas_region=5,
        up_projection_pallas_region=6,
        down_projection_pallas_region=7,
        compiler_names_are_evidence_not_semantic_identity=True,
        dead_custom_call_decoys_forbidden=True,
        identical_reachable_collectives_required=True,
        identical_non_mlp_physical_work_required=True,
        correctness_seeds=SEQAX_SILU_FUSION_CORRECTNESS_SEEDS,
        boundary_seed=SEQAX_SILU_FUSION_BOUNDARY_SEED,
        boundary_mutant_without_intermediate_bf16_round_must_fail=True,
        timing_seed=SEQAX_SILU_FUSION_TIMING_SEED,
        warmup_iterations=5,
        measured_iterations_per_round=5,
        paired_rounds=32,
        bootstrap_samples=100_000,
        confidence_level=0.99,
        minimum_practical_improvement=0.03,
        allow_early_stopping=False,
        allow_timing_retry=False,
        candidates_resident_together=True,
        timing_uses_uninstrumented_binaries=True,
        profile_only_after_decision=True,
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
        candidates=_plans(),
    )
