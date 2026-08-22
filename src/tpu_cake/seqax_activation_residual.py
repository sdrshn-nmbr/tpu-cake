from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256, semantic_seed
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy

SEQAX_ACTIVATION_RESIDUAL_SCHEMA = "seqax-activation-residual-v1"
SEQAX_ACTIVATION_RESIDUAL_COMPILATION_ROOT = "/home/sudarshan/tpu-cake-main"
SEQAX_ACTIVATION_RESIDUAL_QUESTION = (
    "On TPU7x d=2,t=4, does a one-MiB BF16 residual message created by scaling batch and "
    "sequence favor reduce-scatter, local residual, and deferred all-gather over residual "
    "all-reduce by more than 3% in resident whole-forward latency?"
)
SEQAX_ACTIVATION_RESIDUAL_SCOPE = "fixed-batch64-sequence64-model256-layer1-bf16-pallas-performance"
SEQAX_ACTIVATION_RESIDUAL_PARENT_FAILURE_RECORD_ID = (
    "1f62ff3394dedf1eca6cba4df92d6d2f316202a907897f75766a1017c48eab94"
)
SEQAX_ACTIVATION_RESIDUAL_CORRECTNESS_SEEDS = tuple(
    semantic_seed(SEQAX_ACTIVATION_RESIDUAL_SCHEMA, f"correctness:{index}") for index in range(5)
)
SEQAX_ACTIVATION_RESIDUAL_TIMING_SEED = semantic_seed(
    SEQAX_ACTIVATION_RESIDUAL_SCHEMA,
    "timing",
)


def _zero_collectives() -> CompilerCollectiveAnalysis:
    return CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=0,
        stablehlo_all_gather_count=0,
        compiler_reduce_scatter_count=0,
        compiler_all_reduce_count=0,
        compiler_all_gather_count=0,
        sparse_core_reduce_scatter_count=0,
        sparse_core_all_gather_count=0,
    )


class SeqaxActivationResidualPlanContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_pallas_compiler_collectives: CompilerCollectiveAnalysis
    expected_pallas_regions: int = Field(gt=0)
    expected_all_gathers: int = Field(ge=0)
    expected_all_reduces: int = Field(ge=0)
    expected_reduce_scatters: int = Field(ge=0)
    expected_ring_equivalent_ici_bytes_per_device: int = Field(gt=0)
    expected_peak_vmem_bytes_per_device: int = Field(gt=0)


class SeqaxActivationResidualDesignContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_ACTIVATION_RESIDUAL_SCHEMA] = SEQAX_ACTIVATION_RESIDUAL_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    question: str
    claim_scope: str
    parent_failure_record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compilation_source_root: str
    source_remote_url: str
    source_branch: Literal["main"]
    compiler_environment: dict[str, str]
    compile_input_mode: Literal["abstract-only"]
    compiler_identity_status: Literal["pending"]
    compile_capture_count: Literal[2]
    correctness_policy_status: Literal["pending-calibration"]
    timing_authorized: Literal[False]
    scaling_book_sources: tuple[str, ...] = Field(min_length=2, max_length=2)
    illustrative_latency_crossover_bytes: int = Field(gt=0)
    latency_crossover_role: str
    full_activation_bf16_bytes_per_data_shard: int = Field(gt=0)
    local_activation_bf16_bytes_per_device: int = Field(gt=0)
    partial_activation_f32_bytes_per_data_shard: int = Field(gt=0)
    prior_small_activation_bf16_bytes_per_data_shard: int = Field(gt=0)
    predicted_ring_savings_bytes_per_device: int = Field(gt=0)
    predicted_ring_savings_fraction: float = Field(gt=0, lt=1)
    numerical_rationale: str
    failed_workload_boundary_relation: str
    expected_standard_boundary_chains: Literal[2]
    boundary_reduce_scatter_input: str
    boundary_reduce_scatter_output: str
    boundary_residual_output: str
    boundary_all_gather_output: str
    top1_role: Literal["diagnostic-only"]
    parameters: dict[str, int | str]
    baseline: Literal[SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE]
    candidate: Literal[SeqaxResidualNormStrategy.STANDARD]
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    timing_seed: int
    warmup_iterations: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    paired_rounds: int = Field(ge=24)
    bootstrap_samples: int = Field(ge=10_000)
    confidence_level: float = Field(gt=0, lt=1)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    allow_early_stopping: Literal[False]
    allow_retry: Literal[False]
    candidates_resident_together: Literal[True]
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
    candidates: tuple[SeqaxActivationResidualPlanContract, ...] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxActivationResidualDesignContract:
        expected = default_seqax_activation_residual_design_contract(self.runtime)
        if self.model_dump(exclude_computed_fields=True) != expected.model_dump(
            exclude_computed_fields=True
        ):
            raise ValueError("Seqax activation residual design protocol mismatch")
        return self

    @computed_field
    @property
    def design_id(self) -> str:
        return model_identity_sha256(self)


def _parameters() -> dict[str, int | str]:
    return {
        "batch": 64,
        "data_mesh": 2,
        "feed_forward": 256,
        "head": 64,
        "key_value_heads": 4,
        "layers": 1,
        "model": 256,
        "numerical_semantics": "typed_bf16_hidden_v2",
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 64,
        "tensor_mesh": 4,
        "vocabulary": 256,
    }


def _plans() -> tuple[SeqaxActivationResidualPlanContract, ...]:
    zero = "0" * 64
    zero_collectives = _zero_collectives()
    return (
        SeqaxActivationResidualPlanContract(
            candidate=SeqaxResidualNormStrategy.STANDARD,
            distributed_schedule_sha256=(
                "9c4175fe6bb1691a47ccc359ad49ef470586906f1654911bfb90f2bed090b703"
            ),
            physical_schedule_sha256=(
                "6ee34bf23cfeb82d98aaeeb66710e9eaaaa41db822b2f49a7e43d9eb6b2b9fc3"
            ),
            pallas_source_sha256=(
                "ddf9aaabbfb3ee01459830f683297f3db0541d5d269b6e5b4a5fbf8c58ebbaba"
            ),
            pallas_manifest_sha256=(
                "d95024afcbbef237ad8a9770035859e09a8d380a660fc3a88a48a02d2a3e7994"
            ),
            pallas_stablehlo_sha256=zero,
            control_stablehlo_sha256=zero,
            expected_pallas_compiler_collectives=zero_collectives,
            expected_pallas_regions=9,
            expected_all_gathers=17,
            expected_all_reduces=0,
            expected_reduce_scatters=3,
            expected_ring_equivalent_ici_bytes_per_device=12_948_736,
            expected_peak_vmem_bytes_per_device=5_177_600,
        ),
        SeqaxActivationResidualPlanContract(
            candidate=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
            distributed_schedule_sha256=(
                "20ef7196d0b823840cf13511576da1f3f6284dfce9c1fb8c8248c9375d9b7196"
            ),
            physical_schedule_sha256=(
                "8c6c836bd6d8c5e27521c9f0d48657cddb532c2b2b5795983b0347975a1fd595"
            ),
            pallas_source_sha256=(
                "f39f34f62f7716ac9de75cbd6427883abf91caf46b0c59f1860abd1101e7b1a7"
            ),
            pallas_manifest_sha256=(
                "e6d3e90f224113f18648ffdbc39ed9f0bf99facfc6144c481491094f785c16e1"
            ),
            pallas_stablehlo_sha256=zero,
            control_stablehlo_sha256=zero,
            expected_pallas_compiler_collectives=zero_collectives,
            expected_pallas_regions=9,
            expected_all_gathers=15,
            expected_all_reduces=2,
            expected_reduce_scatters=1,
            expected_ring_equivalent_ici_bytes_per_device=16_094_464,
            expected_peak_vmem_bytes_per_device=5_177_600,
        ),
    )


def default_seqax_activation_residual_design_contract(
    runtime: RuntimeIdentity,
) -> SeqaxActivationResidualDesignContract:
    return SeqaxActivationResidualDesignContract.model_construct(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        question=SEQAX_ACTIVATION_RESIDUAL_QUESTION,
        claim_scope=SEQAX_ACTIVATION_RESIDUAL_SCOPE,
        parent_failure_record_id=SEQAX_ACTIVATION_RESIDUAL_PARENT_FAILURE_RECORD_ID,
        compilation_source_root=SEQAX_ACTIVATION_RESIDUAL_COMPILATION_ROOT,
        source_remote_url="https://github.com/sdrshn-nmbr/tpu-cake.git",
        source_branch="main",
        compiler_environment={
            "LIBTPU_INIT_ARGS": " --xla_tpu_use_enhanced_launch_barrier=true",
            "TPU_LIBRARY_PATH": "/home/sudarshan/tpu-cake-main/.venv/lib/python3.12/site-packages/libtpu/libtpu.so",
        },
        compile_input_mode="abstract-only",
        compiler_identity_status="pending",
        compile_capture_count=2,
        correctness_policy_status="pending-calibration",
        timing_authorized=False,
        scaling_book_sources=(
            "https://jax-ml.github.io/scaling-book/sharding/",
            "https://jax-ml.github.io/scaling-book/inference/#appendix-c-latency-bound-communications",
        ),
        illustrative_latency_crossover_bytes=360_000,
        latency_crossover_role="illustrative-hardware-specific-not-a-tpu7x-threshold",
        full_activation_bf16_bytes_per_data_shard=1_048_576,
        local_activation_bf16_bytes_per_device=262_144,
        partial_activation_f32_bytes_per_data_shard=2_097_152,
        prior_small_activation_bf16_bytes_per_data_shard=512,
        predicted_ring_savings_bytes_per_device=3_145_728,
        predicted_ring_savings_fraction=0.19545403935166775,
        numerical_rationale=(
            "scale the collective message through batch and sequence while preserving the "
            "previously validated model width and frozen input generator"
        ),
        failed_workload_boundary_relation=(
            "same-one-mib-boundary-with-sixteen-times-shorter-model-axis-reductions"
        ),
        expected_standard_boundary_chains=2,
        boundary_reduce_scatter_input="f32[32,64,256]",
        boundary_reduce_scatter_output="f32[32,64,64]",
        boundary_residual_output="bf16[32,64,64]",
        boundary_all_gather_output="bf16[32,64,256]",
        top1_role="diagnostic-only",
        parameters=_parameters(),
        baseline=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        candidate=SeqaxResidualNormStrategy.STANDARD,
        correctness_seeds=SEQAX_ACTIVATION_RESIDUAL_CORRECTNESS_SEEDS,
        timing_seed=SEQAX_ACTIVATION_RESIDUAL_TIMING_SEED,
        warmup_iterations=5,
        measured_iterations=5,
        paired_rounds=32,
        bootstrap_samples=100_000,
        confidence_level=0.99,
        minimum_practical_improvement=0.03,
        allow_early_stopping=False,
        allow_retry=False,
        candidates_resident_together=True,
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
