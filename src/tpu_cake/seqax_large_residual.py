from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256, semantic_seed
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy

SEQAX_LARGE_RESIDUAL_SCHEMA = "seqax-large-residual-v1"
SEQAX_LARGE_RESIDUAL_COMPILATION_ROOT = "/home/sudarshan/tpu-cake-main"
SEQAX_LARGE_RESIDUAL_QUESTION = (
    "On TPU7x d=2,t=4, does the typed BF16 reduce-scatter, local residual, "
    "and deferred all-gather schedule beat residual all-reduce by more than 3% "
    "in resident whole-forward latency?"
)
SEQAX_LARGE_RESIDUAL_SCOPE = "fixed-model4096-layer1-sequence128-bf16-pallas-performance"
SEQAX_LARGE_RESIDUAL_TIMING_SEED = semantic_seed(
    SEQAX_LARGE_RESIDUAL_SCHEMA,
    "timing",
)
SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS = tuple(
    semantic_seed(SEQAX_LARGE_RESIDUAL_SCHEMA, f"correctness:{index}") for index in range(5)
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


class SeqaxLargeResidualPlanContract(BaseModel):
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


class SeqaxLargeResidualContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_LARGE_RESIDUAL_SCHEMA] = SEQAX_LARGE_RESIDUAL_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    question: str
    claim_scope: str
    compilation_source_root: str
    compiler_identity_status: Literal["pending", "pinned"]
    compile_capture_count: int = Field(gt=0)
    scaling_book_sources: tuple[str, ...] = Field(min_length=2, max_length=2)
    illustrative_latency_crossover_bytes: int = Field(gt=0)
    full_activation_bf16_bytes_per_data_shard: int = Field(gt=0)
    local_activation_bf16_bytes_per_device: int = Field(gt=0)
    partial_activation_f32_bytes_per_data_shard: int = Field(gt=0)
    prior_small_activation_bf16_bytes_per_data_shard: int = Field(gt=0)
    parameters: dict[str, int | str]
    baseline: SeqaxResidualNormStrategy
    candidate: SeqaxResidualNormStrategy
    timing_seed: int
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    warmup_iterations: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    paired_rounds: int = Field(ge=24)
    bootstrap_samples: int = Field(ge=10_000)
    confidence_level: float = Field(gt=0, lt=1)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    allow_early_stopping: bool
    allow_further_retry: bool
    candidates_resident_together: bool
    profile_only_after_decision: bool
    require_native_standard_reduce_scatters: int = Field(gt=0)
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    mesh: dict[str, int]
    candidates: tuple[SeqaxLargeResidualPlanContract, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxLargeResidualContract:
        expected = default_seqax_large_residual_contract(self.runtime)
        if self.model_dump(
            exclude={"candidates"}, exclude_computed_fields=True
        ) != expected.model_dump(
            exclude={"candidates"},
            exclude_computed_fields=True,
        ):
            raise ValueError("Seqax large residual protocol mismatch")
        expected_static = tuple(
            value.model_dump(
                exclude={
                    "pallas_stablehlo_sha256",
                    "control_stablehlo_sha256",
                    "expected_pallas_compiler_collectives",
                }
            )
            for value in expected.candidates
        )
        observed_static = tuple(
            value.model_dump(
                exclude={
                    "pallas_stablehlo_sha256",
                    "control_stablehlo_sha256",
                    "expected_pallas_compiler_collectives",
                }
            )
            for value in self.candidates
        )
        if observed_static != expected_static:
            raise ValueError("Seqax large residual plan mismatch")
        zero = "0" * 64
        zero_collectives = _zero_collectives()
        if self.compiler_identity_status == "pending":
            if any(
                plan.pallas_stablehlo_sha256 != zero
                or plan.control_stablehlo_sha256 != zero
                or plan.expected_pallas_compiler_collectives != zero_collectives
                for plan in self.candidates
            ):
                raise ValueError("Pending Seqax large residual compiler identities must be zero")
        elif any(
            plan.pallas_stablehlo_sha256 == zero
            or plan.control_stablehlo_sha256 == zero
            or plan.expected_pallas_compiler_collectives == zero_collectives
            for plan in self.candidates
        ):
            raise ValueError("Pinned Seqax large residual compiler identities are incomplete")
        return self

    @computed_field
    @property
    def contract_id(self) -> str:
        return model_identity_sha256(self)


def _parameters() -> dict[str, int | str]:
    return {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 1024,
        "head": 64,
        "key_value_heads": 4,
        "layers": 1,
        "model": 4096,
        "numerical_semantics": "typed_bf16_hidden_v2",
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 128,
        "tensor_mesh": 4,
        "vocabulary": 256,
    }


def _pending_plans() -> tuple[SeqaxLargeResidualPlanContract, ...]:
    zero = "0" * 64
    return (
        SeqaxLargeResidualPlanContract(
            candidate=SeqaxResidualNormStrategy.STANDARD,
            distributed_schedule_sha256=(
                "b72a0fa72040601cae8d24185ebb6d0890f9b08e4be059e046426bee16c69ca0"
            ),
            physical_schedule_sha256=(
                "c725424e3a7c4c71feb175f1d6243704dceeafd6f72a7cb9c71c67790bca2820"
            ),
            pallas_source_sha256=(
                "a90357c41554d452fdfeaee05d451cc9a9a1a79a5d4c5d7f2ec73d7448e1b5db"
            ),
            pallas_manifest_sha256=(
                "a503daf982286e87327b23b25ac28f6af5dce463dffccad55387ed8db8d343b4"
            ),
            pallas_stablehlo_sha256=zero,
            control_stablehlo_sha256=zero,
            expected_pallas_compiler_collectives=_zero_collectives(),
            expected_pallas_regions=9,
            expected_all_gathers=17,
            expected_all_reduces=0,
            expected_reduce_scatters=3,
            expected_ring_equivalent_ici_bytes_per_device=23_154_688,
            expected_peak_vmem_bytes_per_device=12_357_632,
        ),
        SeqaxLargeResidualPlanContract(
            candidate=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
            distributed_schedule_sha256=(
                "3b4129d67948a82620f9fa595f9690fd00705ee95af6bbc9641e486071c865c0"
            ),
            physical_schedule_sha256=(
                "b9b692967b4a82d15362ddcd0201e2427cb10cb4b1ac26bfc5dadfde8cc52ddc"
            ),
            pallas_source_sha256=(
                "069d5499a03523d2b945f532f91f53b4b3f7dbbdb9ac5ea396e8e4ec76b08e3f"
            ),
            pallas_manifest_sha256=(
                "18308733f03c8a7e0ee0a0c7b6c21c759c6e63475e918b17203072179eb196e5"
            ),
            pallas_stablehlo_sha256=zero,
            control_stablehlo_sha256=zero,
            expected_pallas_compiler_collectives=_zero_collectives(),
            expected_pallas_regions=9,
            expected_all_gathers=15,
            expected_all_reduces=2,
            expected_reduce_scatters=1,
            expected_ring_equivalent_ici_bytes_per_device=26_300_416,
            expected_peak_vmem_bytes_per_device=13_391_872,
        ),
    )


def default_seqax_large_residual_contract(runtime: RuntimeIdentity) -> SeqaxLargeResidualContract:
    return SeqaxLargeResidualContract.model_construct(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        question=SEQAX_LARGE_RESIDUAL_QUESTION,
        claim_scope=SEQAX_LARGE_RESIDUAL_SCOPE,
        compilation_source_root=SEQAX_LARGE_RESIDUAL_COMPILATION_ROOT,
        compiler_identity_status="pending",
        compile_capture_count=2,
        scaling_book_sources=(
            "https://jax-ml.github.io/scaling-book/sharding/",
            "https://jax-ml.github.io/scaling-book/inference/#appendix-c-latency-bound-communications",
        ),
        illustrative_latency_crossover_bytes=360_000,
        full_activation_bf16_bytes_per_data_shard=1_048_576,
        local_activation_bf16_bytes_per_device=262_144,
        partial_activation_f32_bytes_per_data_shard=2_097_152,
        prior_small_activation_bf16_bytes_per_data_shard=512,
        parameters=_parameters(),
        baseline=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        candidate=SeqaxResidualNormStrategy.STANDARD,
        timing_seed=SEQAX_LARGE_RESIDUAL_TIMING_SEED,
        correctness_seeds=SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS,
        warmup_iterations=5,
        measured_iterations=5,
        paired_rounds=32,
        bootstrap_samples=100_000,
        confidence_level=0.99,
        minimum_practical_improvement=0.03,
        allow_early_stopping=False,
        allow_further_retry=False,
        candidates_resident_together=True,
        profile_only_after_decision=True,
        require_native_standard_reduce_scatters=3,
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        mesh={"d": 2, "t": 4},
        candidates=_pending_plans(),
    )
