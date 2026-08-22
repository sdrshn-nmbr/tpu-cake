from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256, semantic_seed
from tpu_cake.inkling_gmm_route_corpus import InklingGmmRouteCorpusReport

INKLING_GMM_TILE_SEARCH_SCHEMA = "inkling-gmm-tile-search-v1"
GMM_COMPLETION_STEPS = tuple(range(2, 66))
GMM_LAYER_INDICES = tuple(range(2, 42))
GMM_CORRECTNESS_SEEDS = tuple(
    semantic_seed(INKLING_GMM_TILE_SEARCH_SCHEMA, "correctness", str(index)) for index in range(5)
)
GMM_IMPLEMENTATION_SOURCE_PATHS = (
    "engine/sglang-jax/python/sgl_jax/srt/models/inkling.py",
    "engine/sglang-jax/python/sgl_jax/srt/configs/model_config.py",
    "engine/sglang-jax/python/sgl_jax/srt/server_args.py",
    "engine/sglang-jax/python/sgl_jax/srt/eplb/eplb_algorithms.py",
    "engine/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py",
    "engine/sglang-jax/python/sgl_jax/srt/layers/moe.py",
    "engine/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py",
    "engine/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py",
    "engine/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py",
    "engine/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py",
    "engine/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py",
    "engine/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py",
    "engine/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py",
)


class GmmOperation(StrEnum):
    GATE = "gate"
    UP = "up"
    DOWN = "down"


class GmmArmName(StrEnum):
    INCUMBENT = "incumbent"
    SPARSE_M64 = "sparse-m64"
    SPARSE_M32 = "sparse-m32"
    SPLIT_N = "split-n"
    SPARSE_M64_SPLIT_N = "sparse-m64-split-n"


class RouteCorpusBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_scope: Literal["operational-assumption-no-emitted-step-id"] = (
        "operational-assumption-no-emitted-step-id"
    )


class GmmKernelAbi(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: GmmOperation
    lhs_dtype: Literal["bf16", "fp32"]
    rhs_dtype: Literal["bf16"] = "bf16"
    accumulator_dtype: Literal["fp32"] = "fp32"
    output_dtype: Literal["fp32"] = "fp32"
    k: int = Field(gt=0)
    n: int = Field(gt=0)
    zero_initialize: bool
    quantized: Literal[False] = False
    has_scale: Literal[False] = False
    has_bias: Literal[False] = False

    @model_validator(mode="after")
    def matches_production_call(self) -> GmmKernelAbi:
        expected = {
            GmmOperation.GATE: ("bf16", 4096, 2048, False),
            GmmOperation.UP: ("bf16", 4096, 2048, False),
            GmmOperation.DOWN: ("fp32", 2048, 4096, True),
        }[self.operation]
        if (self.lhs_dtype, self.k, self.n, self.zero_initialize) != expected:
            raise ValueError(f"{self.operation.value} GMM ABI mismatch")
        return self


class GmmProductionAbi(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    m: Literal[288] = 288
    global_group_count: Literal[256] = 256
    device_count: Literal[8] = 8
    local_experts_per_device: Literal[32] = 32
    group_offset_rule: Literal["device_index*32"] = "device_index*32"
    expert_location: Literal["trivial-identity-no-redundant-experts"] = (
        "trivial-identity-no-redundant-experts"
    )
    lhs_distribution: Literal["same-global-expert-sorted-lhs-per-device"] = (
        "same-global-expert-sorted-lhs-per-device"
    )
    kernels: tuple[GmmKernelAbi, GmmKernelAbi, GmmKernelAbi]

    @model_validator(mode="after")
    def operation_inventory_is_exact(self) -> GmmProductionAbi:
        if tuple(kernel.operation for kernel in self.kernels) != tuple(GmmOperation):
            raise ValueError("GMM operation inventory mismatch")
        if self.device_count * self.local_experts_per_device != self.global_group_count:
            raise ValueError("GMM expert partition mismatch")
        return self


class GmmCorpusProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    completion_steps: tuple[int, ...]
    layer_indices: tuple[int, ...]
    order: Literal["completion-major-layer-minor"] = "completion-major-layer-minor"
    timing_unit: Literal["one-ordered-64-step-by-40-layer-corpus-block"] = (
        "one-ordered-64-step-by-40-layer-corpus-block"
    )
    groups_are_independent_samples: Literal[False] = False

    @property
    def group_count(self) -> int:
        return len(self.completion_steps) * len(self.layer_indices)

    @model_validator(mode="after")
    def inventory_is_exact(self) -> GmmCorpusProtocol:
        if self.completion_steps != GMM_COMPLETION_STEPS:
            raise ValueError("GMM corpus completion-step inventory mismatch")
        if self.layer_indices != GMM_LAYER_INDICES:
            raise ValueError("GMM corpus layer inventory mismatch")
        return self


class GmmTileArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: GmmArmName
    tile_m: Literal[32, 64, 128]
    tile_k: Literal["K"] = "K"
    tile_n: Literal["N", "N/2"]

    def resolve_tiles(self, kernel: GmmKernelAbi) -> tuple[int, int, int]:
        tile_n = kernel.n if self.tile_n == "N" else kernel.n // 2
        return self.tile_m, kernel.k, tile_n


class GmmSearchProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    can_promote: Literal[False] = False
    selection_unit: Literal["full-corpus-block"] = "full-corpus-block"
    order: Literal["balanced-forward-reverse-latin-square"] = (
        "balanced-forward-reverse-latin-square"
    )
    layer_weight_banks: Literal[40] = 40
    layer_weight_banks_are_distinct: Literal[True] = True
    layer_weight_bank_order: Literal["moe-layer-order"] = "moe-layer-order"
    executables_resident: Literal[True] = True
    operands_resident: Literal[True] = True
    operands_shared_across_arms: Literal[True] = True
    compiler_preflight: Literal["reachable-exact-gmm-v2-scope-label-per-operation"] = (
        "reachable-exact-gmm-v2-scope-label-per-operation"
    )


class GmmTargetRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_type: Literal["TPU v7x"] = "TPU v7x"
    device_count: Literal[8] = 8
    server_tp_size: Literal[8] = 8
    server_ep_size: Literal[8] = 8
    gmm_expert_axis_size: Literal[8] = 8
    gmm_tensor_axis_size: Literal[1] = 1
    jax: Literal["0.11.0"] = "0.11.0"
    jaxlib: Literal["0.11.0"] = "0.11.0"
    libtpu: Literal["0.0.44.1"] = "0.0.44.1"


class GmmConfirmationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paired_rounds: Literal[32] = 32
    order_balance: Literal["alternating-ab-ba"] = "alternating-ab-ba"
    samples_per_arm_per_round: Literal[5] = 5
    samples_synchronized: Literal[True] = True
    sample_unit: Literal["full-corpus-block"] = "full-corpus-block"
    within_round_reduction: Literal["median-of-five-synchronized-full-corpus-blocks"] = (
        "median-of-five-synchronized-full-corpus-blocks"
    )
    statistic: Literal["paired-median-improvement"] = "paired-median-improvement"
    bootstrap: Literal["deterministic-paired-median"] = "deterministic-paired-median"
    bootstrap_seed_rule: Literal["semantic-seed(search-id,finalist)"] = (
        "semantic-seed(search-id,finalist)"
    )
    bootstrap_samples: Literal[100000] = 100_000
    confidence_level: Literal[0.99] = 0.99
    minimum_practical_improvement: Literal[0.03] = 0.03
    lower_bound_must_exceed_threshold: Literal[True] = True
    allow_early_stopping: Literal[False] = False
    allow_retry: Literal[False] = False
    executables_resident: Literal[True] = True
    operands_resident: Literal[True] = True


class GmmCorrectnessProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    numerical_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    seeds: tuple[int, ...]
    comparison: Literal["final-down-output-vs-incumbent"] = "final-down-output-vs-incumbent"
    cpu_oracle: Literal["fixed-order-active-spans"] = "fixed-order-active-spans"
    absolute_tolerance: float = Field(gt=0)
    relative_tolerance: float = Field(gt=0)
    tolerances_frozen_before_timing: Literal[True] = True
    require_finite_outputs: Literal[True] = True
    require_down_zero_outside_local_spans: Literal[True] = True

    @model_validator(mode="after")
    def seeds_are_exact(self) -> GmmCorrectnessProtocol:
        if self.seeds != GMM_CORRECTNESS_SEEDS:
            raise ValueError("GMM correctness seeds are not canonical")
        return self


class InklingGmmTileSearchContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["inkling-gmm-tile-search-v1"] = INKLING_GMM_TILE_SEARCH_SCHEMA
    name: str = Field(min_length=1)
    route_corpus: RouteCorpusBinding
    inkling_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    inkling_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    target_runtime: GmmTargetRuntime
    production_abi: GmmProductionAbi
    corpus: GmmCorpusProtocol
    arms: tuple[GmmTileArm, ...]
    search: GmmSearchProtocol
    confirmation: GmmConfirmationProtocol
    correctness: GmmCorrectnessProtocol

    @model_validator(mode="after")
    def contract_surface_is_exact(self) -> InklingGmmTileSearchContract:
        paths = tuple(source.path for source in self.implementation_source_manifest)
        if paths != GMM_IMPLEMENTATION_SOURCE_PATHS:
            raise ValueError("GMM implementation source manifest is incomplete or reordered")
        if len(paths) != len(set(paths)):
            raise ValueError("GMM implementation source manifest paths must be unique")
        expected_arms = (
            (GmmArmName.INCUMBENT, 128, "K", "N"),
            (GmmArmName.SPARSE_M64, 64, "K", "N"),
            (GmmArmName.SPARSE_M32, 32, "K", "N"),
            (GmmArmName.SPLIT_N, 128, "K", "N/2"),
            (GmmArmName.SPARSE_M64_SPLIT_N, 64, "K", "N/2"),
        )
        observed_arms = tuple((arm.name, arm.tile_m, arm.tile_k, arm.tile_n) for arm in self.arms)
        if observed_arms != expected_arms:
            raise ValueError("GMM tile-search arm inventory mismatch")
        return self

    @computed_field
    @property
    def search_id(self) -> str:
        return model_identity_sha256(self)


def default_gmm_tile_search_contract(
    *,
    accepted_route_report_id: str,
    accepted_route_report_sha256: str,
    accepted_route_corpus_sha256: str,
    inkling_git_commit: str,
    inkling_uv_lock_sha256: str,
    implementation_source_manifest: tuple[SourceFileContract, ...],
    numerical_contract_id: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> InklingGmmTileSearchContract:
    gate_up = {
        "lhs_dtype": "bf16",
        "rhs_dtype": "bf16",
        "accumulator_dtype": "fp32",
        "output_dtype": "fp32",
        "k": 4096,
        "n": 2048,
        "zero_initialize": False,
    }
    return InklingGmmTileSearchContract(
        name="inkling-small-v7x-gmm-tile-search",
        route_corpus=RouteCorpusBinding(
            report_id=accepted_route_report_id,
            report_sha256=accepted_route_report_sha256,
            corpus_sha256=accepted_route_corpus_sha256,
        ),
        inkling_git_commit=inkling_git_commit,
        inkling_uv_lock_sha256=inkling_uv_lock_sha256,
        implementation_source_manifest=implementation_source_manifest,
        target_runtime=GmmTargetRuntime(),
        production_abi=GmmProductionAbi(
            kernels=(
                GmmKernelAbi(operation=GmmOperation.GATE, **gate_up),
                GmmKernelAbi(operation=GmmOperation.UP, **gate_up),
                GmmKernelAbi(
                    operation=GmmOperation.DOWN,
                    lhs_dtype="fp32",
                    rhs_dtype="bf16",
                    accumulator_dtype="fp32",
                    output_dtype="fp32",
                    k=2048,
                    n=4096,
                    zero_initialize=True,
                ),
            )
        ),
        corpus=GmmCorpusProtocol(
            completion_steps=GMM_COMPLETION_STEPS,
            layer_indices=GMM_LAYER_INDICES,
        ),
        arms=(
            GmmTileArm(name=GmmArmName.INCUMBENT, tile_m=128, tile_n="N"),
            GmmTileArm(name=GmmArmName.SPARSE_M64, tile_m=64, tile_n="N"),
            GmmTileArm(name=GmmArmName.SPARSE_M32, tile_m=32, tile_n="N"),
            GmmTileArm(name=GmmArmName.SPLIT_N, tile_m=128, tile_n="N/2"),
            GmmTileArm(name=GmmArmName.SPARSE_M64_SPLIT_N, tile_m=64, tile_n="N/2"),
        ),
        search=GmmSearchProtocol(),
        confirmation=GmmConfirmationProtocol(),
        correctness=GmmCorrectnessProtocol(
            numerical_contract_id=numerical_contract_id,
            seeds=GMM_CORRECTNESS_SEEDS,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
    )


def local_active_span(group_sizes: tuple[int, ...], *, device_index: int) -> tuple[int, int]:
    if len(group_sizes) != 256 or any(value < 0 for value in group_sizes):
        raise ValueError("GMM group sizes must contain 256 nonnegative values")
    if not 0 <= device_index < 8:
        raise ValueError("GMM device index must be in [0, 8)")
    first_group = device_index * 32
    start = sum(group_sizes[:first_group])
    return start, start + sum(group_sizes[first_group : first_group + 32])


def validate_route_corpus_binding(
    contract: InklingGmmTileSearchContract,
    report: InklingGmmRouteCorpusReport,
    *,
    report_bytes: bytes,
) -> None:
    binding = contract.route_corpus
    if (
        report.report_id == "0" * 64
        or model_identity_sha256(report, exclude={"report_id"}) != report.report_id
    ):
        raise ValueError("GMM route report must have a valid final identity")
    if report.report_id != binding.report_id:
        raise ValueError("GMM route report identity mismatch")
    if hashlib.sha256(report_bytes).hexdigest() != binding.report_sha256:
        raise ValueError("GMM route report content hash mismatch")
    try:
        parsed_report = InklingGmmRouteCorpusReport.model_validate_json(report_bytes)
    except ValueError as error:
        raise ValueError("GMM route report bytes do not encode the accepted report") from error
    if parsed_report != report:
        raise ValueError("GMM route report bytes do not encode the accepted report")
    if report.corpus_sha256 != binding.corpus_sha256:
        raise ValueError("GMM route corpus hash mismatch")
    if report.cohort_scope != binding.cohort_scope:
        raise ValueError("GMM route corpus cohort scope mismatch")
    if (
        report.concurrency != 48
        or report.num_experts_per_token != 6
        or report.num_routed_experts != 256
        or len(report.request_state_slots) != 48
        or len(report.recurrent_state_slots) != 48
    ):
        raise ValueError("GMM route corpus production workload mismatch")
    if (
        report.selected_completion_steps != contract.corpus.completion_steps
        or tuple(range(report.first_moe_layer, report.num_layers)) != contract.corpus.layer_indices
    ):
        raise ValueError("GMM route corpus inventory mismatch")
    if any(
        len(group.group_sizes) != contract.production_abi.global_group_count
        or sum(group.group_sizes) != contract.production_abi.m
        for group in report.group_sizes
    ):
        raise ValueError("GMM route corpus ABI mismatch")
