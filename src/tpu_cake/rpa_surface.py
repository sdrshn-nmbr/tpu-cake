from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, semantic_seed
from tpu_cake.rpa_lowering import lower_inkling_sharded_rpa_to_pallas
from tpu_cake.workloads.inkling_rpa import inkling_sharded_fused_rpa_schedule

INKLING_SHARDED_RPA_SURFACE_SCHEMA = "inkling-sharded-rpa-surface-v1"
INKLING_SHARDED_RPA_COMPILATION_ROOT = "/home/sudarshan/tpu-cake-main"
INKLING_SHARDED_RPA_BACKEND_PYTHON_PATH = "/home/sudarshan/inkle/engine/sglang-jax/python"
INKLING_SHARDED_RPA_BACKEND_IMPORT_PACKAGES = (
    "fastapi==0.116.1",
    "orjson==3.11.1",
    "psutil==7.0.0",
    "pyzmq==27.0.1",
)
INKLING_SHARDED_RPA_CORRECTNESS_SEEDS = tuple(
    semantic_seed(INKLING_SHARDED_RPA_SURFACE_SCHEMA, str(index)) for index in range(5)
)
INKLING_SHARDED_RPA_STABLEHLO_SHA256 = (
    "f5da8c8caa28f42ff79c9bb14cb5cd638d01de85eca4156d6116ec97d14f1c7e"
)


class InklingShardedRpaPlanContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_authority: Literal["receipt-bound-raw-bytes-not-reproducible-identity"]
    mesh_axes: tuple[str, str]
    mesh_shape: tuple[int, int]
    local_input_shapes: tuple[tuple[int, ...], ...] = Field(min_length=11, max_length=11)
    global_input_shapes: tuple[tuple[int, ...], ...] = Field(min_length=11, max_length=11)
    global_output_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    input_partition_specs: tuple[tuple[str, ...], ...] = Field(min_length=11, max_length=11)
    output_partition_specs: tuple[tuple[str, ...], tuple[str, ...]]
    input_dtypes: tuple[str, ...] = Field(min_length=11, max_length=11)
    output_dtypes: tuple[str, str]
    decode_block_sizes: tuple[int, int, int, int]
    backend_repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    backend_file_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    backend_manifest: tuple[SourceFileContract, ...] = Field(min_length=4, max_length=4)


def _plan_contract() -> InklingShardedRpaPlanContract:
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    return InklingShardedRpaPlanContract(
        schedule_sha256=plan.schedule_sha256,
        execution_sha256=plan.source_sha256(),
        stablehlo_sha256=INKLING_SHARDED_RPA_STABLEHLO_SHA256,
        compiler_hlo_authority="receipt-bound-raw-bytes-not-reproducible-identity",
        mesh_axes=plan.mesh_axes,
        mesh_shape=plan.mesh_shape,
        local_input_shapes=plan.local_plan.input_shapes,
        global_input_shapes=plan.global_input_shapes,
        global_output_shapes=plan.global_output_shapes,
        input_partition_specs=plan.input_partition_specs,
        output_partition_specs=plan.output_partition_specs,
        input_dtypes=plan.local_plan.input_dtypes,
        output_dtypes=plan.local_plan.output_dtypes,
        decode_block_sizes=plan.local_plan.decode_block_sizes,
        backend_repository_revision=plan.local_plan.backend_repository_revision,
        backend_file_revision=plan.local_plan.backend_file_revision,
        backend_manifest=tuple(
            SourceFileContract(path=path, sha256=sha256) for path, sha256 in plan.backend_manifest
        ),
    )


class InklingShardedRpaSurfaceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    surface_schema: Literal[INKLING_SHARDED_RPA_SURFACE_SCHEMA]
    identity_schema: Literal[SEMANTIC_IDENTITY_SCHEMA]
    compilation_source_root: Literal[INKLING_SHARDED_RPA_COMPILATION_ROOT]
    backend_python_path: Literal[INKLING_SHARDED_RPA_BACKEND_PYTHON_PATH]
    backend_import_packages: tuple[str, ...] = Field(min_length=4, max_length=4)
    hlo_identity_status: Literal["pinned"]
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    timing_seed: int
    repeat_executions: int = Field(ge=2)
    warmup_iterations: int = Field(gt=0)
    timing_rounds: int = Field(ge=12)
    samples_per_round: int = Field(ge=3)
    output_maximum_absolute_error: float = Field(gt=0)
    output_relative_l2_error: float = Field(gt=0)
    require_exact_cache: bool
    runtime: RuntimeIdentity
    backend: Literal["tpu"]
    device_kind: Literal["TPU7x"]
    device_count: Literal[8]
    process_count: Literal[1]
    plan: InklingShardedRpaPlanContract
    claim_scope: Literal[
        "fixed-inkling-hq32-hkv16-d128-contexts128-512-1024-2048-owned-2x4-sharding"
    ]

    @model_validator(mode="after")
    def contract_is_canonical(self) -> InklingShardedRpaSurfaceContract:
        expected_runtime = RuntimeIdentity(
            python="3.12.3",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla=" --xla_tpu_use_enhanced_launch_barrier=true",
        )
        if self.correctness_seeds != INKLING_SHARDED_RPA_CORRECTNESS_SEEDS:
            raise ValueError("sharded RPA correctness seeds are not canonical")
        if self.backend_import_packages != INKLING_SHARDED_RPA_BACKEND_IMPORT_PACKAGES:
            raise ValueError("sharded RPA backend import packages are not canonical")
        if self.timing_seed != self.correctness_seeds[0]:
            raise ValueError("sharded RPA timing seed must be the first checked seed")
        if (
            self.repeat_executions,
            self.warmup_iterations,
            self.timing_rounds,
            self.samples_per_round,
        ) != (2, 5, 16, 5):
            raise ValueError("sharded RPA measurement protocol is not canonical")
        if (
            self.output_maximum_absolute_error,
            self.output_relative_l2_error,
            self.require_exact_cache,
        ) != (0.001, 0.006, True):
            raise ValueError("sharded RPA numerical policy is not canonical")
        if self.runtime != expected_runtime:
            raise ValueError("sharded RPA runtime is not canonical")
        if self.plan != _plan_contract():
            raise ValueError("sharded RPA plan is not canonical")
        return self

    @computed_field
    @property
    def surface_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def default_inkling_sharded_rpa_surface_contract() -> InklingShardedRpaSurfaceContract:
    return InklingShardedRpaSurfaceContract(
        surface_schema=INKLING_SHARDED_RPA_SURFACE_SCHEMA,
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        compilation_source_root=INKLING_SHARDED_RPA_COMPILATION_ROOT,
        backend_python_path=INKLING_SHARDED_RPA_BACKEND_PYTHON_PATH,
        backend_import_packages=INKLING_SHARDED_RPA_BACKEND_IMPORT_PACKAGES,
        hlo_identity_status="pinned",
        correctness_seeds=INKLING_SHARDED_RPA_CORRECTNESS_SEEDS,
        timing_seed=INKLING_SHARDED_RPA_CORRECTNESS_SEEDS[0],
        repeat_executions=2,
        warmup_iterations=5,
        timing_rounds=16,
        samples_per_round=5,
        output_maximum_absolute_error=0.001,
        output_relative_l2_error=0.006,
        require_exact_cache=True,
        runtime=RuntimeIdentity(
            python="3.12.3",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla=" --xla_tpu_use_enhanced_launch_barrier=true",
        ),
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        process_count=1,
        plan=_plan_contract(),
        claim_scope=("fixed-inkling-hq32-hkv16-d128-contexts128-512-1024-2048-owned-2x4-sharding"),
    )


class InklingShardedRpaDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0)
    process_index: Literal[0]
    platform: Literal["tpu"]
    device_kind: Literal["TPU7x"]


class InklingShardedRpaCorrectnessObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: int
    input_sha256: tuple[str, ...] = Field(min_length=11, max_length=11)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeat_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeat_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeated_output_exact: bool
    repeated_cache_exact: bool
    maximum_absolute_error: float = Field(ge=0)
    relative_l2_error: float = Field(ge=0)
    passed: bool

    @model_validator(mode="after")
    def metrics_are_finite(self) -> InklingShardedRpaCorrectnessObservation:
        if not math.isfinite(self.maximum_absolute_error) or not math.isfinite(
            self.relative_l2_error
        ):
            raise ValueError("sharded RPA correctness metrics must be finite")
        return self


class InklingShardedRpaTimingRound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    round_index: int = Field(ge=0)
    samples_ns: tuple[int, ...] = Field(min_length=3)
    median_ns: float = Field(gt=0)

    @model_validator(mode="after")
    def samples_are_valid(self) -> InklingShardedRpaTimingRound:
        if any(value <= 0 for value in self.samples_ns):
            raise ValueError("sharded RPA timing samples must be positive")
        if self.median_ns != float(statistics.median(self.samples_ns)):
            raise ValueError("sharded RPA timing median mismatch")
        return self


class InklingShardedRpaSurfaceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    runtime: RuntimeIdentity
    devices: tuple[InklingShardedRpaDevice, ...] = Field(min_length=8, max_length=8)
    plan: InklingShardedRpaPlanContract
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness: tuple[InklingShardedRpaCorrectnessObservation, ...] = Field(
        min_length=5,
        max_length=5,
    )
    timing_input_sha256: tuple[str, ...] = Field(min_length=11, max_length=11)
    pre_timing_output_sha256: tuple[str, str]
    rounds: tuple[InklingShardedRpaTimingRound, ...] = Field(min_length=12)
    post_timing_output_sha256: tuple[str, str]
    median_round_duration_ns: float = Field(gt=0)
    p90_round_duration_ns: float = Field(gt=0)
    coefficient_of_variation: float = Field(ge=0)
    accepted: Literal[True]
    claim_scope: Literal[
        "fixed-inkling-hq32-hkv16-d128-contexts128-512-1024-2048-owned-2x4-sharding"
    ]

    @model_validator(mode="after")
    def result_is_internally_consistent(self) -> InklingShardedRpaSurfaceResult:
        seeds = tuple(value.seed for value in self.correctness)
        if len(seeds) != len(set(seeds)):
            raise ValueError("sharded RPA correctness seeds must be unique")
        if any(
            not value.passed
            or not value.repeated_output_exact
            or not value.repeated_cache_exact
            or value.output_sha256 != value.repeat_output_sha256
            or value.cache_sha256 != value.repeat_cache_sha256
            or value.cache_sha256 != value.oracle_cache_sha256
            for value in self.correctness
        ):
            raise ValueError("sharded RPA correctness evidence must pass exactly")
        if tuple(value.round_index for value in self.rounds) != tuple(range(len(self.rounds))):
            raise ValueError("sharded RPA timing rounds must be ordered")
        medians = tuple(value.median_ns for value in self.rounds)
        ordered = sorted(medians)
        p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
        coefficient = statistics.pstdev(medians) / statistics.mean(medians)
        if (
            self.median_round_duration_ns != float(statistics.median(medians))
            or self.p90_round_duration_ns != float(p90)
            or not math.isclose(
                self.coefficient_of_variation,
                coefficient,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("sharded RPA aggregate timing statistics mismatch")
        return self


class InklingShardedRpaSurfaceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    receipt_schema: Literal["inkling-sharded-rpa-surface-receipt-v1"]
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_count: int = Field(gt=0)
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    accepted: Literal[True]
    claim_scope: Literal[
        "fixed-inkling-hq32-hkv16-d128-contexts128-512-1024-2048-owned-2x4-sharding"
    ]

    @model_validator(mode="after")
    def artifact_inventory_is_exact(self) -> InklingShardedRpaSurfaceReceipt:
        paths = tuple(value.path for value in self.artifacts)
        if self.artifact_count != len(self.artifacts) or len(paths) != len(set(paths)):
            raise ValueError("sharded RPA receipt artifact inventory mismatch")
        if tuple(sorted(paths)) != paths:
            raise ValueError("sharded RPA receipt artifacts must be ordered")
        return self
