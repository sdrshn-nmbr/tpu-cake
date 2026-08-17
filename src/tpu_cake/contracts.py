from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.metrics import Metric


class WorkloadStage(StrEnum):
    CONTROL = "control"
    PREFILL = "prefill"
    STEADY_DECODE = "steady_decode"
    MIXED = "mixed"


class RunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"


class ProfileExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    stage: WorkloadStage
    minimum_tpu_device_planes: int = Field(default=1, ge=1)
    require_tensor_core_activity: bool = True
    require_hbm_read_counters: bool = False
    require_hbm_write_counters: bool = False
    require_cycle_counters: bool = False
    required_timed_hlo_markers: tuple[str, ...] = ()
    forbidden_timed_hlo_fragments: tuple[str, ...] = ()


class TargetHardware(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accelerator: str = Field(min_length=1)
    topology: str = Field(min_length=1)
    chip_count: int = Field(ge=1)
    vmem_budget_bytes_per_core: int = Field(gt=0)
    smem_budget_bytes_per_core: int = Field(gt=0)
    peak_hbm_bytes_per_second: int | None = Field(default=None, gt=0)
    peak_flops_per_second: int | None = Field(default=None, gt=0)
    runtime_target: str = Field(min_length=1)


class TensorContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    shape: tuple[int, ...]
    logical_shape: tuple[str, ...]
    dtype: str = Field(min_length=1)
    sharding: tuple[str, ...]

    @model_validator(mode="after")
    def ranks_match(self) -> TensorContract:
        if len(self.shape) != len(self.logical_shape) or len(self.shape) != len(self.sharding):
            raise ValueError(
                "tensor physical shape, logical shape, and sharding must have equal rank"
            )
        if any(size <= 0 for size in self.shape):
            raise ValueError("tensor dimensions must be positive")
        return self


class NumericalContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference: str = Field(min_length=1)
    absolute_tolerance: float = Field(ge=0)
    relative_tolerance: float = Field(ge=0)
    deterministic: bool = True


class BenchmarkProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(ge=1)
    synchronization: str = Field(min_length=1)
    statistic: str = Field(min_length=1)


class SearchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    objective_metric: str = Field(min_length=1)
    require_correctness: bool = True
    require_profile_acceptance: bool = True


class WorkloadContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    stage: WorkloadStage
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    numerical: NumericalContract


class KernelExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    workload: WorkloadContract
    target: TargetHardware
    benchmark: BenchmarkProtocol
    search: SearchPolicy
    profile: ProfileExpectation
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @computed_field
    @property
    def experiment_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"experiment_id"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CorrectnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passed: bool
    oracle: str = Field(min_length=1)
    maximum_absolute_error: float | None = Field(default=None, ge=0)
    maximum_relative_error: float | None = Field(default=None, ge=0)


class RuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    python: str = Field(min_length=1)
    jax: str | None = None
    jaxlib: str | None = None
    libtpu: str | None = None
    xla: str | None = None


class RunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RunStatus
    runtime: RuntimeIdentity
    correctness: CorrectnessResult
    metrics: tuple[Metric, ...]
    artifacts: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def pass_requires_correctness(self) -> RunReceipt:
        if self.status is RunStatus.PASSED and not self.correctness.passed:
            raise ValueError("a passed receipt requires passed correctness")
        return self
