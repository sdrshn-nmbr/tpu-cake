from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

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


class EvidencePhaseName(StrEnum):
    TIMING = "timing"
    TRACE = "trace"
    COUNTERS = "counters"
    FINALIZER = "finalizer"
    AGGREGATE = "aggregate"


class EvidenceProfile(StrEnum):
    DISTRIBUTED_MATMUL = "distributed_matmul_v1"
    OPAQUE_RPA_ADAPTER = "opaque_rpa_adapter_v1"
    SEQAX_DISTRIBUTED_FORWARD = "seqax_distributed_forward_v1"


class ArtifactRole(StrEnum):
    EXPERIMENT = "experiment"
    DISTRIBUTED_IR = "distributed_ir"
    PHYSICAL_IR = "physical_ir"
    PALLAS_SOURCE = "pallas_source"
    JAX_SOURCE = "jax_source"
    PLAN_MANIFEST = "plan_manifest"
    STABLEHLO = "stablehlo"
    COMPILER_HLO = "compiler_hlo"
    CORRECTNESS_INPUT = "correctness_input"
    CORRECTNESS_OUTPUT = "correctness_output"
    ORACLE_OUTPUT = "oracle_output"
    TIMING_SAMPLES = "timing_samples"
    TIMING_TRACE = "timing_trace"
    COUNTER_TRACE = "counter_trace"
    PROFILE_ASSESSMENT = "profile_assessment"
    COST_MODEL_INPUT = "cost_model_input"
    COST_MODEL = "cost_model"
    ROOFLINE_INPUT = "roofline_input"
    ROOFLINE_REPORT = "roofline_report"
    ROOFLINE_METRICS = "roofline_metrics"
    EXECUTION_LEDGER = "execution_ledger"
    TRACE_RESULT = "trace_result"
    COUNTER_RESULT = "counter_result"
    HLO_STATS = "hlo_stats"
    INVOCATION = "invocation"
    PROFILER_CONFIG = "profiler_config"
    SOURCE_STATE = "source_state"
    SOURCE_DIFF = "source_diff"
    BACKEND_MANIFEST = "backend_manifest"
    PREFLIGHT_RESULT = "preflight_result"
    XPROF_EXPORT = "xprof_export"
    SEARCH_CONTRACT = "search_contract"
    SEARCH_RESULT = "search_result"
    SEARCH_EVIDENCE = "search_evidence"


class SemanticPropertyKind(StrEnum):
    PREFIX_INVARIANCE = "prefix_invariance"
    PREFILL_DECODE_EQUIVALENCE = "prefill_decode_equivalence"
    STEPWISE_EQUIVALENCE = "stepwise_equivalence"
    BATCH_PERMUTATION_INVARIANCE = "batch_permutation_invariance"
    STATE_ISOLATION = "state_isolation"
    CACHE_EQUIVALENCE = "cache_equivalence"


class ProfileExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    stage: WorkloadStage
    minimum_tpu_device_planes: int = Field(default=1, ge=1)
    require_tensor_core_activity: bool = True
    require_hbm_read_counters: bool = False
    require_hbm_write_counters: bool = False
    require_cycle_counters: bool = False
    minimum_counter_device_planes: int = Field(default=0, ge=0)
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
    semantic_properties: tuple[SemanticPropertyKind, ...] = ()


class SourceFileContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    executor: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    preflight: str | None = None
    source_revision: str | None = None
    source_manifest: tuple[SourceFileContract, ...] = ()

    @model_validator(mode="after")
    def source_paths_are_unique(self) -> ExecutionContract:
        paths = [source.path for source in self.source_manifest]
        if len(paths) != len(set(paths)):
            raise ValueError("execution source manifest paths must be unique")
        return self


class SemanticPropertyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    property: SemanticPropertyKind
    passed: bool
    maximum_absolute_error: float = Field(ge=0)
    details: str = Field(min_length=1)


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
    execution: ExecutionContract | None = None


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
        payload = self.model_dump(
            mode="json",
            exclude={"experiment_id"},
        )
        if payload["workload"]["execution"] is None:
            del payload["workload"]["execution"]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def experiment_artifact_json(experiment: KernelExperiment, *, indent: int = 2) -> str:
    exclude: dict[str, object] | None = None
    if experiment.workload.execution is None:
        exclude = {"workload": {"execution"}}
    return experiment.model_dump_json(
        indent=indent,
        exclude_computed_fields=True,
        exclude=exclude,
    )


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: ArtifactRole

    @field_validator("path")
    @classmethod
    def path_stays_inside_bundle(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or path.is_absolute()
            or ".." in path.parts
            or value != path.as_posix()
            or path.as_posix() == "."
        ):
            raise ValueError("artifact path must be a canonical bundle-relative POSIX path")
        return value


PHASE_REQUIRED_ROLES: dict[EvidencePhaseName, frozenset[ArtifactRole]] = {
    EvidencePhaseName.TIMING: frozenset(
        {
            ArtifactRole.EXPERIMENT,
            ArtifactRole.DISTRIBUTED_IR,
            ArtifactRole.PHYSICAL_IR,
            ArtifactRole.PALLAS_SOURCE,
            ArtifactRole.STABLEHLO,
            ArtifactRole.COMPILER_HLO,
            ArtifactRole.CORRECTNESS_INPUT,
            ArtifactRole.CORRECTNESS_OUTPUT,
            ArtifactRole.ORACLE_OUTPUT,
            ArtifactRole.TIMING_SAMPLES,
            ArtifactRole.EXECUTION_LEDGER,
            ArtifactRole.COST_MODEL_INPUT,
            ArtifactRole.COST_MODEL,
            ArtifactRole.INVOCATION,
            ArtifactRole.PROFILER_CONFIG,
            ArtifactRole.SOURCE_STATE,
            ArtifactRole.SOURCE_DIFF,
        }
    ),
    EvidencePhaseName.TRACE: frozenset(
        {
            ArtifactRole.TRACE_RESULT,
            ArtifactRole.EXECUTION_LEDGER,
            ArtifactRole.TIMING_TRACE,
            ArtifactRole.HLO_STATS,
            ArtifactRole.INVOCATION,
            ArtifactRole.PROFILER_CONFIG,
            ArtifactRole.SOURCE_STATE,
            ArtifactRole.SOURCE_DIFF,
        }
    ),
    EvidencePhaseName.COUNTERS: frozenset(
        {
            ArtifactRole.COUNTER_RESULT,
            ArtifactRole.EXECUTION_LEDGER,
            ArtifactRole.COUNTER_TRACE,
            ArtifactRole.HLO_STATS,
            ArtifactRole.INVOCATION,
            ArtifactRole.PROFILER_CONFIG,
            ArtifactRole.SOURCE_STATE,
            ArtifactRole.SOURCE_DIFF,
        }
    ),
    EvidencePhaseName.FINALIZER: frozenset({ArtifactRole.SOURCE_STATE, ArtifactRole.SOURCE_DIFF}),
    EvidencePhaseName.AGGREGATE: frozenset(
        {
            ArtifactRole.PROFILE_ASSESSMENT,
            ArtifactRole.ROOFLINE_INPUT,
            ArtifactRole.ROOFLINE_REPORT,
            ArtifactRole.ROOFLINE_METRICS,
        }
    ),
}

RPA_PHASE_REQUIRED_ROLES: dict[EvidencePhaseName, frozenset[ArtifactRole]] = {
    EvidencePhaseName.TIMING: frozenset(
        {
            ArtifactRole.EXPERIMENT,
            ArtifactRole.PHYSICAL_IR,
            ArtifactRole.PALLAS_SOURCE,
            ArtifactRole.STABLEHLO,
            ArtifactRole.COMPILER_HLO,
            ArtifactRole.CORRECTNESS_INPUT,
            ArtifactRole.CORRECTNESS_OUTPUT,
            ArtifactRole.ORACLE_OUTPUT,
            ArtifactRole.TIMING_SAMPLES,
            ArtifactRole.EXECUTION_LEDGER,
            ArtifactRole.INVOCATION,
            ArtifactRole.PROFILER_CONFIG,
            ArtifactRole.SOURCE_STATE,
            ArtifactRole.SOURCE_DIFF,
            ArtifactRole.BACKEND_MANIFEST,
            ArtifactRole.PREFLIGHT_RESULT,
        }
    ),
    EvidencePhaseName.TRACE: frozenset(
        {
            ArtifactRole.TRACE_RESULT,
            ArtifactRole.EXECUTION_LEDGER,
            ArtifactRole.TIMING_TRACE,
            ArtifactRole.HLO_STATS,
            ArtifactRole.INVOCATION,
            ArtifactRole.PROFILER_CONFIG,
            ArtifactRole.SOURCE_STATE,
            ArtifactRole.SOURCE_DIFF,
            ArtifactRole.BACKEND_MANIFEST,
            ArtifactRole.PREFLIGHT_RESULT,
        }
    ),
    EvidencePhaseName.COUNTERS: frozenset(
        {
            ArtifactRole.COUNTER_RESULT,
            ArtifactRole.EXECUTION_LEDGER,
            ArtifactRole.COUNTER_TRACE,
            ArtifactRole.HLO_STATS,
            ArtifactRole.INVOCATION,
            ArtifactRole.PROFILER_CONFIG,
            ArtifactRole.SOURCE_STATE,
            ArtifactRole.SOURCE_DIFF,
            ArtifactRole.BACKEND_MANIFEST,
            ArtifactRole.PREFLIGHT_RESULT,
        }
    ),
    EvidencePhaseName.FINALIZER: frozenset({ArtifactRole.SOURCE_STATE, ArtifactRole.SOURCE_DIFF}),
    EvidencePhaseName.AGGREGATE: frozenset({ArtifactRole.PROFILE_ASSESSMENT}),
}

PHASE_REQUIRED_ROLES_BY_PROFILE = {
    EvidenceProfile.DISTRIBUTED_MATMUL: PHASE_REQUIRED_ROLES,
    EvidenceProfile.OPAQUE_RPA_ADAPTER: RPA_PHASE_REQUIRED_ROLES,
}


class EvidencePhase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: EvidencePhaseName
    artifact_paths: tuple[str, ...]

    @model_validator(mode="after")
    def paths_are_unique(self) -> EvidencePhase:
        if not self.artifact_paths or len(self.artifact_paths) != len(set(self.artifact_paths)):
            raise ValueError("evidence phase artifact paths must be non-empty and unique")
        return self


class CorrectnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passed: bool
    oracle: str = Field(min_length=1)
    maximum_absolute_error: float | None = Field(default=None, ge=0)
    maximum_relative_error: float | None = Field(default=None, ge=0)
    semantic_properties: tuple[SemanticPropertyResult, ...] = ()

    @model_validator(mode="after")
    def passed_result_has_no_failed_or_duplicate_properties(self) -> CorrectnessResult:
        properties = [result.property for result in self.semantic_properties]
        if len(properties) != len(set(properties)):
            raise ValueError("correctness result has duplicate semantic properties")
        if self.passed and any(not result.passed for result in self.semantic_properties):
            raise ValueError("passed correctness cannot contain a failed semantic property")
        return self


class RuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    python: str = Field(min_length=1)
    jax: str | None = None
    jaxlib: str | None = None
    libtpu: str | None = None
    xla: str | None = None


class SearchProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    winner: str = Field(min_length=1)
    tile_m: int = Field(gt=0)
    tile_n: int = Field(gt=0)
    winner_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_count: int = Field(gt=0)


class RpaSearchSelection(StrEnum):
    CHALLENGER_PROMOTED = "challenger_promoted"
    BASELINE_RETAINED_NO_PROMOTABLE_CHALLENGER = (
        "baseline_retained_no_promotable_challenger"
    )
    BASELINE_RETAINED_CONFIRMATION_FAILED = "baseline_retained_confirmation_failed"


class RpaSearchProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection: RpaSearchSelection
    baseline: str = Field(min_length=1)
    selected_candidate: str = Field(min_length=1)
    selected_block_sizes: tuple[int, int, int, int]
    selected_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_run_path: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9._/-]+$")
    selected_run_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_profiler_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_count: int = Field(gt=0)


class RunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_profile: EvidenceProfile = EvidenceProfile.DISTRIBUTED_MATMUL
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RunStatus
    runtime: RuntimeIdentity
    correctness: CorrectnessResult
    required_semantic_properties: tuple[SemanticPropertyKind, ...]
    metrics: tuple[Metric, ...]
    artifacts: tuple[ArtifactReference, ...]
    phases: tuple[EvidencePhase, ...]
    search_provenance: SearchProvenance | None = None
    rpa_search_provenance: RpaSearchProvenance | None = None

    @model_validator(mode="after")
    def pass_requires_correctness(self) -> RunReceipt:
        if self.status is RunStatus.PASSED and not self.correctness.passed:
            raise ValueError("a passed receipt requires passed correctness")
        required = set(self.required_semantic_properties)
        if len(required) != len(self.required_semantic_properties):
            raise ValueError("receipt semantic-property requirements must be unique")
        observed = {result.property for result in self.correctness.semantic_properties}
        if self.status is RunStatus.PASSED and observed != required:
            raise ValueError("a passed receipt must contain every required semantic property")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("receipt artifact paths must be unique")
        if self.search_provenance is not None and self.rpa_search_provenance is not None:
            raise ValueError("receipt cannot contain two search provenance contracts")
        if self.status is RunStatus.PASSED:
            required_roles = set().union(
                *PHASE_REQUIRED_ROLES_BY_PROFILE[self.evidence_profile].values()
            )
            if self.evidence_profile is EvidenceProfile.DISTRIBUTED_MATMUL:
                required_roles.update(
                    {
                        ArtifactRole.EXPERIMENT,
                        ArtifactRole.DISTRIBUTED_IR,
                        ArtifactRole.PHYSICAL_IR,
                        ArtifactRole.PALLAS_SOURCE,
                        ArtifactRole.STABLEHLO,
                        ArtifactRole.COMPILER_HLO,
                        ArtifactRole.CORRECTNESS_INPUT,
                        ArtifactRole.CORRECTNESS_OUTPUT,
                        ArtifactRole.ORACLE_OUTPUT,
                        ArtifactRole.TIMING_SAMPLES,
                        ArtifactRole.TIMING_TRACE,
                        ArtifactRole.COUNTER_TRACE,
                        ArtifactRole.PROFILE_ASSESSMENT,
                        ArtifactRole.COST_MODEL_INPUT,
                        ArtifactRole.COST_MODEL,
                        ArtifactRole.ROOFLINE_INPUT,
                        ArtifactRole.ROOFLINE_REPORT,
                        ArtifactRole.ROOFLINE_METRICS,
                        ArtifactRole.EXECUTION_LEDGER,
                        ArtifactRole.TRACE_RESULT,
                        ArtifactRole.COUNTER_RESULT,
                        ArtifactRole.HLO_STATS,
                        ArtifactRole.INVOCATION,
                        ArtifactRole.PROFILER_CONFIG,
                        ArtifactRole.SOURCE_STATE,
                        ArtifactRole.SOURCE_DIFF,
                    }
                )
            roles = {artifact.role for artifact in self.artifacts}
            missing = sorted(role.value for role in required_roles - roles)
            if missing:
                raise ValueError(f"passed receipt is missing artifact roles: {missing}")
            if (
                self.search_provenance is not None
                or self.rpa_search_provenance is not None
            ):
                search_roles = {
                    ArtifactRole.SEARCH_CONTRACT,
                    ArtifactRole.SEARCH_RESULT,
                    ArtifactRole.SEARCH_EVIDENCE,
                }
                missing_search = sorted(role.value for role in search_roles - roles)
                if missing_search:
                    raise ValueError(
                        f"search provenance is missing artifact roles: {missing_search}"
                    )
            phase_names = tuple(phase.name for phase in self.phases)
            if set(phase_names) != set(EvidencePhaseName) or len(phase_names) != len(
                EvidencePhaseName
            ):
                raise ValueError("passed receipt needs every evidence phase exactly once")
            phase_paths = [path for phase in self.phases for path in phase.artifact_paths]
            if len(phase_paths) != len(set(phase_paths)) or set(phase_paths) != set(paths):
                raise ValueError("passed receipt phases must partition every artifact exactly once")
            artifacts_by_path = {artifact.path: artifact for artifact in self.artifacts}
            for phase in self.phases:
                phase_roles = {artifacts_by_path[path].role for path in phase.artifact_paths}
                missing_phase_roles = (
                    PHASE_REQUIRED_ROLES_BY_PROFILE[self.evidence_profile][phase.name] - phase_roles
                )
                if missing_phase_roles:
                    missing_names = sorted(role.value for role in missing_phase_roles)
                    raise ValueError(
                        f"evidence phase {phase.name.value} is missing roles: {missing_names}"
                    )
        return self
