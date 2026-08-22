from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA
from tpu_cake.runner import RunMode
from tpu_cake.seqax_numerical import (
    SeqaxBf16OutputAssessment,
    default_seqax_bf16_validation_contract,
)
from tpu_cake.seqax_pallas_search import SEQAX_PALLAS_CORRECTNESS_SEEDS, SeqaxPallasDevice
from tpu_cake.seqax_runner import expected_seqax_profiler_contract
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy

SEQAX_RESIDUAL_PROFILE_SCHEMA = "seqax-residual-all-reduce-profile-v1"
SEQAX_RESIDUAL_PROFILE_WARMUPS = 5
SEQAX_RESIDUAL_PROFILE_ITERATIONS = 50
SEQAX_RESIDUAL_PROFILE_TIMING_SEED = 16669898698845158318
SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT = "/home/sudarshan/tpu-cake-main"


class SeqaxResidualProfileCandidateContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_pallas_regions: int = Field(gt=0)
    expected_all_gathers: int = Field(ge=0)
    expected_all_reduces: int = Field(ge=0)
    expected_reduce_scatters: int = Field(ge=0)
    expected_semantic_all_gather_rows: int = Field(ge=0)
    expected_semantic_all_reduce_rows: int = Field(ge=0)
    expected_semantic_reduce_scatter_rows: int = Field(ge=0)
    expected_async_collective_completion_rows: int = Field(ge=0)
    expected_ring_equivalent_ici_bytes_per_device: int = Field(ge=0)
    trace_step_event: str = Field(min_length=1)
    counter_step_event: str = Field(min_length=1)


class SeqaxResidualProfileContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    profile_schema: str = SEQAX_RESIDUAL_PROFILE_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    numerical_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compilation_source_root: str
    hlo_identity_status: str = Field(pattern=r"^(pending|pinned)$")
    timing_seed: int
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    warmup_iterations: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    parameters: dict[str, int | str]
    trace_profiler_config: dict[str, object]
    counter_profiler_config: dict[str, object]
    candidates: tuple[SeqaxResidualProfileCandidateContract, ...] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxResidualProfileContract:
        if self.profile_schema != SEQAX_RESIDUAL_PROFILE_SCHEMA:
            raise ValueError("Seqax residual profile schema mismatch")
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("Seqax residual profile identity schema mismatch")
        if self.numerical_contract_id != default_seqax_bf16_validation_contract().contract_id:
            raise ValueError("Seqax residual profile numerical contract mismatch")
        if self.compilation_source_root != SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT:
            raise ValueError("Seqax residual profile compilation root mismatch")
        if self.timing_seed != SEQAX_RESIDUAL_PROFILE_TIMING_SEED:
            raise ValueError("Seqax residual profile timing seed mismatch")
        if self.correctness_seeds != SEQAX_PALLAS_CORRECTNESS_SEEDS:
            raise ValueError("Seqax residual profile correctness seeds mismatch")
        if (self.warmup_iterations, self.measured_iterations) != (
            SEQAX_RESIDUAL_PROFILE_WARMUPS,
            SEQAX_RESIDUAL_PROFILE_ITERATIONS,
        ):
            raise ValueError("Seqax residual profile iteration protocol mismatch")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax residual profile requires eight TPU7x devices")
        if self.parameters != _parameters():
            raise ValueError("Seqax residual profile parameters mismatch")
        if self.trace_profiler_config != expected_seqax_profiler_contract(RunMode.TRACE):
            raise ValueError("Seqax residual trace profiler contract mismatch")
        if self.counter_profiler_config != expected_seqax_profiler_contract(RunMode.COUNTERS):
            raise ValueError("Seqax residual counter profiler contract mismatch")
        if self.candidates != _candidate_contracts():
            raise ValueError("Seqax residual profile candidates mismatch")
        hashes = tuple(
            value
            for candidate in self.candidates
            for value in (
                candidate.pallas_stablehlo_sha256,
                candidate.pallas_compiler_hlo_sha256,
                candidate.control_stablehlo_sha256,
                candidate.control_compiler_hlo_sha256,
            )
        )
        zero = "0" * 64
        if self.hlo_identity_status == "pending" and any(value != zero for value in hashes):
            raise ValueError("Pending Seqax residual HLO identities must be zero")
        if self.hlo_identity_status == "pinned" and (
            any(value == zero for value in hashes) or len(set(hashes)) != len(hashes)
        ):
            raise ValueError("Pinned Seqax residual HLO identities are incomplete")
        return self

    @computed_field
    @property
    def profile_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SeqaxResidualCorrectnessObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    seed: int
    input_sha256: tuple[str, ...] = Field(min_length=13, max_length=13)
    cpu_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment: SeqaxBf16OutputAssessment

    @model_validator(mode="after")
    def numerical_policy_passed(self) -> SeqaxResidualCorrectnessObservation:
        if not self.assessment.final_outputs_satisfy_policy:
            raise ValueError("Seqax residual profile correctness policy failed")
        return self


class SeqaxResidualProfileSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    mode: RunMode
    module_execution_count: int = Field(gt=0)
    module_median_duration_ns: float = Field(gt=0)
    module_p90_duration_ns: float = Field(gt=0)
    pallas_average_self_time_sum_ns_per_device: float = Field(gt=0)
    collective_completion_average_self_time_sum_ns_per_device: float = Field(ge=0)
    all_reduce_average_self_time_sum_ns_per_device: float = Field(ge=0)
    semantic_all_gather_rows: int = Field(ge=0)
    semantic_all_reduce_rows: int = Field(ge=0)
    semantic_reduce_scatter_rows: int = Field(ge=0)
    async_collective_completion_rows: int = Field(ge=0)
    static_all_gathers: int = Field(ge=0)
    static_all_reduces: int = Field(ge=0)
    static_reduce_scatters: int = Field(ge=0)
    pallas_regions: int = Field(gt=0)
    ring_equivalent_ici_bytes_per_device: int = Field(ge=0)

    @model_validator(mode="after")
    def values_are_valid(self) -> SeqaxResidualProfileSummary:
        values = (
            self.module_median_duration_ns,
            self.module_p90_duration_ns,
            self.pallas_average_self_time_sum_ns_per_device,
            self.collective_completion_average_self_time_sum_ns_per_device,
            self.all_reduce_average_self_time_sum_ns_per_device,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Seqax residual profile summary is nonfinite")
        if self.module_p90_duration_ns < self.module_median_duration_ns:
            raise ValueError("Seqax residual profile p90 is below median")
        if self.async_collective_completion_rows != (
            self.semantic_all_gather_rows + self.semantic_reduce_scatter_rows
        ):
            raise ValueError("Seqax residual profile observed collective inventory mismatch")
        return self


class SeqaxResidualProfileCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    mode: RunMode
    step_event: str = Field(min_length=1)
    profiler_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    xplane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_id: str = Field(min_length=1)
    summary: SeqaxResidualProfileSummary
    periodic_counter_names: tuple[str, ...]
    periodic_counter_samples_per_core: dict[str, int]
    hbm_read_counter_names: int = Field(ge=0)
    hbm_write_counter_names: int = Field(ge=0)
    cycle_counter_names: int = Field(ge=0)

    @model_validator(mode="after")
    def capture_is_complete(self) -> SeqaxResidualProfileCapture:
        if self.summary.candidate is not self.candidate or self.summary.mode is not self.mode:
            raise ValueError("Seqax residual profile capture identity mismatch")
        if self.mode is RunMode.TRACE:
            if (
                self.periodic_counter_names
                or self.periodic_counter_samples_per_core
                or self.hbm_read_counter_names
                or self.hbm_write_counter_names
                or self.cycle_counter_names
            ):
                raise ValueError("Seqax residual trace carries counter claims")
        elif (
            not self.periodic_counter_names
            or set(self.periodic_counter_samples_per_core) != {"0", "2", "4", "6"}
            or any(value < 2 for value in self.periodic_counter_samples_per_core.values())
            or not any(name.startswith("COUNT_MXU_BUSY") for name in self.periodic_counter_names)
            or self.hbm_read_counter_names <= 0
            or self.hbm_write_counter_names <= 0
            or self.cycle_counter_names <= 0
        ):
            raise ValueError("Seqax residual counter capture is incomplete")
        return self


class SeqaxResidualCandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_compiler_analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_compiler_analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timing_input_sha256: tuple[str, ...] = Field(min_length=13, max_length=13)
    timing_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness: tuple[SeqaxResidualCorrectnessObservation, ...] = Field(
        min_length=5,
        max_length=5,
    )
    trace: SeqaxResidualProfileCapture
    counters: SeqaxResidualProfileCapture

    @model_validator(mode="after")
    def identity_is_consistent(self) -> SeqaxResidualCandidateResult:
        if (
            any(value.candidate is not self.candidate for value in self.correctness)
            or self.trace.candidate is not self.candidate
            or self.counters.candidate is not self.candidate
            or self.trace.mode is not RunMode.TRACE
            or self.counters.mode is not RunMode.COUNTERS
        ):
            raise ValueError("Seqax residual candidate result identity mismatch")
        if tuple(value.seed for value in self.correctness) != SEQAX_PALLAS_CORRECTNESS_SEEDS:
            raise ValueError("Seqax residual candidate correctness seed mismatch")
        static_fields = (
            "static_all_gathers",
            "static_all_reduces",
            "static_reduce_scatters",
            "pallas_regions",
            "ring_equivalent_ici_bytes_per_device",
            "semantic_all_gather_rows",
            "semantic_all_reduce_rows",
            "semantic_reduce_scatter_rows",
            "async_collective_completion_rows",
        )
        if any(
            getattr(self.trace.summary, field) != getattr(self.counters.summary, field)
            for field in static_fields
        ):
            raise ValueError("Seqax residual trace and counters disagree")
        return self


class SeqaxResidualProfileComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    baseline: SeqaxResidualNormStrategy
    candidate: SeqaxResidualNormStrategy
    trace_module_median_change: float
    counter_module_median_change: float
    trace_collective_completion_change: float
    counter_collective_completion_change: float
    trace_all_reduce_self_time_change: float
    counter_all_reduce_self_time_change: float
    trace_pallas_self_time_change: float
    counter_pallas_self_time_change: float
    static_all_gathers_eliminated: int
    static_all_reduces_added: int
    static_reduce_scatters_eliminated: int
    ring_equivalent_ici_bytes_added_per_device: int
    trace_semantic_collective_rows_eliminated: int
    counter_semantic_collective_rows_eliminated: int
    interpretation: tuple[str, ...]

    @model_validator(mode="after")
    def values_are_finite(self) -> SeqaxResidualProfileComparison:
        values = (
            self.trace_module_median_change,
            self.counter_module_median_change,
            self.trace_collective_completion_change,
            self.counter_collective_completion_change,
            self.trace_all_reduce_self_time_change,
            self.counter_all_reduce_self_time_change,
            self.trace_pallas_self_time_change,
            self.counter_pallas_self_time_change,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Seqax residual profile comparison is nonfinite")
        return self


class SeqaxResidualProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeIdentity
    devices: tuple[SeqaxPallasDevice, ...] = Field(min_length=8, max_length=8)
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...]
    compiler_strategy_surface_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[SeqaxResidualCandidateResult, ...] = Field(min_length=2, max_length=2)
    comparison: SeqaxResidualProfileComparison
    correctness_scope: str = "independent-bf16-numerical-diagnostic-on-fixed-surface"
    accepted: bool = False

    @model_validator(mode="after")
    def result_is_diagnostic(self) -> SeqaxResidualProfileResult:
        if (
            tuple(value.candidate for value in self.candidates)
            != (
                SeqaxResidualNormStrategy.STANDARD,
                SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
            )
            or self.comparison.baseline is not SeqaxResidualNormStrategy.STANDARD
            or self.comparison.candidate is not SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE
            or self.correctness_scope != "independent-bf16-numerical-diagnostic-on-fixed-surface"
            or self.accepted
        ):
            raise ValueError("Seqax residual profile result scope mismatch")
        return self


class SeqaxResidualProfileReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: str = Field(pattern=r"^passed$")
    profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)


def _parameters() -> dict[str, int | str]:
    return {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 16,
        "head": 4,
        "key_value_heads": 4,
        "layers": 1,
        "model": 256,
        "numerical_semantics": "typed_bf16_hidden_v2",
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 1,
        "tensor_mesh": 4,
        "vocabulary": 16,
    }


def _candidate_contracts() -> tuple[SeqaxResidualProfileCandidateContract, ...]:
    return (
        SeqaxResidualProfileCandidateContract(
            candidate=SeqaxResidualNormStrategy.STANDARD,
            distributed_schedule_sha256=(
                "c96e1c08df86f0722c3464f470790f9a1bee1f47dfd78e2e8cc262cc20bb4e39"
            ),
            physical_schedule_sha256=(
                "a9910167578c14c0d0a0a944a64477dfa471282024134d961e3994eb24e39cc5"
            ),
            pallas_source_sha256=(
                "7226cc31e4e279ec3ef68c9c25a262978774cbddd6b0891c84ed890f29ef024f"
            ),
            pallas_manifest_sha256=(
                "8de9903ccd4e74e4ded51589cbde9ec869f5a85d34de4ac3a1acbaadff772665"
            ),
            pallas_stablehlo_sha256=(
                "11e044eb217cf74dd9ba968290b0326511b2e454b222daf7688c33404462a3d5"
            ),
            pallas_compiler_hlo_sha256=(
                "9a4f95b321c0d342eb850c08316f1cc7869eb064c21d362881dd1b7e4e0cfd46"
            ),
            control_stablehlo_sha256=(
                "05f377de78b292c90d020b8d865285c807ed2e5d3814c0b4977da09c629cffa6"
            ),
            control_compiler_hlo_sha256=(
                "1d1b379f1b31534cf1a4de4d8616358d40bf10089e4d29aee2a26aedfaffed6e"
            ),
            expected_pallas_regions=9,
            expected_all_gathers=17,
            expected_all_reduces=0,
            expected_reduce_scatters=3,
            expected_semantic_all_gather_rows=8,
            expected_semantic_all_reduce_rows=5,
            expected_semantic_reduce_scatter_rows=3,
            expected_async_collective_completion_rows=11,
            expected_ring_equivalent_ici_bytes_per_device=34_048,
            trace_step_event="seqax_residual_profile_standard_trace",
            counter_step_event="seqax_residual_profile_standard_counters",
        ),
        SeqaxResidualProfileCandidateContract(
            candidate=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
            distributed_schedule_sha256=(
                "e1f3f82306357b67d72f9739a36b0feb68c1a31a3a09b32ae5f6af2625a1c83e"
            ),
            physical_schedule_sha256=(
                "e9e2c0006672bab2a5981b109dc5eff67c30cb53664ed521175752f3ca748701"
            ),
            pallas_source_sha256=(
                "01918b28a8df579c63b3590da07d993302ce9336fadb3574ab31310ee957e7ea"
            ),
            pallas_manifest_sha256=(
                "cb1e97845fc0fe78ed5d74a7e956cabed246fd14262261c0a21329ba4082f27f"
            ),
            pallas_stablehlo_sha256=(
                "015f34a68e49c654f714cbde064d0e7f66082e20d15b702ba4aa5502b48757c8"
            ),
            pallas_compiler_hlo_sha256=(
                "85b3cc1138d01cac9a528a056fd759a377b5185d845f53a0210b87520fbab288"
            ),
            control_stablehlo_sha256=(
                "e3a7902d71030a1cb180d4d44b77f290d0f3e0933f09df9d84d301cbeba416b2"
            ),
            control_compiler_hlo_sha256=(
                "f36bd56bef8b331e1bac2f183d345c5b06488527d6ebe19c6e5f335bce8518ad"
            ),
            expected_pallas_regions=9,
            expected_all_gathers=15,
            expected_all_reduces=2,
            expected_reduce_scatters=1,
            expected_semantic_all_gather_rows=8,
            expected_semantic_all_reduce_rows=5,
            expected_semantic_reduce_scatter_rows=1,
            expected_async_collective_completion_rows=9,
            expected_ring_equivalent_ici_bytes_per_device=35_584,
            trace_step_event="seqax_residual_profile_residual_all_reduce_trace",
            counter_step_event="seqax_residual_profile_residual_all_reduce_counters",
        ),
    )


def default_seqax_residual_profile_contract(
    runtime: RuntimeIdentity,
) -> SeqaxResidualProfileContract:
    return SeqaxResidualProfileContract(
        numerical_contract_id=default_seqax_bf16_validation_contract().contract_id,
        compilation_source_root=SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT,
        hlo_identity_status="pinned",
        timing_seed=SEQAX_RESIDUAL_PROFILE_TIMING_SEED,
        correctness_seeds=SEQAX_PALLAS_CORRECTNESS_SEEDS,
        warmup_iterations=SEQAX_RESIDUAL_PROFILE_WARMUPS,
        measured_iterations=SEQAX_RESIDUAL_PROFILE_ITERATIONS,
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        parameters=_parameters(),
        trace_profiler_config=expected_seqax_profiler_contract(RunMode.TRACE),
        counter_profiler_config=expected_seqax_profiler_contract(RunMode.COUNTERS),
        candidates=_candidate_contracts(),
    )


def compare_residual_profiles(
    baseline: SeqaxResidualCandidateResult,
    candidate: SeqaxResidualCandidateResult,
) -> SeqaxResidualProfileComparison:
    def change(new: float, old: float) -> float:
        return new / old - 1

    def rows(summary: SeqaxResidualProfileSummary) -> int:
        return (
            summary.semantic_all_gather_rows
            + summary.semantic_all_reduce_rows
            + summary.semantic_reduce_scatter_rows
        )

    return SeqaxResidualProfileComparison(
        baseline=baseline.candidate,
        candidate=candidate.candidate,
        trace_module_median_change=change(
            candidate.trace.summary.module_median_duration_ns,
            baseline.trace.summary.module_median_duration_ns,
        ),
        counter_module_median_change=change(
            candidate.counters.summary.module_median_duration_ns,
            baseline.counters.summary.module_median_duration_ns,
        ),
        trace_collective_completion_change=change(
            candidate.trace.summary.collective_completion_average_self_time_sum_ns_per_device,
            baseline.trace.summary.collective_completion_average_self_time_sum_ns_per_device,
        ),
        counter_collective_completion_change=change(
            candidate.counters.summary.collective_completion_average_self_time_sum_ns_per_device,
            baseline.counters.summary.collective_completion_average_self_time_sum_ns_per_device,
        ),
        trace_pallas_self_time_change=change(
            candidate.trace.summary.pallas_average_self_time_sum_ns_per_device,
            baseline.trace.summary.pallas_average_self_time_sum_ns_per_device,
        ),
        counter_pallas_self_time_change=change(
            candidate.counters.summary.pallas_average_self_time_sum_ns_per_device,
            baseline.counters.summary.pallas_average_self_time_sum_ns_per_device,
        ),
        trace_all_reduce_self_time_change=change(
            candidate.trace.summary.all_reduce_average_self_time_sum_ns_per_device,
            baseline.trace.summary.all_reduce_average_self_time_sum_ns_per_device,
        ),
        counter_all_reduce_self_time_change=change(
            candidate.counters.summary.all_reduce_average_self_time_sum_ns_per_device,
            baseline.counters.summary.all_reduce_average_self_time_sum_ns_per_device,
        ),
        static_all_gathers_eliminated=(
            baseline.trace.summary.static_all_gathers - candidate.trace.summary.static_all_gathers
        ),
        static_all_reduces_added=(
            candidate.trace.summary.static_all_reduces - baseline.trace.summary.static_all_reduces
        ),
        static_reduce_scatters_eliminated=(
            baseline.trace.summary.static_reduce_scatters
            - candidate.trace.summary.static_reduce_scatters
        ),
        ring_equivalent_ici_bytes_added_per_device=(
            candidate.trace.summary.ring_equivalent_ici_bytes_per_device
            - baseline.trace.summary.ring_equivalent_ici_bytes_per_device
        ),
        trace_semantic_collective_rows_eliminated=(
            rows(baseline.trace.summary) - rows(candidate.trace.summary)
        ),
        counter_semantic_collective_rows_eliminated=(
            rows(baseline.counters.summary) - rows(candidate.counters.summary)
        ),
        interpretation=(
            "Module medians compare separate profiler-instrumented captures and are descriptive, not promotion timings.",
            "Asynchronous completion time covers all-gather and reduce-scatter rows; all-reduce self time is reported separately.",
            "Collective rows are an observed XProf inventory and are not an additive critical path.",
            "Static collective counts come from the authenticated compiled program; observed rows may be aggregated by XProf.",
            "The candidate moves more ring-equivalent bytes and can win only by reducing latency-bearing collective boundaries.",
            "Pallas self time covers contractions only; residual injection, all-reduce, casts, and shard extraction remain JAX/XLA owned.",
        ),
    )
