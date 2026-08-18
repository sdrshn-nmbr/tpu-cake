from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA
from tpu_cake.runner import RunMode
from tpu_cake.seqax_runner import expected_seqax_profiler_contract
from tpu_cake.seqax_weight_placement import SeqaxWeightPlacementName

SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA = "seqax-weight-data-placement-diagnostic-v1"
SEQAX_WEIGHT_PLACEMENT_SEARCH_ID = (
    "81085697c16f46649c708ef858d1d3339db71a4986bd58a550e5e536b55652f5"
)
SEQAX_WEIGHT_PLACEMENT_SEARCH_RECEIPT_SHA256 = (
    "41c86a2267a435070a087e6de89ae92925f7de318314f410c394c0212f75a347"
)
SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_WARMUPS = 5
SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS = 50
SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_STEP_PREFIX = "seqax_weight_placement_diagnostic"


class SeqaxWeightPlacementDiagnosticCandidateContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: SeqaxWeightPlacementName
    expected_high_level_all_gathers: int = Field(ge=0)
    expected_physical_collectives: int = Field(ge=0)
    expected_stablehlo_all_gathers: int = Field(ge=0)
    expected_pallas_regions: int = Field(gt=0)
    expected_parameter_bytes_per_device: int = Field(gt=0)
    expected_ring_equivalent_ici_bytes_per_device: int = Field(ge=0)
    trace_step_event: str = Field(min_length=1)
    counter_step_event: str = Field(min_length=1)


class SeqaxWeightPlacementDiagnosticContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_schema: str = SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: SeqaxWeightPlacementName
    timing_seed: int
    warmup_iterations: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    trace_profiler_config: dict[str, object]
    counter_profiler_config: dict[str, object]
    candidates: tuple[SeqaxWeightPlacementDiagnosticCandidateContract, ...] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxWeightPlacementDiagnosticContract:
        if self.diagnostic_schema != SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA:
            raise ValueError("Seqax weight-placement diagnostic schema mismatch")
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("Seqax weight-placement diagnostic identity schema mismatch")
        if (
            self.search_id != SEQAX_WEIGHT_PLACEMENT_SEARCH_ID
            or self.search_receipt_sha256 != SEQAX_WEIGHT_PLACEMENT_SEARCH_RECEIPT_SHA256
        ):
            raise ValueError("Seqax weight-placement diagnostic search authority mismatch")
        if (
            self.warmup_iterations,
            self.measured_iterations,
        ) != (
            SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_WARMUPS,
            SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS,
        ):
            raise ValueError("Seqax weight-placement diagnostic protocol mismatch")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax weight-placement diagnostic requires TPU7x")
        if self.baseline is not SeqaxWeightPlacementName.SHARDED:
            raise ValueError("Seqax weight-placement diagnostic baseline mismatch")
        if self.trace_profiler_config != expected_seqax_profiler_contract(RunMode.TRACE):
            raise ValueError("Seqax weight-placement trace profiler contract mismatch")
        if self.counter_profiler_config != expected_seqax_profiler_contract(RunMode.COUNTERS):
            raise ValueError("Seqax weight-placement counter profiler contract mismatch")
        expected = _candidate_contracts()
        if self.candidates != expected:
            raise ValueError("Seqax weight-placement diagnostic candidates mismatch")
        return self

    @computed_field
    @property
    def diagnostic_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SeqaxWeightPlacementProfileSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: SeqaxWeightPlacementName
    mode: RunMode
    module_execution_count: int = Field(gt=0)
    module_median_duration_ns: float = Field(gt=0)
    module_p90_duration_ns: float = Field(gt=0)
    pallas_average_self_time_sum_ns_per_device: float = Field(gt=0)
    collective_completion_average_self_time_sum_ns_per_device: float = Field(gt=0)
    semantic_all_gather_rows: int = Field(ge=0)
    semantic_reduce_scatter_rows: int = Field(ge=0)
    async_collective_completion_rows: int = Field(ge=0)
    high_level_all_gathers: int = Field(ge=0)
    physical_collectives: int = Field(ge=0)
    stablehlo_all_gathers: int = Field(ge=0)
    pallas_regions: int = Field(gt=0)
    parameter_bytes_per_device: int = Field(gt=0)
    ring_equivalent_ici_bytes_per_device: int = Field(ge=0)

    @model_validator(mode="after")
    def values_are_finite(self) -> SeqaxWeightPlacementProfileSummary:
        values = (
            self.module_median_duration_ns,
            self.module_p90_duration_ns,
            self.pallas_average_self_time_sum_ns_per_device,
            self.collective_completion_average_self_time_sum_ns_per_device,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Seqax weight-placement diagnostic summary is nonfinite")
        if self.module_p90_duration_ns < self.module_median_duration_ns:
            raise ValueError("Seqax weight-placement diagnostic p90 is below median")
        if self.async_collective_completion_rows != (
            self.semantic_all_gather_rows + self.semantic_reduce_scatter_rows
        ):
            raise ValueError("Seqax weight-placement collective inventory mismatch")
        return self


class SeqaxWeightPlacementCandidateProfiles(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: SeqaxWeightPlacementName
    trace: SeqaxWeightPlacementProfileSummary
    counters: SeqaxWeightPlacementProfileSummary

    @model_validator(mode="after")
    def modes_and_identity_match(self) -> SeqaxWeightPlacementCandidateProfiles:
        if (
            self.trace.candidate is not self.candidate
            or self.counters.candidate is not self.candidate
            or self.trace.mode is not RunMode.TRACE
            or self.counters.mode is not RunMode.COUNTERS
        ):
            raise ValueError("Seqax weight-placement diagnostic capture identity mismatch")
        static_fields = (
            "high_level_all_gathers",
            "physical_collectives",
            "stablehlo_all_gathers",
            "pallas_regions",
            "parameter_bytes_per_device",
            "ring_equivalent_ici_bytes_per_device",
            "semantic_all_gather_rows",
            "semantic_reduce_scatter_rows",
            "async_collective_completion_rows",
        )
        if any(
            getattr(self.trace, field) != getattr(self.counters, field) for field in static_fields
        ):
            raise ValueError("Seqax weight-placement diagnostic captures disagree")
        return self


class SeqaxWeightPlacementDiagnosticComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: SeqaxWeightPlacementName
    candidate: SeqaxWeightPlacementName
    trace_module_median_change: float
    counter_module_median_change: float
    trace_collective_completion_change: float
    counter_collective_completion_change: float
    trace_pallas_self_time_change: float
    counter_pallas_self_time_change: float
    high_level_all_gathers_eliminated: int = Field(ge=0)
    physical_collectives_eliminated: int = Field(ge=0)
    stablehlo_all_gathers_eliminated: int = Field(ge=0)
    ring_equivalent_ici_bytes_eliminated_per_device: int = Field(ge=0)
    parameter_bytes_added_per_device: int = Field(ge=0)
    interpretation: tuple[str, ...]

    @model_validator(mode="after")
    def values_are_finite(self) -> SeqaxWeightPlacementDiagnosticComparison:
        values = (
            self.trace_module_median_change,
            self.counter_module_median_change,
            self.trace_collective_completion_change,
            self.counter_collective_completion_change,
            self.trace_pallas_self_time_change,
            self.counter_pallas_self_time_change,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Seqax weight-placement diagnostic comparison is nonfinite")
        return self


def _candidate_contracts() -> tuple[SeqaxWeightPlacementDiagnosticCandidateContract, ...]:
    return (
        SeqaxWeightPlacementDiagnosticCandidateContract(
            candidate=SeqaxWeightPlacementName.SHARDED,
            expected_high_level_all_gathers=14,
            expected_physical_collectives=20,
            expected_stablehlo_all_gathers=17,
            expected_pallas_regions=9,
            expected_parameter_bytes_per_device=22_912,
            expected_ring_equivalent_ici_bytes_per_device=34_048,
            trace_step_event=(f"{SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_STEP_PREFIX}.sharded.trace"),
            counter_step_event=(
                f"{SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_STEP_PREFIX}.sharded.counters"
            ),
        ),
        SeqaxWeightPlacementDiagnosticCandidateContract(
            candidate=SeqaxWeightPlacementName.EMBEDDING_MLP,
            expected_high_level_all_gathers=9,
            expected_physical_collectives=15,
            expected_stablehlo_all_gathers=12,
            expected_pallas_regions=9,
            expected_parameter_bytes_per_device=33_152,
            expected_ring_equivalent_ici_bytes_per_device=23_808,
            trace_step_event=(
                f"{SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_STEP_PREFIX}.embedding-mlp.trace"
            ),
            counter_step_event=(
                f"{SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_STEP_PREFIX}.embedding-mlp.counters"
            ),
        ),
    )


def default_seqax_weight_placement_diagnostic_contract(
    runtime: RuntimeIdentity,
) -> SeqaxWeightPlacementDiagnosticContract:
    return SeqaxWeightPlacementDiagnosticContract(
        search_id=SEQAX_WEIGHT_PLACEMENT_SEARCH_ID,
        search_receipt_sha256=SEQAX_WEIGHT_PLACEMENT_SEARCH_RECEIPT_SHA256,
        baseline=SeqaxWeightPlacementName.SHARDED,
        timing_seed=12655767603698703491,
        warmup_iterations=SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_WARMUPS,
        measured_iterations=SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS,
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        trace_profiler_config=expected_seqax_profiler_contract(RunMode.TRACE),
        counter_profiler_config=expected_seqax_profiler_contract(RunMode.COUNTERS),
        candidates=_candidate_contracts(),
    )


def compare_weight_placement_profiles(
    contract: SeqaxWeightPlacementDiagnosticContract,
    profiles: tuple[SeqaxWeightPlacementCandidateProfiles, ...],
) -> SeqaxWeightPlacementDiagnosticComparison:
    if tuple(value.candidate for value in profiles) != tuple(
        value.candidate for value in contract.candidates
    ):
        raise ValueError("Seqax weight-placement diagnostic profile order mismatch")
    for expected, observed in zip(contract.candidates, profiles, strict=True):
        for summary in (observed.trace, observed.counters):
            if summary.module_execution_count != contract.measured_iterations:
                raise ValueError("Seqax weight-placement diagnostic execution count mismatch")
            values = (
                summary.high_level_all_gathers,
                summary.physical_collectives,
                summary.stablehlo_all_gathers,
                summary.pallas_regions,
                summary.parameter_bytes_per_device,
                summary.ring_equivalent_ici_bytes_per_device,
            )
            expected_values = (
                expected.expected_high_level_all_gathers,
                expected.expected_physical_collectives,
                expected.expected_stablehlo_all_gathers,
                expected.expected_pallas_regions,
                expected.expected_parameter_bytes_per_device,
                expected.expected_ring_equivalent_ici_bytes_per_device,
            )
            if values != expected_values:
                raise ValueError("Seqax weight-placement diagnostic static evidence mismatch")
    baseline, candidate = profiles

    def change(candidate_value: float, baseline_value: float) -> float:
        return (candidate_value - baseline_value) / baseline_value

    return SeqaxWeightPlacementDiagnosticComparison(
        baseline=baseline.candidate,
        candidate=candidate.candidate,
        trace_module_median_change=change(
            candidate.trace.module_median_duration_ns,
            baseline.trace.module_median_duration_ns,
        ),
        counter_module_median_change=change(
            candidate.counters.module_median_duration_ns,
            baseline.counters.module_median_duration_ns,
        ),
        trace_collective_completion_change=change(
            candidate.trace.collective_completion_average_self_time_sum_ns_per_device,
            baseline.trace.collective_completion_average_self_time_sum_ns_per_device,
        ),
        counter_collective_completion_change=change(
            candidate.counters.collective_completion_average_self_time_sum_ns_per_device,
            baseline.counters.collective_completion_average_self_time_sum_ns_per_device,
        ),
        trace_pallas_self_time_change=change(
            candidate.trace.pallas_average_self_time_sum_ns_per_device,
            baseline.trace.pallas_average_self_time_sum_ns_per_device,
        ),
        counter_pallas_self_time_change=change(
            candidate.counters.pallas_average_self_time_sum_ns_per_device,
            baseline.counters.pallas_average_self_time_sum_ns_per_device,
        ),
        high_level_all_gathers_eliminated=(
            baseline.trace.high_level_all_gathers - candidate.trace.high_level_all_gathers
        ),
        physical_collectives_eliminated=(
            baseline.trace.physical_collectives - candidate.trace.physical_collectives
        ),
        stablehlo_all_gathers_eliminated=(
            baseline.trace.stablehlo_all_gathers - candidate.trace.stablehlo_all_gathers
        ),
        ring_equivalent_ici_bytes_eliminated_per_device=(
            baseline.trace.ring_equivalent_ici_bytes_per_device
            - candidate.trace.ring_equivalent_ici_bytes_per_device
        ),
        parameter_bytes_added_per_device=(
            candidate.trace.parameter_bytes_per_device - baseline.trace.parameter_bytes_per_device
        ),
        interpretation=(
            "Trace and counter captures are isolated profiler-instrumented diagnostics, not unprofiled promotion measurements.",
            "Pallas and collective rows are non-additive XProf attribution inventories, not critical-path decomposition.",
            "Counter evidence establishes selected hardware series and does not derive MBU or MFU.",
            "The accepted search remains a no-winner result; this comparison cannot promote either placement.",
        ),
    )
