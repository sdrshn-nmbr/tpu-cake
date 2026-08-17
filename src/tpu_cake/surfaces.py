from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import WorkloadStage


class OutputEquivalencePolicy(StrEnum):
    EXACT_IDENTITY = "exact_identity"
    INDEPENDENT_ORACLE_AND_CROSS_MODE_TOLERANCE = (
        "independent_oracle_and_cross_mode_tolerance"
    )


class AttentionScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    stage: WorkloadStage
    batch_size: int = Field(gt=0)
    query_tokens_per_request: int = Field(gt=0)
    context_lengths: tuple[int, ...] = Field(min_length=1)
    page_size: int = Field(gt=0)
    dtype: str = Field(min_length=1)
    sharding: tuple[str, ...]
    weight: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def lengths_match_the_batch(self) -> AttentionScenario:
        if len(self.context_lengths) != self.batch_size:
            raise ValueError("one context length is required for every request")
        if any(length <= 0 for length in self.context_lengths):
            raise ValueError("context lengths must be positive")
        if self.stage is WorkloadStage.STEADY_DECODE and self.query_tokens_per_request != 1:
            raise ValueError("steady decode scenarios must use one query token per request")
        return self

    @computed_field
    @property
    def allocated_pages(self) -> int:
        return sum((length + self.page_size - 1) // self.page_size for length in self.context_lengths)

    @computed_field
    @property
    def page_occupancy(self) -> Decimal:
        return Decimal(sum(self.context_lengths)) / Decimal(
            self.allocated_pages * self.page_size
        )


class AttentionWorkloadSurface(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    scenarios: tuple[AttentionScenario, ...] = Field(min_length=1)
    minimum_practical_improvement: Decimal = Field(gt=0, lt=1)
    maximum_scenario_regression: Decimal = Field(default=Decimal("0.01"), ge=0, lt=1)
    bootstrap_samples: int = Field(default=10_000, ge=1_000)
    output_equivalence: OutputEquivalencePolicy = OutputEquivalencePolicy.EXACT_IDENTITY

    @model_validator(mode="after")
    def scenario_names_are_unique(self) -> AttentionWorkloadSurface:
        names = tuple(scenario.name for scenario in self.scenarios)
        if len(names) != len(set(names)):
            raise ValueError("workload surface scenario names must be unique")
        return self

    @computed_field
    @property
    def surface_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"surface_id"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SeqaxForwardScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    batch: int = Field(gt=0)
    sequence: int = Field(gt=0)
    model: int = Field(gt=0)
    vocabulary: int = Field(gt=0)
    feed_forward: int = Field(gt=0)
    query_groups: int = Field(gt=0)
    key_value_heads: int = Field(gt=0)
    head: int = Field(gt=0)
    layers: int = Field(gt=0)
    data_mesh: int = Field(gt=0)
    tensor_mesh: int = Field(gt=0)
    rope_max_timescale: int = Field(gt=1)
    weight: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def dimensions_fit_the_mesh(self) -> SeqaxForwardScenario:
        if self.batch % self.data_mesh:
            raise ValueError("batch must divide evenly over the data mesh")
        if self.model % self.data_mesh or self.model % self.tensor_mesh:
            raise ValueError("model width must divide evenly over both mesh axes")
        if self.vocabulary % self.tensor_mesh or self.feed_forward % self.tensor_mesh:
            raise ValueError("vocabulary and feed-forward widths must divide over tensor mesh")
        if self.key_value_heads % self.tensor_mesh:
            raise ValueError("key/value heads must divide evenly over tensor mesh")
        if self.head % 2:
            raise ValueError("head width must be even for rotary embedding")
        return self

    def parameters(self) -> dict[str, int]:
        return {
            key: value
            for key, value in self.model_dump(mode="python").items()
            if key not in {"name", "weight"}
        }


class SeqaxForwardWorkloadSurface(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    scenarios: tuple[SeqaxForwardScenario, ...] = Field(min_length=2)
    minimum_practical_improvement: Decimal = Field(gt=0, lt=1)
    maximum_scenario_regression: Decimal = Field(default=Decimal("0.01"), ge=0, lt=1)
    bootstrap_samples: int = Field(default=10_000, ge=1_000)
    output_equivalence: OutputEquivalencePolicy = OutputEquivalencePolicy.EXACT_IDENTITY
    oracle_quantization_decimals: int = Field(default=6, ge=0, le=9)

    @model_validator(mode="after")
    def scenario_names_are_unique(self) -> SeqaxForwardWorkloadSurface:
        names = tuple(scenario.name for scenario in self.scenarios)
        if len(names) != len(set(names)):
            raise ValueError("workload surface scenario names must be unique")
        return self

    @computed_field
    @property
    def surface_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"surface_id"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


WorkloadSurface = AttentionWorkloadSurface | SeqaxForwardWorkloadSurface


class ScenarioObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: str
    round_medians_ns: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=5)
    round_samples_ns: tuple[tuple[Annotated[int, Field(gt=0)], ...], ...] = ()
    ran_first: tuple[bool, ...] = Field(min_length=5)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profiled: bool
    passed: bool

    @model_validator(mode="after")
    def timing_rounds_have_positions(self) -> ScenarioObservation:
        if len(self.ran_first) != len(self.round_medians_ns):
            raise ValueError("every timing round needs an execution position")
        if self.round_samples_ns:
            if len(self.round_samples_ns) != len(self.round_medians_ns):
                raise ValueError("every timing round needs its raw samples")
            if any(
                Decimal(median) != Decimal(str(float(np.median(samples))))
                for median, samples in zip(
                    self.round_medians_ns, self.round_samples_ns, strict=True
                )
            ):
                raise ValueError("round medians must match the preserved raw samples")
        return self


class SurfaceCandidateObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str = Field(min_length=1)
    scenarios: tuple[ScenarioObservation, ...]

    @model_validator(mode="after")
    def scenario_names_are_unique(self) -> SurfaceCandidateObservation:
        names = tuple(value.scenario for value in self.scenarios)
        if len(names) != len(set(names)):
            raise ValueError("candidate scenario names must be unique")
        return self


class SurfaceComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: str
    candidate: str
    weighted_median_improvement: Decimal
    confidence_interval: tuple[Decimal, Decimal]
    scenario_improvements: dict[str, Decimal]
    promotable: bool


def compare_surface_candidates(
    surface: WorkloadSurface,
    baseline: SurfaceCandidateObservation,
    candidate: SurfaceCandidateObservation,
) -> SurfaceComparison:
    baseline_by_name = {value.scenario: value for value in baseline.scenarios}
    candidate_by_name = {value.scenario: value for value in candidate.scenarios}
    expected_names = tuple(scenario.name for scenario in surface.scenarios)
    if set(baseline_by_name) != set(expected_names) or set(candidate_by_name) != set(
        expected_names
    ):
        raise ValueError("surface observations must cover every declared scenario exactly once")
    if baseline.candidate == candidate.candidate:
        raise ValueError("surface comparison needs distinct candidates")
    round_counts = {
        len(observation.round_medians_ns)
        for observation in (*baseline.scenarios, *candidate.scenarios)
    }
    if len(round_counts) != 1:
        raise ValueError("surface observations need the same number of independent rounds")
    if any(
        not baseline_by_name[name].passed or not candidate_by_name[name].passed
        for name in expected_names
    ):
        raise ValueError("surface candidates must pass their numerical contracts")
    if (
        surface.output_equivalence is OutputEquivalencePolicy.EXACT_IDENTITY
        and any(
            baseline_by_name[name].output_sha256
            != candidate_by_name[name].output_sha256
            for name in expected_names
        )
    ):
        raise ValueError("surface candidates need matched correct outputs")
    if any(
        baseline_by_name[name].input_sha256 != candidate_by_name[name].input_sha256
        or baseline_by_name[name].runtime_sha256
        != candidate_by_name[name].runtime_sha256
        for name in expected_names
    ):
        raise ValueError("surface candidates need matched inputs and runtimes")
    if any(
        baseline_by_name[name].profiled or candidate_by_name[name].profiled
        for name in expected_names
    ):
        raise ValueError("surface promotion requires unprofiled timing")
    if any(
        tuple(not position for position in baseline_by_name[name].ran_first)
        != candidate_by_name[name].ran_first
        or any(
            left == right
            for left, right in zip(
                baseline_by_name[name].ran_first,
                baseline_by_name[name].ran_first[1:],
            )
        )
        for name in expected_names
    ):
        raise ValueError("surface candidates need alternating matched order")

    total_weight = sum(scenario.weight for scenario in surface.scenarios)
    normalized_weights = {
        scenario.name: scenario.weight / total_weight for scenario in surface.scenarios
    }
    scenario_round_improvements: dict[str, tuple[Decimal, ...]] = {}
    for name in expected_names:
        base = baseline_by_name[name].round_medians_ns
        contender = candidate_by_name[name].round_medians_ns
        scenario_round_improvements[name] = tuple(
            (Decimal(base_value) - Decimal(candidate_value)) / Decimal(base_value)
            for base_value, candidate_value in zip(base, contender, strict=True)
        )
    rounds = next(iter(round_counts))
    generator = np.random.default_rng(
        int(hashlib.sha256(f"{surface.surface_id}:{candidate.candidate}".encode()).hexdigest()[:16], 16)
    )
    estimates = np.empty(surface.bootstrap_samples, dtype=np.float64)
    for index in range(surface.bootstrap_samples):
        estimates[index] = float(
            sum(
                normalized_weights[name]
                * Decimal(
                    str(
                        float(
                            np.median(
                                generator.choice(
                                    np.asarray(
                                        [
                                            float(value)
                                            for value in scenario_round_improvements[name]
                                        ],
                                        dtype=np.float64,
                                    ),
                                    rounds,
                                    replace=True,
                                )
                            )
                        )
                    )
                )
                for name in expected_names
            )
        )
    low, high = np.quantile(estimates, (0.025, 0.975))
    scenario_improvements = {
        name: Decimal(str(float(np.median([float(value) for value in values]))))
        for name, values in scenario_round_improvements.items()
    }
    weighted_median = sum(
        normalized_weights[name] * scenario_improvements[name] for name in expected_names
    )
    confidence = (Decimal(str(float(low))), Decimal(str(float(high))))
    promotable = (
        confidence[0] > surface.minimum_practical_improvement
        and min(scenario_improvements.values()) >= -surface.maximum_scenario_regression
    )
    return SurfaceComparison(
        surface_id=surface.surface_id,
        baseline=baseline.candidate,
        candidate=candidate.candidate,
        weighted_median_improvement=weighted_median,
        confidence_interval=confidence,
        scenario_improvements=scenario_improvements,
        promotable=promotable,
    )
