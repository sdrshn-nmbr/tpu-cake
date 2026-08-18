from __future__ import annotations

import hashlib
import json
import statistics
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, semantic_seed

SEQAX_PALLAS_SEARCH_SCHEMA = "seqax-pallas-tile-search-v1"
SEQAX_PALLAS_SEARCH_PARAMETERS = {
    "batch": 2,
    "data_mesh": 2,
    "feed_forward": 16,
    "head": 4,
    "key_value_heads": 4,
    "layers": 1,
    "model": 256,
    "query_groups": 2,
    "rope_max_timescale": 256,
    "sequence": 1,
    "tensor_mesh": 4,
    "vocabulary": 16,
}
SEQAX_PALLAS_CORRECTNESS_SEEDS = tuple(
    semantic_seed("seqax-pallas-tiled-einsum-v1", str(index)) for index in range(5)
)
SEQAX_PALLAS_TIMING_SEED = SEQAX_PALLAS_CORRECTNESS_SEEDS[0]


class TilePolicy(StrEnum):
    FULL = "full"
    SPLIT_K = "split-k"
    SPLIT_N = "split-n"
    SPLIT_KN = "split-kn"


class SeqaxPallasSearchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    policy: TilePolicy
    maximum_tile_k: int | None = Field(default=None, gt=0)
    maximum_tile_n: int | None = Field(default=None, gt=0)
    expected_changed_regions: int = Field(ge=0)

    @model_validator(mode="after")
    def policy_matches_limits(self) -> SeqaxPallasSearchCandidate:
        expected = {
            TilePolicy.FULL: (None, None, 0),
            TilePolicy.SPLIT_K: (128, None, 5),
            TilePolicy.SPLIT_N: (None, 128, 2),
            TilePolicy.SPLIT_KN: (128, 128, 7),
        }[self.policy]
        if (
            self.maximum_tile_k,
            self.maximum_tile_n,
            self.expected_changed_regions,
        ) != expected:
            raise ValueError("Seqax Pallas tile policy does not match its limits")
        return self


class SeqaxPallasSearchContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_schema: str = SEQAX_PALLAS_SEARCH_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    baseline: str
    parameters: dict[str, int]
    correctness_seeds: tuple[int, ...] = Field(min_length=5)
    timing_seed: int
    primitive_absolute_tolerance: float = Field(gt=0)
    primitive_relative_tolerance: float = Field(gt=0)
    warmup_iterations: int = Field(ge=1)
    measured_iterations: int = Field(ge=3)
    rounds: int = Field(ge=8)
    confirmation_rounds: int = Field(ge=6)
    bootstrap_samples: int = Field(ge=1_000)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    candidates: tuple[SeqaxPallasSearchCandidate, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def protocol_is_fixed_and_balanced(self) -> SeqaxPallasSearchContract:
        if self.search_schema != SEQAX_PALLAS_SEARCH_SCHEMA:
            raise ValueError("Seqax Pallas search schema mismatch")
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("Seqax Pallas search identity schema mismatch")
        if self.parameters != SEQAX_PALLAS_SEARCH_PARAMETERS:
            raise ValueError("Seqax Pallas search parameters are not canonical")
        if self.correctness_seeds != SEQAX_PALLAS_CORRECTNESS_SEEDS:
            raise ValueError("Seqax Pallas search correctness seeds are not canonical")
        if self.timing_seed != SEQAX_PALLAS_TIMING_SEED:
            raise ValueError("Seqax Pallas search timing seed is not canonical")
        if (self.primitive_absolute_tolerance, self.primitive_relative_tolerance) != (
            1e-3,
            1e-3,
        ):
            raise ValueError("Seqax Pallas primitive tolerance is not canonical")
        if (
            self.warmup_iterations,
            self.measured_iterations,
            self.rounds,
            self.confirmation_rounds,
            self.bootstrap_samples,
            self.minimum_practical_improvement,
        ) != (5, 5, 8, 6, 10_000, 0.03):
            raise ValueError("Seqax Pallas search measurement protocol is not canonical")
        if self.measured_iterations % 2 == 0:
            raise ValueError("Seqax Pallas timing needs an odd sample count")
        names = tuple(candidate.name for candidate in self.candidates)
        policies = tuple(candidate.policy for candidate in self.candidates)
        if len(names) != len(set(names)) or len(policies) != len(set(policies)):
            raise ValueError("Seqax Pallas search candidates must be unique")
        if names != ("incumbent", "split-k", "split-n", "split-kn") or policies != (
            TilePolicy.FULL,
            TilePolicy.SPLIT_K,
            TilePolicy.SPLIT_N,
            TilePolicy.SPLIT_KN,
        ):
            raise ValueError("Seqax Pallas search candidate set is not canonical")
        if self.baseline not in names:
            raise ValueError("Seqax Pallas search baseline is not a candidate")
        baseline = self.candidates[names.index(self.baseline)]
        if baseline.policy is not TilePolicy.FULL:
            raise ValueError("Seqax Pallas search baseline must use full tiles")
        if self.rounds % (2 * len(self.candidates)):
            raise ValueError(
                "Seqax Pallas search rounds must complete forward and reverse Latin squares"
            )
        if self.confirmation_rounds % 2:
            raise ValueError("Seqax Pallas confirmation rounds must balance both orders")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax Pallas search requires the fixed TPU7x device contract")
        return self

    @computed_field
    @property
    def search_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SeqaxPallasPrimitiveObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    shape_index: int = Field(ge=0)
    seed: int
    lhs_shape: tuple[int, ...] = Field(min_length=1)
    lhs_names: tuple[str, ...] = Field(min_length=1)
    rhs_shape: tuple[int, ...] = Field(min_length=1)
    rhs_names: tuple[str, ...] = Field(min_length=1)
    output_shape: tuple[int, ...] = Field(min_length=1)
    output_names: tuple[str, ...] = Field(min_length=1)
    contracting_dimensions: tuple[str, ...] = Field(min_length=1)
    tiles: tuple[int, int, int]
    dtype: str = Field(pattern=r"^(bf16|f32)$")
    lhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)
    passed: bool


class SeqaxPallasCandidatePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    policy: TilePolicy
    tiles: tuple[tuple[int, int, int], ...]
    changed_region_count: int = Field(ge=0)
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SeqaxPallasCandidateCorrectness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    input_sha256: tuple[tuple[str, ...], ...] = Field(min_length=5)
    output_sha256: tuple[str, ...] = Field(min_length=5)
    baseline_output_sha256: tuple[str, ...] = Field(min_length=5)
    exact_baseline_parity: bool
    cpu_oracle_sha256: tuple[str, ...] = Field(min_length=5)
    cpu_oracle_maximum_absolute_error: tuple[float, ...] = Field(min_length=5)
    cpu_oracle_maximum_relative_error: tuple[float, ...] = Field(min_length=5)
    cpu_oracle_passed: tuple[bool, ...] = Field(min_length=5)


class SeqaxPallasRoundObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_index: int = Field(ge=0)
    position: int = Field(ge=0)
    candidate: str
    samples_ns: tuple[int, ...] = Field(min_length=3)
    median_ns: float = Field(gt=0)

    @model_validator(mode="after")
    def median_matches_samples(self) -> SeqaxPallasRoundObservation:
        if self.median_ns != float(statistics.median(self.samples_ns)):
            raise ValueError("Seqax Pallas round median does not match raw samples")
        if any(value <= 0 for value in self.samples_ns):
            raise ValueError("Seqax Pallas timing samples must be positive")
        return self


class SeqaxPallasCandidateStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    run_count: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    median_round_ns: float = Field(gt=0)
    p90_round_ns: float = Field(gt=0)
    median_absolute_deviation_ns: float = Field(ge=0)
    coefficient_of_variation: float = Field(ge=0)
    improvement_over_baseline: float
    improvement_confidence_interval: tuple[float, float]
    promotable: bool


class SeqaxPallasConfirmationStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: str
    candidate: str
    execution_orders: tuple[tuple[str, str], ...]
    median_improvement: float
    improvement_confidence_interval: tuple[float, float]
    confirmed: bool


class SeqaxPallasDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=0)
    process_index: int = Field(ge=0)
    platform: str
    device_kind: str


class SeqaxPallasSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: str
    runtime: RuntimeIdentity
    device_kind: str
    device_count: int = Field(gt=0)
    devices: tuple[SeqaxPallasDevice, ...]
    timing_input_sha256: tuple[str, ...] = Field(min_length=1)
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    plans: tuple[SeqaxPallasCandidatePlan, ...]
    primitive_observations: tuple[SeqaxPallasPrimitiveObservation, ...]
    correctness: tuple[SeqaxPallasCandidateCorrectness, ...]
    execution_orders: tuple[tuple[str, ...], ...]
    rounds: tuple[SeqaxPallasRoundObservation, ...]
    candidates: tuple[SeqaxPallasCandidateStatistics, ...]
    provisional_winner: str | None
    confirmation_rounds: tuple[SeqaxPallasRoundObservation, ...]
    confirmation: SeqaxPallasConfirmationStatistics | None
    winner: str | None


class SeqaxPallasSearchReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str = Field(pattern=r"^passed$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)


def default_seqax_pallas_search_contract(
    runtime: RuntimeIdentity,
) -> SeqaxPallasSearchContract:
    return SeqaxPallasSearchContract(
        baseline="incumbent",
        parameters=SEQAX_PALLAS_SEARCH_PARAMETERS,
        correctness_seeds=SEQAX_PALLAS_CORRECTNESS_SEEDS,
        timing_seed=SEQAX_PALLAS_TIMING_SEED,
        primitive_absolute_tolerance=1e-3,
        primitive_relative_tolerance=1e-3,
        warmup_iterations=5,
        measured_iterations=5,
        rounds=8,
        confirmation_rounds=6,
        bootstrap_samples=10_000,
        minimum_practical_improvement=0.03,
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        candidates=(
            SeqaxPallasSearchCandidate(
                name="incumbent",
                policy=TilePolicy.FULL,
                expected_changed_regions=0,
            ),
            SeqaxPallasSearchCandidate(
                name="split-k",
                policy=TilePolicy.SPLIT_K,
                maximum_tile_k=128,
                expected_changed_regions=5,
            ),
            SeqaxPallasSearchCandidate(
                name="split-n",
                policy=TilePolicy.SPLIT_N,
                maximum_tile_n=128,
                expected_changed_regions=2,
            ),
            SeqaxPallasSearchCandidate(
                name="split-kn",
                policy=TilePolicy.SPLIT_KN,
                maximum_tile_k=128,
                maximum_tile_n=128,
                expected_changed_regions=7,
            ),
        ),
    )


def candidate_tiles(
    full_tiles: tuple[tuple[int, int, int], ...],
    candidate: SeqaxPallasSearchCandidate,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            tile_m,
            candidate.maximum_tile_k
            if candidate.maximum_tile_k is not None
            and tile_k > candidate.maximum_tile_k
            and tile_k % candidate.maximum_tile_k == 0
            else tile_k,
            candidate.maximum_tile_n
            if candidate.maximum_tile_n is not None
            and tile_n > candidate.maximum_tile_n
            and tile_n % candidate.maximum_tile_n == 0
            else tile_n,
        )
        for tile_m, tile_k, tile_n in full_tiles
    )


def execution_orders(
    contract: SeqaxPallasSearchContract,
) -> tuple[tuple[str, ...], ...]:
    names = tuple(candidate.name for candidate in contract.candidates)
    orders = []
    for round_index in range(contract.rounds):
        square = round_index // len(names)
        basis = names if square % 2 == 0 else tuple(reversed(names))
        offset = round_index % len(names)
        orders.append(basis[offset:] + basis[:offset])
    return tuple(orders)


def _improvement_interval(
    values: tuple[float, ...],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    observed = np.asarray(values, dtype=np.float64)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = np.median(generator.choice(observed, len(observed), replace=True))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def candidate_statistics(
    contract: SeqaxPallasSearchContract,
    observations: tuple[SeqaxPallasRoundObservation, ...],
) -> tuple[SeqaxPallasCandidateStatistics, ...]:
    expected_positions = tuple(
        (round_index, position, name)
        for round_index, order in enumerate(execution_orders(contract))
        for position, name in enumerate(order)
    )
    observed_positions = tuple(
        (item.round_index, item.position, item.candidate) for item in observations
    )
    if observed_positions != expected_positions:
        raise ValueError("Seqax Pallas search execution order mismatch")
    if any(len(item.samples_ns) != contract.measured_iterations for item in observations):
        raise ValueError("Seqax Pallas search sample count mismatch")
    by_name = {
        candidate.name: tuple(item for item in observations if item.candidate == candidate.name)
        for candidate in contract.candidates
    }
    if any(len(values) != contract.rounds for values in by_name.values()):
        raise ValueError("Seqax Pallas search observation count mismatch")
    baseline = by_name[contract.baseline]
    if tuple(item.round_index for item in baseline) != tuple(range(contract.rounds)):
        raise ValueError("Seqax Pallas baseline rounds are not canonical")
    result = []
    for candidate_index, candidate in enumerate(contract.candidates):
        runs = by_name[candidate.name]
        if tuple(item.round_index for item in runs) != tuple(range(contract.rounds)):
            raise ValueError(f"Seqax Pallas candidate rounds are not canonical: {candidate.name}")
        improvements = tuple(
            (base.median_ns - observed.median_ns) / base.median_ns
            for base, observed in zip(baseline, runs, strict=True)
        )
        interval = _improvement_interval(
            improvements,
            samples=contract.bootstrap_samples,
            seed=int(contract.search_id[:16], 16) ^ candidate_index,
        )
        medians = tuple(item.median_ns for item in runs)
        median = float(statistics.median(medians))
        result.append(
            SeqaxPallasCandidateStatistics(
                name=candidate.name,
                run_count=len(runs),
                sample_count=sum(len(item.samples_ns) for item in runs),
                median_round_ns=median,
                p90_round_ns=float(np.quantile(medians, 0.9)),
                median_absolute_deviation_ns=float(
                    statistics.median(abs(value - median) for value in medians)
                ),
                coefficient_of_variation=(
                    statistics.pstdev(medians) / statistics.mean(medians)
                    if len(medians) > 1 and statistics.mean(medians)
                    else 0.0
                ),
                improvement_over_baseline=float(statistics.median(improvements)),
                improvement_confidence_interval=interval,
                promotable=(
                    candidate.name != contract.baseline
                    and interval[0] > contract.minimum_practical_improvement
                ),
            )
        )
    return tuple(result)


def confirmation_statistics(
    contract: SeqaxPallasSearchContract,
    candidate: str,
    observations: tuple[SeqaxPallasRoundObservation, ...],
) -> SeqaxPallasConfirmationStatistics:
    if candidate == contract.baseline or candidate not in {
        value.name for value in contract.candidates
    }:
        raise ValueError("Seqax Pallas confirmation candidate is invalid")
    orders = tuple(
        ((contract.baseline, candidate) if round_index % 2 == 0 else (candidate, contract.baseline))
        for round_index in range(contract.confirmation_rounds)
    )
    expected = tuple(
        (round_index, position, name)
        for round_index, order in enumerate(orders)
        for position, name in enumerate(order)
    )
    observed = tuple((item.round_index, item.position, item.candidate) for item in observations)
    if observed != expected:
        raise ValueError("Seqax Pallas confirmation execution order mismatch")
    if any(len(item.samples_ns) != contract.measured_iterations for item in observations):
        raise ValueError("Seqax Pallas confirmation sample count mismatch")
    baseline = tuple(item for item in observations if item.candidate == contract.baseline)
    challenger = tuple(item for item in observations if item.candidate == candidate)
    improvements = tuple(
        (base.median_ns - contender.median_ns) / base.median_ns
        for base, contender in zip(baseline, challenger, strict=True)
    )
    interval = _improvement_interval(
        improvements,
        samples=contract.bootstrap_samples,
        seed=int(contract.search_id[16:32], 16),
    )
    return SeqaxPallasConfirmationStatistics(
        baseline=contract.baseline,
        candidate=candidate,
        execution_orders=orders,
        median_improvement=float(statistics.median(improvements)),
        improvement_confidence_interval=interval,
        confirmed=interval[0] > contract.minimum_practical_improvement,
    )
