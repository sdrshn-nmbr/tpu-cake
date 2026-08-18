from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA
from tpu_cake.jax_lowering import JaxTensorContract
from tpu_cake.seqax_pallas_search import (
    SEQAX_PALLAS_CORRECTNESS_SEEDS,
    SEQAX_PALLAS_SEARCH_PARAMETERS,
    SEQAX_PALLAS_TIMING_SEED,
    SeqaxPallasCandidateCorrectness,
    SeqaxPallasCandidateStatistics,
    SeqaxPallasConfirmationStatistics,
    SeqaxPallasDevice,
    SeqaxPallasRoundObservation,
)
from tpu_cake.workloads.seqax_forward import (
    REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA,
    SEQAX_FORWARD_INPUT_NAMES,
    SHARDED_WEIGHT_DATA,
    SeqaxDataAxisPlacement,
    SeqaxWeightDataPlacement,
)

SEQAX_WEIGHT_PLACEMENT_SCHEMA = "seqax-weight-data-placement-search-v1"


class SeqaxWeightPlacementName(StrEnum):
    SHARDED = "sharded"
    EMBEDDING_MLP = "embedding-mlp"


class SeqaxWeightPlacementPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding: SeqaxDataAxisPlacement
    attention: SeqaxDataAxisPlacement
    feed_forward: SeqaxDataAxisPlacement

    def schedule_policy(self) -> SeqaxWeightDataPlacement:
        return SeqaxWeightDataPlacement(
            embedding=self.embedding,
            attention=self.attention,
            feed_forward=self.feed_forward,
        )


class SeqaxWeightPlacementCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: SeqaxWeightPlacementName
    policy: SeqaxWeightPlacementPolicy
    expected_high_level_all_gathers: int = Field(ge=0)
    expected_physical_collectives: int = Field(ge=0)
    expected_stablehlo_all_gathers: int = Field(ge=0)
    expected_parameter_bytes_per_device: int = Field(gt=0)


class SeqaxWeightPlacementContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_schema: str = SEQAX_WEIGHT_PLACEMENT_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    baseline: SeqaxWeightPlacementName
    parameters: dict[str, int]
    correctness_seeds: tuple[int, ...] = Field(min_length=5)
    expected_incumbent_cpu_oracle_passed: tuple[bool, ...] = Field(min_length=5)
    timing_seed: int
    cpu_oracle_replay_absolute_tolerance: float = Field(gt=0)
    warmup_iterations: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    rounds: int = Field(ge=8)
    confirmation_rounds: int = Field(ge=6)
    bootstrap_samples: int = Field(ge=1_000)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    require_isolated_memory_observation: bool
    candidates: tuple[SeqaxWeightPlacementCandidate, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxWeightPlacementContract:
        if self.search_schema != SEQAX_WEIGHT_PLACEMENT_SCHEMA:
            raise ValueError("Seqax weight-placement schema mismatch")
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("Seqax weight-placement identity schema mismatch")
        if self.parameters != SEQAX_PALLAS_SEARCH_PARAMETERS:
            raise ValueError("Seqax weight-placement parameters are not canonical")
        if self.correctness_seeds != SEQAX_PALLAS_CORRECTNESS_SEEDS:
            raise ValueError("Seqax weight-placement correctness seeds are not canonical")
        if self.expected_incumbent_cpu_oracle_passed != (
            True,
            True,
            True,
            False,
            False,
        ):
            raise ValueError("Seqax weight-placement oracle verdict scope is not canonical")
        if self.timing_seed != SEQAX_PALLAS_TIMING_SEED:
            raise ValueError("Seqax weight-placement timing seed is not canonical")
        if self.cpu_oracle_replay_absolute_tolerance != 2e-6:
            raise ValueError("Seqax weight-placement oracle replay tolerance is not canonical")
        if (
            self.warmup_iterations,
            self.measured_iterations,
            self.rounds,
            self.confirmation_rounds,
            self.bootstrap_samples,
            self.minimum_practical_improvement,
        ) != (5, 5, 12, 12, 10_000, 0.03):
            raise ValueError("Seqax weight-placement measurement protocol is not canonical")
        if self.measured_iterations % 2 == 0:
            raise ValueError("Seqax weight-placement sample count must be odd")
        if self.rounds % (2 * len(self.candidates)):
            raise ValueError("Seqax weight-placement rounds must form complete Latin squares")
        if self.confirmation_rounds % 2:
            raise ValueError("Seqax weight-placement confirmation order must balance")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax weight-placement search requires the TPU7x contract")
        if not self.require_isolated_memory_observation:
            raise ValueError("Seqax weight-placement search requires isolated memory evidence")
        names = tuple(candidate.name for candidate in self.candidates)
        if names != (
            SeqaxWeightPlacementName.SHARDED,
            SeqaxWeightPlacementName.EMBEDDING_MLP,
        ):
            raise ValueError("Seqax weight-placement candidates are not canonical")
        if self.baseline is not SeqaxWeightPlacementName.SHARDED:
            raise ValueError("Seqax weight-placement baseline must be sharded")
        expected = (
            SeqaxWeightPlacementCandidate(
                name=SeqaxWeightPlacementName.SHARDED,
                policy=_policy(SHARDED_WEIGHT_DATA),
                expected_high_level_all_gathers=14,
                expected_physical_collectives=20,
                expected_stablehlo_all_gathers=17,
                expected_parameter_bytes_per_device=22_912,
            ),
            SeqaxWeightPlacementCandidate(
                name=SeqaxWeightPlacementName.EMBEDDING_MLP,
                policy=_policy(REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA),
                expected_high_level_all_gathers=9,
                expected_physical_collectives=15,
                expected_stablehlo_all_gathers=12,
                expected_parameter_bytes_per_device=33_152,
            ),
        )
        if self.candidates != expected:
            raise ValueError("Seqax weight-placement candidate contracts are not canonical")
        return self

    @computed_field
    @property
    def search_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SeqaxWeightResidencyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: SeqaxWeightPlacementName
    runtime: RuntimeIdentity
    devices: tuple[SeqaxPallasDevice, ...] = Field(min_length=8, max_length=8)
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    timing_input_sha256: tuple[str, ...] = Field(min_length=1)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_bytes_per_device: int = Field(gt=0)
    device_bytes_limit: tuple[int, ...] = Field(min_length=8, max_length=8)
    peak_bytes_in_use: tuple[int, ...] = Field(min_length=8, max_length=8)
    largest_allocation_bytes: tuple[int, ...] = Field(min_length=8, max_length=8)
    isolated_process: bool
    fits_observed_device_memory: bool

    @model_validator(mode="after")
    def memory_values_are_consistent(self) -> SeqaxWeightResidencyObservation:
        if not self.isolated_process:
            raise ValueError("Seqax weight residency must come from an isolated process")
        if any(value <= 0 for value in self.device_bytes_limit):
            raise ValueError("Seqax weight residency needs positive device limits")
        if any(value <= 0 for value in self.peak_bytes_in_use):
            raise ValueError("Seqax weight residency needs positive peak usage")
        if any(value <= 0 for value in self.largest_allocation_bytes):
            raise ValueError("Seqax weight residency needs positive allocation sizes")
        if any(
            largest > peak
            for largest, peak in zip(
                self.largest_allocation_bytes,
                self.peak_bytes_in_use,
                strict=True,
            )
        ):
            raise ValueError("Seqax weight residency allocation exceeds peak usage")
        if any(value < self.parameter_bytes_per_device for value in self.peak_bytes_in_use):
            raise ValueError("Seqax weight residency peak cannot be below parameter residency")
        expected = all(
            peak <= limit
            for peak, limit in zip(
                self.peak_bytes_in_use,
                self.device_bytes_limit,
                strict=True,
            )
        )
        if self.fits_observed_device_memory is not expected:
            raise ValueError("Seqax weight residency feasibility does not match observations")
        return self


class SeqaxWeightPlacementPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: SeqaxWeightPlacementName
    policy: SeqaxWeightPlacementPolicy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    high_level_all_gathers: int = Field(ge=0)
    physical_collectives: int = Field(ge=0)
    stablehlo_all_gathers: int = Field(ge=0)
    pallas_regions: int = Field(gt=0)
    parameter_bytes_per_device: int = Field(gt=0)


class SeqaxWeightPlacementResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: SeqaxWeightPlacementName
    runtime: RuntimeIdentity
    devices: tuple[SeqaxPallasDevice, ...] = Field(min_length=8, max_length=8)
    timing_input_sha256: tuple[str, ...] = Field(min_length=1)
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    plans: tuple[SeqaxWeightPlacementPlan, ...] = Field(min_length=2, max_length=2)
    memory: tuple[SeqaxWeightResidencyObservation, ...] = Field(min_length=2, max_length=2)
    correctness: tuple[SeqaxPallasCandidateCorrectness, ...] = Field(min_length=2, max_length=2)
    execution_orders: tuple[tuple[str, ...], ...]
    rounds: tuple[SeqaxPallasRoundObservation, ...]
    candidates: tuple[SeqaxPallasCandidateStatistics, ...]
    provisional_winner: str | None
    confirmation_rounds: tuple[SeqaxPallasRoundObservation, ...]
    confirmation: SeqaxPallasConfirmationStatistics | None
    winner: str | None
    correctness_scope: str = Field(pattern=r"^incumbent-bit-exact$")

    @model_validator(mode="after")
    def evidence_sets_are_consistent(self) -> SeqaxWeightPlacementResult:
        plan_names = tuple(value.candidate for value in self.plans)
        memory_names = tuple(value.candidate for value in self.memory)
        correctness_names = tuple(value.name for value in self.correctness)
        statistics_names = tuple(value.name for value in self.candidates)
        if len(set(plan_names)) != len(plan_names):
            raise ValueError("Seqax weight-placement plans must be unique")
        if not (plan_names == memory_names == correctness_names == statistics_names):
            raise ValueError("Seqax weight-placement evidence candidate sets differ")
        return self


class SeqaxWeightPlacementReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str = Field(pattern=r"^passed$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)


def _policy(value: SeqaxWeightDataPlacement) -> SeqaxWeightPlacementPolicy:
    return SeqaxWeightPlacementPolicy(
        embedding=value.embedding,
        attention=value.attention,
        feed_forward=value.feed_forward,
    )


def default_seqax_weight_placement_contract(
    runtime: RuntimeIdentity,
) -> SeqaxWeightPlacementContract:
    return SeqaxWeightPlacementContract(
        baseline=SeqaxWeightPlacementName.SHARDED,
        parameters=SEQAX_PALLAS_SEARCH_PARAMETERS,
        correctness_seeds=SEQAX_PALLAS_CORRECTNESS_SEEDS,
        expected_incumbent_cpu_oracle_passed=(True, True, True, False, False),
        timing_seed=SEQAX_PALLAS_TIMING_SEED,
        cpu_oracle_replay_absolute_tolerance=2e-6,
        warmup_iterations=5,
        measured_iterations=5,
        rounds=12,
        confirmation_rounds=12,
        bootstrap_samples=10_000,
        minimum_practical_improvement=0.03,
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        require_isolated_memory_observation=True,
        candidates=(
            SeqaxWeightPlacementCandidate(
                name=SeqaxWeightPlacementName.SHARDED,
                policy=_policy(SHARDED_WEIGHT_DATA),
                expected_high_level_all_gathers=14,
                expected_physical_collectives=20,
                expected_stablehlo_all_gathers=17,
                expected_parameter_bytes_per_device=22_912,
            ),
            SeqaxWeightPlacementCandidate(
                name=SeqaxWeightPlacementName.EMBEDDING_MLP,
                policy=_policy(REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA),
                expected_high_level_all_gathers=9,
                expected_physical_collectives=15,
                expected_stablehlo_all_gathers=12,
                expected_parameter_bytes_per_device=33_152,
            ),
        ),
    )


def parameter_residency_bytes_per_device(
    input_contracts: tuple[JaxTensorContract, ...],
    *,
    mesh: dict[str, int],
) -> int:
    if len(input_contracts) != len(SEQAX_FORWARD_INPUT_NAMES):
        raise ValueError("Seqax weight residency input contract count mismatch")
    dtype_bytes = {
        "bool": 1,
        "uint32": 4,
        "int32": 4,
        "float32": 4,
        "bfloat16": 2,
    }
    total = 0
    for name, contract in zip(SEQAX_FORWARD_INPUT_NAMES, input_contracts, strict=True):
        if name in {"tokens", "sequence_starts"}:
            continue
        if contract.dtype not in dtype_bytes:
            raise ValueError(f"Seqax weight residency dtype unsupported: {contract.dtype}")
        total += math.prod(contract.local_shape(mesh)) * dtype_bytes[contract.dtype]
    return total
