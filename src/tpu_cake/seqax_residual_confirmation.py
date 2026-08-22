from __future__ import annotations

import math
import statistics
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis
from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256, semantic_seed
from tpu_cake.seqax_numerical import default_seqax_bf16_validation_contract
from tpu_cake.seqax_pallas_search import SEQAX_PALLAS_CORRECTNESS_SEEDS, SeqaxPallasDevice
from tpu_cake.seqax_residual_profile import (
    SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT,
    SEQAX_RESIDUAL_PROFILE_TIMING_SEED,
    SeqaxResidualCorrectnessObservation,
    SeqaxResidualProfileCandidateContract,
    default_seqax_residual_profile_contract,
)
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy

SEQAX_RESIDUAL_CONFIRMATION_SCHEMA = "seqax-residual-all-reduce-confirmation-v1"
SOURCE_PROFILE_ID = "1a7c702597d6794ba6159314fcf4b508a6506e9d759ce6bbf872dff7535a7576"
SOURCE_PROFILE_ARCHIVE_SHA256 = "8308631742c9af50a90b48962aa5e730f966d607ed7e0b856a99ea53de298e43"
SOURCE_PROFILE_RECEIPT_SHA256 = "cc9bdbfee9c935ef0e0d719017a3424d3cf5575faf573c00002bcf5693d9fe6c"
SOURCE_PROFILE_RESULT_SHA256 = "6f14b3abea1545b73883930a4155f7a9fb166a4597a6adf2706b1927b0948e22"


class SeqaxResidualConfirmationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_schema: str = SEQAX_RESIDUAL_CONFIRMATION_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    source_profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_profile_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_profile_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_profile_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compilation_source_root: str
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
    analysis_index: int = Field(ge=2)
    allow_early_stopping: bool
    allow_further_retry: bool
    candidates_resident_together: bool
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    parameters: dict[str, int | str]
    plans: tuple[SeqaxResidualProfileCandidateContract, ...] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxResidualConfirmationContract:
        if self.confirmation_schema != SEQAX_RESIDUAL_CONFIRMATION_SCHEMA:
            raise ValueError("Seqax residual confirmation schema mismatch")
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("Seqax residual confirmation identity schema mismatch")
        if (
            self.source_profile_id,
            self.source_profile_archive_sha256,
            self.source_profile_receipt_sha256,
            self.source_profile_result_sha256,
        ) != (
            SOURCE_PROFILE_ID,
            SOURCE_PROFILE_ARCHIVE_SHA256,
            SOURCE_PROFILE_RECEIPT_SHA256,
            SOURCE_PROFILE_RESULT_SHA256,
        ):
            raise ValueError("Seqax residual confirmation profile provenance mismatch")
        numerical = default_seqax_bf16_validation_contract()
        profile = default_seqax_residual_profile_contract(self.runtime)
        if self.numerical_contract_id != numerical.contract_id:
            raise ValueError("Seqax residual confirmation numerical contract mismatch")
        if self.compilation_source_root != SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT:
            raise ValueError("Seqax residual confirmation compilation root mismatch")
        if (self.baseline, self.candidate) != (
            SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
            SeqaxResidualNormStrategy.STANDARD,
        ):
            raise ValueError("Seqax residual confirmation candidate pair mismatch")
        if self.timing_seed != SEQAX_RESIDUAL_PROFILE_TIMING_SEED:
            raise ValueError("Seqax residual confirmation timing seed mismatch")
        if self.correctness_seeds != SEQAX_PALLAS_CORRECTNESS_SEEDS:
            raise ValueError("Seqax residual confirmation correctness seeds mismatch")
        if (
            self.warmup_iterations,
            self.measured_iterations,
            self.paired_rounds,
            self.bootstrap_samples,
            self.confidence_level,
            self.minimum_practical_improvement,
            self.analysis_index,
            self.allow_early_stopping,
            self.allow_further_retry,
            self.candidates_resident_together,
        ) != (5, 5, 32, 100_000, 0.99, 0.03, 3, False, False, True):
            raise ValueError("Seqax residual confirmation measurement protocol mismatch")
        if self.measured_iterations % 2 == 0 or self.paired_rounds % 2:
            raise ValueError("Seqax residual confirmation protocol must be balanced")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax residual confirmation TPU contract mismatch")
        if self.parameters != profile.parameters or self.plans != profile.candidates:
            raise ValueError("Seqax residual confirmation plan surface mismatch")
        return self

    @computed_field
    @property
    def confirmation_id(self) -> str:
        return model_identity_sha256(self)


class SeqaxResidualConfirmationRunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_RESIDUAL_CONFIRMATION_SCHEMA] = SEQAX_RESIDUAL_CONFIRMATION_SCHEMA
    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class SeqaxResidualConfirmationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_pallas_compiler_collectives: CompilerCollectiveAnalysis
    pallas_regions: int = Field(gt=0)
    all_gathers: int = Field(ge=0)
    all_reduces: int = Field(ge=0)
    reduce_scatters: int = Field(ge=0)


class SeqaxResidualTimingRound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    round_index: int = Field(ge=0)
    position: int = Field(ge=0, le=1)
    candidate: SeqaxResidualNormStrategy
    samples_ns: tuple[int, ...] = Field(min_length=3)
    median_ns: float = Field(gt=0)

    @model_validator(mode="after")
    def samples_are_valid(self) -> SeqaxResidualTimingRound:
        if any(value <= 0 for value in self.samples_ns):
            raise ValueError("Seqax residual confirmation samples must be positive")
        if self.median_ns != float(statistics.median(self.samples_ns)):
            raise ValueError("Seqax residual confirmation median mismatch")
        return self


class SeqaxResidualConfirmationStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    baseline: SeqaxResidualNormStrategy
    candidate: SeqaxResidualNormStrategy
    round_count: int = Field(gt=0)
    paired_improvements: tuple[float, ...] = Field(min_length=1)
    median_improvement: float
    mean_improvement: float
    improvement_confidence_interval: tuple[float, float]
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_seed: int
    bootstrap_samples: int = Field(gt=0)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    confirmed: bool

    @model_validator(mode="after")
    def statistics_are_valid(self) -> SeqaxResidualConfirmationStatistics:
        values = (
            *self.paired_improvements,
            self.median_improvement,
            self.mean_improvement,
            *self.improvement_confidence_interval,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Seqax residual confirmation statistics must be finite")
        if self.improvement_confidence_interval[0] > self.improvement_confidence_interval[1]:
            raise ValueError("Seqax residual confirmation interval is inverted")
        return self


class SeqaxResidualTimingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SeqaxResidualConfirmationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_profile_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeIdentity
    devices: tuple[SeqaxPallasDevice, ...] = Field(min_length=8, max_length=8)
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    plans: tuple[SeqaxResidualConfirmationPlan, ...] = Field(min_length=2, max_length=2)
    correctness: tuple[SeqaxResidualCorrectnessObservation, ...] = Field(
        min_length=10,
        max_length=10,
    )
    timing_input_sha256: tuple[str, ...] = Field(min_length=13, max_length=13)
    pre_timing_outputs: tuple[SeqaxResidualTimingOutput, ...] = Field(
        min_length=2,
        max_length=2,
    )
    execution_orders: tuple[tuple[SeqaxResidualNormStrategy, SeqaxResidualNormStrategy], ...]
    rounds: tuple[SeqaxResidualTimingRound, ...]
    post_timing_outputs: tuple[SeqaxResidualTimingOutput, ...] = Field(
        min_length=2,
        max_length=2,
    )
    statistics: SeqaxResidualConfirmationStatistics
    winner: SeqaxResidualNormStrategy | None
    claim_scope: str = Field(pattern=r"^fixed-model256-layer1-sequence1-bf16-pallas-performance$")

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> SeqaxResidualConfirmationResult:
        candidates = (
            SeqaxResidualNormStrategy.STANDARD,
            SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        )
        if tuple(value.candidate for value in self.plans) != candidates:
            raise ValueError("Seqax residual confirmation plan order mismatch")
        if tuple(value.candidate for value in self.pre_timing_outputs) != candidates:
            raise ValueError("Seqax residual confirmation pre-output order mismatch")
        if tuple(value.candidate for value in self.post_timing_outputs) != candidates:
            raise ValueError("Seqax residual confirmation post-output order mismatch")
        expected_correctness = tuple(
            (candidate, seed) for candidate in candidates for seed in SEQAX_PALLAS_CORRECTNESS_SEEDS
        )
        observed_correctness = tuple((value.candidate, value.seed) for value in self.correctness)
        if observed_correctness != expected_correctness:
            raise ValueError("Seqax residual confirmation correctness inventory mismatch")
        if self.statistics.confirmed is (self.winner is None):
            raise ValueError("Seqax residual confirmation winner contradicts statistics")
        return self


class SeqaxResidualConfirmationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str = Field(pattern=r"^passed$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)


def default_seqax_residual_confirmation_contract(
    runtime: RuntimeIdentity,
) -> SeqaxResidualConfirmationContract:
    profile = default_seqax_residual_profile_contract(runtime)
    return SeqaxResidualConfirmationContract(
        source_profile_id=SOURCE_PROFILE_ID,
        source_profile_archive_sha256=SOURCE_PROFILE_ARCHIVE_SHA256,
        source_profile_receipt_sha256=SOURCE_PROFILE_RECEIPT_SHA256,
        source_profile_result_sha256=SOURCE_PROFILE_RESULT_SHA256,
        numerical_contract_id=default_seqax_bf16_validation_contract().contract_id,
        compilation_source_root=SEQAX_RESIDUAL_PROFILE_COMPILATION_ROOT,
        baseline=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        candidate=SeqaxResidualNormStrategy.STANDARD,
        timing_seed=SEQAX_RESIDUAL_PROFILE_TIMING_SEED,
        correctness_seeds=SEQAX_PALLAS_CORRECTNESS_SEEDS,
        warmup_iterations=5,
        measured_iterations=5,
        paired_rounds=32,
        bootstrap_samples=100_000,
        confidence_level=0.99,
        minimum_practical_improvement=0.03,
        analysis_index=3,
        allow_early_stopping=False,
        allow_further_retry=False,
        candidates_resident_together=True,
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        parameters=profile.parameters,
        plans=profile.candidates,
    )


def confirmation_orders(
    contract: SeqaxResidualConfirmationContract,
) -> tuple[tuple[SeqaxResidualNormStrategy, SeqaxResidualNormStrategy], ...]:
    forward = (contract.baseline, contract.candidate)
    reverse = (contract.candidate, contract.baseline)
    return tuple(forward if index % 2 == 0 else reverse for index in range(contract.paired_rounds))


def confirmation_statistics(
    contract: SeqaxResidualConfirmationContract,
    rounds: tuple[SeqaxResidualTimingRound, ...],
) -> SeqaxResidualConfirmationStatistics:
    orders = confirmation_orders(contract)
    if len(rounds) != contract.paired_rounds * 2:
        raise ValueError("Seqax residual confirmation observation count mismatch")
    improvements = []
    for round_index, order in enumerate(orders):
        observed = tuple(value for value in rounds if value.round_index == round_index)
        if tuple(value.candidate for value in observed) != order:
            raise ValueError("Seqax residual confirmation execution order mismatch")
        if tuple(value.position for value in observed) != (0, 1):
            raise ValueError("Seqax residual confirmation positions mismatch")
        if any(len(value.samples_ns) != contract.measured_iterations for value in observed):
            raise ValueError("Seqax residual confirmation sample count mismatch")
        medians = {value.candidate: value.median_ns for value in observed}
        improvements.append(1.0 - medians[contract.candidate] / medians[contract.baseline])
    values = np.asarray(improvements, dtype=np.float64)
    bootstrap_seed = semantic_seed(
        "seqax-residual-confirmation-bootstrap-v1",
        contract.confirmation_id,
    )
    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(
        0,
        len(values),
        size=(contract.bootstrap_samples, len(values)),
    )
    bootstrap = np.median(values[indices], axis=1)
    tail = (1.0 - contract.confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap, (tail, 1.0 - tail), method="linear")
    confirmed = bool(float(lower) > contract.minimum_practical_improvement)
    return SeqaxResidualConfirmationStatistics(
        baseline=contract.baseline,
        candidate=contract.candidate,
        round_count=len(values),
        paired_improvements=tuple(float(value) for value in values),
        median_improvement=float(np.median(values)),
        mean_improvement=float(np.mean(values)),
        improvement_confidence_interval=(float(lower), float(upper)),
        confidence_level=contract.confidence_level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=contract.bootstrap_samples,
        minimum_practical_improvement=contract.minimum_practical_improvement,
        confirmed=confirmed,
    )
