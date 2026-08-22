from __future__ import annotations

import math
import statistics

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256, semantic_seed
from tpu_cake.seqax_pallas_search import (
    SEQAX_PALLAS_CORRECTNESS_SEEDS,
    SEQAX_PALLAS_SEARCH_PARAMETERS,
    SEQAX_PALLAS_TIMING_SEED,
    SeqaxPallasCandidateCorrectness,
    SeqaxPallasDevice,
    SeqaxPallasRoundObservation,
)
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementName,
    SeqaxWeightPlacementPlan,
    default_seqax_weight_placement_contract,
)

SEQAX_WEIGHT_CONFIRMATION_SCHEMA = "seqax-weight-placement-confirmation-v1"
SOURCE_SEARCH_ID = "81085697c16f46649c708ef858d1d3339db71a4986bd58a550e5e536b55652f5"
SOURCE_SEARCH_RECEIPT_SHA256 = "41c86a2267a435070a087e6de89ae92925f7de318314f410c394c0212f75a347"
SOURCE_DIAGNOSTIC_ID = "b35419f66644bcf0008ac51dc7f43e25156a4cc47e40ab4d0e48196a76b74d76"
SOURCE_DIAGNOSTIC_RECEIPT_SHA256 = (
    "08f25e62c67e659e088c9ced6116045d5c3ff8959d0807179fb2508392cf3ace"
)


class SeqaxWeightConfirmationPlanIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: SeqaxWeightPlacementName
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


ACCEPTED_SEARCH_PLAN_IDENTITIES = (
    SeqaxWeightConfirmationPlanIdentity(
        candidate=SeqaxWeightPlacementName.SHARDED,
        distributed_schedule_sha256=(
            "7329886614acbc40053590195455a41ba0779247274f9a27ba6c0f999e5f650b"
        ),
        physical_schedule_sha256=(
            "7c0ae576c1223cb900c2f76914173204a4c1087e2e31de7d817e4704ff04c4d1"
        ),
        pallas_source_sha256=("e9ec2b17823a39c632ba1b468ce73fe81efa1e540806f179cefd3825b73fcf36"),
        stablehlo_sha256=("906ed9c814950bfdd560e5e7a3f315d4512076bceadf0bb6a1212d2fc3b6a6f6"),
        compiler_hlo_sha256=("005f33e40b6c8451eb144f765e06b656b23d174be05f50c4f79046657fe0c5a7"),
    ),
    SeqaxWeightConfirmationPlanIdentity(
        candidate=SeqaxWeightPlacementName.EMBEDDING_MLP,
        distributed_schedule_sha256=(
            "3557eaf671703274ca47d6a90f2190903f60e5b8eb4ef215cbcb9ae919c4297a"
        ),
        physical_schedule_sha256=(
            "2e2253b4631bbc871b7752df22d5df22d402e2d28020a73a073a6616a8a907a3"
        ),
        pallas_source_sha256=("937782206f2483eea3bf128a1c5f5db47faa0dc09ec6a221195c2f71dff0f18a"),
        stablehlo_sha256=("4ece103be808492f2155db1ea9f4cd5daed7ad88caa417f27374ba3ed85a703a"),
        compiler_hlo_sha256=("3ec6ba9308b494f4af57108fd475c1cc73f9f35b1ef4e0656cf567e774ed3c9e"),
    ),
)


class SeqaxWeightConfirmationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation_schema: str = SEQAX_WEIGHT_CONFIRMATION_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    source_search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_search_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diagnostic_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diagnostic_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_search_plans: tuple[SeqaxWeightConfirmationPlanIdentity, ...] = Field(
        min_length=2,
        max_length=2,
    )
    baseline: SeqaxWeightPlacementName
    candidate: SeqaxWeightPlacementName
    parameters: dict[str, int]
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    expected_incumbent_cpu_oracle_passed: tuple[bool, ...] = Field(
        min_length=5,
        max_length=5,
    )
    timing_seed: int
    cpu_oracle_replay_absolute_tolerance: float = Field(gt=0)
    warmup_iterations: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    paired_rounds: int = Field(ge=24)
    bootstrap_samples: int = Field(ge=10_000)
    confidence_level: float = Field(gt=0, lt=1)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    analysis_index: int = Field(ge=2)
    allow_early_stopping: bool
    allow_further_retry: bool
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxWeightConfirmationContract:
        if self.confirmation_schema != SEQAX_WEIGHT_CONFIRMATION_SCHEMA:
            raise ValueError("Seqax weight confirmation schema mismatch")
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("Seqax weight confirmation identity schema mismatch")
        provenance = (
            self.source_search_id,
            self.source_search_receipt_sha256,
            self.source_diagnostic_id,
            self.source_diagnostic_receipt_sha256,
        )
        if provenance != (
            SOURCE_SEARCH_ID,
            SOURCE_SEARCH_RECEIPT_SHA256,
            SOURCE_DIAGNOSTIC_ID,
            SOURCE_DIAGNOSTIC_RECEIPT_SHA256,
        ):
            raise ValueError("Seqax weight confirmation provenance is not canonical")
        if self.accepted_search_plans != ACCEPTED_SEARCH_PLAN_IDENTITIES:
            raise ValueError("Seqax weight confirmation search plan identities are not canonical")
        if self.parameters != SEQAX_PALLAS_SEARCH_PARAMETERS:
            raise ValueError("Seqax weight confirmation parameters are not canonical")
        if self.correctness_seeds != SEQAX_PALLAS_CORRECTNESS_SEEDS:
            raise ValueError("Seqax weight confirmation seeds are not canonical")
        if self.expected_incumbent_cpu_oracle_passed != (
            True,
            True,
            True,
            False,
            False,
        ):
            raise ValueError("Seqax weight confirmation oracle scope is not canonical")
        if self.timing_seed != SEQAX_PALLAS_TIMING_SEED:
            raise ValueError("Seqax weight confirmation timing seed is not canonical")
        if self.cpu_oracle_replay_absolute_tolerance != 2e-6:
            raise ValueError("Seqax weight confirmation oracle tolerance is not canonical")
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
        ) != (5, 5, 32, 100_000, 0.99, 0.03, 2, False, False):
            raise ValueError("Seqax weight confirmation measurement protocol is not canonical")
        if self.measured_iterations % 2 == 0 or self.paired_rounds % 2:
            raise ValueError("Seqax weight confirmation protocol must be balanced and odd-sampled")
        if (self.baseline, self.candidate) != (
            SeqaxWeightPlacementName.SHARDED,
            SeqaxWeightPlacementName.EMBEDDING_MLP,
        ):
            raise ValueError("Seqax weight confirmation candidate pair is not canonical")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax weight confirmation requires the TPU7x contract")
        return self

    @computed_field
    @property
    def confirmation_id(self) -> str:
        return model_identity_sha256(self)


class SeqaxWeightConfirmationStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: SeqaxWeightPlacementName
    candidate: SeqaxWeightPlacementName
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
    def values_are_finite(self) -> SeqaxWeightConfirmationStatistics:
        values = (
            *self.paired_improvements,
            self.median_improvement,
            self.mean_improvement,
            *self.improvement_confidence_interval,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Seqax weight confirmation statistics must be finite")
        if self.improvement_confidence_interval[0] > self.improvement_confidence_interval[1]:
            raise ValueError("Seqax weight confirmation interval is inverted")
        return self


class SeqaxWeightConfirmationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeIdentity
    devices: tuple[SeqaxPallasDevice, ...] = Field(min_length=8, max_length=8)
    timing_input_sha256: tuple[str, ...] = Field(min_length=1)
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    plans: tuple[SeqaxWeightPlacementPlan, ...] = Field(min_length=2, max_length=2)
    correctness: tuple[SeqaxPallasCandidateCorrectness, ...] = Field(
        min_length=2,
        max_length=2,
    )
    execution_orders: tuple[tuple[SeqaxWeightPlacementName, SeqaxWeightPlacementName], ...]
    rounds: tuple[SeqaxPallasRoundObservation, ...]
    post_timing_output_sha256: tuple[str, str]
    statistics: SeqaxWeightConfirmationStatistics
    winner: SeqaxWeightPlacementName | None
    correctness_scope: str = Field(pattern=r"^incumbent-bit-exact$")

    @model_validator(mode="after")
    def evidence_sets_are_consistent(self) -> SeqaxWeightConfirmationResult:
        plan_names = tuple(value.candidate for value in self.plans)
        correctness_names = tuple(value.name for value in self.correctness)
        if plan_names != correctness_names or len(set(plan_names)) != 2:
            raise ValueError("Seqax weight confirmation evidence candidate sets differ")
        if self.statistics.confirmed is (self.winner is None):
            raise ValueError("Seqax weight confirmation winner contradicts statistics")
        return self


class SeqaxWeightConfirmationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str = Field(pattern=r"^passed$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)


def default_seqax_weight_confirmation_contract(
    runtime: RuntimeIdentity,
) -> SeqaxWeightConfirmationContract:
    return SeqaxWeightConfirmationContract(
        source_search_id=SOURCE_SEARCH_ID,
        source_search_receipt_sha256=SOURCE_SEARCH_RECEIPT_SHA256,
        source_diagnostic_id=SOURCE_DIAGNOSTIC_ID,
        source_diagnostic_receipt_sha256=SOURCE_DIAGNOSTIC_RECEIPT_SHA256,
        accepted_search_plans=ACCEPTED_SEARCH_PLAN_IDENTITIES,
        baseline=SeqaxWeightPlacementName.SHARDED,
        candidate=SeqaxWeightPlacementName.EMBEDDING_MLP,
        parameters=SEQAX_PALLAS_SEARCH_PARAMETERS,
        correctness_seeds=SEQAX_PALLAS_CORRECTNESS_SEEDS,
        expected_incumbent_cpu_oracle_passed=(True, True, True, False, False),
        timing_seed=SEQAX_PALLAS_TIMING_SEED,
        cpu_oracle_replay_absolute_tolerance=2e-6,
        warmup_iterations=5,
        measured_iterations=5,
        paired_rounds=32,
        bootstrap_samples=100_000,
        confidence_level=0.99,
        minimum_practical_improvement=0.03,
        analysis_index=2,
        allow_early_stopping=False,
        allow_further_retry=False,
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
    )


def confirmation_orders(
    contract: SeqaxWeightConfirmationContract,
) -> tuple[tuple[SeqaxWeightPlacementName, SeqaxWeightPlacementName], ...]:
    pair = (contract.baseline, contract.candidate)
    reverse = (contract.candidate, contract.baseline)
    return tuple(pair if index % 2 == 0 else reverse for index in range(contract.paired_rounds))


def confirmation_statistics(
    contract: SeqaxWeightConfirmationContract,
    rounds: tuple[SeqaxPallasRoundObservation, ...],
) -> SeqaxWeightConfirmationStatistics:
    expected_orders = confirmation_orders(contract)
    if len(rounds) != contract.paired_rounds * 2:
        raise ValueError("Seqax weight confirmation observation count mismatch")
    improvements = []
    for round_index, order in enumerate(expected_orders):
        observed = tuple(value for value in rounds if value.round_index == round_index)
        if tuple(value.candidate for value in observed) != order:
            raise ValueError("Seqax weight confirmation execution order mismatch")
        if tuple(value.position for value in observed) != (0, 1):
            raise ValueError("Seqax weight confirmation positions are not canonical")
        if any(len(value.samples_ns) != contract.measured_iterations for value in observed):
            raise ValueError("Seqax weight confirmation sample count mismatch")
        if any(
            value.median_ns != float(statistics.median(value.samples_ns))
            or any(sample <= 0 for sample in value.samples_ns)
            for value in observed
        ):
            raise ValueError("Seqax weight confirmation samples are invalid")
        by_name = {value.candidate: value.median_ns for value in observed}
        improvements.append(1.0 - by_name[contract.candidate] / by_name[contract.baseline])
    values = np.asarray(improvements, dtype=np.float64)
    bootstrap_seed = semantic_seed(
        "seqax-weight-confirmation-bootstrap-v1",
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
    median = float(np.median(values))
    confirmed = bool(float(lower) > contract.minimum_practical_improvement)
    return SeqaxWeightConfirmationStatistics(
        baseline=contract.baseline,
        candidate=contract.candidate,
        round_count=len(values),
        paired_improvements=tuple(float(value) for value in values),
        median_improvement=median,
        mean_improvement=float(np.mean(values)),
        improvement_confidence_interval=(float(lower), float(upper)),
        confidence_level=contract.confidence_level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=contract.bootstrap_samples,
        minimum_practical_improvement=contract.minimum_practical_improvement,
        confirmed=confirmed,
    )


def base_weight_placement_contract(
    contract: SeqaxWeightConfirmationContract,
) -> SeqaxWeightPlacementContract:
    return default_seqax_weight_placement_contract(contract.runtime)
