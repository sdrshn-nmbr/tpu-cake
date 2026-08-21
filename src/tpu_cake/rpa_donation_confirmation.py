from __future__ import annotations

import hashlib
import json
import math
import statistics
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, semantic_seed, semantic_sha256
from tpu_cake.rpa_surface import (
    INKLING_SHARDED_RPA_BACKEND_IMPORT_PACKAGES,
    INKLING_SHARDED_RPA_BACKEND_PYTHON_PATH,
    INKLING_SHARDED_RPA_COMPILATION_ROOT,
    InklingShardedRpaDevice,
    InklingShardedRpaPlanContract,
    default_inkling_sharded_rpa_surface_contract,
)

INKLING_RPA_DONATION_CONFIRMATION_SCHEMA = "inkling-rpa-donation-confirmation-v1"
INKLING_RPA_DONATION_CONFIRMATION_RECEIPT_SCHEMA = "inkling-rpa-donation-confirmation-receipt-v1"
INKLING_RPA_DONATION_CONFIRMATION_CLAIM_SCOPE = (
    "fixed-inkling-hq32-hkv16-d128-contexts128-512-1024-2048-"
    "upstream-query-cache-donation-resident-five-call-latency"
)
SOURCE_SURFACE_ID = "38abd645484ad1acc3f209d9076a2bdc6ea25533426c19e0b4e8cbf7ee520b17"
SOURCE_SURFACE_ARCHIVE_SHA256 = "3dfc4204040c9a537694f2bdddfc8daf74efc23b49934a517e5bbe05323ba88b"
SOURCE_SURFACE_RECEIPT_SHA256 = "c641a831d5de098323278cda2e289f98e750ebce0d1c160abcb605086aa8f4b0"
SOURCE_SURFACE_RESULT_SHA256 = "d33fac8c03bf56074498cad2cf906142a61f367ea9fcdc17c609c1b5fbd1cda8"
INKLING_RPA_DONATION_CORRECTNESS_SEEDS = tuple(
    semantic_seed(INKLING_RPA_DONATION_CONFIRMATION_SCHEMA, "correctness", str(index))
    for index in range(5)
)
INKLING_RPA_DONATION_TIMING_SEED = semantic_seed(
    INKLING_RPA_DONATION_CONFIRMATION_SCHEMA,
    "timing",
)
INKLING_RPA_INSPECTED_SURFACE_SEEDS = tuple(
    semantic_seed(f"inkling-sharded-rpa-surface-v{version}", str(index))
    for version in (1, 2, 3)
    for index in range(5)
)


class InklingRpaDonationArm(StrEnum):
    NON_DONATING = "non_donating"
    DONATING = "donating"


class InklingRpaDonationArmContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    arm: InklingRpaDonationArm
    external_donate_argnums: tuple[int, ...]
    execution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_alias_contract: Literal[
        "no-query-cache-alias",
        "query-output-aliases-arg0-cache-output-aliases-arg3",
    ]


class InklingRpaDonationHloCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    arm: InklingRpaDonationArm
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_alias_contract: Literal[
        "no-query-cache-alias",
        "query-output-aliases-arg0-cache-output-aliases-arg3",
    ]


class InklingRpaDonationHloCaptureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    runtime: RuntimeIdentity
    devices: tuple[InklingShardedRpaDevice, ...] = Field(min_length=8, max_length=8)
    captures: tuple[InklingRpaDonationHloCapture, InklingRpaDonationHloCapture]

    @model_validator(mode="after")
    def capture_order_is_exact(self) -> InklingRpaDonationHloCaptureResult:
        if tuple(value.arm for value in self.captures) != (
            InklingRpaDonationArm.NON_DONATING,
            InklingRpaDonationArm.DONATING,
        ):
            raise ValueError("RPA donation HLO capture arm order mismatch")
        return self


class InklingRpaDonationConfirmationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_schema: Literal[INKLING_RPA_DONATION_CONFIRMATION_SCHEMA]
    identity_schema: Literal[SEMANTIC_IDENTITY_SCHEMA]
    source_surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_surface_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_surface_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_surface_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compilation_source_root: Literal[INKLING_SHARDED_RPA_COMPILATION_ROOT]
    backend_python_path: Literal[INKLING_SHARDED_RPA_BACKEND_PYTHON_PATH]
    backend_import_packages: tuple[str, ...] = Field(min_length=4, max_length=4)
    hlo_identity_status: Literal["pending", "pinned"]
    baseline: Literal[InklingRpaDonationArm.NON_DONATING]
    candidate: Literal[InklingRpaDonationArm.DONATING]
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    timing_seed: int
    repeat_executions: Literal[2]
    warmup_blocks: Literal[5]
    calls_per_block: Literal[5]
    paired_rounds: Literal[32]
    bootstrap_samples: Literal[100000]
    confidence_level: Literal[0.99]
    minimum_practical_improvement: Literal[0.03]
    analysis_index: Literal[2]
    allow_early_stopping: Literal[False]
    allow_further_retry: Literal[False]
    candidates_resident_together: Literal[True]
    output_maximum_absolute_error: Literal[0.001]
    output_relative_l2_error: Literal[0.006]
    require_exact_cache: Literal[True]
    runtime: RuntimeIdentity
    backend: Literal["tpu"]
    device_kind: Literal["TPU7x"]
    device_count: Literal[8]
    process_count: Literal[1]
    producer_system: Literal["Linux"]
    producer_machine: Literal["x86_64"]
    plan: InklingShardedRpaPlanContract
    arms: tuple[InklingRpaDonationArmContract, InklingRpaDonationArmContract]
    claim_scope: Literal[INKLING_RPA_DONATION_CONFIRMATION_CLAIM_SCOPE]

    @model_validator(mode="after")
    def contract_is_canonical(self) -> InklingRpaDonationConfirmationContract:
        surface = default_inkling_sharded_rpa_surface_contract()
        if (
            self.source_surface_id,
            self.source_surface_archive_sha256,
            self.source_surface_receipt_sha256,
            self.source_surface_result_sha256,
        ) != (
            SOURCE_SURFACE_ID,
            SOURCE_SURFACE_ARCHIVE_SHA256,
            SOURCE_SURFACE_RECEIPT_SHA256,
            SOURCE_SURFACE_RESULT_SHA256,
        ):
            raise ValueError("RPA donation confirmation surface provenance mismatch")
        if self.backend_import_packages != INKLING_SHARDED_RPA_BACKEND_IMPORT_PACKAGES:
            raise ValueError("RPA donation confirmation backend packages mismatch")
        if self.correctness_seeds != INKLING_RPA_DONATION_CORRECTNESS_SEEDS:
            raise ValueError("RPA donation confirmation correctness seeds mismatch")
        if self.timing_seed != INKLING_RPA_DONATION_TIMING_SEED:
            raise ValueError("RPA donation confirmation timing seed mismatch")
        inspected = {
            *INKLING_RPA_INSPECTED_SURFACE_SEEDS,
            20260821,
            29101,
            39103,
            49109,
            59113,
            69119,
            79133,
            89137,
        }
        if inspected.intersection((*self.correctness_seeds, self.timing_seed)):
            raise ValueError("RPA donation confirmation seeds overlap prior observations")
        expected_runtime = RuntimeIdentity(
            python="3.12.3",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla=" --xla_tpu_use_enhanced_launch_barrier=true",
        )
        if self.runtime != expected_runtime:
            raise ValueError("RPA donation confirmation runtime mismatch")
        if self.plan != surface.plan:
            raise ValueError("RPA donation confirmation plan mismatch")
        if tuple(value.arm for value in self.arms) != (self.baseline, self.candidate):
            raise ValueError("RPA donation confirmation arm order mismatch")
        if self.arms[0].external_donate_argnums or self.arms[0].compiler_hlo_alias_contract != (
            "no-query-cache-alias"
        ):
            raise ValueError("RPA donation confirmation baseline contract mismatch")
        if (
            self.arms[1].external_donate_argnums != (0, 3)
            or self.arms[1].compiler_hlo_alias_contract
            != "query-output-aliases-arg0-cache-output-aliases-arg3"
        ):
            raise ValueError("RPA donation confirmation candidate contract mismatch")
        expected_execution_hashes = tuple(
            _arm_execution_sha256(surface.plan, value.external_donate_argnums)
            for value in self.arms
        )
        if tuple(value.execution_sha256 for value in self.arms) != expected_execution_hashes:
            raise ValueError("RPA donation confirmation execution identity mismatch")
        hashes = tuple(value.stablehlo_sha256 for value in self.arms)
        if self.hlo_identity_status == "pending":
            if hashes != ("0" * 64, "0" * 64):
                raise ValueError("pending RPA donation HLO identities must be zero")
        elif any(value == "0" * 64 for value in hashes) or len(set(hashes)) != 2:
            raise ValueError("pinned RPA donation HLO identities must be distinct")
        return self

    @computed_field
    @property
    def confirmation_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class InklingRpaDonationCorrectnessObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    arm: InklingRpaDonationArm
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
    cross_arm_output_exact: bool
    cross_arm_cache_exact: bool
    maximum_absolute_error: float = Field(ge=0)
    relative_l2_error: float = Field(ge=0)
    passed: bool

    @model_validator(mode="after")
    def metrics_are_finite(self) -> InklingRpaDonationCorrectnessObservation:
        if not math.isfinite(self.maximum_absolute_error) or not math.isfinite(
            self.relative_l2_error
        ):
            raise ValueError("RPA donation correctness metrics must be finite")
        return self


class InklingRpaDonationTimingRound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    round_index: int = Field(ge=0)
    position: int = Field(ge=0, le=1)
    arm: InklingRpaDonationArm
    samples_ns: tuple[int, ...] = Field(min_length=5, max_length=5)
    median_ns: float = Field(gt=0)
    terminal_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def samples_are_valid(self) -> InklingRpaDonationTimingRound:
        if any(value <= 0 for value in self.samples_ns):
            raise ValueError("RPA donation timing samples must be positive")
        if self.median_ns != float(statistics.median(self.samples_ns)):
            raise ValueError("RPA donation timing median mismatch")
        return self


class InklingRpaDonationStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    baseline: Literal[InklingRpaDonationArm.NON_DONATING]
    candidate: Literal[InklingRpaDonationArm.DONATING]
    round_count: Literal[32]
    paired_improvements: tuple[float, ...] = Field(min_length=32, max_length=32)
    median_improvement: float
    mean_improvement: float
    positive_rounds: int = Field(ge=0, le=32)
    improvement_confidence_interval: tuple[float, float]
    confidence_level: Literal[0.99]
    bootstrap_seed: int
    bootstrap_samples: Literal[100000]
    minimum_practical_improvement: Literal[0.03]
    confirmed: bool

    @model_validator(mode="after")
    def statistics_are_valid(self) -> InklingRpaDonationStatistics:
        values = (
            *self.paired_improvements,
            self.median_improvement,
            self.mean_improvement,
            *self.improvement_confidence_interval,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("RPA donation statistics must be finite")
        if self.positive_rounds != sum(value > 0 for value in self.paired_improvements):
            raise ValueError("RPA donation positive-round count mismatch")
        if self.improvement_confidence_interval[0] > self.improvement_confidence_interval[1]:
            raise ValueError("RPA donation confidence interval is inverted")
        return self


class InklingRpaDonationConfirmationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_surface_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    runtime: RuntimeIdentity
    devices: tuple[InklingShardedRpaDevice, ...] = Field(min_length=8, max_length=8)
    plan: InklingShardedRpaPlanContract
    correctness: tuple[InklingRpaDonationCorrectnessObservation, ...] = Field(
        min_length=10,
        max_length=10,
    )
    timing_input_sha256: tuple[str, ...] = Field(min_length=11, max_length=11)
    execution_orders: tuple[tuple[InklingRpaDonationArm, InklingRpaDonationArm], ...] = Field(
        min_length=32,
        max_length=32,
    )
    rounds: tuple[InklingRpaDonationTimingRound, ...] = Field(min_length=64, max_length=64)
    statistics: InklingRpaDonationStatistics
    winner: InklingRpaDonationArm | None
    accepted: bool
    claim_scope: Literal[INKLING_RPA_DONATION_CONFIRMATION_CLAIM_SCOPE]

    @model_validator(mode="after")
    def result_is_consistent(self) -> InklingRpaDonationConfirmationResult:
        expected = tuple(
            (arm, seed)
            for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
            for seed in INKLING_RPA_DONATION_CORRECTNESS_SEEDS
        )
        observed = tuple((value.arm, value.seed) for value in self.correctness)
        if observed != expected:
            raise ValueError("RPA donation correctness inventory mismatch")
        if self.statistics.confirmed != (self.winner is InklingRpaDonationArm.DONATING):
            raise ValueError("RPA donation winner contradicts statistics")
        if self.accepted != self.statistics.confirmed:
            raise ValueError("RPA donation acceptance contradicts statistics")
        return self


class InklingRpaDonationConfirmationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    receipt_schema: Literal[INKLING_RPA_DONATION_CONFIRMATION_RECEIPT_SCHEMA]
    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_count: int = Field(gt=0)
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    accepted: bool
    claim_scope: Literal[INKLING_RPA_DONATION_CONFIRMATION_CLAIM_SCOPE]

    @model_validator(mode="after")
    def artifacts_are_exact(self) -> InklingRpaDonationConfirmationReceipt:
        paths = tuple(value.path for value in self.artifacts)
        if self.artifact_count != len(paths) or tuple(sorted(paths)) != paths:
            raise ValueError("RPA donation receipt artifact inventory mismatch")
        if len(paths) != len(set(paths)):
            raise ValueError("RPA donation receipt artifact paths must be unique")
        return self


def _arm_execution_sha256(
    plan: InklingShardedRpaPlanContract,
    donate_argnums: tuple[int, ...],
) -> str:
    return semantic_sha256(
        "inkling-rpa-donation-arm-execution-v1",
        plan.schedule_sha256,
        plan.execution_sha256,
        repr(donate_argnums),
    )


def default_inkling_rpa_donation_confirmation_contract() -> InklingRpaDonationConfirmationContract:
    surface = default_inkling_sharded_rpa_surface_contract()
    return InklingRpaDonationConfirmationContract(
        confirmation_schema=INKLING_RPA_DONATION_CONFIRMATION_SCHEMA,
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        source_surface_id=SOURCE_SURFACE_ID,
        source_surface_archive_sha256=SOURCE_SURFACE_ARCHIVE_SHA256,
        source_surface_receipt_sha256=SOURCE_SURFACE_RECEIPT_SHA256,
        source_surface_result_sha256=SOURCE_SURFACE_RESULT_SHA256,
        compilation_source_root=INKLING_SHARDED_RPA_COMPILATION_ROOT,
        backend_python_path=INKLING_SHARDED_RPA_BACKEND_PYTHON_PATH,
        backend_import_packages=INKLING_SHARDED_RPA_BACKEND_IMPORT_PACKAGES,
        hlo_identity_status="pending",
        baseline=InklingRpaDonationArm.NON_DONATING,
        candidate=InklingRpaDonationArm.DONATING,
        correctness_seeds=INKLING_RPA_DONATION_CORRECTNESS_SEEDS,
        timing_seed=INKLING_RPA_DONATION_TIMING_SEED,
        repeat_executions=2,
        warmup_blocks=5,
        calls_per_block=5,
        paired_rounds=32,
        bootstrap_samples=100000,
        confidence_level=0.99,
        minimum_practical_improvement=0.03,
        analysis_index=2,
        allow_early_stopping=False,
        allow_further_retry=False,
        candidates_resident_together=True,
        output_maximum_absolute_error=0.001,
        output_relative_l2_error=0.006,
        require_exact_cache=True,
        runtime=surface.runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        process_count=1,
        producer_system="Linux",
        producer_machine="x86_64",
        plan=surface.plan,
        arms=(
            InklingRpaDonationArmContract(
                arm=InklingRpaDonationArm.NON_DONATING,
                external_donate_argnums=(),
                execution_sha256=_arm_execution_sha256(surface.plan, ()),
                stablehlo_sha256="0" * 64,
                compiler_hlo_alias_contract="no-query-cache-alias",
            ),
            InklingRpaDonationArmContract(
                arm=InklingRpaDonationArm.DONATING,
                external_donate_argnums=(0, 3),
                execution_sha256=_arm_execution_sha256(surface.plan, (0, 3)),
                stablehlo_sha256="0" * 64,
                compiler_hlo_alias_contract=("query-output-aliases-arg0-cache-output-aliases-arg3"),
            ),
        ),
        claim_scope=INKLING_RPA_DONATION_CONFIRMATION_CLAIM_SCOPE,
    )


def donation_confirmation_orders(
    contract: InklingRpaDonationConfirmationContract,
) -> tuple[tuple[InklingRpaDonationArm, InklingRpaDonationArm], ...]:
    forward = (contract.baseline, contract.candidate)
    reverse = (contract.candidate, contract.baseline)
    return tuple(forward if index % 2 == 0 else reverse for index in range(contract.paired_rounds))


def donation_confirmation_statistics(
    contract: InklingRpaDonationConfirmationContract,
    rounds: tuple[InklingRpaDonationTimingRound, ...],
) -> InklingRpaDonationStatistics:
    orders = donation_confirmation_orders(contract)
    if len(rounds) != contract.paired_rounds * 2:
        raise ValueError("RPA donation timing observation count mismatch")
    improvements = []
    for round_index, order in enumerate(orders):
        observed = tuple(value for value in rounds if value.round_index == round_index)
        if tuple(value.arm for value in observed) != order:
            raise ValueError("RPA donation timing execution order mismatch")
        if tuple(value.position for value in observed) != (0, 1):
            raise ValueError("RPA donation timing position mismatch")
        if any(len(value.samples_ns) != contract.calls_per_block for value in observed):
            raise ValueError("RPA donation timing sample count mismatch")
        if (
            len({(value.terminal_output_sha256, value.terminal_cache_sha256) for value in observed})
            != 1
        ):
            raise ValueError("RPA donation timing terminal states differ")
        medians = {value.arm: value.median_ns for value in observed}
        improvements.append(1.0 - medians[contract.candidate] / medians[contract.baseline])
    values = np.asarray(improvements, dtype=np.float64)
    bootstrap_seed = semantic_seed(
        "inkling-rpa-donation-confirmation-bootstrap-v1",
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
    return InklingRpaDonationStatistics(
        baseline=contract.baseline,
        candidate=contract.candidate,
        round_count=contract.paired_rounds,
        paired_improvements=tuple(float(value) for value in values),
        median_improvement=float(np.median(values)),
        mean_improvement=float(np.mean(values)),
        positive_rounds=int(np.count_nonzero(values > 0)),
        improvement_confidence_interval=(float(lower), float(upper)),
        confidence_level=contract.confidence_level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=contract.bootstrap_samples,
        minimum_practical_improvement=contract.minimum_practical_improvement,
        confirmed=confirmed,
    )
