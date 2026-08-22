from __future__ import annotations

import math
import statistics
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256, semantic_seed
from tpu_cake.runner import MatmulCollectiveStrategy

MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA = "matmul-collective-confirmation-v1"
MATMUL_COLLECTIVE_COMPILATION_ROOT = "/home/sudarshan/tpu-cake-main"
MATMUL_COLLECTIVE_SOURCE_REMOTE_URL = "https://github.com/sdrshn-nmbr/tpu-cake.git"
MATMUL_COLLECTIVE_HOSTNAME = "tpu-cake-v7x-rsag-wx7r"
MATMUL_COLLECTIVE_PROJECT = "astral-medley-465922-b2"
MATMUL_COLLECTIVE_ZONE = "us-central1-c"
MATMUL_COLLECTIVE_MACHINE_TYPE = "tpu7x-standard-4t"
MATMUL_COLLECTIVE_INSTANCE_ID = "5064039476077763048"
MATMUL_COLLECTIVE_NUMERIC_PROJECT_ID = "541760035156"
MATMUL_COLLECTIVE_INSTANCE_HOSTNAME = (
    "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
)
MATMUL_COLLECTIVE_CPU_PLATFORM = "Intel Emerald Rapids"
MATMUL_COLLECTIVE_ATTEMPT_REGISTRY_ROOT = (
    "/home/sudarshan/tpu-cake-evidence/matmul-collective-confirmation-attempts"
)
MATMUL_COLLECTIVE_CORRECTNESS_SEEDS = (0, 1, 2, 3, 4)
MATMUL_COLLECTIVE_TIMING_INPUT_SCHEMA = "distributed-matmul-workload/device-run/attempt-0"
MATMUL_COLLECTIVE_TIMING_LHS_SHA256 = (
    "0c134eb9045ac7593695eb1a23bd6ef471446cfd476977e7ef66bd10efe38be4"
)
MATMUL_COLLECTIVE_TIMING_RHS_SHA256 = (
    "79572770dc4e2e23c6ccc5e612a58eb000cd19f2bdbb408f3f31adc57b9d4f92"
)
MATMUL_COLLECTIVE_DIAGNOSTIC_ARCHIVE_SHA256 = (
    "0893037a5ab36637a2d62cccec7a58f6d36600318239d26b49f3188c299dc5a6"
)
MATMUL_COLLECTIVE_PARAMETERS: dict[str, int | str] = {
    "mesh_size": 8,
    "m": 1024,
    "k": 65536,
    "n": 1024,
    "tile_m": 128,
    "tile_k": 8192,
    "tile_n": 128,
    "input_dtype": "bfloat16",
    "output_dtype": "float32",
}
MATMUL_COLLECTIVE_RUNTIME = RuntimeIdentity(
    python="3.12.3",
    jax="0.11.0",
    jaxlib="0.11.0",
    libtpu="0.0.44.1",
    xla=" --xla_tpu_use_enhanced_launch_barrier=true",
)


class MatmulCollectiveDiagnosticAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy: MatmulCollectiveStrategy
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timing_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


MATMUL_COLLECTIVE_DIAGNOSTICS = (
    MatmulCollectiveDiagnosticAuthority(
        strategy=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
        source_commit="57b3229a008b98462f6317dadedbcac415894e6e",
        experiment_id="643fe052b9cabf4a0a02f2e8147558dadd9364730b2e419b8289deee53bf23a3",
        receipt_sha256="be689c3a2c8f74a280a7cb86e6d5a08244257ec0da27f2bfd7ccc4863dfabb32",
        timing_result_sha256="6d6a33187737884576abd7341c238fbf4cd0c7d036a2cdbe1bca8820e485483f",
        profile_assessment_sha256=(
            "2f3786ac6876ecd441f5f8534e6f0affd406ca5b1501f6c04bd3067b752c47c8"
        ),
        schedule_sha256="9593beb3ad1607b65f12ac02cfe4bf2adbf6279405aa307f46c155e430d40990",
        pallas_source_sha256=(
            "54e9b0f642b127e84478220ff0ba6e11de184cfcb4a279a8cbbc9b9fac70f833"
        ),
        stablehlo_sha256="a788751c6ec89c44c907b9a341a2d5ef8a6e749857fcc22091e89a6a82e18220",
        compiler_hlo_sha256=(
            "144cb09030ece96f70880dd1b12cb8324ed6cf255bf9c959f1b51b23b33e92b6"
        ),
    ),
    MatmulCollectiveDiagnosticAuthority(
        strategy=MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING,
        source_commit="57b3229a008b98462f6317dadedbcac415894e6e",
        experiment_id="b023dcd2db9e80ad897c729ccb783896e17c5a349944e5d4406f0d595fad1e19",
        receipt_sha256="11362485533e43e065adfc136d3d7a900441f56c3f1bd4b437af4b5213911702",
        timing_result_sha256="6c6824e19a777fd34986620119fccccfea2cc376efdee4940125c8ab30d9580e",
        profile_assessment_sha256=(
            "ba0352f4a664968961bca95dacae288e2fcd3ada3319285fca6456d70562fde7"
        ),
        schedule_sha256="23917a172499112cebd52bb26333f11b367f1b67190546e90af847ec8b71fe97",
        pallas_source_sha256=(
            "bb04aad3790ac0884f7e4481510bedd85e413aaba9981cfc15313ebd1484b09a"
        ),
        stablehlo_sha256="30e88595dd1d061a3b44841d2e295c7e299c3b59a3c766ad853edd9018bc28af",
        compiler_hlo_sha256=(
            "627fdb10965eece6b55fdcf1aad8243280fdc8e3b8e54d3155ce94bcedc22a13"
        ),
    ),
)


class MatmulCollectiveConfirmationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_schema: str = MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    compilation_source_root: str
    source_branch: str
    source_remote_url: str
    require_origin_main: bool
    attempt_registry_root: str
    diagnostic_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostics: tuple[MatmulCollectiveDiagnosticAuthority, ...] = Field(
        min_length=2,
        max_length=2,
    )
    baseline: MatmulCollectiveStrategy
    candidate: MatmulCollectiveStrategy
    timing_input_schema: str
    timing_lhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timing_rhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    warmup_iterations: int = Field(gt=0)
    calls_per_position: int = Field(ge=3)
    paired_rounds: int = Field(ge=24)
    bootstrap_samples: int = Field(ge=10_000)
    confidence_level: float = Field(gt=0, lt=1)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    allow_early_stopping: bool
    allow_retry: bool
    allow_outlier_removal: bool
    reuse_diagnostic_timing_samples: bool
    candidates_resident_together: bool
    numerical_reference: str
    absolute_tolerance: float = Field(gt=0)
    relative_tolerance: float = Field(gt=0)
    runtime: RuntimeIdentity
    project: str
    zone: str
    hostname: str
    machine_type: str
    instance_id: str
    numeric_project_id: str
    instance_hostname: str
    cpu_platform: str
    xla_flags: str | None
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    parameters: dict[str, int | str]

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> MatmulCollectiveConfirmationContract:
        if self.confirmation_schema != MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA:
            raise ValueError("Matmul collective confirmation schema mismatch")
        if self.identity_schema != SEMANTIC_IDENTITY_SCHEMA:
            raise ValueError("Matmul collective confirmation identity schema mismatch")
        if self.compilation_source_root != MATMUL_COLLECTIVE_COMPILATION_ROOT:
            raise ValueError("Matmul collective confirmation compilation root mismatch")
        if (self.source_branch, self.require_origin_main) != ("main", True):
            raise ValueError("Matmul collective confirmation source policy mismatch")
        if self.source_remote_url != MATMUL_COLLECTIVE_SOURCE_REMOTE_URL:
            raise ValueError("Matmul collective confirmation source remote mismatch")
        if self.attempt_registry_root != MATMUL_COLLECTIVE_ATTEMPT_REGISTRY_ROOT:
            raise ValueError("Matmul collective confirmation attempt registry mismatch")
        if self.diagnostic_archive_sha256 != MATMUL_COLLECTIVE_DIAGNOSTIC_ARCHIVE_SHA256:
            raise ValueError("Matmul collective confirmation diagnostic archive mismatch")
        if self.diagnostics != MATMUL_COLLECTIVE_DIAGNOSTICS:
            raise ValueError("Matmul collective confirmation diagnostic authority mismatch")
        if (self.baseline, self.candidate) != (
            MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
            MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING,
        ):
            raise ValueError("Matmul collective confirmation strategy pair mismatch")
        if (
            self.timing_input_schema,
            self.timing_lhs_sha256,
            self.timing_rhs_sha256,
            self.correctness_seeds,
            self.warmup_iterations,
            self.calls_per_position,
            self.paired_rounds,
            self.bootstrap_samples,
            self.confidence_level,
            self.minimum_practical_improvement,
            self.allow_early_stopping,
            self.allow_retry,
            self.allow_outlier_removal,
            self.reuse_diagnostic_timing_samples,
            self.candidates_resident_together,
            self.numerical_reference,
            self.absolute_tolerance,
            self.relative_tolerance,
        ) != (
            MATMUL_COLLECTIVE_TIMING_INPUT_SCHEMA,
            MATMUL_COLLECTIVE_TIMING_LHS_SHA256,
            MATMUL_COLLECTIVE_TIMING_RHS_SHA256,
            MATMUL_COLLECTIVE_CORRECTNESS_SEEDS,
            20,
            5,
            32,
            100_000,
            0.99,
            0.03,
            False,
            False,
            False,
            False,
            True,
            "numpy.matmul on exact BF16-quantized inputs",
            0.001,
            0.001,
        ):
            raise ValueError("Matmul collective confirmation measurement protocol mismatch")
        if self.calls_per_position % 2 == 0 or self.paired_rounds % 2:
            raise ValueError("Matmul collective confirmation protocol must be balanced")
        if (
            self.project,
            self.zone,
            self.hostname,
            self.machine_type,
            self.instance_id,
            self.numeric_project_id,
            self.instance_hostname,
            self.cpu_platform,
            self.xla_flags,
            self.backend,
            self.device_kind,
            self.device_count,
        ) != (
            MATMUL_COLLECTIVE_PROJECT,
            MATMUL_COLLECTIVE_ZONE,
            MATMUL_COLLECTIVE_HOSTNAME,
            MATMUL_COLLECTIVE_MACHINE_TYPE,
            MATMUL_COLLECTIVE_INSTANCE_ID,
            MATMUL_COLLECTIVE_NUMERIC_PROJECT_ID,
            MATMUL_COLLECTIVE_INSTANCE_HOSTNAME,
            MATMUL_COLLECTIVE_CPU_PLATFORM,
            None,
            "tpu",
            "TPU7x",
            8,
        ):
            raise ValueError("Matmul collective confirmation TPU contract mismatch")
        if self.runtime != MATMUL_COLLECTIVE_RUNTIME:
            raise ValueError("Matmul collective confirmation runtime mismatch")
        if self.parameters != MATMUL_COLLECTIVE_PARAMETERS:
            raise ValueError("Matmul collective confirmation workload mismatch")
        return self

    @computed_field
    @property
    def confirmation_id(self) -> str:
        return model_identity_sha256(self)


class MatmulCollectiveConfirmationRunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA] = (
        MATMUL_COLLECTIVE_CONFIRMATION_SCHEMA
    )
    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class MatmulCollectiveDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0)
    process_index: int = Field(ge=0)
    platform: str
    device_kind: str


class MatmulCollectiveHost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project: str
    zone: str
    hostname: str
    machine_type: str
    instance_id: str
    numeric_project_id: str
    instance_hostname: str
    cpu_platform: str
    zone_resource: str
    machine_type_resource: str


class MatmulCollectivePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy: MatmulCollectiveStrategy
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MatmulCollectiveCorrectnessObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy: MatmulCollectiveStrategy
    seed: int = Field(ge=0)
    lhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)
    passed: bool


class MatmulCollectiveTimingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy: MatmulCollectiveStrategy
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MatmulCollectiveTimingRound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    round_index: int = Field(ge=0)
    position: int = Field(ge=0, le=1)
    strategy: MatmulCollectiveStrategy
    samples_ns: tuple[int, ...] = Field(min_length=3)
    median_ns: float = Field(gt=0)

    @model_validator(mode="after")
    def samples_are_valid(self) -> MatmulCollectiveTimingRound:
        if any(value <= 0 for value in self.samples_ns):
            raise ValueError("Matmul collective confirmation samples must be positive")
        if self.median_ns != float(statistics.median(self.samples_ns)):
            raise ValueError("Matmul collective confirmation median mismatch")
        return self


class MatmulCollectiveConfirmationStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    baseline: MatmulCollectiveStrategy
    candidate: MatmulCollectiveStrategy
    round_count: int = Field(gt=0)
    paired_improvements: tuple[float, ...] = Field(min_length=1)
    median_improvement: float
    mean_improvement: float
    ab_median_improvement: float
    ba_median_improvement: float
    position_order_effect: float
    improvement_confidence_interval: tuple[float, float]
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_seed: int
    bootstrap_samples: int = Field(gt=0)
    minimum_practical_improvement: float = Field(gt=0, lt=1)
    decision: Literal["promote_candidate", "keep_baseline", "inconclusive"]
    candidate_promoted: bool
    selected_strategy: MatmulCollectiveStrategy

    @model_validator(mode="after")
    def statistics_are_valid(self) -> MatmulCollectiveConfirmationStatistics:
        values = (
            *self.paired_improvements,
            self.median_improvement,
            self.mean_improvement,
            self.ab_median_improvement,
            self.ba_median_improvement,
            self.position_order_effect,
            *self.improvement_confidence_interval,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Matmul collective confirmation statistics must be finite")
        if self.improvement_confidence_interval[0] > self.improvement_confidence_interval[1]:
            raise ValueError("Matmul collective confirmation interval is inverted")
        expected_selection = {
            "promote_candidate": self.candidate,
            "keep_baseline": self.baseline,
            "inconclusive": self.baseline,
        }[self.decision]
        if self.selected_strategy is not expected_selection or self.candidate_promoted is not (
            self.decision == "promote_candidate"
        ):
            raise ValueError("Matmul collective confirmation decision is inconsistent")
        return self


class MatmulCollectiveConfirmationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    producer_output_root: str
    diagnostic_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_receipt_sha256: tuple[str, ...] = Field(min_length=2, max_length=2)
    runtime: RuntimeIdentity
    host: MatmulCollectiveHost
    xla_flags: str | None
    devices: tuple[MatmulCollectiveDevice, ...] = Field(min_length=8, max_length=8)
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    plans: tuple[MatmulCollectivePlan, ...] = Field(min_length=2, max_length=2)
    correctness_execution_orders: tuple[
        tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy], ...
    ] = Field(min_length=5, max_length=5)
    correctness: tuple[MatmulCollectiveCorrectnessObservation, ...] = Field(
        min_length=10,
        max_length=10,
    )
    timing_lhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timing_rhs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_timing_outputs: tuple[MatmulCollectiveTimingOutput, ...] = Field(
        min_length=2,
        max_length=2,
    )
    execution_orders: tuple[
        tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy], ...
    ]
    rounds: tuple[MatmulCollectiveTimingRound, ...]
    post_timing_outputs: tuple[MatmulCollectiveTimingOutput, ...] = Field(
        min_length=2,
        max_length=2,
    )
    timing_attempt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    statistics: MatmulCollectiveConfirmationStatistics
    claim_scope: Literal[
        "mesh8-m1024-k65536-n1024-bf16-f32-tile-mn128-k8192-standalone-matmul-collective"
    ]


class MatmulCollectiveConfirmationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confirmation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["passed"] = "passed"
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)


def default_matmul_collective_confirmation_contract(
    runtime: RuntimeIdentity,
) -> MatmulCollectiveConfirmationContract:
    return MatmulCollectiveConfirmationContract(
        compilation_source_root=MATMUL_COLLECTIVE_COMPILATION_ROOT,
        source_branch="main",
        source_remote_url=MATMUL_COLLECTIVE_SOURCE_REMOTE_URL,
        require_origin_main=True,
        attempt_registry_root=MATMUL_COLLECTIVE_ATTEMPT_REGISTRY_ROOT,
        diagnostic_archive_sha256=MATMUL_COLLECTIVE_DIAGNOSTIC_ARCHIVE_SHA256,
        diagnostics=MATMUL_COLLECTIVE_DIAGNOSTICS,
        baseline=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
        candidate=MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING,
        timing_input_schema=MATMUL_COLLECTIVE_TIMING_INPUT_SCHEMA,
        timing_lhs_sha256=MATMUL_COLLECTIVE_TIMING_LHS_SHA256,
        timing_rhs_sha256=MATMUL_COLLECTIVE_TIMING_RHS_SHA256,
        correctness_seeds=MATMUL_COLLECTIVE_CORRECTNESS_SEEDS,
        warmup_iterations=20,
        calls_per_position=5,
        paired_rounds=32,
        bootstrap_samples=100_000,
        confidence_level=0.99,
        minimum_practical_improvement=0.03,
        allow_early_stopping=False,
        allow_retry=False,
        allow_outlier_removal=False,
        reuse_diagnostic_timing_samples=False,
        candidates_resident_together=True,
        numerical_reference="numpy.matmul on exact BF16-quantized inputs",
        absolute_tolerance=0.001,
        relative_tolerance=0.001,
        runtime=runtime,
        project=MATMUL_COLLECTIVE_PROJECT,
        zone=MATMUL_COLLECTIVE_ZONE,
        hostname=MATMUL_COLLECTIVE_HOSTNAME,
        machine_type=MATMUL_COLLECTIVE_MACHINE_TYPE,
        instance_id=MATMUL_COLLECTIVE_INSTANCE_ID,
        numeric_project_id=MATMUL_COLLECTIVE_NUMERIC_PROJECT_ID,
        instance_hostname=MATMUL_COLLECTIVE_INSTANCE_HOSTNAME,
        cpu_platform=MATMUL_COLLECTIVE_CPU_PLATFORM,
        xla_flags=None,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        parameters=MATMUL_COLLECTIVE_PARAMETERS,
    )


def collective_confirmation_orders(
    contract: MatmulCollectiveConfirmationContract,
) -> tuple[tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy], ...]:
    forward = (contract.baseline, contract.candidate)
    reverse = (contract.candidate, contract.baseline)
    return tuple(forward if index % 2 == 0 else reverse for index in range(contract.paired_rounds))


def collective_confirmation_statistics(
    contract: MatmulCollectiveConfirmationContract,
    rounds: tuple[MatmulCollectiveTimingRound, ...],
) -> MatmulCollectiveConfirmationStatistics:
    orders = collective_confirmation_orders(contract)
    expected_inventory = tuple(
        (round_index, position, strategy)
        for round_index, order in enumerate(orders)
        for position, strategy in enumerate(order)
    )
    observed_inventory = tuple(
        (value.round_index, value.position, value.strategy) for value in rounds
    )
    if observed_inventory != expected_inventory:
        raise ValueError("Matmul collective confirmation execution order mismatch")
    if any(len(value.samples_ns) != contract.calls_per_position for value in rounds):
        raise ValueError("Matmul collective confirmation sample count mismatch")
    improvements = []
    for round_index in range(contract.paired_rounds):
        pair = rounds[round_index * 2 : round_index * 2 + 2]
        medians = {value.strategy: value.median_ns for value in pair}
        improvements.append(1.0 - medians[contract.candidate] / medians[contract.baseline])
    values = np.asarray(improvements, dtype=np.float64)
    ab_values = values[::2]
    ba_values = values[1::2]
    ab_median = float(np.median(ab_values))
    ba_median = float(np.median(ba_values))
    bootstrap_seed = semantic_seed(
        "matmul-collective-confirmation-bootstrap-v1",
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
    threshold = contract.minimum_practical_improvement
    lower_exceeds_threshold = float(lower) > threshold and not math.isclose(
        float(lower),
        threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    upper_below_negative_threshold = float(upper) < -threshold and not math.isclose(
        float(upper),
        -threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if lower_exceeds_threshold:
        decision = "promote_candidate"
        selected = contract.candidate
    elif upper_below_negative_threshold:
        decision = "keep_baseline"
        selected = contract.baseline
    else:
        decision = "inconclusive"
        selected = contract.baseline
    return MatmulCollectiveConfirmationStatistics(
        baseline=contract.baseline,
        candidate=contract.candidate,
        round_count=len(values),
        paired_improvements=tuple(float(value) for value in values),
        median_improvement=float(np.median(values)),
        mean_improvement=float(np.mean(values)),
        ab_median_improvement=ab_median,
        ba_median_improvement=ba_median,
        position_order_effect=ab_median - ba_median,
        improvement_confidence_interval=(float(lower), float(upper)),
        confidence_level=contract.confidence_level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=contract.bootstrap_samples,
        minimum_practical_improvement=contract.minimum_practical_improvement,
        decision=decision,
        candidate_promoted=decision == "promote_candidate",
        selected_strategy=selected,
    )
