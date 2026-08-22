from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ArtifactReference, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA, model_identity_sha256
from tpu_cake.seqax_large_residual import (
    SEQAX_LARGE_RESIDUAL_COMPILATION_ROOT,
    SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS,
)
from tpu_cake.seqax_numerical import (
    SeqaxBf16NumericalPolicy,
    SeqaxBf16OutputAssessment,
    default_seqax_bf16_validation_contract,
)
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy

SEQAX_LARGE_RESIDUAL_QUALIFICATION_SCHEMA = "seqax-large-residual-qualification-v2"
SEQAX_LARGE_RESIDUAL_QUALIFICATION_QUESTION = (
    "Does the pinned model-4096 Seqax workload compile with the declared native residual "
    "boundary and satisfy the frozen BF16 final-output policy before any timing is authorized?"
)
SEQAX_LARGE_RESIDUAL_CONTRACT_ID = (
    "55a9da09cc188d8a582ddad566757d2f94413d3e9449ddc30f8603cb3661d9c1"
)
SEQAX_LARGE_RESIDUAL_COMPILER_CAPTURE_RECORD_ID = (
    "da420165c7ea57ac4d5b2830062803942236a2726411c5dc9b3bcc0bb1a63ff6"
)
SEQAX_LARGE_RESIDUAL_COMPILER_CAPTURE_SOURCE_COMMIT = "9ccd3705c63ab772f8a4c325eb407ba35037f749"
SEQAX_LARGE_RESIDUAL_COMPILER_CAPTURE_UV_LOCK_SHA256 = (
    "03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9"
)
SEQAX_LARGE_RESIDUAL_QUALIFICATION_CLAIM_ROOT = (
    "/home/sudarshan/tpu-cake-evidence/seqax-large-residual-qualification-attempts-v2"
)
_SUPERSEDED_QUALIFICATION_ID = "5aaac3984ba05fcc995576b533ec82908ddb204f0a8c0ef8a0323a8504e6f341"
_SUPERSEDED_ATTEMPT_ID = "3478e675d52b0157a4d392dcb5be56f7120e97a857946898524095868009164e"
_SUPERSEDED_SOURCE_COMMIT = "2ac952f0b2a44f36c5fe283a7bc94aaf00804996"
_SUPERSEDED_FAILURE_PATH = (
    "/home/sudarshan/tpu-cake-evidence/"
    "seqax-large-residual-qualification-2ac952f-5aaac39/failure.json"
)
_SUPERSEDED_FAILURE_SHA256 = "85533374507bde0e4671d8896a231905a0d3258dc22c884fa4631f02a1797ff1"
_SUPERSEDED_LOG_PATH = (
    "/home/sudarshan/tpu-cake-evidence/seqax-large-residual-qualification-2ac952f.log"
)
_SUPERSEDED_LOG_SHA256 = "b9cc14f78f6754281f62a252de1ec16702a3237a4f0cf0200a234b205dc15cd9"
_FORENSIC_CAPTURE_PATH = (
    "/home/sudarshan/tpu-cake-evidence/"
    "seqax-large-residual-qualification-failure-forensic-2ac952f.log"
)
_FORENSIC_CAPTURE_SHA256 = "1cd8afbe7735f9046b98934621af33306433572c7fc400ed0f481d45feabdc0d"
SEQAX_LARGE_RESIDUAL_QUALIFICATION_FAILURE_RECORD_SCHEMA = (
    "seqax-large-residual-qualification-failure-record-v1"
)
_CAPTURE_INVOCATION_IDS = (
    "4206cd2d9dff42e199aeb04ebb27ef61",
    "4f85b0144dc249abbe22131b9afa2a3a",
)
_CAPTURE_LOG_SHA256 = (
    "81e9fe9e432ac46f961a48d219a470925449a9a8fbd7778f6ce7a118a45cbf6b",
    "81e9fe9e432ac46f961a48d219a470925449a9a8fbd7778f6ce7a118a45cbf6b",
)
_PARAMETERS = {
    "batch": 2,
    "data_mesh": 2,
    "feed_forward": 1024,
    "head": 64,
    "key_value_heads": 4,
    "layers": 1,
    "model": 4096,
    "query_groups": 2,
    "rope_max_timescale": 256,
    "sequence": 128,
    "tensor_mesh": 4,
    "vocabulary": 256,
}
_CANDIDATES = (
    SeqaxResidualNormStrategy.STANDARD,
    SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
)


class SeqaxLargeResidualBoundaryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chain_count: int = Field(ge=0)
    reduce_scatter_starts: tuple[str, ...]
    reduce_scatter_dones: tuple[str, ...]
    residual_fusions: tuple[str, ...]
    all_gather_starts: tuple[str, ...]

    @model_validator(mode="after")
    def exactly_two_distinct_chains(self) -> SeqaxLargeResidualBoundaryAnalysis:
        inventories = (
            self.reduce_scatter_starts,
            self.reduce_scatter_dones,
            self.residual_fusions,
            self.all_gather_starts,
        )
        if self.chain_count != 2 or any(len(values) != 2 for values in inventories):
            raise ValueError("SEQAX_LARGE_RESIDUAL_BOUNDARY_CHAIN_MISMATCH")
        if any(len(set(values)) != 2 for values in inventories):
            raise ValueError("SEQAX_LARGE_RESIDUAL_BOUNDARY_CHAIN_DUPLICATE")
        return self


def _instructions(compiler_hlo: str) -> tuple[tuple[str, str], ...]:
    instructions = []
    for line in compiler_hlo.splitlines():
        match = re.match(r"\s*%([^\s=]+)\s*=\s*(.+)$", line)
        if match is not None:
            instructions.append((match.group(1), match.group(2)))
    return tuple(instructions)


def _references(body: str, name: str) -> bool:
    return re.search(rf"%{re.escape(name)}(?=[,\s)])", body) is not None


def analyze_large_residual_boundary(
    compiler_hlo: str,
) -> SeqaxLargeResidualBoundaryAnalysis:
    instructions = _instructions(compiler_hlo)
    starts = tuple(
        (name, body)
        for name, body in instructions
        if "f32[1,128,4096]" in body
        and "f32[1,128,1024]" in body
        and "call-start(" in body
        and 'async_execution_thread="sparsecore"' in body
        and 'op_name="jit(physical_call)/shard_map/reduce_scatter"' in body
        and '"offload":"OFFLOAD_COLLECTIVE"' in body
    )
    chains = []
    for start_name, _start_body in starts:
        dones = tuple(
            (name, body)
            for name, body in instructions
            if "f32[1,128,1024]" in body
            and "call-done(" in body
            and _references(body, start_name)
            and 'op_name="jit(physical_call)/shard_map/reduce_scatter"' in body
        )
        if len(dones) != 1:
            continue
        done_name, _done_body = dones[0]
        fusions = tuple(
            (name, body)
            for name, body in instructions
            if name.startswith("convert_add_fusion")
            and "bf16[1,128,1024]" in body
            and " fusion(" in f" {body}"
            and _references(body, done_name)
            and 'op_name="jit(physical_call)/shard_map/add"' in body
        )
        if len(fusions) != 1:
            continue
        fusion_name, _fusion_body = fusions[0]
        gathers = tuple(
            (name, body)
            for name, body in instructions
            if "bf16[1,128,1024]" in body
            and "bf16[1,128,4096]" in body
            and "call-start(" in body
            and _references(body, fusion_name)
            and 'async_execution_thread="sparsecore"' in body
            and 'op_name="jit(physical_call)/shard_map/all_gather"' in body
            and '"offload":"OFFLOAD_COLLECTIVE"' in body
        )
        if len(gathers) == 1:
            chains.append((start_name, done_name, fusion_name, gathers[0][0]))
    return SeqaxLargeResidualBoundaryAnalysis(
        chain_count=len(chains),
        reduce_scatter_starts=tuple(value[0] for value in chains),
        reduce_scatter_dones=tuple(value[1] for value in chains),
        residual_fusions=tuple(value[2] for value in chains),
        all_gather_starts=tuple(value[3] for value in chains),
    )


class SeqaxLargeResidualQualificationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_LARGE_RESIDUAL_QUALIFICATION_SCHEMA] = (
        SEQAX_LARGE_RESIDUAL_QUALIFICATION_SCHEMA
    )
    identity_schema: str = SEMANTIC_IDENTITY_SCHEMA
    question: str
    large_residual_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_capture_record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_capture_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    compiler_capture_uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_capture_invocation_ids: tuple[str, str]
    compiler_capture_log_sha256: tuple[str, str]
    numerical_policy_source_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_policy: SeqaxBf16NumericalPolicy
    compilation_source_root: str
    attempt_registry_root: str
    parameters: dict[str, int]
    candidates: tuple[SeqaxResidualNormStrategy, SeqaxResidualNormStrategy]
    correctness_seeds: tuple[int, ...] = Field(min_length=5, max_length=5)
    repeat_executions: int = Field(ge=2)
    cpu_oracle_executions_per_seed: int = Field(ge=1)
    expected_standard_boundary_chains: int = Field(gt=0)
    inputs_stored_once: bool
    candidates_use_shared_resident_inputs: bool
    capture_memory_before_and_after_residency: bool
    collect_timings: bool
    collect_profile: bool
    allow_resume: bool
    allow_retry: bool
    correctness_scope: str
    compiler_hlo_replay_rule: str
    superseded_qualification_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    superseded_attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    superseded_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    superseded_failure_path: str
    superseded_failure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    superseded_log_path: str
    superseded_log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    forensic_capture_path: str
    forensic_capture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_point: str
    producer_receipt_status: str
    independent_replay_required_for_acceptance: bool
    project: str
    numeric_project_id: str
    zone: str
    hostname: str
    instance_hostname: str
    machine_type: str
    instance_id: str
    cpu_platform: str
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    mesh: dict[str, int]

    @model_validator(mode="after")
    def protocol_is_canonical(self) -> SeqaxLargeResidualQualificationContract:
        expected = default_seqax_large_residual_qualification_contract(self.runtime)
        if self.model_dump(exclude_computed_fields=True) != expected.model_dump(
            exclude_computed_fields=True
        ):
            raise ValueError("Seqax large residual qualification protocol mismatch")
        return self

    @computed_field
    @property
    def qualification_id(self) -> str:
        return model_identity_sha256(self)


class SeqaxLargeResidualQualificationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["seqax-large-residual-qualification-attempt-claim-v1"] = (
        "seqax-large-residual-qualification-attempt-claim-v1"
    )
    qualification_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal["claimed"] = "claimed"


class SeqaxLargeResidualDeviceMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    device_id: int = Field(ge=0)
    bytes_limit: int = Field(gt=0)
    bytes_in_use: int = Field(ge=0)
    peak_bytes_in_use: int = Field(ge=0)
    largest_alloc_size: int = Field(ge=0)


class SeqaxLargeResidualHost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project: str
    numeric_project_id: str
    zone: str
    hostname: str
    instance_hostname: str
    machine_type: str
    instance_id: str
    cpu_platform: str
    zone_resource: str
    machine_type_resource: str


class SeqaxLargeResidualQualificationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    seed: int
    input_sha256: tuple[str, ...] = Field(min_length=13, max_length=13)
    cpu_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_repeat_sha256: tuple[str, str]
    control_repeat_sha256: tuple[str, str]
    assessment: SeqaxBf16OutputAssessment

    @model_validator(mode="after")
    def repeats_and_policy_pass(self) -> SeqaxLargeResidualQualificationObservation:
        if (
            len(set(self.pallas_repeat_sha256)) != 1
            or len(set(self.control_repeat_sha256)) != 1
            or self.pallas_output_sha256 != self.pallas_repeat_sha256[0]
            or self.control_output_sha256 != self.control_repeat_sha256[0]
        ):
            raise ValueError("Seqax large residual qualification repeat mismatch")
        if not self.assessment.final_outputs_satisfy_policy:
            raise ValueError("Seqax large residual qualification numerical policy failed")
        return self


class SeqaxLargeResidualCrossCandidateObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: int
    standard_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    residual_all_reduce_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    standard_as_pallas_residual_as_control: SeqaxBf16OutputAssessment

    @model_validator(mode="after")
    def policy_passed(self) -> SeqaxLargeResidualCrossCandidateObservation:
        if not self.standard_as_pallas_residual_as_control.final_outputs_satisfy_policy:
            raise ValueError("Seqax large residual direct candidate comparison failed")
        return self


class SeqaxLargeResidualQualificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    qualification_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_capture_record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    host: SeqaxLargeResidualHost
    standard_boundary: SeqaxLargeResidualBoundaryAnalysis
    shared_input_sha256: tuple[tuple[str, ...], ...] = Field(min_length=5, max_length=5)
    cpu_output_sha256: tuple[str, ...] = Field(min_length=5, max_length=5)
    observations: tuple[SeqaxLargeResidualQualificationObservation, ...] = Field(
        min_length=10,
        max_length=10,
    )
    cross_candidate: tuple[SeqaxLargeResidualCrossCandidateObservation, ...] = Field(
        min_length=5,
        max_length=5,
    )
    memory_before_residency: tuple[SeqaxLargeResidualDeviceMemory, ...] = Field(
        min_length=8,
        max_length=8,
    )
    memory_after_residency: tuple[SeqaxLargeResidualDeviceMemory, ...] = Field(
        min_length=8,
        max_length=8,
    )
    producer_passed: bool
    independent_replay_performed: bool

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> SeqaxLargeResidualQualificationResult:
        expected_inventory = tuple(
            (seed, candidate)
            for seed in SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS
            for candidate in _CANDIDATES
        )
        if tuple((value.seed, value.candidate) for value in self.observations) != (
            expected_inventory
        ):
            raise ValueError("Seqax large residual qualification observation inventory mismatch")
        if tuple(value.seed for value in self.cross_candidate) != (
            SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS
        ):
            raise ValueError(
                "Seqax large residual qualification cross-candidate inventory mismatch"
            )
        for seed_index, seed in enumerate(SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS):
            pair = tuple(value for value in self.observations if value.seed == seed)
            if any(value.input_sha256 != self.shared_input_sha256[seed_index] for value in pair):
                raise ValueError("Seqax large residual qualification inputs were not shared")
            if any(value.cpu_output_sha256 != self.cpu_output_sha256[seed_index] for value in pair):
                raise ValueError("Seqax large residual qualification CPU oracle was not shared")
            cross = self.cross_candidate[seed_index]
            if (
                cross.standard_output_sha256 != pair[0].pallas_output_sha256
                or cross.residual_all_reduce_output_sha256 != pair[1].pallas_output_sha256
            ):
                raise ValueError("Seqax large residual cross-candidate output mismatch")
        expected_devices = tuple(range(8))
        if (
            tuple(value.device_id for value in self.memory_before_residency) != expected_devices
            or tuple(value.device_id for value in self.memory_after_residency) != expected_devices
        ):
            raise ValueError("Seqax large residual memory inventory mismatch")
        if not self.producer_passed or self.independent_replay_performed:
            raise ValueError("Seqax large residual qualification authority mismatch")
        return self


class SeqaxLargeResidualQualificationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    qualification_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["producer_passed_independent_replay_pending"] = (
        "producer_passed_independent_replay_pending"
    )
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    independent_replay_performed: Literal[False] = False
    immutable_archive_created: Literal[False] = False
    acceptance_authorized: Literal[False] = False


class SeqaxLargeResidualQualificationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["seqax-large-residual-qualification-failure-v1"] = (
        "seqax-large-residual-qualification-failure-v1"
    )
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    error_type: str = Field(min_length=1)
    error: str = Field(min_length=1)
    claim_consumed: Literal[True] = True
    retry_authorized: Literal[False] = False


def default_seqax_large_residual_qualification_contract(
    runtime: RuntimeIdentity,
) -> SeqaxLargeResidualQualificationContract:
    numerical = default_seqax_bf16_validation_contract()
    return SeqaxLargeResidualQualificationContract.model_construct(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        question=SEQAX_LARGE_RESIDUAL_QUALIFICATION_QUESTION,
        large_residual_contract_id=SEQAX_LARGE_RESIDUAL_CONTRACT_ID,
        compiler_capture_record_id=SEQAX_LARGE_RESIDUAL_COMPILER_CAPTURE_RECORD_ID,
        compiler_capture_source_commit=SEQAX_LARGE_RESIDUAL_COMPILER_CAPTURE_SOURCE_COMMIT,
        compiler_capture_uv_lock_sha256=SEQAX_LARGE_RESIDUAL_COMPILER_CAPTURE_UV_LOCK_SHA256,
        compiler_capture_invocation_ids=_CAPTURE_INVOCATION_IDS,
        compiler_capture_log_sha256=_CAPTURE_LOG_SHA256,
        numerical_policy_source_contract_id=numerical.contract_id,
        numerical_policy=numerical.policy,
        compilation_source_root=SEQAX_LARGE_RESIDUAL_COMPILATION_ROOT,
        attempt_registry_root=SEQAX_LARGE_RESIDUAL_QUALIFICATION_CLAIM_ROOT,
        parameters=dict(_PARAMETERS),
        candidates=_CANDIDATES,
        correctness_seeds=SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS,
        repeat_executions=2,
        cpu_oracle_executions_per_seed=1,
        expected_standard_boundary_chains=2,
        inputs_stored_once=True,
        candidates_use_shared_resident_inputs=True,
        capture_memory_before_and_after_residency=True,
        collect_timings=False,
        collect_profile=False,
        allow_resume=False,
        allow_retry=False,
        correctness_scope="final-output-plus-semantic-compiler-boundary-v2",
        compiler_hlo_replay_rule=(
            "stablehlo-exact-compiler-collectives-memory-and-boundary-lineage-v2"
        ),
        superseded_qualification_id=_SUPERSEDED_QUALIFICATION_ID,
        superseded_attempt_id=_SUPERSEDED_ATTEMPT_ID,
        superseded_source_commit=_SUPERSEDED_SOURCE_COMMIT,
        superseded_failure_path=_SUPERSEDED_FAILURE_PATH,
        superseded_failure_sha256=_SUPERSEDED_FAILURE_SHA256,
        superseded_log_path=_SUPERSEDED_LOG_PATH,
        superseded_log_sha256=_SUPERSEDED_LOG_SHA256,
        forensic_capture_path=_FORENSIC_CAPTURE_PATH,
        forensic_capture_sha256=_FORENSIC_CAPTURE_SHA256,
        claim_point=(
            "after-source-device-root-and-superseded-failure-preflight-before-input-"
            "materialization-or-compilation-v2"
        ),
        producer_receipt_status="producer_passed_independent_replay_pending",
        independent_replay_required_for_acceptance=True,
        project="astral-medley-465922-b2",
        numeric_project_id="541760035156",
        zone="us-central1-c",
        hostname="tpu-cake-v7x-rsag-wx7r",
        instance_hostname=(
            "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
        ),
        machine_type="tpu7x-standard-4t",
        instance_id="5064039476077763048",
        cpu_platform="Intel Emerald Rapids",
        runtime=runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        mesh={"d": 2, "t": 4},
    )


class SeqaxLargeResidualQualificationFailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_LARGE_RESIDUAL_QUALIFICATION_FAILURE_RECORD_SCHEMA] = (
        SEQAX_LARGE_RESIDUAL_QUALIFICATION_FAILURE_RECORD_SCHEMA
    )
    large_residual_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    failure_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    service_log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_member_count: int = Field(gt=0)
    seed: int
    candidate: Literal[SeqaxResidualNormStrategy.STANDARD]
    cpu_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_repeat_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_repeat_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment: SeqaxBf16OutputAssessment
    pallas_repeat_exact: Literal[True]
    control_repeat_exact: Literal[True]
    qualification_passed: Literal[False]
    timing_authorized: Literal[False]
    conclusion: Literal["model-4096-workload-failed-frozen-cpu-facing-policy-before-timing-v1"]

    @model_validator(mode="after")
    def record_is_canonical(self) -> SeqaxLargeResidualQualificationFailureRecord:
        expected = default_seqax_large_residual_qualification_failure_record()
        if self.model_dump(exclude_computed_fields=True) != expected.model_dump(
            exclude_computed_fields=True
        ):
            raise ValueError("Seqax large residual qualification failure record mismatch")
        return self

    @computed_field
    @property
    def record_id(self) -> str:
        return model_identity_sha256(self)


def default_seqax_large_residual_qualification_failure_record() -> (
    SeqaxLargeResidualQualificationFailureRecord
):
    return SeqaxLargeResidualQualificationFailureRecord.model_construct(
        large_residual_contract_id=SEQAX_LARGE_RESIDUAL_CONTRACT_ID,
        qualification_id=("38481a975a61e8c65b06f24020876397f0644fb5953975e4984aaf75e56fe19c"),
        attempt_id="f040ef497072fbe87a4e7b1e26929c0577ed1f92a979ac8991ade6abe807a825",
        source_commit="53a60057c47c72548d9dde0170afd194a587ea92",
        failure_artifact_sha256=(
            "a4d09f2cb301c353e224c7a3156427ef23352b95edb8a64105537e542f49997f"
        ),
        service_log_sha256=("bdc97afc1cc2ba392868a315d0a3d64976d471d58b5c7d6053ec83a7a929ae9d"),
        archive_path=(
            "/home/sudarshan/tpu-cake-evidence/"
            "seqax-large-residual-qualification-53a6005-38481a9.failed.tar.zst"
        ),
        archive_sha256=("01a322639e15c4f0a8ac973cf22f12e27463b2c728c44287f7a28f2abb64ba69"),
        archive_member_count=119,
        seed=SEQAX_LARGE_RESIDUAL_CORRECTNESS_SEEDS[0],
        candidate=SeqaxResidualNormStrategy.STANDARD,
        cpu_output_sha256=("a97e1131ea2e6599523e050e9590def8b1c8a2cf0c21de8d24dfe41c6f7d3df1"),
        pallas_output_sha256=("61ce2e73bf7fca9745b553db9f71d40ec1a4bf939c127ff868761b7889ce7686"),
        pallas_repeat_sha256=("61ce2e73bf7fca9745b553db9f71d40ec1a4bf939c127ff868761b7889ce7686"),
        control_output_sha256=("709aaf1d5743dd4160e0cda752626aa3bf0b57379460d2ead38400f0a6f5e551"),
        control_repeat_sha256=("709aaf1d5743dd4160e0cda752626aa3bf0b57379460d2ead38400f0a6f5e551"),
        assessment=SeqaxBf16OutputAssessment(
            cpu_pallas_relative_l2=0.02887471382092,
            cpu_control_relative_l2=0.028874912381819,
            cross_path_relative_l2=0.000676910067889,
            cpu_pallas_row_scaled_max=0.19160372148691,
            cpu_control_row_scaled_max=0.191603736483618,
            cross_path_row_scaled_max=0.004154127070847,
            pallas_top1_matches_cpu=False,
            control_top1_matches_cpu=False,
            pallas_top1_matches_control=True,
            final_outputs_satisfy_policy=False,
        ),
        pallas_repeat_exact=True,
        control_repeat_exact=True,
        qualification_passed=False,
        timing_authorized=False,
        conclusion="model-4096-workload-failed-frozen-cpu-facing-policy-before-timing-v1",
    )
