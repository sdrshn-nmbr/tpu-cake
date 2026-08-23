from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis, CompilerExecutableAnalysis
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity, SourceFileContract
from tpu_cake.identity import json_sha256
from tpu_cake.ledger import RunState
from tpu_cake.seqax_contract_types import SeqaxFeedForwardFusion
from tpu_cake.seqax_silu_fusion import SeqaxSiluFusionDesignContract

_VECTOR_SHAPE = "bf16[128,1,1024]"
_STRICT_KERNELS = {
    "seqax_strict_bf16_silu",
    "seqax_strict_bf16_multiply",
    "seqax_strict_bf16_silu_multiply",
}
_COMPUTATION_REFERENCES = (
    "to_apply",
    "calls",
    "condition",
    "body",
    "fused_computation",
)
SEQAX_SILU_FUSION_COMPILER_CLAIM_SCHEMA = "seqax-silu-fusion-compiler-claim-v1"
SEQAX_SILU_FUSION_COMPILER_CAPTURE_SCHEMA = "seqax-silu-fusion-compiler-capture-v1"
SEQAX_SILU_FUSION_COMPILER_RECEIPT_SCHEMA = "seqax-silu-fusion-compiler-receipt-v1"
SEQAX_SILU_FUSION_COMPILER_FAILURE_RECEIPT_SCHEMA = "seqax-silu-fusion-compiler-failure-receipt-v1"
SEQAX_SILU_FUSION_COMPILER_FAILURE_REPLAY_SEAL_SCHEMA = (
    "seqax-silu-fusion-compiler-failure-replay-seal-v1"
)
SEQAX_SILU_FUSION_COMPILER_PAIR_SCHEMA = "seqax-silu-fusion-compiler-pair-v1"
SEQAX_SILU_FUSION_COMPILER_REPLAY_SEAL_SCHEMA = "seqax-silu-fusion-compiler-replay-seal-v1"
SEQAX_SILU_FUSION_COMPILER_ARTIFACT_ROLES = frozenset(
    {
        ArtifactRole.EXPERIMENT,
        ArtifactRole.DISTRIBUTED_IR,
        ArtifactRole.PHYSICAL_IR,
        ArtifactRole.PALLAS_SOURCE,
        ArtifactRole.PLAN_MANIFEST,
        ArtifactRole.STABLEHLO,
        ArtifactRole.COMPILER_HLO,
        ArtifactRole.COMPILER_ANALYSIS,
        ArtifactRole.COST_MODEL,
        ArtifactRole.EXECUTION_LEDGER,
        ArtifactRole.INVOCATION,
        ArtifactRole.SOURCE_STATE,
    }
)


def _semantic_instruction_body(body: str) -> str:
    return re.split(
        r",\s*(?:metadata|frontend_attributes|backend_config|sharding|control-predecessors)=",
        body,
        maxsplit=1,
    )[0]


class SeqaxSiluFusionCompilerSourceAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch: Literal["main"]
    origin_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_url: str
    source_root: str
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cli_sha256: Literal["355040b20f7e48683811b009fc77f460652617fafcdc44c68a3d7309fd71f740"]
    design_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pair_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...] = Field(min_length=1)
    runtime: RuntimeIdentity

    @model_validator(mode="after")
    def main_is_exact(self) -> SeqaxSiluFusionCompilerSourceAuthority:
        if (
            self.source_commit != self.origin_main_commit
            or self.source_commit != self.remote_main_commit
        ):
            raise ValueError("SEQAX_SILU_FUSION_SOURCE_MAIN_MISMATCH")
        paths = tuple(value.path for value in self.source_manifest)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("SEQAX_SILU_FUSION_SOURCE_MANIFEST_INVALID")
        return self


class SeqaxSiluFusionCompilerHostIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project: str
    numeric_project_id: str
    zone: str
    hostname: str
    instance_hostname: str
    machine_type: str
    instance_id: str
    cpu_platform: str


class SeqaxSiluFusionCompilerDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0, le=7)
    process_index: Literal[0]
    platform: Literal["tpu"]
    device_kind: Literal["TPU7x"]


class SeqaxSiluFusionCompilerAttemptClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_COMPILER_CLAIM_SCHEMA] = (
        SEQAX_SILU_FUSION_COMPILER_CLAIM_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_ordinal: int = Field(ge=0, le=1)
    invocation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    output_root: str

    @computed_field
    @property
    def claim_id(self) -> str:
        return json_sha256(self.model_dump(exclude_computed_fields=True, mode="json"))


class SeqaxSiluFusionCompilerCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxFeedForwardFusion
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_optimization_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_analysis: CompilerExecutableAnalysis
    reachable_collectives: CompilerCollectiveAnalysis
    fusion_analysis: SeqaxSiluFusionCompilerAnalysis
    buffer_assignment_size_bytes: int = Field(ge=0)
    buffer_assignment_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    allocated_vmem_bytes_per_device: int = Field(gt=0)
    peak_live_vmem_bytes_per_device: int = Field(gt=0)
    ring_equivalent_ici_bytes_per_device: int = Field(gt=0)

    @model_validator(mode="after")
    def buffer_assignment_is_bound(self) -> SeqaxSiluFusionCompilerCandidate:
        memory = self.compiler_analysis.memory
        if (
            memory.buffer_assignment_size_bytes != self.buffer_assignment_size_bytes
            or memory.buffer_assignment_sha256 != self.buffer_assignment_sha256
            or self.fusion_analysis.candidate is not self.candidate
        ):
            raise ValueError("SEQAX_SILU_FUSION_BUFFER_ASSIGNMENT_MISMATCH")
        return self

    @computed_field
    @property
    def semantic_id(self) -> str:
        return json_sha256(
            {
                "candidate": self.candidate,
                "distributed_schedule_sha256": self.distributed_schedule_sha256,
                "physical_schedule_sha256": self.physical_schedule_sha256,
                "pallas_source_sha256": self.pallas_source_sha256,
                "pallas_manifest_sha256": self.pallas_manifest_sha256,
                "reachable_collectives": self.reachable_collectives.model_dump(mode="json"),
                "fusion_semantic_id": self.fusion_analysis.semantic_id,
                "allocated_vmem_bytes_per_device": self.allocated_vmem_bytes_per_device,
                "peak_live_vmem_bytes_per_device": self.peak_live_vmem_bytes_per_device,
                "ring_equivalent_ici_bytes_per_device": (self.ring_equivalent_ici_bytes_per_device),
            }
        )


class SeqaxSiluFusionCompilerCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_COMPILER_CAPTURE_SCHEMA] = (
        SEQAX_SILU_FUSION_COMPILER_CAPTURE_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_ordinal: int = Field(ge=0, le=1)
    invocation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: SeqaxSiluFusionCompilerSourceAuthority
    host: SeqaxSiluFusionCompilerHostIdentity
    worker_environment: dict[str, str]
    compiler_environment: dict[str, str]
    source_import_root: str
    compile_input_mode: Literal["abstract-only"]
    devices: tuple[SeqaxSiluFusionCompilerDevice, ...] = Field(min_length=8, max_length=8)
    worker_pid: int = Field(gt=0)
    worker_nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    candidates: tuple[SeqaxSiluFusionCompilerCandidate, SeqaxSiluFusionCompilerCandidate]
    model_outputs_executed: Literal[False]
    correctness_outputs_collected: Literal[False]
    timing_collected: Literal[False]
    profile_collected: Literal[False]

    @model_validator(mode="after")
    def compile_scope_is_exact(self) -> SeqaxSiluFusionCompilerCapture:
        if tuple(device.id for device in self.devices) != tuple(range(8)):
            raise ValueError("SEQAX_SILU_FUSION_DEVICE_ORDER_MISMATCH")
        if tuple(value.candidate for value in self.candidates) != (
            SeqaxFeedForwardFusion.SEPARATE,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
        ):
            raise ValueError("SEQAX_SILU_FUSION_CANDIDATE_ORDER_MISMATCH")
        if self.candidates[0].reachable_collectives != self.candidates[1].reachable_collectives:
            raise ValueError("SEQAX_SILU_FUSION_COLLECTIVE_PARITY_MISMATCH")
        return self

    @computed_field
    @property
    def capture_id(self) -> str:
        return json_sha256(self.model_dump(exclude_computed_fields=True, mode="json"))

    @computed_field
    @property
    def semantic_pair_id(self) -> str:
        return json_sha256([value.semantic_id for value in self.candidates])


class SeqaxSiluFusionCompilerWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claim: SeqaxSiluFusionCompilerAttemptClaim
    design: SeqaxSiluFusionDesignContract
    source: SeqaxSiluFusionCompilerSourceAuthority

    def wire_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_computed_fields=True)


class SeqaxSiluFusionCompilerWorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capture: SeqaxSiluFusionCompilerCapture

    def wire_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_computed_fields=True)


class SeqaxSiluFusionCompilerWorkerFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    returncode: int
    stdout: str
    stderr: str
    worker_process_started: Literal[True]
    model_outputs_executed: Literal[False]
    correctness_outputs_collected: Literal[False]
    timing_collected: Literal[False]
    profile_collected: Literal[False]

    @model_validator(mode="after")
    def process_failed_with_diagnostic(self) -> SeqaxSiluFusionCompilerWorkerFailure:
        if self.returncode == 0 or not (self.stdout or self.stderr):
            raise ValueError("SEQAX_SILU_FUSION_WORKER_FAILURE_INVALID")
        return self


class SeqaxSiluFusionCompilerFailureReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_COMPILER_FAILURE_RECEIPT_SCHEMA] = (
        SEQAX_SILU_FUSION_COMPILER_FAILURE_RECEIPT_SCHEMA
    )
    claim: SeqaxSiluFusionCompilerAttemptClaim
    source: SeqaxSiluFusionCompilerSourceAuthority
    final_ledger_state: Literal[RunState.CREATED, RunState.VERIFIED, RunState.LOWERED]
    failure: SeqaxSiluFusionCompilerWorkerFailure
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    independent_replay_required: Literal[True]
    independent_replay_performed_at_receipt_creation: Literal[False]
    retry_authorized: Literal[False]
    ordinal_one_launched: bool

    @model_validator(mode="after")
    def artifacts_are_failed_compile_only(self) -> SeqaxSiluFusionCompilerFailureReceipt:
        if any(
            value.role not in SEQAX_SILU_FUSION_COMPILER_ARTIFACT_ROLES for value in self.artifacts
        ) or any(
            value.path in {"receipt.json", "worker-result.json", "failure-receipt.json"}
            for value in self.artifacts
        ):
            raise ValueError("SEQAX_SILU_FUSION_FAILURE_ARTIFACT_INVALID")
        if self.ordinal_one_launched != (self.claim.capture_ordinal == 1):
            raise ValueError("SEQAX_SILU_FUSION_FAILURE_ORDINAL_MISMATCH")
        return self

    @computed_field
    @property
    def failure_receipt_id(self) -> str:
        return json_sha256(self.model_dump(exclude_computed_fields=True, mode="json"))


class SeqaxSiluFusionCompilerFailureReplaySeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_COMPILER_FAILURE_REPLAY_SEAL_SCHEMA] = (
        SEQAX_SILU_FUSION_COMPILER_FAILURE_REPLAY_SEAL_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_ordinal: int = Field(ge=0, le=1)
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_receipt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    output_root: str
    independent_replay_performed: Literal[True]
    retry_authorized: Literal[False]
    ordinal_one_launched: bool

    @model_validator(mode="after")
    def ordinal_matches_launch_state(self) -> SeqaxSiluFusionCompilerFailureReplaySeal:
        if self.ordinal_one_launched != (self.capture_ordinal == 1):
            raise ValueError("SEQAX_SILU_FUSION_FAILURE_REPLAY_ORDINAL_MISMATCH")
        return self

    @computed_field
    @property
    def failure_replay_seal_id(self) -> str:
        return json_sha256(self.model_dump(exclude_computed_fields=True, mode="json"))


class SeqaxSiluFusionCompilerReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_COMPILER_RECEIPT_SCHEMA] = (
        SEQAX_SILU_FUSION_COMPILER_RECEIPT_SCHEMA
    )
    capture: SeqaxSiluFusionCompilerCapture
    final_ledger_state: Literal[RunState.COMPILED]
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def artifacts_are_compile_only(self) -> SeqaxSiluFusionCompilerReceipt:
        if any(
            value.role not in SEQAX_SILU_FUSION_COMPILER_ARTIFACT_ROLES for value in self.artifacts
        ):
            raise ValueError("SEQAX_SILU_FUSION_NONCOMPILER_ARTIFACT")
        return self

    @computed_field
    @property
    def receipt_id(self) -> str:
        return json_sha256(self.model_dump(exclude_computed_fields=True, mode="json"))


class SeqaxSiluFusionCompilerReplaySeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_COMPILER_REPLAY_SEAL_SCHEMA] = (
        SEQAX_SILU_FUSION_COMPILER_REPLAY_SEAL_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_ordinal: int = Field(ge=0, le=1)
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_pair_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    output_root: str
    independent_replay_performed: Literal[True]

    @computed_field
    @property
    def replay_seal_id(self) -> str:
        return json_sha256(self.model_dump(exclude_computed_fields=True, mode="json"))


class SeqaxSiluFusionCompilerPairMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capture_root: str
    capture_ordinal: int = Field(ge=0, le=1)
    capture_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_seal_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    worker_pid: int = Field(gt=0)
    worker_nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_authority_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    host_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_environment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_environment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    device_inventory_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_pair_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_semantic_ids: tuple[str, str]

    @model_validator(mode="after")
    def root_is_canonical(self) -> SeqaxSiluFusionCompilerPairMember:
        root = Path(self.capture_root)
        if (
            not root.is_absolute()
            or root.parent != Path("/home/sudarshan/tpu-cake-evidence")
            or re.fullmatch(
                rf"seqax-silu-fusion-compiler-[0-9a-f]{{7}}-{self.capture_ordinal}-[0-9a-f]{{8}}",
                root.name,
            )
            is None
        ):
            raise ValueError("SEQAX_SILU_FUSION_PAIR_CAPTURE_ROOT_INVALID")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.candidate_semantic_ids
        ):
            raise ValueError("SEQAX_SILU_FUSION_PAIR_CANDIDATE_ID_INVALID")
        return self


class SeqaxSiluFusionCompilerPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_SILU_FUSION_COMPILER_PAIR_SCHEMA] = (
        SEQAX_SILU_FUSION_COMPILER_PAIR_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    captures: tuple[
        SeqaxSiluFusionCompilerPairMember,
        SeqaxSiluFusionCompilerPairMember,
    ]
    independent_replay_performed: Literal[True]
    model_outputs_executed: Literal[False]
    correctness_outputs_collected: Literal[False]
    timing_collected: Literal[False]
    profile_collected: Literal[False]

    @model_validator(mode="after")
    def captures_are_independent_repetitions(self) -> SeqaxSiluFusionCompilerPair:
        if tuple(value.capture_ordinal for value in self.captures) != (0, 1):
            raise ValueError("SEQAX_SILU_FUSION_PAIR_ORDINAL_MISMATCH")
        if any(
            not Path(value.capture_root).name.startswith(
                f"seqax-silu-fusion-compiler-{self.design_id[:7]}-{value.capture_ordinal}-"
            )
            for value in self.captures
        ):
            raise ValueError("SEQAX_SILU_FUSION_PAIR_DESIGN_ROOT_MISMATCH")
        unique_fields = (
            "capture_root",
            "capture_id",
            "receipt_id",
            "replay_seal_id",
            "claim_id",
            "invocation_id",
            "worker_nonce",
        )
        if any(
            len({getattr(value, field) for value in self.captures}) != 2 for field in unique_fields
        ):
            raise ValueError("SEQAX_SILU_FUSION_PAIR_INDEPENDENCE_MISMATCH")
        shared_fields = (
            "source_commit",
            "source_tree",
            "source_authority_id",
            "host_id",
            "compiler_environment_id",
            "worker_environment_id",
            "device_inventory_id",
            "semantic_pair_id",
            "candidate_semantic_ids",
        )
        if any(
            len({getattr(value, field) for value in self.captures}) != 1 for field in shared_fields
        ):
            raise ValueError("SEQAX_SILU_FUSION_PAIR_SEMANTIC_MISMATCH")
        return self

    @computed_field
    @property
    def pair_id(self) -> str:
        return json_sha256(self.model_dump(exclude_computed_fields=True, mode="json"))


def seqax_silu_fusion_compiler_pair_member(
    root: Path,
    receipt: SeqaxSiluFusionCompilerReceipt,
    replay_seal: SeqaxSiluFusionCompilerReplaySeal,
) -> SeqaxSiluFusionCompilerPairMember:
    capture = receipt.capture
    if (
        replay_seal.design_id != capture.design_id
        or replay_seal.capture_ordinal != capture.capture_ordinal
        or replay_seal.claim_id != capture.claim_id
        or replay_seal.capture_id != capture.capture_id
        or replay_seal.receipt_id != receipt.receipt_id
        or replay_seal.semantic_pair_id != capture.semantic_pair_id
        or replay_seal.source_commit != capture.source.source_commit
        or replay_seal.source_tree != capture.source.source_tree
        or replay_seal.output_root != str(root)
    ):
        raise ValueError("SEQAX_SILU_FUSION_REPLAY_SEAL_MISMATCH")
    return SeqaxSiluFusionCompilerPairMember(
        capture_root=str(root),
        capture_ordinal=capture.capture_ordinal,
        capture_id=capture.capture_id,
        receipt_id=receipt.receipt_id,
        replay_seal_id=replay_seal.replay_seal_id,
        claim_id=capture.claim_id,
        invocation_id=capture.invocation_id,
        worker_pid=capture.worker_pid,
        worker_nonce=capture.worker_nonce,
        source_commit=capture.source.source_commit,
        source_tree=capture.source.source_tree,
        source_authority_id=json_sha256(capture.source.model_dump(mode="json")),
        host_id=json_sha256(capture.host.model_dump(mode="json")),
        compiler_environment_id=json_sha256(capture.compiler_environment),
        worker_environment_id=json_sha256(capture.worker_environment),
        device_inventory_id=json_sha256(
            [value.model_dump(mode="json") for value in capture.devices]
        ),
        semantic_pair_id=capture.semantic_pair_id,
        candidate_semantic_ids=tuple(value.semantic_id for value in capture.candidates),
    )


class SeqaxSiluFusionCompilerCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ordinal: int = Field(ge=0, le=1)
    kernel: str
    output_shape: str
    operand_count: int = Field(ge=1, le=2)
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector_region_index: int = Field(ge=0, le=1)
    implementation: str
    instruction_name: str = Field(min_length=1)
    operand_names: tuple[str, ...] = Field(min_length=1, max_length=2)


class SeqaxSiluFusionCompilerAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxFeedForwardFusion
    strict_vector_call_count: int = Field(ge=1, le=2)
    calls: tuple[SeqaxSiluFusionCompilerCall, ...] = Field(min_length=1, max_length=2)
    all_strict_vector_calls_are_live: bool
    gate_and_up_projection_lineages_are_distinct: bool
    silu_output_feeds_multiply: bool
    vector_output_feeds_one_down_projection: bool

    @model_validator(mode="after")
    def boundary_is_exact(self) -> SeqaxSiluFusionCompilerAnalysis:
        kernels = tuple(call.kernel for call in self.calls)
        expected = (
            ("seqax_strict_bf16_silu", "seqax_strict_bf16_multiply")
            if self.candidate is SeqaxFeedForwardFusion.SEPARATE
            else ("seqax_strict_bf16_silu_multiply",)
        )
        expected_operand_counts = (
            (1, 2) if self.candidate is SeqaxFeedForwardFusion.SEPARATE else (2,)
        )
        if (
            kernels != expected
            or tuple(call.operand_count for call in self.calls) != expected_operand_counts
            or self.strict_vector_call_count != len(expected)
            or tuple(call.ordinal for call in self.calls) != tuple(range(len(expected)))
            or any(call.output_shape != _VECTOR_SHAPE for call in self.calls)
            or not self.all_strict_vector_calls_are_live
            or not self.gate_and_up_projection_lineages_are_distinct
            or not self.vector_output_feeds_one_down_projection
        ):
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_BOUNDARY_MISMATCH")
        if (self.candidate is SeqaxFeedForwardFusion.SEPARATE) != self.silu_output_feeds_multiply:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_DEPENDENCY_MISMATCH")
        return self

    @computed_field
    @property
    def semantic_id(self) -> str:
        return json_sha256(
            {
                "candidate": self.candidate,
                "kernels": [
                    {
                        "kernel": call.kernel,
                        "output_shape": call.output_shape,
                        "operand_count": call.operand_count,
                        "schedule_sha256": call.schedule_sha256,
                        "vector_region_index": call.vector_region_index,
                        "implementation": call.implementation,
                    }
                    for call in self.calls
                ],
                "all_strict_vector_calls_are_live": self.all_strict_vector_calls_are_live,
                "gate_and_up_projection_lineages_are_distinct": (
                    self.gate_and_up_projection_lineages_are_distinct
                ),
                "silu_output_feeds_multiply": self.silu_output_feeds_multiply,
                "vector_output_feeds_one_down_projection": (
                    self.vector_output_feeds_one_down_projection
                ),
            }
        )


class _Instruction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    body: str
    root: bool

    @property
    def references(self) -> tuple[str, ...]:
        body = _semantic_instruction_body(self.body)
        operation = re.search(r"\s+[A-Za-z][A-Za-z0-9.-]*\(", body)
        if operation is None:
            return ()
        start = operation.end() - 1
        depth = 0
        end = None
        for index, character in enumerate(body[start:], start=start):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_OPERAND_EXPRESSION_INVALID")
        return tuple(re.findall(r"%([A-Za-z0-9_.$-]+)", body[start : end + 1]))


class _Computation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    entry: bool
    instructions: tuple[_Instruction, ...]

    @property
    def by_name(self) -> dict[str, _Instruction]:
        return {instruction.name: instruction for instruction in self.instructions}

    @property
    def root(self) -> _Instruction:
        roots = tuple(instruction for instruction in self.instructions if instruction.root)
        if len(roots) != 1:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_ROOT_MISMATCH")
        return roots[0]


def _computations(compiler_hlo: str) -> dict[str, _Computation]:
    lines = compiler_hlo.splitlines()
    header_pattern = re.compile(
        r"\s*(?P<entry>ENTRY\s+)?%?(?P<name>[A-Za-z0-9_.$-]+)"
        r"(?:\s+\([^\n]*\)\s*->\s*[^\n{]+)?\s*\{\s*"
    )
    headers = tuple(
        (index, match.group("name"), match.group("entry") is not None)
        for index, line in enumerate(lines)
        if (match := header_pattern.fullmatch(line)) is not None
    )
    if len(tuple(value for value in headers if value[2])) != 1:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_ENTRY_MISMATCH")
    computations = {}
    instruction_pattern = re.compile(r"\s*(?P<root>ROOT\s+)?%(?P<name>[^\s=]+)\s*=\s*(?P<body>.+)$")
    for header_index, (start, name, entry) in enumerate(headers):
        limit = headers[header_index + 1][0] if header_index + 1 < len(headers) else len(lines)
        endings = tuple(
            index
            for index in range(start + 1, limit)
            if re.fullmatch(r'\s*}(?:,\s*execution_thread="[^"]+")?\s*', lines[index])
        )
        if not endings:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_COMPUTATION_INVALID")
        end = endings[-1]
        starts = tuple(
            (index, match)
            for index in range(start + 1, end)
            if (match := instruction_pattern.match(lines[index])) is not None
        )
        instructions = []
        for instruction_index, (line_index, match) in enumerate(starts):
            next_line = (
                starts[instruction_index + 1][0] if instruction_index + 1 < len(starts) else end
            )
            body = "\n".join([match.group("body"), *lines[line_index + 1 : next_line]])
            instructions.append(
                _Instruction(
                    name=match.group("name"),
                    body=body,
                    root=match.group("root") is not None,
                )
            )
        computations[name] = _Computation(
            name=name,
            entry=entry,
            instructions=tuple(instructions),
        )
    return computations


def _called_computations(body: str) -> tuple[str, ...]:
    body = _semantic_instruction_body(body)
    called = []
    for attribute in _COMPUTATION_REFERENCES:
        called.extend(re.findall(rf"\b{attribute}=%?([A-Za-z0-9_.$-]+)", body))
    for attribute in ("branch_computations", "called_computations"):
        for values in re.findall(rf"\b{attribute}=\{{([^}}]*)\}}", body):
            called.extend(re.findall(r"%?([A-Za-z0-9_.$-]+)", values))
    return tuple(called)


def _reachable_and_live(
    compiler_hlo: str,
) -> tuple[dict[str, _Computation], set[tuple[str, str]]]:
    computations = _computations(compiler_hlo)
    entry = next(computation for computation in computations.values() if computation.entry)
    live: set[tuple[str, str]] = set()
    pending = deque([(entry.name, entry.root.name)])
    while pending:
        computation_name, instruction_name = pending.popleft()
        key = (computation_name, instruction_name)
        if key in live:
            continue
        computation = computations.get(computation_name)
        if computation is None:
            raise ValueError(
                f"SEQAX_SILU_FUSION_COMPILER_REFERENCE_MISSING computation={computation_name}"
            )
        instruction = computation.by_name.get(instruction_name)
        if instruction is None:
            continue
        live.add(key)
        pending.extend(
            (computation_name, reference)
            for reference in instruction.references
            if reference in computation.by_name
        )
        for called in _called_computations(instruction.body):
            called_computation = computations.get(called)
            if called_computation is None:
                raise ValueError(
                    f"SEQAX_SILU_FUSION_COMPILER_REFERENCE_MISSING computation={called}"
                )
            pending.append((called, called_computation.root.name))
    return computations, live


def live_seqax_silu_fusion_compiler_hlo(compiler_hlo: str) -> str:
    computations, live = _reachable_and_live(compiler_hlo)
    return "\n".join(
        instruction.body
        for computation_name, computation in computations.items()
        for instruction in computation.instructions
        if (computation_name, instruction.name) in live
    )


def _kernel_name(body: str) -> str | None:
    if "custom-call(" not in body or 'custom_call_target="tpu_custom_call"' not in body:
        return None
    matches = tuple(
        dict.fromkeys(
            re.findall(
                r'op_name="[^"]*/(seqax_[A-Za-z0-9_]+?)/pallas_call"',
                body,
            )
        )
    )
    if len(matches) > 1:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_KERNEL_METADATA_AMBIGUOUS")
    return matches[0] if matches else None


def _output_shape(body: str) -> str:
    match = re.match(r"\s*([^\s]+)\s+custom-call\(", body)
    if match is None:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_CUSTOM_CALL_INVALID")
    return match.group(1).split("{")[0]


def _custom_call_operands(body: str) -> tuple[str, ...]:
    match = re.search(r"\bcustom-call\((?P<operands>[^)]*)\)", body)
    if match is None:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_CUSTOM_CALL_INVALID")
    return tuple(re.findall(r"%([A-Za-z0-9_.$-]+)", match.group("operands")))


def _kernel_metadata(body: str, key: str) -> str:
    matches = re.findall(rf'"{re.escape(key)}":(?:")?([^",\n}}]+)', body)
    if not matches:
        raise ValueError(f"SEQAX_SILU_FUSION_COMPILER_METADATA_MISSING key={key}")
    if len(matches) != 1:
        raise ValueError(f"SEQAX_SILU_FUSION_COMPILER_METADATA_AMBIGUOUS key={key}")
    return matches[0]


def _depends_on(
    computation: _Computation,
    instruction_name: str,
    ancestor_name: str,
) -> bool:
    pending = [instruction_name]
    visited = set()
    while pending:
        name = pending.pop()
        if name == ancestor_name:
            return True
        if name in visited:
            continue
        visited.add(name)
        instruction = computation.by_name.get(name)
        if instruction is not None:
            pending.extend(instruction.references)
    return False


def _nearest_einsum_ancestors(
    computation: _Computation,
    instruction_name: str,
) -> frozenset[str]:
    found = set()
    visited = set()
    pending = deque([instruction_name])
    while pending:
        name = pending.popleft()
        if name in visited:
            continue
        visited.add(name)
        instruction = computation.by_name.get(name)
        if instruction is None:
            continue
        if _kernel_name(instruction.body) == "seqax_named_einsum":
            found.add(name)
            continue
        pending.extend(instruction.references)
    return frozenset(found)


def _projection_metadata_matches(
    computation: _Computation,
    names: frozenset[str],
    *,
    region_index: int,
    schedule_sha256: str,
) -> bool:
    if len(names) != 1:
        return False
    instruction = computation.by_name[next(iter(names))]
    return (
        int(_kernel_metadata(instruction.body, "region_index")) == region_index
        and _kernel_metadata(instruction.body, "schedule_sha256") == schedule_sha256
    )


def analyze_seqax_silu_fusion_compiler_hlo(
    compiler_hlo: str,
    candidate: SeqaxFeedForwardFusion,
    *,
    expected_schedule_sha256: str,
) -> SeqaxSiluFusionCompilerAnalysis:
    computations, live = _reachable_and_live(compiler_hlo)
    all_strict = []
    live_strict = []
    unknown_strict = []
    for computation_name, computation in computations.items():
        for instruction in computation.instructions:
            kernel = _kernel_name(instruction.body)
            if kernel is None:
                continue
            if kernel.startswith("seqax_strict_bf16_") and kernel not in _STRICT_KERNELS:
                unknown_strict.append(kernel)
            if kernel in _STRICT_KERNELS:
                value = (computation_name, instruction, kernel)
                all_strict.append(value)
                if (computation_name, instruction.name) in live:
                    live_strict.append(value)
    if unknown_strict:
        raise ValueError(
            f"SEQAX_SILU_FUSION_COMPILER_UNKNOWN_STRICT_KERNEL kernels={unknown_strict}"
        )
    if len(all_strict) != len(live_strict):
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_DEAD_STRICT_VECTOR_CALL")
    expected_kernels = (
        ("seqax_strict_bf16_silu", "seqax_strict_bf16_multiply")
        if candidate is SeqaxFeedForwardFusion.SEPARATE
        else ("seqax_strict_bf16_silu_multiply",)
    )
    by_kernel = {
        kernel: (computation_name, instruction)
        for computation_name, instruction, kernel in live_strict
    }
    if set(by_kernel) != set(expected_kernels) or len(live_strict) != len(expected_kernels):
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_STRICT_VECTOR_CALL_MISMATCH")
    ordered = tuple((*by_kernel[kernel], kernel) for kernel in expected_kernels)
    if len({computation_name for computation_name, _instruction, _kernel in ordered}) != 1:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_BOUNDARY_COMPUTATION_MISMATCH")
    computation = computations[ordered[0][0]]
    calls = tuple(
        SeqaxSiluFusionCompilerCall(
            ordinal=ordinal,
            kernel=kernel,
            output_shape=_output_shape(instruction.body),
            operand_count=len(_custom_call_operands(instruction.body)),
            schedule_sha256=_kernel_metadata(instruction.body, "schedule_sha256"),
            vector_region_index=int(_kernel_metadata(instruction.body, "vector_region_index")),
            implementation=_kernel_metadata(instruction.body, "implementation"),
            instruction_name=instruction.name,
            operand_names=_custom_call_operands(instruction.body),
        )
        for ordinal, (_computation_name, instruction, kernel) in enumerate(ordered)
    )
    if any(
        call.schedule_sha256 != expected_schedule_sha256
        or call.vector_region_index != call.ordinal
        or call.implementation != "pallas_full_local"
        for call in calls
    ):
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_VECTOR_METADATA_MISMATCH")
    required_operand_counts = (1, 2) if candidate is SeqaxFeedForwardFusion.SEPARATE else (2,)
    if tuple(call.operand_count for call in calls) != required_operand_counts:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_VECTOR_ARITY_MISMATCH")
    if candidate is SeqaxFeedForwardFusion.SEPARATE:
        silu, multiply = calls
        silu_feeds_multiply = _depends_on(
            computation,
            multiply.instruction_name,
            silu.instruction_name,
        )
        gate_ancestors = _nearest_einsum_ancestors(computation, silu.operand_names[0])
        silu_operands = tuple(
            operand
            for operand in multiply.operand_names
            if _depends_on(computation, operand, silu.instruction_name)
        )
        if len(silu_operands) != 1:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_SILU_MULTIPLY_EDGE_MISMATCH")
        up_operand = next(
            (operand for operand in multiply.operand_names if operand not in silu_operands),
            None,
        )
        up_ancestors = (
            frozenset()
            if up_operand is None
            else _nearest_einsum_ancestors(computation, up_operand)
        )
        output_name = multiply.instruction_name
    else:
        (fused,) = calls
        silu_feeds_multiply = False
        if len(fused.operand_names) != 2:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_FUSED_OPERAND_MISMATCH")
        gate_ancestors = _nearest_einsum_ancestors(computation, fused.operand_names[0])
        up_ancestors = _nearest_einsum_ancestors(computation, fused.operand_names[1])
        output_name = fused.instruction_name
    distinct_lineages = (
        gate_ancestors.isdisjoint(up_ancestors)
        and _projection_metadata_matches(
            computation,
            gate_ancestors,
            region_index=5,
            schedule_sha256=expected_schedule_sha256,
        )
        and _projection_metadata_matches(
            computation,
            up_ancestors,
            region_index=6,
            schedule_sha256=expected_schedule_sha256,
        )
    )
    down_projections = tuple(
        instruction
        for instruction in computation.instructions
        if (computation.name, instruction.name) in live
        and _kernel_name(instruction.body) == "seqax_named_einsum"
        and _depends_on(computation, instruction.name, output_name)
        and int(_kernel_metadata(instruction.body, "region_index")) == 7
    )
    down_operands = (
        () if len(down_projections) != 1 else _custom_call_operands(down_projections[0].body)
    )
    down_vector_operands = tuple(
        operand for operand in down_operands if _depends_on(computation, operand, output_name)
    )
    if (
        not distinct_lineages
        or len(down_projections) != 1
        or len(down_operands) != 2
        or len(down_vector_operands) != 1
        or _kernel_metadata(
            down_projections[0].body,
            "schedule_sha256",
        )
        != expected_schedule_sha256
    ):
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_PROJECTION_METADATA_MISMATCH")
    return SeqaxSiluFusionCompilerAnalysis(
        candidate=candidate,
        strict_vector_call_count=len(calls),
        calls=calls,
        all_strict_vector_calls_are_live=True,
        gate_and_up_projection_lineages_are_distinct=distinct_lineages,
        silu_output_feeds_multiply=silu_feeds_multiply,
        vector_output_feeds_one_down_projection=len(down_projections) == 1,
    )
