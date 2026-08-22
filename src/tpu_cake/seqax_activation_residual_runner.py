from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from xdsl.dialects.builtin import ModuleOp

from tpu_cake.artifacts import (
    build_artifact_manifest,
    file_sha256,
    validate_artifact_manifest,
)
from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import (
    CompilerCollectiveAnalysis,
    CompilerExecutableAnalysis,
    analyze_compiler_collectives,
    capture_compiler_analysis,
    validate_compiler_analysis,
)
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import json_sha256, model_identity_sha256
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.physical_cost_model import analyze_physical_kernel
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_activation_residual import (
    SeqaxActivationResidualDesignContract,
    SeqaxActivationResidualPlanContract,
    default_seqax_activation_residual_design_contract,
)
from tpu_cake.seqax_pallas_lowering import SeqaxPallasPlan, lower_seqax_physical_to_pallas
from tpu_cake.seqax_pallas_runner import (
    _compiler_hlo,
    _physical_collective_inventory,
    _validate_compiled_program,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.stablehlo import StableHloInspector
from tpu_cake.workloads.seqax_forward import (
    SeqaxNumericalSemantics,
    SeqaxResidualNormStrategy,
    seqax_forward_schedule,
)

SEQAX_ACTIVATION_RESIDUAL_CAPTURE_SCHEMA = "seqax-activation-residual-compiler-capture-v1"
SEQAX_ACTIVATION_RESIDUAL_RECEIPT_SCHEMA = "seqax-activation-residual-compiler-receipt-v1"
_EVIDENCE_ROOT = Path("/home/sudarshan/tpu-cake-evidence")
_BOUNDARY_SIGNATURE = (
    "f32[32,64,256]",
    "f32[32,64,64]",
    "bf16[32,64,64]",
    "bf16[32,64,256]",
)
_COMPILER_ENVIRONMENT_PREFIXES = ("JAX_", "XLA_", "PJRT_", "LIBTPU_")


class SeqaxActivationResidualSourceAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch: Literal["main"] = "main"
    origin_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_url: str
    source_root: str
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeIdentity

    @model_validator(mode="after")
    def main_is_exact(self) -> SeqaxActivationResidualSourceAuthority:
        if (
            self.source_commit != self.origin_main_commit
            or self.source_commit != self.remote_main_commit
        ):
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_SOURCE_MAIN_MISMATCH")
        return self


class SeqaxActivationResidualHostIdentity(BaseModel):
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


class SeqaxActivationResidualDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=0, le=7)
    process_index: Literal[0]
    platform: Literal["tpu"]
    device_kind: Literal["TPU7x"]


class SeqaxActivationResidualBoundarySignature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reduce_scatter_input: Literal["f32[32,64,256]"]
    reduce_scatter_output: Literal["f32[32,64,64]"]
    residual_output: Literal["bf16[32,64,64]"]
    all_gather_output: Literal["bf16[32,64,256]"]
    reduce_scatter_native: Literal[True]
    reduce_scatter_sparse_core_offload: Literal[True]
    residual_is_bf16_add_of_converted_f32: Literal[True]
    all_gather_sparse_core_offload: Literal[True]


class SeqaxActivationResidualBoundaryTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ordinal: int = Field(ge=0, le=1)
    signature: SeqaxActivationResidualBoundarySignature
    reduce_scatter_start: str = Field(min_length=1)
    reduce_scatter_done: str = Field(min_length=1)
    residual_operand: str = Field(min_length=1)
    residual_fusion: str = Field(min_length=1)
    all_gather_start: str = Field(min_length=1)


class SeqaxActivationResidualBoundaryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chain_count: Literal[2]
    traces: tuple[
        SeqaxActivationResidualBoundaryTrace,
        SeqaxActivationResidualBoundaryTrace,
    ]

    @model_validator(mode="after")
    def two_distinct_exact_chains(self) -> SeqaxActivationResidualBoundaryAnalysis:
        if tuple(trace.ordinal for trace in self.traces) != (0, 1):
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_BOUNDARY_ORDER_MISMATCH")
        for field in (
            "reduce_scatter_start",
            "reduce_scatter_done",
            "residual_operand",
            "residual_fusion",
            "all_gather_start",
        ):
            if len({getattr(trace, field) for trace in self.traces}) != 2:
                raise ValueError("SEQAX_ACTIVATION_RESIDUAL_BOUNDARY_DUPLICATE")
        return self

    @computed_field
    @property
    def semantic_id(self) -> str:
        return json_sha256(
            {
                "chain_count": self.chain_count,
                "signatures": [trace.signature.model_dump(mode="json") for trace in self.traces],
            }
        )


class SeqaxActivationResidualCompilerCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_analysis: CompilerExecutableAnalysis
    control_analysis: CompilerExecutableAnalysis
    pallas_reachable_collectives: CompilerCollectiveAnalysis
    control_reachable_collectives: CompilerCollectiveAnalysis
    pallas_boundary: SeqaxActivationResidualBoundaryAnalysis | None
    physical_peak_vmem_bytes_per_device: int = Field(gt=0)
    ring_equivalent_ici_bytes_per_device: int = Field(gt=0)

    @model_validator(mode="after")
    def boundary_matches_candidate(self) -> SeqaxActivationResidualCompilerCandidate:
        if self.candidate is SeqaxResidualNormStrategy.STANDARD:
            if self.pallas_boundary is None:
                raise ValueError("SEQAX_ACTIVATION_RESIDUAL_STANDARD_BOUNDARY_MISSING")
        elif self.pallas_boundary is not None:
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_BASELINE_BOUNDARY_PRESENT")
        return self

    @computed_field
    @property
    def semantic_id(self) -> str:
        def compiler_semantics(
            value: CompilerExecutableAnalysis,
            reachable_collectives: CompilerCollectiveAnalysis,
        ) -> dict[str, object]:
            return {
                "stablehlo_sha256": value.stablehlo_sha256,
                "peak_memory_in_bytes": value.memory.peak_memory_in_bytes,
                "reachable_collectives": reachable_collectives.model_dump(mode="json"),
            }

        return json_sha256(
            {
                "candidate": self.candidate,
                "distributed_schedule_sha256": self.distributed_schedule_sha256,
                "physical_schedule_sha256": self.physical_schedule_sha256,
                "pallas_source_sha256": self.pallas_source_sha256,
                "pallas_manifest_sha256": self.pallas_manifest_sha256,
                "pallas": compiler_semantics(
                    self.pallas_analysis,
                    self.pallas_reachable_collectives,
                ),
                "control": compiler_semantics(
                    self.control_analysis,
                    self.control_reachable_collectives,
                ),
                "pallas_boundary": (
                    None if self.pallas_boundary is None else self.pallas_boundary.semantic_id
                ),
                "physical_peak_vmem_bytes_per_device": self.physical_peak_vmem_bytes_per_device,
                "ring_equivalent_ici_bytes_per_device": self.ring_equivalent_ici_bytes_per_device,
            }
        )


class SeqaxActivationResidualCompilerCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_ACTIVATION_RESIDUAL_CAPTURE_SCHEMA] = (
        SEQAX_ACTIVATION_RESIDUAL_CAPTURE_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source: SeqaxActivationResidualSourceAuthority
    host: SeqaxActivationResidualHostIdentity
    compiler_environment: dict[str, str]
    compile_input_mode: Literal["abstract-only"]
    devices: tuple[SeqaxActivationResidualDevice, ...] = Field(min_length=8, max_length=8)
    candidates: tuple[
        SeqaxActivationResidualCompilerCandidate,
        SeqaxActivationResidualCompilerCandidate,
    ]
    model_outputs_executed: Literal[False]
    correctness_outputs_collected: Literal[False]
    timing_collected: Literal[False]
    profile_collected: Literal[False]

    @model_validator(mode="after")
    def capture_scope_is_compile_only(self) -> SeqaxActivationResidualCompilerCapture:
        if tuple(device.id for device in self.devices) != tuple(range(8)):
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_DEVICE_ORDER_MISMATCH")
        if tuple(candidate.candidate for candidate in self.candidates) != (
            SeqaxResidualNormStrategy.STANDARD,
            SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        ):
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_CANDIDATE_ORDER_MISMATCH")
        return self

    @computed_field
    @property
    def capture_id(self) -> str:
        return model_identity_sha256(self)


class SeqaxActivationResidualCompilerReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[SEQAX_ACTIVATION_RESIDUAL_RECEIPT_SCHEMA] = (
        SEQAX_ACTIVATION_RESIDUAL_RECEIPT_SCHEMA
    )
    capture: SeqaxActivationResidualCompilerCapture
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def artifacts_are_compile_only(self) -> SeqaxActivationResidualCompilerReceipt:
        forbidden = {
            ArtifactRole.CORRECTNESS_INPUT,
            ArtifactRole.CORRECTNESS_OUTPUT,
            ArtifactRole.ORACLE_OUTPUT,
            ArtifactRole.TIMING_SAMPLES,
            ArtifactRole.TIMING_TRACE,
            ArtifactRole.COUNTER_TRACE,
            ArtifactRole.PROFILER_CONFIG,
        }
        if any(artifact.role in forbidden for artifact in self.artifacts):
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_NONCOMPILER_ARTIFACT")
        return self

    @computed_field
    @property
    def receipt_id(self) -> str:
        return model_identity_sha256(self)


@dataclass(frozen=True)
class _PreparedCandidate:
    expected: SeqaxActivationResidualPlanContract
    distributed: ModuleOp
    physical: ModuleOp
    plan: SeqaxPallasPlan


@dataclass(frozen=True)
class _CompiledProgram:
    prepared: _PreparedCandidate
    pallas_executable: Any
    control_executable: Any
    mesh: Any
    pallas_stablehlo: str
    pallas_compiler_hlo: str
    control_stablehlo: str
    control_compiler_hlo: str
    pallas_compiler_analysis: CompilerExecutableAnalysis
    control_compiler_analysis: CompilerExecutableAnalysis
    pallas_reachable_collectives: CompilerCollectiveAnalysis
    control_reachable_collectives: CompilerCollectiveAnalysis


@dataclass(frozen=True)
class _CompiledCandidate:
    value: _CompiledProgram
    pallas_boundary: SeqaxActivationResidualBoundaryAnalysis | None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _write_json_exclusive(path: Path, value: object) -> None:
    _write_bytes_exclusive(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_authority(
    repository_root: Path,
    design: SeqaxActivationResidualDesignContract,
) -> SeqaxActivationResidualSourceAuthority:
    repository_root = repository_root.resolve(strict=True)
    if repository_root != Path(design.compilation_source_root):
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_SOURCE_ROOT_MISMATCH")
    branch = _git(repository_root, "branch", "--show-current")
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != design.source_branch or status:
        raise ValueError(
            "SEQAX_ACTIVATION_RESIDUAL_SOURCE_CHECKOUT_INVALID "
            f"branch={branch!r} status={status.splitlines()}"
        )
    remote = subprocess.run(
        ["/usr/bin/git", "ls-remote", design.source_remote_url, "refs/heads/main"],
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(remote) != 2 or remote[1] != "refs/heads/main":
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_REMOTE_MAIN_INVALID")
    source_commit = _git(repository_root, "rev-parse", "HEAD")
    authority = SeqaxActivationResidualSourceAuthority(
        source_commit=source_commit,
        source_tree=_git(repository_root, "rev-parse", "HEAD^{tree}"),
        origin_main_commit=_git(repository_root, "rev-parse", "origin/main"),
        remote_main_commit=remote[0],
        remote_url=design.source_remote_url,
        source_root=design.compilation_source_root,
        uv_lock_sha256=file_sha256(repository_root / "uv.lock"),
        design_file_sha256=file_sha256(
            repository_root / "contracts" / "seqax-activation-residual-v1.json"
        ),
        runner_source_sha256=file_sha256(Path(__file__)),
        runtime=_runtime_identity(),
    )
    if authority.runtime != design.runtime:
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_RUNTIME_MISMATCH")
    return authority


class _RejectMetadataRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_METADATA_REDIRECT code={code} url={newurl}")


def _metadata(path: str) -> str:
    request = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectMetadataRedirects(),
    )
    with opener.open(request, timeout=5) as response:
        if response.headers.get("Metadata-Flavor") != "Google":
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_METADATA_HEADER_MISSING")
        payload = response.read(4097)
    if len(payload) > 4096:
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_METADATA_RESPONSE_TOO_LARGE")
    return payload.decode().strip()


def _host_identity() -> SeqaxActivationResidualHostIdentity:
    zone_resource = _metadata("instance/zone")
    machine_type_resource = _metadata("instance/machine-type")
    return SeqaxActivationResidualHostIdentity(
        project=_metadata("project/project-id"),
        numeric_project_id=_metadata("project/numeric-project-id"),
        zone=zone_resource.rsplit("/", maxsplit=1)[-1],
        hostname=_metadata("instance/name"),
        instance_hostname=_metadata("instance/hostname"),
        machine_type=machine_type_resource.rsplit("/", maxsplit=1)[-1],
        instance_id=_metadata("instance/id"),
        cpu_platform=_metadata("instance/cpu-platform"),
        zone_resource=zone_resource,
        machine_type_resource=machine_type_resource,
    )


def _expected_host(
    design: SeqaxActivationResidualDesignContract,
) -> SeqaxActivationResidualHostIdentity:
    return SeqaxActivationResidualHostIdentity(
        project=design.project,
        numeric_project_id=design.numeric_project_id,
        zone=design.zone,
        hostname=design.hostname,
        instance_hostname=design.instance_hostname,
        machine_type=design.machine_type,
        instance_id=design.instance_id,
        cpu_platform=design.cpu_platform,
        zone_resource=f"projects/{design.numeric_project_id}/zones/{design.zone}",
        machine_type_resource=(
            f"projects/{design.numeric_project_id}/machineTypes/{design.machine_type}"
        ),
    )


def _compiler_environment(design: SeqaxActivationResidualDesignContract) -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in design.compiler_environment}
    forbidden = {
        key: value
        for key, value in os.environ.items()
        if (key == "TPU_LIBRARY_PATH" or key.startswith(_COMPILER_ENVIRONMENT_PREFIXES))
        and key not in design.compiler_environment
    }
    if observed != design.compiler_environment or forbidden:
        raise ValueError(
            "SEQAX_ACTIVATION_RESIDUAL_COMPILER_ENVIRONMENT_MISMATCH "
            f"observed={observed} forbidden={forbidden}"
        )
    return dict(design.compiler_environment)


def _device_inventory() -> tuple[SeqaxActivationResidualDevice, ...]:
    return tuple(
        SeqaxActivationResidualDevice(
            id=int(device.id),
            process_index=int(device.process_index),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        for device in jax.devices()
    )


def _validate_authority(
    design: SeqaxActivationResidualDesignContract,
    host: SeqaxActivationResidualHostIdentity,
    devices: tuple[SeqaxActivationResidualDevice, ...],
) -> None:
    if host != _expected_host(design):
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_HOST_MISMATCH")
    if tuple(device.id for device in devices) != tuple(range(8)):
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_DEVICE_INVENTORY_MISMATCH")
    if jax.default_backend() != design.backend or len(devices) != design.device_count:
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_BACKEND_MISMATCH")


def _compiler_computations(compiler_hlo: str) -> tuple[dict[str, str], str]:
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
    entries = tuple(name for _index, name, is_entry in headers if is_entry)
    if len(entries) != 1:
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_COMPILER_ENTRY_MISMATCH")
    computations = {}
    for header_index, (start, name, _is_entry) in enumerate(headers):
        limit = headers[header_index + 1][0] if header_index + 1 < len(headers) else len(lines)
        endings = tuple(
            index
            for index in range(start + 1, limit)
            if re.fullmatch(
                r'\s*}(?:,\s*execution_thread="[^"]+")?\s*',
                lines[index],
            )
        )
        if not endings:
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_COMPILER_COMPUTATION_INVALID")
        end = endings[-1]
        computations[name] = "\n".join(lines[start : end + 1])
    return computations, entries[0]


def _reachable_compiler_computations(compiler_hlo: str) -> dict[str, str]:
    computations, entry = _compiler_computations(compiler_hlo)
    reachable = {}
    pending = [entry]
    reference_attributes = (
        "to_apply",
        "calls",
        "condition",
        "body",
        "fused_computation",
    )
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        computation = computations.get(name)
        if computation is None:
            raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_COMPILER_REFERENCE_MISSING name={name}")
        reachable[name] = computation
        for attribute in reference_attributes:
            pending.extend(
                re.findall(
                    rf"\b{attribute}=%?([A-Za-z0-9_.$-]+)",
                    computation,
                )
            )
        for attribute in ("branch_computations", "called_computations"):
            for values in re.findall(rf"\b{attribute}=\{{([^}}]*)\}}", computation):
                pending.extend(re.findall(r"%?([A-Za-z0-9_.$-]+)", values))
    return reachable


def _reachable_compiler_hlo(compiler_hlo: str) -> str:
    return "\n".join(_reachable_compiler_computations(compiler_hlo).values())


def _live_instruction_groups(
    compiler_hlo: str,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    groups = []
    for computation in _reachable_compiler_computations(compiler_hlo).values():
        instructions: dict[str, str] = {}
        roots = []
        for line in computation.splitlines()[1:-1]:
            match = re.match(
                r"\s*(?P<root>ROOT\s+)?%(?P<name>[^\s=]+)\s*=\s*(?P<body>.+)$",
                line,
            )
            if match is None:
                continue
            name = match.group("name")
            instructions[name] = match.group("body")
            if match.group("root") is not None:
                roots.append(name)
        if len(roots) != 1:
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_COMPILER_ROOT_MISMATCH")
        live = set()
        pending = list(roots)
        while pending:
            name = pending.pop()
            if name in live:
                continue
            body = instructions.get(name)
            if body is None:
                continue
            live.add(name)
            pending.extend(
                reference
                for reference in re.findall(r"%([A-Za-z0-9_.$-]+)", body)
                if reference in instructions
            )
        groups.append(tuple((name, body) for name, body in instructions.items() if name in live))
    return tuple(groups)


def _references(body: str, name: str) -> bool:
    return (
        re.search(
            rf"%{re.escape(name)}(?![A-Za-z0-9_.$-])",
            body,
        )
        is not None
    )


def _instruction_references(body: str) -> tuple[str, ...]:
    return tuple(re.findall(r"%([A-Za-z0-9_.$-]+)", body))


def _fusion_operands(body: str) -> tuple[str, ...]:
    match = re.search(r"\bfusion\((?P<operands>[^)]*)\)", body)
    if match is None:
        return ()
    return tuple(re.findall(r"%([A-Za-z0-9_.$-]+)", match.group("operands")))


def _fusion_is_bf16_residual_add(compiler_hlo: str, fusion_body: str) -> bool:
    called = tuple(re.findall(r"\bcalls=%?([A-Za-z0-9_.$-]+)", fusion_body))
    if len(called) != 1:
        return False
    computation = _compiler_computations(compiler_hlo)[0].get(called[0])
    if computation is None:
        return False
    local_f32 = _BOUNDARY_SIGNATURE[1]
    local_bf16 = _BOUNDARY_SIGNATURE[2]
    instructions = {}
    root = None
    for line in computation.splitlines()[1:-1]:
        match = re.match(
            r"\s*(?P<root>ROOT\s+)?%(?P<name>[^\s=]+)\s*=\s*(?P<body>.+)$",
            line,
        )
        if match is None:
            continue
        instructions[match.group("name")] = match.group("body")
        if match.group("root") is not None:
            root = match.group("name")
    if root is None:
        return False
    converts = {
        name
        for name, body in instructions.items()
        if local_bf16 in body
        and " convert(" in f" {body}"
        and any(
            local_f32 in instructions.get(reference, "")
            for reference in _instruction_references(body)
        )
    }
    root_body = instructions[root]
    root_references = set(_instruction_references(root_body))
    return (
        local_bf16 in root_body
        and " add(" in f" {root_body}"
        and bool(converts & root_references)
        and len(root_references) == 2
    )


def analyze_activation_residual_boundary(
    compiler_hlo: str,
) -> SeqaxActivationResidualBoundaryAnalysis:
    full_f32, local_f32, local_bf16, full_bf16 = _BOUNDARY_SIGNATURE
    traces = []
    for instructions in _live_instruction_groups(compiler_hlo):
        starts = tuple(
            (name, body)
            for name, body in instructions
            if full_f32 in body
            and local_f32 in body
            and "call-start(" in body
            and 'async_execution_thread="sparsecore"' in body
            and 'op_name="jit(physical_call)/shard_map/reduce_scatter"' in body
            and '"offload":"OFFLOAD_COLLECTIVE"' in body
        )
        for start_name, _start_body in starts:
            dones = tuple(
                (name, body)
                for name, body in instructions
                if local_f32 in body
                and "call-done(" in body
                and _references(body, start_name)
                and 'op_name="jit(physical_call)/shard_map/reduce_scatter"' in body
            )
            if len(dones) != 1:
                continue
            done_name, _done_body = dones[0]
            fusions = tuple(
                (name, body, _fusion_operands(body))
                for name, body in instructions
                if local_bf16 in body
                and " fusion(" in f" {body}"
                and _references(body, done_name)
                and 'op_name="jit(physical_call)/shard_map/add"' in body
                and len(_fusion_operands(body)) == 2
                and done_name in _fusion_operands(body)
                and all(operand in dict(instructions) for operand in _fusion_operands(body))
                and _fusion_is_bf16_residual_add(compiler_hlo, body)
            )
            if len(fusions) != 1:
                continue
            fusion_name, _fusion_body, fusion_inputs = fusions[0]
            residual_operand = next(name for name in fusion_inputs if name != done_name)
            gathers = tuple(
                (name, body)
                for name, body in instructions
                if local_bf16 in body
                and full_bf16 in body
                and "call-start(" in body
                and _references(body, fusion_name)
                and 'async_execution_thread="sparsecore"' in body
                and 'op_name="jit(physical_call)/shard_map/all_gather"' in body
                and '"offload":"OFFLOAD_COLLECTIVE"' in body
            )
            if len(gathers) == 1:
                traces.append((start_name, done_name, residual_operand, fusion_name, gathers[0][0]))
    signature = SeqaxActivationResidualBoundarySignature(
        reduce_scatter_input=full_f32,
        reduce_scatter_output=local_f32,
        residual_output=local_bf16,
        all_gather_output=full_bf16,
        reduce_scatter_native=True,
        reduce_scatter_sparse_core_offload=True,
        residual_is_bf16_add_of_converted_f32=True,
        all_gather_sparse_core_offload=True,
    )
    return SeqaxActivationResidualBoundaryAnalysis(
        chain_count=len(traces),
        traces=tuple(
            SeqaxActivationResidualBoundaryTrace(
                ordinal=index,
                signature=signature,
                reduce_scatter_start=trace[0],
                reduce_scatter_done=trace[1],
                residual_operand=trace[2],
                residual_fusion=trace[3],
                all_gather_start=trace[4],
            )
            for index, trace in enumerate(traces)
        ),
    )


def _canonical_hlo(value: str) -> str:
    return value.rstrip("\n") + "\n"


def _parameters(design: SeqaxActivationResidualDesignContract) -> dict[str, int | Any]:
    parameters = dict(design.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    return parameters


def _prepare_candidates(
    design: SeqaxActivationResidualDesignContract,
) -> tuple[_PreparedCandidate, ...]:
    prepared = []
    for expected in design.candidates:
        distributed = seqax_forward_schedule(
            **_parameters(design),
            residual_norm_strategy=expected.candidate,
        )
        physical = lower_seqax_forward_to_physical(distributed).module
        plan = lower_seqax_physical_to_pallas(distributed, physical)
        observed = (
            plan.distributed_schedule_sha256,
            plan.physical_schedule_sha256,
            plan.source_sha256(),
            json_sha256(plan.manifest()),
            plan.pallas_region_count,
            _physical_collective_inventory(physical),
        )
        required = (
            expected.distributed_schedule_sha256,
            expected.physical_schedule_sha256,
            expected.pallas_source_sha256,
            expected.pallas_manifest_sha256,
            expected.expected_pallas_regions,
            (
                expected.expected_all_gathers,
                expected.expected_all_reduces,
                expected.expected_reduce_scatters,
            ),
        )
        if observed != required:
            raise ValueError(
                "SEQAX_ACTIVATION_RESIDUAL_PLAN_IDENTITY_MISMATCH "
                f"candidate={expected.candidate} expected={required} observed={observed}"
            )
        prepared.append(
            _PreparedCandidate(
                expected=expected,
                distributed=distributed,
                physical=physical,
                plan=plan,
            )
        )
    return tuple(prepared)


def _abstract_inputs(prepared: _PreparedCandidate, mesh: Any) -> tuple[Any, ...]:
    dtypes = {
        "bfloat16": jnp.bfloat16,
        "bool": jnp.bool_,
        "float32": jnp.float32,
        "int32": jnp.int32,
        "uint32": jnp.uint32,
    }
    values = []
    for contract in prepared.plan.input_contracts:
        if contract.dtype not in dtypes:
            raise ValueError(
                f"SEQAX_ACTIVATION_RESIDUAL_ABSTRACT_DTYPE_UNSUPPORTED dtype={contract.dtype}"
            )
        values.append(
            jax.ShapeDtypeStruct(
                tuple(size for _name, size in contract.shape),
                dtypes[contract.dtype],
                sharding=NamedSharding(mesh, contract.partition_spec()),
            )
        )
    return tuple(values)


def _validate_pallas_collectives(
    expected: SeqaxActivationResidualPlanContract,
    collectives: CompilerCollectiveAnalysis,
) -> None:
    observed = (
        collectives.stablehlo_all_gather_count,
        collectives.stablehlo_reduce_scatter_count,
        collectives.compiler_all_gather_count,
        collectives.compiler_all_reduce_count,
        collectives.compiler_reduce_scatter_count,
        collectives.sparse_core_all_gather_count,
        collectives.sparse_core_reduce_scatter_count,
    )
    required = (
        expected.expected_all_gathers,
        expected.expected_reduce_scatters,
        expected.expected_all_gathers,
        expected.expected_all_reduces,
        expected.expected_reduce_scatters,
        expected.expected_all_gathers,
        expected.expected_reduce_scatters,
    )
    if observed != required:
        raise ValueError(
            "SEQAX_ACTIVATION_RESIDUAL_NATIVE_COLLECTIVE_MISMATCH "
            f"candidate={expected.candidate} expected={required} observed={observed}"
        )


def _validate_control_collectives(
    expected: SeqaxActivationResidualPlanContract,
    collectives: CompilerCollectiveAnalysis,
) -> None:
    if (
        collectives.stablehlo_reduce_scatter_count != expected.expected_reduce_scatters
        or collectives.compiler_reduce_scatter_count != expected.expected_reduce_scatters
        or collectives.sparse_core_reduce_scatter_count != expected.expected_reduce_scatters
        or collectives.compiler_all_reduce_count != expected.expected_all_reduces
        or collectives.stablehlo_all_gather_count != collectives.compiler_all_gather_count
        or collectives.compiler_all_gather_count != collectives.sparse_core_all_gather_count
        or not 0 < collectives.compiler_all_gather_count <= expected.expected_all_gathers
    ):
        raise ValueError(
            "SEQAX_ACTIVATION_RESIDUAL_CONTROL_COLLECTIVE_MISMATCH "
            f"candidate={expected.candidate} observed={collectives}"
        )


def _compile_candidate(
    prepared: _PreparedCandidate,
    devices: tuple[Any, ...],
) -> _CompiledCandidate:
    pallas_callable, mesh = prepared.plan.build(interpret=False, devices=devices)
    pallas_lowered = pallas_callable.lower(*_abstract_inputs(prepared, mesh))
    pallas_stablehlo = _canonical_hlo(str(pallas_lowered.compiler_ir(dialect="stablehlo")))
    pallas_pre_optimization_hlo = _canonical_hlo(_compiler_hlo(pallas_lowered))
    expected = prepared.expected
    _validate_compiled_program(
        pallas_stablehlo,
        pallas_pre_optimization_hlo,
        pallas_region_count=prepared.plan.pallas_region_count,
        pallas_vector_region_count=prepared.plan.pallas_vector_region_count,
        all_gather_count=expected.expected_all_gathers,
        all_reduce_count=expected.expected_all_reduces,
        reduce_scatter_count=expected.expected_reduce_scatters,
    )
    counts = StableHloInspector.parse(pallas_stablehlo).live_collective_counts()
    observed_counts = tuple(counts[name] for name in ("all_gather", "all_reduce", "reduce_scatter"))
    required_counts = (
        expected.expected_all_gathers,
        expected.expected_all_reduces,
        expected.expected_reduce_scatters,
    )
    if observed_counts != required_counts:
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_STABLEHLO_COLLECTIVE_MISMATCH")

    control_callable, control_mesh = lower_distributed_program_to_jax_mesh(
        prepared.distributed
    ).build(devices=devices)
    control_lowered = control_callable.lower(*_abstract_inputs(prepared, control_mesh))
    control_stablehlo = _canonical_hlo(str(control_lowered.compiler_ir(dialect="stablehlo")))
    pallas_executable = pallas_lowered.compile()
    control_executable = control_lowered.compile()
    pallas_compiler_hlo = _canonical_hlo(pallas_executable.as_text())
    control_compiler_hlo = _canonical_hlo(control_executable.as_text())
    pallas_analysis = capture_compiler_analysis(
        pallas_executable,
        stablehlo=pallas_stablehlo.rstrip("\n"),
        compiler_hlo=pallas_compiler_hlo.rstrip("\n"),
    )
    control_analysis = capture_compiler_analysis(
        control_executable,
        stablehlo=control_stablehlo.rstrip("\n"),
        compiler_hlo=control_compiler_hlo.rstrip("\n"),
    )
    pallas_reachable_collectives = analyze_compiler_collectives(
        stablehlo=pallas_stablehlo,
        compiler_hlo=_reachable_compiler_hlo(pallas_compiler_hlo),
    )
    control_reachable_collectives = analyze_compiler_collectives(
        stablehlo=control_stablehlo,
        compiler_hlo=_reachable_compiler_hlo(control_compiler_hlo),
    )
    _validate_pallas_collectives(expected, pallas_reachable_collectives)
    _validate_control_collectives(expected, control_reachable_collectives)
    if expected.candidate is SeqaxResidualNormStrategy.STANDARD:
        pallas_boundary = analyze_activation_residual_boundary(pallas_compiler_hlo)
    else:
        pallas_boundary = None
    return _CompiledCandidate(
        value=_CompiledProgram(
            prepared=prepared,
            pallas_executable=pallas_executable,
            control_executable=control_executable,
            mesh=mesh,
            pallas_stablehlo=pallas_stablehlo,
            pallas_compiler_hlo=pallas_compiler_hlo,
            control_stablehlo=control_stablehlo,
            control_compiler_hlo=control_compiler_hlo,
            pallas_compiler_analysis=pallas_analysis,
            control_compiler_analysis=control_analysis,
            pallas_reachable_collectives=pallas_reachable_collectives,
            control_reachable_collectives=control_reachable_collectives,
        ),
        pallas_boundary=pallas_boundary,
    )


def _candidate_record(compiled: _CompiledCandidate) -> SeqaxActivationResidualCompilerCandidate:
    value = compiled.value
    expected = value.prepared.expected
    physical_report = analyze_physical_kernel(
        value.prepared.physical,
        hardware=tpu7x_tensorcore_rates(),
    )
    ring_bytes = int(physical_report.devices[0].collective_ring_equivalent_bytes)
    peak_vmem = physical_report.memory.peak_live_vmem_bytes_per_device
    if (
        ring_bytes != expected.expected_ring_equivalent_ici_bytes_per_device
        or peak_vmem != expected.expected_peak_vmem_bytes_per_device
    ):
        raise ValueError(
            f"SEQAX_ACTIVATION_RESIDUAL_RESOURCE_MISMATCH candidate={expected.candidate}"
        )
    return SeqaxActivationResidualCompilerCandidate(
        candidate=expected.candidate,
        distributed_schedule_sha256=value.prepared.plan.distributed_schedule_sha256,
        physical_schedule_sha256=value.prepared.plan.physical_schedule_sha256,
        pallas_source_sha256=value.prepared.plan.source_sha256(),
        pallas_manifest_sha256=json_sha256(value.prepared.plan.manifest()),
        pallas_analysis=value.pallas_compiler_analysis,
        control_analysis=value.control_compiler_analysis,
        pallas_reachable_collectives=value.pallas_reachable_collectives,
        control_reachable_collectives=value.control_reachable_collectives,
        pallas_boundary=compiled.pallas_boundary,
        physical_peak_vmem_bytes_per_device=peak_vmem,
        ring_equivalent_ici_bytes_per_device=ring_bytes,
    )


def _write_candidate_artifacts(root: Path, compiled: _CompiledCandidate) -> None:
    value = compiled.value
    candidate_root = root / "candidates" / value.prepared.expected.candidate.value
    _write_bytes_exclusive(
        candidate_root / "distributed.xdsl",
        canonical_text(value.prepared.distributed).encode(),
    )
    _write_bytes_exclusive(
        candidate_root / "physical.xdsl",
        canonical_text(value.prepared.physical).encode(),
    )
    _write_bytes_exclusive(
        candidate_root / "lowered_pallas.py",
        value.prepared.plan.render_executable_source().encode(),
    )
    _write_json_exclusive(candidate_root / "plan_manifest.json", value.prepared.plan.manifest())
    _write_bytes_exclusive(candidate_root / "pallas_stablehlo.txt", value.pallas_stablehlo.encode())
    _write_bytes_exclusive(
        candidate_root / "pallas_compiler_hlo.txt", value.pallas_compiler_hlo.encode()
    )
    _write_bytes_exclusive(
        candidate_root / "control_stablehlo.txt", value.control_stablehlo.encode()
    )
    _write_bytes_exclusive(
        candidate_root / "control_compiler_hlo.txt", value.control_compiler_hlo.encode()
    )
    _write_json_exclusive(
        candidate_root / "pallas_compiler_analysis.json",
        value.pallas_compiler_analysis.model_dump(mode="json"),
    )
    _write_json_exclusive(
        candidate_root / "control_compiler_analysis.json",
        value.control_compiler_analysis.model_dump(mode="json"),
    )
    if compiled.pallas_boundary is not None:
        _write_json_exclusive(
            candidate_root / "pallas_boundary.json",
            compiled.pallas_boundary.model_dump(mode="json", exclude_computed_fields=True),
        )


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    fixed = {
        "invocation.json": ArtifactRole.INVOCATION,
        "contract.json": ArtifactRole.SEARCH_CONTRACT,
        "source.json": ArtifactRole.SOURCE_STATE,
        "host.json": ArtifactRole.PREFLIGHT_RESULT,
        "compiler_environment.json": ArtifactRole.PREFLIGHT_RESULT,
        "devices.json": ArtifactRole.BACKEND_MANIFEST,
    }
    if relative in fixed:
        return fixed[relative]
    if relative.startswith("candidates/"):
        roles = {
            "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
            "physical.xdsl": ArtifactRole.PHYSICAL_IR,
            "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
            "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
            "pallas_stablehlo.txt": ArtifactRole.STABLEHLO,
            "pallas_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
            "control_stablehlo.txt": ArtifactRole.STABLEHLO,
            "control_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
            "pallas_compiler_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
            "control_compiler_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
            "pallas_boundary.json": ArtifactRole.SEARCH_EVIDENCE,
        }
        if path.name in roles:
            return roles[path.name]
    raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_ARTIFACT_UNRECOGNIZED path={relative}")


def _require_safe_new_root(root: Path, repository_root: Path) -> None:
    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_ROOT_INVALID path={root}")
    resolved_root = root.resolve(strict=False)
    resolved_evidence = _EVIDENCE_ROOT.resolve(strict=True)
    resolved_repository = repository_root.resolve(strict=True)
    if resolved_root != root:
        raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_ROOT_NOT_NORMALIZED path={root}")
    if resolved_root.parent != resolved_evidence or not resolved_root.name.startswith(
        "seqax-activation-residual-compiler-"
    ):
        raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_ROOT_OUT_OF_SCOPE path={root}")
    if resolved_repository == resolved_root or resolved_repository in resolved_root.parents:
        raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_ROOT_IN_REPOSITORY path={root}")
    current = Path(root.anchor)
    for part in root.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_PARENT_SYMLINK path={current}")
    if not root.parent.is_dir():
        raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_PARENT_MISSING path={root.parent}")


def _preflight_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError(f"SEQAX_ACTIVATION_RESIDUAL_CAPTURE_LINK_INVALID path={path}")


def _validate_candidate_artifacts(
    root: Path,
    observed: SeqaxActivationResidualCompilerCandidate,
    prepared: _PreparedCandidate,
) -> None:
    candidate_root = root / "candidates" / observed.candidate.value
    expected = prepared.expected
    if (
        observed.distributed_schedule_sha256 != expected.distributed_schedule_sha256
        or observed.physical_schedule_sha256 != expected.physical_schedule_sha256
        or observed.pallas_source_sha256 != expected.pallas_source_sha256
        or observed.pallas_manifest_sha256 != expected.pallas_manifest_sha256
        or (candidate_root / "distributed.xdsl").read_text() != canonical_text(prepared.distributed)
        or (candidate_root / "physical.xdsl").read_text() != canonical_text(prepared.physical)
        or (candidate_root / "lowered_pallas.py").read_text()
        != prepared.plan.render_executable_source()
        or json.loads((candidate_root / "plan_manifest.json").read_text())
        != prepared.plan.manifest()
    ):
        raise ValueError(
            f"SEQAX_ACTIVATION_RESIDUAL_STATIC_ARTIFACT_MISMATCH candidate={observed.candidate}"
        )
    physical_report = analyze_physical_kernel(
        prepared.physical,
        hardware=tpu7x_tensorcore_rates(),
    )
    if (
        observed.physical_peak_vmem_bytes_per_device
        != physical_report.memory.peak_live_vmem_bytes_per_device
        or observed.ring_equivalent_ici_bytes_per_device
        != int(physical_report.devices[0].collective_ring_equivalent_bytes)
    ):
        raise ValueError(
            f"SEQAX_ACTIVATION_RESIDUAL_RESOURCE_REPLAY_MISMATCH candidate={observed.candidate}"
        )
    pallas_analysis = validate_compiler_analysis(
        candidate_root / "pallas_compiler_analysis.json",
        stablehlo_path=candidate_root / "pallas_stablehlo.txt",
        compiler_hlo_path=candidate_root / "pallas_compiler_hlo.txt",
    )
    control_analysis = validate_compiler_analysis(
        candidate_root / "control_compiler_analysis.json",
        stablehlo_path=candidate_root / "control_stablehlo.txt",
        compiler_hlo_path=candidate_root / "control_compiler_hlo.txt",
    )
    if pallas_analysis != observed.pallas_analysis or control_analysis != observed.control_analysis:
        raise ValueError(
            f"SEQAX_ACTIVATION_RESIDUAL_COMPILER_ARTIFACT_MISMATCH candidate={observed.candidate}"
        )
    pallas_reachable_collectives = analyze_compiler_collectives(
        stablehlo=(candidate_root / "pallas_stablehlo.txt").read_text(),
        compiler_hlo=_reachable_compiler_hlo(
            (candidate_root / "pallas_compiler_hlo.txt").read_text()
        ),
    )
    control_reachable_collectives = analyze_compiler_collectives(
        stablehlo=(candidate_root / "control_stablehlo.txt").read_text(),
        compiler_hlo=_reachable_compiler_hlo(
            (candidate_root / "control_compiler_hlo.txt").read_text()
        ),
    )
    if (
        pallas_reachable_collectives != observed.pallas_reachable_collectives
        or control_reachable_collectives != observed.control_reachable_collectives
    ):
        raise ValueError(
            f"SEQAX_ACTIVATION_RESIDUAL_REACHABLE_COLLECTIVE_MISMATCH candidate={observed.candidate}"
        )
    _validate_pallas_collectives(expected, pallas_reachable_collectives)
    _validate_control_collectives(expected, control_reachable_collectives)
    if observed.candidate is SeqaxResidualNormStrategy.STANDARD:
        pallas_boundary = analyze_activation_residual_boundary(
            (candidate_root / "pallas_compiler_hlo.txt").read_text()
        )
        saved_pallas = SeqaxActivationResidualBoundaryAnalysis.model_validate_json(
            (candidate_root / "pallas_boundary.json").read_text()
        )
        if pallas_boundary != saved_pallas or saved_pallas != observed.pallas_boundary:
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_BOUNDARY_ARTIFACT_MISMATCH")


def verify_seqax_activation_residual_compiler_capture(
    root: Path,
    design: SeqaxActivationResidualDesignContract,
) -> SeqaxActivationResidualCompilerReceipt:
    _preflight_root(root)
    canonical = default_seqax_activation_residual_design_contract(design.runtime)
    if design != canonical or design.compiler_identity_status != "pending":
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_CAPTURE_DESIGN_MISMATCH")
    receipt = SeqaxActivationResidualCompilerReceipt.model_validate_json(
        (root / "receipt.json").read_text()
    )
    saved_design = SeqaxActivationResidualDesignContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_design != design or receipt.capture.design_id != design.design_id:
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_CAPTURE_IDENTITY_MISMATCH")
    validate_artifact_manifest(
        root,
        receipt.artifacts,
        role_for_path=_artifact_role,
        duplicate_error="SEQAX_ACTIVATION_RESIDUAL_ARTIFACT_DUPLICATE",
        closed_world_error="SEQAX_ACTIVATION_RESIDUAL_ARTIFACT_CLOSED_WORLD_MISMATCH",
        mismatch_error=lambda path: f"SEQAX_ACTIVATION_RESIDUAL_ARTIFACT_MISMATCH path={path}",
        symlink_error="SEQAX_ACTIVATION_RESIDUAL_ARTIFACT_SYMLINK",
        excluded_paths=("receipt.json",),
    )
    source = SeqaxActivationResidualSourceAuthority.model_validate_json(
        (root / "source.json").read_text()
    )
    invocation = json.loads((root / "invocation.json").read_text())
    host = SeqaxActivationResidualHostIdentity.model_validate_json((root / "host.json").read_text())
    environment = json.loads((root / "compiler_environment.json").read_text())
    devices = tuple(
        SeqaxActivationResidualDevice.model_validate(value)
        for value in json.loads((root / "devices.json").read_text())
    )
    if (
        source != receipt.capture.source
        or invocation != {"invocation_id": receipt.capture.invocation_id}
        or source.remote_url != design.source_remote_url
        or source.source_root != design.compilation_source_root
        or source.runtime != design.runtime
        or host != receipt.capture.host
        or environment != receipt.capture.compiler_environment
        or devices != receipt.capture.devices
        or host != _expected_host(design)
        or environment != design.compiler_environment
        or receipt.capture.compile_input_mode != design.compile_input_mode
    ):
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_AUTHORITY_ARTIFACT_MISMATCH")
    repository_root = Path(__file__).resolve().parents[2]
    if repository_root.resolve() == Path(design.compilation_source_root) and (
        source != _source_authority(repository_root, design)
        or host != _host_identity()
        or environment != _compiler_environment(design)
        or devices != _device_inventory()
    ):
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_LIVE_AUTHORITY_MISMATCH")
    prepared = _prepare_candidates(design)
    for observed, expected in zip(receipt.capture.candidates, prepared, strict=True):
        _validate_candidate_artifacts(root, observed, expected)
    return receipt


def run_seqax_activation_residual_compiler_capture(
    root: Path,
    design: SeqaxActivationResidualDesignContract,
    *,
    invocation_id: str,
) -> SeqaxActivationResidualCompilerReceipt:
    repository_root = Path(__file__).resolve().parents[2]
    canonical = default_seqax_activation_residual_design_contract(_runtime_identity())
    if design != canonical or design.compiler_identity_status != "pending":
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_CAPTURE_DESIGN_MISMATCH")
    if not re.fullmatch(r"[0-9a-f]{32}", invocation_id):
        raise ValueError("SEQAX_ACTIVATION_RESIDUAL_INVOCATION_ID_INVALID")
    environment = _compiler_environment(design)
    source = _source_authority(repository_root, design)
    host = _host_identity()
    devices = _device_inventory()
    _validate_authority(design, host, devices)
    _require_safe_new_root(root, repository_root)
    root.mkdir(mode=0o700)
    _fsync_directory(root.parent)
    try:
        _write_json_exclusive(root / "invocation.json", {"invocation_id": invocation_id})
        _write_json_exclusive(
            root / "contract.json",
            design.model_dump(mode="json", exclude_computed_fields=True),
        )
        _write_json_exclusive(
            root / "source.json",
            source.model_dump(mode="json", exclude_computed_fields=True),
        )
        _write_json_exclusive(
            root / "host.json",
            host.model_dump(mode="json", exclude_computed_fields=True),
        )
        _write_json_exclusive(root / "compiler_environment.json", environment)
        _write_json_exclusive(
            root / "devices.json",
            [device.model_dump(mode="json") for device in devices],
        )
        prepared = _prepare_candidates(design)
        if (
            prepared[0].plan.input_contracts != prepared[1].plan.input_contracts
            or prepared[0].plan.output_contracts != prepared[1].plan.output_contracts
        ):
            raise ValueError("SEQAX_ACTIVATION_RESIDUAL_CANDIDATE_ABI_MISMATCH")
        raw_devices = tuple(jax.devices())
        compiled = tuple(_compile_candidate(value, raw_devices) for value in prepared)
        for value in compiled:
            _write_candidate_artifacts(root, value)
        capture = SeqaxActivationResidualCompilerCapture(
            design_id=design.design_id,
            invocation_id=invocation_id,
            source=source,
            host=host,
            compiler_environment=environment,
            compile_input_mode=design.compile_input_mode,
            devices=devices,
            candidates=tuple(_candidate_record(value) for value in compiled),
            model_outputs_executed=False,
            correctness_outputs_collected=False,
            timing_collected=False,
            profile_collected=False,
        )
        artifacts = build_artifact_manifest(
            root,
            role_for_path=_artifact_role,
            excluded_paths=("receipt.json",),
        )
        receipt = SeqaxActivationResidualCompilerReceipt(
            capture=capture,
            artifacts=artifacts,
        )
        _write_json_exclusive(
            root / "receipt.json",
            receipt.model_dump(mode="json", exclude_computed_fields=True),
        )
        return verify_seqax_activation_residual_compiler_capture(root, design)
    except BaseException as error:
        failure_path = root / "failure.json"
        if not failure_path.exists():
            _write_json_exclusive(
                failure_path,
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "invocation_id": invocation_id,
                    "design_id": design.design_id,
                },
            )
        raise


def _load_design(path: Path) -> SeqaxActivationResidualDesignContract:
    return SeqaxActivationResidualDesignContract.model_validate_json(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("capture", "verify"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invocation-id")
    args = parser.parse_args()
    design = _load_design(args.contract)
    if args.command == "capture":
        if args.invocation_id is None:
            parser.error("capture requires --invocation-id")
        receipt = run_seqax_activation_residual_compiler_capture(
            args.output,
            design,
            invocation_id=args.invocation_id,
        )
    else:
        if args.invocation_id is not None:
            parser.error("verify does not accept --invocation-id")
        receipt = verify_seqax_activation_residual_compiler_capture(args.output, design)
    print(receipt.model_dump_json(exclude_computed_fields=False))


if __name__ == "__main__":
    main()
