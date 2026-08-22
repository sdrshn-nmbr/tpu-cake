from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import jax
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import model_identity_sha256
from tpu_cake.physical_cost_model import analyze_physical_kernel
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_large_residual import (
    SeqaxLargeResidualContract,
    default_seqax_large_residual_contract,
)
from tpu_cake.seqax_residual_profile_runner import (
    _compile,
    _device_inventory,
    _json_sha256,
    _prepare_candidates,
    _text_sha256,
    _validate_devices,
)
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


class SeqaxLargeResidualCompilerCaptureCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxResidualNormStrategy
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_compiler_collectives: CompilerCollectiveAnalysis
    control_compiler_collectives: CompilerCollectiveAnalysis
    pallas_peak_memory_bytes: int = Field(gt=0)
    control_peak_memory_bytes: int = Field(gt=0)
    physical_peak_vmem_bytes_per_device: int = Field(gt=0)
    ring_equivalent_ici_bytes_per_device: int = Field(gt=0)


class SeqaxLargeResidualCompilerCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeIdentity
    device_ids: tuple[int, ...] = Field(min_length=8, max_length=8)
    candidates: tuple[SeqaxLargeResidualCompilerCaptureCandidate, ...] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def native_boundary_is_present(self) -> SeqaxLargeResidualCompilerCapture:
        if tuple(value.candidate for value in self.candidates) != (
            SeqaxResidualNormStrategy.STANDARD,
            SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
        ):
            raise ValueError("Seqax large residual compiler capture order mismatch")
        standard, residual_all_reduce = self.candidates
        if (
            standard.pallas_compiler_collectives.compiler_reduce_scatter_count < 3
            or standard.pallas_compiler_collectives.sparse_core_reduce_scatter_count < 3
            or standard.pallas_compiler_collectives.compiler_reduce_scatter_count
            <= residual_all_reduce.pallas_compiler_collectives.compiler_reduce_scatter_count
        ):
            raise ValueError("Seqax large residual native reduce-scatter boundary was rewritten")
        return self

    @property
    def capture_id(self) -> str:
        return model_identity_sha256(self)


def _require_clean_repository(repository_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_LARGE_RESIDUAL_SOURCE_DIRTY status={status}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _peak_memory_bytes(executable_analysis: object) -> int:
    memory = executable_analysis.memory
    value = memory.peak_memory_in_bytes
    if type(value) is not int or value <= 0:
        raise ValueError("SEQAX_LARGE_RESIDUAL_COMPILER_MEMORY_INVALID")
    return value


def capture_seqax_large_residual_compiler(
    contract: SeqaxLargeResidualContract,
) -> SeqaxLargeResidualCompilerCapture:
    repository_root = Path(__file__).resolve().parents[2]
    runtime = _runtime_identity()
    canonical = default_seqax_large_residual_contract(runtime)
    if contract != canonical or contract.compiler_identity_status != "pending":
        raise ValueError("SEQAX_LARGE_RESIDUAL_CAPTURE_CONTRACT_MISMATCH")
    if repository_root.resolve() != Path(contract.compilation_source_root):
        raise ValueError("SEQAX_LARGE_RESIDUAL_COMPILATION_ROOT_MISMATCH")
    source_commit = _require_clean_repository(repository_root)
    devices = tuple(jax.devices())
    _validate_devices(devices, contract)
    parameters = dict(contract.parameters)
    parameters.pop("numerical_semantics")
    host_inputs = tuple(
        np.asarray(value) for value in seqax_forward_inputs(seed=contract.timing_seed, **parameters)
    )
    prepared = _prepare_candidates(contract)
    compiled = tuple(
        _compile(
            value,
            host_inputs,
            devices,
            enforce_hlo_identity=False,
            enforce_compiler_collectives=False,
        )
        for value in prepared
    )
    candidates = []
    for value in compiled:
        expected = value.prepared.expected
        physical_report = analyze_physical_kernel(
            value.prepared.physical,
            hardware=tpu7x_tensorcore_rates(),
        )
        ring_bytes = int(physical_report.devices[0].collective_ring_equivalent_bytes)
        if (
            physical_report.memory.peak_live_vmem_bytes_per_device
            != expected.expected_peak_vmem_bytes_per_device
            or ring_bytes != expected.expected_ring_equivalent_ici_bytes_per_device
        ):
            raise ValueError(
                f"SEQAX_LARGE_RESIDUAL_RESOURCE_MISMATCH candidate={expected.candidate}"
            )
        candidates.append(
            SeqaxLargeResidualCompilerCaptureCandidate(
                candidate=expected.candidate,
                distributed_schedule_sha256=value.prepared.plan.distributed_schedule_sha256,
                physical_schedule_sha256=value.prepared.plan.physical_schedule_sha256,
                pallas_source_sha256=value.prepared.plan.source_sha256(),
                pallas_manifest_sha256=_json_sha256(value.prepared.plan.manifest()),
                pallas_stablehlo_sha256=_text_sha256(value.pallas_stablehlo),
                pallas_compiler_hlo_sha256=_text_sha256(value.pallas_compiler_hlo),
                control_stablehlo_sha256=_text_sha256(value.control_stablehlo),
                control_compiler_hlo_sha256=_text_sha256(value.control_compiler_hlo),
                pallas_compiler_collectives=value.pallas_compiler_analysis.collectives,
                control_compiler_collectives=value.control_compiler_analysis.collectives,
                pallas_peak_memory_bytes=_peak_memory_bytes(value.pallas_compiler_analysis),
                control_peak_memory_bytes=_peak_memory_bytes(value.control_compiler_analysis),
                physical_peak_vmem_bytes_per_device=(
                    physical_report.memory.peak_live_vmem_bytes_per_device
                ),
                ring_equivalent_ici_bytes_per_device=ring_bytes,
            )
        )
    inventory = _device_inventory(devices)
    return SeqaxLargeResidualCompilerCapture(
        contract_id=contract.contract_id,
        source_commit=source_commit,
        uv_lock_sha256=hashlib.sha256((repository_root / "uv.lock").read_bytes()).hexdigest(),
        runtime=runtime,
        device_ids=tuple(value.id for value in inventory),
        candidates=tuple(candidates),
    )
