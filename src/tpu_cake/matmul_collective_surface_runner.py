from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256, semantic_sha256
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceArmPlan,
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceModel,
    SurfaceCalibrationObservation,
    derive_matmul_collective_surface_design_report,
    fit_surface_model,
)
from tpu_cake.receipt import _validate_matmul_compiler_strategy
from tpu_cake.runner import MatmulCollectiveStrategy, _runtime_identity

MATMUL_COLLECTIVE_SURFACE_COMPILE_SCHEMA = "matmul-collective-surface-compile-v1"
MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SEAL_SCHEMA = "matmul-collective-surface-calibration-seal-v1"

SURFACE_EXECUTABLE_DEPENDENCIES = (
    "tpu_cake/__init__.py",
    "tpu_cake/artifacts.py",
    "tpu_cake/canonical.py",
    "tpu_cake/compiler_analysis.py",
    "tpu_cake/contracts.py",
    "tpu_cake/cost_model.py",
    "tpu_cake/dialects/__init__.py",
    "tpu_cake/dialects/distributed_tensor.py",
    "tpu_cake/dialects/tpu_schedule.py",
    "tpu_cake/distributed_frontend.py",
    "tpu_cake/evidence.py",
    "tpu_cake/frontend.py",
    "tpu_cake/identity.py",
    "tpu_cake/ledger.py",
    "tpu_cake/lowering.py",
    "tpu_cake/matmul_collective_surface_prediction.py",
    "tpu_cake/matmul_collective_surface_runner.py",
    "tpu_cake/metrics.py",
    "tpu_cake/pallas_lowering.py",
    "tpu_cake/receipt.py",
    "tpu_cake/receipt_metrics.py",
    "tpu_cake/rpa_lowering.py",
    "tpu_cake/rpa_owned_kernel.py",
    "tpu_cake/runner.py",
    "tpu_cake/search.py",
    "tpu_cake/source.py",
    "tpu_cake/stablehlo.py",
    "tpu_cake/workloads/__init__.py",
    "tpu_cake/workloads/distributed_matmul.py",
    "tpu_cake/workloads/inkling_rpa.py",
    "tpu_cake/workloads/matmul.py",
    "tpu_cake/xprof_evidence.py",
)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _semantic_compiler_hlo(value: str) -> str:
    canonical = value.rstrip("\n") + "\n"
    metadata_start = canonical.find("\nFileNames\n")
    if metadata_start >= 0:
        computation_starts = tuple(
            offset
            for marker in ("\n%", "\nENTRY ")
            if (offset := canonical.find(marker, metadata_start)) >= 0
        )
        if not computation_starts:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILER_HLO_INVALID")
        canonical = canonical[:metadata_start] + canonical[min(computation_starts) :]
    return re.sub(r" stack_frame_id=\d+", "", canonical)


class SurfaceCompileStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SurfaceEvidenceKind(StrEnum):
    UNPROFILED_TIMING = "unprofiled_timing"
    TRACE = "trace"
    COUNTERS = "counters"


class SurfacePhase(StrEnum):
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    CALIBRATION = "calibration"
    CALIBRATION_SEALED = "calibration_sealed"
    HOLDOUT = "holdout"


class MatmulCollectiveSurfaceSourceAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch: str
    origin_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_main_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    remote_url: str
    compilation_source_root: str
    runtime: dict[str, str | None]
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies: tuple[SourceFileContract, ...] = Field(min_length=1)

    @computed_field
    @property
    def authority_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceInputIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    lhs_seed_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rhs_seed_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CompileCaptureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    repetition: int = Field(ge=1, le=2)
    input_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: SurfaceCompileStatus
    stablehlo: str = Field(min_length=1)
    compiler_hlo: str = Field(min_length=1)
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    error_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def status_matches_error(self) -> CompileCaptureRecord:
        if (self.status is SurfaceCompileStatus.FAILED) != (self.error_sha256 is not None):
            raise ValueError("Matmul collective surface compile status/error mismatch")
        return self


class MatmulCollectiveSurfaceCompileReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-compile-v1"] = (
        MATMUL_COLLECTIVE_SURFACE_COMPILE_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captures: tuple[CompileCaptureRecord, ...] = Field(min_length=1)

    @computed_field
    @property
    def report_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    evidence_kind: SurfaceEvidenceKind
    median_ns: float = Field(gt=0)


class SurfaceCorrectnessObservation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    pattern: str
    input_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    absolute_tolerance: float = Field(ge=0)
    relative_tolerance: float = Field(ge=0)
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)
    mismatched_element_count: int = Field(ge=0)

    @computed_field
    @property
    def passed(self) -> bool:
        return self.mismatched_element_count == 0


class SurfaceCorrectnessArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: tuple[SurfaceCorrectnessObservation, ...] = Field(
        min_length=200,
        max_length=200,
    )

    @computed_field
    @property
    def artifact_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCalibrationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: tuple[SurfaceMeasurement, ...] = Field(min_length=32, max_length=32)

    @computed_field
    @property
    def artifact_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceHoldoutPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    predicted_median_ns: float = Field(gt=0)


class MatmulCollectiveSurfaceCalibrationSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-seal-v1"] = (
        MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SEAL_SCHEMA
    )
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_batch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: tuple[SurfaceMeasurement, ...] = Field(min_length=32, max_length=32)
    model: MatmulCollectiveSurfaceModel
    holdout_predictions: tuple[SurfaceHoldoutPrediction, ...] = Field(
        min_length=8,
        max_length=8,
    )

    @computed_field
    @property
    def semantic_sha256(self) -> str:
        return model_identity_sha256(self)


class CalibrationSealEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seal: MatmulCollectiveSurfaceCalibrationSeal
    seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def outer_hash_matches(self) -> CalibrationSealEnvelope:
        if self.seal_sha256 != self.seal.semantic_sha256:
            raise ValueError("Matmul collective surface calibration seal hash mismatch")
        return self


class SurfacePhaseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(gt=0)
    phase: SurfacePhase
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SurfacePhaseLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    events: tuple[SurfacePhaseEvent, ...] = ()

    @computed_field
    @property
    def current_phase(self) -> SurfacePhase | None:
        return self.events[-1].phase if self.events else None

    @model_validator(mode="after")
    def history_is_canonical(self) -> SurfacePhaseLedger:
        if len(self.events) > len(_PHASE_ORDER):
            raise ValueError("Matmul collective surface phase ledger is not canonical")
        for index, event in enumerate(self.events):
            if event.sequence != index + 1 or event.phase is not _PHASE_ORDER[index]:
                raise ValueError("Matmul collective surface phase ledger is not canonical")
        return self


_PHASE_ORDER = (
    SurfacePhase.COMPILE,
    SurfacePhase.CORRECTNESS,
    SurfacePhase.CALIBRATION,
    SurfacePhase.CALIBRATION_SEALED,
    SurfacePhase.HOLDOUT,
)


def _revalidate_model[ModelT: BaseModel](value: ModelT) -> ModelT:
    return cast(
        ModelT,
        type(value).model_validate(value.model_dump(mode="python", exclude_computed_fields=True)),
    )


def derive_surface_input_identities(
    contract: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceInputIdentity, ...]:
    contract = _revalidate_model(contract)
    identities = []
    for scenario in contract.scenarios:
        workload = semantic_sha256(
            "matmul-collective-surface-input",
            contract.design_id,
            scenario.name,
            str(scenario.m),
            str(scenario.k),
            str(scenario.n),
        )
        lhs = semantic_sha256(workload, "lhs", contract.input_dtype)
        rhs = semantic_sha256(workload, "rhs", contract.input_dtype)
        identities.append(
            SurfaceInputIdentity(
                scenario_name=scenario.name,
                lhs_seed_identity_sha256=lhs,
                rhs_seed_identity_sha256=rhs,
                input_contract_sha256=semantic_sha256(lhs, rhs),
            )
        )
    return tuple(identities)


def make_compile_capture_record(
    *,
    scenario_name: str,
    strategy: MatmulCollectiveStrategy,
    repetition: int,
    input_contract_sha256: str,
    distributed_schedule_sha256: str,
    physical_schedule_sha256: str,
    pallas_source_sha256: str,
    stablehlo: str,
    compiler_hlo: str,
) -> CompileCaptureRecord:
    stablehlo = stablehlo.rstrip("\n") + "\n"
    compiler_hlo = compiler_hlo.rstrip("\n") + "\n"
    _validate_matmul_compiler_strategy(stablehlo, compiler_hlo, strategy)
    return CompileCaptureRecord(
        scenario_name=scenario_name,
        strategy=strategy,
        repetition=repetition,
        input_contract_sha256=input_contract_sha256,
        distributed_schedule_sha256=distributed_schedule_sha256,
        physical_schedule_sha256=physical_schedule_sha256,
        pallas_source_sha256=pallas_source_sha256,
        status=SurfaceCompileStatus.SUCCEEDED,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
        stablehlo_sha256=_text_sha256(stablehlo),
        semantic_stablehlo_sha256=_text_sha256(stablehlo),
        compiler_hlo_sha256=_text_sha256(compiler_hlo),
        semantic_compiler_hlo_sha256=_text_sha256(_semantic_compiler_hlo(compiler_hlo)),
    )


@lru_cache(maxsize=4)
def _read_committed_source_blob_items(
    repository_root: str,
    source_commit: str,
) -> tuple[tuple[str, bytes], ...]:
    root = Path(repository_root)
    if not root.is_dir():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_SOURCE_REPOSITORY_UNAVAILABLE")
    blobs = []
    for path in (*SURFACE_EXECUTABLE_DEPENDENCIES, "uv.lock"):
        repository_path = path if path == "uv.lock" else f"src/{path}"
        try:
            blob = subprocess.run(
                ["git", "show", f"{source_commit}:{repository_path}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"MATMUL_COLLECTIVE_SURFACE_SOURCE_COMMIT_UNAVAILABLE path={path}"
            ) from error
        blobs.append((path, blob))
    return tuple(blobs)


def _read_committed_source_blobs(
    repository_root: Path,
    source_commit: str,
) -> dict[str, bytes]:
    return dict(
        _read_committed_source_blob_items(
            str(repository_root.resolve()),
            source_commit,
        )
    )


def validate_surface_source_authority(
    authority: MatmulCollectiveSurfaceSourceAuthority,
    contract: MatmulCollectiveSurfaceDesignContract,
    source_blobs: Mapping[str, bytes],
) -> None:
    authority = _revalidate_model(authority)
    contract = _revalidate_model(contract)
    if (
        authority.branch != contract.source_branch
        or authority.source_commit != authority.origin_main_commit
        or authority.source_commit != authority.remote_main_commit
        or authority.remote_url != contract.source_remote_url
        or authority.compilation_source_root != contract.compilation_source_root
        or authority.runtime != contract.runtime
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_SOURCE_AUTHORITY_MISMATCH")
    paths = tuple(value.path for value in authority.dependencies)
    if paths != SURFACE_EXECUTABLE_DEPENDENCIES or tuple(source_blobs) != (*paths, "uv.lock"):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_SOURCE_DEPENDENCY_INVENTORY_MISMATCH")
    for dependency in authority.dependencies:
        if hashlib.sha256(source_blobs[dependency.path]).hexdigest() != dependency.sha256:
            raise ValueError(
                f"MATMUL_COLLECTIVE_SURFACE_SOURCE_DEPENDENCY_MISMATCH path={dependency.path}"
            )
    if hashlib.sha256(source_blobs["uv.lock"]).hexdigest() != authority.uv_lock_sha256:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_UV_LOCK_MISMATCH")
    committed_blobs = _read_committed_source_blobs(
        Path(authority.compilation_source_root),
        authority.source_commit,
    )
    if committed_blobs != dict(source_blobs):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_SOURCE_COMMIT_BLOB_MISMATCH")


def capture_surface_source_authority(
    repository_root: Path,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> tuple[MatmulCollectiveSurfaceSourceAuthority, dict[str, bytes]]:
    if repository_root.resolve() != Path(contract.compilation_source_root):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILATION_ROOT_MISMATCH")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain=v1")
    if status:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_SOURCE_DIRTY")
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    origin_main = git("rev-parse", "origin/main")
    remote_url = git("remote", "get-url", "origin")
    remote_record = git("ls-remote", "origin", "refs/heads/main").split()
    if len(remote_record) != 2 or remote_record[1] != "refs/heads/main":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_REMOTE_MAIN_UNAVAILABLE")
    blobs = _read_committed_source_blobs(repository_root, commit)
    uv_lock = blobs["uv.lock"]
    authority = MatmulCollectiveSurfaceSourceAuthority(
        source_commit=commit,
        branch=branch,
        origin_main_commit=origin_main,
        remote_main_commit=remote_record[0],
        remote_url=remote_url,
        compilation_source_root=str(repository_root.resolve()),
        runtime=_runtime_identity().model_dump(mode="python"),
        uv_lock_sha256=hashlib.sha256(uv_lock).hexdigest(),
        dependencies=tuple(
            SourceFileContract(path=path, sha256=hashlib.sha256(blob).hexdigest())
            for path, blob in blobs.items()
            if path != "uv.lock"
        ),
    )
    validate_surface_source_authority(authority, contract, blobs)
    return authority, blobs


def _validate_stablehlo_static_abi(
    capture: CompileCaptureRecord,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> None:
    scenario = next(
        (value for value in contract.scenarios if value.name == capture.scenario_name),
        None,
    )
    if scenario is None:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_SCENARIO_UNKNOWN")
    uncommented = "\n".join(
        line for line in capture.stablehlo.splitlines() if not line.lstrip().startswith("//")
    )
    signatures = re.findall(
        r"func\.func\s+public\s+@main\s*\((.*?)\)\s*->\s*\(?\s*(tensor<[^>]+>)",
        uncommented,
        flags=re.DOTALL,
    )
    if len(signatures) != 1:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_STABLEHLO_MAIN_ABI_INVALID")
    parameters, result = signatures[0]
    parameter_types = tuple(re.findall(r"tensor<[^>]+>", parameters))
    observed = (*parameter_types, result)
    expected = (
        f"tensor<{scenario.m}x{scenario.k}xbf16>",
        f"tensor<{scenario.k}x{scenario.n}xbf16>",
        f"tensor<{scenario.m}x{scenario.n}xf32>",
    )
    if observed != expected:
        raise ValueError(
            f"MATMUL_COLLECTIVE_SURFACE_STABLEHLO_STATIC_ABI_MISMATCH scenario={scenario.name}"
        )


def _validate_compiler_hlo_static_abi(
    capture: CompileCaptureRecord,
    contract: MatmulCollectiveSurfaceDesignContract,
) -> None:
    scenario = next(
        (value for value in contract.scenarios if value.name == capture.scenario_name),
        None,
    )
    if scenario is None:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_SCENARIO_UNKNOWN")
    header_lines = tuple(
        line.strip()
        for line in capture.compiler_hlo.splitlines()
        if line.strip().startswith("HloModule ")
    )
    if len(header_lines) != 1 or "entry_computation_layout=" not in header_lines[0]:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILER_HLO_MAIN_ABI_INVALID")
    header = header_lines[0]
    observed = tuple(
        (dtype, tuple(int(value) for value in dimensions.split(",") if value))
        for dtype, dimensions in re.findall(r"\b(bf16|f32)\[([0-9,]*)\]", header)
    )
    expected = (
        ("bf16", (scenario.m, scenario.k)),
        ("bf16", (scenario.k, scenario.n)),
        ("f32", (scenario.m, scenario.n)),
    )
    if observed != expected:
        raise ValueError(
            f"MATMUL_COLLECTIVE_SURFACE_COMPILER_HLO_STATIC_ABI_MISMATCH scenario={scenario.name}"
        )


def validate_compile_capture_report(
    report: MatmulCollectiveSurfaceCompileReport,
    contract: MatmulCollectiveSurfaceDesignContract,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    report = _revalidate_model(report)
    contract = _revalidate_model(contract)
    source = _revalidate_model(source)
    validate_surface_source_authority(source, contract, source_blobs)
    if (
        report.design_id != contract.design_id
        or report.source_authority_sha256 != source.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_AUTHORITY_MISMATCH")
    expected = tuple(
        (scenario.name, strategy, repetition)
        for scenario in contract.scenarios
        for strategy in contract.strategies
        for repetition in range(1, contract.compiler_capture_repetitions + 1)
    )
    observed = tuple(
        (value.scenario_name, value.strategy, value.repetition) for value in report.captures
    )
    if observed != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_INVENTORY_MISMATCH")
    design = derive_matmul_collective_surface_design_report(contract)
    design_arms = {(value.scenario_name, value.strategy): value for value in design.arms}
    identity_records = derive_surface_input_identities(contract)
    identities = {value.scenario_name: value.input_contract_sha256 for value in identity_records}
    if len(identities) != len(identity_records) or len(set(identities.values())) != len(identities):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_INPUT_CONTRACT_IDENTITY_COLLISION")
    by_arm: dict[tuple[str, MatmulCollectiveStrategy], list[CompileCaptureRecord]] = {}
    for capture in report.captures:
        if capture.status is SurfaceCompileStatus.FAILED:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_FAILED_NO_RETRY")
        if capture.input_contract_sha256 != identities[capture.scenario_name]:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_INPUT_IDENTITY_MISMATCH")
        arm = design_arms[(capture.scenario_name, capture.strategy)]
        if (
            capture.distributed_schedule_sha256 != arm.distributed_schedule_sha256
            or capture.physical_schedule_sha256 != arm.physical_schedule_sha256
            or capture.pallas_source_sha256 != arm.pallas_source_sha256
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_COMPILE_ARM_IDENTITY_MISMATCH")
        if (
            capture.stablehlo_sha256 != _text_sha256(capture.stablehlo)
            or capture.semantic_stablehlo_sha256
            != _text_sha256(capture.stablehlo.rstrip("\n") + "\n")
            or capture.compiler_hlo_sha256 != _text_sha256(capture.compiler_hlo)
            or capture.semantic_compiler_hlo_sha256
            != _text_sha256(_semantic_compiler_hlo(capture.compiler_hlo))
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_HLO_HASH_MISMATCH")
        _validate_stablehlo_static_abi(capture, contract)
        _validate_compiler_hlo_static_abi(capture, contract)
        _validate_matmul_compiler_strategy(
            capture.stablehlo,
            capture.compiler_hlo,
            capture.strategy,
        )
        by_arm.setdefault((capture.scenario_name, capture.strategy), []).append(capture)
    for key, repetitions in by_arm.items():
        stablehlo_hashes = {value.semantic_stablehlo_sha256 for value in repetitions}
        compiler_hashes = {value.semantic_compiler_hlo_sha256 for value in repetitions}
        if len(stablehlo_hashes) != 1 or len(compiler_hashes) != 1:
            raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_UNSTABLE_COMPILER_HASH arm={key}")


def validate_surface_correctness_artifact(
    artifact: SurfaceCorrectnessArtifact,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    artifact = _revalidate_model(artifact)
    contract = _revalidate_model(contract)
    compile_report = _revalidate_model(compile_report)
    source = _revalidate_model(source)
    validate_compile_capture_report(compile_report, contract, source, source_blobs)
    if (
        artifact.design_id != contract.design_id
        or artifact.compile_report_sha256 != compile_report.report_sha256
        or artifact.source_authority_sha256 != source.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_AUTHORITY_MISMATCH")
    expected = tuple(
        (scenario.name, strategy, pattern)
        for scenario in contract.scenarios
        for strategy in contract.strategies
        for pattern in contract.correctness_patterns
    )
    observed = tuple(
        (value.scenario_name, value.strategy, value.pattern) for value in artifact.observations
    )
    if observed != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_INVENTORY_MISMATCH")
    identities = {
        value.scenario_name: value.input_contract_sha256
        for value in derive_surface_input_identities(contract)
    }
    oracle_hashes: dict[tuple[str, str], set[str]] = {}
    for observation in artifact.observations:
        if (
            observation.input_contract_sha256 != identities[observation.scenario_name]
            or observation.absolute_tolerance != 1e-3
            or observation.relative_tolerance != 1e-3
            or observation.mismatched_element_count != 0
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_FAILED")
        oracle_hashes.setdefault(
            (observation.scenario_name, observation.pattern),
            set(),
        ).add(observation.oracle_output_sha256)
    if any(len(values) != 1 for values in oracle_hashes.values()):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ORACLE_MISMATCH")


def validate_surface_calibration_batch(
    batch: SurfaceCalibrationBatch,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    batch = _revalidate_model(batch)
    contract = _revalidate_model(contract)
    compile_report = _revalidate_model(compile_report)
    source = _revalidate_model(source)
    validate_compile_capture_report(compile_report, contract, source, source_blobs)
    if (
        batch.design_id != contract.design_id
        or batch.compile_report_sha256 != compile_report.report_sha256
        or batch.source_authority_sha256 != source.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_BATCH_AUTHORITY_MISMATCH")
    design = derive_matmul_collective_surface_design_report(contract)
    expected = tuple((value.scenario_name, value.strategy) for value in design.calibration_arms)
    observed = tuple((value.scenario_name, value.strategy) for value in batch.observations)
    if observed != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_INVENTORY_MISMATCH")
    if any(
        value.evidence_kind is not SurfaceEvidenceKind.UNPROFILED_TIMING
        for value in batch.observations
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_PROFILE_DATA_FORBIDDEN_IN_FIT")


def record_surface_phase(
    ledger: SurfacePhaseLedger,
    phase: SurfacePhase,
    artifact_sha256: str,
) -> SurfacePhaseLedger:
    ledger = _revalidate_model(ledger)
    expected = _PHASE_ORDER[len(ledger.events)] if len(ledger.events) < len(_PHASE_ORDER) else None
    if phase is not expected:
        current = ledger.current_phase.value if ledger.current_phase is not None else "none"
        raise ValueError(
            f"MATMUL_COLLECTIVE_SURFACE_PHASE_ORDER current={current} requested={phase.value}"
        )
    if phase is SurfacePhase.HOLDOUT:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_HOLDOUT_REQUIRES_VALIDATED_SEAL")
    return _append_surface_phase(ledger, phase, artifact_sha256)


def _append_surface_phase(
    ledger: SurfacePhaseLedger,
    phase: SurfacePhase,
    artifact_sha256: str,
) -> SurfacePhaseLedger:
    event = SurfacePhaseEvent(
        sequence=len(ledger.events) + 1,
        phase=phase,
        artifact_sha256=artifact_sha256,
    )
    return ledger.model_copy(update={"events": (*ledger.events, event)})


def _prediction(
    contract: MatmulCollectiveSurfaceDesignContract,
    arm: MatmulCollectiveSurfaceArmPlan,
    model: MatmulCollectiveSurfaceModel,
) -> float:
    coefficients = model.coefficients
    first = arm.strategy is contract.strategies[0]
    divisor = contract.feature_scale_divisor_ns
    predicted = (
        coefficients[0 if first else 1]
        + coefficients[2] * float(arm.compute_time_floor_ns) / divisor
        + coefficients[3] * float(arm.hbm_time_floor_ns) / divisor
        + coefficients[4 if first else 5] * float(arm.ici_time_floor_ns) / divisor
    )
    if not math.isfinite(predicted) or predicted <= 0:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_NONFINITE_PREDICTION")
    return predicted


def _replay_calibration_seal(
    seal: MatmulCollectiveSurfaceCalibrationSeal,
    contract: MatmulCollectiveSurfaceDesignContract,
    batch: SurfaceCalibrationBatch,
) -> None:
    if seal.observations != batch.observations:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_BATCH_SEAL_MISMATCH")
    fitted = fit_surface_model(
        contract,
        tuple(
            SurfaceCalibrationObservation(
                scenario_name=value.scenario_name,
                strategy=value.strategy,
                median_ns=value.median_ns,
            )
            for value in seal.observations
        ),
    )
    if fitted != seal.model:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_MODEL_REPLAY_MISMATCH")
    design = derive_matmul_collective_surface_design_report(contract)
    holdouts = tuple(value for value in design.arms if value not in design.calibration_arms)
    expected_predictions = tuple(
        SurfaceHoldoutPrediction(
            scenario_name=arm.scenario_name,
            strategy=arm.strategy,
            predicted_median_ns=_prediction(contract, arm, fitted),
        )
        for arm in holdouts
    )
    if seal.holdout_predictions != expected_predictions:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_HOLDOUT_PREDICTION_REPLAY_MISMATCH")


def seal_surface_calibration(
    ledger: SurfacePhaseLedger,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
    correctness: SurfaceCorrectnessArtifact,
    batch: SurfaceCalibrationBatch,
) -> tuple[SurfacePhaseLedger, CalibrationSealEnvelope]:
    ledger = _revalidate_model(ledger)
    contract = _revalidate_model(contract)
    compile_report = _revalidate_model(compile_report)
    source = _revalidate_model(source)
    correctness = _revalidate_model(correctness)
    batch = _revalidate_model(batch)
    if ledger.current_phase is not SurfacePhase.CALIBRATION:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_PHASE_ORDER calibration seal unavailable")
    validate_compile_capture_report(compile_report, contract, source, source_blobs)
    validate_surface_correctness_artifact(
        correctness,
        contract,
        compile_report,
        source,
        source_blobs,
    )
    validate_surface_calibration_batch(batch, contract, compile_report, source, source_blobs)
    expected_ledger_hashes = (
        compile_report.report_sha256,
        correctness.artifact_sha256,
        batch.artifact_sha256,
    )
    if tuple(value.artifact_sha256 for value in ledger.events) != expected_ledger_hashes:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_PHASE_ARTIFACT_MISMATCH")
    model = fit_surface_model(
        contract,
        tuple(
            SurfaceCalibrationObservation(
                scenario_name=value.scenario_name,
                strategy=value.strategy,
                median_ns=value.median_ns,
            )
            for value in batch.observations
        ),
    )
    design = derive_matmul_collective_surface_design_report(contract)
    holdouts = tuple(value for value in design.arms if value not in design.calibration_arms)
    seal = MatmulCollectiveSurfaceCalibrationSeal(
        design_id=contract.design_id,
        compile_report_sha256=compile_report.report_sha256,
        source_authority_sha256=source.authority_sha256,
        correctness_artifact_sha256=correctness.artifact_sha256,
        calibration_batch_sha256=batch.artifact_sha256,
        observations=batch.observations,
        model=model,
        holdout_predictions=tuple(
            SurfaceHoldoutPrediction(
                scenario_name=arm.scenario_name,
                strategy=arm.strategy,
                predicted_median_ns=_prediction(contract, arm, model),
            )
            for arm in holdouts
        ),
    )
    envelope = CalibrationSealEnvelope(seal=seal, seal_sha256=seal.semantic_sha256)
    return (
        record_surface_phase(
            ledger,
            SurfacePhase.CALIBRATION_SEALED,
            envelope.seal_sha256,
        ),
        envelope,
    )


def validate_calibration_seal(
    envelope: CalibrationSealEnvelope,
    ledger: SurfacePhaseLedger,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
    correctness: SurfaceCorrectnessArtifact,
    batch: SurfaceCalibrationBatch,
) -> None:
    envelope = _revalidate_model(envelope)
    ledger = _revalidate_model(ledger)
    contract = _revalidate_model(contract)
    compile_report = _revalidate_model(compile_report)
    source = _revalidate_model(source)
    correctness = _revalidate_model(correctness)
    batch = _revalidate_model(batch)
    validate_compile_capture_report(compile_report, contract, source, source_blobs)
    validate_surface_correctness_artifact(
        correctness,
        contract,
        compile_report,
        source,
        source_blobs,
    )
    validate_surface_calibration_batch(batch, contract, compile_report, source, source_blobs)
    seal = envelope.seal
    expected_ledger_hashes = (
        compile_report.report_sha256,
        correctness.artifact_sha256,
        batch.artifact_sha256,
        envelope.seal_sha256,
    )
    if (
        tuple(value.artifact_sha256 for value in ledger.events) != expected_ledger_hashes
        or envelope.seal_sha256 != seal.semantic_sha256
        or seal.design_id != contract.design_id
        or seal.compile_report_sha256 != compile_report.report_sha256
        or seal.source_authority_sha256 != source.authority_sha256
        or seal.correctness_artifact_sha256 != correctness.artifact_sha256
        or seal.calibration_batch_sha256 != batch.artifact_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SEAL_AUTHORITY_MISMATCH")
    _replay_calibration_seal(seal, contract, batch)


def begin_surface_holdout(
    ledger: SurfacePhaseLedger,
    seal_path: Path,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
    correctness: SurfaceCorrectnessArtifact,
    batch: SurfaceCalibrationBatch,
) -> SurfacePhaseLedger:
    ledger = _revalidate_model(ledger)
    contract = _revalidate_model(contract)
    compile_report = _revalidate_model(compile_report)
    source = _revalidate_model(source)
    correctness = _revalidate_model(correctness)
    batch = _revalidate_model(batch)
    envelope = CalibrationSealEnvelope.model_validate_json(seal_path.read_text())
    if (
        ledger.current_phase is not SurfacePhase.CALIBRATION_SEALED
        or ledger.events[-1].artifact_sha256 != envelope.seal_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_HOLDOUT_SEAL_LEDGER_MISMATCH")
    replay_calibration_seal(
        seal_path,
        ledger,
        contract,
        compile_report,
        source,
        source_blobs,
        correctness,
        batch,
    )
    return _append_surface_phase(ledger, SurfacePhase.HOLDOUT, envelope.seal_sha256)


def create_surface_attempt_root(root: Path, attempt_id: str) -> None:
    SurfacePhaseLedger(attempt_id=attempt_id)
    try:
        root.mkdir(parents=False, mode=0o700)
    except FileExistsError:
        raise ValueError(f"MATMUL_COLLECTIVE_SURFACE_ATTEMPT_ROOT_EXISTS path={root}") from None
    payload = json.dumps({"attempt_id": attempt_id}, sort_keys=True) + "\n"
    descriptor = os.open(
        root / "attempt.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_model_exclusive(path: Path, model: BaseModel) -> None:
    model = _revalidate_model(model)
    payload = (
        json.dumps(
            model.model_dump(mode="json", exclude_computed_fields=True),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except FileExistsError:
        raise ValueError(
            f"MATMUL_COLLECTIVE_SURFACE_IMMUTABLE_ARTIFACT_EXISTS path={path}"
        ) from None
    with os.fdopen(descriptor, "w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_compile_capture_report(
    path: Path,
    report: MatmulCollectiveSurfaceCompileReport,
    contract: MatmulCollectiveSurfaceDesignContract,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    validate_compile_capture_report(report, contract, source, source_blobs)
    _write_model_exclusive(path, report)


def write_surface_correctness_artifact(
    path: Path,
    artifact: SurfaceCorrectnessArtifact,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    validate_surface_correctness_artifact(
        artifact,
        contract,
        compile_report,
        source,
        source_blobs,
    )
    _write_model_exclusive(path, artifact)


def write_surface_calibration_batch(
    path: Path,
    batch: SurfaceCalibrationBatch,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
) -> None:
    validate_surface_calibration_batch(batch, contract, compile_report, source, source_blobs)
    _write_model_exclusive(path, batch)


def replay_compile_capture_report(
    path: Path,
    contract: MatmulCollectiveSurfaceDesignContract,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
) -> MatmulCollectiveSurfaceCompileReport:
    report = MatmulCollectiveSurfaceCompileReport.model_validate_json(path.read_text())
    validate_compile_capture_report(report, contract, source, source_blobs)
    return report


def write_calibration_seal(
    path: Path,
    envelope: CalibrationSealEnvelope,
    ledger: SurfacePhaseLedger,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
    correctness: SurfaceCorrectnessArtifact,
    batch: SurfaceCalibrationBatch,
) -> None:
    validate_calibration_seal(
        envelope,
        ledger,
        contract,
        compile_report,
        source,
        source_blobs,
        correctness,
        batch,
    )
    _write_model_exclusive(path, envelope)


def replay_calibration_seal(
    path: Path,
    ledger: SurfacePhaseLedger,
    contract: MatmulCollectiveSurfaceDesignContract,
    compile_report: MatmulCollectiveSurfaceCompileReport,
    source: MatmulCollectiveSurfaceSourceAuthority,
    source_blobs: Mapping[str, bytes],
    correctness: SurfaceCorrectnessArtifact,
    batch: SurfaceCalibrationBatch,
) -> CalibrationSealEnvelope:
    envelope = CalibrationSealEnvelope.model_validate_json(path.read_text())
    validate_calibration_seal(
        envelope,
        ledger,
        contract,
        compile_report,
        source,
        source_blobs,
        correctness,
        batch,
    )
    return envelope
