from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from xdsl.dialects.builtin import BFloat16Type, Float32Type
from xprof import profile_data

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    ProfileExpectation,
    RuntimeIdentity,
    SourceFileContract,
    WorkloadStage,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import BufferType, MxuEinsumOp
from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.metrics import MetricSource
from tpu_cake.runner import (
    RunMode,
    _profiler_contract,
    _profiler_options,
    _runtime_identity,
    _source_state,
)
from tpu_cake.seqax_cost_model import SeqaxCostModelReport, estimate_seqax_forward
from tpu_cake.seqax_pallas_runner import _physical_collective_counts
from tpu_cake.seqax_pallas_search import (
    SeqaxPallasRoundObservation,
    SeqaxPallasSearchContract,
    SeqaxPallasSearchReceipt,
    SeqaxPallasSearchResult,
    candidate_statistics,
    default_seqax_pallas_search_contract,
    execution_orders,
)
from tpu_cake.seqax_pallas_search_runner import (
    _compile_candidate,
    _compiler_tile_metadata,
    _execute,
    _resident_inputs,
    _search_source_manifest,
    prepare_seqax_pallas_candidates,
)
from tpu_cake.seqax_runner import expected_seqax_profiler_contract
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs
from tpu_cake.xprof_evidence import assess_capture
from tpu_cake.xprof_export import XProfExportManifest, export_xprof_capture

SEQAX_PALLAS_DIAGNOSTIC_SCHEMA = "seqax-pallas-incumbent-diagnostic-v1"
SEQAX_PALLAS_DIAGNOSTIC_WARMUPS = 5
SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS = 50
SEQAX_PALLAS_CANONICAL_SEARCH_ID = (
    "704fe1cdf7958216dc2295d54ff470f5092895eafc65c987af68dece02620f97"
)
SEQAX_PALLAS_CANONICAL_SEARCH_RECEIPT_SHA256 = (
    "01854528b255fecc942c1a7c54a78170182920b9e21f769aaf2136f955d0737f"
)
SEQAX_PALLAS_CANONICAL_SEARCH_COMMIT = "c6b23b154fcee7d167803218f1a4683cc715b956"
_STEP_EVENT = "seqax_pallas_incumbent_diagnostic"
_PROFILE_MARKERS = ("pallas_call", "all-gather", "reduce_scatter")


class SeqaxPallasDiagnosticContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_schema: str
    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: str
    timing_seed: int
    warmup_iterations: int = Field(ge=1)
    measured_iterations: int = Field(ge=1)
    runtime: RuntimeIdentity
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)

    @model_validator(mode="after")
    def protocol_is_fixed(self) -> SeqaxPallasDiagnosticContract:
        if (
            self.diagnostic_schema != SEQAX_PALLAS_DIAGNOSTIC_SCHEMA
            or self.search_id != SEQAX_PALLAS_CANONICAL_SEARCH_ID
            or self.search_receipt_sha256 != SEQAX_PALLAS_CANONICAL_SEARCH_RECEIPT_SHA256
            or self.candidate != "incumbent"
            or self.warmup_iterations != SEQAX_PALLAS_DIAGNOSTIC_WARMUPS
            or self.measured_iterations != SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS
            or (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8)
        ):
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_PROTOCOL_MISMATCH")
        return self


class SeqaxPallasDiagnosticDevice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=0)
    process_index: int = Field(ge=0)
    platform: str
    device_kind: str


class SeqaxPallasRegionAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    region_index: int = Field(ge=0)
    hlo_op_name: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    contracting_dimensions: tuple[str, ...]
    tiles: tuple[int, int, int]
    local_flops: int = Field(gt=0)
    minimum_operand_result_bytes: int = Field(gt=0)
    arithmetic_intensity_flops_per_byte: float = Field(gt=0)
    observed_occurrences: int = Field(gt=0)
    observed_average_self_time_ns: float = Field(gt=0)
    advertised_compute_floor_ns: float = Field(gt=0)
    observed_to_compute_floor_ratio: float = Field(gt=0)

    @model_validator(mode="after")
    def values_are_finite(self) -> SeqaxPallasRegionAttribution:
        values = (
            self.arithmetic_intensity_flops_per_byte,
            self.observed_average_self_time_ns,
            self.advertised_compute_floor_ns,
            self.observed_to_compute_floor_ratio,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_REGION_NONFINITE")
        return self


class SeqaxPallasHloCategoryAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    row_count: int = Field(gt=0)
    average_self_time_sum_ns: float = Field(ge=0)

    @model_validator(mode="after")
    def value_is_finite(self) -> SeqaxPallasHloCategoryAttribution:
        if not math.isfinite(self.average_self_time_sum_ns):
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_CATEGORY_NONFINITE")
        return self


class SeqaxPallasDiagnosticAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    program_id: str
    module_execution_count: int = Field(gt=0)
    module_median_duration_ns: float = Field(gt=0)
    module_p90_duration_ns: float = Field(gt=0)
    pallas_average_self_time_sum_ns_per_device: float = Field(gt=0)
    collective_completion_average_self_time_sum_ns_per_device: float = Field(ge=0)
    cost_model_idealized_floor_ns: float = Field(gt=0)
    cost_model_materialized_hbm_floor_ns: float = Field(gt=0)
    module_to_idealized_floor_ratio: float = Field(gt=0)
    regions: tuple[SeqaxPallasRegionAttribution, ...]
    categories: tuple[SeqaxPallasHloCategoryAttribution, ...]
    interpretation: tuple[str, ...]

    @model_validator(mode="after")
    def values_are_finite(self) -> SeqaxPallasDiagnosticAttribution:
        values = (
            self.module_median_duration_ns,
            self.module_p90_duration_ns,
            self.pallas_average_self_time_sum_ns_per_device,
            self.collective_completion_average_self_time_sum_ns_per_device,
            self.cost_model_idealized_floor_ns,
            self.cost_model_materialized_hbm_floor_ns,
            self.module_to_idealized_floor_ratio,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_ATTRIBUTION_NONFINITE")
        return self


class SeqaxPallasDiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract: SeqaxPallasDiagnosticContract
    runtime: RuntimeIdentity
    device_kind: str
    device_count: int = Field(gt=0)
    devices: tuple[SeqaxPallasDiagnosticDevice, ...]
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: tuple[SourceFileContract, ...]
    distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_profiler_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counter_profiler_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: tuple[str, ...]
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_incumbent_parity: bool
    trace_step_count: int = Field(gt=0)
    counter_step_count: int = Field(gt=0)
    attribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counter_assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    periodic_counter_names: tuple[str, ...]
    periodic_counter_samples_per_core: dict[str, int]
    hbm_read_counter_names: int = Field(gt=0)
    hbm_write_counter_names: int = Field(gt=0)
    cycle_counter_names: int = Field(gt=0)


class SeqaxPallasDiagnosticReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_schema: str
    status: str = Field(pattern=r"^passed$")
    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[ArtifactReference, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "canonical.py",
        package / "contracts.py",
        package / "cost_model.py",
        package / "dialects" / "distributed_tensor.py",
        package / "dialects" / "tpu_schedule.py",
        package / "evidence.py",
        package / "frontend.py",
        package / "identity.py",
        package / "ledger.py",
        package / "metrics.py",
        package / "runner.py",
        package / "seqax_cost_model.py",
        package / "seqax_pallas_diagnostic.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_pallas_runner.py",
        package / "seqax_pallas_search.py",
        package / "seqax_pallas_search_runner.py",
        package / "seqax_physical_execution.py",
        package / "seqax_physical_lowering.py",
        package / "seqax_runner.py",
        package / "xprof_evidence.py",
        package / "xprof_export.py",
        package / "workloads" / "seqax_forward.py",
        package / "workloads" / "seqax_oracle.py",
    )
    return tuple(
        SourceFileContract(
            path=path.relative_to(package.parent).as_posix(),
            sha256=_sha256(path),
        )
        for path in paths
    )


def _require_safe_new_root(root: Path, search_root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == value or root in value.parents for value in protected):
        raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_UNSAFE_ROOT path={root}")
    if root == search_root or root in search_root.parents or search_root in root.parents:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_ROOT_OVERLAP")
    if root.exists():
        raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_ROOT_EXISTS path={root}")


def _require_clean_repository(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_SOURCE_DIRTY status={status}")


def _close_ledger(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    if path.with_name(f"{path.name}-shm").exists() or path.with_name(f"{path.name}-wal").exists():
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_LEDGER_SIDECAR")


def _canonical_assessment(value: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value))

    def normalize_paths(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "path" and isinstance(child, str):
                    item[key] = Path(child).name
                else:
                    normalize_paths(child)
        elif isinstance(item, list):
            for child in item:
                normalize_paths(child)

    normalize_paths(normalized)
    capture = normalized.get("capture")
    if isinstance(capture, dict):
        if isinstance(capture.get("timed_program_ids"), list):
            capture["timed_program_ids"] = sorted(capture["timed_program_ids"])
        programs = capture.get("programs")
        if isinstance(programs, list):
            for program in programs:
                if not isinstance(program, dict):
                    continue
                hlo = program.get("hlo")
                if isinstance(hlo, dict):
                    hlo.pop("sha256", None)
    return normalized


def _replay_search_with_recorded_validator(root: Path, commit: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="tpu-cake-search-validator-") as directory:
        source_root = Path(directory)
        archive = subprocess.run(
            ["git", "archive", "--format=tar", commit],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        archive_path = source_root / "source.tar"
        archive_path.write_bytes(archive)
        with tarfile.open(archive_path) as stream:
            stream.extractall(source_root, filter="data")
        archive_path.unlink()
        script = """
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from tpu_cake.seqax_pallas_search import SeqaxPallasSearchContract
import tpu_cake.seqax_pallas_search_runner as runner

root = Path(sys.argv[1])
commit = sys.argv[2]
contract = SeqaxPallasSearchContract.model_validate_json((root / "contract.json").read_text())
original_run = subprocess.run
def recorded_commit_run(args, *positional, **keywords):
    if tuple(args) == ("git", "rev-parse", "HEAD"):
        return SimpleNamespace(stdout=commit + "\\n", stderr="", returncode=0)
    return original_run(args, *positional, **keywords)
runner.subprocess.run = recorded_commit_run
runner.validate_seqax_pallas_search(root, contract)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root / "src")
        environment["JAX_PLATFORMS"] = "cpu"
        try:
            subprocess.run(
                [sys.executable, "-c", script, str(root), commit],
                cwd=source_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "no validator output").strip()
            raise ValueError(
                f"SEQAX_PALLAS_DIAGNOSTIC_RECORDED_SEARCH_REPLAY_FAILED detail={detail[-2000:]}"
            ) from error


def _expected_profile(*, counters: bool) -> ProfileExpectation:
    return ProfileExpectation(
        name="seqax-pallas-incumbent-diagnostic",
        stage=WorkloadStage.CONTROL,
        minimum_tpu_device_planes=8,
        require_tensor_core_activity=False,
        require_hbm_read_counters=counters,
        require_hbm_write_counters=counters,
        require_cycle_counters=counters,
        minimum_counter_device_planes=4 if counters else 0,
        required_timed_hlo_markers=_PROFILE_MARKERS,
    )


def _validate_counter_evidence(assessment: Any) -> None:
    counters = assessment.capture.counters
    if (
        set(counters.periodic_samples_per_tpu_core) != {"0", "2", "4", "6"}
        or any(value < 2 for value in counters.periodic_samples_per_tpu_core.values())
        or not any(name.startswith("COUNT_MXU_BUSY") for name in counters.periodic_counter_names)
        or counters.hbm_read_names <= 0
        or counters.hbm_write_names <= 0
        or counters.cycle_names <= 0
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_COUNTER_EVIDENCE_MISMATCH")


def _bound_program(assessment: Any) -> tuple[str, str]:
    matches = tuple(
        (program.program_id, program.name)
        for program in assessment.capture.programs
        if program.program_id in assessment.capture.timed_program_ids
        and program.timed_self_us > 0
        and all(program.marker_counts.get(marker, 0) > 0 for marker in _PROFILE_MARKERS)
    )
    if len(matches) != 1:
        raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_BOUND_PROGRAM_MISMATCH observed={matches}")
    return matches[0]


def _profile_replay(
    xplane: Path,
    program_name: str,
    *,
    step_event: str = _STEP_EVENT,
    iterations: int = SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS,
) -> tuple[int, tuple[float, ...]]:
    profile = profile_data.ProfileData.from_file(xplane)
    try:
        steps = 0
        durations = []
        for plane in profile.planes:
            for line in plane.lines:
                for event in line.events:
                    steps += event.name == step_event
                    if (
                        plane.name == "/device:TPU:0"
                        and line.name == "XLA Modules"
                        and event.name == program_name
                    ):
                        durations.append(float(event.duration_ns))
    finally:
        profile.close()
    if steps != iterations:
        raise ValueError(
            f"SEQAX_PALLAS_DIAGNOSTIC_STEP_COUNT_MISMATCH expected={iterations} observed={steps}"
        )
    if len(durations) != iterations or any(value <= 0 for value in durations):
        raise ValueError(
            "SEQAX_PALLAS_DIAGNOSTIC_MODULE_COUNT_MISMATCH "
            f"expected={iterations} observed={len(durations)}"
        )
    return steps, tuple(durations)


def _gviz_rows(path: Path) -> tuple[dict[str, Any], ...]:
    table = json.loads(path.read_text())
    columns = tuple(column["id"] for column in table["cols"])
    rows = []
    for row in table["rows"]:
        values = tuple(cell.get("v") for cell in row["c"])
        if len(values) != len(columns):
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_HLO_STATS_WIDTH_MISMATCH")
        rows.append(dict(zip(columns, values, strict=True)))
    return tuple(rows)


def _buffer_bytes(value: BufferType) -> int:
    element_type = value.storage.element_type
    if isinstance(element_type, BFloat16Type):
        width = 2
    elif isinstance(element_type, Float32Type):
        width = 4
    else:
        raise TypeError(f"SEQAX_PALLAS_DIAGNOSTIC_DTYPE_UNSUPPORTED dtype={element_type}")
    return math.prod(value.storage.get_shape()) * width


def _region_flops(operation: MxuEinsumOp) -> int:
    output = operation.accumulator.type
    lhs = operation.lhs.type
    if not isinstance(output, BufferType) or not isinstance(lhs, BufferType):
        raise TypeError("Seqax diagnostic expects physical buffer types")
    lhs_names = tuple(value.data for value in lhs.shape.dimensions)
    lhs_extents = dict(zip(lhs_names, lhs.storage.get_shape(), strict=True))
    contraction = math.prod(lhs_extents[value.data] for value in operation.contracting_dimensions)
    return 2 * math.prod(output.storage.get_shape()) * contraction


def _cost_metric(report: SeqaxCostModelReport, name: str) -> float:
    matches = tuple(metric for metric in report.metrics if metric.name == name)
    if len(matches) != 1:
        raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_COST_METRIC_MISSING name={name}")
    return float(matches[0].quantity.value)


def _attribution(
    *,
    physical: Any,
    program_id: str,
    durations: tuple[float, ...],
    hlo_stats: Path,
    cost_report: SeqaxCostModelReport,
    iterations: int = SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS,
) -> SeqaxPallasDiagnosticAttribution:
    rows = tuple(row for row in _gviz_rows(hlo_stats) if str(row["program_id"]) == program_id)
    operations = tuple(
        operation for operation in physical.walk() if isinstance(operation, MxuEinsumOp)
    )
    by_region: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not str(row.get("hlo_op_name", "")).startswith("seqax_named_einsum"):
            continue
        expression = str(row["hlo_op_expression"])
        region_match = re.search(r'"region_index"\s*:\s*(\d+)', expression)
        schedule_match = re.search(r'"schedule_sha256"\s*:\s*"([0-9a-f]{64})"', expression)
        tile_m_match = re.search(r'"tile_m"\s*:\s*(\d+)', expression)
        tile_k_match = re.search(r'"tile_k"\s*:\s*(\d+)', expression)
        tile_n_match = re.search(r'"tile_n"\s*:\s*(\d+)', expression)
        if any(
            value is None
            for value in (
                region_match,
                schedule_match,
                tile_m_match,
                tile_k_match,
                tile_n_match,
            )
        ):
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_REGION_METADATA_MISSING")
        assert region_match is not None
        index = int(region_match.group(1))
        if index in by_region:
            raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_REGION_DUPLICATE index={index}")
        by_region[index] = row
    if tuple(sorted(by_region)) != tuple(range(len(operations))):
        raise ValueError(
            "SEQAX_PALLAS_DIAGNOSTIC_REGION_SET_MISMATCH "
            f"expected={tuple(range(len(operations)))} observed={tuple(sorted(by_region))}"
        )
    hardware = tpu7x_tensorcore_rates()
    physical_schedule_sha256 = schedule_sha256(physical)
    regions = []
    for index, operation in enumerate(operations):
        row = by_region[index]
        lhs = operation.lhs.type
        rhs = operation.rhs.type
        output = operation.accumulator.type
        if not all(isinstance(value, BufferType) for value in (lhs, rhs, output)):
            raise TypeError("Seqax diagnostic expects physical buffer types")
        flops = _region_flops(operation)
        byte_count = _buffer_bytes(lhs) + _buffer_bytes(rhs) + _buffer_bytes(output)
        observed_ns = float(row["avg_self_time"]) * 1_000
        floor_ns = flops * 1_000_000_000 / hardware.compute_flops_per_second
        occurrences = int(row["occurrences"])
        expression = str(row["hlo_op_expression"])
        metadata = {
            name: re.search(pattern, expression)
            for name, pattern in {
                "schedule": r'"schedule_sha256"\s*:\s*"([0-9a-f]{64})"',
                "tile_m": r'"tile_m"\s*:\s*(\d+)',
                "tile_k": r'"tile_k"\s*:\s*(\d+)',
                "tile_n": r'"tile_n"\s*:\s*(\d+)',
            }.items()
        }
        if occurrences != iterations * 8:
            raise ValueError(
                "SEQAX_PALLAS_DIAGNOSTIC_REGION_OCCURRENCE_MISMATCH "
                f"region={index} observed={occurrences}"
            )
        if (
            row.get("category") != "custom-call"
            or "pallas_call" not in str(row.get("tf_op_name", ""))
            or any(value is None for value in metadata.values())
            or metadata["schedule"].group(1) != physical_schedule_sha256
            or tuple(int(metadata[name].group(1)) for name in ("tile_m", "tile_k", "tile_n"))
            != (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
        ):
            raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_REGION_IDENTITY_MISMATCH region={index}")
        regions.append(
            SeqaxPallasRegionAttribution(
                region_index=index,
                hlo_op_name=str(row["hlo_op_name"]),
                lhs_shape=lhs.storage.get_shape(),
                rhs_shape=rhs.storage.get_shape(),
                output_shape=output.storage.get_shape(),
                contracting_dimensions=tuple(
                    value.data for value in operation.contracting_dimensions
                ),
                tiles=(operation.tile_m.data, operation.tile_k.data, operation.tile_n.data),
                local_flops=flops,
                minimum_operand_result_bytes=byte_count,
                arithmetic_intensity_flops_per_byte=flops / byte_count,
                observed_occurrences=occurrences,
                observed_average_self_time_ns=observed_ns,
                advertised_compute_floor_ns=floor_ns,
                observed_to_compute_floor_ratio=observed_ns / floor_ns,
            )
        )
    category_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category") or "unknown")
        category_values[category].append(float(row.get("avg_self_time") or 0) * 1_000)
    categories = tuple(
        SeqaxPallasHloCategoryAttribution(
            category=category,
            row_count=len(values),
            average_self_time_sum_ns=sum(values),
        )
        for category, values in sorted(category_values.items())
    )
    all_gather_count, reduce_scatter_count = _physical_collective_counts(physical)
    if all_gather_count <= 0 or reduce_scatter_count <= 0:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_PHYSICAL_COLLECTIVES_MISSING")
    semantic_collective_rows = tuple(
        row for row in rows if row.get("category") in {"all-gather", "reduce-scatter"}
    )
    if not semantic_collective_rows or any(
        int(row["occurrences"]) != iterations or float(row["avg_self_time"]) <= 0
        for row in semantic_collective_rows
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_COLLECTIVE_ROWS_MISMATCH")
    collective_completion_rows = tuple(
        row
        for row in rows
        if row.get("category") == "async-done"
        and str(row.get("hlo_op_name", "")).startswith(("all-gather", "reduce-scatter"))
        and "call-done" in str(row.get("hlo_op_name", ""))
    )
    if len(collective_completion_rows) != len(semantic_collective_rows) or any(
        int(row["occurrences"]) != iterations * 8 or float(row["avg_self_time"]) <= 0
        for row in collective_completion_rows
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_COLLECTIVE_COMPLETION_ROWS_MISMATCH")
    collective_ns = sum(float(row["avg_self_time"]) * 1_000 for row in collective_completion_rows)
    pallas_ns = sum(value.observed_average_self_time_ns for value in regions)
    idealized_ns = _cost_metric(cost_report, "seqax_idealized_time_floor")
    materialized_ns = _cost_metric(cost_report, "seqax_materialized_hbm_time")
    module_median = float(statistics.median(durations))
    ordered = sorted(durations)
    module_p90 = ordered[round((len(ordered) - 1) * 0.9)]
    return SeqaxPallasDiagnosticAttribution(
        program_id=program_id,
        module_execution_count=len(durations),
        module_median_duration_ns=module_median,
        module_p90_duration_ns=module_p90,
        pallas_average_self_time_sum_ns_per_device=pallas_ns,
        collective_completion_average_self_time_sum_ns_per_device=collective_ns,
        cost_model_idealized_floor_ns=idealized_ns,
        cost_model_materialized_hbm_floor_ns=materialized_ns,
        module_to_idealized_floor_ratio=module_median / idealized_ns,
        regions=tuple(regions),
        categories=categories,
        interpretation=(
            "Per-region Pallas self time is XProf HLO average self time for one local-device invocation.",
            "Collective time sums only asynchronous completion rows; start and semantic opcode rows are not double-counted.",
            "Collective completion rows and local Pallas rows are diagnostic inventories, not an additive critical-path estimate.",
            "Analytical floors use advertised TPU7x rates and omit launch, vector, synchronization, and collective startup time.",
            "This diagnostic ranks measured boundaries; it is not a throughput benchmark or a promotion receipt.",
        ),
    )


def _export_xprof(profile_root: Path, output_root: Path) -> None:
    xplanes = tuple(profile_root.rglob("*.xplane.pb"))
    if len(xplanes) != 1:
        raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_XPLANE_COUNT observed={xplanes}")
    temporary = output_root.with_name(f"{output_root.name}.tmp")
    staging = temporary / ".xprof-input"
    staging.mkdir(parents=True)
    staged_xplane = staging / xplanes[0].name
    shutil.copy2(xplanes[0], staged_xplane)
    try:
        manifest = export_xprof_capture(staging, temporary)
        derived = temporary / "derived"
        derived.mkdir()
        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path == staged_xplane:
                continue
            if not path.name.endswith((".hlo_proto.pb", "ALL_HOSTS.op_stats_v2.pb")):
                raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_XPROF_DERIVED_UNRECOGNIZED path={path}")
            destination = derived / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, destination)
        shutil.rmtree(staging)
        portable = manifest.model_copy(
            update={
                "xplane": xplanes[0].relative_to(profile_root.parent),
                "exports": tuple(
                    export.model_copy(update={"output": Path("xprof") / export.output.name})
                    for export in manifest.exports
                ),
            }
        )
        (temporary / "manifest.json").write_text(
            json.dumps(portable.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _write_json(
            temporary / "derived_manifest.json",
            [
                {
                    "path": (Path("xprof") / path.relative_to(temporary)).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(derived.rglob("*"))
                if path.is_file()
            ],
        )
        if not (temporary / "hlo_stats.json").is_file():
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_HLO_STATS_MISSING")
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_xprof(root: Path, xplane: Path) -> None:
    output_root = root / "xprof"
    manifest = XProfExportManifest.model_validate_json((output_root / "manifest.json").read_text())
    if (
        manifest.xplane != xplane.relative_to(root)
        or tuple(sorted(set(manifest.available_tools))) != manifest.available_tools
        or len({value.tool for value in manifest.exports}) != len(manifest.exports)
        or not any(value.tool == "hlo_stats" for value in manifest.exports)
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_XPROF_MANIFEST_MISMATCH")
    for export in manifest.exports:
        suffix = ".json" if export.mime_type == "application/json" else ".bin"
        expected = Path("xprof") / f"{export.tool}{suffix}"
        path = root / expected
        if (
            export.output != expected
            or not path.is_file()
            or path.stat().st_size != export.size_bytes
        ):
            raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_XPROF_EXPORT_MISMATCH tool={export.tool}")
    derived = json.loads((output_root / "derived_manifest.json").read_text())
    declared = set()
    for value in derived:
        relative = value["path"]
        path = output_root / Path(relative).relative_to("xprof")
        if (
            relative in declared
            or not relative.startswith("xprof/derived/")
            or not path.is_file()
            or path.stat().st_size != value["size_bytes"]
            or _sha256(path) != value["sha256"]
        ):
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_XPROF_DERIVED_MISMATCH")
        declared.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in (output_root / "derived").rglob("*")
        if path.is_file()
    }
    if (
        declared != observed
        or not any(value.endswith(".hlo_proto.pb") for value in observed)
        or not any(value.endswith("ALL_HOSTS.op_stats_v2.pb") for value in observed)
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_XPROF_DERIVED_SET_MISMATCH")
    expected_files = {
        "xprof/manifest.json",
        "xprof/derived_manifest.json",
        *(value.output.as_posix() for value in manifest.exports),
        *declared,
    }
    observed_files = {
        path.relative_to(root).as_posix() for path in output_root.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_XPROF_CLOSED_WORLD_MISMATCH")


def _validate_xprof_replay(saved_root: Path, replay_root: Path) -> None:
    saved_paths = {
        path.relative_to(saved_root / "xprof").as_posix(): path
        for path in (saved_root / "xprof").rglob("*")
        if path.is_file()
    }
    replayed_paths = {
        path.relative_to(replay_root / "xprof").as_posix(): path
        for path in (replay_root / "xprof").rglob("*")
        if path.is_file()
    }
    if set(saved_paths) != set(replayed_paths):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_XPROF_REPLAY_MISMATCH")
    for relative, saved_path in saved_paths.items():
        if relative == "derived_manifest.json" or relative.startswith("derived/"):
            continue
        replayed_path = replayed_paths[relative]
        if relative == "op_profile.json":
            matches = _canonical_op_profile(saved_path) == _canonical_op_profile(replayed_path)
        else:
            matches = _sha256(saved_path) == _sha256(replayed_path)
        if not matches:
            raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_XPROF_REPLAY_MISMATCH path={relative}")
    saved_derived = tuple(
        (value["path"], value["size_bytes"])
        for value in json.loads((saved_root / "xprof" / "derived_manifest.json").read_text())
    )
    replayed_derived = tuple(
        (value["path"], value["size_bytes"])
        for value in json.loads((replay_root / "xprof" / "derived_manifest.json").read_text())
    )
    if saved_derived != replayed_derived:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_XPROF_DERIVED_REPLAY_MISMATCH")


def _canonical_op_profile(path: Path) -> object:
    def canonicalize(value: object) -> object:
        if isinstance(value, dict):
            canonical = {key: canonicalize(item) for key, item in value.items()}
            children = canonical.get("children")
            if isinstance(children, list):
                canonical["children"] = sorted(
                    children,
                    key=lambda child: json.dumps(
                        child,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            return canonical
        if isinstance(value, list):
            return [canonicalize(item) for item in value]
        return value

    return canonicalize(json.loads(path.read_text()))


def _profile_files(profile_root: Path) -> tuple[Path, tuple[Path, ...]]:
    files = tuple(sorted(path for path in profile_root.rglob("*") if path.is_file()))
    xplanes = tuple(path for path in files if path.name.endswith(".xplane.pb"))
    traces = tuple(path for path in files if path.name.endswith(".trace.json.gz"))
    if len(xplanes) != 1 or len(traces) != 1 or set(files) != {*xplanes, *traces}:
        raise ValueError(
            "SEQAX_PALLAS_DIAGNOSTIC_PROFILE_FILE_SET_MISMATCH "
            f"files={[path.name for path in files]}"
        )
    return xplanes[0], traces


def _capture_phase(
    phase_root: Path,
    compiled: Any,
    resident: tuple[jax.Array, ...],
    mode: RunMode,
    *,
    step_event: str = _STEP_EVENT,
    iterations: int = SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS,
) -> tuple[Path, Any, int, tuple[float, ...]]:
    profiler = _profiler_contract(mode)
    if profiler != expected_seqax_profiler_contract(mode):
        raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_PROFILER_CONTRACT_MISMATCH mode={mode.value}")
    _write_json(phase_root / "profiler_config.json", profiler)
    profile_root = phase_root / "profile"
    jax.profiler.start_trace(profile_root, profiler_options=_profiler_options(mode))
    try:
        for step in range(iterations):
            with jax.profiler.StepTraceAnnotation(step_event, step_num=step):
                jax.block_until_ready(compiled.compiled(*resident))
    finally:
        jax.profiler.stop_trace()
    xplane, _trace_files = _profile_files(profile_root)
    _export_xprof(profile_root, phase_root / "xprof")
    _validate_xprof(phase_root, xplane)
    assessment = assess_capture(phase_root, _expected_profile(counters=mode is RunMode.COUNTERS))
    if not assessment.accepted:
        raise ValueError(
            "SEQAX_PALLAS_DIAGNOSTIC_PROFILE_REJECTED "
            f"mode={mode.value} findings={[value.code for value in assessment.findings]}"
        )
    if mode is RunMode.COUNTERS:
        _validate_counter_evidence(assessment)
    _write_json(
        phase_root / "profile_assessment.json",
        _canonical_assessment(assessment.model_dump(mode="json")),
    )
    _program_id, program_name = _bound_program(assessment)
    steps, durations = _profile_replay(
        xplane,
        program_name,
        step_event=step_event,
        iterations=iterations,
    )
    return xplane, assessment, steps, durations


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    if relative.startswith("search/"):
        return ArtifactRole.SEARCH_EVIDENCE
    fixed = {
        "contract.json": ArtifactRole.EXPERIMENT,
        "result.json": ArtifactRole.TRACE_RESULT,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "source_manifest.json": ArtifactRole.BACKEND_MANIFEST,
        "trace/profiler_config.json": ArtifactRole.PROFILER_CONFIG,
        "counters/profiler_config.json": ArtifactRole.PROFILER_CONFIG,
        "trace/profile_assessment.json": ArtifactRole.PROFILE_ASSESSMENT,
        "counters/profile_assessment.json": ArtifactRole.PROFILE_ASSESSMENT,
        "attribution.json": ArtifactRole.SEARCH_EVIDENCE,
        "cost_model.json": ArtifactRole.COST_MODEL,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
    }
    if relative in fixed:
        return fixed[relative]
    if relative.startswith("inputs/"):
        return ArtifactRole.CORRECTNESS_INPUT
    if relative.startswith("trace/profile/") and relative.endswith(".xplane.pb"):
        return ArtifactRole.TIMING_TRACE
    if relative.startswith("counters/profile/") and relative.endswith(".xplane.pb"):
        return ArtifactRole.COUNTER_TRACE
    if relative.startswith(("trace/profile/", "counters/profile/")) and relative.endswith(
        ".trace.json.gz"
    ):
        return ArtifactRole.PROFILE_AUXILIARY
    if relative.startswith(("trace/xprof/", "counters/xprof/")):
        return ArtifactRole.XPROF_EXPORT
    raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_ARTIFACT_UNRECOGNIZED path={relative}")


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    artifacts = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root)
        if relative.as_posix() == "receipt.json":
            continue
        artifacts.append(
            ArtifactReference(
                path=relative.as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
                role=_artifact_role(relative),
            )
        )
    return tuple(artifacts)


def _validate_manifest(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    declared = tuple(value.path for value in artifacts)
    if len(declared) != len(set(declared)):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_ARTIFACT_DUPLICATE")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "receipt.json"
    }
    if set(declared) != observed:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_CLOSED_WORLD_MISMATCH")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SYMLINK")
    for artifact in artifacts:
        path = root / artifact.path
        if (
            path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != artifact.size_bytes
            or _sha256(path) != artifact.sha256
            or _artifact_role(Path(artifact.path)) is not artifact.role
        ):
            raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_ARTIFACT_MISMATCH path={artifact.path}")


def _validate_search_snapshot(
    root: Path,
    trusted_contract: SeqaxPallasSearchContract,
    *,
    replay_recorded: bool = True,
) -> SeqaxPallasSearchResult:
    root = root.resolve()
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_SYMLINK")
    saved_contract = SeqaxPallasSearchContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != trusted_contract:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_CONTRACT_MISMATCH")
    receipt = SeqaxPallasSearchReceipt.model_validate_json((root / "receipt.json").read_text())
    result = SeqaxPallasSearchResult.model_validate_json((root / "result.json").read_text())
    if (
        receipt.search_id != trusted_contract.search_id
        or receipt.result_sha256 != _sha256(root / "result.json")
        or receipt.ledger_sha256 != _sha256(root / "ledger.sqlite")
        or result.search_id != trusted_contract.search_id
        or result.runtime != trusted_contract.runtime
        or result.winner is not None
        or result.provisional_winner is not None
        or result.confirmation is not None
        or result.confirmation_rounds
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_RESULT_MISMATCH")
    declared = tuple(artifact.path for artifact in receipt.artifacts)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "receipt.json"
    }
    if len(declared) != len(set(declared)) or set(declared) != observed:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_CLOSED_WORLD_MISMATCH")
    for artifact in receipt.artifacts:
        path = root / artifact.path
        if (
            path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != artifact.size_bytes
            or _sha256(path) != artifact.sha256
        ):
            raise ValueError(
                f"SEQAX_PALLAS_DIAGNOSTIC_SEARCH_ARTIFACT_MISMATCH path={artifact.path}"
            )
    source_state = json.loads((root / "source_state.json").read_text())
    commit = source_state.get("git_commit")
    repository_root = Path(__file__).resolve().parents[2]
    if (
        not isinstance(commit, str)
        or commit != SEQAX_PALLAS_CANONICAL_SEARCH_COMMIT
        or source_state.get("git_dirty") is not False
        or source_state.get("git_status") != []
        or source_state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
        or (root / "source_diff.patch").read_bytes() != b""
        or result.source_state_sha256 != _sha256(root / "source_state.json")
        or result.search_source_manifest != _search_source_manifest()
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_SOURCE_STATE_MISMATCH")
    if replay_recorded:
        _replay_search_with_recorded_validator(root, commit)
    for source in result.search_source_manifest:
        blob = subprocess.run(
            ["git", "show", f"{commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(
                f"SEQAX_PALLAS_DIAGNOSTIC_SEARCH_SOURCE_BLOB_MISMATCH path={source.path}"
            )
    distributed, prepared = prepare_seqax_pallas_candidates(trusted_contract)
    if (root / "distributed.xdsl").read_text() != canonical_text(distributed):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_DISTRIBUTED_IR_MISMATCH")
    if len(result.plans) != len(prepared):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_PLAN_COUNT_MISMATCH")
    for record, expected in zip(result.plans, prepared, strict=True):
        plan_root = root / "plans" / expected.candidate.name
        expected_tiles = tuple(
            (index, expected.plan.physical_schedule_sha256, *tiles)
            for index, tiles in enumerate(expected.tiles)
        )
        if (
            record.name != expected.candidate.name
            or record.policy != expected.candidate.policy
            or record.tiles != expected.tiles
            or record.changed_region_count != expected.candidate.expected_changed_regions
            or record.physical_schedule_sha256 != expected.plan.physical_schedule_sha256
            or record.pallas_source_sha256 != expected.plan.source_sha256()
            or record.stablehlo_sha256 != _sha256(plan_root / "stablehlo.txt")
            or record.compiler_hlo_sha256 != _sha256(plan_root / "compiler_hlo.txt")
            or (plan_root / "physical.xdsl").read_text() != canonical_text(expected.physical)
            or (plan_root / "lowered_pallas.py").read_text()
            != expected.plan.render_executable_source()
            or _compiler_tile_metadata((plan_root / "compiler_hlo.txt").read_text())
            != expected_tiles
        ):
            raise ValueError(
                f"SEQAX_PALLAS_DIAGNOSTIC_SEARCH_PLAN_REPLAY_MISMATCH candidate={record.name}"
            )
    rounds = tuple(
        SeqaxPallasRoundObservation.model_validate(value)
        for value in json.loads((root / "rounds.json").read_text())
    )
    if (
        rounds != result.rounds
        or result.execution_orders != execution_orders(trusted_contract)
        or result.candidates != candidate_statistics(trusted_contract, rounds)
        or any(value.promotable for value in result.candidates)
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_SELECTION_REPLAY_MISMATCH")
    return result


def _diagnostic_contract(
    search_root: Path,
    search_contract: SeqaxPallasSearchContract,
) -> SeqaxPallasDiagnosticContract:
    receipt_path = search_root / "receipt.json"
    return SeqaxPallasDiagnosticContract(
        diagnostic_schema=SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
        search_id=search_contract.search_id,
        search_receipt_sha256=_sha256(receipt_path),
        candidate=search_contract.baseline,
        timing_seed=search_contract.timing_seed,
        warmup_iterations=SEQAX_PALLAS_DIAGNOSTIC_WARMUPS,
        measured_iterations=SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS,
        runtime=search_contract.runtime,
        backend=search_contract.backend,
        device_kind=search_contract.device_kind,
        device_count=search_contract.device_count,
    )


def run_seqax_pallas_incumbent_diagnostic(
    root: Path,
    search_root: Path,
    search_contract: SeqaxPallasSearchContract,
) -> SeqaxPallasDiagnosticReceipt:
    root = root.resolve()
    search_root = search_root.resolve()
    _require_safe_new_root(root, search_root)
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
    search_result = _validate_search_snapshot(search_root, search_contract)
    if search_result.winner is not None:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_REQUIRES_RETAINED_INCUMBENT")
    runtime = _runtime_identity()
    contract = _diagnostic_contract(search_root, search_contract)
    if runtime != contract.runtime:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_RUNTIME_MISMATCH")
    devices = tuple(jax.devices())
    if (
        jax.default_backend() != "tpu"
        or len(devices) != 8
        or any(device.platform != "tpu" for device in devices)
        or any(device.device_kind not in {"TPU7x", "TPU v7x"} for device in devices)
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_DEVICE_MISMATCH")
    root.mkdir(parents=True)
    shutil.copytree(search_root, root / "search")
    _validate_search_snapshot(root / "search", search_contract, replay_recorded=False)
    _write_json(root / "contract.json", contract.model_dump(mode="json"))
    _write_json(
        root / "source_manifest.json",
        [value.model_dump(mode="json") for value in _source_manifest()],
    )
    _source_state(repository_root, root)
    captured_source = json.loads((root / "source_state.json").read_text())
    if captured_source.get("git_dirty") is not False or captured_source.get("git_status") != []:
        raise ValueError(
            "SEQAX_PALLAS_DIAGNOSTIC_OUTPUT_DIRTY_SOURCE "
            f"status={captured_source.get('git_status')}"
        )
    for mode in (RunMode.TRACE, RunMode.COUNTERS):
        profiler = _profiler_contract(mode)
        if profiler != expected_seqax_profiler_contract(mode):
            raise ValueError(
                f"SEQAX_PALLAS_DIAGNOSTIC_PROFILER_CONTRACT_MISMATCH mode={mode.value}"
            )
        _write_json(root / mode.value / "profiler_config.json", profiler)
    run_id = semantic_sha256(
        SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
        contract.search_id,
        contract.search_receipt_sha256,
        _sha256(root / "source_state.json"),
        _sha256(root / "source_manifest.json"),
        _sha256(root / "trace" / "profiler_config.json"),
        _sha256(root / "counters" / "profiler_config.json"),
    )
    ledger_path = root / "ledger.sqlite"
    with ExperimentLedger(ledger_path) as ledger:
        ledger.create(run_id, {"contract": contract.model_dump(mode="json")})

    distributed, prepared = prepare_seqax_pallas_candidates(search_contract)
    incumbent = next(value for value in prepared if value.candidate.name == contract.candidate)
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.VERIFIED,
            {
                "distributed_schedule_sha256": incumbent.plan.distributed_schedule_sha256,
                "physical_schedule_sha256": incumbent.plan.physical_schedule_sha256,
            },
        )
        ledger.transition(
            run_id,
            RunState.LOWERED,
            {"pallas_source_sha256": incumbent.plan.source_sha256()},
        )

    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **search_contract.parameters,
        )
    )
    if arrays_sha256(host_inputs) != search_result.timing_input_sha256:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_INPUT_IDENTITY_MISMATCH")
    compiled = _compile_candidate(incumbent, host_inputs, devices)
    saved_plan = next(value for value in search_result.plans if value.name == contract.candidate)
    if (
        _sha256(root / "search" / "plans" / contract.candidate / "stablehlo.txt")
        != saved_plan.stablehlo_sha256
        or compiled.stablehlo + "\n"
        != (root / "search" / "plans" / contract.candidate / "stablehlo.txt").read_text()
        or compiled.compiler_hlo + "\n"
        != (root / "search" / "plans" / contract.candidate / "compiler_hlo.txt").read_text()
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_COMPILED_PROGRAM_MISMATCH")
    expected_tiles = tuple(
        (index, incumbent.plan.physical_schedule_sha256, *tiles)
        for index, tiles in enumerate(incumbent.tiles)
    )
    if _compiler_tile_metadata(compiled.compiler_hlo) != expected_tiles:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_COMPILER_TILE_MISMATCH")
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.COMPILED,
            {
                "stablehlo_sha256": saved_plan.stablehlo_sha256,
                "compiler_hlo_sha256": saved_plan.compiler_hlo_sha256,
            },
        )

    resident = _resident_inputs(host_inputs, compiled)
    actual = _execute(compiled, resident)
    expected_path = (
        root
        / "search"
        / "correctness"
        / str(contract.timing_seed)
        / "outputs"
        / f"{contract.candidate}.npy"
    )
    expected = np.load(expected_path, allow_pickle=False)
    exact = (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and np.array_equal(actual, expected)
    )
    if not exact:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_INCUMBENT_PARITY_FAILED")
    inputs_root = root / "inputs"
    inputs_root.mkdir()
    for index, value in enumerate(host_inputs):
        np.save(inputs_root / f"{index:02d}.npy", value, allow_pickle=False)
    np.save(root / "output.npy", actual, allow_pickle=False)
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.CORRECT,
            {
                "input_sha256": arrays_sha256(host_inputs),
                "output_sha256": array_sha256(actual),
                "expected_output_sha256": array_sha256(expected),
            },
        )

    for _ in range(contract.warmup_iterations):
        jax.block_until_ready(compiled.compiled(*resident))
    trace_xplane, trace_assessment, trace_steps, trace_durations = _capture_phase(
        root / "trace",
        compiled,
        resident,
        RunMode.TRACE,
    )
    counter_xplane, counter_assessment, counter_steps, _counter_durations = _capture_phase(
        root / "counters",
        compiled,
        resident,
        RunMode.COUNTERS,
    )
    trace_program_id, _trace_program_name = _bound_program(trace_assessment)
    counter_program_id, _counter_program_name = _bound_program(counter_assessment)
    if trace_program_id != counter_program_id:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_PROGRAM_ID_MISMATCH")
    cost_report = estimate_seqax_forward(
        distributed,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=incumbent.plan.distributed_schedule_sha256,
            artifact_path="search/distributed.xdsl",
            tool="tpu-cake",
            field="canonical distributed tensor program",
        ),
        expected_schedule_sha256=incumbent.plan.distributed_schedule_sha256,
    )
    _write_json(root / "cost_model.json", cost_report.model_dump(mode="json"))
    attribution = _attribution(
        physical=incumbent.physical,
        program_id=trace_program_id,
        durations=trace_durations,
        hlo_stats=root / "trace" / "xprof" / "hlo_stats.json",
        cost_report=cost_report,
    )
    _write_json(root / "attribution.json", attribution.model_dump(mode="json"))
    result = SeqaxPallasDiagnosticResult(
        run_id=run_id,
        contract=contract,
        runtime=runtime,
        device_kind="TPU7x",
        device_count=len(devices),
        devices=tuple(
            SeqaxPallasDiagnosticDevice(
                id=device.id,
                process_index=device.process_index,
                platform=device.platform,
                device_kind=device.device_kind,
            )
            for device in devices
        ),
        source_state_sha256=_sha256(root / "source_state.json"),
        source_manifest_sha256=_sha256(root / "source_manifest.json"),
        source_manifest=_source_manifest(),
        distributed_schedule_sha256=incumbent.plan.distributed_schedule_sha256,
        physical_schedule_sha256=incumbent.plan.physical_schedule_sha256,
        pallas_source_sha256=incumbent.plan.source_sha256(),
        stablehlo_sha256=saved_plan.stablehlo_sha256,
        compiler_hlo_sha256=saved_plan.compiler_hlo_sha256,
        trace_profiler_config_sha256=_sha256(root / "trace" / "profiler_config.json"),
        counter_profiler_config_sha256=_sha256(root / "counters" / "profiler_config.json"),
        input_sha256=arrays_sha256(host_inputs),
        output_sha256=array_sha256(actual),
        expected_output_sha256=array_sha256(expected),
        exact_incumbent_parity=exact,
        trace_step_count=trace_steps,
        counter_step_count=counter_steps,
        attribution_sha256=_sha256(root / "attribution.json"),
        cost_model_sha256=_sha256(root / "cost_model.json"),
        trace_assessment_sha256=_sha256(root / "trace" / "profile_assessment.json"),
        counter_assessment_sha256=_sha256(root / "counters" / "profile_assessment.json"),
        periodic_counter_names=counter_assessment.capture.counters.periodic_counter_names,
        periodic_counter_samples_per_core=(
            counter_assessment.capture.counters.periodic_samples_per_tpu_core
        ),
        hbm_read_counter_names=counter_assessment.capture.counters.hbm_read_names,
        hbm_write_counter_names=counter_assessment.capture.counters.hbm_write_names,
        cycle_counter_names=counter_assessment.capture.counters.cycle_names,
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.COUNTERED,
            {
                "trace_xplane_sha256": _sha256(trace_xplane),
                "counter_xplane_sha256": _sha256(counter_xplane),
                "trace_step_count": trace_steps,
                "counter_step_count": counter_steps,
                "attribution_sha256": result.attribution_sha256,
            },
        )
    _close_ledger(ledger_path)
    _validate_diagnostic(root, require_accepted=False)
    with ExperimentLedger(ledger_path) as ledger:
        ledger.transition(
            run_id,
            RunState.ACCEPTED,
            {"result_sha256": _sha256(root / "result.json")},
        )
    _close_ledger(ledger_path)
    receipt = SeqaxPallasDiagnosticReceipt(
        diagnostic_schema=SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
        status="passed",
        search_id=contract.search_id,
        run_id=run_id,
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(ledger_path),
        artifacts=_artifact_manifest(root),
    )
    _write_json(root / "receipt.json", receipt.model_dump(mode="json"))
    return validate_seqax_pallas_incumbent_diagnostic(root)


def _validate_source(root: Path, result: SeqaxPallasDiagnosticResult) -> None:
    manifest = tuple(
        SourceFileContract.model_validate(value)
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    if (
        manifest != result.source_manifest
        or manifest != _source_manifest()
        or result.source_state_sha256 != _sha256(root / "source_state.json")
        or result.source_manifest_sha256 != _sha256(root / "source_manifest.json")
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SOURCE_MANIFEST_MISMATCH")
    source_state = json.loads((root / "source_state.json").read_text())
    repository_root = Path(__file__).resolve().parents[2]
    commit = source_state.get("git_commit")
    if (
        not isinstance(commit, str)
        or source_state.get("git_dirty") is not False
        or source_state.get("git_status") != []
        or source_state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
        or (root / "source_diff.patch").read_bytes() != b""
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SOURCE_STATE_MISMATCH")
    for source in manifest:
        blob = subprocess.run(
            ["git", "show", f"{commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_SOURCE_BLOB_MISMATCH path={source.path}")


def _validate_diagnostic(root: Path, *, require_accepted: bool) -> SeqaxPallasDiagnosticReceipt:
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_ROOT_INVALID")
    saved_receipt = None
    if require_accepted:
        saved_receipt = SeqaxPallasDiagnosticReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        _validate_manifest(root, saved_receipt.artifacts)
    result = SeqaxPallasDiagnosticResult.model_validate_json((root / "result.json").read_text())
    contract = SeqaxPallasDiagnosticContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if result.contract != contract:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_CONTRACT_MISMATCH")
    trusted_search = default_seqax_pallas_search_contract(contract.runtime)
    if (
        contract.search_id != trusted_search.search_id
        or contract.timing_seed != trusted_search.timing_seed
        or contract.search_receipt_sha256 != _sha256(root / "search" / "receipt.json")
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_BINDING_MISMATCH")
    search_result = _validate_search_snapshot(
        root / "search",
        trusted_search,
        replay_recorded=require_accepted,
    )
    if search_result.winner is not None:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_SEARCH_WINNER_MISMATCH")
    _validate_source(root, result)
    distributed, prepared = prepare_seqax_pallas_candidates(trusted_search)
    incumbent = next(value for value in prepared if value.candidate.name == contract.candidate)
    saved_plan = next(value for value in search_result.plans if value.name == contract.candidate)
    if (
        result.runtime != contract.runtime
        or (result.device_kind, result.device_count) != ("TPU7x", 8)
        or tuple(value.id for value in result.devices) != tuple(range(8))
        or len({value.process_index for value in result.devices}) != 1
        or any(value.platform != "tpu" for value in result.devices)
        or any(value.device_kind not in {"TPU7x", "TPU v7x"} for value in result.devices)
        or tuple(value.model_dump() for value in result.devices)
        != tuple(value.model_dump() for value in search_result.devices)
        or result.distributed_schedule_sha256 != incumbent.plan.distributed_schedule_sha256
        or result.physical_schedule_sha256 != incumbent.plan.physical_schedule_sha256
        or result.pallas_source_sha256 != incumbent.plan.source_sha256()
        or result.stablehlo_sha256 != saved_plan.stablehlo_sha256
        or result.compiler_hlo_sha256 != saved_plan.compiler_hlo_sha256
        or result.trace_profiler_config_sha256 != _sha256(root / "trace" / "profiler_config.json")
        or result.counter_profiler_config_sha256
        != _sha256(root / "counters" / "profiler_config.json")
        or json.loads((root / "trace" / "profiler_config.json").read_text())
        != expected_seqax_profiler_contract(RunMode.TRACE)
        or json.loads((root / "counters" / "profiler_config.json").read_text())
        != expected_seqax_profiler_contract(RunMode.COUNTERS)
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_EXECUTION_IDENTITY_MISMATCH")
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **trusted_search.parameters,
        )
    )
    saved_inputs = tuple(
        np.load(root / "inputs" / f"{index:02d}.npy", allow_pickle=False)
        for index in range(len(host_inputs))
    )
    if any(
        saved.shape != expected.shape
        or saved.dtype != expected.dtype
        or not np.array_equal(saved, expected)
        for saved, expected in zip(saved_inputs, host_inputs, strict=True)
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_INPUT_REPLAY_MISMATCH")
    expected = np.load(
        root
        / "search"
        / "correctness"
        / str(contract.timing_seed)
        / "outputs"
        / f"{contract.candidate}.npy",
        allow_pickle=False,
    )
    actual = np.load(root / "output.npy", allow_pickle=False)
    if (
        result.input_sha256 != arrays_sha256(saved_inputs)
        or result.output_sha256 != array_sha256(actual)
        or result.expected_output_sha256 != array_sha256(expected)
        or not result.exact_incumbent_parity
        or actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not np.array_equal(actual, expected)
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_OUTPUT_REPLAY_MISMATCH")
    cost_report = estimate_seqax_forward(
        distributed,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=incumbent.plan.distributed_schedule_sha256,
            artifact_path="search/distributed.xdsl",
            tool="tpu-cake",
            field="canonical distributed tensor program",
        ),
        expected_schedule_sha256=incumbent.plan.distributed_schedule_sha256,
    )
    saved_cost = SeqaxCostModelReport.model_validate_json((root / "cost_model.json").read_text())
    if saved_cost != cost_report:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_COST_REPLAY_MISMATCH")
    profile_replays: dict[str, tuple[Path, Any, str, int, tuple[float, ...]]] = {}
    with tempfile.TemporaryDirectory(prefix="tpu-cake-xprof-replay-") as directory:
        replay_parent = Path(directory)
        for phase, counters in (("trace", False), ("counters", True)):
            phase_root = root / phase
            xplane, _trace_files = _profile_files(phase_root / "profile")
            _validate_xprof(phase_root, xplane)
            replay_root = replay_parent / phase
            replay_profile = replay_root / "profile"
            replay_xplane = replay_profile / xplane.relative_to(phase_root / "profile")
            replay_xplane.parent.mkdir(parents=True)
            shutil.copy2(xplane, replay_xplane)
            _export_xprof(replay_profile, replay_root / "xprof")
            _validate_xprof(replay_root, replay_xplane)
            _validate_xprof_replay(phase_root, replay_root)
            expectation = _expected_profile(counters=counters)
            saved_assessment = assess_capture(phase_root, expectation)
            assessment = assess_capture(replay_root, expectation)
            if not assessment.accepted:
                raise ValueError(f"SEQAX_PALLAS_DIAGNOSTIC_PROFILE_REPLAY_REJECTED phase={phase}")
            if not saved_assessment.accepted or _canonical_assessment(
                saved_assessment.model_dump(mode="json")
            ) != _canonical_assessment(assessment.model_dump(mode="json")):
                raise ValueError(
                    f"SEQAX_PALLAS_DIAGNOSTIC_XPROF_SEMANTIC_REPLAY_MISMATCH phase={phase}"
                )
            if counters:
                _validate_counter_evidence(saved_assessment)
                _validate_counter_evidence(assessment)
            expected_assessment = _canonical_assessment(assessment.model_dump(mode="json"))
            if (
                json.loads((phase_root / "profile_assessment.json").read_text())
                != expected_assessment
            ):
                raise ValueError(
                    f"SEQAX_PALLAS_DIAGNOSTIC_ASSESSMENT_REPLAY_MISMATCH phase={phase}"
                )
            program_id, program_name = _bound_program(assessment)
            steps, durations = _profile_replay(xplane, program_name)
            profile_replays[phase] = (xplane, assessment, program_id, steps, durations)
        trace_xplane, _trace_assessment, trace_program_id, trace_steps, trace_durations = (
            profile_replays["trace"]
        )
        counter_xplane, counter_assessment, counter_program_id, counter_steps, _ = profile_replays[
            "counters"
        ]
        if trace_program_id != counter_program_id:
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_PROGRAM_ID_MISMATCH")
        attribution = _attribution(
            physical=incumbent.physical,
            program_id=trace_program_id,
            durations=trace_durations,
            hlo_stats=replay_parent / "trace" / "xprof" / "hlo_stats.json",
            cost_report=cost_report,
        )
    saved_attribution = SeqaxPallasDiagnosticAttribution.model_validate_json(
        (root / "attribution.json").read_text()
    )
    if saved_attribution != attribution:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_ATTRIBUTION_REPLAY_MISMATCH")
    if (
        result.trace_step_count != trace_steps
        or result.counter_step_count != counter_steps
        or result.attribution_sha256 != _sha256(root / "attribution.json")
        or result.cost_model_sha256 != _sha256(root / "cost_model.json")
        or result.trace_assessment_sha256 != _sha256(root / "trace" / "profile_assessment.json")
        or result.counter_assessment_sha256
        != _sha256(root / "counters" / "profile_assessment.json")
        or result.periodic_counter_names
        != counter_assessment.capture.counters.periodic_counter_names
        or result.periodic_counter_samples_per_core
        != counter_assessment.capture.counters.periodic_samples_per_tpu_core
        or result.hbm_read_counter_names != counter_assessment.capture.counters.hbm_read_names
        or result.hbm_write_counter_names != counter_assessment.capture.counters.hbm_write_names
        or result.cycle_counter_names != counter_assessment.capture.counters.cycle_names
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_RESULT_REPLAY_MISMATCH")
    run_id = semantic_sha256(
        SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
        contract.search_id,
        contract.search_receipt_sha256,
        _sha256(root / "source_state.json"),
        _sha256(root / "source_manifest.json"),
        _sha256(root / "trace" / "profiler_config.json"),
        _sha256(root / "counters" / "profiler_config.json"),
    )
    if result.run_id != run_id:
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_RUN_ID_MISMATCH")
    payloads = (
        (RunState.CREATED, {"contract": contract.model_dump(mode="json")}),
        (
            RunState.VERIFIED,
            {
                "distributed_schedule_sha256": incumbent.plan.distributed_schedule_sha256,
                "physical_schedule_sha256": incumbent.plan.physical_schedule_sha256,
            },
        ),
        (RunState.LOWERED, {"pallas_source_sha256": incumbent.plan.source_sha256()}),
        (
            RunState.COMPILED,
            {
                "stablehlo_sha256": saved_plan.stablehlo_sha256,
                "compiler_hlo_sha256": saved_plan.compiler_hlo_sha256,
            },
        ),
        (
            RunState.CORRECT,
            {
                "input_sha256": arrays_sha256(host_inputs),
                "output_sha256": array_sha256(actual),
                "expected_output_sha256": array_sha256(expected),
            },
        ),
        (
            RunState.COUNTERED,
            {
                "trace_xplane_sha256": _sha256(trace_xplane),
                "counter_xplane_sha256": _sha256(counter_xplane),
                "trace_step_count": trace_steps,
                "counter_step_count": counter_steps,
                "attribution_sha256": result.attribution_sha256,
            },
        ),
    )
    expected_payloads = (
        payloads + ((RunState.ACCEPTED, {"result_sha256": _sha256(root / "result.json")}),)
        if require_accepted
        else payloads
    )
    history = read_ledger_history(root / "ledger.sqlite", run_id)
    if tuple(event.state for event in history) != tuple(value[0] for value in expected_payloads):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_LEDGER_STATE_MISMATCH")
    if tuple(event.payload_sha256 for event in history) != tuple(
        ExperimentLedger.payload_sha256(value[1]) for value in expected_payloads
    ):
        raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_LEDGER_PAYLOAD_MISMATCH")
    if require_accepted:
        assert saved_receipt is not None
        expected_receipt = SeqaxPallasDiagnosticReceipt(
            diagnostic_schema=SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
            status="passed",
            search_id=contract.search_id,
            run_id=run_id,
            result_sha256=_sha256(root / "result.json"),
            ledger_sha256=_sha256(root / "ledger.sqlite"),
            artifacts=_artifact_manifest(root),
        )
        if saved_receipt != expected_receipt:
            raise ValueError("SEQAX_PALLAS_DIAGNOSTIC_RECEIPT_REPLAY_MISMATCH")
        return saved_receipt
    _validate_manifest(root, _artifact_manifest(root))
    return SeqaxPallasDiagnosticReceipt(
        diagnostic_schema=SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
        status="passed",
        search_id=contract.search_id,
        run_id=run_id,
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(root / "ledger.sqlite"),
        artifacts=_artifact_manifest(root),
    )


def validate_seqax_pallas_incumbent_diagnostic(
    root: Path,
) -> SeqaxPallasDiagnosticReceipt:
    return _validate_diagnostic(root.resolve(), require_accepted=True)
