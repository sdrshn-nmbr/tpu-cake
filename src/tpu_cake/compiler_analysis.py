from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.artifacts import file_sha256, write_json

COMPILER_EXECUTABLE_ANALYSIS_SCHEMA = "compiler-executable-analysis-v2"
COMPILER_COLLECTIVE_STRATEGY_SURFACE_SCHEMA = "compiler-collective-strategy-surface-v1"

_MEMORY_FIELDS = (
    "generated_code_size_in_bytes",
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "host_generated_code_size_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
    "peak_memory_in_bytes",
)


def compiler_hlo_text(lowered: Any) -> str:
    computation = lowered.compiler_ir(dialect="hlo")
    return computation.as_hlo_text() if hasattr(computation, "as_hlo_text") else str(computation)


def _text_artifact_sha256(value: str) -> str:
    return hashlib.sha256((value + "\n").encode()).hexdigest()


class CompilerCostMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    raw_value: float
    value: float | None = Field(default=None, ge=0)
    available: bool

    @model_validator(mode="after")
    def availability_is_consistent(self) -> CompilerCostMetric:
        if not math.isfinite(self.raw_value):
            raise ValueError("compiler cost metric must be finite")
        if self.available:
            if self.value is None or self.raw_value != self.value:
                raise ValueError("available compiler metric must preserve its raw value")
        elif self.raw_value >= 0 or self.value is not None:
            raise ValueError("unsupported compiler metric unavailable state")
        return self


class CompilerMemoryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_code_size_in_bytes: int = Field(ge=0)
    argument_size_in_bytes: int = Field(ge=0)
    output_size_in_bytes: int = Field(ge=0)
    alias_size_in_bytes: int = Field(ge=0)
    temp_size_in_bytes: int = Field(ge=0)
    host_generated_code_size_in_bytes: int = Field(ge=0)
    host_argument_size_in_bytes: int = Field(ge=0)
    host_output_size_in_bytes: int = Field(ge=0)
    host_alias_size_in_bytes: int = Field(ge=0)
    host_temp_size_in_bytes: int = Field(ge=0)
    peak_memory_in_bytes: int = Field(ge=0)
    buffer_assignment_available: bool
    buffer_assignment_size_bytes: int = Field(ge=0)
    buffer_assignment_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def buffer_assignment_availability_is_consistent(self) -> CompilerMemoryAnalysis:
        if self.buffer_assignment_available:
            if self.buffer_assignment_size_bytes == 0 or self.buffer_assignment_sha256 is None:
                raise ValueError("available buffer assignment must have bytes and a digest")
        elif self.buffer_assignment_size_bytes != 0 or self.buffer_assignment_sha256 is not None:
            raise ValueError("unavailable buffer assignment cannot claim bytes or a digest")
        return self


class CompilerCollectiveAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stablehlo_reduce_scatter_count: int = Field(ge=0)
    stablehlo_all_gather_count: int = Field(ge=0)
    compiler_reduce_scatter_count: int = Field(ge=0)
    compiler_all_reduce_count: int = Field(ge=0)
    compiler_all_gather_count: int = Field(ge=0)
    sparse_core_reduce_scatter_count: int = Field(ge=0)
    sparse_core_all_gather_count: int = Field(ge=0)

    @model_validator(mode="after")
    def offload_counts_do_not_exceed_emitted_counts(self) -> CompilerCollectiveAnalysis:
        if self.sparse_core_reduce_scatter_count > self.compiler_reduce_scatter_count:
            raise ValueError("sparse-core reduce-scatter count exceeds emitted count")
        if self.sparse_core_all_gather_count > self.compiler_all_gather_count:
            raise ValueError("sparse-core all-gather count exceeds emitted count")
        return self


class CompilerExecutableAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_schema: str = COMPILER_EXECUTABLE_ANALYSIS_SCHEMA
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_metrics: tuple[CompilerCostMetric, ...] = Field(min_length=1)
    memory: CompilerMemoryAnalysis
    collectives: CompilerCollectiveAnalysis

    @model_validator(mode="after")
    def contents_are_canonical(self) -> CompilerExecutableAnalysis:
        if self.analysis_schema != COMPILER_EXECUTABLE_ANALYSIS_SCHEMA:
            raise ValueError("compiler analysis schema mismatch")
        names = tuple(metric.name for metric in self.cost_metrics)
        if names != tuple(sorted(names)):
            raise ValueError("compiler cost metrics must be sorted")
        if len(names) != len(set(names)):
            raise ValueError("compiler cost metric names must be unique")
        return self


class CompilerCollectiveStrategyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    payload_bytes_per_device: int = Field(gt=0)
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    collectives: CompilerCollectiveAnalysis

    @model_validator(mode="after")
    def payload_matches_shape(self) -> CompilerCollectiveStrategyPoint:
        if self.payload_bytes_per_device != self.rows * self.columns * 2:
            raise ValueError("BF16 compiler strategy payload does not match shape")
        return self


class CompilerCollectiveStrategySurface(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    surface_schema: str = COMPILER_COLLECTIVE_STRATEGY_SURFACE_SCHEMA
    mesh_axes: tuple[tuple[str, int], ...]
    dtype: str
    points: tuple[CompilerCollectiveStrategyPoint, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def surface_is_canonical(self) -> CompilerCollectiveStrategySurface:
        if self.surface_schema != COMPILER_COLLECTIVE_STRATEGY_SURFACE_SCHEMA:
            raise ValueError("compiler collective strategy surface schema mismatch")
        if self.mesh_axes != (("d", 2), ("t", 4)) or self.dtype != "bfloat16":
            raise ValueError("compiler collective strategy surface geometry mismatch")
        columns = tuple(point.columns for point in self.points)
        if columns != tuple(sorted(set(columns))):
            raise ValueError(
                "compiler collective strategy surface points must be sorted and unique"
            )
        return self


def _compiler_operation_lines(value: str, operation: str) -> tuple[str, ...]:
    pattern = re.compile(rf"=\s*[^\n=]*\b{re.escape(operation)}\(")
    return tuple(line for line in value.splitlines() if pattern.search(line))


def _stablehlo_operation_count(value: str, operation: str) -> int:
    pattern = re.compile(rf'=\s*"?stablehlo\.{re.escape(operation)}"?(?=[\s(<])')
    return sum(bool(pattern.search(line)) for line in value.splitlines())


def analyze_compiler_collectives(
    *,
    stablehlo: str,
    compiler_hlo: str,
) -> CompilerCollectiveAnalysis:
    reduce_scatter_lines = _compiler_operation_lines(compiler_hlo, "reduce-scatter")
    all_gather_lines = _compiler_operation_lines(compiler_hlo, "all-gather")
    return CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=_stablehlo_operation_count(
            stablehlo,
            "reduce_scatter",
        ),
        stablehlo_all_gather_count=_stablehlo_operation_count(stablehlo, "all_gather"),
        compiler_reduce_scatter_count=len(reduce_scatter_lines),
        compiler_all_reduce_count=len(_compiler_operation_lines(compiler_hlo, "all-reduce")),
        compiler_all_gather_count=len(all_gather_lines),
        sparse_core_reduce_scatter_count=sum(
            "reduce_scatter_offload_config" in line
            and '"device_type":"DEVICE_TYPE_SPARSECORE"' in line
            for line in reduce_scatter_lines
        ),
        sparse_core_all_gather_count=sum(
            "all_gather_offload_config" in line and '"device_type":"DEVICE_TYPE_SPARSECORE"' in line
            for line in all_gather_lines
        ),
    )


def _capture_cost_metrics(value: object) -> tuple[CompilerCostMetric, ...]:
    if value is None:
        raise ValueError("COMPILER_ANALYSIS_COST_UNAVAILABLE")
    if not isinstance(value, dict):
        raise TypeError(f"COMPILER_ANALYSIS_COST_TYPE_INVALID type={type(value).__name__}")
    if not value:
        raise ValueError("COMPILER_ANALYSIS_COST_EMPTY")
    metrics = []
    for name, metric_value in value.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(float(metric_value))
        ):
            raise ValueError(
                f"COMPILER_ANALYSIS_COST_VALUE_INVALID name={name!r} value={metric_value!r}"
            )
        raw_value = float(metric_value)
        if raw_value < 0:
            metrics.append(
                CompilerCostMetric(
                    name=name,
                    raw_value=raw_value,
                    available=False,
                )
            )
        else:
            metrics.append(
                CompilerCostMetric(
                    name=name,
                    raw_value=raw_value,
                    value=raw_value,
                    available=True,
                )
            )
    return tuple(sorted(metrics, key=lambda metric: metric.name))


def _capture_memory(value: object) -> CompilerMemoryAnalysis:
    if value is None:
        raise ValueError("COMPILER_ANALYSIS_MEMORY_UNAVAILABLE")
    fields: dict[str, int] = {}
    for name in _MEMORY_FIELDS:
        observed = getattr(value, name, None)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(
                f"COMPILER_ANALYSIS_MEMORY_FIELD_INVALID name={name} value={observed!r}"
            )
        fields[name] = observed
    buffer_assignment = getattr(value, "serialized_buffer_assignment_proto", None)
    if not isinstance(buffer_assignment, bytes):
        raise TypeError("COMPILER_ANALYSIS_BUFFER_ASSIGNMENT_INVALID")
    buffer_assignment_available = bool(buffer_assignment)
    return CompilerMemoryAnalysis(
        **fields,
        buffer_assignment_available=buffer_assignment_available,
        buffer_assignment_size_bytes=len(buffer_assignment),
        buffer_assignment_sha256=(
            hashlib.sha256(buffer_assignment).hexdigest() if buffer_assignment_available else None
        ),
    )


def capture_compiler_analysis(
    executable: Any,
    *,
    stablehlo: str,
    compiler_hlo: str,
) -> CompilerExecutableAnalysis:
    executable_hlo = executable.as_text()
    if not isinstance(executable_hlo, str) or not executable_hlo:
        raise TypeError("COMPILER_ANALYSIS_EXECUTABLE_HLO_INVALID")
    if executable_hlo.rstrip("\n") != compiler_hlo.rstrip("\n"):
        raise ValueError("COMPILER_ANALYSIS_EXECUTABLE_HLO_MISMATCH")
    return CompilerExecutableAnalysis(
        stablehlo_sha256=_text_artifact_sha256(stablehlo),
        compiler_hlo_sha256=_text_artifact_sha256(compiler_hlo),
        cost_metrics=_capture_cost_metrics(executable.cost_analysis()),
        memory=_capture_memory(executable.memory_analysis()),
        collectives=analyze_compiler_collectives(
            stablehlo=stablehlo,
            compiler_hlo=compiler_hlo,
        ),
    )


def write_compiler_analysis(
    path: Path,
    analysis: CompilerExecutableAnalysis,
) -> None:
    write_json(path, analysis.model_dump(mode="json"))


def validate_compiler_analysis(
    path: Path,
    *,
    stablehlo_path: Path,
    compiler_hlo_path: Path,
) -> CompilerExecutableAnalysis:
    analysis = CompilerExecutableAnalysis.model_validate_json(path.read_text())
    if analysis.stablehlo_sha256 != file_sha256(
        stablehlo_path
    ) or analysis.compiler_hlo_sha256 != file_sha256(compiler_hlo_path):
        raise ValueError("COMPILER_ANALYSIS_PROGRAM_MISMATCH")
    observed_collectives = analyze_compiler_collectives(
        stablehlo=stablehlo_path.read_text(),
        compiler_hlo=compiler_hlo_path.read_text(),
    )
    if analysis.collectives != observed_collectives:
        raise ValueError("COMPILER_ANALYSIS_COLLECTIVE_MISMATCH")
    return analysis
