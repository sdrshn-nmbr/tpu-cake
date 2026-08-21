from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.artifacts import file_sha256, write_json

COMPILER_EXECUTABLE_ANALYSIS_SCHEMA = "compiler-executable-analysis-v1"

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


def _text_artifact_sha256(value: str) -> str:
    return hashlib.sha256((value + "\n").encode()).hexdigest()


class CompilerCostMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    value: float = Field(ge=0)

    @model_validator(mode="after")
    def value_is_finite(self) -> CompilerCostMetric:
        if not math.isfinite(self.value):
            raise ValueError("compiler cost metric must be finite")
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
    buffer_assignment_size_bytes: int = Field(gt=0)
    buffer_assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CompilerExecutableAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_schema: str = COMPILER_EXECUTABLE_ANALYSIS_SCHEMA
    stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_metrics: tuple[CompilerCostMetric, ...] = Field(min_length=1)
    memory: CompilerMemoryAnalysis

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


def _capture_cost_metrics(value: object) -> tuple[CompilerCostMetric, ...]:
    if value is None:
        raise ValueError("COMPILER_ANALYSIS_COST_UNAVAILABLE")
    if not isinstance(value, dict):
        raise TypeError(
            f"COMPILER_ANALYSIS_COST_TYPE_INVALID type={type(value).__name__}"
        )
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
            or metric_value < 0
        ):
            raise ValueError(
                "COMPILER_ANALYSIS_COST_VALUE_INVALID "
                f"name={name!r} value={metric_value!r}"
            )
        metrics.append(CompilerCostMetric(name=name, value=float(metric_value)))
    return tuple(sorted(metrics, key=lambda metric: metric.name))


def _capture_memory(value: object) -> CompilerMemoryAnalysis:
    if value is None:
        raise ValueError("COMPILER_ANALYSIS_MEMORY_UNAVAILABLE")
    fields: dict[str, int] = {}
    for name in _MEMORY_FIELDS:
        observed = getattr(value, name, None)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(
                "COMPILER_ANALYSIS_MEMORY_FIELD_INVALID "
                f"name={name} value={observed!r}"
            )
        fields[name] = observed
    buffer_assignment = getattr(value, "serialized_buffer_assignment_proto", None)
    if not isinstance(buffer_assignment, bytes) or not buffer_assignment:
        raise ValueError("COMPILER_ANALYSIS_BUFFER_ASSIGNMENT_INVALID")
    return CompilerMemoryAnalysis(
        **fields,
        buffer_assignment_size_bytes=len(buffer_assignment),
        buffer_assignment_sha256=hashlib.sha256(buffer_assignment).hexdigest(),
    )


def capture_compiler_analysis(
    executable: Any,
    *,
    stablehlo: str,
    compiler_hlo: str,
) -> CompilerExecutableAnalysis:
    return CompilerExecutableAnalysis(
        stablehlo_sha256=_text_artifact_sha256(stablehlo),
        compiler_hlo_sha256=_text_artifact_sha256(compiler_hlo),
        cost_metrics=_capture_cost_metrics(executable.cost_analysis()),
        memory=_capture_memory(executable.memory_analysis()),
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
    if (
        analysis.stablehlo_sha256 != file_sha256(stablehlo_path)
        or analysis.compiler_hlo_sha256 != file_sha256(compiler_hlo_path)
    ):
        raise ValueError("COMPILER_ANALYSIS_PROGRAM_MISMATCH")
    return analysis
