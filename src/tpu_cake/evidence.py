from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.contracts import ProfileExpectation


class FindingSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ArtifactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PlaneEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    device_type: str | None = None
    line_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    tensor_core_event_count: int = Field(ge=0)


class CounterEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hbm_read_names: int = Field(ge=0)
    hbm_write_names: int = Field(ge=0)
    cycle_names: int = Field(ge=0)
    snapshots_per_tpu_core: dict[str, int]

    @computed_field
    @property
    def rates_derivable(self) -> bool:
        return bool(self.snapshots_per_tpu_core) and all(
            snapshots >= 2 for snapshots in self.snapshots_per_tpu_core.values()
        )


class ProgramEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    program_id: str
    name: str
    timed_self_us: float = Field(ge=0)
    hlo: ArtifactEvidence
    marker_counts: dict[str, int]
    forbidden_fragment_hits: dict[str, int]


class CaptureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    xplane: ArtifactEvidence
    planes: tuple[PlaneEvidence, ...]
    counters: CounterEvidence
    programs: tuple[ProgramEvidence, ...]
    timed_program_ids: frozenset[str]

    @model_validator(mode="after")
    def program_ids_are_unique(self) -> CaptureEvidence:
        program_ids = [program.program_id for program in self.programs]
        if len(program_ids) != len(set(program_ids)):
            raise ValueError("program IDs must be unique")
        return self


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: FindingSeverity
    message: str
    evidence: tuple[str, ...] = ()


class CaptureAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expectation: ProfileExpectation
    capture: CaptureEvidence
    findings: tuple[Finding, ...]

    @computed_field
    @property
    def accepted(self) -> bool:
        return not any(finding.severity is FindingSeverity.ERROR for finding in self.findings)
