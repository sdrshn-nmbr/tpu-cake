from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class WorkloadStage(StrEnum):
    CONTROL = "control"
    PREFILL = "prefill"
    STEADY_DECODE = "steady_decode"
    MIXED = "mixed"


class ProfileExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    stage: WorkloadStage
    minimum_tpu_device_planes: int = Field(default=1, ge=1)
    require_tensor_core_activity: bool = True
    require_hbm_read_counters: bool = False
    require_hbm_write_counters: bool = False
    require_cycle_counters: bool = False
    required_timed_hlo_markers: tuple[str, ...] = ()
    forbidden_timed_hlo_fragments: tuple[str, ...] = ()
