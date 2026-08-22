from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import model_identity_sha256
from tpu_cake.matmul_collective_surface_correctness_protocol import (
    MatmulCollectiveSurfaceCorrectnessProtocol,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceSplit,
)
from tpu_cake.runner import MatmulCollectiveStrategy

CORRECTNESS_EVIDENCE_SCHEMA = "matmul-collective-surface-correctness-evidence-v1"


class SurfaceCorrectnessSlice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    start: int = Field(ge=0)
    stop: int = Field(gt=0)
    step: Literal[1] = 1

    @model_validator(mode="after")
    def bounds_are_nonempty(self) -> SurfaceCorrectnessSlice:
        if self.stop <= self.start:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SLICE_EMPTY")
        return self


class SurfaceCorrectnessSentinel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ordinal: int = Field(ge=0, le=31)
    global_coordinate: tuple[int, int]
    local_coordinate: tuple[int, int]
    expected_bfloat16_hex: str = Field(pattern=r"^[0-9a-f]{4}$")
    observed_bfloat16_hex: str = Field(pattern=r"^[0-9a-f]{4}$")

    @model_validator(mode="after")
    def observed_matches_expected(self) -> SurfaceCorrectnessSentinel:
        if self.observed_bfloat16_hex != self.expected_bfloat16_hex:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SENTINEL_MISMATCH")
        return self


class SurfaceCorrectnessShardIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["lhs", "rhs"]
    shard_index: int = Field(ge=0, le=7)
    device_id: int = Field(ge=0, le=7)
    process_index: Literal[0] = 0
    global_shape: tuple[int, int]
    sharding: Literal["PartitionSpec(None, 't')", "PartitionSpec('t', None)"]
    global_slice: tuple[SurfaceCorrectnessSlice, SurfaceCorrectnessSlice]
    local_shape: tuple[int, int]
    logical_dtype: Literal["bfloat16"] = "bfloat16"
    numpy_dtype_str: Literal["<V2"] = "<V2"
    payload_byte_order: Literal["little"] = "little"
    payload_nbytes: int = Field(gt=0)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sentinels: tuple[SurfaceCorrectnessSentinel, ...] = Field(min_length=32, max_length=32)

    @model_validator(mode="after")
    def shard_is_internally_canonical(self) -> SurfaceCorrectnessShardIdentity:
        expected_sharding = (
            "PartitionSpec(None, 't')" if self.role == "lhs" else "PartitionSpec('t', None)"
        )
        slices = self.global_slice
        expected_local = tuple(value.stop - value.start for value in slices)
        coordinates = tuple(value.global_coordinate for value in self.sentinels)
        local_coordinates = tuple(value.local_coordinate for value in self.sentinels)
        if (
            self.shard_index != self.device_id
            or self.sharding != expected_sharding
            or self.local_shape != expected_local
            or len(set(coordinates)) != len(coordinates)
            or coordinates != tuple(sorted(coordinates))
            or tuple(value.ordinal for value in self.sentinels) != tuple(range(32))
            or self.payload_nbytes != self.local_shape[0] * self.local_shape[1] * 2
            or any(
                not all(
                    bound.start <= coordinate < bound.stop
                    for coordinate, bound in zip(sentinel.global_coordinate, slices, strict=True)
                )
                for sentinel in self.sentinels
            )
            or local_coordinates
            != tuple(
                tuple(
                    coordinate - bound.start
                    for coordinate, bound in zip(
                        sentinel.global_coordinate,
                        slices,
                        strict=True,
                    )
                )
                for sentinel in self.sentinels
            )
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SHARD_IDENTITY_INVALID")
        return self

    @computed_field
    @property
    def identity_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCorrectnessInputCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    pattern: str
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    pattern_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lhs_shards: tuple[SurfaceCorrectnessShardIdentity, ...] = Field(min_length=8, max_length=8)
    rhs_shards: tuple[SurfaceCorrectnessShardIdentity, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def shard_inventory_is_ordered(self) -> SurfaceCorrectnessInputCase:
        for role, values in (("lhs", self.lhs_shards), ("rhs", self.rhs_shards)):
            if tuple(value.device_id for value in values) != tuple(range(8)) or any(
                value.role != role for value in values
            ):
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SHARD_INVENTORY_INVALID")
        return self

    @computed_field
    @property
    def input_identity_sha256(self) -> str:
        return model_identity_sha256(self)

    @computed_field
    @property
    def lhs_identity_set_sha256(self) -> str:
        return model_identity_sha256(_ShardIdentitySet(shards=self.lhs_shards))

    @computed_field
    @property
    def rhs_identity_set_sha256(self) -> str:
        return model_identity_sha256(_ShardIdentitySet(shards=self.rhs_shards))


class _ShardIdentitySet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    shards: tuple[SurfaceCorrectnessShardIdentity, ...]


class SurfaceCorrectnessSavedArray(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    array_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shape: tuple[int, int]
    dtype: Literal["float32"] = "float32"
    numpy_dtype_str: Literal["<f4"] = "<f4"
    nan_count: Literal[0] = 0
    positive_infinity_count: Literal[0] = 0
    negative_infinity_count: Literal[0] = 0

    @model_validator(mode="after")
    def path_is_canonical(self) -> SurfaceCorrectnessSavedArray:
        parts = self.path.split("/")
        if (
            not self.path.endswith(".npy")
            or self.path.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_OUTPUT_PATH_INVALID")
        return self


class SurfaceCorrectnessCandidateExecution(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    sequence: int = Field(gt=0)
    position: int = Field(ge=1, le=4)
    strategy: MatmulCollectiveStrategy
    strategy_repetition: int = Field(ge=1, le=2)
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    fresh_compile_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lhs_identity_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rhs_identity_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_array_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_output_sharding: Literal["PartitionSpec(None, 't')"] = "PartitionSpec(None, 't')"
    output: SurfaceCorrectnessSavedArray
    mismatched_element_count: int = Field(ge=0)
    maximum_absolute_error: float = Field(ge=0)
    maximum_normalized_error: float = Field(ge=0)

    @model_validator(mode="after")
    def execution_passed(self) -> SurfaceCorrectnessCandidateExecution:
        if (
            self.output.nan_count != 0
            or self.output.positive_infinity_count != 0
            or self.output.negative_infinity_count != 0
            or self.mismatched_element_count != 0
            or self.maximum_normalized_error > 1.0
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_EXECUTION_FAILED")
        return self


class SurfaceCorrectnessCaseEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    input: SurfaceCorrectnessInputCase
    oracle: SurfaceCorrectnessSavedArray
    executions: tuple[SurfaceCorrectnessCandidateExecution, ...] = Field(
        min_length=4,
        max_length=4,
    )

    @model_validator(mode="after")
    def repetitions_are_deterministic(self) -> SurfaceCorrectnessCaseEvidence:
        by_strategy: dict[MatmulCollectiveStrategy, list[SurfaceCorrectnessCandidateExecution]] = {}
        for execution in self.executions:
            by_strategy.setdefault(execution.strategy, []).append(execution)
        if len(by_strategy) != 2 or any(
            tuple(value.strategy_repetition for value in values) != (1, 2)
            or len({value.output.array_sha256 for value in values}) != 1
            for values in by_strategy.values()
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_REPEAT_MISMATCH")
        if len({value.output.path for value in self.executions}) != 4:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_OUTPUT_PATH_REUSED")
        return self

    @computed_field
    @property
    def case_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCompileContinuityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    stablehlo_path: str
    stablehlo_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_hlo_path: str
    compiler_hlo_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_physical_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_pallas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_semantic_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_semantic_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_semantic_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_semantic_compiler_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identities_match_parent(self) -> SurfaceCompileContinuityEvidence:
        if any(
            parent != observed
            for parent, observed in (
                (
                    self.parent_distributed_schedule_sha256,
                    self.observed_distributed_schedule_sha256,
                ),
                (self.parent_physical_schedule_sha256, self.observed_physical_schedule_sha256),
                (self.parent_pallas_source_sha256, self.observed_pallas_source_sha256),
                (self.parent_semantic_stablehlo_sha256, self.observed_semantic_stablehlo_sha256),
                (
                    self.parent_semantic_compiler_hlo_sha256,
                    self.observed_semantic_compiler_hlo_sha256,
                ),
            )
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_COMPILE_CONTINUITY_FAILED")
        return self

    @computed_field
    @property
    def compile_record_sha256(self) -> str:
        return model_identity_sha256(self)


class MatmulCollectiveSurfaceCorrectnessEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-correctness-evidence-v1"] = (
        CORRECTNESS_EVIDENCE_SCHEMA
    )
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: MatmulCollectiveSurfaceSplit
    parent_compile_manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    continuity: tuple[SurfaceCompileContinuityEvidence, ...] = Field(
        min_length=8,
        max_length=32,
    )
    cases: tuple[SurfaceCorrectnessCaseEvidence, ...] = Field(min_length=20, max_length=80)

    @computed_field
    @property
    def evidence_sha256(self) -> str:
        return model_identity_sha256(self)


def validate_surface_correctness_evidence(
    evidence: MatmulCollectiveSurfaceCorrectnessEvidence,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> None:
    evidence = MatmulCollectiveSurfaceCorrectnessEvidence.model_validate(
        evidence.model_dump(mode="python", exclude_computed_fields=True)
    )
    protocol = MatmulCollectiveSurfaceCorrectnessProtocol.model_validate(
        protocol.model_dump(mode="python", exclude_computed_fields=True)
    )
    design = MatmulCollectiveSurfaceDesignContract.model_validate(
        design.model_dump(mode="python", exclude_computed_fields=True)
    )
    if (
        evidence.protocol_id != protocol.protocol_id
        or evidence.parent_compile_manifest_file_sha256
        != protocol.parent_compile.manifest_file_sha256
        or design.design_id != protocol.parent_compile.design_id
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_EVIDENCE_AUTHORITY_MISMATCH")
    expected_scenarios = (
        protocol.calibration_scenarios
        if evidence.split is MatmulCollectiveSurfaceSplit.CALIBRATION
        else protocol.holdout_scenarios
    )
    expected_continuity = tuple(
        (scenario, strategy) for scenario in expected_scenarios for strategy in protocol.strategies
    )
    if tuple((value.scenario_name, value.strategy) for value in evidence.continuity) != (
        expected_continuity
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CONTINUITY_INVENTORY_MISMATCH")
    expected_cases = tuple(
        (scenario, pattern)
        for scenario in expected_scenarios
        for pattern in protocol.patterns.ordered_patterns
    )
    if tuple((value.input.scenario_name, value.input.pattern) for value in evidence.cases) != (
        expected_cases
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CASE_INVENTORY_MISMATCH")
    scenarios = {value.name: value for value in design.scenarios}
    for case in evidence.cases:
        scenario = scenarios[case.input.scenario_name]
        if (
            case.input.protocol_id != protocol.protocol_id
            or case.input.pattern_contract_sha256 != protocol.patterns.contract_sha256
            or case.oracle.shape != (scenario.m, scenario.n)
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CASE_AUTHORITY_MISMATCH")
        _validate_case_shards(case.input, scenario.m, scenario.k, scenario.n, protocol)
        pattern_index = protocol.patterns.ordered_patterns.index(case.input.pattern)
        first, second = protocol.strategies
        expected_order = (first, second, second, first)
        if pattern_index % 2:
            expected_order = (second, first, first, second)
        if tuple(value.strategy for value in case.executions) != expected_order or tuple(
            value.position for value in case.executions
        ) != (1, 2, 3, 4):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_EXECUTION_ORDER_MISMATCH")
        if any(value.output.shape != (scenario.m, scenario.n) for value in case.executions):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_OUTPUT_ABI_MISMATCH")
        continuity = {
            value.strategy: value
            for value in evidence.continuity
            if value.scenario_name == scenario.name
        }
        if any(
            value.fresh_compile_record_sha256 != continuity[value.strategy].compile_record_sha256
            or value.lhs_identity_set_sha256 != case.input.lhs_identity_set_sha256
            or value.rhs_identity_set_sha256 != case.input.rhs_identity_set_sha256
            or value.oracle_array_sha256 != case.oracle.array_sha256
            for value in case.executions
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_EXECUTION_BINDING_MISMATCH")
    executions = tuple(value for case in evidence.cases for value in case.executions)
    if tuple(value.sequence for value in executions) != tuple(range(1, len(executions) + 1)):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_EXECUTION_SEQUENCE_MISMATCH")
    if len({value.output.path for value in executions}) != len(executions):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_OUTPUT_PATH_REUSED")


def _validate_case_shards(
    case: SurfaceCorrectnessInputCase,
    m: int,
    k: int,
    n: int,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
) -> None:
    local_k = k // 8
    for role, values, global_shape, sharding in (
        ("lhs", case.lhs_shards, (m, k), protocol.lhs_sharding),
        ("rhs", case.rhs_shards, (k, n), protocol.rhs_sharding),
    ):
        for device_id, shard in enumerate(values):
            expected_slice = (
                (
                    SurfaceCorrectnessSlice(start=0, stop=m),
                    SurfaceCorrectnessSlice(
                        start=device_id * local_k,
                        stop=(device_id + 1) * local_k,
                    ),
                )
                if role == "lhs"
                else (
                    SurfaceCorrectnessSlice(
                        start=device_id * local_k,
                        stop=(device_id + 1) * local_k,
                    ),
                    SurfaceCorrectnessSlice(start=0, stop=n),
                )
            )
            if (
                shard.device_id != device_id
                or shard.global_shape != global_shape
                or shard.sharding != sharding
                or shard.global_slice != expected_slice
            ):
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SHARD_PLACEMENT_MISMATCH")
