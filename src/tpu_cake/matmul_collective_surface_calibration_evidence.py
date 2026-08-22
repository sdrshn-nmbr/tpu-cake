from __future__ import annotations

from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import model_identity_sha256
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    MatmulCollectiveSurfaceCalibrationProtocol,
)
from tpu_cake.matmul_collective_surface_correctness_evidence import (
    SurfaceCompileContinuityEvidence,
    SurfaceCorrectnessInputCase,
    SurfaceCorrectnessSavedArray,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
)
from tpu_cake.runner import MatmulCollectiveStrategy

MATMUL_COLLECTIVE_SURFACE_CALIBRATION_EVIDENCE_SCHEMA = (
    "matmul-collective-surface-calibration-evidence-v1"
)


class SurfaceCalibrationTimingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    parent_case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_xla_array_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_pallas_array_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input: SurfaceCorrectnessInputCase
    oracle: SurfaceCorrectnessSavedArray

    @model_validator(mode="after")
    def input_and_oracle_are_bound(self) -> SurfaceCalibrationTimingInput:
        if (
            self.input.scenario_name != self.scenario_name
            or self.input.pattern != "signed-periodic"
            or self.oracle.path != f"oracles/{self.scenario_name}.npy"
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_INPUT_INVALID")
        return self


class SurfaceCalibrationResidentPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scenario_name: str
    xla_compile_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pallas_compile_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)

    @computed_field
    @property
    def resident_pair_sha256(self) -> str:
        return model_identity_sha256(self)


class SurfaceCalibrationOutputGate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    scenario_name: str
    strategy: MatmulCollectiveStrategy
    phase: Literal["before_timing", "after_timing"]
    resident_pair_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    start_ns: int = Field(ge=0)
    stop_ns: int = Field(gt=0)
    oracle_array_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output: SurfaceCorrectnessSavedArray
    mismatched_element_count: Literal[0] = 0
    maximum_absolute_error: float = Field(ge=0)
    maximum_normalized_error: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def output_gate_passed(self) -> SurfaceCalibrationOutputGate:
        expected_path = f"outputs/{self.scenario_name}/{self.strategy.value}-{self.phase}.npy"
        if self.stop_ns <= self.start_ns or self.output.path != expected_path:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_OUTPUT_GATE_INVALID")
        return self


class SurfaceCalibrationWarmupExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(gt=0)
    scenario_name: str
    scenario_position: int = Field(ge=1, le=16)
    strategy: MatmulCollectiveStrategy
    strategy_repetition: int = Field(ge=1, le=10)
    resident_pair_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    start_ns: int = Field(ge=0)
    stop_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def completed(self) -> SurfaceCalibrationWarmupExecution:
        if self.stop_ns <= self.start_ns:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_WARMUP_DURATION_INVALID")
        return self


class SurfaceCalibrationCallSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(gt=0)
    round_index: int = Field(ge=0, le=15)
    scenario_name: str
    scenario_position: int = Field(ge=1, le=16)
    strategy: MatmulCollectiveStrategy
    arm_position: int = Field(ge=1, le=2)
    call_index: int = Field(ge=0, le=4)
    resident_pair_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    start_ns: int = Field(ge=0)
    stop_ns: int = Field(gt=0)
    duration_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def duration_matches_clock(self) -> SurfaceCalibrationCallSample:
        if self.stop_ns - self.start_ns != self.duration_ns:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SAMPLE_DURATION_INVALID")
        return self


class MatmulCollectiveSurfaceCalibrationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["matmul-collective-surface-calibration-evidence-v1"] = (
        MATMUL_COLLECTIVE_SURFACE_CALIBRATION_EVIDENCE_SCHEMA
    )
    protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent_attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_parent_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_pid: int = Field(gt=0)
    continuity: tuple[SurfaceCompileContinuityEvidence, ...] = Field(
        min_length=32,
        max_length=32,
    )
    inputs: tuple[SurfaceCalibrationTimingInput, ...] = Field(
        min_length=16,
        max_length=16,
    )
    resident_pairs: tuple[SurfaceCalibrationResidentPair, ...] = Field(
        min_length=16,
        max_length=16,
    )
    output_gates: tuple[SurfaceCalibrationOutputGate, ...] = Field(
        min_length=64,
        max_length=64,
    )
    warmups: tuple[SurfaceCalibrationWarmupExecution, ...] = Field(
        min_length=320,
        max_length=320,
    )
    samples: tuple[SurfaceCalibrationCallSample, ...] = Field(
        min_length=2560,
        max_length=2560,
    )

    @computed_field
    @property
    def evidence_sha256(self) -> str:
        return model_identity_sha256(self)


def validate_surface_calibration_evidence(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
    *,
    expected_protocol_file_sha256: str,
    expected_design_file_sha256: str,
    expected_execution_authority_sha256: str,
    expected_invocation_nonce: str,
    expected_worker_pid: int,
) -> None:
    evidence = MatmulCollectiveSurfaceCalibrationEvidence.model_validate(
        evidence.model_dump(mode="python", exclude_computed_fields=True)
    )
    if (
        evidence.protocol_id != protocol.protocol_id
        or evidence.protocol_file_sha256 != expected_protocol_file_sha256
        or evidence.design_id != design.design_id
        or evidence.design_id != protocol.design_id
        or evidence.design_file_sha256 != expected_design_file_sha256
        or evidence.correctness_parent_attempt_id != protocol.correctness_parent.attempt_id
        or evidence.correctness_parent_evidence_sha256
        != protocol.correctness_parent.evidence_sha256
        or evidence.correctness_parent_receipt_sha256 != protocol.correctness_parent.receipt_sha256
        or evidence.calibration_execution_authority_sha256 != expected_execution_authority_sha256
        or evidence.invocation_nonce != expected_invocation_nonce
        or evidence.worker_pid != expected_worker_pid
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_EVIDENCE_AUTHORITY_MISMATCH")
    _validate_continuity(evidence, protocol)
    _validate_inputs(evidence, protocol, design)
    pairs = _validate_resident_pairs(evidence, protocol)
    _validate_output_gates(evidence, protocol, pairs)
    _validate_warmups(evidence, protocol, pairs)
    _validate_samples(evidence, protocol, pairs)
    _validate_global_clock_order(evidence)


def _validate_continuity(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> None:
    expected = tuple(
        (scenario, strategy) for scenario in protocol.scenarios for strategy in protocol.strategies
    )
    if tuple((value.scenario_name, value.strategy) for value in evidence.continuity) != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CONTINUITY_INVENTORY_MISMATCH")


def _validate_inputs(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> None:
    scenarios = {value.name: value for value in design.calibration_scenarios}
    if tuple(value.scenario_name for value in evidence.inputs) != protocol.scenarios:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_INPUT_INVENTORY_MISMATCH")
    for value in evidence.inputs:
        scenario = scenarios[value.scenario_name]
        if (
            value.input.protocol_id != protocol.correctness_parent.protocol_id
            or value.oracle.shape != (scenario.m, scenario.n)
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_INPUT_AUTHORITY_MISMATCH")


def _validate_resident_pairs(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> dict[str, SurfaceCalibrationResidentPair]:
    if tuple(value.scenario_name for value in evidence.resident_pairs) != protocol.scenarios:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_RESIDENCY_INVENTORY_MISMATCH")
    pairs = {value.scenario_name: value for value in evidence.resident_pairs}
    continuity = {
        (value.scenario_name, value.strategy): value.compile_record_sha256
        for value in evidence.continuity
    }
    first, second = protocol.strategies
    for value in evidence.resident_pairs:
        if (
            value.xla_compile_record_sha256 != continuity[(value.scenario_name, first)]
            or value.pallas_compile_record_sha256 != continuity[(value.scenario_name, second)]
            or value.invocation_nonce != evidence.invocation_nonce
            or value.worker_pid != evidence.worker_pid
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_RESIDENCY_BINDING_MISMATCH")
    return pairs


def _validate_output_gates(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    pairs: dict[str, SurfaceCalibrationResidentPair],
) -> None:
    expected = tuple(
        (scenario, strategy, phase)
        for phase in ("before_timing", "after_timing")
        for scenario in protocol.scenarios
        for strategy in protocol.strategies
    )
    observed = tuple(
        (value.scenario_name, value.strategy, value.phase) for value in evidence.output_gates
    )
    if observed != expected:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_OUTPUT_GATE_INVENTORY_MISMATCH")
    inputs = {value.scenario_name: value for value in evidence.inputs}
    by_arm: dict[tuple[str, MatmulCollectiveStrategy], list[SurfaceCalibrationOutputGate]] = {}
    for gate in evidence.output_gates:
        pair = pairs[gate.scenario_name]
        if (
            gate.resident_pair_sha256 != pair.resident_pair_sha256
            or gate.invocation_nonce != evidence.invocation_nonce
            or gate.worker_pid != evidence.worker_pid
            or gate.oracle_array_sha256 != inputs[gate.scenario_name].oracle.array_sha256
            or gate.output.shape != inputs[gate.scenario_name].oracle.shape
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_OUTPUT_GATE_BINDING_MISMATCH")
        expected_parent = (
            inputs[gate.scenario_name].parent_xla_array_sha256
            if gate.strategy is protocol.strategies[0]
            else inputs[gate.scenario_name].parent_pallas_array_sha256
        )
        if gate.output.array_sha256 != expected_parent:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_OUTPUT_PARENT_MISMATCH")
        by_arm.setdefault((gate.scenario_name, gate.strategy), []).append(gate)
    if any(
        len(values) != 2
        or values[0].output.array_sha256 != values[1].output.array_sha256
        or values[0].output.array_sha256 != values[0].oracle_array_sha256
        for values in by_arm.values()
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_OUTPUT_GATE_REPEAT_MISMATCH")


def _validate_warmups(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    pairs: dict[str, SurfaceCalibrationResidentPair],
) -> None:
    expected = []
    repetitions: dict[tuple[str, MatmulCollectiveStrategy], int] = {}
    for scenario_index, scenario in enumerate(protocol.scenarios):
        for strategy in protocol.warmup_strategy_order(scenario_index):
            key = (scenario, strategy)
            repetitions[key] = repetitions.get(key, 0) + 1
            expected.append(
                (
                    scenario,
                    scenario_index + 1,
                    strategy,
                    repetitions[key],
                    pairs[scenario].resident_pair_sha256,
                )
            )
    observed = tuple(
        (
            value.scenario_name,
            value.scenario_position,
            value.strategy,
            value.strategy_repetition,
            value.resident_pair_sha256,
        )
        for value in evidence.warmups
    )
    if (
        observed != tuple(expected)
        or tuple(value.sequence for value in evidence.warmups) != tuple(range(1, 321))
        or any(
            value.invocation_nonce != evidence.invocation_nonce
            or value.worker_pid != evidence.worker_pid
            for value in evidence.warmups
        )
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_WARMUP_SEQUENCE_MISMATCH")


def _validate_samples(
    evidence: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    pairs: dict[str, SurfaceCalibrationResidentPair],
) -> None:
    expected = []
    for round_index in range(protocol.paired_rounds):
        for scenario_position, scenario in enumerate(protocol.scenario_order(round_index), start=1):
            for arm_position, strategy in enumerate(
                protocol.strategy_order(round_index),
                start=1,
            ):
                for call_index in range(protocol.calls_per_position):
                    expected.append(
                        (
                            round_index,
                            scenario,
                            scenario_position,
                            strategy,
                            arm_position,
                            call_index,
                            pairs[scenario].resident_pair_sha256,
                        )
                    )
    observed = tuple(
        (
            value.round_index,
            value.scenario_name,
            value.scenario_position,
            value.strategy,
            value.arm_position,
            value.call_index,
            value.resident_pair_sha256,
        )
        for value in evidence.samples
    )
    if (
        observed != tuple(expected)
        or tuple(value.sequence for value in evidence.samples) != tuple(range(1, 2561))
        or any(
            value.invocation_nonce != evidence.invocation_nonce
            or value.worker_pid != evidence.worker_pid
            for value in evidence.samples
        )
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_SAMPLE_SEQUENCE_MISMATCH")


def _validate_global_clock_order(evidence: MatmulCollectiveSurfaceCalibrationEvidence) -> None:
    before = tuple(value for value in evidence.output_gates if value.phase == "before_timing")
    after = tuple(value for value in evidence.output_gates if value.phase == "after_timing")
    timeline = (*before, *evidence.warmups, *evidence.samples, *after)
    if any(left.stop_ns > right.start_ns for left, right in pairwise(timeline)):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_CLOCK_ORDER_MISMATCH")
