from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import array_sha256, model_identity_sha256
from tpu_cake.inkling_gmm_route_corpus import InklingGmmRouteCorpusReport
from tpu_cake.inkling_gmm_tile_search import (
    GmmArmName,
    GmmPolicyPair,
    InklingGmmTileSearchContract,
    local_active_span,
)

GMM_CORRECTNESS_GATE_SCHEMA = "inkling-gmm-tile-search-correctness-v1"
GMM_CONFIRMATION_CORRECTNESS_SCHEMA = "inkling-gmm-confirmation-correctness-v1"
_OPERAND_NAMES = ("inputs", "gate", "up", "down")


class GmmCorrectnessGateError(RuntimeError):
    pass


def _fail(code: str, **context: object) -> None:
    fields = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
    raise GmmCorrectnessGateError(f"INKLING_GMM_CORRECTNESS_{code} {fields}".rstrip())


@dataclass(frozen=True)
class GmmCorrectnessOperands:
    inputs: Any
    gate_weights: Any
    up_weights: Any
    down_weights: Any

    def as_mapping(self) -> Mapping[str, Any]:
        return {
            "inputs": self.inputs,
            "gate": self.gate_weights,
            "up": self.up_weights,
            "down": self.down_weights,
        }


@dataclass(frozen=True)
class GmmStageOutputs:
    gate: Any
    up: Any
    down: Any


@dataclass(frozen=True)
class GmmCpuOracleValues:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


class OperandSentinelMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    shape: tuple[int, ...]
    dtype: str
    indices: tuple[tuple[int, ...], ...]
    descriptor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sentinel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StageOutputMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nonfinite_count: int = Field(ge=0)
    outside_nonzero_count: int = Field(ge=0)


class OutputMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate: StageOutputMeasurement
    up: StageOutputMeasurement
    down: StageOutputMeasurement


class CpuOracleMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    stage: str
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)


class PolicyCorrectnessMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: GmmPolicyPair
    outputs: OutputMeasurement


class ProfileCorrectnessMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_index: int = Field(ge=0, lt=5)
    seed: int = Field(ge=0)
    completion_step: int
    layer_index: int
    group_sizes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operands: tuple[OperandSentinelMeasurement, ...]
    policies: tuple[PolicyCorrectnessMeasurement, ...]
    cpu_oracle: tuple[CpuOracleMeasurement, ...]

    @model_validator(mode="after")
    def measurement_inventory_is_exact(self) -> ProfileCorrectnessMeasurement:
        if tuple(item.name for item in self.operands) != _OPERAND_NAMES:
            raise ValueError("GMM correctness operand measurement inventory mismatch")
        policies = tuple(item.policy for item in self.policies)
        if not policies or len(policies) != len(set(policies)):
            raise ValueError("GMM correctness policy measurements must be unique")
        return self


class GmmCorrectnessGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GMM_CORRECTNESS_GATE_SCHEMA
    report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    profiles: tuple[ProfileCorrectnessMeasurement, ...]

    @computed_field
    @property
    def profile_count(self) -> int:
        return len(self.profiles)

    @model_validator(mode="after")
    def report_inventory_is_exact(self) -> GmmCorrectnessGateReport:
        if tuple(profile.profile_index for profile in self.profiles) != tuple(range(5)):
            raise ValueError("GMM correctness profile report inventory mismatch")
        if sum(len(profile.cpu_oracle) for profile in self.profiles) != 24:
            raise ValueError("GMM correctness CPU oracle measurement inventory mismatch")
        if any(
            tuple(item.policy for item in profile.policies) != unique_correctness_policies()
            for profile in self.profiles
        ):
            raise ValueError("GMM correctness policy measurement inventory mismatch")
        return self


class GmmConfirmationCorrectnessReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GMM_CONFIRMATION_CORRECTNESS_SCHEMA
    report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: GmmPolicyPair
    profiles: tuple[ProfileCorrectnessMeasurement, ...]

    @model_validator(mode="after")
    def report_inventory_is_exact(self) -> GmmConfirmationCorrectnessReport:
        baseline = GmmPolicyPair(
            gate_up=GmmArmName.INCUMBENT,
            down=GmmArmName.INCUMBENT,
        )
        expected = (baseline, self.candidate)
        if self.candidate == baseline:
            raise ValueError("GMM confirmation correctness candidate must differ from baseline")
        if tuple(profile.profile_index for profile in self.profiles) != tuple(range(5)):
            raise ValueError("GMM confirmation correctness profile inventory mismatch")
        if sum(len(profile.cpu_oracle) for profile in self.profiles) != 24:
            raise ValueError("GMM confirmation correctness CPU oracle inventory mismatch")
        if any(
            tuple(item.policy for item in profile.policies) != expected
            for profile in self.profiles
        ):
            raise ValueError("GMM confirmation correctness policy inventory mismatch")
        return self


OperandFactory = Callable[[int, int, int], GmmCorrectnessOperands]
CorrectnessExecutor = Callable[
    [GmmPolicyPair, GmmCorrectnessOperands, tuple[int, ...]],
    GmmStageOutputs,
]


def unique_correctness_policies() -> tuple[GmmPolicyPair, ...]:
    incumbent = GmmArmName.INCUMBENT
    alternatives = tuple(arm for arm in GmmArmName if arm is not incumbent)
    return (
        GmmPolicyPair(gate_up=incumbent, down=incumbent),
        *(GmmPolicyPair(gate_up=arm, down=incumbent) for arm in alternatives),
        *(GmmPolicyPair(gate_up=incumbent, down=arm) for arm in alternatives),
    )


def fixed_order_fp32_row_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_f32 = np.asarray(lhs, dtype=np.float32)
    rhs_f32 = np.asarray(rhs, dtype=np.float32)
    if lhs_f32.ndim != 1 or rhs_f32.ndim != 2 or lhs_f32.shape[0] != rhs_f32.shape[0]:
        raise ValueError("fixed-order row matmul ABI mismatch")
    result = np.zeros(rhs_f32.shape[1], dtype=np.float32)
    for index in range(lhs_f32.shape[0]):
        product = np.multiply(lhs_f32[index], rhs_f32[index], dtype=np.float32)
        result = np.add(result, product, dtype=np.float32)
    return result


def selected_row_cpu_oracle(
    inputs: np.ndarray,
    gate_weights: np.ndarray,
    up_weights: np.ndarray,
    down_weights: np.ndarray,
    *,
    row_index: int,
    local_expert_index: int,
    down_columns: tuple[int, ...],
) -> GmmCpuOracleValues:
    inputs_array = np.asarray(inputs)
    gate_array = np.asarray(gate_weights)
    up_array = np.asarray(up_weights)
    down_array = np.asarray(down_weights)
    gate = fixed_order_fp32_row_matmul(inputs_array[row_index], gate_array[local_expert_index])
    up = fixed_order_fp32_row_matmul(inputs_array[row_index], up_array[local_expert_index])
    negative_gate = np.negative(gate, dtype=np.float32)
    denominator = np.add(np.float32(1.0), np.exp(negative_gate, dtype=np.float32), dtype=np.float32)
    silu = np.divide(gate, denominator, dtype=np.float32)
    hidden = np.multiply(silu, up, dtype=np.float32)
    selected_down = down_array[local_expert_index][:, down_columns]
    down = fixed_order_fp32_row_matmul(hidden, selected_down)
    return GmmCpuOracleValues(gate=gate, up=up, down=down)


def _sentinel_indices(
    seed: int,
    name: str,
    shape: tuple[int, ...],
    count: int,
) -> tuple[tuple[int, ...], ...]:
    if not shape or any(dimension <= 0 for dimension in shape) or count <= 0:
        raise ValueError("operand sentinel shape/count mismatch")
    indices: list[tuple[int, ...]] = []
    ordinal = 0
    while len(indices) < min(count, int(np.prod(shape))):
        digest = hashlib.sha256(f"{seed}:{name}:{ordinal}".encode()).digest()
        candidate = tuple(
            int.from_bytes(digest[axis * 4 : axis * 4 + 4], "big") % dimension
            for axis, dimension in enumerate(shape)
        )
        if candidate not in indices:
            indices.append(candidate)
        ordinal += 1
    return tuple(indices)


def measure_operand_sentinels(
    arrays: Mapping[str, Any],
    *,
    seed: int,
    sentinel_count: int = 16,
) -> tuple[OperandSentinelMeasurement, ...]:
    if tuple(arrays) != _OPERAND_NAMES:
        raise ValueError("operand inventory mismatch")
    measurements = []
    for name, value in arrays.items():
        shape = tuple(int(dimension) for dimension in value.shape)
        dtype = str(value.dtype)
        indices = _sentinel_indices(seed, name, shape, sentinel_count)
        coordinates = tuple(
            np.asarray([index[axis] for index in indices], dtype=np.int32)
            for axis in range(len(shape))
        )
        sentinels = np.asarray(value[coordinates])
        descriptor = {
            "schema": "jax-stateless-uniform-v1",
            "seed": seed,
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "minimum": -0.02,
            "maximum": 0.02,
            "indices": indices,
        }
        measurements.append(
            OperandSentinelMeasurement(
                name=name,
                shape=shape,
                dtype=dtype,
                indices=indices,
                descriptor_sha256=hashlib.sha256(
                    json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                sentinel_sha256=array_sha256(sentinels),
            )
        )
    return tuple(measurements)


def _active_array(array: np.ndarray, spans: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.concatenate(
        tuple(array[device_index, start:end] for device_index, (start, end) in enumerate(spans)),
        axis=0,
    )


def _outside_nonzero(array: np.ndarray, spans: tuple[tuple[int, int], ...]) -> int:
    count = 0
    for device_index, (start, end) in enumerate(spans):
        count += int(np.count_nonzero(array[device_index, :start]))
        count += int(np.count_nonzero(array[device_index, end:]))
    return count


def measure_outputs(
    outputs: GmmStageOutputs,
    spans: tuple[tuple[int, int], ...],
) -> OutputMeasurement:
    arrays = {
        "gate": np.asarray(outputs.gate),
        "up": np.asarray(outputs.up),
        "down": np.asarray(outputs.down),
    }
    if any(value.dtype != np.float32 or value.ndim != 3 for value in arrays.values()):
        _fail("OUTPUT_ABI")
    if any(value.shape[0] != len(spans) for value in arrays.values()):
        _fail("OUTPUT_DEVICE_COUNT")
    measurements: dict[str, StageOutputMeasurement] = {}
    for name, value in arrays.items():
        active = _active_array(value, spans)
        nonfinite_count = int(np.count_nonzero(~np.isfinite(active)))
        if name == "down":
            nonfinite_count += sum(
                int(np.count_nonzero(~np.isfinite(value[device_index, :start])))
                + int(np.count_nonzero(~np.isfinite(value[device_index, end:])))
                for device_index, (start, end) in enumerate(spans)
            )
        if nonfinite_count:
            _fail("NONFINITE_ACTIVE_OUTPUT", stage=name, count=nonfinite_count)
        outside_nonzero_count = _outside_nonzero(value, spans) if name == "down" else 0
        if outside_nonzero_count:
            _fail("DOWN_OUTSIDE_LOCAL_SPAN", count=outside_nonzero_count)
        measurements[name] = StageOutputMeasurement(
            active_sha256=array_sha256(active),
            nonfinite_count=nonfinite_count,
            outside_nonzero_count=outside_nonzero_count,
        )
    return OutputMeasurement(**measurements)


def compare_active_spans(
    baseline: GmmStageOutputs,
    candidate: GmmStageOutputs,
    spans: tuple[tuple[int, int], ...],
    *,
    policy_name: str,
) -> None:
    for stage in ("gate", "up", "down"):
        baseline_active = _active_array(np.asarray(getattr(baseline, stage)), spans)
        candidate_active = _active_array(np.asarray(getattr(candidate, stage)), spans)
        if array_sha256(baseline_active) != array_sha256(candidate_active):
            _fail("ACTIVE_SPAN_MISMATCH", stage=stage, policy=policy_name)


def validate_cpu_values(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    stage: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> CpuOracleMeasurement:
    expected_f32 = np.asarray(expected, dtype=np.float32)
    actual_f32 = np.asarray(actual, dtype=np.float32)
    if expected_f32.shape != actual_f32.shape or not np.all(np.isfinite(actual_f32)):
        _fail("CPU_ORACLE_ABI", stage=stage)
    absolute = np.abs(actual_f32 - expected_f32)
    bound = absolute_tolerance + relative_tolerance * np.abs(expected_f32)
    if np.any(absolute > bound):
        _fail(
            "CPU_ORACLE_MISMATCH",
            stage=stage,
            maximum_absolute_error=float(np.max(absolute, initial=0.0)),
        )
    denominator = np.maximum(np.abs(expected_f32), np.finfo(np.float32).tiny)
    relative = absolute / denominator
    return CpuOracleMeasurement(
        stage=stage,
        expected_sha256=array_sha256(expected_f32),
        actual_sha256=array_sha256(actual_f32),
        maximum_absolute_error=float(np.max(absolute, initial=0.0)),
        maximum_relative_error=float(np.max(relative, initial=0.0)),
    )


def _group_sizes_hash(group_sizes: tuple[int, ...]) -> str:
    encoded = json.dumps(group_sizes, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _local_expert_for_row(
    group_sizes: tuple[int, ...],
    *,
    device_index: int,
    row_index: int,
) -> int:
    start, end = local_active_span(group_sizes, device_index=device_index)
    if not start <= row_index < end:
        _fail("CPU_ORACLE_ROW_OUTSIDE_SHARD", device=device_index, row=row_index)
    cursor = 0
    for expert_index, size in enumerate(group_sizes):
        if cursor <= row_index < cursor + size:
            local_expert = expert_index - device_index * 32
            if not 0 <= local_expert < 32:
                _fail("CPU_ORACLE_EXPERT_SHARD", device=device_index, expert=expert_index)
            return local_expert
        cursor += size
    _fail("CPU_ORACLE_ROW_UNROUTED", row=row_index)


def _validate_production_operand_abi(operands: GmmCorrectnessOperands) -> None:
    expected_shapes = {
        "inputs": (8, 288, 4096),
        "gate": (8, 32, 4096, 2048),
        "up": (8, 32, 4096, 2048),
        "down": (8, 32, 2048, 4096),
    }
    for name, value in operands.as_mapping().items():
        if tuple(value.shape) != expected_shapes[name] or str(value.dtype) != "bfloat16":
            _fail("OPERAND_ABI", operand=name, shape=value.shape, dtype=value.dtype)


def _validate_production_output_abi(outputs: GmmStageOutputs) -> None:
    expected_shapes = {
        "gate": (8, 288, 2048),
        "up": (8, 288, 2048),
        "down": (8, 288, 4096),
    }
    for name, value in (("gate", outputs.gate), ("up", outputs.up), ("down", outputs.down)):
        if tuple(value.shape) != expected_shapes[name] or str(value.dtype) != "float32":
            _fail("OUTPUT_ABI", stage=name, shape=value.shape, dtype=value.dtype)


def _cpu_oracle_measurements(
    contract: InklingGmmTileSearchContract,
    operands: GmmCorrectnessOperands,
    baseline: GmmStageOutputs,
    group_sizes: tuple[int, ...],
    profile_index: int,
) -> tuple[CpuOracleMeasurement, ...]:
    measurements = []
    for row in contract.correctness.cpu_oracle_rows:
        if row.profile_index != profile_index:
            continue
        local_expert = _local_expert_for_row(
            group_sizes,
            device_index=row.device_index,
            row_index=row.row_index,
        )
        device = row.device_index
        oracle = selected_row_cpu_oracle(
            np.asarray(operands.inputs[device]),
            np.asarray(operands.gate_weights[device, local_expert : local_expert + 1]),
            np.asarray(operands.up_weights[device, local_expert : local_expert + 1]),
            np.asarray(operands.down_weights[device, local_expert : local_expert + 1]),
            row_index=row.row_index,
            local_expert_index=0,
            down_columns=row.down_columns,
        )
        actuals = (
            ("gate", oracle.gate, np.asarray(baseline.gate[device, row.row_index])),
            ("up", oracle.up, np.asarray(baseline.up[device, row.row_index])),
            (
                "down",
                oracle.down,
                np.asarray(baseline.down[device, row.row_index, list(row.down_columns)]),
            ),
        )
        measurements.extend(
            validate_cpu_values(
                expected,
                actual,
                stage=f"device-{device}/{stage}",
                absolute_tolerance=contract.correctness.absolute_tolerance,
                relative_tolerance=contract.correctness.relative_tolerance,
            )
            for stage, expected, actual in actuals
        )
    return tuple(measurements)


def run_correctness_gate(
    contract: InklingGmmTileSearchContract,
    report: InklingGmmRouteCorpusReport,
    *,
    operand_factory: OperandFactory,
    execute: CorrectnessExecutor,
) -> GmmCorrectnessGateReport:
    policies = unique_correctness_policies()
    baseline_policy = policies[0]
    profiles = []
    for profile_index, (profile, seed) in enumerate(
        zip(contract.correctness.profiles, contract.correctness.seeds, strict=True)
    ):
        group = report.group_sizes[profile.corpus_index]
        group_sizes = group.group_sizes
        if (
            group.completion_step != profile.completion_step
            or group.layer_index != profile.layer_index
            or _group_sizes_hash(group_sizes) != profile.group_sizes_sha256
        ):
            _fail("PROFILE_BINDING", profile=profile_index)
        operands = operand_factory(profile_index, seed, profile.layer_index - 2)
        _validate_production_operand_abi(operands)
        operand_measurements = measure_operand_sentinels(operands.as_mapping(), seed=seed)
        spans = tuple(
            local_active_span(group_sizes, device_index=device_index)
            for device_index in range(contract.production_abi.device_count)
        )
        baseline = execute(baseline_policy, operands, group_sizes)
        _validate_production_output_abi(baseline)
        policy_measurements = [
            PolicyCorrectnessMeasurement(
                policy=baseline_policy,
                outputs=measure_outputs(baseline, spans),
            )
        ]
        cpu_measurements = _cpu_oracle_measurements(
            contract,
            operands,
            baseline,
            group_sizes,
            profile_index,
        )
        for policy in policies[1:]:
            candidate = execute(policy, operands, group_sizes)
            _validate_production_output_abi(candidate)
            candidate_measurement = measure_outputs(candidate, spans)
            compare_active_spans(baseline, candidate, spans, policy_name=policy.name)
            policy_measurements.append(
                PolicyCorrectnessMeasurement(policy=policy, outputs=candidate_measurement)
            )
        profiles.append(
            ProfileCorrectnessMeasurement(
                profile_index=profile_index,
                seed=seed,
                completion_step=profile.completion_step,
                layer_index=profile.layer_index,
                group_sizes_sha256=profile.group_sizes_sha256,
                operands=operand_measurements,
                policies=tuple(policy_measurements),
                cpu_oracle=cpu_measurements,
            )
        )
    provisional = GmmCorrectnessGateReport(
        report_id="0" * 64,
        search_id=contract.search_id,
        route_report_id=report.report_id,
        numerical_contract_id=contract.correctness.numerical_contract_id,
        profiles=tuple(profiles),
    )
    return provisional.model_copy(
        update={"report_id": model_identity_sha256(provisional, exclude={"report_id"})}
    )


def run_confirmation_correctness_gate(
    contract: InklingGmmTileSearchContract,
    report: InklingGmmRouteCorpusReport,
    candidate: GmmPolicyPair,
    *,
    operand_factory: OperandFactory,
    execute: CorrectnessExecutor,
) -> GmmConfirmationCorrectnessReport:
    baseline_policy = GmmPolicyPair(
        gate_up=GmmArmName.INCUMBENT,
        down=GmmArmName.INCUMBENT,
    )
    if candidate == baseline_policy:
        _fail("CONFIRMATION_CANDIDATE_IS_BASELINE")
    profiles = []
    for profile_index, (profile, seed) in enumerate(
        zip(contract.correctness.profiles, contract.correctness.seeds, strict=True)
    ):
        group = report.group_sizes[profile.corpus_index]
        group_sizes = group.group_sizes
        if (
            group.completion_step != profile.completion_step
            or group.layer_index != profile.layer_index
            or _group_sizes_hash(group_sizes) != profile.group_sizes_sha256
        ):
            _fail("PROFILE_BINDING", profile=profile_index)
        operands = operand_factory(profile_index, seed, profile.layer_index - 2)
        _validate_production_operand_abi(operands)
        operand_measurements = measure_operand_sentinels(operands.as_mapping(), seed=seed)
        spans = tuple(
            local_active_span(group_sizes, device_index=device_index)
            for device_index in range(contract.production_abi.device_count)
        )
        baseline = execute(baseline_policy, operands, group_sizes)
        candidate_outputs = execute(candidate, operands, group_sizes)
        _validate_production_output_abi(baseline)
        _validate_production_output_abi(candidate_outputs)
        compare_active_spans(baseline, candidate_outputs, spans, policy_name=candidate.name)
        profiles.append(
            ProfileCorrectnessMeasurement(
                profile_index=profile_index,
                seed=seed,
                completion_step=profile.completion_step,
                layer_index=profile.layer_index,
                group_sizes_sha256=profile.group_sizes_sha256,
                operands=operand_measurements,
                policies=(
                    PolicyCorrectnessMeasurement(
                        policy=baseline_policy,
                        outputs=measure_outputs(baseline, spans),
                    ),
                    PolicyCorrectnessMeasurement(
                        policy=candidate,
                        outputs=measure_outputs(candidate_outputs, spans),
                    ),
                ),
                cpu_oracle=_cpu_oracle_measurements(
                    contract,
                    operands,
                    baseline,
                    group_sizes,
                    profile_index,
                ),
            )
        )
    provisional = GmmConfirmationCorrectnessReport(
        report_id="0" * 64,
        search_id=contract.search_id,
        route_report_id=report.report_id,
        numerical_contract_id=contract.correctness.numerical_contract_id,
        candidate=candidate,
        profiles=tuple(profiles),
    )
    return provisional.model_copy(
        update={"report_id": model_identity_sha256(provisional, exclude={"report_id"})}
    )


def replay_correctness_gate(
    expected: GmmCorrectnessGateReport,
    contract: InklingGmmTileSearchContract,
    report: InklingGmmRouteCorpusReport,
    *,
    operand_factory: OperandFactory,
    execute: CorrectnessExecutor,
) -> GmmCorrectnessGateReport:
    observed = run_correctness_gate(
        contract,
        report,
        operand_factory=operand_factory,
        execute=execute,
    )
    if observed != expected:
        _fail(
            "REPLAY_MISMATCH",
            expected_report_id=expected.report_id,
            observed_report_id=observed.report_id,
        )
    return observed


def write_correctness_report(path: Path, report: GmmCorrectnessGateReport) -> None:
    with path.open("x") as stream:
        stream.write(
            json.dumps(
                report.model_dump(mode="json", exclude_computed_fields=True),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def write_confirmation_correctness_report(
    path: Path,
    report: GmmConfirmationCorrectnessReport,
) -> None:
    with path.open("x") as stream:
        stream.write(
            json.dumps(
                report.model_dump(mode="json", exclude_computed_fields=True),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
