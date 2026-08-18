from __future__ import annotations

import hashlib
import json
from collections import deque
from enum import StrEnum

import ml_dtypes
import numpy as np
from jax._src.interpreters import mlir
from jaxlib.mlir import ir
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import semantic_seed
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)

BF16_UNIT_ROUNDOFF = 2.0**-8
SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA = "bf16-forward-numerical-v1"
_REGION_TERMINATORS = frozenset({"sdy.return", "stablehlo.return"})
_CALIBRATION_PARAMETERS = {
    "batch": 2,
    "data_mesh": 2,
    "feed_forward": 16,
    "head": 4,
    "key_value_heads": 4,
    "layers": 1,
    "model": 256,
    "query_groups": 2,
    "rope_max_timescale": 256,
    "sequence": 1,
    "tensor_mesh": 4,
    "vocabulary": 16,
}
_CALIBRATION_SEEDS = tuple(
    semantic_seed("seqax-pallas-tiled-einsum-v1", str(index)) for index in range(5)
)
_HELD_OUT_PARAMETERS = {
    "m128-b2-s3-l2": {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 24,
        "head": 4,
        "key_value_heads": 4,
        "layers": 2,
        "model": 128,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 3,
        "tensor_mesh": 4,
        "vocabulary": 32,
    },
    "m256-b4-s2-l1": {
        "batch": 4,
        "data_mesh": 2,
        "feed_forward": 32,
        "head": 4,
        "key_value_heads": 4,
        "layers": 1,
        "model": 256,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 2,
        "tensor_mesh": 4,
        "vocabulary": 32,
    },
    "m384-b2-s2-l1": {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 24,
        "head": 4,
        "key_value_heads": 4,
        "layers": 1,
        "model": 384,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 2,
        "tensor_mesh": 4,
        "vocabulary": 32,
    },
}


class SeqaxNumericalScenarioRole(StrEnum):
    CALIBRATION = "calibration"
    HELD_OUT = "held_out"


class SeqaxInputMutation(StrEnum):
    DROP_EMBEDDING_SHARD = "drop_embedding_shard"
    ROLL_MODEL_SHARD = "roll_model_shard"
    OMIT_MLP_TERM = "omit_mlp_term"
    SWAP_GATE_UP = "swap_gate_up"


class SeqaxNumericalDiscriminator(StrEnum):
    REMOVE_INPUT_BARRIER = "remove_input_barrier"
    REMOVE_OUTPUT_BARRIER = "remove_output_barrier"
    IDENTITY_SILU = "identity_silu"
    RELU_SILU = "relu_silu"
    DROP_REDUCTION_COLLECTIVE = "drop_reduction_collective"
    DROP_EMBEDDING_SHARD = "drop_embedding_shard"
    ROLL_MODEL_SHARD = "roll_model_shard"
    OMIT_MLP_TERM = "omit_mlp_term"
    SWAP_GATE_UP = "swap_gate_up"
    LOCALIZED_SPIKE = "localized_spike"
    DISTRIBUTED_DRIFT = "distributed_drift"
    NONFINITE_OUTPUT = "nonfinite_output"
    DTYPE_OUTPUT = "dtype_output"
    SHAPE_OUTPUT = "shape_output"


class SeqaxDiscriminatorClause(StrEnum):
    STRICT_HLO_STRUCTURE = "strict_hlo_structure"
    PHYSICAL_SCHEDULE_VERIFICATION = "physical_schedule_verification"
    FORWARD_NUMERICAL_POLICY = "forward_numerical_policy"
    ROW_SCALED_MAXIMUM = "row_scaled_maximum"
    RELATIVE_L2 = "relative_l2"
    FINITE_OUTPUT = "finite_output"
    OUTPUT_DTYPE = "output_dtype"
    OUTPUT_SHAPE = "output_shape"


_DISCRIMINATOR_CLAUSES = {
    SeqaxNumericalDiscriminator.REMOVE_INPUT_BARRIER: (
        SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE
    ),
    SeqaxNumericalDiscriminator.REMOVE_OUTPUT_BARRIER: (
        SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE
    ),
    SeqaxNumericalDiscriminator.IDENTITY_SILU: (SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE),
    SeqaxNumericalDiscriminator.RELU_SILU: (SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE),
    SeqaxNumericalDiscriminator.DROP_REDUCTION_COLLECTIVE: (
        SeqaxDiscriminatorClause.PHYSICAL_SCHEDULE_VERIFICATION
    ),
    SeqaxNumericalDiscriminator.DROP_EMBEDDING_SHARD: (
        SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY
    ),
    SeqaxNumericalDiscriminator.ROLL_MODEL_SHARD: (
        SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY
    ),
    SeqaxNumericalDiscriminator.OMIT_MLP_TERM: (SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY),
    SeqaxNumericalDiscriminator.SWAP_GATE_UP: (SeqaxDiscriminatorClause.FORWARD_NUMERICAL_POLICY),
    SeqaxNumericalDiscriminator.LOCALIZED_SPIKE: (SeqaxDiscriminatorClause.ROW_SCALED_MAXIMUM),
    SeqaxNumericalDiscriminator.DISTRIBUTED_DRIFT: SeqaxDiscriminatorClause.RELATIVE_L2,
    SeqaxNumericalDiscriminator.NONFINITE_OUTPUT: SeqaxDiscriminatorClause.FINITE_OUTPUT,
    SeqaxNumericalDiscriminator.DTYPE_OUTPUT: SeqaxDiscriminatorClause.OUTPUT_DTYPE,
    SeqaxNumericalDiscriminator.SHAPE_OUTPUT: SeqaxDiscriminatorClause.OUTPUT_SHAPE,
}


def seqax_discriminator_clause(
    discriminator: SeqaxNumericalDiscriminator,
) -> SeqaxDiscriminatorClause:
    if not isinstance(discriminator, SeqaxNumericalDiscriminator):
        raise TypeError("Seqax numerical discriminator must be typed")
    return _DISCRIMINATOR_CLAUSES[discriminator]


class SeqaxBf16NumericalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA
    numerical_semantics: str = "typed_bf16_v1"
    unit_roundoff: float = BF16_UNIT_ROUNDOFF
    cpu_relative_l2_units: float = Field(gt=0)
    cpu_row_scaled_max_units: float = Field(gt=0)
    cross_path_relative_l2_units: float = Field(gt=0)
    cross_path_row_scaled_max_units: float = Field(gt=0)
    row_scale_floor: float = Field(gt=0)
    cpu_reference: str = "seqax_forward_canonical_reference"
    cpu_reference_quantization_decimals: int = 6
    checkpoint_storage_dtype: str = "uint16"
    checkpoint_logical_dtype: str = "bfloat16"
    checkpoint_encoding: str = "bf16-bit-pattern-v1"
    require_float32_output: bool = True
    require_finite_output: bool = True
    require_exact_mathematical_silu: bool = True

    @model_validator(mode="after")
    def policy_is_canonical(self) -> SeqaxBf16NumericalPolicy:
        if self.schema_version != SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA:
            raise ValueError("Seqax BF16 numerical policy schema mismatch")
        if self.numerical_semantics != "typed_bf16_v1":
            raise ValueError("Seqax BF16 numerical policy requires typed BF16 semantics")
        if self.unit_roundoff != BF16_UNIT_ROUNDOFF:
            raise ValueError("Seqax BF16 numerical policy unit roundoff mismatch")
        if (
            self.cpu_relative_l2_units,
            self.cpu_row_scaled_max_units,
            self.cross_path_relative_l2_units,
            self.cross_path_row_scaled_max_units,
            self.row_scale_floor,
            self.cpu_reference,
            self.cpu_reference_quantization_decimals,
            self.checkpoint_storage_dtype,
            self.checkpoint_logical_dtype,
            self.checkpoint_encoding,
            self.require_float32_output,
            self.require_finite_output,
            self.require_exact_mathematical_silu,
        ) != (
            2.0,
            4.0,
            0.5,
            1.0,
            1.0,
            "seqax_forward_canonical_reference",
            6,
            "uint16",
            "bfloat16",
            "bf16-bit-pattern-v1",
            True,
            True,
            True,
        ):
            raise ValueError("Seqax BF16 numerical policy is not canonical")
        return self


class SeqaxBf16ScenarioParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    batch: int = Field(gt=0)
    data_mesh: int = Field(gt=0)
    feed_forward: int = Field(gt=0)
    head: int = Field(gt=0)
    key_value_heads: int = Field(gt=0)
    layers: int = Field(gt=0)
    model: int = Field(gt=0)
    query_groups: int = Field(gt=0)
    rope_max_timescale: int = Field(gt=0)
    sequence: int = Field(gt=0)
    tensor_mesh: int = Field(gt=0)
    vocabulary: int = Field(gt=0)


class SeqaxNumericalTensorContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    shape: tuple[int, ...] = Field(min_length=1)
    dtype: str = Field(min_length=1)

    @model_validator(mode="after")
    def dimensions_are_positive(self) -> SeqaxNumericalTensorContract:
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("Seqax numerical tensor dimensions must be positive")
        return self


def _scenario_abi(
    parameters: SeqaxBf16ScenarioParameters,
) -> tuple[
    tuple[SeqaxNumericalTensorContract, ...],
    SeqaxNumericalTensorContract,
    tuple[SeqaxNumericalTensorContract, ...],
    tuple[SeqaxNumericalTensorContract, ...],
]:
    batch = parameters.batch
    sequence = parameters.sequence
    model = parameters.model
    vocabulary = parameters.vocabulary
    feed_forward = parameters.feed_forward
    layers = parameters.layers
    query_groups = parameters.query_groups
    key_value_heads = parameters.key_value_heads
    head = parameters.head

    def tensor(
        name: str, shape: tuple[int, ...], dtype: str = "float32"
    ) -> SeqaxNumericalTensorContract:
        return SeqaxNumericalTensorContract(name=name, shape=shape, dtype=dtype)

    inputs = (
        tensor("tokens", (batch, sequence), "uint32"),
        tensor("sequence_starts", (batch, sequence), "bool"),
        tensor("embedding", (vocabulary, model)),
        tensor("layer_norm_1", (layers, model)),
        tensor("layer_norm_2", (layers, model)),
        tensor(
            "query_weights",
            (layers, model, query_groups, key_value_heads, head),
        ),
        tensor(
            "key_value_weights",
            (layers, 2, model, key_value_heads, head),
        ),
        tensor(
            "output_weights",
            (layers, model, query_groups, key_value_heads, head),
        ),
        tensor("gate_weights", (layers, model, feed_forward)),
        tensor("up_weights", (layers, model, feed_forward)),
        tensor("down_weights", (layers, model, feed_forward)),
        tensor("final_layer_norm", (model,)),
        tensor("unembedding", (vocabulary, model)),
    )
    output = tensor("logits", (batch, sequence, vocabulary))
    gate_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_gate",
            (batch, sequence, feed_forward),
            "bfloat16",
        )
        for layer in range(layers)
    )
    silu_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_silu",
            (batch, sequence, feed_forward),
            "bfloat16",
        )
        for layer in range(layers)
    )
    return inputs, output, gate_checkpoints, silu_checkpoints


class SeqaxBf16NumericalScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    role: SeqaxNumericalScenarioRole
    parameters: SeqaxBf16ScenarioParameters
    seeds: tuple[int, ...] = Field(min_length=4)
    inputs: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=13, max_length=13)
    output: SeqaxNumericalTensorContract
    gate_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    silu_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def parameter_schema_is_complete(self) -> SeqaxBf16NumericalScenario:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seqax BF16 numerical scenario seeds must be unique")
        expected_inputs, expected_output, expected_gates, expected_silu = _scenario_abi(
            self.parameters
        )
        if self.inputs != expected_inputs:
            raise ValueError("Seqax BF16 numerical scenario input ABI mismatch")
        if self.output != expected_output:
            raise ValueError("Seqax BF16 numerical scenario output ABI mismatch")
        if self.gate_checkpoints != expected_gates:
            raise ValueError("Seqax BF16 numerical scenario gate checkpoint ABI mismatch")
        if self.silu_checkpoints != expected_silu:
            raise ValueError("Seqax BF16 numerical scenario SiLU checkpoint ABI mismatch")
        return self


class SeqaxBf16ValidationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA
    policy: SeqaxBf16NumericalPolicy
    scenarios: tuple[SeqaxBf16NumericalScenario, ...] = Field(min_length=4)
    required_discriminators: tuple[SeqaxNumericalDiscriminator, ...]
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    acceptance_authority: str = "authenticated-runner-and-relocated-public-replay"
    require_instrumented_output_parity: bool = True
    require_discriminator_artifact_replay: bool = True

    @model_validator(mode="after")
    def validation_surface_is_canonical(self) -> SeqaxBf16ValidationContract:
        if self.schema_version != SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA:
            raise ValueError("Seqax BF16 validation schema mismatch")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax BF16 validation hardware contract mismatch")
        if (
            self.acceptance_authority,
            self.require_instrumented_output_parity,
            self.require_discriminator_artifact_replay,
        ) != (
            "authenticated-runner-and-relocated-public-replay",
            True,
            True,
        ):
            raise ValueError("Seqax BF16 validation acceptance authority mismatch")
        if self.required_discriminators != tuple(SeqaxNumericalDiscriminator):
            raise ValueError("Seqax BF16 validation discriminators are not canonical")
        if tuple(scenario.name for scenario in self.scenarios) != (
            "calibration-m256-b2-s1-l1",
            *_HELD_OUT_PARAMETERS,
        ):
            raise ValueError("Seqax BF16 validation scenarios are not canonical")
        for scenario in self.scenarios:
            expected_parameters = (
                _CALIBRATION_PARAMETERS
                if scenario.role is SeqaxNumericalScenarioRole.CALIBRATION
                else _HELD_OUT_PARAMETERS.get(scenario.name)
            )
            if expected_parameters is None or scenario.parameters != (
                SeqaxBf16ScenarioParameters(**expected_parameters)
            ):
                raise ValueError(
                    f"Seqax BF16 validation scenario parameters mismatch: {scenario.name}"
                )
            expected_seeds = (
                _CALIBRATION_SEEDS
                if scenario.role is SeqaxNumericalScenarioRole.CALIBRATION
                else tuple(
                    semantic_seed(self.schema_version, f"{scenario.name}:{index}")
                    for index in range(4)
                )
            )
            if scenario.seeds != expected_seeds:
                raise ValueError(f"Seqax BF16 validation scenario seeds mismatch: {scenario.name}")
        all_seeds = tuple(seed for scenario in self.scenarios for seed in scenario.seeds)
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("Seqax BF16 validation seeds must be unique across scenarios")
        return self

    @computed_field
    @property
    def contract_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"contract_id"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SeqaxBf16NumericalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cpu_pallas_relative_l2: float = Field(ge=0)
    cpu_control_relative_l2: float = Field(ge=0)
    cross_path_relative_l2: float = Field(ge=0)
    cpu_pallas_row_scaled_max: float = Field(ge=0)
    cpu_control_row_scaled_max: float = Field(ge=0)
    cross_path_row_scaled_max: float = Field(ge=0)
    pallas_top1_matches_cpu: bool
    control_top1_matches_cpu: bool
    pallas_top1_matches_control: bool
    gate_cross_path_exact: bool
    pallas_silu_matches_mathematical: bool
    control_silu_matches_mathematical: bool
    silu_cross_path_exact: bool
    final_outputs_satisfy_policy: bool
    checkpoint_values_consistent: bool


def default_seqax_bf16_validation_contract() -> SeqaxBf16ValidationContract:
    calibration_parameters = SeqaxBf16ScenarioParameters(**_CALIBRATION_PARAMETERS)
    calibration_inputs, calibration_output, calibration_gates, calibration_silu = _scenario_abi(
        calibration_parameters
    )
    scenarios = [
        SeqaxBf16NumericalScenario(
            name="calibration-m256-b2-s1-l1",
            role=SeqaxNumericalScenarioRole.CALIBRATION,
            parameters=calibration_parameters,
            seeds=_CALIBRATION_SEEDS,
            inputs=calibration_inputs,
            output=calibration_output,
            gate_checkpoints=calibration_gates,
            silu_checkpoints=calibration_silu,
        )
    ]
    for name, raw_parameters in _HELD_OUT_PARAMETERS.items():
        parameters = SeqaxBf16ScenarioParameters(**raw_parameters)
        inputs, output, gates, silu = _scenario_abi(parameters)
        scenarios.append(
            SeqaxBf16NumericalScenario(
                name=name,
                role=SeqaxNumericalScenarioRole.HELD_OUT,
                parameters=parameters,
                seeds=tuple(
                    semantic_seed(
                        SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA,
                        f"{name}:{index}",
                    )
                    for index in range(4)
                ),
                inputs=inputs,
                output=output,
                gate_checkpoints=gates,
                silu_checkpoints=silu,
            )
        )
    return SeqaxBf16ValidationContract(
        policy=SeqaxBf16NumericalPolicy(
            cpu_relative_l2_units=2.0,
            cpu_row_scaled_max_units=4.0,
            cross_path_relative_l2_units=0.5,
            cross_path_row_scaled_max_units=1.0,
            row_scale_floor=1.0,
        ),
        scenarios=tuple(scenarios),
        required_discriminators=tuple(SeqaxNumericalDiscriminator),
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
    )


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    denominator = max(float(np.linalg.norm(expected.astype(np.float64).ravel())), 1e-30)
    return float(np.linalg.norm(difference.ravel()) / denominator)


def _row_scaled_max(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    scale_floor: float,
) -> float:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    row_scale = np.maximum(
        np.max(np.abs(expected.astype(np.float64)), axis=-1),
        scale_floor,
    )
    return float(np.max(np.max(difference, axis=-1) / row_scale))


def _validate_bf16_checkpoints(
    values: tuple[np.ndarray, ...],
    expected: tuple[SeqaxNumericalTensorContract, ...],
    *,
    label: str,
) -> tuple[np.ndarray, ...]:
    if not values or len(values) != len(expected):
        raise ValueError(f"Seqax BF16 {label} checkpoint count does not match the contract")
    arrays = tuple(np.asarray(value) for value in values)
    expected_dtype = np.dtype(ml_dtypes.bfloat16)
    for checkpoint, contract in zip(arrays, expected, strict=True):
        if checkpoint.dtype != expected_dtype:
            raise TypeError(f"Seqax BF16 {label} checkpoints must use bfloat16")
        if checkpoint.shape != contract.shape:
            raise ValueError(f"Seqax BF16 {label} checkpoint shape does not match the contract")
        if not np.all(np.isfinite(checkpoint.astype(np.float32))):
            raise ValueError(f"Seqax BF16 {label} checkpoints must be finite")
    return arrays


def assess_seqax_bf16_forward(
    pallas: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    inputs: tuple[np.ndarray, ...],
    pallas_gate_checkpoints: tuple[np.ndarray, ...],
    control_gate_checkpoints: tuple[np.ndarray, ...],
    pallas_silu_checkpoints: tuple[np.ndarray, ...],
    control_silu_checkpoints: tuple[np.ndarray, ...],
    policy: SeqaxBf16NumericalPolicy,
    scenario: SeqaxBf16NumericalScenario,
) -> SeqaxBf16NumericalAssessment:
    if type(seed) is not int or seed not in scenario.seeds:
        raise ValueError("Seqax BF16 numerical seed is not declared by the scenario")
    validate_seqax_numerical_inputs(inputs, scenario)
    expected_inputs = seqax_forward_inputs(
        seed=seed,
        **scenario.parameters.model_dump(),
    )
    for actual, expected, contract in zip(inputs, expected_inputs, scenario.inputs, strict=True):
        if not np.array_equal(actual, expected):
            raise ValueError(f"Seqax numerical deterministic input mismatch: {contract.name}")
    cpu_reference = seqax_forward_canonical_reference(
        expected_inputs,
        quantization_decimals=policy.cpu_reference_quantization_decimals,
        **scenario.parameters.model_dump(),
    )
    arrays = tuple(np.asarray(value) for value in (pallas, control, cpu_reference))
    if len({value.shape for value in arrays}) != 1 or arrays[0].shape != scenario.output.shape:
        raise ValueError("Seqax BF16 numerical output shape does not match the contract")
    if policy.require_float32_output and any(value.dtype != np.float32 for value in arrays):
        raise TypeError("Seqax BF16 numerical outputs must use float32")
    if policy.require_finite_output and any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Seqax BF16 numerical outputs must be finite")
    pallas_relative = _relative_l2(arrays[0], arrays[2])
    control_relative = _relative_l2(arrays[1], arrays[2])
    cross_relative = _relative_l2(arrays[0], arrays[1])
    pallas_scaled = _row_scaled_max(arrays[0], arrays[2], scale_floor=policy.row_scale_floor)
    control_scaled = _row_scaled_max(arrays[1], arrays[2], scale_floor=policy.row_scale_floor)
    cross_scaled = _row_scaled_max(arrays[0], arrays[1], scale_floor=policy.row_scale_floor)
    pallas_gates = _validate_bf16_checkpoints(
        pallas_gate_checkpoints,
        scenario.gate_checkpoints,
        label="gate",
    )
    control_gates = _validate_bf16_checkpoints(
        control_gate_checkpoints,
        scenario.gate_checkpoints,
        label="gate",
    )
    pallas_silu = _validate_bf16_checkpoints(
        pallas_silu_checkpoints,
        scenario.silu_checkpoints,
        label="SiLU",
    )
    control_silu = _validate_bf16_checkpoints(
        control_silu_checkpoints,
        scenario.silu_checkpoints,
        label="SiLU",
    )
    pallas_mathematical = tuple(rounded_mathematical_silu_bf16(value) for value in pallas_gates)
    control_mathematical = tuple(rounded_mathematical_silu_bf16(value) for value in control_gates)
    gate_cross_path = all(
        np.array_equal(pallas_value, control_value)
        for pallas_value, control_value in zip(pallas_gates, control_gates, strict=True)
    )
    pallas_silu_mathematical = all(
        np.array_equal(actual, expected)
        for actual, expected in zip(pallas_silu, pallas_mathematical, strict=True)
    )
    control_silu_mathematical = all(
        np.array_equal(actual, expected)
        for actual, expected in zip(control_silu, control_mathematical, strict=True)
    )
    silu_cross_path = all(
        np.array_equal(pallas_value, control_value)
        for pallas_value, control_value in zip(pallas_silu, control_silu, strict=True)
    )
    unit = policy.unit_roundoff
    final_outputs_satisfy_policy = (
        pallas_relative <= policy.cpu_relative_l2_units * unit
        and control_relative <= policy.cpu_relative_l2_units * unit
        and cross_relative <= policy.cross_path_relative_l2_units * unit
        and pallas_scaled <= policy.cpu_row_scaled_max_units * unit
        and control_scaled <= policy.cpu_row_scaled_max_units * unit
        and cross_scaled <= policy.cross_path_row_scaled_max_units * unit
    )
    checkpoint_values_consistent = (
        gate_cross_path
        and pallas_silu_mathematical
        and control_silu_mathematical
        and silu_cross_path
    )
    pallas_top1 = np.argmax(arrays[0], axis=-1)
    control_top1 = np.argmax(arrays[1], axis=-1)
    cpu_top1 = np.argmax(arrays[2], axis=-1)
    return SeqaxBf16NumericalAssessment(
        cpu_pallas_relative_l2=pallas_relative,
        cpu_control_relative_l2=control_relative,
        cross_path_relative_l2=cross_relative,
        cpu_pallas_row_scaled_max=pallas_scaled,
        cpu_control_row_scaled_max=control_scaled,
        cross_path_row_scaled_max=cross_scaled,
        pallas_top1_matches_cpu=bool(np.array_equal(pallas_top1, cpu_top1)),
        control_top1_matches_cpu=bool(np.array_equal(control_top1, cpu_top1)),
        pallas_top1_matches_control=bool(np.array_equal(pallas_top1, control_top1)),
        gate_cross_path_exact=gate_cross_path,
        pallas_silu_matches_mathematical=pallas_silu_mathematical,
        control_silu_matches_mathematical=control_silu_mathematical,
        silu_cross_path_exact=silu_cross_path,
        final_outputs_satisfy_policy=final_outputs_satisfy_policy,
        checkpoint_values_consistent=checkpoint_values_consistent,
    )


def validate_seqax_numerical_inputs(
    inputs: tuple[np.ndarray, ...],
    scenario: SeqaxBf16NumericalScenario,
) -> None:
    if len(inputs) != len(scenario.inputs):
        raise ValueError("Seqax numerical input count does not match the contract")
    dtype_by_name = {
        "bool": np.dtype(np.bool_),
        "float32": np.dtype(np.float32),
        "uint32": np.dtype(np.uint32),
    }
    for value, contract in zip(inputs, scenario.inputs, strict=True):
        array = np.asarray(value)
        if array.shape != contract.shape:
            raise ValueError(f"Seqax numerical input shape mismatch: {contract.name}")
        if array.dtype != dtype_by_name[contract.dtype]:
            raise TypeError(f"Seqax numerical input dtype mismatch: {contract.name}")


def encode_seqax_bf16_checkpoint(
    value: np.ndarray,
    contract: SeqaxNumericalTensorContract,
) -> np.ndarray:
    array = np.asarray(value)
    if contract.dtype != "bfloat16":
        raise ValueError("Seqax BF16 checkpoint contract must declare bfloat16")
    if array.dtype != np.dtype(ml_dtypes.bfloat16):
        raise TypeError("Seqax BF16 checkpoint must use logical bfloat16")
    if array.shape != contract.shape:
        raise ValueError("Seqax BF16 checkpoint shape does not match the contract")
    if not np.all(np.isfinite(array.astype(np.float32))):
        raise ValueError("Seqax BF16 checkpoint must be finite")
    return array.view(np.uint16).copy()


def decode_seqax_bf16_checkpoint(
    stored: np.ndarray,
    contract: SeqaxNumericalTensorContract,
) -> np.ndarray:
    array = np.asarray(stored)
    if contract.dtype != "bfloat16":
        raise ValueError("Seqax BF16 checkpoint contract must declare bfloat16")
    if array.dtype != np.dtype(np.uint16):
        raise TypeError("Seqax BF16 checkpoint storage must use uint16")
    if array.shape != contract.shape:
        raise ValueError("Seqax BF16 checkpoint shape does not match the contract")
    result = array.view(np.dtype(ml_dtypes.bfloat16))
    if not np.all(np.isfinite(result.astype(np.float32))):
        raise ValueError("Seqax BF16 checkpoint must be finite")
    return result


def mutate_seqax_forward_inputs(
    inputs: tuple[np.ndarray, ...],
    mutation: SeqaxInputMutation,
) -> tuple[np.ndarray, ...]:
    if not isinstance(mutation, SeqaxInputMutation):
        raise TypeError("Seqax input mutation must be a SeqaxInputMutation")
    values = [np.asarray(value).copy() for value in inputs]
    if len(values) != 13:
        raise ValueError("Seqax numerical mutation expects the exact 13-input ABI")
    if mutation is SeqaxInputMutation.DROP_EMBEDDING_SHARD:
        midpoint = values[2].shape[1] // 2
        values[2][:, midpoint:] = 0
    elif mutation is SeqaxInputMutation.ROLL_MODEL_SHARD:
        midpoint = values[12].shape[1] // 2
        values[12] = np.roll(values[12], midpoint, axis=1)
    elif mutation is SeqaxInputMutation.OMIT_MLP_TERM:
        values[10].fill(0)
    elif mutation is SeqaxInputMutation.SWAP_GATE_UP:
        values[8], values[9] = values[9], values[8]
    return tuple(values)


def _is_bf16_tensor(value: ir.Value) -> bool:
    value_type = value.type.maybe_downcast()
    return isinstance(value_type, ir.RankedTensorType) and isinstance(
        value_type.element_type.maybe_downcast(), ir.BF16Type
    )


def _as_operation(value: object) -> ir.Operation | None:
    if isinstance(value, ir.Operation):
        return value
    operation = getattr(value, "operation", None)
    return operation if isinstance(operation, ir.Operation) else None


def _result_reaches_function_return(result: ir.Value) -> bool:
    pending = deque([result])
    visited: set[ir.Value] = set()
    while pending:
        value = pending.popleft()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            consumer = _as_operation(use.owner)
            assert consumer is not None
            if consumer.name == "func.return":
                return True
            if consumer.name in _REGION_TERMINATORS:
                parent = _as_operation(consumer.parent)
                if parent is not None and use.operand_number < len(parent.results):
                    pending.append(parent.results[use.operand_number])
            pending.extend(consumer.results)
    return False


def _function_operations(function: ir.Operation) -> tuple[ir.Operation, ...]:
    operations: list[ir.Operation] = []

    def visit(operation: ir.Operation) -> None:
        operations.append(operation)
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    visit(function)
    return tuple(operations)


def validate_strict_silu_stablehlo(stablehlo: str, *, expected_count: int) -> None:
    if expected_count <= 0:
        raise ValueError("strict SiLU StableHLO expected count must be positive")
    try:
        with mlir.make_ir_context():
            module = ir.Module.parse(stablehlo)
            module.operation.verify()
            strict_chains = 0
            for top_level in module.body:
                if top_level.operation.name != "func.func":
                    continue
                operations = _function_operations(top_level.operation)
                silu_calls = tuple(
                    operation
                    for operation in operations
                    if operation.name == "func.call"
                    and str(operation.attributes["callee"]) == "@silu"
                )
                for silu_call in silu_calls:
                    if len(silu_call.operands) != 1 or len(silu_call.results) != 1:
                        raise ValueError("strict SiLU call must have one input and one result")
                    source = silu_call.operands[0]
                    result = silu_call.results[0]
                    if not _is_bf16_tensor(source) or not _is_bf16_tensor(result):
                        raise ValueError("strict SiLU call must use BF16 tensors")
                    input_barrier = _as_operation(source.owner)
                    if input_barrier is None or input_barrier.name != (
                        "stablehlo.optimization_barrier"
                    ):
                        raise ValueError("strict SiLU StableHLO is missing its input barrier")
                    if tuple(_as_operation(use.owner) for use in source.uses) != (silu_call,):
                        raise ValueError("strict SiLU input barrier must feed only its SiLU call")
                    result_uses = tuple(result.uses)
                    if (
                        len(result_uses) != 1
                        or _as_operation(result_uses[0].owner).name
                        != "stablehlo.optimization_barrier"
                    ):
                        raise ValueError("strict SiLU result must feed only its result barrier")
                    result_barrier = _as_operation(result_uses[0].owner)
                    assert result_barrier is not None
                    if len(result_barrier.results) != 1:
                        raise ValueError(
                            "strict SiLU StableHLO is missing its unique result barrier"
                        )
                    barrier_result = result_barrier.results[0]
                    barrier_uses = tuple(barrier_result.uses)
                    if (
                        len(barrier_uses) != 1
                        or _as_operation(barrier_uses[0].owner).name != "stablehlo.multiply"
                    ):
                        raise ValueError(
                            "strict SiLU result barrier must feed exactly one BF16 multiply"
                        )
                    multiply = _as_operation(barrier_uses[0].owner)
                    assert multiply is not None
                    if not all(_is_bf16_tensor(value) for value in multiply.operands):
                        raise ValueError(
                            "strict SiLU result barrier must feed exactly one BF16 multiply"
                        )
                    if len(multiply.results) != 1 or not _is_bf16_tensor(multiply.results[0]):
                        raise ValueError(
                            "strict SiLU result barrier must feed exactly one BF16 multiply"
                        )
                    if not _result_reaches_function_return(multiply.results[0]):
                        raise ValueError("strict SiLU BF16 multiply must reach its function return")
                    strict_chains += 1
            if strict_chains != expected_count:
                raise ValueError(
                    f"strict SiLU StableHLO expected {expected_count} calls, found {strict_chains}"
                )
    except ir.MLIRError as error:
        raise ValueError("strict SiLU StableHLO is not valid MLIR") from error


def rounded_mathematical_silu_bf16(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.dtype(ml_dtypes.bfloat16):
        raise TypeError("mathematical SiLU reference requires BF16 input")
    source = value.astype(np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("mathematical SiLU reference requires finite input")
    sigmoid = np.empty_like(source)
    nonnegative = source >= 0
    sigmoid[nonnegative] = 1.0 / (1.0 + np.exp(-source[nonnegative]))
    exponential = np.exp(source[~nonnegative])
    sigmoid[~nonnegative] = exponential / (1.0 + exponential)
    return np.asarray(source * sigmoid, dtype=ml_dtypes.bfloat16)
