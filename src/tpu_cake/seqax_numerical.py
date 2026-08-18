from __future__ import annotations

import hashlib
import json
import math
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
    metric_quantization_decimals: int = Field(ge=12, le=15)
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
            self.metric_quantization_decimals,
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
            15,
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
    runtime: SeqaxBf16RuntimeContract
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    acceptance_authority: str = "authenticated-runner-and-relocated-public-replay"
    checkpoint_capture: str = "typed-extra-outputs-v1"
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
            self.checkpoint_capture,
            self.require_instrumented_output_parity,
            self.require_discriminator_artifact_replay,
        ) != (
            "authenticated-runner-and-relocated-public-replay",
            "typed-extra-outputs-v1",
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


class SeqaxBf16RuntimeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    python_major_minor: str = Field(pattern=r"^3\.12$")
    jax: str = Field(pattern=r"^0\.11\.0$")
    jaxlib: str = Field(pattern=r"^0\.11\.0$")
    libtpu: str = Field(pattern=r"^0\.0\.44\.1$")
    ml_dtypes: str = Field(pattern=r"^0\.6\.0$")
    libtpu_init_args: str = Field(pattern=r"^ --xla_tpu_use_enhanced_launch_barrier=true$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
            metric_quantization_decimals=15,
        ),
        scenarios=tuple(scenarios),
        required_discriminators=tuple(SeqaxNumericalDiscriminator),
        runtime=SeqaxBf16RuntimeContract(
            python_major_minor="3.12",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            ml_dtypes="0.6.0",
            libtpu_init_args=" --xla_tpu_use_enhanced_launch_barrier=true",
            uv_lock_sha256="7790b780e29c426595854b93c7bbde10571afe93bc13134c3ebc83df5e4f4c7b",
        ),
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
    )


def _relative_l2(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    quantization_decimals: int,
) -> float:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    numerator_squared = math.fsum(float(value) ** 2 for value in difference.ravel())
    denominator_squared = math.fsum(
        float(value) ** 2 for value in expected.astype(np.float64).ravel()
    )
    value = math.sqrt(numerator_squared / max(denominator_squared, 1e-60))
    return round(value, quantization_decimals)


def _row_scaled_max(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    scale_floor: float,
    quantization_decimals: int,
) -> float:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    row_scale = np.maximum(
        np.max(np.abs(expected.astype(np.float64)), axis=-1),
        scale_floor,
    )
    value = float(np.max(np.max(difference, axis=-1) / row_scale))
    return round(value, quantization_decimals)


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
    metric_decimals = policy.metric_quantization_decimals
    pallas_relative = _relative_l2(arrays[0], arrays[2], quantization_decimals=metric_decimals)
    control_relative = _relative_l2(arrays[1], arrays[2], quantization_decimals=metric_decimals)
    cross_relative = _relative_l2(arrays[0], arrays[1], quantization_decimals=metric_decimals)
    pallas_scaled = _row_scaled_max(
        arrays[0],
        arrays[2],
        scale_floor=policy.row_scale_floor,
        quantization_decimals=metric_decimals,
    )
    control_scaled = _row_scaled_max(
        arrays[1],
        arrays[2],
        scale_floor=policy.row_scale_floor,
        quantization_decimals=metric_decimals,
    )
    cross_scaled = _row_scaled_max(
        arrays[0],
        arrays[1],
        scale_floor=policy.row_scale_floor,
        quantization_decimals=metric_decimals,
    )
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


def _require_checkpoint_function_result(use: ir.OpOperand, expected_position: int) -> None:
    terminator = _as_operation(use.owner)
    if terminator is None or terminator.name not in _REGION_TERMINATORS:
        raise ValueError("instrumented strict SiLU checkpoint must leave its region")
    parent = _as_operation(terminator.parent)
    if parent is None or use.operand_number >= len(parent.results):
        raise ValueError("instrumented strict SiLU checkpoint has no parent result")
    parent_uses = tuple(parent.results[use.operand_number].uses)
    if (
        len(parent_uses) != 1
        or _as_operation(parent_uses[0].owner) is None
        or _as_operation(parent_uses[0].owner).name != "func.return"
        or parent_uses[0].operand_number != expected_position
    ):
        raise ValueError("instrumented strict SiLU checkpoint is not an exact function result")


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


def _require_single_result(operation: ir.Operation, expected_type: ir.Type) -> ir.Value:
    if len(operation.results) != 1 or operation.results[0].type != expected_type:
        raise ValueError("strict SiLU implementation has an invalid result type")
    return operation.results[0]


def _require_operands(operation: ir.Operation, expected: tuple[ir.Value, ...]) -> None:
    if tuple(operation.operands) != expected:
        raise ValueError("strict SiLU implementation has invalid operand wiring")


def _require_attribute_names(operation: ir.Operation, expected: frozenset[str]) -> None:
    observed = frozenset(str(name) for name in operation.attributes)
    if observed != expected:
        raise ValueError(
            "strict SiLU implementation has invalid operation attributes "
            f"operation={operation.name} expected={sorted(expected)} "
            f"observed={sorted(observed)}"
        )


def _require_bf16_one(operation: ir.Operation) -> ir.Value:
    _require_attribute_names(operation, frozenset({"value"}))
    if len(operation.results) != 1 or not _is_bf16_tensor(operation.results[0]):
        raise ValueError("strict SiLU implementation must use an exact BF16 one")
    result = operation.results[0]
    if ir.RankedTensorType(result.type).rank != 0:
        raise ValueError("strict SiLU implementation must use an exact BF16 one")
    value = ir.DenseElementsAttr(operation.attributes["value"])
    if not value.is_splat or str(value.get_splat_value()) != "1.000000e+00 : bf16":
        raise ValueError("strict SiLU implementation must use an exact BF16 one")
    return result


def _validate_silu_function(function: ir.Operation) -> None:
    _require_attribute_names(
        function,
        frozenset({"function_type", "sym_name", "sym_visibility"}),
    )
    if str(function.attributes["sym_visibility"]) != '"private"':
        raise ValueError("strict SiLU implementation must be private")
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise ValueError("strict SiLU implementation must have one body block")
    block = function.regions[0].blocks[0]
    if len(block.arguments) != 1 or not _is_bf16_tensor(block.arguments[0]):
        raise ValueError("strict SiLU implementation must take one BF16 tensor")
    argument = block.arguments[0]
    operations = tuple(operation.operation for operation in block.operations)
    expected_names = (
        "stablehlo.negate",
        "stablehlo.exponential",
        "stablehlo.constant",
        "stablehlo.broadcast_in_dim",
        "stablehlo.add",
        "stablehlo.constant",
        "stablehlo.broadcast_in_dim",
        "stablehlo.divide",
        "stablehlo.multiply",
        "func.return",
    )
    if tuple(operation.name for operation in operations) != expected_names:
        raise ValueError("strict SiLU implementation has an invalid operation sequence")

    (
        negate,
        exponential,
        one,
        broadcast_one,
        add,
        numerator,
        broadcast_numerator,
        divide,
        multiply,
        result,
    ) = operations
    tensor_type = argument.type
    for operation in (negate, exponential, add, divide, multiply, result):
        _require_attribute_names(operation, frozenset())
    for operation in (broadcast_one, broadcast_numerator):
        _require_attribute_names(operation, frozenset({"broadcast_dimensions"}))
    _require_operands(negate, (argument,))
    negated = _require_single_result(negate, tensor_type)
    _require_operands(exponential, (negated,))
    exponentiated = _require_single_result(exponential, tensor_type)
    one_value = _require_bf16_one(one)
    _require_operands(broadcast_one, (one_value,))
    broadcast_one_value = _require_single_result(broadcast_one, tensor_type)
    if str(broadcast_one.attributes["broadcast_dimensions"]) != "array<i64>":
        raise ValueError("strict SiLU implementation must scalar-broadcast BF16 one")
    _require_operands(add, (broadcast_one_value, exponentiated))
    denominator = _require_single_result(add, tensor_type)
    numerator_value = _require_bf16_one(numerator)
    _require_operands(broadcast_numerator, (numerator_value,))
    broadcast_numerator_value = _require_single_result(broadcast_numerator, tensor_type)
    if str(broadcast_numerator.attributes["broadcast_dimensions"]) != "array<i64>":
        raise ValueError("strict SiLU implementation must scalar-broadcast BF16 one")
    _require_operands(divide, (broadcast_numerator_value, denominator))
    sigmoid = _require_single_result(divide, tensor_type)
    _require_operands(multiply, (argument, sigmoid))
    silu = _require_single_result(multiply, tensor_type)
    _require_operands(result, (silu,))


def _validate_strict_silu_stablehlo(
    stablehlo: str,
    *,
    expected_count: int,
    instrumented: bool,
    leading_result_count: int = 0,
    allow_callbacks: bool = False,
) -> None:
    if expected_count <= 0:
        raise ValueError("strict SiLU StableHLO expected count must be positive")
    if leading_result_count not in {0, 1} or (leading_result_count != 0 and not instrumented):
        raise ValueError("strict SiLU StableHLO result offset is invalid")
    if leading_result_count != 0 and not allow_callbacks:
        raise ValueError("strict SiLU StableHLO callback policy is invalid")
    try:
        with mlir.make_ir_context():
            module = ir.Module.parse(stablehlo)
            module.operation.verify()
            silu_functions = tuple(
                operation.operation
                for operation in module.body
                if operation.operation.name == "func.func"
                and str(operation.operation.attributes["sym_name"]) == '"silu"'
            )
            if len(silu_functions) != 1:
                raise ValueError("strict SiLU StableHLO must define exactly one @silu function")
            _validate_silu_function(silu_functions[0])
            functions = tuple(
                operation.operation
                for operation in module.body
                if operation.operation.name == "func.func"
            )
            entry_functions = tuple(
                function
                for function in functions
                if str(function.attributes["sym_name"]) == '"main"'
                and str(function.attributes.get("sym_visibility")) == '"public"'
            )
            if len(entry_functions) != 1:
                raise ValueError("strict SiLU StableHLO must define one public @main entry")
            entry_function = entry_functions[0]
            entry_returns = tuple(
                operation
                for operation in _function_operations(entry_function)
                if operation.name == "func.return"
            )
            if instrumented and (
                len(entry_returns) != 1
                or len(entry_returns[0].operands) != leading_result_count + 1 + 2 * expected_count
            ):
                raise ValueError("instrumented strict SiLU has an invalid function result ABI")
            for function in functions:
                if function is entry_function:
                    continue
                if any(
                    operation.name == "func.call" and str(operation.attributes["callee"]) == "@silu"
                    for operation in _function_operations(function)
                ):
                    raise ValueError("strict SiLU calls are permitted only in public @main")
            strict_chains = 0
            operations = _function_operations(entry_function)
            silu_calls = tuple(
                operation
                for operation in operations
                if operation.name == "func.call" and str(operation.attributes["callee"]) == "@silu"
            )
            for layer, silu_call in enumerate(silu_calls):
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
                source_consumers = tuple(_as_operation(use.owner) for use in source.uses)
                result_consumers = tuple(_as_operation(use.owner) for use in result.uses)
                checkpoint_return: ir.Operation | None = None
                if instrumented:
                    checkpoint_uses = tuple(
                        use for use in source.uses if _as_operation(use.owner) != silu_call
                    )
                    if (
                        len(source_consumers) != 2
                        or not any(operation == silu_call for operation in source_consumers)
                        or len(checkpoint_uses) != 1
                        or _as_operation(checkpoint_uses[0].owner) is None
                        or _as_operation(checkpoint_uses[0].owner).name not in _REGION_TERMINATORS
                        or checkpoint_uses[0].operand_number != leading_result_count + 1 + 2 * layer
                    ):
                        raise ValueError(
                            "instrumented strict SiLU must return the real gate checkpoint"
                        )
                    _require_checkpoint_function_result(
                        checkpoint_uses[0],
                        leading_result_count + 1 + 2 * layer,
                    )
                    checkpoint_return = _as_operation(checkpoint_uses[0].owner)
                elif source_consumers != (silu_call,):
                    raise ValueError("strict SiLU input barrier must feed only its SiLU call")
                if (
                    len(result_consumers) != 1
                    or result_consumers[0] is None
                    or result_consumers[0].name != "stablehlo.optimization_barrier"
                ):
                    raise ValueError("strict SiLU result must feed only its result barrier")
                result_barrier = result_consumers[0]
                assert result_barrier is not None
                if len(result_barrier.results) != 1:
                    raise ValueError("strict SiLU StableHLO is missing its unique result barrier")
                barrier_result = result_barrier.results[0]
                barrier_uses = tuple(barrier_result.uses)
                multiply_uses = tuple(
                    use
                    for use in barrier_uses
                    if _as_operation(use.owner) is not None
                    and _as_operation(use.owner).name == "stablehlo.multiply"
                )
                if len(multiply_uses) != 1:
                    raise ValueError(
                        "strict SiLU result barrier must feed exactly one BF16 multiply"
                    )
                if instrumented:
                    checkpoint_uses = tuple(
                        use
                        for use in barrier_uses
                        if _as_operation(use.owner) is not None
                        and _as_operation(use.owner).name in _REGION_TERMINATORS
                    )
                    if (
                        len(barrier_uses) != 2
                        or len(checkpoint_uses) != 1
                        or _as_operation(checkpoint_uses[0].owner) != checkpoint_return
                        or checkpoint_uses[0].operand_number != leading_result_count + 2 + 2 * layer
                    ):
                        raise ValueError(
                            "instrumented strict SiLU must return the real SiLU checkpoint"
                        )
                    _require_checkpoint_function_result(
                        checkpoint_uses[0],
                        leading_result_count + 2 + 2 * layer,
                    )
                elif len(barrier_uses) != 1:
                    raise ValueError(
                        "strict SiLU result barrier must feed exactly one BF16 multiply"
                    )
                multiply = _as_operation(multiply_uses[0].owner)
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
            observed_callbacks = tuple(
                operation
                for operation in operations
                if operation.name == "stablehlo.custom_call"
                and "callback" in str(operation.attributes.get("call_target_name", "")).lower()
            )
            if observed_callbacks and not allow_callbacks:
                raise ValueError("strict SiLU StableHLO must not contain callbacks")
    except ir.MLIRError as error:
        raise ValueError("strict SiLU StableHLO is not valid MLIR") from error


def validate_strict_silu_stablehlo(stablehlo: str, *, expected_count: int) -> None:
    _validate_strict_silu_stablehlo(
        stablehlo,
        expected_count=expected_count,
        instrumented=False,
    )


def validate_instrumented_strict_silu_stablehlo(
    stablehlo: str,
    *,
    expected_count: int,
) -> None:
    _validate_strict_silu_stablehlo(
        stablehlo,
        expected_count=expected_count,
        instrumented=True,
    )


def _validate_relu_function(function: ir.Operation) -> None:
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise ValueError("ReLU discriminator must have one body block")
    block = function.regions[0].blocks[0]
    if len(block.arguments) != 1 or not _is_bf16_tensor(block.arguments[0]):
        raise ValueError("ReLU discriminator must take one BF16 tensor")
    operations = tuple(operation.operation for operation in block.operations)
    if tuple(operation.name for operation in operations) != (
        "stablehlo.constant",
        "stablehlo.broadcast_in_dim",
        "stablehlo.maximum",
        "func.return",
    ):
        raise ValueError("ReLU discriminator has an invalid operation sequence")
    zero, broadcast, maximum, result = operations
    _require_attribute_names(zero, frozenset({"value"}))
    if (
        len(zero.results) != 1
        or str(zero.results[0].type) != "tensor<bf16>"
        or str(zero.attributes["value"]) != "dense<0.000000e+00> : tensor<bf16>"
    ):
        raise ValueError("ReLU discriminator must use exact BF16 zero")
    _require_attribute_names(broadcast, frozenset({"broadcast_dimensions"}))
    _require_operands(broadcast, (zero.results[0],))
    broadcast_value = _require_single_result(broadcast, block.arguments[0].type)
    if str(broadcast.attributes["broadcast_dimensions"]) != "array<i64>":
        raise ValueError("ReLU discriminator must scalar-broadcast BF16 zero")
    _require_attribute_names(maximum, frozenset())
    _require_operands(maximum, (block.arguments[0], broadcast_value))
    maximum_value = _require_single_result(maximum, block.arguments[0].type)
    _require_attribute_names(result, frozenset())
    _require_operands(result, (maximum_value,))


def _validate_mutant_chain_result(result: ir.Value) -> None:
    result_uses = tuple(result.uses)
    if (
        len(result_uses) != 1
        or _as_operation(result_uses[0].owner).name != "stablehlo.optimization_barrier"
    ):
        raise ValueError("activation discriminator result must feed one result barrier")
    barrier = _as_operation(result_uses[0].owner)
    assert barrier is not None
    if len(barrier.results) != 1:
        raise ValueError("activation discriminator result barrier must have one result")
    barrier_uses = tuple(barrier.results[0].uses)
    if len(barrier_uses) != 1 or _as_operation(barrier_uses[0].owner).name != "stablehlo.multiply":
        raise ValueError("activation discriminator result barrier must feed one multiply")
    multiply = _as_operation(barrier_uses[0].owner)
    assert multiply is not None
    if (
        len(multiply.results) != 1
        or not all(_is_bf16_tensor(value) for value in multiply.operands)
        or not _is_bf16_tensor(multiply.results[0])
        or not _result_reaches_function_return(multiply.results[0])
    ):
        raise ValueError("activation discriminator multiply must reach the entrypoint return")


def validate_activation_mutant_stablehlo(
    stablehlo: str,
    *,
    expected_count: int,
    relu: bool,
) -> None:
    if expected_count <= 0:
        raise ValueError("activation discriminator expected count must be positive")
    try:
        with mlir.make_ir_context():
            module = ir.Module.parse(stablehlo)
            module.operation.verify()
            functions = tuple(
                operation.operation
                for operation in module.body
                if operation.operation.name == "func.func"
            )
            entries = tuple(
                function
                for function in functions
                if str(function.attributes["sym_name"]) == '"main"'
                and str(function.attributes.get("sym_visibility")) == '"public"'
            )
            if len(entries) != 1:
                raise ValueError("activation discriminator must have one public @main")
            entry = entries[0]
            operations = _function_operations(entry)
            if any(
                operation.name == "func.call" and str(operation.attributes["callee"]) == "@silu"
                for operation in operations
            ):
                raise ValueError("activation discriminator must not call strict SiLU")
            chains = 0
            if relu:
                relu_functions = tuple(
                    function
                    for function in functions
                    if str(function.attributes["sym_name"]) == '"relu"'
                )
                if len(relu_functions) != 1:
                    raise ValueError("ReLU discriminator must define exactly one @relu")
                _validate_relu_function(relu_functions[0])
                calls = tuple(
                    operation
                    for operation in operations
                    if operation.name == "func.call"
                    and str(operation.attributes["callee"]) == "@relu"
                )
                for call in calls:
                    if len(call.operands) != 1 or len(call.results) != 1:
                        raise ValueError("ReLU discriminator call has invalid arity")
                    source = call.operands[0]
                    source_owner = _as_operation(source.owner)
                    if (
                        source_owner is None
                        or source_owner.name != "stablehlo.optimization_barrier"
                        or tuple(_as_operation(use.owner) for use in source.uses) != (call,)
                        or not _is_bf16_tensor(source)
                        or not _is_bf16_tensor(call.results[0])
                    ):
                        raise ValueError("ReLU discriminator must consume the strict input barrier")
                    _validate_mutant_chain_result(call.results[0])
                    chains += 1
            else:
                for barrier in operations:
                    if (
                        barrier.name != "stablehlo.optimization_barrier"
                        or len(barrier.results) != 1
                    ):
                        continue
                    source = barrier.results[0]
                    source_uses = tuple(source.uses)
                    if (
                        len(source_uses) != 1
                        or _as_operation(source_uses[0].owner).name
                        != "stablehlo.optimization_barrier"
                    ):
                        continue
                    if not _is_bf16_tensor(source):
                        raise ValueError("identity discriminator must use BF16 barriers")
                    second_barrier = _as_operation(source_uses[0].owner)
                    assert second_barrier is not None
                    if len(second_barrier.results) != 1:
                        raise ValueError("identity discriminator result barrier is invalid")
                    _validate_mutant_chain_result(source)
                    chains += 1
            if chains != expected_count:
                raise ValueError(
                    f"activation discriminator expected {expected_count} chains, found {chains}"
                )
    except ir.MLIRError as error:
        raise ValueError("activation discriminator StableHLO is not valid MLIR") from error


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
