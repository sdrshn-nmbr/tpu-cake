from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable
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
SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA = "bf16-forward-numerical-v3"
SEQAX_BF16_HLO_IDENTITY_STATUS = "pending"
SEQAX_BF16_COMPILATION_SOURCE_ROOT = "/home/sudarshan/tpu-cake-main"
_CALIBRATION_SCHEMA = "bf16-forward-numerical-v1"
_V2_CALIBRATION_SCHEMA = "bf16-forward-numerical-v2"
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
_CALIBRATION_SURFACE_PARAMETERS = {
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
_CALIBRATION_SURFACE_SEEDS = {
    name: tuple(
        semantic_seed(
            _CALIBRATION_SCHEMA,
            f"{name}:sqrt-depth:{index}",
        )
        for index in range(4)
    )
    for name in _CALIBRATION_SURFACE_PARAMETERS
}
_V2_CALIBRATION_PARAMETERS = {
    "m192-b2-s4-l2": {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 32,
        "head": 4,
        "key_value_heads": 4,
        "layers": 2,
        "model": 192,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 4,
        "tensor_mesh": 4,
        "vocabulary": 48,
    },
    "m320-b4-s3-l1": {
        "batch": 4,
        "data_mesh": 2,
        "feed_forward": 40,
        "head": 4,
        "key_value_heads": 4,
        "layers": 1,
        "model": 320,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 3,
        "tensor_mesh": 4,
        "vocabulary": 48,
    },
    "m256-b2-s8-l4": {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 32,
        "head": 4,
        "key_value_heads": 4,
        "layers": 4,
        "model": 256,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 8,
        "tensor_mesh": 4,
        "vocabulary": 64,
    },
}
_V2_CALIBRATION_SEEDS = {
    name: tuple(
        semantic_seed(
            _V2_CALIBRATION_SCHEMA,
            f"{name}:held-out:{index}",
        )
        for index in range(4)
    )
    for name in _V2_CALIBRATION_PARAMETERS
}
_HELD_OUT_PARAMETERS = {
    "m224-b4-s5-l2": {
        "batch": 4,
        "data_mesh": 2,
        "feed_forward": 56,
        "head": 4,
        "key_value_heads": 4,
        "layers": 2,
        "model": 224,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 5,
        "tensor_mesh": 4,
        "vocabulary": 64,
    },
    "m352-b2-s6-l1": {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 44,
        "head": 4,
        "key_value_heads": 4,
        "layers": 1,
        "model": 352,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 6,
        "tensor_mesh": 4,
        "vocabulary": 64,
    },
    "m288-b4-s4-l3": {
        "batch": 4,
        "data_mesh": 2,
        "feed_forward": 48,
        "head": 4,
        "key_value_heads": 4,
        "layers": 3,
        "model": 288,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 4,
        "tensor_mesh": 4,
        "vocabulary": 64,
    },
}
_STABLEHLO_SHA256 = {
    "calibration-m256-b2-s1-l1": {
        "pallas": "f914d06da5716168c9ca447ef9f26b37ba84f42be6cace2860f4ab03d730425b",
        "control": "e012666b6b40f2d9e0efac1517312ebcb29e2a27ae6999cb63cdd112b12a331f",
        "instrumented_pallas": "a2af894fa96cb8e9658ecdb854d5645d79d4743f2d78abf4f26c868ebd5a4a03",
        "instrumented_control": "bf118a132a3a6be5fcd62becce670364d676ee778eb029a53ce7744b583ebfb1",
    },
    "m128-b2-s3-l2": {
        "pallas": "a48ec8246f774059681d055362912a92775b4a8b11a7d0032ba0f49ca8590b3a",
        "control": "f027b8411006ac97e2b548aaf26f9561b02c920dfa4af7979db36e91aaf6561c",
        "instrumented_pallas": "41c5bc0bd02743f70f7d16523c241b9b9637a81041e215ae730cd8ccf9b00897",
        "instrumented_control": "90fd1818ce0fd577c992ec569ca97145b7dbbd78b8ca355c924608b9b613c7b9",
    },
    "m256-b4-s2-l1": {
        "pallas": "c1e3bc28366ad47032b2b23245d5628b3dc7826cfb0200957daff685558a0029",
        "control": "176cde5eda31748aa89ce5ed5634c4732877a77d7fefa6f5883fa2438655e159",
        "instrumented_pallas": "018e93660c463bbc7d7ce56eceee4e2f03066c3e8f5c56d1e107f95b1f630314",
        "instrumented_control": "50e9200f6093b3279f0f78bf8ce7e1ce596a96ccd9a2b8e28f0b90f164da0565",
    },
    "m384-b2-s2-l1": {
        "pallas": "6a3032320ea43d96e973e430e8290673b71448b5c0611edbf9679359d08fea34",
        "control": "8efc8cec47c7f7901bae8dfcb042a32ac3c1645ac960381749d3ef00dbd4c740",
        "instrumented_pallas": "5c6e9f7c31449be1b3dedd2158b6025e9a8b331e592da227a17d90ad1ecaf30a",
        "instrumented_control": "f7525f6eef598329764b57c1517b9e392df11619cd5173e0382c023d2f9bfc2f",
    },
    "m192-b2-s4-l2": {
        "pallas": "f82161148b6f2e36afccca3adfbcf8ea700a879df5297482390405fd3d35c973",
        "control": "7cc8b16442e5d98a70a0a513e96c9de608146bcae7431bf184be8e859112de50",
        "instrumented_pallas": "baeb21604a50cd77807fbd9cc90603a425677af053fba8659186492394646cf1",
        "instrumented_control": "4c541d8832bb03534f7989cbd9c2dc1387c9d6f8620746ea0bf4cf7fa1ee2641",
    },
    "m320-b4-s3-l1": {
        "pallas": "acd2361cc4a14ebeb0d097299a0f079671d8db1ef7045d63076b17c56aace960",
        "control": "14d9569435ab5b8892fdf98eeba6a1da7e9999e1dd998cf6f61efc7a46521340",
        "instrumented_pallas": "e03e5ed1b6f1935dff582d62dc6a0e063a1acf752bdb98560b92aca9ab04f0fc",
        "instrumented_control": "fc1107a6331d5321091aa2d3bbee0fe13e738daab72300efce0861d53c9d6822",
    },
    "m256-b2-s8-l4": {
        "pallas": "220306763eb8c308e331c17931145bedcf1c504559b0e9aaaedb8b2be58c248e",
        "control": "68c42cf8d7b5e98f41a449c536e3092982b863c87698546193bc47df47456b06",
        "instrumented_pallas": "e1744714945f2960f6434b7a6fa318f22779a5a218f9b23ee7d7677ac7983159",
        "instrumented_control": "0215702f5ca20e8841d33cd818e0fd41f257e5eb5dbe82ecba1cc744343079d3",
    },
    "m224-b4-s5-l2": {
        "pallas": "1aa604c425d158797cc5e95ab4e27ba73fe7e4cffe8b76631208caf36e37be2c",
        "control": "5b5c87f00b189ad942fe6da3b4c62b00dd3cd7fea75ea7e65250e6db8f1ab285",
        "instrumented_pallas": "06b51213587d4ed05289631bc093ef5d847bbe52a3a165f3cc82230343e0d13b",
        "instrumented_control": "41395212f0b09255f015184bf595224c33c660a2bafe2d784b005c482cc22c44",
    },
    "m352-b2-s6-l1": {
        "pallas": "b92905ad6c3b031f3db23958a5e1a4d721d761cc5418e3ffe9de84d180fe7a69",
        "control": "580e88491762c2907e3ac853486c8cd2c3207762e07f0d869cba42dc1bdd9062",
        "instrumented_pallas": "6ebb99f3c9d73faba3a8b19e2d49f56694d0a828f7825fd48fbc0cf4091f97f7",
        "instrumented_control": "617d65edb907f42e1885b848576aa1402324aa422f97461bd25874417e776271",
    },
    "m288-b4-s4-l3": {
        "pallas": "8b5c8e95eee7a2f9b7a73c72a9e596b1ff3b2a10f2bdd8419b5aa1cbe7572e88",
        "control": "5a32f34548ddbab5998586d3d6f489e45581eca8a88b9c7fd1798f500f533e54",
        "instrumented_pallas": "1f174e03cc3c95ffe99f0f71a648be9854ce6e1b5b2a7f1065314c1272236070",
        "instrumented_control": "df32c73e17a9912ef72b372c98aa3714dc808d0b3a80106c1b37d71ed208e6d9",
    },
}
_ACTIVATION_MUTANT_STABLEHLO_SHA256 = {
    "identity_silu": {
        "pallas": "32f83731b4e76caff7ea6f41476e38e2059aef863ccf3aefc31e651371ea150e",
        "control": "cef7589996b812a807a93c1d2113fd22fa27ca851a265240ac44372c486f9a89",
    },
    "relu_silu": {
        "pallas": "9b266c4b2b1000a7dc52815f9881b7ddeedc99f45bfd5e89f0d0e470b077ab27",
        "control": "1ae5bdc96b55a530bbae97f950507572caa2d40dbce27b7b3d22aa32acf42f85",
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
    REMOVE_HIDDEN_BARRIER = "remove_hidden_barrier"
    IDENTITY_SILU = "identity_silu"
    RELU_SILU = "relu_silu"
    CORRUPT_DOWN_CHECKPOINT = "corrupt_down_checkpoint"
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
    DOWN_PROJECTION_ORACLE = "down_projection_oracle"
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
    SeqaxNumericalDiscriminator.REMOVE_HIDDEN_BARRIER: (
        SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE
    ),
    SeqaxNumericalDiscriminator.IDENTITY_SILU: (SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE),
    SeqaxNumericalDiscriminator.RELU_SILU: (SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE),
    SeqaxNumericalDiscriminator.CORRUPT_DOWN_CHECKPOINT: (
        SeqaxDiscriminatorClause.DOWN_PROJECTION_ORACLE
    ),
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


def canonical_seqax_stablehlo(stablehlo: str) -> str:
    return stablehlo.rstrip("\n") + "\n"


def seqax_stablehlo_sha256(stablehlo: str) -> str:
    return hashlib.sha256(canonical_seqax_stablehlo(stablehlo).encode()).hexdigest()


def _require_stablehlo_identity(stablehlo: str, expected_sha256: str) -> None:
    if seqax_stablehlo_sha256(stablehlo) != expected_sha256:
        raise ValueError("Seqax StableHLO trusted identity mismatch")


class SeqaxBf16NumericalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA
    numerical_semantics: str = "typed_bf16_hidden_v2"
    unit_roundoff: float = BF16_UNIT_ROUNDOFF
    cpu_relative_l2_units: float = Field(gt=0)
    cpu_row_scaled_max_units: float = Field(gt=0)
    cross_path_relative_l2_units: float = Field(gt=0)
    cross_path_row_scaled_max_units: float = Field(gt=0)
    depth_scaling: str = "sqrt_layers"
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
        if self.numerical_semantics != "typed_bf16_hidden_v2":
            raise ValueError("Seqax BF16 numerical policy requires typed BF16 semantics")
        if self.unit_roundoff != BF16_UNIT_ROUNDOFF:
            raise ValueError("Seqax BF16 numerical policy unit roundoff mismatch")
        if (
            self.cpu_relative_l2_units,
            self.cpu_row_scaled_max_units,
            self.cross_path_relative_l2_units,
            self.cross_path_row_scaled_max_units,
            self.depth_scaling,
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
            3.0,
            8.0,
            2.0,
            2.0,
            "sqrt_layers",
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

    def depth_scale(self, layers: int) -> float:
        if type(layers) is not int or layers <= 0:
            raise ValueError("Seqax BF16 numerical depth must be a positive integer")
        return math.sqrt(layers)


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
    tuple[SeqaxNumericalTensorContract, ...],
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
    up_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_up",
            (batch, sequence, feed_forward),
            "bfloat16",
        )
        for layer in range(layers)
    )
    hidden_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_hidden",
            (batch, sequence, feed_forward),
            "bfloat16",
        )
        for layer in range(layers)
    )
    down_float32_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_down_float32",
            (batch, sequence, model),
            "float32",
        )
        for layer in range(layers)
    )
    down_bfloat16_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_down_bfloat16",
            (batch, sequence, model),
            "bfloat16",
        )
        for layer in range(layers)
    )
    return (
        inputs,
        output,
        gate_checkpoints,
        silu_checkpoints,
        up_checkpoints,
        hidden_checkpoints,
        down_float32_checkpoints,
        down_bfloat16_checkpoints,
    )


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
    up_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    hidden_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    down_float32_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    down_bfloat16_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrumented_control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def parameter_schema_is_complete(self) -> SeqaxBf16NumericalScenario:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seqax BF16 numerical scenario seeds must be unique")
        (
            expected_inputs,
            expected_output,
            expected_gates,
            expected_silu,
            expected_up,
            expected_hidden,
            expected_down_float32,
            expected_down_bfloat16,
        ) = _scenario_abi(self.parameters)
        if self.inputs != expected_inputs:
            raise ValueError("Seqax BF16 numerical scenario input ABI mismatch")
        if self.output != expected_output:
            raise ValueError("Seqax BF16 numerical scenario output ABI mismatch")
        if self.gate_checkpoints != expected_gates:
            raise ValueError("Seqax BF16 numerical scenario gate checkpoint ABI mismatch")
        if self.silu_checkpoints != expected_silu:
            raise ValueError("Seqax BF16 numerical scenario SiLU checkpoint ABI mismatch")
        if self.up_checkpoints != expected_up:
            raise ValueError("Seqax BF16 numerical scenario up checkpoint ABI mismatch")
        if self.hidden_checkpoints != expected_hidden:
            raise ValueError("Seqax BF16 numerical scenario hidden checkpoint ABI mismatch")
        if self.down_float32_checkpoints != expected_down_float32:
            raise ValueError("Seqax BF16 numerical scenario float32 down checkpoint ABI mismatch")
        if self.down_bfloat16_checkpoints != expected_down_bfloat16:
            raise ValueError("Seqax BF16 numerical scenario BF16 down checkpoint ABI mismatch")
        expected_hlo = _STABLEHLO_SHA256.get(self.name)
        if expected_hlo is None or (
            self.pallas_stablehlo_sha256,
            self.control_stablehlo_sha256,
            self.instrumented_pallas_stablehlo_sha256,
            self.instrumented_control_stablehlo_sha256,
        ) != (
            expected_hlo["pallas"],
            expected_hlo["control"],
            expected_hlo["instrumented_pallas"],
            expected_hlo["instrumented_control"],
        ):
            raise ValueError("Seqax BF16 numerical scenario StableHLO identity mismatch")
        return self


class SeqaxBf16ActivationMutantStablehloContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    discriminator: SeqaxNumericalDiscriminator
    pallas_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_stablehlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_is_canonical(self) -> SeqaxBf16ActivationMutantStablehloContract:
        expected = _ACTIVATION_MUTANT_STABLEHLO_SHA256.get(self.discriminator.value)
        if expected is None or (
            self.pallas_stablehlo_sha256,
            self.control_stablehlo_sha256,
        ) != (expected["pallas"], expected["control"]):
            raise ValueError("Seqax BF16 activation mutant StableHLO identity mismatch")
        return self


class SeqaxBf16ValidationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA
    policy: SeqaxBf16NumericalPolicy
    scenarios: tuple[SeqaxBf16NumericalScenario, ...] = Field(min_length=10, max_length=10)
    activation_mutant_stablehlo: tuple[SeqaxBf16ActivationMutantStablehloContract, ...] = Field(
        min_length=2, max_length=2
    )
    required_discriminators: tuple[SeqaxNumericalDiscriminator, ...]
    runtime: SeqaxBf16RuntimeContract
    compilation_source_root: str
    hlo_identity_status: str = Field(pattern=r"^(pending|pinned)$")
    backend: str
    device_kind: str
    device_count: int = Field(gt=0)
    acceptance_authority: str = "authenticated-runner-and-relocated-public-replay"
    checkpoint_capture: str = "typed-strict-mlp-extra-outputs-v3"
    require_instrumented_output_parity: bool = True
    require_discriminator_artifact_replay: bool = True

    @model_validator(mode="after")
    def validation_surface_is_canonical(self) -> SeqaxBf16ValidationContract:
        if self.schema_version != SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA:
            raise ValueError("Seqax BF16 validation schema mismatch")
        if (self.backend, self.device_kind, self.device_count) != ("tpu", "TPU7x", 8):
            raise ValueError("Seqax BF16 validation hardware contract mismatch")
        if self.compilation_source_root != SEQAX_BF16_COMPILATION_SOURCE_ROOT:
            raise ValueError("Seqax BF16 validation compilation source root mismatch")
        if (
            self.acceptance_authority,
            self.checkpoint_capture,
            self.require_instrumented_output_parity,
            self.require_discriminator_artifact_replay,
        ) != (
            "authenticated-runner-and-relocated-public-replay",
            "typed-strict-mlp-extra-outputs-v3",
            True,
            True,
        ):
            raise ValueError("Seqax BF16 validation acceptance authority mismatch")
        if self.required_discriminators != tuple(SeqaxNumericalDiscriminator):
            raise ValueError("Seqax BF16 validation discriminators are not canonical")
        if tuple(value.discriminator for value in self.activation_mutant_stablehlo) != (
            SeqaxNumericalDiscriminator.IDENTITY_SILU,
            SeqaxNumericalDiscriminator.RELU_SILU,
        ):
            raise ValueError("Seqax BF16 activation mutant StableHLO order mismatch")
        if tuple(scenario.name for scenario in self.scenarios) != (
            "calibration-m256-b2-s1-l1",
            *_CALIBRATION_SURFACE_PARAMETERS,
            *_V2_CALIBRATION_PARAMETERS,
            *_HELD_OUT_PARAMETERS,
        ):
            raise ValueError("Seqax BF16 validation scenarios are not canonical")
        for scenario in self.scenarios:
            if scenario.name == "calibration-m256-b2-s1-l1":
                expected_role = SeqaxNumericalScenarioRole.CALIBRATION
                expected_parameters = _CALIBRATION_PARAMETERS
                expected_seeds = _CALIBRATION_SEEDS
            elif scenario.name in _CALIBRATION_SURFACE_PARAMETERS:
                expected_role = SeqaxNumericalScenarioRole.CALIBRATION
                expected_parameters = _CALIBRATION_SURFACE_PARAMETERS[scenario.name]
                expected_seeds = _CALIBRATION_SURFACE_SEEDS[scenario.name]
            elif scenario.name in _V2_CALIBRATION_PARAMETERS:
                expected_role = SeqaxNumericalScenarioRole.CALIBRATION
                expected_parameters = _V2_CALIBRATION_PARAMETERS[scenario.name]
                expected_seeds = _V2_CALIBRATION_SEEDS[scenario.name]
            else:
                expected_role = SeqaxNumericalScenarioRole.HELD_OUT
                expected_parameters = _HELD_OUT_PARAMETERS.get(scenario.name)
                expected_seeds = tuple(
                    semantic_seed(
                        self.schema_version,
                        f"{scenario.name}:held-out:{index}",
                    )
                    for index in range(4)
                )
            if scenario.role is not expected_role:
                raise ValueError(f"Seqax BF16 validation scenario role mismatch: {scenario.name}")
            if expected_parameters is None or scenario.parameters != (
                SeqaxBf16ScenarioParameters(**expected_parameters)
            ):
                raise ValueError(
                    f"Seqax BF16 validation scenario parameters mismatch: {scenario.name}"
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
    up_cross_path_exact: bool
    pallas_hidden_matches_product: bool
    control_hidden_matches_product: bool
    hidden_cross_path_exact: bool
    pallas_down_float32_max_bound_ratio: float = Field(ge=0)
    control_down_float32_max_bound_ratio: float = Field(ge=0)
    pallas_down_float32_within_bound: bool
    control_down_float32_within_bound: bool
    pallas_down_bfloat16_matches_float32: bool
    control_down_bfloat16_matches_float32: bool
    down_bfloat16_cross_path_exact: bool
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
    def scenario_hlo(name: str) -> dict[str, str]:
        expected = _STABLEHLO_SHA256[name]
        return {
            "pallas_stablehlo_sha256": expected["pallas"],
            "control_stablehlo_sha256": expected["control"],
            "instrumented_pallas_stablehlo_sha256": expected["instrumented_pallas"],
            "instrumented_control_stablehlo_sha256": expected["instrumented_control"],
        }

    calibration_parameters = SeqaxBf16ScenarioParameters(**_CALIBRATION_PARAMETERS)
    (
        calibration_inputs,
        calibration_output,
        calibration_gates,
        calibration_silu,
        calibration_up,
        calibration_hidden,
        calibration_down_float32,
        calibration_down_bfloat16,
    ) = _scenario_abi(calibration_parameters)
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
            up_checkpoints=calibration_up,
            hidden_checkpoints=calibration_hidden,
            down_float32_checkpoints=calibration_down_float32,
            down_bfloat16_checkpoints=calibration_down_bfloat16,
            **scenario_hlo("calibration-m256-b2-s1-l1"),
        )
    ]
    for name, raw_parameters in _CALIBRATION_SURFACE_PARAMETERS.items():
        parameters = SeqaxBf16ScenarioParameters(**raw_parameters)
        inputs, output, gates, silu, up, hidden, down_float32, down_bfloat16 = _scenario_abi(
            parameters
        )
        scenarios.append(
            SeqaxBf16NumericalScenario(
                name=name,
                role=SeqaxNumericalScenarioRole.CALIBRATION,
                parameters=parameters,
                seeds=_CALIBRATION_SURFACE_SEEDS[name],
                inputs=inputs,
                output=output,
                gate_checkpoints=gates,
                silu_checkpoints=silu,
                up_checkpoints=up,
                hidden_checkpoints=hidden,
                down_float32_checkpoints=down_float32,
                down_bfloat16_checkpoints=down_bfloat16,
                **scenario_hlo(name),
            )
        )
    for name, raw_parameters in _V2_CALIBRATION_PARAMETERS.items():
        parameters = SeqaxBf16ScenarioParameters(**raw_parameters)
        inputs, output, gates, silu, up, hidden, down_float32, down_bfloat16 = _scenario_abi(
            parameters
        )
        scenarios.append(
            SeqaxBf16NumericalScenario(
                name=name,
                role=SeqaxNumericalScenarioRole.CALIBRATION,
                parameters=parameters,
                seeds=_V2_CALIBRATION_SEEDS[name],
                inputs=inputs,
                output=output,
                gate_checkpoints=gates,
                silu_checkpoints=silu,
                up_checkpoints=up,
                hidden_checkpoints=hidden,
                down_float32_checkpoints=down_float32,
                down_bfloat16_checkpoints=down_bfloat16,
                **scenario_hlo(name),
            )
        )
    for name, raw_parameters in _HELD_OUT_PARAMETERS.items():
        parameters = SeqaxBf16ScenarioParameters(**raw_parameters)
        inputs, output, gates, silu, up, hidden, down_float32, down_bfloat16 = _scenario_abi(
            parameters
        )
        scenarios.append(
            SeqaxBf16NumericalScenario(
                name=name,
                role=SeqaxNumericalScenarioRole.HELD_OUT,
                parameters=parameters,
                seeds=tuple(
                    semantic_seed(
                        SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA,
                        f"{name}:held-out:{index}",
                    )
                    for index in range(4)
                ),
                inputs=inputs,
                output=output,
                gate_checkpoints=gates,
                silu_checkpoints=silu,
                up_checkpoints=up,
                hidden_checkpoints=hidden,
                down_float32_checkpoints=down_float32,
                down_bfloat16_checkpoints=down_bfloat16,
                **scenario_hlo(name),
            )
        )
    return SeqaxBf16ValidationContract(
        policy=SeqaxBf16NumericalPolicy(
            cpu_relative_l2_units=3.0,
            cpu_row_scaled_max_units=8.0,
            cross_path_relative_l2_units=2.0,
            cross_path_row_scaled_max_units=2.0,
            row_scale_floor=1.0,
            metric_quantization_decimals=15,
        ),
        scenarios=tuple(scenarios),
        activation_mutant_stablehlo=tuple(
            SeqaxBf16ActivationMutantStablehloContract(
                discriminator=SeqaxNumericalDiscriminator(name),
                pallas_stablehlo_sha256=value["pallas"],
                control_stablehlo_sha256=value["control"],
            )
            for name, value in _ACTIVATION_MUTANT_STABLEHLO_SHA256.items()
        ),
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
        compilation_source_root=SEQAX_BF16_COMPILATION_SOURCE_ROOT,
        hlo_identity_status=SEQAX_BF16_HLO_IDENTITY_STATUS,
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


def _validate_float32_checkpoints(
    values: tuple[np.ndarray, ...],
    expected: tuple[SeqaxNumericalTensorContract, ...],
    *,
    label: str,
) -> tuple[np.ndarray, ...]:
    if not values or len(values) != len(expected):
        raise ValueError(f"Seqax BF16 {label} checkpoint count does not match the contract")
    arrays = tuple(np.asarray(value) for value in values)
    for checkpoint, contract in zip(arrays, expected, strict=True):
        if checkpoint.dtype != np.float32:
            raise TypeError(f"Seqax BF16 {label} checkpoints must use float32")
        if checkpoint.shape != contract.shape:
            raise ValueError(f"Seqax BF16 {label} checkpoint shape does not match the contract")
        if not np.all(np.isfinite(checkpoint)):
            raise ValueError(f"Seqax BF16 {label} checkpoints must be finite")
    return arrays


def _down_projection_reference_components(
    hidden: np.ndarray,
    down_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden64 = hidden.astype(np.float64)
    weight64 = down_weight.astype(ml_dtypes.bfloat16).astype(np.float64)
    output_shape = (*hidden.shape[:-1], down_weight.shape[0])
    reference = np.empty(output_shape, dtype=np.float64)
    absolute_sum = np.empty(output_shape, dtype=np.float64)
    for batch, sequence, model in np.ndindex(output_shape):
        products = tuple(
            float(hidden64[batch, sequence, feed_forward]) * float(weight64[model, feed_forward])
            for feed_forward in range(hidden.shape[-1])
        )
        reference[batch, sequence, model] = math.fsum(products)
        absolute_sum[batch, sequence, model] = math.fsum(abs(value) for value in products)
    return reference, absolute_sum


def seqax_down_projection_reference_float32(
    hidden: np.ndarray,
    down_weight: np.ndarray,
) -> np.ndarray:
    hidden_array = np.asarray(hidden)
    down_weight_array = np.asarray(down_weight)
    if hidden_array.dtype != np.dtype(ml_dtypes.bfloat16):
        raise TypeError("Seqax down projection reference requires BF16 hidden values")
    if down_weight_array.dtype != np.float32:
        raise TypeError("Seqax down projection reference requires float32 down weights")
    if hidden_array.ndim != 3 or down_weight_array.ndim != 2:
        raise ValueError(
            "Seqax down projection reference requires rank-3 hidden and rank-2 weights"
        )
    if hidden_array.shape[-1] != down_weight_array.shape[-1]:
        raise ValueError("Seqax down projection reference contraction shape mismatch")
    if not np.all(np.isfinite(hidden_array)) or not np.all(np.isfinite(down_weight_array)):
        raise ValueError("Seqax down projection reference requires finite inputs")
    reference, _absolute_sum = _down_projection_reference_components(
        hidden_array, down_weight_array
    )
    return reference.astype(np.float32)


def _down_projection_bound_ratio(
    actual: np.ndarray,
    hidden: np.ndarray,
    down_weight: np.ndarray,
    *,
    tensor_mesh: int,
) -> float:
    reference, absolute_sum = _down_projection_reference_components(hidden, down_weight)
    additions = hidden.shape[-1] + tensor_mesh
    float32_unit_roundoff = 2.0**-24
    gamma = additions * float32_unit_roundoff / (1.0 - additions * float32_unit_roundoff)
    bound = gamma * absolute_sum + np.finfo(np.float32).tiny
    ratio = np.abs(actual.astype(np.float64) - reference) / bound
    return round(float(np.max(ratio)), 15)


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
    pallas_up_checkpoints: tuple[np.ndarray, ...],
    control_up_checkpoints: tuple[np.ndarray, ...],
    pallas_hidden_checkpoints: tuple[np.ndarray, ...],
    control_hidden_checkpoints: tuple[np.ndarray, ...],
    pallas_down_float32_checkpoints: tuple[np.ndarray, ...],
    control_down_float32_checkpoints: tuple[np.ndarray, ...],
    pallas_down_bfloat16_checkpoints: tuple[np.ndarray, ...],
    control_down_bfloat16_checkpoints: tuple[np.ndarray, ...],
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
    pallas_up = _validate_bf16_checkpoints(
        pallas_up_checkpoints,
        scenario.up_checkpoints,
        label="up",
    )
    control_up = _validate_bf16_checkpoints(
        control_up_checkpoints,
        scenario.up_checkpoints,
        label="up",
    )
    pallas_hidden = _validate_bf16_checkpoints(
        pallas_hidden_checkpoints,
        scenario.hidden_checkpoints,
        label="hidden",
    )
    control_hidden = _validate_bf16_checkpoints(
        control_hidden_checkpoints,
        scenario.hidden_checkpoints,
        label="hidden",
    )
    pallas_down_float32 = _validate_float32_checkpoints(
        pallas_down_float32_checkpoints,
        scenario.down_float32_checkpoints,
        label="float32 down",
    )
    control_down_float32 = _validate_float32_checkpoints(
        control_down_float32_checkpoints,
        scenario.down_float32_checkpoints,
        label="float32 down",
    )
    pallas_down_bfloat16 = _validate_bf16_checkpoints(
        pallas_down_bfloat16_checkpoints,
        scenario.down_bfloat16_checkpoints,
        label="BF16 down",
    )
    control_down_bfloat16 = _validate_bf16_checkpoints(
        control_down_bfloat16_checkpoints,
        scenario.down_bfloat16_checkpoints,
        label="BF16 down",
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
    up_cross_path = all(
        np.array_equal(pallas_value, control_value)
        for pallas_value, control_value in zip(pallas_up, control_up, strict=True)
    )
    pallas_hidden_mathematical = all(
        np.array_equal(
            actual,
            np.asarray(
                silu.astype(np.float32) * up.astype(np.float32),
                dtype=ml_dtypes.bfloat16,
            ),
        )
        for actual, silu, up in zip(pallas_hidden, pallas_silu, pallas_up, strict=True)
    )
    control_hidden_mathematical = all(
        np.array_equal(
            actual,
            np.asarray(
                silu.astype(np.float32) * up.astype(np.float32),
                dtype=ml_dtypes.bfloat16,
            ),
        )
        for actual, silu, up in zip(control_hidden, control_silu, control_up, strict=True)
    )
    hidden_cross_path = all(
        np.array_equal(pallas_value, control_value)
        for pallas_value, control_value in zip(pallas_hidden, control_hidden, strict=True)
    )
    down_weights = expected_inputs[10]
    pallas_down_ratios = tuple(
        _down_projection_bound_ratio(
            actual,
            hidden,
            down_weights[layer],
            tensor_mesh=scenario.parameters.tensor_mesh,
        )
        for layer, (actual, hidden) in enumerate(
            zip(pallas_down_float32, pallas_hidden, strict=True)
        )
    )
    control_down_ratios = tuple(
        _down_projection_bound_ratio(
            actual,
            hidden,
            down_weights[layer],
            tensor_mesh=scenario.parameters.tensor_mesh,
        )
        for layer, (actual, hidden) in enumerate(
            zip(control_down_float32, control_hidden, strict=True)
        )
    )
    pallas_down_ratio = max(pallas_down_ratios)
    control_down_ratio = max(control_down_ratios)
    pallas_down_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(pallas_down_bfloat16, pallas_down_float32, strict=True)
    )
    control_down_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(control_down_bfloat16, control_down_float32, strict=True)
    )
    down_bfloat16_cross_path = all(
        np.array_equal(pallas_value, control_value)
        for pallas_value, control_value in zip(
            pallas_down_bfloat16, control_down_bfloat16, strict=True
        )
    )
    unit = policy.unit_roundoff
    depth_scale = policy.depth_scale(scenario.parameters.layers)
    final_outputs_satisfy_policy = (
        pallas_relative <= policy.cpu_relative_l2_units * unit * depth_scale
        and control_relative <= policy.cpu_relative_l2_units * unit * depth_scale
        and cross_relative <= policy.cross_path_relative_l2_units * unit * depth_scale
        and pallas_scaled <= policy.cpu_row_scaled_max_units * unit * depth_scale
        and control_scaled <= policy.cpu_row_scaled_max_units * unit * depth_scale
        and cross_scaled <= policy.cross_path_row_scaled_max_units * unit * depth_scale
    )
    checkpoint_values_consistent = (
        gate_cross_path
        and pallas_silu_mathematical
        and control_silu_mathematical
        and silu_cross_path
        and up_cross_path
        and pallas_hidden_mathematical
        and control_hidden_mathematical
        and hidden_cross_path
        and pallas_down_ratio <= 1.0
        and control_down_ratio <= 1.0
        and pallas_down_bfloat16_matches
        and control_down_bfloat16_matches
        and down_bfloat16_cross_path
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
        up_cross_path_exact=up_cross_path,
        pallas_hidden_matches_product=pallas_hidden_mathematical,
        control_hidden_matches_product=control_hidden_mathematical,
        hidden_cross_path_exact=hidden_cross_path,
        pallas_down_float32_max_bound_ratio=pallas_down_ratio,
        control_down_float32_max_bound_ratio=control_down_ratio,
        pallas_down_float32_within_bound=pallas_down_ratio <= 1.0,
        control_down_float32_within_bound=control_down_ratio <= 1.0,
        pallas_down_bfloat16_matches_float32=pallas_down_bfloat16_matches,
        control_down_bfloat16_matches_float32=control_down_bfloat16_matches,
        down_bfloat16_cross_path_exact=down_bfloat16_cross_path,
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


def _is_f32_tensor(value: ir.Value) -> bool:
    value_type = value.type.maybe_downcast()
    return isinstance(value_type, ir.RankedTensorType) and isinstance(
        value_type.element_type.maybe_downcast(), ir.F32Type
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


def _follow_shape_only_reshapes(
    value: ir.Value,
    *,
    ignored_terminators: bool,
    expected_dtype: Callable[[ir.Value], bool],
) -> ir.OpOperand:
    current = value
    first = True
    while True:
        uses = tuple(
            use
            for use in current.uses
            if not (
                first
                and ignored_terminators
                and _as_operation(use.owner) is not None
                and _as_operation(use.owner).name in _REGION_TERMINATORS
            )
        )
        if len(uses) != 1:
            raise ValueError("strict MLP shape-only path must have one semantic use")
        use = uses[0]
        operation = _as_operation(use.owner)
        if operation is None or operation.name != "stablehlo.reshape":
            return use
        if use.operand_number != 0:
            raise ValueError("strict MLP reshape must consume operand zero")
        _require_attribute_names(operation, frozenset())
        if len(operation.results) != 1 or not expected_dtype(operation.results[0]):
            raise ValueError("strict MLP reshape has an invalid result")
        current = operation.results[0]
        first = False


def _require_f32_one(operation: ir.Operation) -> ir.Value:
    _require_attribute_names(operation, frozenset({"value"}))
    if len(operation.results) != 1 or not _is_f32_tensor(operation.results[0]):
        raise ValueError("strict SiLU implementation must use an exact float32 one")
    result = operation.results[0]
    if ir.RankedTensorType(result.type).rank != 0:
        raise ValueError("strict SiLU implementation must use an exact float32 one")
    value = ir.DenseElementsAttr(operation.attributes["value"])
    if not value.is_splat or str(value.get_splat_value()) != "1.000000e+00 : f32":
        raise ValueError("strict SiLU implementation must use an exact float32 one")
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
    if len(block.arguments) != 1 or not _is_f32_tensor(block.arguments[0]):
        raise ValueError("strict SiLU implementation must take one float32 tensor")
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
    one_value = _require_f32_one(one)
    _require_operands(broadcast_one, (one_value,))
    broadcast_one_value = _require_single_result(broadcast_one, tensor_type)
    if str(broadcast_one.attributes["broadcast_dimensions"]) != "array<i64>":
        raise ValueError("strict SiLU implementation must scalar-broadcast float32 one")
    _require_operands(add, (broadcast_one_value, exponentiated))
    denominator = _require_single_result(add, tensor_type)
    numerator_value = _require_f32_one(numerator)
    _require_operands(broadcast_numerator, (numerator_value,))
    broadcast_numerator_value = _require_single_result(broadcast_numerator, tensor_type)
    if str(broadcast_numerator.attributes["broadcast_dimensions"]) != "array<i64>":
        raise ValueError("strict SiLU implementation must scalar-broadcast float32 one")
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
    require_hidden_down: bool = True,
) -> None:
    if expected_count <= 0:
        raise ValueError("strict SiLU StableHLO expected count must be positive")
    if leading_result_count not in {0, 1} or (leading_result_count != 0 and not instrumented):
        raise ValueError("strict SiLU StableHLO result offset is invalid")
    if leading_result_count != 0 and not allow_callbacks:
        raise ValueError("strict SiLU StableHLO callback policy is invalid")
    if instrumented and not require_hidden_down:
        raise ValueError("instrumented strict MLP validation requires hidden/down authority")
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
                or len(entry_returns[0].operands) != leading_result_count + 1 + 6 * expected_count
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
                call_source = silu_call.operands[0]
                call_result = silu_call.results[0]
                if not _is_f32_tensor(call_source) or not _is_f32_tensor(call_result):
                    raise ValueError("strict SiLU call must use float32 tensors")
                input_convert = _as_operation(call_source.owner)
                if (
                    input_convert is None
                    or input_convert.name != "stablehlo.convert"
                    or len(input_convert.operands) != 1
                    or len(input_convert.results) != 1
                    or input_convert.results[0] != call_source
                    or not _is_bf16_tensor(input_convert.operands[0])
                    or tuple(_as_operation(use.owner) for use in call_source.uses) != (silu_call,)
                ):
                    raise ValueError("strict SiLU must promote its barrier output to float32")
                _require_attribute_names(input_convert, frozenset())
                source = input_convert.operands[0]
                input_barrier = _as_operation(source.owner)
                if input_barrier is None or input_barrier.name != (
                    "stablehlo.optimization_barrier"
                ):
                    raise ValueError("strict SiLU StableHLO is missing its input barrier")
                source_consumers = tuple(_as_operation(use.owner) for use in source.uses)
                call_result_consumers = tuple(_as_operation(use.owner) for use in call_result.uses)
                checkpoint_return: ir.Operation | None = None
                if instrumented:
                    checkpoint_uses = tuple(
                        use for use in source.uses if _as_operation(use.owner) != input_convert
                    )
                    if (
                        len(source_consumers) != 2
                        or not any(operation == input_convert for operation in source_consumers)
                        or len(checkpoint_uses) != 1
                        or _as_operation(checkpoint_uses[0].owner) is None
                        or _as_operation(checkpoint_uses[0].owner).name not in _REGION_TERMINATORS
                        or checkpoint_uses[0].operand_number != leading_result_count + 1 + 6 * layer
                    ):
                        raise ValueError(
                            "instrumented strict SiLU must return the real gate checkpoint"
                        )
                    _require_checkpoint_function_result(
                        checkpoint_uses[0],
                        leading_result_count + 1 + 6 * layer,
                    )
                    checkpoint_return = _as_operation(checkpoint_uses[0].owner)
                elif source_consumers != (input_convert,):
                    raise ValueError("strict SiLU input barrier must feed only its promotion")
                if (
                    len(call_result_consumers) != 1
                    or call_result_consumers[0] is None
                    or call_result_consumers[0].name != "stablehlo.convert"
                ):
                    raise ValueError("strict SiLU result must feed only its BF16 conversion")
                output_convert = call_result_consumers[0]
                assert output_convert is not None
                if (
                    len(output_convert.operands) != 1
                    or len(output_convert.results) != 1
                    or output_convert.operands[0] != call_result
                    or not _is_bf16_tensor(output_convert.results[0])
                ):
                    raise ValueError("strict SiLU must round its float32 result to BF16")
                _require_attribute_names(output_convert, frozenset())
                converted_result = output_convert.results[0]
                converted_consumers = tuple(
                    _as_operation(use.owner) for use in converted_result.uses
                )
                if (
                    len(converted_consumers) != 1
                    or converted_consumers[0] is None
                    or converted_consumers[0].name != "stablehlo.optimization_barrier"
                ):
                    raise ValueError("strict SiLU BF16 result must feed only its result barrier")
                result_barrier = converted_consumers[0]
                assert result_barrier is not None
                if len(result_barrier.results) != 1:
                    raise ValueError("strict SiLU StableHLO is missing its unique result barrier")
                barrier_result = result_barrier.results[0]
                barrier_uses = tuple(barrier_result.uses)
                expected_multiply_consumer = (
                    "stablehlo.optimization_barrier"
                    if require_hidden_down
                    else "stablehlo.multiply"
                )
                multiply_input_uses = tuple(
                    use
                    for use in barrier_uses
                    if _as_operation(use.owner) is not None
                    and _as_operation(use.owner).name == expected_multiply_consumer
                )
                if len(multiply_input_uses) != 1:
                    raise ValueError("strict SiLU result barrier has invalid multiply dataflow")
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
                        or checkpoint_uses[0].operand_number != leading_result_count + 2 + 6 * layer
                    ):
                        raise ValueError(
                            "instrumented strict SiLU must return the real SiLU checkpoint"
                        )
                    _require_checkpoint_function_result(
                        checkpoint_uses[0],
                        leading_result_count + 2 + 6 * layer,
                    )
                elif len(barrier_uses) != 1:
                    raise ValueError(
                        "strict SiLU result barrier must feed only the strict multiply input barrier"
                    )
                if require_hidden_down:
                    multiply_input_barrier = _as_operation(multiply_input_uses[0].owner)
                    assert multiply_input_barrier is not None
                    _require_attribute_names(multiply_input_barrier, frozenset())
                    if len(multiply_input_barrier.results) != 1 or not _is_bf16_tensor(
                        multiply_input_barrier.results[0]
                    ):
                        raise ValueError("strict multiply input barrier has an invalid result")
                    multiply_input = multiply_input_barrier.results[0]
                    multiply_uses = tuple(multiply_input.uses)
                    if (
                        len(multiply_uses) != 1
                        or _as_operation(multiply_uses[0].owner) is None
                        or _as_operation(multiply_uses[0].owner).name != "stablehlo.multiply"
                    ):
                        raise ValueError(
                            "strict multiply input barrier must feed exactly one BF16 multiply"
                        )
                    multiply = _as_operation(multiply_uses[0].owner)
                    assert multiply is not None
                else:
                    multiply = _as_operation(multiply_input_uses[0].owner)
                    assert multiply is not None
                if not all(_is_bf16_tensor(value) for value in multiply.operands):
                    raise ValueError(
                        "strict SiLU result barrier must feed exactly one BF16 multiply"
                    )
                if len(multiply.results) != 1 or not _is_bf16_tensor(multiply.results[0]):
                    raise ValueError(
                        "strict SiLU result barrier must feed exactly one BF16 multiply"
                    )
                if require_hidden_down:
                    up_operands = tuple(
                        operand for operand in multiply.operands if operand != multiply_input
                    )
                    if len(up_operands) != 1 or not _is_bf16_tensor(up_operands[0]):
                        raise ValueError("strict hidden multiply must have one BF16 up operand")
                    up_operand = up_operands[0]
                    up_uses = tuple(up_operand.uses)
                    up_checkpoint_uses = tuple(
                        use
                        for use in up_uses
                        if _as_operation(use.owner) is not None
                        and _as_operation(use.owner).name in _REGION_TERMINATORS
                    )
                    if instrumented:
                        up_position = leading_result_count + 3 + 6 * layer
                        if (
                            len(up_uses) != 2
                            or len(up_checkpoint_uses) != 1
                            or up_checkpoint_uses[0].operand_number != up_position
                        ):
                            raise ValueError(
                                "instrumented strict MLP must return the real up checkpoint"
                            )
                        _require_checkpoint_function_result(up_checkpoint_uses[0], up_position)
                    elif len(up_uses) != 1:
                        raise ValueError("strict hidden up operand must feed only its multiply")
                if not require_hidden_down:
                    if not _result_reaches_function_return(multiply.results[0]):
                        raise ValueError("strict SiLU BF16 multiply must reach its function return")
                    strict_chains += 1
                    continue
                hidden = multiply.results[0]
                hidden_consumers = tuple(_as_operation(use.owner) for use in hidden.uses)
                if (
                    len(hidden_consumers) != 1
                    or hidden_consumers[0] is None
                    or hidden_consumers[0].name != "stablehlo.optimization_barrier"
                ):
                    raise ValueError(
                        "strict hidden BF16 multiply must feed only its result barrier"
                    )
                hidden_barrier = hidden_consumers[0]
                assert hidden_barrier is not None
                _require_attribute_names(hidden_barrier, frozenset())
                if len(hidden_barrier.results) != 1 or not _is_bf16_tensor(
                    hidden_barrier.results[0]
                ):
                    raise ValueError("strict hidden materialization has an invalid result")
                materialized_hidden = hidden_barrier.results[0]
                hidden_uses = tuple(materialized_hidden.uses)
                hidden_checkpoint_uses = tuple(
                    use
                    for use in hidden_uses
                    if _as_operation(use.owner) is not None
                    and _as_operation(use.owner).name in _REGION_TERMINATORS
                )
                if instrumented:
                    hidden_position = leading_result_count + 4 + 6 * layer
                    if (
                        len(hidden_checkpoint_uses) != 1
                        or hidden_checkpoint_uses[0].operand_number != hidden_position
                    ):
                        raise ValueError(
                            "instrumented strict MLP must return the materialized hidden checkpoint"
                        )
                    _require_checkpoint_function_result(hidden_checkpoint_uses[0], hidden_position)
                elif len(hidden_uses) != 1:
                    raise ValueError(
                        "strict hidden materialization must feed only the down projection"
                    )
                if allow_callbacks:
                    if not _result_reaches_function_return(materialized_hidden):
                        raise ValueError(
                            "CPU interpret hidden checkpoint must reach its function return"
                        )
                    strict_chains += 1
                    continue

                def is_down_projection(use: ir.OpOperand) -> bool:
                    operation = _as_operation(use.owner)
                    if operation is None:
                        return False
                    if operation.name == "stablehlo.dot_general":
                        return True
                    return (
                        operation.name == "stablehlo.custom_call"
                        and str(operation.attributes.get("call_target_name")) == '"tpu_custom_call"'
                        and str(operation.attributes.get("kernel_name")) == '"seqax_named_einsum"'
                    )

                hidden_dot_use = _follow_shape_only_reshapes(
                    materialized_hidden,
                    ignored_terminators=instrumented,
                    expected_dtype=_is_bf16_tensor,
                )
                expected_hidden_use_count = 2 if instrumented else 1
                if (
                    len(hidden_uses) != expected_hidden_use_count
                    or not is_down_projection(hidden_dot_use)
                    or hidden_dot_use.operand_number != 0
                ):
                    raise ValueError(
                        "strict hidden materialization must feed the down projection lhs"
                    )
                down_dot = _as_operation(hidden_dot_use.owner)
                assert down_dot is not None
                if len(down_dot.results) != 1 or not _is_f32_tensor(down_dot.results[0]):
                    raise ValueError("strict MLP down projection must produce float32")
                down_dot_use = _follow_shape_only_reshapes(
                    down_dot.results[0],
                    ignored_terminators=False,
                    expected_dtype=_is_f32_tensor,
                )
                if (
                    _as_operation(down_dot_use.owner) is None
                    or _as_operation(down_dot_use.owner).name != "stablehlo.reduce_scatter"
                ):
                    raise ValueError("strict MLP down projection must feed one reduce-scatter")
                down_reduce_scatter = _as_operation(down_dot_use.owner)
                assert down_reduce_scatter is not None
                if len(down_reduce_scatter.results) != 1 or not _is_f32_tensor(
                    down_reduce_scatter.results[0]
                ):
                    raise ValueError("strict MLP reduce-scatter must produce float32")
                down_float32 = down_reduce_scatter.results[0]
                down_float32_uses = tuple(down_float32.uses)
                down_converts = tuple(
                    use
                    for use in down_float32_uses
                    if _as_operation(use.owner) is not None
                    and _as_operation(use.owner).name == "stablehlo.convert"
                )
                down_float32_checkpoints = tuple(
                    use
                    for use in down_float32_uses
                    if _as_operation(use.owner) is not None
                    and _as_operation(use.owner).name in _REGION_TERMINATORS
                )
                if len(down_converts) != 1:
                    raise ValueError("strict MLP float32 down result must have one BF16 cast")
                if instrumented:
                    down_float32_position = leading_result_count + 5 + 6 * layer
                    if (
                        len(down_float32_uses) != 2
                        or len(down_float32_checkpoints) != 1
                        or down_float32_checkpoints[0].operand_number != down_float32_position
                    ):
                        raise ValueError(
                            "instrumented strict MLP must return the float32 down checkpoint"
                        )
                    _require_checkpoint_function_result(
                        down_float32_checkpoints[0], down_float32_position
                    )
                elif len(down_float32_uses) != 1:
                    raise ValueError("strict MLP float32 down result must feed only its BF16 cast")
                down_convert = _as_operation(down_converts[0].owner)
                assert down_convert is not None
                _require_attribute_names(down_convert, frozenset())
                if len(down_convert.results) != 1 or not _is_bf16_tensor(down_convert.results[0]):
                    raise ValueError("strict MLP down cast must produce BF16")
                down_bfloat16 = down_convert.results[0]
                down_bfloat16_uses = tuple(down_bfloat16.uses)
                residual_uses = tuple(
                    use
                    for use in down_bfloat16_uses
                    if _as_operation(use.owner) is not None
                    and _as_operation(use.owner).name == "stablehlo.add"
                )
                down_bfloat16_checkpoints = tuple(
                    use
                    for use in down_bfloat16_uses
                    if _as_operation(use.owner) is not None
                    and _as_operation(use.owner).name in _REGION_TERMINATORS
                )
                if len(residual_uses) != 1:
                    raise ValueError("strict MLP BF16 down result must feed one residual add")
                if instrumented:
                    down_bfloat16_position = leading_result_count + 6 + 6 * layer
                    if (
                        len(down_bfloat16_uses) != 2
                        or len(down_bfloat16_checkpoints) != 1
                        or down_bfloat16_checkpoints[0].operand_number != down_bfloat16_position
                    ):
                        raise ValueError(
                            "instrumented strict MLP must return the BF16 down checkpoint"
                        )
                    _require_checkpoint_function_result(
                        down_bfloat16_checkpoints[0], down_bfloat16_position
                    )
                elif len(down_bfloat16_uses) != 1:
                    raise ValueError("strict MLP BF16 down result must feed only the residual add")
                residual = _as_operation(residual_uses[0].owner)
                assert residual is not None
                if not _result_reaches_function_return(residual.results[0]):
                    raise ValueError("strict MLP residual must reach its function return")
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


def validate_strict_silu_stablehlo(
    stablehlo: str,
    *,
    expected_count: int,
    expected_sha256: str,
) -> None:
    _require_stablehlo_identity(stablehlo, expected_sha256)
    _validate_strict_silu_stablehlo(
        stablehlo,
        expected_count=expected_count,
        instrumented=False,
    )


def validate_instrumented_strict_silu_stablehlo(
    stablehlo: str,
    *,
    expected_count: int,
    expected_sha256: str,
) -> None:
    _require_stablehlo_identity(stablehlo, expected_sha256)
    _validate_strict_silu_stablehlo(
        stablehlo,
        expected_count=expected_count,
        instrumented=True,
    )


def _validate_relu_function(function: ir.Operation) -> None:
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise ValueError("ReLU discriminator must have one body block")
    block = function.regions[0].blocks[0]
    if len(block.arguments) != 1 or not _is_f32_tensor(block.arguments[0]):
        raise ValueError("ReLU discriminator must take one float32 tensor")
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
        or str(zero.results[0].type) != "tensor<f32>"
        or str(zero.attributes["value"]) != "dense<0.000000e+00> : tensor<f32>"
    ):
        raise ValueError("ReLU discriminator must use exact float32 zero")
    _require_attribute_names(broadcast, frozenset({"broadcast_dimensions"}))
    _require_operands(broadcast, (zero.results[0],))
    broadcast_value = _require_single_result(broadcast, block.arguments[0].type)
    if str(broadcast.attributes["broadcast_dimensions"]) != "array<i64>":
        raise ValueError("ReLU discriminator must scalar-broadcast float32 zero")
    _require_attribute_names(maximum, frozenset())
    _require_operands(maximum, (block.arguments[0], broadcast_value))
    maximum_value = _require_single_result(maximum, block.arguments[0].type)
    _require_attribute_names(result, frozenset())
    _require_operands(result, (maximum_value,))


def _validate_mutant_chain_result(result: ir.Value) -> None:
    result_uses = tuple(result.uses)
    if len(result_uses) != 1 or _as_operation(result_uses[0].owner).name != "stablehlo.convert":
        raise ValueError("activation discriminator result must feed one BF16 conversion")
    conversion = _as_operation(result_uses[0].owner)
    assert conversion is not None
    if (
        len(conversion.operands) != 1
        or len(conversion.results) != 1
        or not _is_f32_tensor(conversion.operands[0])
        or not _is_bf16_tensor(conversion.results[0])
    ):
        raise ValueError("activation discriminator result must round from float32 to BF16")
    _require_attribute_names(conversion, frozenset())
    converted_uses = tuple(conversion.results[0].uses)
    if (
        len(converted_uses) != 1
        or _as_operation(converted_uses[0].owner).name != "stablehlo.optimization_barrier"
    ):
        raise ValueError("activation discriminator BF16 result must feed one result barrier")
    barrier = _as_operation(converted_uses[0].owner)
    assert barrier is not None
    if len(barrier.results) != 1:
        raise ValueError("activation discriminator result barrier must have one result")
    barrier_uses = tuple(barrier.results[0].uses)
    if (
        len(barrier_uses) != 1
        or _as_operation(barrier_uses[0].owner) is None
        or _as_operation(barrier_uses[0].owner).name != "stablehlo.optimization_barrier"
    ):
        raise ValueError(
            "activation discriminator result barrier must feed the strict multiply barrier"
        )
    multiply_barrier = _as_operation(barrier_uses[0].owner)
    assert multiply_barrier is not None
    _require_attribute_names(multiply_barrier, frozenset())
    if len(multiply_barrier.results) != 1 or not _is_bf16_tensor(multiply_barrier.results[0]):
        raise ValueError("activation discriminator multiply barrier is invalid")
    multiply_uses = tuple(multiply_barrier.results[0].uses)
    if (
        len(multiply_uses) != 1
        or _as_operation(multiply_uses[0].owner) is None
        or _as_operation(multiply_uses[0].owner).name != "stablehlo.multiply"
    ):
        raise ValueError("activation discriminator multiply barrier must feed one multiply")
    multiply = _as_operation(multiply_uses[0].owner)
    assert multiply is not None
    if (
        len(multiply.results) != 1
        or not all(_is_bf16_tensor(value) for value in multiply.operands)
        or not _is_bf16_tensor(multiply.results[0])
    ):
        raise ValueError("activation discriminator multiply must use BF16 tensors")
    hidden_uses = tuple(multiply.results[0].uses)
    if (
        len(hidden_uses) != 1
        or _as_operation(hidden_uses[0].owner) is None
        or _as_operation(hidden_uses[0].owner).name != "stablehlo.optimization_barrier"
    ):
        raise ValueError("activation discriminator hidden value must feed its result barrier")
    hidden_barrier = _as_operation(hidden_uses[0].owner)
    assert hidden_barrier is not None
    _require_attribute_names(hidden_barrier, frozenset())
    if len(hidden_barrier.results) != 1 or not _is_bf16_tensor(hidden_barrier.results[0]):
        raise ValueError("activation discriminator hidden result barrier is invalid")
    down_uses = tuple(hidden_barrier.results[0].uses)
    if len(down_uses) != 1 or down_uses[0].operand_number != 0:
        raise ValueError("activation discriminator hidden result must feed one down projection")
    down = _as_operation(down_uses[0].owner)
    if down is None or down.name not in {"stablehlo.dot_general", "stablehlo.custom_call"}:
        raise ValueError("activation discriminator hidden result must feed one down projection")
    if down.name == "stablehlo.custom_call" and not (
        str(down.attributes.get("call_target_name")) == '"tpu_custom_call"'
        and str(down.attributes.get("kernel_name")) == '"seqax_named_einsum"'
    ):
        raise ValueError("activation discriminator hidden result has an invalid down projection")
    if len(down.results) != 1 or not _is_f32_tensor(down.results[0]):
        raise ValueError("activation discriminator down projection must produce float32")
    reduce_uses = tuple(down.results[0].uses)
    if (
        len(reduce_uses) != 1
        or _as_operation(reduce_uses[0].owner) is None
        or _as_operation(reduce_uses[0].owner).name != "stablehlo.reduce_scatter"
    ):
        raise ValueError("activation discriminator down projection must feed one reduce-scatter")
    reduce_scatter = _as_operation(reduce_uses[0].owner)
    assert reduce_scatter is not None
    if len(reduce_scatter.results) != 1 or not _is_f32_tensor(reduce_scatter.results[0]):
        raise ValueError("activation discriminator reduce-scatter must produce float32")
    cast_uses = tuple(reduce_scatter.results[0].uses)
    if (
        len(cast_uses) != 1
        or _as_operation(cast_uses[0].owner) is None
        or _as_operation(cast_uses[0].owner).name != "stablehlo.convert"
    ):
        raise ValueError("activation discriminator reduce-scatter must feed one BF16 cast")
    down_cast = _as_operation(cast_uses[0].owner)
    assert down_cast is not None
    _require_attribute_names(down_cast, frozenset())
    if len(down_cast.results) != 1 or not _is_bf16_tensor(down_cast.results[0]):
        raise ValueError("activation discriminator down cast must produce BF16")
    residual_uses = tuple(down_cast.results[0].uses)
    if (
        len(residual_uses) != 1
        or _as_operation(residual_uses[0].owner) is None
        or _as_operation(residual_uses[0].owner).name != "stablehlo.add"
    ):
        raise ValueError("activation discriminator down result must feed one residual add")
    residual = _as_operation(residual_uses[0].owner)
    assert residual is not None
    if len(residual.results) != 1 or not _result_reaches_function_return(residual.results[0]):
        raise ValueError("activation discriminator residual must reach the entrypoint return")


def _validate_activation_mutant_stablehlo(
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
                    gate = (
                        source_owner.operands[0]
                        if source_owner is not None
                        and source_owner.name == "stablehlo.convert"
                        and len(source_owner.operands) == 1
                        else None
                    )
                    gate_owner = _as_operation(gate.owner) if gate is not None else None
                    if (
                        source_owner is None
                        or source_owner.name != "stablehlo.convert"
                        or gate is None
                        or gate_owner is None
                        or gate_owner.name != "stablehlo.optimization_barrier"
                        or tuple(_as_operation(use.owner) for use in source.uses) != (call,)
                        or tuple(_as_operation(use.owner) for use in gate.uses) != (source_owner,)
                        or not _is_bf16_tensor(gate)
                        or not _is_f32_tensor(source)
                        or not _is_f32_tensor(call.results[0])
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
                        or _as_operation(source_uses[0].owner).name != "stablehlo.convert"
                    ):
                        continue
                    if not _is_bf16_tensor(source):
                        raise ValueError("identity discriminator must use BF16 barriers")
                    promotion = _as_operation(source_uses[0].owner)
                    assert promotion is not None
                    if (
                        len(promotion.operands) != 1
                        or len(promotion.results) != 1
                        or not _is_f32_tensor(promotion.results[0])
                    ):
                        raise ValueError("identity discriminator promotion is invalid")
                    _validate_mutant_chain_result(promotion.results[0])
                    chains += 1
            if chains != expected_count:
                raise ValueError(
                    f"activation discriminator expected {expected_count} chains, found {chains}"
                )
    except ir.MLIRError as error:
        raise ValueError("activation discriminator StableHLO is not valid MLIR") from error


def validate_activation_mutant_stablehlo(
    stablehlo: str,
    *,
    expected_count: int,
    expected_sha256: str,
    relu: bool,
) -> None:
    _require_stablehlo_identity(stablehlo, expected_sha256)
    _validate_activation_mutant_stablehlo(
        stablehlo,
        expected_count=expected_count,
        relu=relu,
    )


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
