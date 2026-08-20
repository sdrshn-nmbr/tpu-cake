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
SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA = "bf16-forward-numerical-v5"
SEQAX_BF16_HLO_IDENTITY_STATUS = "pinned"
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
    "m240-b2-s7-l2": {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 60,
        "head": 4,
        "key_value_heads": 4,
        "layers": 2,
        "model": 240,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 7,
        "tensor_mesh": 4,
        "vocabulary": 64,
    },
    "m416-b4-s3-l1": {
        "batch": 4,
        "data_mesh": 2,
        "feed_forward": 52,
        "head": 4,
        "key_value_heads": 4,
        "layers": 1,
        "model": 416,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 3,
        "tensor_mesh": 4,
        "vocabulary": 64,
    },
    "m272-b2-s9-l3": {
        "batch": 2,
        "data_mesh": 2,
        "feed_forward": 68,
        "head": 4,
        "key_value_heads": 4,
        "layers": 3,
        "model": 272,
        "query_groups": 2,
        "rope_max_timescale": 256,
        "sequence": 9,
        "tensor_mesh": 4,
        "vocabulary": 64,
    },
}
_STABLEHLO_SHA256 = {
    "calibration-m256-b2-s1-l1": {
        "pallas": "aa7b6af57d8ab77c06747c46a31091998cdb449c5e0ff0894d425ddfd06641ae",
        "control": "05f377de78b292c90d020b8d865285c807ed2e5d3814c0b4977da09c629cffa6",
        "instrumented_pallas": "82032504f755d6485035fd8142e0323177928ec7369bf7ffbf966b00f9e0b6e5",
        "instrumented_control": "e9980b0fd41a0e1e0898fa59946b6dff911772a57e017b5596f3a5d55a97ad0a",
    },
    "m128-b2-s3-l2": {
        "pallas": "7c18eda7e1951421cefe44430f9f897d23f823210e23fa9a18c9e25425bb3ab6",
        "control": "3fd1c90846f71c80f141a2db8b5b6b15ab8e0c840444bc920c8f8115a1d70021",
        "instrumented_pallas": "db270f066229731aeb5601884e659e08fca2444de66d76a399f42a60f27b95e9",
        "instrumented_control": "7cf32ba85a08866bc731eb4cd8530fe78fbeb989d99e41672153fa7054dc54e2",
    },
    "m256-b4-s2-l1": {
        "pallas": "2e57c9234a684f3453ef5e234bd7889626bad4c31088b45a5dd17481383f3e39",
        "control": "72addb368ded5587116f727253fe983b832cafb644291cdd286dbb9156f9cc74",
        "instrumented_pallas": "384097fb860b225e8785cde71ed26a031c5660c16cf06f5fdf2b8416f8e8ebae",
        "instrumented_control": "7f61b8dfc0a64000a9602c47ce062fa3fcd3cc561d7c11af722561b8244c28c1",
    },
    "m384-b2-s2-l1": {
        "pallas": "44a5569e10046f01d837d7b2ee2d79751be94fe558c4434269cd439ba3528370",
        "control": "35e3b6c7e12ae821fe6edee8ee515c8da1c0109cd02c345762d66b838e2cea76",
        "instrumented_pallas": "c959a6e7c41aefb9bec6f091c7388b2b914ff5e9a45e7fc823e7a9727c2af9fb",
        "instrumented_control": "bccedf167f0b9bfb71151fb81895ac6c978b09941574b797d3a795253b4aa2a6",
    },
    "m192-b2-s4-l2": {
        "pallas": "6249025460ce9418ce087273072d6d96209c05e356d49f26347182db60bb2531",
        "control": "aa7a9344ee40fc5269c4fef332fa51ec02bb958c105142eb23ab31178d7d19a7",
        "instrumented_pallas": "39636de1e87e73f1a923bf8a274a6dae869c229906dc98e6da5e53a740f9f62f",
        "instrumented_control": "e627cfd0d8c6a45483f10dd50f0bc62864410dc3749f18f0319a64092482bba9",
    },
    "m320-b4-s3-l1": {
        "pallas": "750804aa2a24bdabbd65da65e89b819a3487e5fb42738f285960e23b76f0d5b3",
        "control": "87ef688fd0a932c6878894da5a6e564b190236ef6768bc3e86580132de4b4676",
        "instrumented_pallas": "45cb18809686d3044276c7a9d0adf809f5323919cf8ae826a26faf18693e5358",
        "instrumented_control": "4297f162e9acf9d5c6821f89b0a4203561b52b27274330aa6a1cd8022c1ff4a3",
    },
    "m256-b2-s8-l4": {
        "pallas": "19426e64a79ad76fdc75702daaefbf2c43abfe48b84522773244029d3e1da03d",
        "control": "0514db06552295c04707be2bcdc0d0800801ed79326a1556be916d3d31d4cff3",
        "instrumented_pallas": "9e9e098b9db1c9e8685ac9a85e5c20848de78490fbaa3a01b787e7a031d8af3a",
        "instrumented_control": "01c5c72b190f8c52ffc7eac137c182b645649ccd5bfa3472e10d1d60b6c0e7b5",
    },
    "m240-b2-s7-l2": {
        "pallas": "93f28ddd92db10eedcd856a2d723d307288efa638b768c14d79f7a1a38c611e0",
        "control": "3d7413bfd40bfeb64526cd80d53ae20d1ab630437cf4c2d512757c32c6851c22",
        "instrumented_pallas": "07ba1ea9f974d86424317fce090a030c4b1f0eb9b7d7190fdee73c52ed47f042",
        "instrumented_control": "d46a704854d87aa18a4af7d4479f2adf0edf441f4e981e2821465d45dba7c107",
    },
    "m416-b4-s3-l1": {
        "pallas": "0e0d38ac32c605fa697590e4298db1a8038769c7f28b3069a42470e2c8dd4186",
        "control": "6e312627d1501a76e2338219103af7f74cb6eb95e797fe12ee9ac966a678f90f",
        "instrumented_pallas": "6520cc17803155f57592dee3ac69e2ac2c81238c05264b472166dd6d433a5bcb",
        "instrumented_control": "7b356aa5e4067ca0564125017f59fe64b0e73bbbc5c181fa3cdd5a416e991b0f",
    },
    "m272-b2-s9-l3": {
        "pallas": "b647fb19fb6b401c7b7c1468481a0747d409d8c67baf5381cf3b9c4183eb4c73",
        "control": "712966e66b048bc426f17c9d309afcfb271e1d021a4c10bde47f116ae4e4cf69",
        "instrumented_pallas": "694fe5098ef05097cdd091ece373f4cddb01903fd6957c4770aedec663279f5b",
        "instrumented_control": "83ef4a7000b7a54833409a19242a0ec30f485df1b734e05d0076ba4820bd0c8d",
    },
}
_ACTIVATION_MUTANT_STABLEHLO_SHA256 = {
    "identity_silu": {
        "pallas": "0cdf402b2bbc65a0558d0abd74b6ced1d5cf24f8d13c0c6836948dffafc4603d",
        "control": "96af1a60b031e515dc674fafaf8b79964133a13e2972d2a717d2d105af86ee7d",
    },
    "relu_silu": {
        "pallas": "c94cd093d7191755250c6ebaf1d2ca0e105d616b850dd831cdec286c9fa819f0",
        "control": "5f177aff06654d8d656d67c0d8bb846330fe2b7869bd52ccbd0a4b7a6a45b0e1",
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
    BYPASS_RMS_NORM_CHECKPOINT = "bypass_rms_norm_checkpoint"
    WRONG_RMS_SCALE_CHECKPOINT = "wrong_rms_scale_checkpoint"
    CORRUPT_RMS_MEAN_SQUARE_CHECKPOINT = "corrupt_rms_mean_square_checkpoint"
    CORRUPT_RMS_INV_CHECKPOINT = "corrupt_rms_inv_checkpoint"
    CORRUPT_NORMALIZED_FLOAT32_CHECKPOINT = "corrupt_normalized_float32_checkpoint"
    CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT = "corrupt_normalized_bfloat16_checkpoint"
    WRONG_GATE_WEIGHT_CHECKPOINT = "wrong_gate_weight_checkpoint"
    CORRUPT_GATE_FLOAT32_CHECKPOINT = "corrupt_gate_float32_checkpoint"
    CORRUPT_GATE_BFLOAT16_CHECKPOINT = "corrupt_gate_bfloat16_checkpoint"
    WRONG_UP_WEIGHT_CHECKPOINT = "wrong_up_weight_checkpoint"
    CORRUPT_UP_FLOAT32_CHECKPOINT = "corrupt_up_float32_checkpoint"
    CORRUPT_UP_BFLOAT16_CHECKPOINT = "corrupt_up_bfloat16_checkpoint"
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
    RMS_NORM_ORACLE = "rms_norm_oracle"
    RMS_BFLOAT16_CONVERSION = "rms_bfloat16_conversion"
    GATE_PROJECTION_ORACLE = "gate_projection_oracle"
    GATE_BFLOAT16_CONVERSION = "gate_bfloat16_conversion"
    UP_PROJECTION_ORACLE = "up_projection_oracle"
    UP_BFLOAT16_CONVERSION = "up_bfloat16_conversion"
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
    SeqaxNumericalDiscriminator.BYPASS_RMS_NORM_CHECKPOINT: (
        SeqaxDiscriminatorClause.STRICT_HLO_STRUCTURE
    ),
    SeqaxNumericalDiscriminator.WRONG_RMS_SCALE_CHECKPOINT: (
        SeqaxDiscriminatorClause.RMS_NORM_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_RMS_MEAN_SQUARE_CHECKPOINT: (
        SeqaxDiscriminatorClause.RMS_NORM_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_RMS_INV_CHECKPOINT: (
        SeqaxDiscriminatorClause.RMS_NORM_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_FLOAT32_CHECKPOINT: (
        SeqaxDiscriminatorClause.RMS_NORM_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_NORMALIZED_BFLOAT16_CHECKPOINT: (
        SeqaxDiscriminatorClause.RMS_BFLOAT16_CONVERSION
    ),
    SeqaxNumericalDiscriminator.WRONG_GATE_WEIGHT_CHECKPOINT: (
        SeqaxDiscriminatorClause.GATE_PROJECTION_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_GATE_FLOAT32_CHECKPOINT: (
        SeqaxDiscriminatorClause.GATE_PROJECTION_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_GATE_BFLOAT16_CHECKPOINT: (
        SeqaxDiscriminatorClause.GATE_BFLOAT16_CONVERSION
    ),
    SeqaxNumericalDiscriminator.WRONG_UP_WEIGHT_CHECKPOINT: (
        SeqaxDiscriminatorClause.UP_PROJECTION_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_UP_FLOAT32_CHECKPOINT: (
        SeqaxDiscriminatorClause.UP_PROJECTION_ORACLE
    ),
    SeqaxNumericalDiscriminator.CORRUPT_UP_BFLOAT16_CHECKPOINT: (
        SeqaxDiscriminatorClause.UP_BFLOAT16_CONVERSION
    ),
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
    cpu_reference: str = "jax_cpu_reference_v1"
    cpu_reference_quantization_decimals: int = 6
    cpu_replay_rule: str = "cross_path_numerical_bounds"
    checkpoint_storage_dtype: str = "uint16"
    checkpoint_logical_dtype: str = "bfloat16"
    checkpoint_encoding: str = "bf16-bit-pattern-v1"
    require_float32_output: bool = True
    require_finite_output: bool = True
    mathematical_silu_max_ulp: int = Field(ge=0, le=1)
    rms_inverse_relative_error_units: float = Field(gt=0)

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
            self.cpu_replay_rule,
            self.checkpoint_storage_dtype,
            self.checkpoint_logical_dtype,
            self.checkpoint_encoding,
            self.require_float32_output,
            self.require_finite_output,
            self.mathematical_silu_max_ulp,
            self.rms_inverse_relative_error_units,
        ) != (
            3.0,
            8.0,
            2.0,
            2.0,
            "sqrt_layers",
            1.0,
            15,
            "jax_cpu_reference_v1",
            6,
            "cross_path_numerical_bounds",
            "uint16",
            "bfloat16",
            "bf16-bit-pattern-v1",
            True,
            True,
            1,
            4.0,
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
    tuple[SeqaxNumericalTensorContract, ...],
    tuple[SeqaxNumericalTensorContract, ...],
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
    rms_input_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_rms_input",
            (batch, sequence, model),
            "bfloat16",
        )
        for layer in range(layers)
    )
    rms_mean_square_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_rms_mean_square",
            (batch, sequence, 1),
            "float32",
        )
        for layer in range(layers)
    )
    rms_inverse_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_rms_inverse",
            (batch, sequence, 1),
            "float32",
        )
        for layer in range(layers)
    )
    normalized_float32_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_normalized_float32",
            (batch, sequence, model),
            "float32",
        )
        for layer in range(layers)
    )
    normalized_input_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_normalized_input",
            (batch, sequence, model),
            "bfloat16",
        )
        for layer in range(layers)
    )
    gate_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_gate",
            (batch, sequence, feed_forward),
            "bfloat16",
        )
        for layer in range(layers)
    )
    gate_float32_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_gate_float32",
            (batch, sequence, feed_forward),
            "float32",
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
    up_float32_checkpoints = tuple(
        tensor(
            f"layer_{layer:02d}_up_float32",
            (batch, sequence, feed_forward),
            "float32",
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
        rms_input_checkpoints,
        rms_mean_square_checkpoints,
        rms_inverse_checkpoints,
        normalized_float32_checkpoints,
        normalized_input_checkpoints,
        gate_float32_checkpoints,
        gate_checkpoints,
        silu_checkpoints,
        up_float32_checkpoints,
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
    rms_input_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    rms_mean_square_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    rms_inverse_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    normalized_float32_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    normalized_input_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    gate_float32_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    gate_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    silu_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
    up_float32_checkpoints: tuple[SeqaxNumericalTensorContract, ...] = Field(min_length=1)
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
            expected_rms_inputs,
            expected_rms_mean_square,
            expected_rms_inverse,
            expected_normalized_float32,
            expected_normalized_inputs,
            expected_gate_float32,
            expected_gates,
            expected_silu,
            expected_up_float32,
            expected_up,
            expected_hidden,
            expected_down_float32,
            expected_down_bfloat16,
        ) = _scenario_abi(self.parameters)
        if self.inputs != expected_inputs:
            raise ValueError("Seqax BF16 numerical scenario input ABI mismatch")
        if self.output != expected_output:
            raise ValueError("Seqax BF16 numerical scenario output ABI mismatch")
        if self.rms_input_checkpoints != expected_rms_inputs:
            raise ValueError("Seqax BF16 numerical scenario RMS-input checkpoint ABI mismatch")
        if self.rms_mean_square_checkpoints != expected_rms_mean_square:
            raise ValueError("Seqax BF16 numerical scenario RMS-statistic checkpoint ABI mismatch")
        if self.rms_inverse_checkpoints != expected_rms_inverse:
            raise ValueError("Seqax BF16 numerical scenario RMS-inverse checkpoint ABI mismatch")
        if self.normalized_float32_checkpoints != expected_normalized_float32:
            raise ValueError(
                "Seqax BF16 numerical scenario normalized-float32 checkpoint ABI mismatch"
            )
        if self.normalized_input_checkpoints != expected_normalized_inputs:
            raise ValueError(
                "Seqax BF16 numerical scenario normalized-input checkpoint ABI mismatch"
            )
        if self.gate_float32_checkpoints != expected_gate_float32:
            raise ValueError("Seqax BF16 numerical scenario float32 gate checkpoint ABI mismatch")
        if self.gate_checkpoints != expected_gates:
            raise ValueError("Seqax BF16 numerical scenario gate checkpoint ABI mismatch")
        if self.silu_checkpoints != expected_silu:
            raise ValueError("Seqax BF16 numerical scenario SiLU checkpoint ABI mismatch")
        if self.up_float32_checkpoints != expected_up_float32:
            raise ValueError("Seqax BF16 numerical scenario float32 up checkpoint ABI mismatch")
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
    checkpoint_capture: str = "typed-strict-rms-mlp-extra-outputs-v4"
    require_normal_output_policy: bool = True
    require_instrumented_output_policy: bool = True
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
            self.require_normal_output_policy,
            self.require_instrumented_output_policy,
            self.require_discriminator_artifact_replay,
        ) != (
            "authenticated-runner-and-relocated-public-replay",
            "typed-strict-rms-mlp-extra-outputs-v4",
            True,
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


class SeqaxBf16OutputAssessment(BaseModel):
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
    final_outputs_satisfy_policy: bool


class SeqaxBf16CpuReferenceReplayAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    saved_to_fresh_relative_l2: float = Field(ge=0)
    fresh_to_saved_relative_l2: float = Field(ge=0)
    saved_to_fresh_row_scaled_max: float = Field(ge=0)
    fresh_to_saved_row_scaled_max: float = Field(ge=0)
    relative_l2_limit: float = Field(gt=0)
    row_scaled_max_limit: float = Field(gt=0)
    top1_matches: bool
    within_bounds: bool


class SeqaxBf16NumericalAssessment(SeqaxBf16OutputAssessment):
    rms_input_cross_path_max_ulp: int = Field(ge=0)
    pallas_rms_mean_square_max_bound_ratio: float = Field(ge=0)
    control_rms_mean_square_max_bound_ratio: float = Field(ge=0)
    pallas_rms_mean_square_within_bound: bool
    control_rms_mean_square_within_bound: bool
    pallas_rms_inverse_relative_error_units: float = Field(ge=0)
    control_rms_inverse_relative_error_units: float = Field(ge=0)
    pallas_rms_inverse_within_bound: bool
    control_rms_inverse_within_bound: bool
    pallas_normalized_float32_max_bound_ratio: float = Field(ge=0)
    control_normalized_float32_max_bound_ratio: float = Field(ge=0)
    pallas_normalized_float32_within_bound: bool
    control_normalized_float32_within_bound: bool
    pallas_normalized_bfloat16_matches_float32: bool
    control_normalized_bfloat16_matches_float32: bool
    normalized_input_cross_path_max_ulp: int = Field(ge=0)
    pallas_gate_float32_max_bound_ratio: float = Field(ge=0)
    control_gate_float32_max_bound_ratio: float = Field(ge=0)
    pallas_gate_float32_within_bound: bool
    control_gate_float32_within_bound: bool
    pallas_gate_bfloat16_matches_float32: bool
    control_gate_bfloat16_matches_float32: bool
    gate_cross_path_max_ulp: int = Field(ge=0)
    pallas_silu_within_one_ulp_of_mathematical: bool
    control_silu_within_one_ulp_of_mathematical: bool
    silu_cross_path_max_ulp: int = Field(ge=0)
    pallas_up_float32_max_bound_ratio: float = Field(ge=0)
    control_up_float32_max_bound_ratio: float = Field(ge=0)
    pallas_up_float32_within_bound: bool
    control_up_float32_within_bound: bool
    pallas_up_bfloat16_matches_float32: bool
    control_up_bfloat16_matches_float32: bool
    up_bfloat16_cross_path_max_ulp: int = Field(ge=0)
    pallas_hidden_matches_product: bool
    control_hidden_matches_product: bool
    hidden_cross_path_max_ulp: int = Field(ge=0)
    pallas_down_float32_max_bound_ratio: float = Field(ge=0)
    control_down_float32_max_bound_ratio: float = Field(ge=0)
    pallas_down_float32_within_bound: bool
    control_down_float32_within_bound: bool
    pallas_down_bfloat16_matches_float32: bool
    control_down_bfloat16_matches_float32: bool
    down_bfloat16_cross_path_max_ulp: int = Field(ge=0)
    checkpoint_values_consistent: bool


class SeqaxBf16RuntimeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    python_major_minor: str = Field(pattern=r"^3\.12$")
    jax: str = Field(pattern=r"^0\.11\.0$")
    jaxlib: str = Field(pattern=r"^0\.11\.0$")
    libtpu: str = Field(pattern=r"^0\.0\.44\.1$")
    ml_dtypes: str = Field(pattern=r"^0\.6\.0$")
    cpu_machine: str = Field(pattern=r"^x86_64$")
    cpu_system: str = Field(pattern=r"^Linux$")
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
        calibration_rms_inputs,
        calibration_rms_mean_square,
        calibration_rms_inverse,
        calibration_normalized_float32,
        calibration_normalized_inputs,
        calibration_gate_float32,
        calibration_gates,
        calibration_silu,
        calibration_up_float32,
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
            rms_input_checkpoints=calibration_rms_inputs,
            rms_mean_square_checkpoints=calibration_rms_mean_square,
            rms_inverse_checkpoints=calibration_rms_inverse,
            normalized_float32_checkpoints=calibration_normalized_float32,
            normalized_input_checkpoints=calibration_normalized_inputs,
            gate_float32_checkpoints=calibration_gate_float32,
            gate_checkpoints=calibration_gates,
            silu_checkpoints=calibration_silu,
            up_float32_checkpoints=calibration_up_float32,
            up_checkpoints=calibration_up,
            hidden_checkpoints=calibration_hidden,
            down_float32_checkpoints=calibration_down_float32,
            down_bfloat16_checkpoints=calibration_down_bfloat16,
            **scenario_hlo("calibration-m256-b2-s1-l1"),
        )
    ]
    for name, raw_parameters in _CALIBRATION_SURFACE_PARAMETERS.items():
        parameters = SeqaxBf16ScenarioParameters(**raw_parameters)
        (
            inputs,
            output,
            rms_inputs,
            rms_mean_square,
            rms_inverse,
            normalized_float32,
            normalized_inputs,
            gate_float32,
            gates,
            silu,
            up_float32,
            up,
            hidden,
            down_float32,
            down_bfloat16,
        ) = _scenario_abi(parameters)
        scenarios.append(
            SeqaxBf16NumericalScenario(
                name=name,
                role=SeqaxNumericalScenarioRole.CALIBRATION,
                parameters=parameters,
                seeds=_CALIBRATION_SURFACE_SEEDS[name],
                inputs=inputs,
                output=output,
                rms_input_checkpoints=rms_inputs,
                rms_mean_square_checkpoints=rms_mean_square,
                rms_inverse_checkpoints=rms_inverse,
                normalized_float32_checkpoints=normalized_float32,
                normalized_input_checkpoints=normalized_inputs,
                gate_float32_checkpoints=gate_float32,
                gate_checkpoints=gates,
                silu_checkpoints=silu,
                up_float32_checkpoints=up_float32,
                up_checkpoints=up,
                hidden_checkpoints=hidden,
                down_float32_checkpoints=down_float32,
                down_bfloat16_checkpoints=down_bfloat16,
                **scenario_hlo(name),
            )
        )
    for name, raw_parameters in _V2_CALIBRATION_PARAMETERS.items():
        parameters = SeqaxBf16ScenarioParameters(**raw_parameters)
        (
            inputs,
            output,
            rms_inputs,
            rms_mean_square,
            rms_inverse,
            normalized_float32,
            normalized_inputs,
            gate_float32,
            gates,
            silu,
            up_float32,
            up,
            hidden,
            down_float32,
            down_bfloat16,
        ) = _scenario_abi(parameters)
        scenarios.append(
            SeqaxBf16NumericalScenario(
                name=name,
                role=SeqaxNumericalScenarioRole.CALIBRATION,
                parameters=parameters,
                seeds=_V2_CALIBRATION_SEEDS[name],
                inputs=inputs,
                output=output,
                rms_input_checkpoints=rms_inputs,
                rms_mean_square_checkpoints=rms_mean_square,
                rms_inverse_checkpoints=rms_inverse,
                normalized_float32_checkpoints=normalized_float32,
                normalized_input_checkpoints=normalized_inputs,
                gate_float32_checkpoints=gate_float32,
                gate_checkpoints=gates,
                silu_checkpoints=silu,
                up_float32_checkpoints=up_float32,
                up_checkpoints=up,
                hidden_checkpoints=hidden,
                down_float32_checkpoints=down_float32,
                down_bfloat16_checkpoints=down_bfloat16,
                **scenario_hlo(name),
            )
        )
    for name, raw_parameters in _HELD_OUT_PARAMETERS.items():
        parameters = SeqaxBf16ScenarioParameters(**raw_parameters)
        (
            inputs,
            output,
            rms_inputs,
            rms_mean_square,
            rms_inverse,
            normalized_float32,
            normalized_inputs,
            gate_float32,
            gates,
            silu,
            up_float32,
            up,
            hidden,
            down_float32,
            down_bfloat16,
        ) = _scenario_abi(parameters)
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
                rms_input_checkpoints=rms_inputs,
                rms_mean_square_checkpoints=rms_mean_square,
                rms_inverse_checkpoints=rms_inverse,
                normalized_float32_checkpoints=normalized_float32,
                normalized_input_checkpoints=normalized_inputs,
                gate_float32_checkpoints=gate_float32,
                gate_checkpoints=gates,
                silu_checkpoints=silu,
                up_float32_checkpoints=up_float32,
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
            mathematical_silu_max_ulp=1,
            rms_inverse_relative_error_units=4.0,
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
            cpu_machine="x86_64",
            cpu_system="Linux",
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


def _rms_mean_square_reference_components(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value64 = value.astype(np.float64)
    output_shape = (*value.shape[:-1], 1)
    reference = np.empty(output_shape, dtype=np.float64)
    absolute_sum = np.empty(output_shape, dtype=np.float64)
    for batch, sequence in np.ndindex(value.shape[:-1]):
        squares = tuple(float(element) * float(element) for element in value64[batch, sequence])
        total = math.fsum(squares)
        reference[batch, sequence, 0] = total / value.shape[-1]
        absolute_sum[batch, sequence, 0] = total
    return reference, absolute_sum


def _rms_mean_square_bound_ratio(actual: np.ndarray, value: np.ndarray) -> float:
    reference, absolute_sum = _rms_mean_square_reference_components(value)
    rounded_operations = 2 * value.shape[-1]
    float32_unit_roundoff = 2.0**-24
    gamma = (
        rounded_operations
        * float32_unit_roundoff
        / (1.0 - rounded_operations * float32_unit_roundoff)
    )
    bound = gamma * (absolute_sum / value.shape[-1]) + np.finfo(np.float32).tiny
    return float(np.max(np.abs(actual.astype(np.float64) - reference) / bound))


def _rms_inverse_relative_error_units(
    actual: np.ndarray,
    mean_square: np.ndarray,
    *,
    epsilon: float = 0.000001,
) -> float:
    reference = 1.0 / np.sqrt(mean_square.astype(np.float64) + epsilon)
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(actual.astype(np.float64) - reference) / scale) / (2.0**-24))


def _rms_normalized_bound_ratio(
    actual: np.ndarray,
    value: np.ndarray,
    inverse: np.ndarray,
    scale: np.ndarray,
) -> float:
    reference = (
        value.astype(np.float64)
        * inverse.astype(np.float64)
        * scale.astype(np.float64).reshape((1, 1, -1))
    )
    float32_unit_roundoff = 2.0**-24
    gamma = 2 * float32_unit_roundoff / (1.0 - 2 * float32_unit_roundoff)
    bound = gamma * np.abs(reference) + np.finfo(np.float32).tiny
    return float(np.max(np.abs(actual.astype(np.float64) - reference) / bound))


def _mlp_projection_reference_components(
    normalized_input: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized64 = normalized_input.astype(np.float64)
    weight64 = weight.astype(ml_dtypes.bfloat16).astype(np.float64)
    output_shape = (*normalized_input.shape[:-1], weight.shape[-1])
    reference = np.empty(output_shape, dtype=np.float64)
    absolute_sum = np.empty(output_shape, dtype=np.float64)
    for batch, sequence, feed_forward in np.ndindex(output_shape):
        products = tuple(
            float(normalized64[batch, sequence, model]) * float(weight64[model, feed_forward])
            for model in range(normalized_input.shape[-1])
        )
        reference[batch, sequence, feed_forward] = math.fsum(products)
        absolute_sum[batch, sequence, feed_forward] = math.fsum(abs(value) for value in products)
    return reference, absolute_sum


def _seqax_mlp_projection_reference_float32(
    normalized_input: np.ndarray,
    weight: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    normalized_array = np.asarray(normalized_input)
    weight_array = np.asarray(weight)
    if normalized_array.dtype != np.dtype(ml_dtypes.bfloat16):
        raise TypeError(f"Seqax {label} projection reference requires BF16 normalized inputs")
    if weight_array.dtype != np.float32:
        raise TypeError(f"Seqax {label} projection reference requires float32 weights")
    if normalized_array.ndim != 3 or weight_array.ndim != 2:
        raise ValueError(
            f"Seqax {label} projection reference requires rank-3 inputs and rank-2 weights"
        )
    if normalized_array.shape[-1] != weight_array.shape[0]:
        raise ValueError(f"Seqax {label} projection reference contraction shape mismatch")
    if not np.all(np.isfinite(normalized_array)) or not np.all(np.isfinite(weight_array)):
        raise ValueError(f"Seqax {label} projection reference requires finite inputs")
    reference, _absolute_sum = _mlp_projection_reference_components(normalized_array, weight_array)
    return reference.astype(np.float32)


def seqax_gate_projection_reference_float32(
    normalized_input: np.ndarray,
    gate_weight: np.ndarray,
) -> np.ndarray:
    return _seqax_mlp_projection_reference_float32(normalized_input, gate_weight, label="gate")


def seqax_up_projection_reference_float32(
    normalized_input: np.ndarray,
    up_weight: np.ndarray,
) -> np.ndarray:
    return _seqax_mlp_projection_reference_float32(normalized_input, up_weight, label="up")


def _mlp_projection_bound_ratio(
    actual: np.ndarray,
    normalized_input: np.ndarray,
    weight: np.ndarray,
) -> float:
    reference, absolute_sum = _mlp_projection_reference_components(normalized_input, weight)
    rounded_operations = 2 * normalized_input.shape[-1]
    float32_unit_roundoff = 2.0**-24
    gamma = (
        rounded_operations
        * float32_unit_roundoff
        / (1.0 - rounded_operations * float32_unit_roundoff)
    )
    bound = gamma * absolute_sum + np.finfo(np.float32).tiny
    ratio = np.abs(actual.astype(np.float64) - reference) / bound
    return float(np.max(ratio))


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
    return float(np.max(ratio))


def bf16_arrays_within_one_ulp(actual: np.ndarray, expected: np.ndarray) -> bool:
    bf16 = np.dtype(ml_dtypes.bfloat16)
    if actual.dtype != bf16 or expected.dtype != bf16 or actual.shape != expected.shape:
        return False
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        return False
    return _bf16_max_ulp_distance(actual, expected) <= 1


def _bf16_max_ulp_distance(actual: np.ndarray, expected: np.ndarray) -> int:
    def ordered(value: np.ndarray) -> np.ndarray:
        bits = value.view(np.uint16)
        return np.where(
            bits & np.uint16(0x8000),
            np.bitwise_not(bits),
            bits | np.uint16(0x8000),
        ).astype(np.int32)

    return int(np.max(np.abs(ordered(actual) - ordered(expected))))


def _assess_output_arrays(
    pallas: np.ndarray,
    control: np.ndarray,
    cpu_reference: np.ndarray,
    *,
    policy: SeqaxBf16NumericalPolicy,
    scenario: SeqaxBf16NumericalScenario,
) -> SeqaxBf16OutputAssessment:
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
    pallas_top1 = np.argmax(arrays[0], axis=-1)
    control_top1 = np.argmax(arrays[1], axis=-1)
    cpu_top1 = np.argmax(arrays[2], axis=-1)
    return SeqaxBf16OutputAssessment(
        cpu_pallas_relative_l2=pallas_relative,
        cpu_control_relative_l2=control_relative,
        cross_path_relative_l2=cross_relative,
        cpu_pallas_row_scaled_max=pallas_scaled,
        cpu_control_row_scaled_max=control_scaled,
        cross_path_row_scaled_max=cross_scaled,
        pallas_top1_matches_cpu=bool(np.array_equal(pallas_top1, cpu_top1)),
        control_top1_matches_cpu=bool(np.array_equal(control_top1, cpu_top1)),
        pallas_top1_matches_control=bool(np.array_equal(pallas_top1, control_top1)),
        final_outputs_satisfy_policy=final_outputs_satisfy_policy,
    )


def assess_seqax_cpu_reference_replay(
    saved: np.ndarray,
    fresh: np.ndarray,
    *,
    policy: SeqaxBf16NumericalPolicy,
    scenario: SeqaxBf16NumericalScenario,
) -> SeqaxBf16CpuReferenceReplayAssessment:
    if policy.cpu_replay_rule != "cross_path_numerical_bounds":
        raise ValueError("Seqax BF16 CPU replay rule is not supported")
    arrays = tuple(np.asarray(value) for value in (saved, fresh))
    if len({value.shape for value in arrays}) != 1 or arrays[0].shape != scenario.output.shape:
        raise ValueError("Seqax BF16 CPU reference shape does not match the contract")
    if policy.require_float32_output and any(value.dtype != np.float32 for value in arrays):
        raise TypeError("Seqax BF16 CPU references must use float32")
    if policy.require_finite_output and any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Seqax BF16 CPU references must be finite")
    saved_to_fresh_relative_l2 = _relative_l2(
        arrays[0],
        arrays[1],
        quantization_decimals=policy.metric_quantization_decimals,
    )
    fresh_to_saved_relative_l2 = _relative_l2(
        arrays[1],
        arrays[0],
        quantization_decimals=policy.metric_quantization_decimals,
    )
    saved_to_fresh_row_scaled_max = _row_scaled_max(
        arrays[0],
        arrays[1],
        scale_floor=policy.row_scale_floor,
        quantization_decimals=policy.metric_quantization_decimals,
    )
    fresh_to_saved_row_scaled_max = _row_scaled_max(
        arrays[1],
        arrays[0],
        scale_floor=policy.row_scale_floor,
        quantization_decimals=policy.metric_quantization_decimals,
    )
    depth_scale = policy.depth_scale(scenario.parameters.layers)
    relative_l2_limit = policy.cross_path_relative_l2_units * policy.unit_roundoff * depth_scale
    row_scaled_max_limit = (
        policy.cross_path_row_scaled_max_units * policy.unit_roundoff * depth_scale
    )
    return SeqaxBf16CpuReferenceReplayAssessment(
        saved_to_fresh_relative_l2=saved_to_fresh_relative_l2,
        fresh_to_saved_relative_l2=fresh_to_saved_relative_l2,
        saved_to_fresh_row_scaled_max=saved_to_fresh_row_scaled_max,
        fresh_to_saved_row_scaled_max=fresh_to_saved_row_scaled_max,
        relative_l2_limit=relative_l2_limit,
        row_scaled_max_limit=row_scaled_max_limit,
        top1_matches=bool(
            np.array_equal(np.argmax(arrays[0], axis=-1), np.argmax(arrays[1], axis=-1))
        ),
        within_bounds=(
            max(saved_to_fresh_relative_l2, fresh_to_saved_relative_l2) <= relative_l2_limit
            and max(saved_to_fresh_row_scaled_max, fresh_to_saved_row_scaled_max)
            <= row_scaled_max_limit
        ),
    )


def _assess_seqax_bf16_outputs(
    pallas: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    inputs: tuple[np.ndarray, ...],
    policy: SeqaxBf16NumericalPolicy,
    scenario: SeqaxBf16NumericalScenario,
) -> tuple[SeqaxBf16OutputAssessment, tuple[np.ndarray, ...]]:
    if type(seed) is not int or seed not in scenario.seeds:
        raise ValueError("Seqax BF16 numerical seed is not declared by the scenario")
    validate_seqax_numerical_inputs(inputs, scenario)
    expected_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=seed,
            **scenario.parameters.model_dump(),
        )
    )
    for actual, expected, contract in zip(inputs, expected_inputs, scenario.inputs, strict=True):
        if not np.array_equal(actual, expected):
            raise ValueError(f"Seqax numerical deterministic input mismatch: {contract.name}")
    cpu_reference = seqax_forward_canonical_reference(
        expected_inputs,
        quantization_decimals=policy.cpu_reference_quantization_decimals,
        **scenario.parameters.model_dump(),
    )
    return (
        _assess_output_arrays(
            pallas,
            control,
            cpu_reference,
            policy=policy,
            scenario=scenario,
        ),
        expected_inputs,
    )


def assess_seqax_bf16_outputs(
    pallas: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    inputs: tuple[np.ndarray, ...],
    policy: SeqaxBf16NumericalPolicy,
    scenario: SeqaxBf16NumericalScenario,
) -> SeqaxBf16OutputAssessment:
    assessment, _expected_inputs = _assess_seqax_bf16_outputs(
        pallas,
        control,
        seed=seed,
        inputs=inputs,
        policy=policy,
        scenario=scenario,
    )
    return assessment


def assess_seqax_bf16_forward(
    pallas: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    inputs: tuple[np.ndarray, ...],
    pallas_rms_input_checkpoints: tuple[np.ndarray, ...],
    control_rms_input_checkpoints: tuple[np.ndarray, ...],
    pallas_rms_mean_square_checkpoints: tuple[np.ndarray, ...],
    control_rms_mean_square_checkpoints: tuple[np.ndarray, ...],
    pallas_rms_inverse_checkpoints: tuple[np.ndarray, ...],
    control_rms_inverse_checkpoints: tuple[np.ndarray, ...],
    pallas_normalized_float32_checkpoints: tuple[np.ndarray, ...],
    control_normalized_float32_checkpoints: tuple[np.ndarray, ...],
    pallas_normalized_input_checkpoints: tuple[np.ndarray, ...],
    control_normalized_input_checkpoints: tuple[np.ndarray, ...],
    pallas_gate_float32_checkpoints: tuple[np.ndarray, ...],
    control_gate_float32_checkpoints: tuple[np.ndarray, ...],
    pallas_gate_checkpoints: tuple[np.ndarray, ...],
    control_gate_checkpoints: tuple[np.ndarray, ...],
    pallas_silu_checkpoints: tuple[np.ndarray, ...],
    control_silu_checkpoints: tuple[np.ndarray, ...],
    pallas_up_float32_checkpoints: tuple[np.ndarray, ...],
    control_up_float32_checkpoints: tuple[np.ndarray, ...],
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
    output_assessment, expected_inputs = _assess_seqax_bf16_outputs(
        pallas,
        control,
        seed=seed,
        inputs=inputs,
        policy=policy,
        scenario=scenario,
    )
    pallas_rms_inputs = _validate_bf16_checkpoints(
        pallas_rms_input_checkpoints,
        scenario.rms_input_checkpoints,
        label="RMS input",
    )
    control_rms_inputs = _validate_bf16_checkpoints(
        control_rms_input_checkpoints,
        scenario.rms_input_checkpoints,
        label="RMS input",
    )
    pallas_rms_mean_square = _validate_float32_checkpoints(
        pallas_rms_mean_square_checkpoints,
        scenario.rms_mean_square_checkpoints,
        label="RMS mean square",
    )
    control_rms_mean_square = _validate_float32_checkpoints(
        control_rms_mean_square_checkpoints,
        scenario.rms_mean_square_checkpoints,
        label="RMS mean square",
    )
    pallas_rms_inverse = _validate_float32_checkpoints(
        pallas_rms_inverse_checkpoints,
        scenario.rms_inverse_checkpoints,
        label="RMS inverse",
    )
    control_rms_inverse = _validate_float32_checkpoints(
        control_rms_inverse_checkpoints,
        scenario.rms_inverse_checkpoints,
        label="RMS inverse",
    )
    pallas_normalized_float32 = _validate_float32_checkpoints(
        pallas_normalized_float32_checkpoints,
        scenario.normalized_float32_checkpoints,
        label="normalized float32",
    )
    control_normalized_float32 = _validate_float32_checkpoints(
        control_normalized_float32_checkpoints,
        scenario.normalized_float32_checkpoints,
        label="normalized float32",
    )
    pallas_normalized_inputs = _validate_bf16_checkpoints(
        pallas_normalized_input_checkpoints,
        scenario.normalized_input_checkpoints,
        label="normalized input",
    )
    control_normalized_inputs = _validate_bf16_checkpoints(
        control_normalized_input_checkpoints,
        scenario.normalized_input_checkpoints,
        label="normalized input",
    )
    pallas_gate_float32 = _validate_float32_checkpoints(
        pallas_gate_float32_checkpoints,
        scenario.gate_float32_checkpoints,
        label="float32 gate",
    )
    control_gate_float32 = _validate_float32_checkpoints(
        control_gate_float32_checkpoints,
        scenario.gate_float32_checkpoints,
        label="float32 gate",
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
    pallas_up_float32 = _validate_float32_checkpoints(
        pallas_up_float32_checkpoints,
        scenario.up_float32_checkpoints,
        label="float32 up",
    )
    control_up_float32 = _validate_float32_checkpoints(
        control_up_float32_checkpoints,
        scenario.up_float32_checkpoints,
        label="float32 up",
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
    rms_input_cross_path_max_ulp = max(
        _bf16_max_ulp_distance(pallas_value, control_value)
        for pallas_value, control_value in zip(pallas_rms_inputs, control_rms_inputs, strict=True)
    )
    rms_mean_square_pallas_ratios = tuple(
        _rms_mean_square_bound_ratio(actual, value)
        for actual, value in zip(pallas_rms_mean_square, pallas_rms_inputs, strict=True)
    )
    rms_mean_square_control_ratios = tuple(
        _rms_mean_square_bound_ratio(actual, value)
        for actual, value in zip(control_rms_mean_square, control_rms_inputs, strict=True)
    )
    pallas_rms_mean_square_ratio = max(rms_mean_square_pallas_ratios)
    control_rms_mean_square_ratio = max(rms_mean_square_control_ratios)
    pallas_rms_inverse_units = max(
        _rms_inverse_relative_error_units(actual, mean_square)
        for actual, mean_square in zip(pallas_rms_inverse, pallas_rms_mean_square, strict=True)
    )
    control_rms_inverse_units = max(
        _rms_inverse_relative_error_units(actual, mean_square)
        for actual, mean_square in zip(control_rms_inverse, control_rms_mean_square, strict=True)
    )
    rms_scales = expected_inputs[4]
    pallas_normalized_ratios = tuple(
        _rms_normalized_bound_ratio(actual, value, inverse, rms_scales[layer])
        for layer, (actual, value, inverse) in enumerate(
            zip(
                pallas_normalized_float32,
                pallas_rms_inputs,
                pallas_rms_inverse,
                strict=True,
            )
        )
    )
    control_normalized_ratios = tuple(
        _rms_normalized_bound_ratio(actual, value, inverse, rms_scales[layer])
        for layer, (actual, value, inverse) in enumerate(
            zip(
                control_normalized_float32,
                control_rms_inputs,
                control_rms_inverse,
                strict=True,
            )
        )
    )
    pallas_normalized_ratio = max(pallas_normalized_ratios)
    control_normalized_ratio = max(control_normalized_ratios)
    pallas_normalized_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(
            pallas_normalized_inputs, pallas_normalized_float32, strict=True
        )
    )
    control_normalized_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(
            control_normalized_inputs, control_normalized_float32, strict=True
        )
    )
    normalized_input_cross_path_max_ulp = max(
        _bf16_max_ulp_distance(pallas_value, control_value)
        for pallas_value, control_value in zip(
            pallas_normalized_inputs, control_normalized_inputs, strict=True
        )
    )
    gate_weights = expected_inputs[8]
    pallas_gate_ratios = tuple(
        _mlp_projection_bound_ratio(actual, normalized, gate_weights[layer])
        for layer, (actual, normalized) in enumerate(
            zip(pallas_gate_float32, pallas_normalized_inputs, strict=True)
        )
    )
    control_gate_ratios = tuple(
        _mlp_projection_bound_ratio(actual, normalized, gate_weights[layer])
        for layer, (actual, normalized) in enumerate(
            zip(control_gate_float32, control_normalized_inputs, strict=True)
        )
    )
    pallas_gate_ratio = max(pallas_gate_ratios)
    control_gate_ratio = max(control_gate_ratios)
    pallas_gate_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(pallas_gates, pallas_gate_float32, strict=True)
    )
    control_gate_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(control_gates, control_gate_float32, strict=True)
    )
    gate_cross_path_max_ulp = max(
        _bf16_max_ulp_distance(pallas_value, control_value)
        for pallas_value, control_value in zip(pallas_gates, control_gates, strict=True)
    )
    pallas_silu_mathematical = all(
        bf16_arrays_within_one_ulp(actual, expected)
        for actual, expected in zip(pallas_silu, pallas_mathematical, strict=True)
    )
    control_silu_mathematical = all(
        bf16_arrays_within_one_ulp(actual, expected)
        for actual, expected in zip(control_silu, control_mathematical, strict=True)
    )
    silu_cross_path_max_ulp = max(
        _bf16_max_ulp_distance(pallas_value, control_value)
        for pallas_value, control_value in zip(pallas_silu, control_silu, strict=True)
    )
    up_weights = expected_inputs[9]
    pallas_up_ratios = tuple(
        _mlp_projection_bound_ratio(actual, normalized, up_weights[layer])
        for layer, (actual, normalized) in enumerate(
            zip(pallas_up_float32, pallas_normalized_inputs, strict=True)
        )
    )
    control_up_ratios = tuple(
        _mlp_projection_bound_ratio(actual, normalized, up_weights[layer])
        for layer, (actual, normalized) in enumerate(
            zip(control_up_float32, control_normalized_inputs, strict=True)
        )
    )
    pallas_up_ratio = max(pallas_up_ratios)
    control_up_ratio = max(control_up_ratios)
    pallas_up_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(pallas_up, pallas_up_float32, strict=True)
    )
    control_up_bfloat16_matches = all(
        np.array_equal(actual, expected.astype(ml_dtypes.bfloat16))
        for actual, expected in zip(control_up, control_up_float32, strict=True)
    )
    up_cross_path_max_ulp = max(
        _bf16_max_ulp_distance(pallas_value, control_value)
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
    hidden_cross_path_max_ulp = max(
        _bf16_max_ulp_distance(pallas_value, control_value)
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
    down_bfloat16_cross_path_max_ulp = max(
        _bf16_max_ulp_distance(pallas_value, control_value)
        for pallas_value, control_value in zip(
            pallas_down_bfloat16, control_down_bfloat16, strict=True
        )
    )
    checkpoint_values_consistent = (
        pallas_rms_mean_square_ratio <= 1.0
        and control_rms_mean_square_ratio <= 1.0
        and pallas_rms_inverse_units <= policy.rms_inverse_relative_error_units
        and control_rms_inverse_units <= policy.rms_inverse_relative_error_units
        and pallas_normalized_ratio <= 1.0
        and control_normalized_ratio <= 1.0
        and pallas_normalized_bfloat16_matches
        and control_normalized_bfloat16_matches
        and pallas_gate_ratio <= 1.0
        and control_gate_ratio <= 1.0
        and pallas_gate_bfloat16_matches
        and control_gate_bfloat16_matches
        and pallas_silu_mathematical
        and control_silu_mathematical
        and pallas_up_ratio <= 1.0
        and control_up_ratio <= 1.0
        and pallas_up_bfloat16_matches
        and control_up_bfloat16_matches
        and pallas_hidden_mathematical
        and control_hidden_mathematical
        and pallas_down_ratio <= 1.0
        and control_down_ratio <= 1.0
        and pallas_down_bfloat16_matches
        and control_down_bfloat16_matches
    )
    return SeqaxBf16NumericalAssessment(
        **output_assessment.model_dump(),
        rms_input_cross_path_max_ulp=rms_input_cross_path_max_ulp,
        pallas_rms_mean_square_max_bound_ratio=round(
            pallas_rms_mean_square_ratio, policy.metric_quantization_decimals
        ),
        control_rms_mean_square_max_bound_ratio=round(
            control_rms_mean_square_ratio, policy.metric_quantization_decimals
        ),
        pallas_rms_mean_square_within_bound=pallas_rms_mean_square_ratio <= 1.0,
        control_rms_mean_square_within_bound=control_rms_mean_square_ratio <= 1.0,
        pallas_rms_inverse_relative_error_units=round(
            pallas_rms_inverse_units, policy.metric_quantization_decimals
        ),
        control_rms_inverse_relative_error_units=round(
            control_rms_inverse_units, policy.metric_quantization_decimals
        ),
        pallas_rms_inverse_within_bound=(
            pallas_rms_inverse_units <= policy.rms_inverse_relative_error_units
        ),
        control_rms_inverse_within_bound=(
            control_rms_inverse_units <= policy.rms_inverse_relative_error_units
        ),
        pallas_normalized_float32_max_bound_ratio=round(
            pallas_normalized_ratio, policy.metric_quantization_decimals
        ),
        control_normalized_float32_max_bound_ratio=round(
            control_normalized_ratio, policy.metric_quantization_decimals
        ),
        pallas_normalized_float32_within_bound=pallas_normalized_ratio <= 1.0,
        control_normalized_float32_within_bound=control_normalized_ratio <= 1.0,
        pallas_normalized_bfloat16_matches_float32=pallas_normalized_bfloat16_matches,
        control_normalized_bfloat16_matches_float32=control_normalized_bfloat16_matches,
        normalized_input_cross_path_max_ulp=normalized_input_cross_path_max_ulp,
        pallas_gate_float32_max_bound_ratio=round(
            pallas_gate_ratio, policy.metric_quantization_decimals
        ),
        control_gate_float32_max_bound_ratio=round(
            control_gate_ratio, policy.metric_quantization_decimals
        ),
        pallas_gate_float32_within_bound=pallas_gate_ratio <= 1.0,
        control_gate_float32_within_bound=control_gate_ratio <= 1.0,
        pallas_gate_bfloat16_matches_float32=pallas_gate_bfloat16_matches,
        control_gate_bfloat16_matches_float32=control_gate_bfloat16_matches,
        gate_cross_path_max_ulp=gate_cross_path_max_ulp,
        pallas_silu_within_one_ulp_of_mathematical=pallas_silu_mathematical,
        control_silu_within_one_ulp_of_mathematical=control_silu_mathematical,
        silu_cross_path_max_ulp=silu_cross_path_max_ulp,
        pallas_up_float32_max_bound_ratio=round(
            pallas_up_ratio, policy.metric_quantization_decimals
        ),
        control_up_float32_max_bound_ratio=round(
            control_up_ratio, policy.metric_quantization_decimals
        ),
        pallas_up_float32_within_bound=pallas_up_ratio <= 1.0,
        control_up_float32_within_bound=control_up_ratio <= 1.0,
        pallas_up_bfloat16_matches_float32=pallas_up_bfloat16_matches,
        control_up_bfloat16_matches_float32=control_up_bfloat16_matches,
        up_bfloat16_cross_path_max_ulp=up_cross_path_max_ulp,
        pallas_hidden_matches_product=pallas_hidden_mathematical,
        control_hidden_matches_product=control_hidden_mathematical,
        hidden_cross_path_max_ulp=hidden_cross_path_max_ulp,
        pallas_down_float32_max_bound_ratio=round(
            pallas_down_ratio, policy.metric_quantization_decimals
        ),
        control_down_float32_max_bound_ratio=round(
            control_down_ratio, policy.metric_quantization_decimals
        ),
        pallas_down_float32_within_bound=pallas_down_ratio <= 1.0,
        control_down_float32_within_bound=control_down_ratio <= 1.0,
        pallas_down_bfloat16_matches_float32=pallas_down_bfloat16_matches,
        control_down_bfloat16_matches_float32=control_down_bfloat16_matches,
        down_bfloat16_cross_path_max_ulp=down_bfloat16_cross_path_max_ulp,
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


def _follow_shape_only_producer(value: ir.Value) -> ir.Value:
    current = value
    while True:
        operation = _as_operation(current.owner)
        if operation is None or operation.name != "stablehlo.reshape":
            return current
        _require_attribute_names(operation, frozenset())
        if len(operation.operands) != 1 or len(operation.results) != 1:
            raise ValueError("strict MLP reverse reshape path is invalid")
        current = operation.operands[0]


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


def _require_f32_scalar_constant(value: ir.Value, expected: float, *, label: str) -> None:
    operation = _as_operation(value.owner)
    if operation is None or operation.name != "stablehlo.constant":
        raise ValueError(f"strict RMSNorm {label} must be a float32 scalar constant")
    _require_attribute_names(operation, frozenset({"value"}))
    if (
        len(operation.results) != 1
        or operation.results[0] != value
        or not _is_f32_tensor(value)
        or ir.RankedTensorType(value.type).rank != 0
    ):
        raise ValueError(f"strict RMSNorm {label} must be a float32 scalar constant")
    dense = ir.DenseElementsAttr(operation.attributes["value"])
    if not dense.is_splat or float(ir.FloatAttr(dense.get_splat_value()).value) != float(
        np.float32(expected)
    ):
        raise ValueError(f"strict RMSNorm {label} has an invalid value")


def _require_broadcast_dimensions(
    operation: ir.Operation,
    expected: tuple[int, ...],
    *,
    label: str,
) -> None:
    _require_attribute_names(operation, frozenset({"broadcast_dimensions"}))
    encoded = "array<i64>" if not expected else f"array<i64: {', '.join(map(str, expected))}>"
    if str(operation.attributes["broadcast_dimensions"]) != encoded:
        raise ValueError(f"strict RMSNorm {label} has invalid broadcast dimensions")


def _require_rms_scale_source(value: ir.Value, *, manual_argument_number: int) -> None:
    current = value
    while True:
        operation = _as_operation(current.owner)
        if operation is None:
            break
        if operation.name == "stablehlo.reshape":
            _require_attribute_names(operation, frozenset())
            if len(operation.operands) != 1:
                raise ValueError("strict RMSNorm scale reshape is invalid")
            current = operation.operands[0]
        elif operation.name == "stablehlo.broadcast_in_dim":
            if len(operation.operands) != 1:
                raise ValueError("strict RMSNorm scale broadcast is invalid")
            result_rank = ir.RankedTensorType(operation.results[0].type).rank
            input_rank = ir.RankedTensorType(operation.operands[0].type).rank
            expected = tuple(range(result_rank - input_rank, result_rank))
            _require_broadcast_dimensions(operation, expected, label="scale")
            current = operation.operands[0]
        elif operation.name == "stablehlo.all_gather":
            if len(operation.operands) != 1:
                raise ValueError("strict RMSNorm scale all-gather is invalid")
            current = operation.operands[0]
        elif operation.name == "func.call":
            if len(operation.operands) < 1 or not str(
                operation.attributes.get("callee", "")
            ).startswith("@_take"):
                raise ValueError("strict RMSNorm scale must come from layer_norm_2")
            current = operation.operands[0]
        else:
            raise ValueError("strict RMSNorm scale must come from layer_norm_2")

    block = current.owner
    owner = _as_operation(getattr(block, "owner", None))
    argument_number = getattr(current, "arg_number", None)
    if owner is None or argument_number is None:
        raise ValueError("strict RMSNorm scale must come from layer_norm_2")
    if owner.name == "sdy.manual_computation":
        expected_argument = manual_argument_number
    elif owner.name == "func.func":
        expected_argument = 1
    else:
        raise ValueError("strict RMSNorm scale must come from layer_norm_2")
    if argument_number != expected_argument:
        raise ValueError("strict RMSNorm scale must come from layer_norm_2")


def _validate_rms_reducer(operation: ir.Operation) -> None:
    if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
        raise ValueError("strict RMSNorm reduction must have one addition body")
    block = operation.regions[0].blocks[0]
    if len(block.arguments) != 2 or not all(_is_f32_tensor(arg) for arg in block.arguments):
        raise ValueError("strict RMSNorm reduction must have float32 scalar arguments")
    if any(ir.RankedTensorType(arg.type).rank != 0 for arg in block.arguments):
        raise ValueError("strict RMSNorm reduction must have float32 scalar arguments")
    operations = tuple(child.operation for child in block.operations)
    if tuple(child.name for child in operations) != ("stablehlo.add", "stablehlo.return"):
        raise ValueError("strict RMSNorm reduction must use exact float32 addition")
    add, result = operations
    _require_attribute_names(add, frozenset())
    _require_operands(add, tuple(block.arguments))
    added = _require_single_result(add, block.arguments[0].type)
    _require_attribute_names(result, frozenset())
    _require_operands(result, (added,))


def _validate_rmsnorm_dataflow(
    normalized_float32: ir.Value,
    normalized_multiply: ir.Operation,
    *,
    manual_scale_argument: int,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    normalized_type = ir.RankedTensorType(normalized_float32.type)
    normalized_shape = tuple(normalized_type.shape)
    rank = normalized_type.rank
    if rank < 1:
        raise ValueError("strict RMSNorm normalized value must be ranked")

    scaled_candidates = tuple(
        operand
        for operand in normalized_multiply.operands
        if _as_operation(operand.owner) is not None
        and _as_operation(operand.owner).name == "stablehlo.multiply"
    )
    if len(scaled_candidates) != 1:
        raise ValueError("strict RMSNorm must have one input-times-inverse product")
    scaled_input = scaled_candidates[0]
    scale_candidates = tuple(
        operand for operand in normalized_multiply.operands if operand != scaled_input
    )
    if len(scale_candidates) != 1 or not _is_f32_tensor(scale_candidates[0]):
        raise ValueError("strict RMSNorm normalized value must use layer_norm_2 scale")
    _require_rms_scale_source(scale_candidates[0], manual_argument_number=manual_scale_argument)

    scaled_multiply = _as_operation(scaled_input.owner)
    assert scaled_multiply is not None
    _require_attribute_names(scaled_multiply, frozenset())
    rms_input_converts = tuple(
        operand
        for operand in scaled_multiply.operands
        if _as_operation(operand.owner) is not None
        and _as_operation(operand.owner).name == "stablehlo.convert"
        and len(_as_operation(operand.owner).operands) == 1
        and _is_bf16_tensor(_as_operation(operand.owner).operands[0])
    )
    inverse_broadcasts = tuple(
        operand
        for operand in scaled_multiply.operands
        if _as_operation(operand.owner) is not None
        and _as_operation(operand.owner).name == "stablehlo.broadcast_in_dim"
    )
    if len(rms_input_converts) != 1 or len(inverse_broadcasts) != 1:
        raise ValueError("strict RMSNorm must multiply its BF16 input by its inverse RMS")
    rms_input_convert = _as_operation(rms_input_converts[0].owner)
    inverse_broadcast = _as_operation(inverse_broadcasts[0].owner)
    assert rms_input_convert is not None and inverse_broadcast is not None
    _require_attribute_names(rms_input_convert, frozenset())
    rms_input = rms_input_convert.operands[0]
    if tuple(ir.RankedTensorType(rms_input.type).shape) != normalized_shape:
        raise ValueError("strict RMSNorm input and normalized shapes must match")

    square_uses: list[ir.Operation] = []
    for use in rms_input.uses:
        conversion = _as_operation(use.owner)
        if (
            conversion is None
            or conversion.name != "stablehlo.convert"
            or len(conversion.operands) != 1
            or len(conversion.results) != 1
            or conversion.operands[0] != rms_input
        ):
            continue
        _require_attribute_names(conversion, frozenset())
        square_uses.extend(
            square
            for result_use in conversion.results[0].uses
            if (square := _as_operation(result_use.owner)) is not None
            and square.name == "chlo.square"
        )
    if len(square_uses) != 1:
        raise ValueError("strict RMSNorm must square the same BF16 input exactly once")
    square = square_uses[0]
    _require_attribute_names(square, frozenset())
    if len(square.operands) != 1 or len(square.results) != 1:
        raise ValueError("strict RMSNorm square is invalid")
    square_input = _as_operation(square.operands[0].owner)
    if (
        square_input is None
        or square_input.name != "stablehlo.convert"
        or tuple(square_input.operands) != (rms_input,)
        or square.results[0].type != normalized_float32.type
    ):
        raise ValueError("strict RMSNorm square must consume the same BF16 input")
    square_result_uses = tuple(square.results[0].uses)
    if len(square_result_uses) != 1:
        raise ValueError("strict RMSNorm square must feed one reduction")
    reduction = _as_operation(square_result_uses[0].owner)
    if reduction is None or reduction.name != "stablehlo.reduce":
        raise ValueError("strict RMSNorm square must feed one reduction")
    _require_attribute_names(reduction, frozenset({"dimensions"}))
    if str(reduction.attributes["dimensions"]) != f"array<i64: {rank - 1}>":
        raise ValueError("strict RMSNorm reduction must use the model axis")
    if len(reduction.operands) != 2 or reduction.operands[0] != square.results[0]:
        raise ValueError("strict RMSNorm reduction operands are invalid")
    _require_f32_scalar_constant(reduction.operands[1], 0.0, label="reduction zero")
    _validate_rms_reducer(reduction)
    if (
        len(reduction.results) != 1
        or tuple(ir.RankedTensorType(reduction.results[0].type).shape) != normalized_shape[:-1]
    ):
        raise ValueError("strict RMSNorm reduction has an invalid result shape")

    reduction_uses = tuple(reduction.results[0].uses)
    if len(reduction_uses) != 1:
        raise ValueError("strict RMSNorm sum must feed one singleton broadcast")
    sum_broadcast = _as_operation(reduction_uses[0].owner)
    if sum_broadcast is None or sum_broadcast.name != "stablehlo.broadcast_in_dim":
        raise ValueError("strict RMSNorm sum must feed one singleton broadcast")
    _require_broadcast_dimensions(sum_broadcast, tuple(range(rank - 1)), label="sum")
    if (
        len(sum_broadcast.operands) != 1
        or len(sum_broadcast.results) != 1
        or sum_broadcast.operands[0] != reduction.results[0]
        or tuple(ir.RankedTensorType(sum_broadcast.results[0].type).shape)
        != (*normalized_shape[:-1], 1)
    ):
        raise ValueError("strict RMSNorm sum broadcast has an invalid shape")

    if len(inverse_broadcast.operands) != 1:
        raise ValueError("strict RMSNorm inverse broadcast is invalid")
    _require_broadcast_dimensions(inverse_broadcast, tuple(range(rank)), label="inverse")
    rms_inverse = inverse_broadcast.operands[0]
    inverse_operation = _as_operation(rms_inverse.owner)
    if (
        inverse_operation is None
        or inverse_operation.name != "stablehlo.rsqrt"
        or len(inverse_operation.operands) != 1
        or len(inverse_operation.results) != 1
        or inverse_operation.results[0] != rms_inverse
    ):
        raise ValueError("strict RMSNorm inverse checkpoint must come from rsqrt")
    _require_attribute_names(inverse_operation, frozenset())
    epsilon_add = _as_operation(inverse_operation.operands[0].owner)
    if epsilon_add is None or epsilon_add.name != "stablehlo.add":
        raise ValueError("strict RMSNorm rsqrt must consume mean-square plus epsilon")
    _require_attribute_names(epsilon_add, frozenset())
    if len(epsilon_add.operands) != 2:
        raise ValueError("strict RMSNorm rsqrt must consume mean-square plus epsilon")

    mean_square_candidates = tuple(
        operand
        for operand in epsilon_add.operands
        if _as_operation(operand.owner) is not None
        and _as_operation(operand.owner).name == "stablehlo.divide"
    )
    epsilon_candidates = tuple(
        operand
        for operand in epsilon_add.operands
        if _as_operation(operand.owner) is not None
        and _as_operation(operand.owner).name == "stablehlo.broadcast_in_dim"
    )
    if len(mean_square_candidates) != 1 or len(epsilon_candidates) != 1:
        raise ValueError("strict RMSNorm mean-square and epsilon dataflow is invalid")
    rms_mean_square = mean_square_candidates[0]
    mean_square_operation = _as_operation(rms_mean_square.owner)
    epsilon_broadcast = _as_operation(epsilon_candidates[0].owner)
    assert mean_square_operation is not None and epsilon_broadcast is not None
    _require_attribute_names(mean_square_operation, frozenset())
    if (
        len(mean_square_operation.operands) != 2
        or mean_square_operation.operands[0] != sum_broadcast.results[0]
    ):
        raise ValueError("strict RMSNorm mean-square must divide the exact squared sum")
    divisor_broadcast = _as_operation(mean_square_operation.operands[1].owner)
    if (
        divisor_broadcast is None
        or divisor_broadcast.name != "stablehlo.broadcast_in_dim"
        or len(divisor_broadcast.operands) != 1
    ):
        raise ValueError("strict RMSNorm divisor broadcast is invalid")
    _require_broadcast_dimensions(divisor_broadcast, (), label="divisor")
    _require_f32_scalar_constant(
        divisor_broadcast.operands[0], normalized_shape[-1], label="model divisor"
    )
    _require_broadcast_dimensions(epsilon_broadcast, (), label="epsilon")
    if len(epsilon_broadcast.operands) != 1:
        raise ValueError("strict RMSNorm epsilon broadcast is invalid")
    _require_f32_scalar_constant(epsilon_broadcast.operands[0], 1.0e-6, label="epsilon")
    return rms_input, rms_mean_square, rms_inverse


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
                or len(entry_returns[0].operands) != leading_result_count + 1 + 13 * expected_count
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
                gate_projection: ir.Operation | None = None
                gate_is_interpret_callback = False
                if require_hidden_down:
                    if len(input_barrier.operands) != 1 or not _is_bf16_tensor(
                        input_barrier.operands[0]
                    ):
                        raise ValueError("strict SiLU input barrier must consume a BF16 gate")
                    converted_gate = input_barrier.operands[0]
                    if tuple(_as_operation(use.owner) for use in converted_gate.uses) != (
                        input_barrier,
                    ):
                        raise ValueError("strict MLP converted gate must feed only its barrier")
                    gate_convert = _as_operation(converted_gate.owner)
                    if (
                        gate_convert is None
                        or gate_convert.name != "stablehlo.convert"
                        or len(gate_convert.operands) != 1
                        or len(gate_convert.results) != 1
                        or gate_convert.results[0] != converted_gate
                        or not _is_f32_tensor(gate_convert.operands[0])
                    ):
                        raise ValueError("strict MLP gate must come from one float32 conversion")
                    _require_attribute_names(gate_convert, frozenset())
                    gate_float32 = gate_convert.operands[0]
                    gate_float32_uses = tuple(gate_float32.uses)
                    gate_float32_checkpoints = tuple(
                        use
                        for use in gate_float32_uses
                        if _as_operation(use.owner) is not None
                        and _as_operation(use.owner).name in _REGION_TERMINATORS
                    )
                    if instrumented:
                        gate_float32_position = leading_result_count + 6 + 13 * layer
                        if (
                            len(gate_float32_uses) != 2
                            or len(gate_float32_checkpoints) != 1
                            or gate_float32_checkpoints[0].operand_number != gate_float32_position
                        ):
                            raise ValueError(
                                "instrumented strict MLP must return the real float32 gate checkpoint"
                            )
                        _require_checkpoint_function_result(
                            gate_float32_checkpoints[0], gate_float32_position
                        )
                    elif (
                        len(gate_float32_uses) != 1
                        or _as_operation(gate_float32_uses[0].owner) != gate_convert
                    ):
                        raise ValueError(
                            "strict MLP float32 gate result must feed only its BF16 cast"
                        )
                    gate_projection = _as_operation(_follow_shape_only_producer(gate_float32).owner)
                    gate_is_named_pallas = (
                        gate_projection is not None
                        and gate_projection.name == "stablehlo.custom_call"
                        and str(gate_projection.attributes.get("call_target_name"))
                        == '"tpu_custom_call"'
                        and str(gate_projection.attributes.get("kernel_name"))
                        == '"seqax_named_einsum"'
                    )
                    gate_is_interpret_callback = (
                        allow_callbacks
                        and gate_projection is not None
                        and gate_projection.name == "stablehlo.custom_call"
                        and str(gate_projection.attributes.get("call_target_name"))
                        == '"xla_ffi_python_cpu_callback"'
                    )
                    if (
                        gate_projection is None
                        or gate_projection.name
                        not in {"stablehlo.dot_general", "stablehlo.custom_call"}
                        or (
                            gate_projection.name == "stablehlo.custom_call"
                            and not gate_is_named_pallas
                            and not gate_is_interpret_callback
                        )
                        or len(gate_projection.operands) < 2
                    ):
                        raise ValueError(
                            "strict MLP float32 gate checkpoint must come from its projection"
                        )
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
                        or checkpoint_uses[0].operand_number
                        != leading_result_count + 7 + 13 * layer
                    ):
                        raise ValueError(
                            "instrumented strict SiLU must return the real gate checkpoint"
                        )
                    _require_checkpoint_function_result(
                        checkpoint_uses[0],
                        leading_result_count + 7 + 13 * layer,
                    )
                    checkpoint_return = _as_operation(checkpoint_uses[0].owner)
                elif source_consumers != (input_convert,):
                    raise ValueError("strict SiLU input barrier must feed only its promotion")
                rms_input_position = leading_result_count + 1 + 13 * layer
                rms_mean_square_position = leading_result_count + 2 + 13 * layer
                rms_inverse_position = leading_result_count + 3 + 13 * layer
                normalized_float32_position = leading_result_count + 4 + 13 * layer
                normalized_position = leading_result_count + 5 + 13 * layer
                gate_normalized_input = None
                if require_hidden_down:
                    assert gate_projection is not None
                    gate_normalized_input = (
                        checkpoint_return.operands[normalized_position]
                        if gate_is_interpret_callback and checkpoint_return is not None
                        else (
                            None
                            if gate_is_interpret_callback
                            else _follow_shape_only_producer(gate_projection.operands[0])
                        )
                    )
                if gate_normalized_input is not None and not _is_bf16_tensor(gate_normalized_input):
                    raise ValueError(
                        "strict MLP gate projection must consume a BF16 normalized input"
                    )
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
                        or checkpoint_uses[0].operand_number
                        != leading_result_count + 8 + 13 * layer
                    ):
                        raise ValueError(
                            "instrumented strict SiLU must return the real SiLU checkpoint"
                        )
                    _require_checkpoint_function_result(
                        checkpoint_uses[0],
                        leading_result_count + 8 + 13 * layer,
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
                    up_input_barrier = _as_operation(up_operand.owner)
                    if (
                        up_input_barrier is None
                        or up_input_barrier.name != "stablehlo.optimization_barrier"
                        or len(up_input_barrier.operands) != 1
                        or len(up_input_barrier.results) != 1
                        or up_input_barrier.results[0] != up_operand
                        or not _is_bf16_tensor(up_input_barrier.operands[0])
                    ):
                        raise ValueError("strict MLP up value must pass through one input barrier")
                    _require_attribute_names(up_input_barrier, frozenset())
                    converted_up = up_input_barrier.operands[0]
                    if tuple(_as_operation(use.owner) for use in converted_up.uses) != (
                        up_input_barrier,
                    ):
                        raise ValueError("strict MLP converted up value must feed only its barrier")
                    up_convert = _as_operation(converted_up.owner)
                    if (
                        up_convert is None
                        or up_convert.name != "stablehlo.convert"
                        or len(up_convert.operands) != 1
                        or len(up_convert.results) != 1
                        or up_convert.results[0] != converted_up
                        or not _is_f32_tensor(up_convert.operands[0])
                    ):
                        raise ValueError(
                            "strict MLP up value must come from one float32 conversion"
                        )
                    _require_attribute_names(up_convert, frozenset())
                    up_float32 = up_convert.operands[0]
                    up_float32_uses = tuple(up_float32.uses)
                    up_float32_checkpoints = tuple(
                        use
                        for use in up_float32_uses
                        if _as_operation(use.owner) is not None
                        and _as_operation(use.owner).name in _REGION_TERMINATORS
                    )
                    if instrumented:
                        up_float32_position = leading_result_count + 9 + 13 * layer
                        if (
                            len(up_float32_uses) != 2
                            or len(up_float32_checkpoints) != 1
                            or up_float32_checkpoints[0].operand_number != up_float32_position
                        ):
                            raise ValueError(
                                "instrumented strict MLP must return the real float32 up checkpoint"
                            )
                        _require_checkpoint_function_result(
                            up_float32_checkpoints[0], up_float32_position
                        )
                    elif (
                        len(up_float32_uses) != 1
                        or _as_operation(up_float32_uses[0].owner) != up_convert
                    ):
                        raise ValueError(
                            "strict MLP float32 up result must feed only its BF16 cast"
                        )
                    up_projection_result = _follow_shape_only_producer(up_float32)
                    up_projection = _as_operation(up_projection_result.owner)
                    is_named_pallas = (
                        up_projection is not None
                        and up_projection.name == "stablehlo.custom_call"
                        and str(up_projection.attributes.get("call_target_name"))
                        == '"tpu_custom_call"'
                        and str(up_projection.attributes.get("kernel_name"))
                        == '"seqax_named_einsum"'
                    )
                    is_interpret_callback = (
                        allow_callbacks
                        and up_projection is not None
                        and up_projection.name == "stablehlo.custom_call"
                        and str(up_projection.attributes.get("call_target_name"))
                        == '"xla_ffi_python_cpu_callback"'
                    )
                    if (
                        up_projection is None
                        or up_projection.name
                        not in {"stablehlo.dot_general", "stablehlo.custom_call"}
                        or (
                            up_projection.name == "stablehlo.custom_call"
                            and not is_named_pallas
                            and not is_interpret_callback
                        )
                        or len(up_projection.operands) < 2
                    ):
                        raise ValueError(
                            "strict MLP float32 up checkpoint must come from its projection"
                        )
                    normalized_input = (
                        checkpoint_return.operands[normalized_position]
                        if is_interpret_callback and checkpoint_return is not None
                        else _follow_shape_only_producer(up_projection.operands[0])
                    )
                    if not _is_bf16_tensor(normalized_input):
                        raise ValueError(
                            "strict MLP up projection must consume a BF16 normalized input"
                        )
                    if (
                        gate_normalized_input is not None
                        and normalized_input != gate_normalized_input
                    ):
                        raise ValueError(
                            "strict MLP gate and up projections must consume the same normalized input"
                        )
                    normalized_convert = _as_operation(normalized_input.owner)
                    if (
                        normalized_convert is None
                        or normalized_convert.name != "stablehlo.convert"
                        or len(normalized_convert.operands) != 1
                        or not _is_f32_tensor(normalized_convert.operands[0])
                    ):
                        raise ValueError(
                            "strict RMSNorm normalized BF16 checkpoint must come from float32"
                        )
                    _require_attribute_names(normalized_convert, frozenset())
                    normalized_float32 = normalized_convert.operands[0]
                    normalized_multiply = _as_operation(normalized_float32.owner)
                    if (
                        normalized_multiply is None
                        or normalized_multiply.name != "stablehlo.multiply"
                        or len(normalized_multiply.operands) != 2
                    ):
                        raise ValueError(
                            "strict RMSNorm normalized float32 checkpoint must apply its scale"
                        )
                    rms_input, rms_mean_square, rms_inverse = _validate_rmsnorm_dataflow(
                        normalized_float32,
                        normalized_multiply,
                        manual_scale_argument=4 + leading_result_count,
                    )
                    if instrumented:
                        for value, position, label in (
                            (rms_input, rms_input_position, "input"),
                            (rms_mean_square, rms_mean_square_position, "mean-square"),
                            (rms_inverse, rms_inverse_position, "inverse"),
                            (
                                normalized_float32,
                                normalized_float32_position,
                                "normalized float32",
                            ),
                        ):
                            checkpoint_uses = tuple(
                                use
                                for use in value.uses
                                if _as_operation(use.owner) is not None
                                and _as_operation(use.owner).name in _REGION_TERMINATORS
                            )
                            if (
                                len(checkpoint_uses) != 1
                                or checkpoint_uses[0].operand_number != position
                            ):
                                raise ValueError(
                                    f"instrumented strict RMSNorm must return its real {label} checkpoint"
                                )
                            _require_checkpoint_function_result(checkpoint_uses[0], position)
                    normalized_checkpoint_uses = tuple(
                        use
                        for use in normalized_input.uses
                        if _as_operation(use.owner) is not None
                        and _as_operation(use.owner).name in _REGION_TERMINATORS
                    )
                    if instrumented:
                        if (
                            len(normalized_checkpoint_uses) != 1
                            or normalized_checkpoint_uses[0].operand_number != normalized_position
                        ):
                            raise ValueError(
                                "instrumented strict MLP must return the real normalized up input"
                            )
                        _require_checkpoint_function_result(
                            normalized_checkpoint_uses[0], normalized_position
                        )
                    elif normalized_checkpoint_uses:
                        raise ValueError(
                            "uninstrumented strict MLP must not return its normalized input"
                        )
                    up_uses = tuple(up_operand.uses)
                    up_checkpoint_uses = tuple(
                        use
                        for use in up_uses
                        if _as_operation(use.owner) is not None
                        and _as_operation(use.owner).name in _REGION_TERMINATORS
                    )
                    if instrumented:
                        up_position = leading_result_count + 10 + 13 * layer
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
                    hidden_position = leading_result_count + 11 + 13 * layer
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
                    down_float32_position = leading_result_count + 12 + 13 * layer
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
                    down_bfloat16_position = leading_result_count + 13 + 13 * layer
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
