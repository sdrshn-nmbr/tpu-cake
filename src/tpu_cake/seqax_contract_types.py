from __future__ import annotations

from enum import StrEnum


class SeqaxNormScalePlacement(StrEnum):
    SHARDED = "sharded"
    REPLICATED = "replicated"


class SeqaxDataAxisPlacement(StrEnum):
    SHARDED = "sharded"
    REPLICATED = "replicated"


class SeqaxNumericalSemantics(StrEnum):
    LEGACY_FUSED_V0 = "legacy_fused_v0"
    TYPED_BF16_V1 = "typed_bf16_v1"
    TYPED_BF16_HIDDEN_V2 = "typed_bf16_hidden_v2"


class SeqaxFeedForwardFusion(StrEnum):
    SEPARATE = "separate"
    SILU_MULTIPLY = "silu_multiply"


class SeqaxFeedForwardVectorExecution(StrEnum):
    LEGACY_MIXED = "legacy_mixed"
    PALLAS_FULL_LOCAL = "pallas_full_local"


class SeqaxResidualNormStrategy(StrEnum):
    STANDARD = "standard"
    SHARDED_RMS = "sharded_rms"
    RESIDUAL_ALL_REDUCE = "residual_all_reduce"
