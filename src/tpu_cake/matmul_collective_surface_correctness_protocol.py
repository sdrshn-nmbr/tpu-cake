from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import model_identity_sha256
from tpu_cake.runner import MatmulCollectiveStrategy

CORRECTNESS_PROTOCOL_SCHEMA = "matmul-collective-surface-correctness-protocol-v1"
PROTOCOL_PATTERN_SCHEMA = "structured-bf16-analytical-v1"
PROTOCOL_PATTERNS = (
    "constant",
    "one-hot-stripes",
    "signed-periodic",
    "block-diagonal",
    "low-rank",
)
_SIGNED_LHS_SEQUENCE = (1, -2, 3, -4, 2, -1, 4, -3, -1, 3, -2, 4, -4, 2, -3, 1)
_SIGNED_RHS_SEQUENCE = (2, 1, -3, 4, -1, -4, 3, -2, 4, -3, 1, -2, 3, 2, -4, -1)


class SurfaceParentCompileAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    archive_path: str
    manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SurfaceCorrectnessPatternContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["structured-bf16-analytical-v1"] = PROTOCOL_PATTERN_SCHEMA
    ordered_patterns: tuple[str, ...] = PROTOCOL_PATTERNS
    constant_formula: Literal["A=1;B=2^-17;C=K*2^-17"] = "A=1;B=2^-17;C=K*2^-17"
    one_hot_formula: Literal[
        "L=K/8;p=(i%8)*L+((257*i+17)%L);A=1[k=p];"
        "c=(j//16+3*(k%32)+5*(k//L))%8;"
        "B=(-1 if c>=4 else +1)*(c%4+1)*2^-3;C=B[p,j]"
    ] = (
        "L=K/8;p=(i%8)*L+((257*i+17)%L);A=1[k=p];"
        "c=(j//16+3*(k%32)+5*(k//L))%8;"
        "B=(-1 if c>=4 else +1)*(c%4+1)*2^-3;C=B[p,j]"
    )
    signed_lhs_sequence: tuple[int, ...] = (
        1,
        -2,
        3,
        -4,
        2,
        -1,
        4,
        -3,
        -1,
        3,
        -2,
        4,
        -4,
        2,
        -3,
        1,
    )
    signed_rhs_sequence: tuple[int, ...] = (
        2,
        1,
        -3,
        4,
        -1,
        -4,
        3,
        -2,
        4,
        -3,
        1,
        -2,
        3,
        2,
        -4,
        -1,
    )
    signed_formula: Literal[
        "s=k//(K/8);A=a[(k+i)%16]*2^-4;B=b[(k+3*j)%16]*(s+1)*2^-15;"
        "C=(K/128)*36*2^-19*sum_r(a[(r+i)%16]*b[(r+3*j)%16])"
    ] = (
        "s=k//(K/8);A=a[(k+i)%16]*2^-4;B=b[(k+3*j)%16]*(s+1)*2^-15;"
        "C=(K/128)*36*2^-19*sum_r(a[(r+i)%16]*b[(r+3*j)%16])"
    )
    block_formula: Literal[
        "rb=16*i//M;kb=16*k//K;cb=16*j//N;A=1[rb=kb];B=2^-14*1[kb=cb];C=(K/16)*2^-14*1[rb=cb]"
    ] = "rb=16*i//M;kb=16*k//K;cb=16*j//N;A=1[rb=kb];B=2^-14*1[kb=cb];C=(K/16)*2^-14*1[rb=cb]"
    low_rank_formula: Literal[
        "q=[1,(-1)^bit0(k),(-1)^bit1(k),(-1)^bit2(k)];"
        "u=[1,i%3-1,(+1 if i%4<2 else -1),(+1 if i%5<3 else -1)];"
        "v=[(+1 if j%2=0 else -1),j%3-1,(+1 if j%4 in {0,3} else -1),"
        "(+1 if j%5<2 else -1)];"
        "A=sum(u*q);B=2^-17*sum(q*v);C=K*2^-17*sum(u*v)"
    ] = (
        "q=[1,(-1)^bit0(k),(-1)^bit1(k),(-1)^bit2(k)];"
        "u=[1,i%3-1,(+1 if i%4<2 else -1),(+1 if i%5<3 else -1)];"
        "v=[(+1 if j%2=0 else -1),j%3-1,(+1 if j%4 in {0,3} else -1),"
        "(+1 if j%5<2 else -1)];"
        "A=sum(u*q);B=2^-17*sum(q*v);C=K*2^-17*sum(u*v)"
    )

    @model_validator(mode="after")
    def exact_pattern_contract(self) -> SurfaceCorrectnessPatternContract:
        if (
            self.ordered_patterns != PROTOCOL_PATTERNS
            or self.signed_lhs_sequence != _SIGNED_LHS_SEQUENCE
            or self.signed_rhs_sequence != _SIGNED_RHS_SEQUENCE
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PATTERN_CONTRACT_MISMATCH")
        return self

    @computed_field
    @property
    def contract_sha256(self) -> str:
        return model_identity_sha256(self)


class MatmulCollectiveSurfaceCorrectnessProtocol(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["matmul-collective-surface-correctness-protocol-v1"] = (
        CORRECTNESS_PROTOCOL_SCHEMA
    )
    parent_compile: SurfaceParentCompileAuthority
    scenarios: tuple[str, ...] = Field(min_length=20, max_length=20)
    calibration_scenarios: tuple[str, ...] = Field(min_length=16, max_length=16)
    holdout_scenarios: tuple[str, ...] = Field(min_length=4, max_length=4)
    initial_execution_split: Literal["calibration"] = "calibration"
    strategies: tuple[MatmulCollectiveStrategy, ...] = Field(min_length=2, max_length=2)
    patterns: SurfaceCorrectnessPatternContract
    numpy_version: Literal["2.5.2"] = "2.5.2"
    ml_dtypes_version: Literal["0.6.0"] = "0.6.0"
    logical_input_dtype: Literal["bfloat16"] = "bfloat16"
    lhs_sharding: Literal["PartitionSpec(None, 't')"] = "PartitionSpec(None, 't')"
    rhs_sharding: Literal["PartitionSpec('t', None)"] = "PartitionSpec('t', None)"
    output_dtype: Literal["float32"] = "float32"
    output_sharding: Literal["PartitionSpec(None, 't')"] = "PartitionSpec(None, 't')"
    output_file_format: Literal["npy-allow-pickle-false-v1"] = "npy-allow-pickle-false-v1"
    save_global_inputs: Literal[False] = False
    save_oracle_outputs: Literal[True] = True
    save_candidate_outputs: Literal[True] = True
    absolute_tolerance: Literal[0.001] = 0.001
    relative_tolerance: Literal[0.001] = 0.001
    mismatch_rule: Literal["abs(candidate-oracle)>atol+rtol*abs(oracle)-v1"] = (
        "abs(candidate-oracle)>atol+rtol*abs(oracle)-v1"
    )
    normalized_error_rule: Literal["abs(candidate-oracle)/(atol+rtol*abs(oracle))-v1"] = (
        "abs(candidate-oracle)/(atol+rtol*abs(oracle))-v1"
    )
    shard_identity_schema: Literal[
        "logical-dtype-global-shape-sharding-device-slice-host-callback-payload-device-sentinels-v1"
    ] = "logical-dtype-global-shape-sharding-device-slice-host-callback-payload-device-sentinels-v1"
    sentinel_rule: Literal["pattern-support-plus-32-semantic-coordinates-per-device-shard-v1"] = (
        "pattern-support-plus-32-semantic-coordinates-per-device-shard-v1"
    )
    sentinel_count_per_shard: Literal[32] = 32
    strategy_order_rule: Literal["pattern-parity-abba-baab-v1"] = "pattern-parity-abba-baab-v1"
    correctness_repetitions_per_strategy: Literal[2] = 2
    compile_continuity_rule: Literal[
        "fresh-schedule-pallas-semantic-stablehlo-semantic-compilerhlo-v1"
    ] = "fresh-schedule-pallas-semantic-stablehlo-semantic-compilerhlo-v1"
    attempt_registry_root: Literal[
        "/home/sudarshan/tpu-cake-evidence/matmul-collective-surface-correctness-attempts-v1"
    ] = "/home/sudarshan/tpu-cake-evidence/matmul-collective-surface-correctness-attempts-v1"
    allow_retry: Literal[False] = False
    one_shot_attempt_ledger: Literal[True] = True

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> MatmulCollectiveSurfaceCorrectnessProtocol:
        calibration = tuple(f"calibration-{index}" for index in range(16))
        holdout = tuple(f"holdout-{index}" for index in range(4))
        expected_strategies = (
            MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
            MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING,
        )
        if (
            self.scenarios != (*calibration, *holdout)
            or self.calibration_scenarios != calibration
            or self.holdout_scenarios != holdout
            or self.strategies != expected_strategies
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PROTOCOL_INVENTORY_MISMATCH")
        return self

    @computed_field
    @property
    def protocol_id(self) -> str:
        return model_identity_sha256(self)


def default_matmul_collective_surface_correctness_protocol() -> (
    MatmulCollectiveSurfaceCorrectnessProtocol
):
    return MatmulCollectiveSurfaceCorrectnessProtocol(
        parent_compile=SurfaceParentCompileAuthority(
            archive_path=(
                "/home/sudarshan/tpu-cake-evidence/"
                "matmul-collective-surface-compile-6dead4d-f74a1b6c"
            ),
            manifest_file_sha256=(
                "fbb0483bea04e6fc0a55036b36b8cd1efbb430ba03d941fca8998ef423f2f6ac"
            ),
            compile_report_file_sha256=(
                "e29c33e65458d7f6ad918c429006fece81661cb8cc70bea551689fb2cbced553"
            ),
            design_id="f2f8a0eeba4842167780cd3d79043443d0d02392ed037a5250df1a2218691d83",
            attempt_id="f74a1b6c424be9de661909d2b349244ccdd76ab867522a0798364326a5f25252",
            source_commit="6dead4dfa23e912fa6352452d1a9480cca9d1f7b",
            source_authority_sha256=(
                "787f840630a1db875c81ed5dcb756b730ef1c6f25f078031e39547876ae51cb0"
            ),
            execution_authority_sha256=(
                "0932ab6a7f0166da8dd914d76556bacb586da22c30df3b59987975e79445390a"
            ),
            compile_report_sha256=(
                "e5a2bdd9c2735e7c22a0fc2b6ddcb498cf298bed73353eb435a954d8d244ba5b"
            ),
            compile_ledger_sha256=(
                "b91c6cde9a89f39c333d0459d16b2ce355674711282485113dd9a3206da31324"
            ),
        ),
        scenarios=(
            *(f"calibration-{index}" for index in range(16)),
            *(f"holdout-{index}" for index in range(4)),
        ),
        calibration_scenarios=tuple(f"calibration-{index}" for index in range(16)),
        holdout_scenarios=tuple(f"holdout-{index}" for index in range(4)),
        strategies=(
            MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
            MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING,
        ),
        patterns=SurfaceCorrectnessPatternContract(),
    )
