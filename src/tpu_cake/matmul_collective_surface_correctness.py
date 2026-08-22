from __future__ import annotations

from typing import Literal

import ml_dtypes
import numpy as np

SurfaceCorrectnessPattern = Literal[
    "constant",
    "one-hot-stripes",
    "signed-periodic",
    "block-diagonal",
    "low-rank",
]
SurfaceOperandRole = Literal["lhs", "rhs"]

CORRECTNESS_PATTERN_SCHEMA = "structured-bf16-analytical-v1"
CORRECTNESS_PATTERNS: tuple[SurfaceCorrectnessPattern, ...] = (
    "constant",
    "one-hot-stripes",
    "signed-periodic",
    "block-diagonal",
    "low-rank",
)
_BF16 = np.dtype(ml_dtypes.bfloat16)
_SIGNED_LHS = np.asarray(
    (1, -2, 3, -4, 2, -1, 4, -3, -1, 3, -2, 4, -4, 2, -3, 1),
    dtype=np.int8,
)
_SIGNED_RHS = np.asarray(
    (2, 1, -3, 4, -1, -4, 3, -2, 4, -3, 1, -2, 3, 2, -4, -1),
    dtype=np.int8,
)


def make_correctness_operand_shard(
    pattern: SurfaceCorrectnessPattern,
    role: SurfaceOperandRole,
    *,
    m: int,
    k: int,
    n: int,
    k_start: int,
    k_stop: int,
    mesh_size: int = 8,
) -> np.ndarray:
    _validate_problem(pattern, m=m, k=k, n=n, mesh_size=mesh_size)
    local_k = k // mesh_size
    if (
        role not in {"lhs", "rhs"}
        or not 0 <= k_start < k_stop <= k
        or k_stop - k_start != local_k
        or k_start % local_k
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SHARD_INVALID")
    rows = np.arange(m, dtype=np.int64)[:, None]
    columns = np.arange(n, dtype=np.int64)[None, :]
    reduction = np.arange(k_start, k_stop, dtype=np.int64)
    if pattern == "constant":
        shape = (m, reduction.size) if role == "lhs" else (reduction.size, n)
        value = 1.0 if role == "lhs" else 2.0**-17
        return np.full(shape, value, dtype=_BF16)
    if pattern == "one-hot-stripes":
        if role == "lhs":
            positions = (rows % mesh_size) * local_k + ((257 * rows + 17) % local_k)
            return np.ascontiguousarray((positions == reduction[None, :]).astype(_BF16))
        code = (
            columns // 16 + 3 * (reduction[:, None] % 32) + 5 * (reduction[:, None] // local_k)
        ) % 8
        signs = np.where(code >= 4, -1.0, 1.0)
        return np.ascontiguousarray((signs * ((code % 4) + 1) * 2.0**-3).astype(_BF16))
    if pattern == "signed-periodic":
        if role == "lhs":
            indexes = (rows + reduction[None, :]) % _SIGNED_LHS.size
            return np.ascontiguousarray((_SIGNED_LHS[indexes] * 2.0**-4).astype(_BF16))
        indexes = (reduction[:, None] + 3 * columns) % _SIGNED_RHS.size
        shard_weight = k_start // local_k + 1
        return np.ascontiguousarray((_SIGNED_RHS[indexes] * shard_weight * 2.0**-15).astype(_BF16))
    if pattern == "block-diagonal":
        k_blocks = (16 * reduction) // k
        if role == "lhs":
            row_blocks = (16 * rows) // m
            return np.ascontiguousarray((row_blocks == k_blocks[None, :]).astype(_BF16))
        column_blocks = (16 * columns) // n
        return np.ascontiguousarray(((k_blocks[:, None] == column_blocks) * 2.0**-14).astype(_BF16))
    if pattern == "low-rank":
        q = _low_rank_reduction_factors(reduction)
        if role == "lhs":
            return np.ascontiguousarray((_low_rank_lhs_factors(m) @ q).astype(_BF16))
        return np.ascontiguousarray((q.T @ _low_rank_rhs_factors(n) * 2.0**-17).astype(_BF16))
    raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PATTERN_INVALID")


def _low_rank_reduction_factors(reduction: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.ones(reduction.size, dtype=np.int8),
            np.where(reduction & 1, -1, 1),
            np.where(reduction & 2, -1, 1),
            np.where(reduction & 4, -1, 1),
        )
    )


def _low_rank_lhs_factors(m: int) -> np.ndarray:
    rows = np.arange(m, dtype=np.int64)
    return np.stack(
        (
            np.ones(m, dtype=np.int8),
            (rows % 3 - 1).astype(np.int8),
            np.where(rows % 4 < 2, 1, -1).astype(np.int8),
            np.where(rows % 5 < 3, 1, -1).astype(np.int8),
        ),
        axis=1,
    )


def _low_rank_rhs_factors(n: int) -> np.ndarray:
    columns = np.arange(n, dtype=np.int64)
    return np.stack(
        (
            np.where(columns % 2 == 0, 1, -1).astype(np.int8),
            (columns % 3 - 1).astype(np.int8),
            np.where(np.isin(columns % 4, (0, 3)), 1, -1).astype(np.int8),
            np.where(columns % 5 < 2, 1, -1).astype(np.int8),
        )
    )


def _validate_pattern(pattern: str) -> None:
    if pattern not in CORRECTNESS_PATTERNS:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PATTERN_INVALID")


def _validate_problem(
    pattern: str,
    *,
    m: int,
    k: int,
    n: int,
    mesh_size: int,
) -> None:
    _validate_pattern(pattern)
    if min(m, k, n) <= 0 or mesh_size != 8 or k % 16 or k % mesh_size or m % 16 or n % 16:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SHAPE_INVALID")
