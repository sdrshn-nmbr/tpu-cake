from __future__ import annotations

from typing import Literal

import numpy as np

ORACLE_PATTERN_SCHEMA = "structured-bf16-analytical-v1"
OracleCorrectnessPattern = Literal[
    "constant",
    "one-hot-stripes",
    "signed-periodic",
    "block-diagonal",
    "low-rank",
]
ORACLE_PATTERNS: tuple[OracleCorrectnessPattern, ...] = (
    "constant",
    "one-hot-stripes",
    "signed-periodic",
    "block-diagonal",
    "low-rank",
)
_SIGNED_LHS = np.asarray(
    (1, -2, 3, -4, 2, -1, 4, -3, -1, 3, -2, 4, -4, 2, -3, 1),
    dtype=np.int64,
)
_SIGNED_RHS = np.asarray(
    (2, 1, -3, 4, -1, -4, 3, -2, 4, -3, 1, -2, 3, 2, -4, -1),
    dtype=np.int64,
)


def make_correctness_oracle(
    pattern: OracleCorrectnessPattern,
    *,
    m: int,
    k: int,
    n: int,
    mesh_size: int = 8,
) -> np.ndarray:
    _validate_problem(pattern, m=m, k=k, n=n, mesh_size=mesh_size)
    rows = np.arange(m, dtype=np.int64)[:, None]
    columns = np.arange(n, dtype=np.int64)[None, :]
    if pattern == "constant":
        return np.full((m, n), k * 2.0**-17, dtype=np.float32)
    if pattern == "one-hot-stripes":
        local_k = k // mesh_size
        positions = (rows % mesh_size) * local_k + ((257 * rows + 17) % local_k)
        code = (columns // 16 + 3 * (positions % 32) + 5 * (positions // local_k)) % 8
        signs = np.where(code >= 4, -1.0, 1.0)
        return np.ascontiguousarray((signs * ((code % 4) + 1) * 2.0**-3).astype(np.float32))
    if pattern == "signed-periodic":
        left = _SIGNED_LHS[(np.arange(16)[None, :] + rows) % 16]
        right = _SIGNED_RHS[(np.arange(16)[None, :, None] + 3 * columns[:, None, :]) % 16]
        period_dot = np.sum(left[:, :, None] * right, axis=1, dtype=np.int64)
        shard_weight_sum = mesh_size * (mesh_size + 1) // 2
        return np.ascontiguousarray(
            (period_dot * (k // (mesh_size * 16)) * shard_weight_sum * 2.0**-19).astype(np.float32)
        )
    if pattern == "block-diagonal":
        row_blocks = (16 * rows) // m
        column_blocks = (16 * columns) // n
        return np.ascontiguousarray(
            ((row_blocks == column_blocks) * (k // 16) * 2.0**-14).astype(np.float32)
        )
    if pattern != "low-rank":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ORACLE_PATTERN_INVALID")
    lhs_factors = np.stack(
        (
            np.ones(m, dtype=np.int64),
            rows[:, 0] % 3 - 1,
            np.where(rows[:, 0] % 4 < 2, 1, -1),
            np.where(rows[:, 0] % 5 < 3, 1, -1),
        ),
        axis=1,
    )
    rhs_factors = np.stack(
        (
            np.where(columns[0] % 2 == 0, 1, -1),
            columns[0] % 3 - 1,
            np.where(np.isin(columns[0] % 4, (0, 3)), 1, -1),
            np.where(columns[0] % 5 < 2, 1, -1),
        )
    )
    return np.ascontiguousarray((lhs_factors @ rhs_factors * k * 2.0**-17).astype(np.float32))


def _validate_problem(
    pattern: str,
    *,
    m: int,
    k: int,
    n: int,
    mesh_size: int,
) -> None:
    if pattern not in ORACLE_PATTERNS:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ORACLE_PATTERN_INVALID")
    if min(m, k, n) <= 0 or mesh_size != 8 or k % 16 or k % mesh_size or m % 16 or n % 16:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_ORACLE_SHAPE_INVALID")
