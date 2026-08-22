from __future__ import annotations

from typing import Literal

import ml_dtypes
import numpy as np

from tpu_cake.identity import semantic_seed

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


def correctness_sentinel_coordinates(
    pattern: SurfaceCorrectnessPattern,
    role: SurfaceOperandRole,
    *,
    protocol_id: str,
    scenario_name: str,
    m: int,
    k: int,
    n: int,
    device_id: int,
    mesh_size: int = 8,
    count: int = 32,
) -> tuple[tuple[int, int], ...]:
    _validate_problem(pattern, m=m, k=k, n=n, mesh_size=mesh_size)
    if role not in {"lhs", "rhs"} or not 0 <= device_id < mesh_size or count != 32:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SENTINEL_REQUEST_INVALID")
    local_k = k // mesh_size
    k_start = device_id * local_k
    k_stop = (device_id + 1) * local_k
    first_bounds = (0, m) if role == "lhs" else (k_start, k_stop)
    second_bounds = (k_start, k_stop) if role == "lhs" else (0, n)
    coordinates: set[tuple[int, int]] = set()

    def add(first: int, second: int) -> None:
        if (
            first_bounds[0] <= first < first_bounds[1]
            and second_bounds[0] <= second < second_bounds[1]
        ):
            coordinates.add((first, second))

    for first_fraction, second_fraction in (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
    ):
        first = first_bounds[0] + (first_bounds[1] - first_bounds[0] - 1) * first_fraction // 3
        second = second_bounds[0] + (second_bounds[1] - second_bounds[0] - 1) * second_fraction // 3
        add(first, second)
    if pattern == "one-hot-stripes" and role == "lhs":
        matching_rows = tuple(row for row in range(m) if row % mesh_size == device_id)
        for row in (*matching_rows[:4], *matching_rows[-4:]):
            add(row, device_id * local_k + ((257 * row + 17) % local_k))
    elif pattern == "one-hot-stripes":
        for column in (0, min(15, n - 1), min(16, n - 1), n - 1):
            add(k_start, column)
            add(k_stop - 1, column)
    elif pattern == "block-diagonal":
        for block in (device_id * 2, device_id * 2 + 1):
            reduction = block * k // 16
            if role == "lhs":
                add(block * m // 16, reduction)
                add(((block + 1) % 16) * m // 16, reduction)
            else:
                add(reduction, block * n // 16)
                add(reduction, ((block + 1) % 16) * n // 16)
    seed = semantic_seed(
        protocol_id,
        scenario_name,
        pattern,
        role,
        str(device_id),
        "surface-correctness-sentinels-v1",
    )
    first_size = first_bounds[1] - first_bounds[0]
    second_size = second_bounds[1] - second_bounds[0]
    flat_size = first_size * second_size
    counter = 0
    while len(coordinates) < count:
        flat = (seed + counter * 0x9E3779B97F4A7C15) % flat_size
        add(first_bounds[0] + flat // second_size, second_bounds[0] + flat % second_size)
        counter += 1
    return tuple(sorted(coordinates)[:count])


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
