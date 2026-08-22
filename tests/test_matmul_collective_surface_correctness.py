from __future__ import annotations

import numpy as np
import pytest

from tpu_cake.identity import array_sha256
from tpu_cake.matmul_collective_surface_correctness import (
    CORRECTNESS_PATTERNS,
    correctness_sentinel_coordinates,
    make_correctness_operand_shard,
)
from tpu_cake.matmul_collective_surface_correctness_oracle import make_correctness_oracle


@pytest.mark.parametrize("pattern", CORRECTNESS_PATTERNS)
def test_structured_bf16_patterns_match_dense_float32_matmul(pattern: str) -> None:
    dimensions = {"m": 16, "k": 128, "n": 32}
    lhs = np.concatenate(
        tuple(
            make_correctness_operand_shard(
                pattern,
                "lhs",
                **dimensions,
                k_start=shard * 16,
                k_stop=(shard + 1) * 16,
            )
            for shard in range(8)
        ),
        axis=1,
    )
    rhs = np.concatenate(
        tuple(
            make_correctness_operand_shard(
                pattern,
                "rhs",
                **dimensions,
                k_start=shard * 16,
                k_stop=(shard + 1) * 16,
            )
            for shard in range(8)
        ),
        axis=0,
    )
    actual = lhs.astype(np.float32) @ rhs.astype(np.float32)
    expected = make_correctness_oracle(pattern, **dimensions)

    assert expected.dtype == np.dtype("<f4")
    assert expected.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


def test_patterns_have_distinct_real_shape_oracles() -> None:
    hashes = {
        array_sha256(make_correctness_oracle(pattern, m=768, k=131072, n=2048))
        for pattern in CORRECTNESS_PATTERNS
    }

    assert len(hashes) == len(CORRECTNESS_PATTERNS)


def test_signed_periodic_has_distinct_simultaneous_device_contributions() -> None:
    dimensions = {"m": 16, "k": 16384, "n": 16}
    lhs_shards = tuple(
        make_correctness_operand_shard(
            "signed-periodic",
            "lhs",
            **dimensions,
            k_start=device * 2048,
            k_stop=(device + 1) * 2048,
        )
        for device in range(8)
    )
    rhs_shards = tuple(
        make_correctness_operand_shard(
            "signed-periodic",
            "rhs",
            **dimensions,
            k_start=device * 2048,
            k_stop=(device + 1) * 2048,
        )
        for device in range(8)
    )
    partials = tuple(
        lhs.astype(np.float32) @ rhs.astype(np.float32)
        for lhs, rhs in zip(lhs_shards, rhs_shards, strict=True)
    )
    coordinate = next(
        (row, column)
        for row in range(16)
        for column in range(16)
        if all(partial[row, column] != 0 for partial in partials)
    )

    assert len({float(partial[coordinate]) for partial in partials}) == 8
    np.testing.assert_array_equal(
        sum(partials), make_correctness_oracle("signed-periodic", **dimensions)
    )


@pytest.mark.parametrize("pattern", CORRECTNESS_PATTERNS)
@pytest.mark.parametrize("role", ("lhs", "rhs"))
def test_sentinels_are_canonical_and_pattern_support_aware(pattern: str, role: str) -> None:
    dimensions = {"m": 16, "k": 16384, "n": 32}
    device = 3
    coordinates = correctness_sentinel_coordinates(
        pattern,
        role,
        protocol_id="a" * 64,
        scenario_name="calibration-0",
        device_id=device,
        **dimensions,
    )

    assert len(coordinates) == len(set(coordinates)) == 32
    assert coordinates == tuple(sorted(coordinates))
    local_k = dimensions["k"] // 8
    if role == "lhs":
        assert all(0 <= row < dimensions["m"] for row, _ in coordinates)
        assert all(
            device * local_k <= reduction < (device + 1) * local_k for _, reduction in coordinates
        )
    else:
        assert all(
            device * local_k <= reduction < (device + 1) * local_k for reduction, _ in coordinates
        )
        assert all(0 <= column < dimensions["n"] for _, column in coordinates)
    shard = make_correctness_operand_shard(
        pattern,
        role,
        **dimensions,
        k_start=device * local_k,
        k_stop=(device + 1) * local_k,
    )
    if pattern in {"one-hot-stripes", "block-diagonal"}:
        local_coordinates = (
            tuple((first, second - device * local_k) for first, second in coordinates)
            if role == "lhs"
            else tuple((first - device * local_k, second) for first, second in coordinates)
        )
        assert any(shard[value] != 0 for value in local_coordinates)


@pytest.mark.parametrize(
    ("updates", "error"),
    (
        ({"pattern": "invented"}, "PATTERN_INVALID"),
        ({"m": 15}, "SHAPE_INVALID"),
        ({"k": 120}, "SHAPE_INVALID"),
        ({"mesh_size": 4}, "SHAPE_INVALID"),
        ({"k_start": 128}, "SHARD_INVALID"),
        ({"k_stop": 32}, "SHARD_INVALID"),
    ),
)
def test_pattern_generator_rejects_ambiguous_contracts(updates, error: str) -> None:
    arguments = {
        "pattern": "constant",
        "role": "lhs",
        "m": 16,
        "k": 128,
        "n": 32,
        "k_start": 0,
        "k_stop": 16,
        "mesh_size": 8,
    }
    arguments.update(updates)

    with pytest.raises(ValueError, match=error):
        make_correctness_operand_shard(**arguments)
