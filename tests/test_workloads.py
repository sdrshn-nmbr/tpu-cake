import numpy as np

from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.workloads.inkling_rpa import (
    inkling_rpa_inputs,
    inkling_rpa_reference,
    inkling_rpa_schedule,
)
from tpu_cake.workloads.matmul import matmul_inputs, matmul_reference, matmul_schedule


def test_matmul_reference_has_expected_shape() -> None:
    lhs, rhs = matmul_inputs()
    actual = matmul_reference(lhs, rhs)
    np.testing.assert_allclose(actual, lhs @ rhs)
    assert actual.shape == (16, 16)


def test_rpa_reference_is_deterministic_and_finite() -> None:
    inputs = inkling_rpa_inputs()
    first = inkling_rpa_reference(*inputs)
    second = inkling_rpa_reference(*inputs)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 4, 32)
    assert np.isfinite(first).all()


def test_schedule_text_and_hash_are_stable() -> None:
    for build in (matmul_schedule, inkling_rpa_schedule):
        first = build()
        second = build()
        assert canonical_module_text(first) == canonical_module_text(second)
        assert schedule_sha256(first) == schedule_sha256(second)
