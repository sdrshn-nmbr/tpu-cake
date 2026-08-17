import numpy as np

from tpu_cake.workloads.inkling_rpa import (
    FusedRpaOracleMutation,
    inkling_fused_rpa_inputs,
    inkling_fused_rpa_reference,
)


def test_fused_rpa_oracle_is_seed_replayable_and_updates_packed_cache() -> None:
    first_inputs = inkling_fused_rpa_inputs(seed=83)
    second_inputs = inkling_fused_rpa_inputs(seed=83)
    for first, second in zip(first_inputs, second_inputs, strict=True):
        np.testing.assert_array_equal(first, second)

    output, updated_cache = inkling_fused_rpa_reference(first_inputs)

    assert output.shape == (4, 4, 32)
    assert str(output.dtype) == "bfloat16"
    assert updated_cache.shape == (32, 16, 2, 2, 32)
    assert str(updated_cache.dtype) == "bfloat16"
    assert not np.array_equal(updated_cache, first_inputs[3])


def test_fused_rpa_oracle_discriminates_real_kernel_failure_classes() -> None:
    inputs = inkling_fused_rpa_inputs(seed=97)
    expected_output, expected_cache = inkling_fused_rpa_reference(inputs)

    for mutation in FusedRpaOracleMutation:
        wrong_output, wrong_cache = inkling_fused_rpa_reference(
            inputs,
            mutation=mutation,
        )
        assert not (
            np.allclose(wrong_output, expected_output, rtol=0.02, atol=0.02)
            and np.allclose(wrong_cache, expected_cache, rtol=0.02, atol=0.02)
        ), mutation
