import numpy as np
import pytest

from tpu_cake.rpa_runner import fused_rpa_outputs_pass, validate_fused_rpa_run_protocol


def test_fused_rpa_run_protocol_is_predeclared() -> None:
    validate_fused_rpa_run_protocol(seed=97, warmup_iterations=5, measured_iterations=50)

    with pytest.raises(ValueError, match="predeclared benchmark protocol"):
        validate_fused_rpa_run_protocol(
            seed=97,
            warmup_iterations=5,
            measured_iterations=20,
        )

    with pytest.raises(ValueError, match="predeclared benchmark protocol"):
        validate_fused_rpa_run_protocol(
            seed=19,
            warmup_iterations=5,
            measured_iterations=50,
        )


def test_fused_rpa_cache_must_match_exactly() -> None:
    output = np.ones((2, 2), dtype=np.float32)
    cache = np.ones((2, 2), dtype=np.float32)
    perturbed_cache = cache.copy()
    perturbed_cache[0, 0] += 0.001

    assert fused_rpa_outputs_pass(
        (output + 0.001, cache),
        (output, cache),
        output_atol=0.02,
        output_rtol=0.02,
    )
    assert not fused_rpa_outputs_pass(
        (output, perturbed_cache),
        (output, cache),
        output_atol=0.02,
        output_rtol=0.02,
    )
