import numpy as np

from tpu_cake.dtensor_interpreter import interpret_distributed_program
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_inputs,
    seqax_forward_reference,
)


def test_complete_seqax_forward_matches_independent_numerical_oracle() -> None:
    parameters = {
        "batch": 2,
        "sequence": 4,
        "model": 8,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 2,
        "head": 4,
        "layers": 2,
        "data_mesh": 1,
        "tensor_mesh": 1,
        "rope_max_timescale": 256,
    }
    inputs = seqax_forward_inputs(seed=9173, **parameters)
    expected = seqax_forward_reference(inputs, **parameters)
    (actual,) = interpret_distributed_program(
        seqax_forward_schedule(**parameters),
        inputs,
    )

    assert actual.shape == (2, 4, 16)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_seqax_oracle_inputs_are_seed_replayable() -> None:
    parameters = {
        "batch": 2,
        "sequence": 4,
        "model": 8,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 2,
        "head": 4,
        "layers": 2,
    }
    first = seqax_forward_inputs(seed=11, **parameters)
    second = seqax_forward_inputs(seed=11, **parameters)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_seqax_oracle_discriminates_a_wrong_rope_contract() -> None:
    parameters = {
        "batch": 2,
        "sequence": 4,
        "model": 8,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 2,
        "head": 4,
        "layers": 2,
        "data_mesh": 1,
        "tensor_mesh": 1,
        "rope_max_timescale": 256,
    }
    inputs = seqax_forward_inputs(seed=41, **parameters)
    expected = seqax_forward_reference(inputs, **parameters)
    wrong_schedule = seqax_forward_schedule(
        **{**parameters, "rope_max_timescale": 4_096}
    )
    (wrong,) = interpret_distributed_program(wrong_schedule, inputs)

    assert not np.allclose(wrong, expected, rtol=2e-3, atol=2e-3)
