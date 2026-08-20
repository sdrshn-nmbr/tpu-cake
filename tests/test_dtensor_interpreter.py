import ml_dtypes
import numpy as np
import pytest
from xdsl.dialects.builtin import bf16, f32
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.distributed_tensor import ElementwiseMaterialization
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.dtensor_interpreter import (
    execute_distributed_program_jax,
    interpret_distributed_program,
)
from tpu_cake.seqax_numerical import rounded_mathematical_silu_bf16
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
    np.testing.assert_array_equal(actual, expected)


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


def test_strict_typed_silu_rounds_once_from_a_float32_interior() -> None:
    value_type = tensor(bf16, (("F", 5),))
    builder = DistributedProgramBuilder("strict_silu", {}, (value_type,))
    result = builder.elementwise(
        builder.inputs[0],
        result=value_type,
        function="silu",
        materialization=ElementwiseMaterialization.STRICT_TYPED,
    )
    gate = np.asarray(
        [2.4375, 1.6484375, 0.625, -2.375, 2.953125],
        dtype=ml_dtypes.bfloat16,
    )

    (actual,) = execute_distributed_program_jax(builder.module(result), (gate,))

    np.testing.assert_array_equal(actual, rounded_mathematical_silu_bf16(gate))


def test_strict_typed_multiply_requires_bf16_and_rejects_other_operations() -> None:
    f32_type = tensor(f32, (("F", 4),))
    f32_builder = DistributedProgramBuilder("strict_f32_multiply", {}, (f32_type, f32_type))
    f32_result = f32_builder.elementwise(
        *f32_builder.inputs,
        result=f32_type,
        function="multiply",
        materialization=ElementwiseMaterialization.STRICT_TYPED,
    )
    with pytest.raises(VerifyException, match="requires BF16"):
        f32_builder.module(f32_result)

    bf16_type = tensor(bf16, (("F", 4),))
    add_builder = DistributedProgramBuilder("strict_add", {}, (bf16_type, bf16_type))
    add_result = add_builder.elementwise(
        *add_builder.inputs,
        result=bf16_type,
        function="add",
        materialization=ElementwiseMaterialization.STRICT_TYPED,
    )
    with pytest.raises(VerifyException, match="only supported for SiLU and multiply"):
        add_builder.module(add_result)


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
    wrong_schedule = seqax_forward_schedule(**{**parameters, "rope_max_timescale": 4_096})
    (wrong,) = interpret_distributed_program(wrong_schedule, inputs)

    assert not np.allclose(wrong, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("index", "dtype", "message"),
    (
        (0, np.int32, "input dtype int32 does not match uint32"),
        (1, np.uint8, "input dtype uint8 does not match bool"),
        (2, np.float64, "input dtype float64 does not match float32"),
    ),
)
def test_interpreter_rejects_inputs_that_violate_declared_dtypes(
    index: int, dtype: type[np.generic], message: str
) -> None:
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
    inputs = list(seqax_forward_inputs(seed=7, **parameters))
    inputs[index] = inputs[index].astype(dtype)

    with pytest.raises(ValueError, match=message):
        interpret_distributed_program(seqax_forward_schedule(**parameters), tuple(inputs))


def test_seqax_oracle_discriminates_swapped_normalization_scales() -> None:
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
    inputs = seqax_forward_inputs(seed=19, **parameters)
    expected = seqax_forward_reference(inputs, **parameters)
    swapped = (*inputs[:3], inputs[4], inputs[3], *inputs[5:])
    (wrong,) = interpret_distributed_program(seqax_forward_schedule(**parameters), swapped)

    assert not np.allclose(wrong, expected, rtol=2e-3, atol=2e-3)
