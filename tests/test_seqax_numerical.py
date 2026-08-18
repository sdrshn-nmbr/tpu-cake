import ml_dtypes
import numpy as np
import pytest

from tpu_cake.seqax_numerical import (
    BF16_UNIT_ROUNDOFF,
    SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA,
    rounded_mathematical_silu_bf16,
    validate_strict_silu_stablehlo,
)


def test_mathematical_silu_reference_rounds_once_to_bf16() -> None:
    value = np.asarray(
        [-3.625, -2.125, -0.3046875, 0.0, 0.8828125, 2.125, 2.96875, 3.375],
        dtype=ml_dtypes.bfloat16,
    )

    actual = rounded_mathematical_silu_bf16(value)

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [
                -0.09423828125,
                -0.2265625,
                -0.12890625,
                0.0,
                0.625,
                1.8984375,
                2.828125,
                3.265625,
            ],
            dtype=ml_dtypes.bfloat16,
        ),
    )
    assert actual.dtype == np.dtype(ml_dtypes.bfloat16)
    assert BF16_UNIT_ROUNDOFF == 0.00390625
    assert SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA == "bf16-forward-numerical-v1"


def test_mathematical_silu_reference_rejects_non_bf16_or_nonfinite_input() -> None:
    with pytest.raises(TypeError, match="requires BF16"):
        rounded_mathematical_silu_bf16(np.asarray([1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="requires finite"):
        rounded_mathematical_silu_bf16(np.asarray([np.inf, np.nan], dtype=ml_dtypes.bfloat16))


def test_strict_silu_stablehlo_requires_barrier_dataflow_into_multiply() -> None:
    stablehlo = """module {
      func.func private @silu(tensor<1x4xbf16>) -> tensor<1x4xbf16>
      func.func @main(
        %arg0: tensor<1x4xbf16>, %other: tensor<1x4xbf16>
      ) -> tensor<1x4xbf16> {
        %1 = stablehlo.optimization_barrier %arg0 : tensor<1x4xbf16>
        %2 = func.call @silu(%1) : (tensor<1x4xbf16>) -> tensor<1x4xbf16>
        %3 = stablehlo.optimization_barrier %2 : tensor<1x4xbf16>
        %4 = stablehlo.multiply %other, %3 : tensor<1x4xbf16>
        return %4 : tensor<1x4xbf16>
      }
    }"""

    validate_strict_silu_stablehlo(stablehlo, expected_count=1)

    with pytest.raises(ValueError, match="input barrier"):
        validate_strict_silu_stablehlo(
            stablehlo.replace("func.call @silu(%1)", "func.call @silu(%arg0)"),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="result must feed only"):
        validate_strict_silu_stablehlo(
            stablehlo.replace("optimization_barrier %2", "optimization_barrier %arg0"),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="must feed exactly one"):
        validate_strict_silu_stablehlo(
            stablehlo.replace("multiply %other, %3", "multiply %other, %arg0"),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="result must feed only"):
        validate_strict_silu_stablehlo(
            stablehlo.replace(
                "return %4 : tensor<1x4xbf16>",
                "%5 = stablehlo.multiply %other, %2 : tensor<1x4xbf16>\n"
                "        return %4 : tensor<1x4xbf16>",
            ),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="must reach its function return"):
        validate_strict_silu_stablehlo(
            stablehlo.replace(
                "return %4 : tensor<1x4xbf16>",
                "%5 = stablehlo.add %4, %other : tensor<1x4xbf16>\n"
                "        %6 = stablehlo.multiply %other, %arg0 : tensor<1x4xbf16>\n"
                "        return %6 : tensor<1x4xbf16>",
            ),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="expected 2 calls"):
        validate_strict_silu_stablehlo(stablehlo, expected_count=2)


def test_strict_silu_stablehlo_scopes_ssa_values_per_function() -> None:
    function = """
      func.func @{name}(
        %arg0: tensor<1x4xbf16>, %other: tensor<1x4xbf16>
      ) -> tensor<1x4xbf16> {{
        %0 = stablehlo.optimization_barrier %arg0 : tensor<1x4xbf16>
        %1 = func.call @silu(%0) : (tensor<1x4xbf16>) -> tensor<1x4xbf16>
        %2 = stablehlo.optimization_barrier %1 : tensor<1x4xbf16>
        %3 = stablehlo.multiply %other, %2 : tensor<1x4xbf16>
        return %3 : tensor<1x4xbf16>
      }}
    """
    stablehlo = (
        """module {
      func.func private @silu(tensor<1x4xbf16>) -> tensor<1x4xbf16>
    """
        + function.format(name="first")
        + function.format(name="second")
        + "}"
    )

    validate_strict_silu_stablehlo(stablehlo, expected_count=2)
