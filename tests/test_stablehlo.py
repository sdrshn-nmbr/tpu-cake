from __future__ import annotations

import pytest

from tpu_cake.stablehlo import StableHloInspector


def test_inspector_counts_only_live_public_main_operations() -> None:
    stablehlo = """
module @fixture {
  func.func private @decoy() -> tensor<1xf32> {
    %input = stablehlo.constant dense<0.0> : tensor<1xf32>
    %dead = "stablehlo.all_gather"(%input) <{
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0]]> : tensor<1x1xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      use_global_device_ids
    }> : (tensor<1xf32>) -> tensor<1xf32>
    return %input : tensor<1xf32>
  }

  func.func public @main() -> tensor<1xf32> {
    %input = stablehlo.constant dense<0.0> : tensor<1xf32>
    %first = "stablehlo.all_gather"(%input) <{
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0]]> : tensor<1x1xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
      use_global_device_ids
    }> : (tensor<1xf32>) -> tensor<1xf32>
    %second = "stablehlo.all_gather"(%first) <{
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0]]> : tensor<1x1xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
      use_global_device_ids
    }> : (tensor<1xf32>) -> tensor<1xf32>
    %unused = "stablehlo.all_gather"(%input) <{
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0]]> : tensor<1x1xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>,
      use_global_device_ids
    }> : (tensor<1xf32>) -> tensor<1xf32>
    return %second : tensor<1xf32>
  }
}
"""

    inspector = StableHloInspector.parse(stablehlo)

    assert inspector.public_main_operation_count("stablehlo.all_gather") == 3
    assert inspector.live_public_main_operation_count("stablehlo.all_gather") == 2


def test_inspector_rejects_missing_public_main() -> None:
    stablehlo = """
module @fixture {
  func.func private @main() -> tensor<1xf32> {
    %value = stablehlo.constant dense<0.0> : tensor<1xf32>
    return %value : tensor<1xf32>
  }
}
"""

    with pytest.raises(ValueError, match="STABLEHLO_PUBLIC_MAIN_MISSING"):
        StableHloInspector.parse(stablehlo)
