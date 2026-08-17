from __future__ import annotations

import json
import os
import subprocess
import sys

from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def test_physical_matmul_lowers_to_stable_pallas_plan() -> None:
    physical = lower_distributed_matmul(distributed_matmul_schedule())
    first = lower_physical_matmul_to_pallas(physical)
    second = lower_physical_matmul_to_pallas(physical)
    assert first == second
    assert first.global_lhs_shape == (16, 32)
    assert first.global_rhs_shape == (32, 16)
    assert first.global_output_shape == (16, 16)
    assert first.source_sha256() == second.source_sha256()
    source = first.render_source()
    compile(source, "lowered_pallas.py", "exec")
    for required in ("pallas_call", "BlockSpec", "shard_map", "psum_scatter", "check_vma=False"):
        assert required in source


def test_tile_choice_is_part_of_the_canonical_physical_schedule() -> None:
    distributed = distributed_matmul_schedule(mesh_size=4, m=256, k=512, n=256)
    whole = lower_physical_matmul_to_pallas(lower_distributed_matmul(distributed))
    tiled = lower_physical_matmul_to_pallas(
        lower_distributed_matmul(distributed, tile=MatmulTile(128, 128))
    )
    assert (tiled.tile_m, tiled.tile_k, tiled.tile_n) == (128, 128, 128)
    assert tiled.schedule_sha256 != whole.schedule_sha256
    assert tiled.source_sha256() != whole.source_sha256()


def test_distributed_pallas_interpreter_matches_oracle() -> None:
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-m", "tpu_cake.control"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)
    assert result["passed"] is True
    assert result["device_count"] == 4
    assert result["maximum_absolute_error"] <= 1e-4
    assert result["output_shape"] == [16, 16]
    assert result["output_dtype"] == "float32"
