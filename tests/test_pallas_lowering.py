from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest
from xdsl.dialects.builtin import IntAttr, StringAttr

from tpu_cake.canonical import canonical_text
from tpu_cake.dialects.tpu_schedule import (
    CollectiveKind,
    CollectiveOp,
    KernelOp,
)
from tpu_cake.lowering import (
    MatmulTile,
    UnsupportedLoweringError,
    lower_distributed_matmul,
)
from tpu_cake.pallas_lowering import (
    lower_physical_matmul_to_pallas,
    validate_saved_pallas_plan,
)
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
    source = first.render_executable_source()
    compile(source, "lowered_pallas.py", "exec")
    for required in ("PallasMatmulPlan", "PALLAS_EXECUTION_SCHEMA", "PLAN.build"):
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


def test_pallas_lowering_rejects_an_unrepresented_collective() -> None:
    physical = lower_distributed_matmul(distributed_matmul_schedule())
    kernel = next(operation for operation in physical.walk() if isinstance(operation, KernelOp))
    collective = next(
        operation for operation in physical.walk() if isinstance(operation, CollectiveOp)
    )
    extra = CollectiveOp(
        collective.destination,
        collective.destination,
        stage=4,
        kind=CollectiveKind.ALL_REDUCE,
        mesh_axis=collective.mesh_axis.data,
        group_size=collective.group_size.data,
        reducer="sum",
    )
    assert collective.parent is not None
    collective.parent.insert_op_after(extra, collective)
    kernel.properties["ici_link_count"] = IntAttr(2)
    physical.verify()

    with pytest.raises(UnsupportedLoweringError, match="exact supported"):
        lower_physical_matmul_to_pallas(physical)


def test_pallas_lowering_rejects_a_non_sum_reduce_scatter() -> None:
    physical = lower_distributed_matmul(distributed_matmul_schedule())
    collective = next(
        operation for operation in physical.walk() if isinstance(operation, CollectiveOp)
    )
    collective.properties["reducer"] = StringAttr("max")
    physical.verify()

    with pytest.raises(UnsupportedLoweringError, match="sum reduction only"):
        lower_physical_matmul_to_pallas(physical)


def test_saved_physical_tile_and_pallas_rendering_are_bound(tmp_path) -> None:
    physical = lower_distributed_matmul(
        distributed_matmul_schedule(mesh_size=4, m=256, k=512, n=256),
        tile=MatmulTile(128, 128),
    )
    plan = lower_physical_matmul_to_pallas(physical)
    physical_path = tmp_path / "physical.xdsl"
    pallas_path = tmp_path / "lowered_pallas.py"
    physical_path.write_text(canonical_text(physical))
    pallas_path.write_text(plan.render_executable_source())

    replayed = validate_saved_pallas_plan(
        physical_path,
        pallas_path,
        schedule_sha256=plan.schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
    )
    assert (replayed.tile_m, replayed.tile_n) == (128, 128)

    pallas_path.write_text(pallas_path.read_text().replace("TILE_M = 128", "TILE_M = 64"))
    forged_hash = hashlib.sha256(pallas_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SAVED_PALLAS_SOURCE_PLAN_MISMATCH"):
        validate_saved_pallas_plan(
            physical_path,
            pallas_path,
            schedule_sha256=plan.schedule_sha256,
            pallas_source_sha256=forged_hash,
        )


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
