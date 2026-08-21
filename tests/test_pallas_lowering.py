from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from tpu_cake import pallas_lowering
from tpu_cake.canonical import canonical_text
from tpu_cake.dialects.tpu_schedule import (
    CollectiveImplementation,
    CollectiveKind,
    CollectiveOp,
    KernelOp,
)
from tpu_cake.frontend import schedule_sha256
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
    assert first.collective_link_bandwidths == (
        ("link:0-1", 600_000_000_000),
        ("link:1-2", 600_000_000_000),
        ("link:2-3", 600_000_000_000),
    )
    assert first.source_sha256() == second.source_sha256()
    source = first.render_executable_source()
    compile(source, "lowered_pallas.py", "exec")
    for required in ("PallasMatmulPlan", "PALLAS_EXECUTION_SCHEMA", "PLAN.build"):
        assert required in source


def test_legacy_physical_and_pallas_identities_remain_exact() -> None:
    physical = lower_distributed_matmul(distributed_matmul_schedule())
    plan = lower_physical_matmul_to_pallas(physical)

    assert schedule_sha256(physical) == (
        "395c5e60a6ae43e2c86b61b06271666010d79d103175cb43a4c01c9fb22a8819"
    )
    assert plan.source_sha256() == (
        "1ca9459450c4e0a0e332ffefa2899eb836988914fd466abd91454f0e29772033"
    )
    assert plan.collective_implementation is None


def test_pallas_native_reduce_scatter_has_a_bound_resource_contract(tmp_path) -> None:
    physical = lower_distributed_matmul(
        distributed_matmul_schedule(mesh_size=8, m=128, k=1024, n=1024),
        tile=MatmulTile(128, 128),
        collective_implementation=(CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING),
    )
    plan = lower_physical_matmul_to_pallas(physical)

    assert plan.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING
    assert plan.output_local_shape == (128, 128)
    assert plan.collective_hbm_scratch_bytes == 131_072
    assert plan.collective_accumulator_vmem_bytes == 32_768
    assert plan.collective_dma_semaphore_count == 5
    assert plan.collective_capacity_semaphore_count == 2
    assert plan.collective_startup_semaphore_count == 1
    assert plan.collective_startup_barrier_phases == 2
    assert plan.collective_remote_half_output_copy_count == 17
    assert plan.collective_remote_payload_bytes == 557_056
    assert plan.collective_remote_bidirectional_endpoint_bytes == 1_114_112
    source = plan.render_executable_source()
    compile(source, "lowered_pallas.py", "exec")
    assert "native-collective-plan-v3" in source
    assert "COLLECTIVE_HBM_SCRATCH_BYTES = 131072" in source

    physical_path = tmp_path / "physical.xdsl"
    pallas_path = tmp_path / "lowered_pallas.py"
    physical_path.write_text(canonical_text(physical))
    pallas_path.write_text(source)
    replayed = validate_saved_pallas_plan(
        physical_path,
        pallas_path,
        schedule_sha256=plan.schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
    )
    assert replayed == plan


def test_pallas_native_reduce_scatter_rejects_an_unusable_output_block() -> None:
    with pytest.raises(
        VerifyException,
        match="rows divisible by 16 and columns divisible by 128",
    ):
        lower_distributed_matmul(
            distributed_matmul_schedule(),
            collective_implementation=(CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING),
        )


def test_saved_native_collective_resource_contract_is_exact(tmp_path) -> None:
    physical = lower_distributed_matmul(
        distributed_matmul_schedule(mesh_size=8, m=128, k=1024, n=1024),
        tile=MatmulTile(128, 128),
        collective_implementation=(CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING),
    )
    plan = lower_physical_matmul_to_pallas(physical)
    physical_path = tmp_path / "physical.xdsl"
    pallas_path = tmp_path / "lowered_pallas.py"
    physical_path.write_text(canonical_text(physical))
    pallas_path.write_text(
        plan.render_executable_source().replace(
            "COLLECTIVE_DMA_SEMAPHORE_COUNT = 5",
            "COLLECTIVE_DMA_SEMAPHORE_COUNT = 4",
        )
    )

    with pytest.raises(ValueError, match="SAVED_PALLAS_SOURCE_PLAN_MISMATCH"):
        validate_saved_pallas_plan(
            physical_path,
            pallas_path,
            schedule_sha256=plan.schedule_sha256,
            pallas_source_sha256=hashlib.sha256(pallas_path.read_bytes()).hexdigest(),
        )


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


def test_pallas_lowering_rejects_a_non_tpu_target() -> None:
    physical = lower_distributed_matmul(distributed_matmul_schedule())
    kernel = next(operation for operation in physical.walk() if isinstance(operation, KernelOp))
    kernel.properties["target"] = StringAttr("gpu")
    physical.verify()

    with pytest.raises(UnsupportedLoweringError, match="does not support target 'gpu'"):
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


def test_current_standalone_rendering_replays(tmp_path) -> None:
    physical = lower_distributed_matmul(distributed_matmul_schedule())
    plan = lower_physical_matmul_to_pallas(physical)
    physical_path = tmp_path / "physical.xdsl"
    pallas_path = tmp_path / "lowered_pallas.py"
    physical_path.write_text(canonical_text(physical))
    pallas_path.write_text(plan.render_source())

    replayed = validate_saved_pallas_plan(
        physical_path,
        pallas_path,
        schedule_sha256=plan.schedule_sha256,
        pallas_source_sha256=hashlib.sha256(pallas_path.read_bytes()).hexdigest(),
    )
    assert replayed == plan


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


def test_pallas_native_reduce_scatter_interpreter_matches_oracle() -> None:
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    environment["TPU_CAKE_CONTROL_NATIVE_COLLECTIVE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "tpu_cake.control"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)
    assert result["passed"] is True
    assert result["device_count"] == 8
    assert result["collective_implementation"] == "pallas_bidirectional_ring"
    assert result["maximum_absolute_error"] <= 1e-4
    assert result["output_shape"] == [128, 1024]
    assert result["output_dtype"] == "float32"


def test_pallas_native_reduce_scatter_addresses_its_ring_inside_a_2d_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pallas_lowering.lax,
        "axis_index",
        lambda name: {"d": 1, "t": 2}[name],
    )

    assert pallas_lowering._ring_device_id(
        3,
        axis_name="t",
        mesh_axis_names=("d", "t"),
    ) == (1, 3)
    with pytest.raises(ValueError, match="exactly once"):
        pallas_lowering._ring_device_id(
            3,
            axis_name="t",
            mesh_axis_names=("d", "d"),
        )
