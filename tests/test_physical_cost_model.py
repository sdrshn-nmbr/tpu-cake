from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, bf16, f32
from xdsl.ir import Block
from xdsl.utils.exceptions import VerifyException

from tpu_cake.cli import (
    _estimate_physical_cost,
    _estimate_physical_latency,
    _verify_physical_cost,
    _verify_physical_latency,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import (
    CollectiveImplementation,
    CollectiveKind,
    KernelOp,
    LinkAttr,
    MemorySpace,
    MxuEinsumOp,
    MxuMatmulOp,
    Ownership,
    PipelineLoopOp,
    PipelineYieldOp,
    TopologyAttr,
    TransferPlanAttr,
    TransferRouteAttr,
    rectilinear_topology,
)
from tpu_cake.frontend import KernelBuilder, buffer, canonical_module_text, schedule_sha256
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.metrics import MetricSource
from tpu_cake.physical_cost_model import (
    PhysicalCollectiveLatencyCalibration,
    PhysicalImbalanceRecord,
    PhysicalKernelLatencyReport,
    PhysicalKernelResourceReport,
    UnsupportedPhysicalCostModelError,
    analyze_physical_kernel,
    analyze_physical_kernel_latency,
    physical_schedule_source,
    tpu7x_collective_latency_calibration,
    validate_physical_kernel_latency_report,
    validate_physical_kernel_report,
    write_physical_kernel_latency_report,
    write_physical_kernel_report,
)
from tpu_cake.seqax_cost_model import estimate_seqax_forward
from tpu_cake.seqax_pallas_lowering import _einsum_tiles
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads import inkling_rpa_schedule
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule
from tpu_cake.workloads.seqax_forward import (
    SeqaxNumericalSemantics,
    SeqaxResidualNormStrategy,
    seqax_forward_schedule,
)

SMALL_SEQAX = {
    "batch": 2,
    "sequence": 4,
    "model": 8,
    "vocabulary": 16,
    "feed_forward": 16,
    "query_groups": 2,
    "key_value_heads": 4,
    "head": 4,
    "layers": 2,
    "data_mesh": 2,
    "tensor_mesh": 4,
    "rope_max_timescale": 256,
}


def _report(module):
    return analyze_physical_kernel(
        module,
        hardware=tpu7x_tensorcore_rates(),
    )


def _small_seqax_physical():
    return lower_seqax_forward_to_physical(seqax_forward_schedule(**SMALL_SEQAX)).module


def test_complete_seqax_physical_schedule_has_replayable_resource_authority() -> None:
    module = _small_seqax_physical()
    report = _report(module)

    assert report.physical_schedule_sha256 == schedule_sha256(module)
    assert report.execution_authority == "static-declared-schedule-only"
    assert report.mesh_axes == (("d", 2), ("t", 4))
    assert report.device_count == 8
    assert len(report.mxu_regions) == 17
    assert tuple(value.input_dtype for value in report.mxu_regions).count("bf16") == 15
    assert tuple(value.input_dtype for value in report.mxu_regions).count("f32") == 2
    priced_bf16_flops = sum(
        value.total_flops for value in report.mxu_regions if value.input_dtype == "bf16"
    )
    assert (
        next(
            metric
            for metric in report.metrics
            if metric.name == "physical_priced_bf16_mxu_flops_per_device"
        ).quantity.value
        == priced_bf16_flops
    )
    assert any(value.startswith("mxu_flops[f32]=") for value in report.unpriced_work)
    assert sum(value.executions for value in report.vector_work) == 77
    assert len(report.collectives) == 34
    assert report.operation_executions == (
        ("tpu_schedule.collective", 34),
        ("tpu_schedule.dma_start", 14),
        ("tpu_schedule.mxu_einsum", 17),
        ("tpu_schedule.vector_compute", 77),
    )
    assert report.memory.external_hbm_input_footprint_bytes_per_device == 1320
    assert report.memory.external_hbm_output_footprint_bytes_per_device == 64
    assert report.memory.peak_live_vmem_bytes_per_device == 2168
    assert all(value.per_link_traffic_derivable is False for value in report.collectives)
    assert all(value.kind.value in {"all_gather", "reduce_scatter"} for value in report.collectives)
    assert {metric.sources[0].artifact_sha256 for metric in report.metrics} == {
        report.physical_schedule_sha256
    }
    assert all(metric.kind.value == "estimated" for metric in report.metrics)

    validate_physical_kernel_report(
        report,
        module=module,
        hardware=tpu7x_tensorcore_rates(),
    )


def test_physical_counts_cross_check_the_same_distributed_program() -> None:
    distributed = seqax_forward_schedule(**SMALL_SEQAX)
    physical = lower_seqax_forward_to_physical(distributed).module
    physical_report = _report(physical)
    distributed_hash = schedule_sha256(distributed)
    distributed_report = estimate_seqax_forward(
        distributed,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=distributed_hash,
            artifact_path="distributed.xdsl",
            tool="tpu-cake",
            field="canonical-distributed-schedule-xdsl",
        ),
    )

    assert sum(value.total_flops for value in physical_report.mxu_regions) == (
        distributed_report.counts.mxu_flops_per_device
    )
    assert physical_report.memory.explicit_hbm_dma_read_bytes_per_device == (
        distributed_report.counts.minimum_hbm_read_bytes_per_device
    )
    assert physical_report.memory.explicit_hbm_dma_write_bytes_per_device == (
        distributed_report.counts.minimum_hbm_write_bytes_per_device
    )
    assert (
        sum(
            value.total_ring_equivalent_bidirectional_bytes_per_device
            for value in physical_report.collectives
        )
        == distributed_report.counts.ici_bidirectional_bytes_per_device
    )


def test_sharded_rms_physical_collectives_match_distributed_cost_authority() -> None:
    parameters = {
        **SMALL_SEQAX,
        "model": 256,
        "sequence": 1,
        "layers": 1,
        "numerical_semantics": SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
        "residual_norm_strategy": SeqaxResidualNormStrategy.SHARDED_RMS,
    }
    distributed = seqax_forward_schedule(**parameters)
    physical = lower_seqax_forward_to_physical(distributed).module
    physical_report = _report(physical)
    distributed_report = estimate_seqax_forward(
        distributed,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=schedule_sha256(distributed),
            artifact_path="distributed.xdsl",
            tool="tpu-cake",
            field="canonical-distributed-schedule-xdsl",
        ),
    )

    assert (
        sum(value.kind is CollectiveKind.ALL_GATHER for value in physical_report.collectives) == 14
    )
    assert (
        sum(value.kind is CollectiveKind.ALL_REDUCE for value in physical_report.collectives) == 3
    )
    assert (
        sum(value.kind is CollectiveKind.REDUCE_SCATTER for value in physical_report.collectives)
        == 3
    )
    assert (
        sum(
            value.total_ring_equivalent_bidirectional_bytes_per_device
            for value in physical_report.collectives
        )
        == distributed_report.counts.ici_bidirectional_bytes_per_device
    )


def test_residual_all_reduce_physical_collectives_are_exactly_accounted() -> None:
    parameters = {
        **SMALL_SEQAX,
        "model": 256,
        "sequence": 1,
        "layers": 1,
        "numerical_semantics": SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
    }
    standard = _report(lower_seqax_forward_to_physical(seqax_forward_schedule(**parameters)).module)
    candidate = _report(
        lower_seqax_forward_to_physical(
            seqax_forward_schedule(
                **parameters,
                residual_norm_strategy=SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
            )
        ).module
    )

    assert len(standard.collectives) == 20
    assert len(candidate.collectives) == 18
    assert sum(value.kind is CollectiveKind.ALL_GATHER for value in candidate.collectives) == 15
    assert sum(value.kind is CollectiveKind.ALL_REDUCE for value in candidate.collectives) == 2
    assert sum(value.kind is CollectiveKind.REDUCE_SCATTER for value in candidate.collectives) == 1
    assert (
        sum(
            value.total_ring_equivalent_bidirectional_bytes_per_device
            for value in standard.collectives
        )
        == 34_048
    )
    assert (
        sum(
            value.total_ring_equivalent_bidirectional_bytes_per_device
            for value in candidate.collectives
        )
        == 35_584
    )
    assert [
        (value.function, value.executions)
        for value in candidate.vector_work
        if value.function in {"residual_inject", "shard_extract"}
    ] == [
        ("residual_inject", 1),
        ("shard_extract", 1),
        ("residual_inject", 1),
        ("shard_extract", 1),
    ]


def test_exact_tpu7x_collective_latency_distinguishes_standard_and_sharded_rms() -> None:
    parameters = {
        **SMALL_SEQAX,
        "model": 256,
        "sequence": 1,
        "layers": 1,
        "numerical_semantics": SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
    }
    standard_module = lower_seqax_forward_to_physical(seqax_forward_schedule(**parameters)).module
    candidate_module = lower_seqax_forward_to_physical(
        seqax_forward_schedule(
            **parameters,
            residual_norm_strategy=SeqaxResidualNormStrategy.SHARDED_RMS,
        )
    ).module
    standard = _report(standard_module)
    candidate = _report(candidate_module)
    calibration = tpu7x_collective_latency_calibration()
    standard_latency = analyze_physical_kernel_latency(
        standard,
        module=standard_module,
        calibration=calibration,
    )
    candidate_latency = analyze_physical_kernel_latency(
        candidate,
        module=candidate_module,
        calibration=calibration,
    )

    assert len(calibration.points) == 10
    assert all(point.positive_delta_rounds == 21 for point in calibration.points)
    assert len(standard_latency.operations) == 20
    assert len(candidate_latency.operations) == 20
    assert standard_latency.collective_measured_serial_scenario_ns == Decimal("192608.9062500")
    assert candidate_latency.collective_measured_serial_scenario_ns == Decimal("186731.7656250")
    assert (
        candidate_latency.collective_measured_serial_scenario_ns
        - standard_latency.collective_measured_serial_scenario_ns
        == Decimal("-5877.1406250")
    )
    assert standard_latency.collective_latency_excess_scenario_ns > 192_000
    assert candidate_latency.collective_latency_excess_scenario_ns > 186_000
    assert standard_latency.predicted_limiting_priced_resource == "ici"
    assert candidate_latency.predicted_limiting_priced_resource == "ici"
    assert len(standard_latency.metrics) == 7
    assert all(metric.quantity.unit.value == "ns" for metric in standard_latency.metrics)
    assert any(
        source.artifact_sha256 == calibration.archive_sha256
        for metric in standard_latency.metrics
        for source in metric.sources
    )

    forged = standard.model_copy(
        update={
            "priced_compute_time_floor_ns": standard.priced_compute_time_floor_ns + 1,
        }
    )
    with pytest.raises(ValueError, match="PHYSICAL_KERNEL_RESOURCE_REPORT_REPLAY_MISMATCH"):
        analyze_physical_kernel_latency(
            forged,
            module=standard_module,
            calibration=calibration,
        )

    wildcard_operation = next(
        operation
        for operation in standard_latency.operations
        if operation.kind is CollectiveKind.ALL_GATHER
        and any(
            point.mesh_axis == operation.mesh_axis
            and point.group_size == operation.group_size
            and point.kind is operation.kind
            and point.reducer == operation.reducer
            and point.source_dtype_authority == "payload-only"
            and point.payload_bytes_per_device == operation.payload_bytes_per_device
            for point in calibration.points
        )
    )
    wildcard = next(
        point
        for point in calibration.points
        if point.mesh_axis == wildcard_operation.mesh_axis
        and point.group_size == wildcard_operation.group_size
        and point.kind is wildcard_operation.kind
        and point.reducer == wildcard_operation.reducer
        and point.source_dtype_authority == "payload-only"
        and point.payload_bytes_per_device == wildcard_operation.payload_bytes_per_device
    )
    exact = wildcard.model_copy(update={"source_dtype_authority": wildcard_operation.source_dtype})
    ambiguous = PhysicalCollectiveLatencyCalibration.model_validate(
        calibration.model_copy(
            update={
                "points": tuple(
                    sorted(
                        (*calibration.points, exact),
                        key=lambda point: (
                            point.mesh_axis,
                            point.group_size,
                            point.kind.value,
                            point.reducer,
                            point.source_dtype_authority,
                            point.payload_bytes_per_device,
                        ),
                    )
                )
            }
        ).model_dump(mode="python", exclude_computed_fields=True)
    )
    with pytest.raises(UnsupportedPhysicalCostModelError, match="matches=2"):
        analyze_physical_kernel_latency(
            standard,
            module=standard_module,
            calibration=ambiguous,
        )


def test_collective_latency_requires_an_exact_measured_point() -> None:
    module = _small_seqax_physical()
    report = _report(module)

    with pytest.raises(
        UnsupportedPhysicalCostModelError,
        match="exactly one matching point",
    ):
        analyze_physical_kernel_latency(
            report,
            module=module,
            calibration=tpu7x_collective_latency_calibration(),
        )


def test_collective_latency_rejects_an_unmeasured_reducer() -> None:
    external = buffer(
        (1,),
        "X",
        f32,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    )
    builder = KernelBuilder(
        "unmeasured_reducer",
        "tpu7x",
        (external,),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        mesh={"d": 2, "t": 4},
        interconnect_bandwidth_bytes_per_second={
            "d": 600_000_000_000,
            "t": 600_000_000_000,
        },
        argument_modes=("input",),
    )
    source = builder.alloc(
        buffer((1,), "X", f32, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        "source",
    )
    destination = builder.alloc(
        buffer((1,), "X", f32, memory=MemorySpace.VMEM, lifetime=(2, 2)),
        "destination",
    )
    inbound = builder.dma_start(builder.inputs[0], source, builder.semaphore(), stage=0)
    builder.dma_wait(inbound, stage=1)
    builder.collective(
        source,
        destination,
        stage=2,
        kind=CollectiveKind.ALL_REDUCE,
        mesh_axis="t",
        group_size=4,
        reducer="max",
    )
    module = builder.module()

    with pytest.raises(UnsupportedPhysicalCostModelError, match="reducer=max"):
        analyze_physical_kernel_latency(
            _report(module),
            module=module,
            calibration=tpu7x_collective_latency_calibration(),
        )


def test_collective_latency_rejects_an_unmeasured_reduction_dtype() -> None:
    external = buffer(
        (2,),
        "X",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    )
    builder = KernelBuilder(
        "unmeasured_reduction_dtype",
        "tpu7x",
        (external,),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        mesh={"d": 2, "t": 4},
        interconnect_bandwidth_bytes_per_second={
            "d": 600_000_000_000,
            "t": 600_000_000_000,
        },
        argument_modes=("input",),
    )
    source = builder.alloc(
        buffer((2,), "X", bf16, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        "source",
    )
    destination = builder.alloc(
        buffer((2,), "X", bf16, memory=MemorySpace.VMEM, lifetime=(2, 2)),
        "destination",
    )
    inbound = builder.dma_start(builder.inputs[0], source, builder.semaphore(), stage=0)
    builder.dma_wait(inbound, stage=1)
    builder.collective(
        source,
        destination,
        stage=2,
        kind=CollectiveKind.ALL_REDUCE,
        mesh_axis="t",
        group_size=4,
        reducer="sum",
    )
    module = builder.module()

    with pytest.raises(UnsupportedPhysicalCostModelError, match="source_dtype=bf16"):
        analyze_physical_kernel_latency(
            _report(module),
            module=module,
            calibration=tpu7x_collective_latency_calibration(),
        )


def test_collective_latency_report_writes_once_and_replays(tmp_path: Path) -> None:
    parameters = {
        **SMALL_SEQAX,
        "model": 256,
        "sequence": 1,
        "layers": 1,
        "numerical_semantics": SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
    }
    module = lower_seqax_forward_to_physical(seqax_forward_schedule(**parameters)).module
    resource = _report(module)
    calibration = tpu7x_collective_latency_calibration()
    path = tmp_path / "latency.json"
    saved = write_physical_kernel_latency_report(
        path,
        module=module,
        resource_report=resource,
        calibration=calibration,
    )

    replayed = PhysicalKernelLatencyReport.model_validate_json(path.read_text())
    assert replayed == saved
    validate_physical_kernel_latency_report(
        replayed,
        module=module,
        resource_report=resource,
        calibration=calibration,
    )
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        write_physical_kernel_latency_report(
            path,
            module=module,
            resource_report=resource,
            calibration=calibration,
        )
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        validate_physical_kernel_latency_report(
            replayed.model_copy(
                update={
                    "collective_measured_serial_scenario_ns": (
                        replayed.collective_measured_serial_scenario_ns + 1
                    )
                }
            ),
            module=module,
            resource_report=resource,
            calibration=calibration,
        )


def test_tpu7x_collective_latency_external_contract_is_canonical() -> None:
    path = Path(__file__).parents[1] / "contracts" / "tpu7x-collective-latency-v1.json"
    saved = PhysicalCollectiveLatencyCalibration.model_validate_json(path.read_text())

    assert saved == tpu7x_collective_latency_calibration()
    assert saved.calibration_id == (
        "a1c957efbd5810047c748ad7fc1b8bca5cd2daa80f373279a81e8263a9e299c8"
    )


def test_public_cli_derives_and_replays_physical_latency(tmp_path: Path) -> None:
    parameters = {
        **SMALL_SEQAX,
        "model": 256,
        "sequence": 1,
        "layers": 1,
        "numerical_semantics": SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
    }
    module = lower_seqax_forward_to_physical(seqax_forward_schedule(**parameters)).module
    schedule = tmp_path / "physical.xdsl"
    report = tmp_path / "physical-latency.json"
    calibration = Path(__file__).parents[1] / "contracts" / "tpu7x-collective-latency-v1.json"
    schedule.write_text(canonical_module_text(module))

    assert _estimate_physical_latency(schedule, calibration, report) == 0
    assert _verify_physical_latency(report, schedule, calibration) == 0
    relocated = tmp_path / "relocated" / "renamed-schedule.xdsl"
    relocated.parent.mkdir()
    relocated.write_text(schedule.read_text())
    assert _verify_physical_latency(report, relocated, calibration) == 0

    saved = PhysicalKernelLatencyReport.model_validate_json(report.read_text())
    report.write_text(
        saved.model_copy(
            update={
                "assumptions": (*saved.assumptions, "forged assumption"),
            }
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        _verify_physical_latency(report, schedule, calibration)

    wrong_calibration = tmp_path / "wrong-calibration.json"
    canonical = PhysicalCollectiveLatencyCalibration.model_validate_json(calibration.read_text())
    wrong_calibration.write_text(
        canonical.model_copy(update={"warmups": canonical.warmups + 1}).model_dump_json(
            exclude_computed_fields=True
        )
    )
    with pytest.raises(ValueError, match="EXTERNAL_CONTRACT_MISMATCH"):
        _verify_physical_latency(report, schedule, wrong_calibration)


def test_legal_tile_schedule_changes_physical_cost_identity_and_tile_grid() -> None:
    parameters = {**SMALL_SEQAX, "model": 256, "sequence": 1, "layers": 1}
    distributed = seqax_forward_schedule(**parameters)
    full = lower_seqax_forward_to_physical(distributed).module
    full_tiles = _einsum_tiles(full)
    split_tiles = tuple(
        (
            tile_m,
            128 if tile_k > 128 and tile_k % 128 == 0 else tile_k,
            128 if tile_n > 128 and tile_n % 128 == 0 else tile_n,
        )
        for tile_m, tile_k, tile_n in full_tiles
    )
    split = lower_seqax_forward_to_physical(
        distributed,
        einsum_tiles=split_tiles,
    ).module
    full_report = _report(full)
    split_report = _report(split)

    assert sum(left != right for left, right in zip(full_tiles, split_tiles, strict=True)) == 7
    assert full_report.physical_schedule_sha256 != split_report.physical_schedule_sha256
    assert sum(value.total_flops for value in full_report.mxu_regions) == sum(
        value.total_flops for value in split_report.mxu_regions
    )
    assert full_report.memory.external_hbm_input_footprint_bytes_per_device == (
        split_report.memory.external_hbm_input_footprint_bytes_per_device
    )
    assert full_report.priced_ici_time_scenario_ns == split_report.priced_ici_time_scenario_ns
    assert sum(value.total_tile_programs for value in split_report.mxu_regions) > sum(
        value.total_tile_programs for value in full_report.mxu_regions
    )
    assert max(value.accumulator_scratch_bytes for value in split_report.mxu_regions) < max(
        value.accumulator_scratch_bytes for value in full_report.mxu_regions
    )


def test_views_alias_storage_but_charge_their_exact_dma_payload() -> None:
    external_input = buffer(
        (8, 8),
        "MI NI",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 3),
    )
    external_output = buffer(
        (4, 8),
        "MO NO",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 3),
    )
    builder = KernelBuilder(
        "view_cost",
        "tpu7x",
        (external_input, external_output),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        argument_modes=("input", "output"),
    )
    allocation = builder.alloc(
        buffer(
            (8, 8),
            "MI NI",
            bf16,
            memory=MemorySpace.VMEM,
            lifetime=(0, 3),
        ),
        "allocation",
    )
    inbound = builder.dma_start(builder.inputs[0], allocation, builder.semaphore(), stage=0)
    builder.dma_wait(inbound, stage=1)
    view = builder.view(
        allocation,
        buffer(
            (4, 8),
            "MO NO",
            bf16,
            memory=MemorySpace.VMEM,
            lifetime=(2, 3),
        ),
        offsets=(0, 0),
        alias_group="allocation",
    )
    outbound = builder.dma_start(view, builder.inputs[1], builder.semaphore(), stage=2)
    builder.dma_wait(outbound, stage=3)
    report = _report(builder.module())

    assert report.memory.allocated_vmem_bytes_per_device == 128
    assert report.memory.peak_live_vmem_bytes_per_device == 128
    assert report.memory.explicit_hbm_dma_read_bytes_per_device == 128
    assert report.memory.explicit_hbm_dma_write_bytes_per_device == 64


def _pipeline_module(
    *,
    initiation_interval: int,
    accumulator_rotations: int,
    use_einsum: bool = False,
    unrelated_local: bool = False,
    tile_m: int = 8,
    vmem_capacity_bytes: int = 4096,
):
    external = (
        buffer(
            (8, 8),
            "M K",
            bf16,
            memory=MemorySpace.HBM,
            ownership=Ownership.EXTERNAL,
            lifetime=(0, 3),
        ),
        buffer(
            (8, 8),
            "K N",
            bf16,
            memory=MemorySpace.HBM,
            ownership=Ownership.EXTERNAL,
            lifetime=(0, 3),
        ),
        buffer(
            (8, 8),
            "M N",
            f32,
            memory=MemorySpace.HBM,
            ownership=Ownership.EXTERNAL,
            lifetime=(0, 3),
        ),
    )
    builder = KernelBuilder(
        "pipeline_cost",
        "tpu7x",
        external,
        vmem_capacity_bytes=vmem_capacity_bytes,
        smem_capacity_bytes=1024,
        dma_engine_count=3,
        mxu_count=1,
        argument_modes=("input", "input", "input"),
    )
    local_specs = (
        buffer((8, 8), "M K", bf16, memory=MemorySpace.VMEM, lifetime=(0, 1)),
        buffer((8, 8), "K N", bf16, memory=MemorySpace.VMEM, lifetime=(0, 1)),
        buffer((8, 8), "M N", f32, memory=MemorySpace.VMEM, lifetime=(0, 1)),
    )
    captures = []
    transfers = []
    for index, (source, spec) in enumerate(zip(builder.inputs, local_specs, strict=True)):
        destination = builder.alloc(spec, f"capture_{index}")
        captures.append(destination)
        transfers.append(builder.dma_start(source, destination, builder.semaphore(), stage=0))
    for transfer in transfers:
        builder.dma_wait(transfer, stage=1)
    if unrelated_local:
        builder.alloc(
            buffer(
                (8, 8),
                "U V",
                bf16,
                memory=MemorySpace.VMEM,
                lifetime=(0, 3),
            ),
            "unrelated",
        )
    body = Block(arg_types=[capture.buffer.type for capture in captures])
    if use_einsum:
        work = (
            MxuEinsumOp(
                body.args[0],
                body.args[1],
                body.args[2],
                stage=0,
                contracting_dimensions=("K",),
                tile_m=tile_m,
                tile_k=8,
                tile_n=8,
            ),
        )
    else:
        work = (
            MxuMatmulOp(body.args[0], body.args[1], body.args[2], 0),
            MxuMatmulOp(body.args[0], body.args[1], body.args[2], 1),
        )
    body.add_ops((*work, PipelineYieldOp(*body.args)))
    builder.block.add_op(
        PipelineLoopOp(
            tuple(captures),
            body,
            trip_count=4,
            initiation_interval=initiation_interval,
            pipeline_stages=3,
            rotation_counts=(1, 1, accumulator_rotations),
        )
    )
    return builder.module()


def test_pipeline_work_multiplies_by_trip_count_but_rotation_storage_does_not() -> None:
    report = _report(_pipeline_module(initiation_interval=2, accumulator_rotations=2))

    assert report.operation_executions == (
        ("tpu_schedule.dma_start", 3),
        ("tpu_schedule.mxu_matmul", 8),
        ("tpu_schedule.pipeline_loop", 1),
    )
    assert sum(value.total_flops for value in report.mxu_regions) == 8192
    assert report.memory.allocated_vmem_bytes_per_device == 512
    assert report.memory.pipeline_rotation_vmem_bytes_per_device == 256
    assert report.memory.peak_live_vmem_bytes_per_device == 768
    pipeline_stages = tuple(value for value in report.stages if value.scope != "kernel")
    assert max(value.active_mxu for value in pipeline_stages) == 1


def test_pipeline_einsum_scratch_is_shared_dialect_and_report_authority() -> None:
    with pytest.raises(VerifyException, match="pipeline VMEM capacity exceeded.*768 > 600"):
        _pipeline_module(
            initiation_interval=2,
            accumulator_rotations=1,
            use_einsum=True,
            vmem_capacity_bytes=600,
        )

    report = _report(
        _pipeline_module(
            initiation_interval=2,
            accumulator_rotations=1,
            use_einsum=True,
            vmem_capacity_bytes=768,
        )
    )
    assert report.memory.peak_live_vmem_bytes_per_device == 768


def test_pipeline_memory_combines_uncaptured_locals_rotations_and_scratch() -> None:
    with pytest.raises(VerifyException, match="pipeline VMEM capacity exceeded.*896 > 800"):
        _pipeline_module(
            initiation_interval=2,
            accumulator_rotations=1,
            use_einsum=True,
            unrelated_local=True,
            vmem_capacity_bytes=800,
        )

    report = _report(
        _pipeline_module(
            initiation_interval=2,
            accumulator_rotations=1,
            use_einsum=True,
            unrelated_local=True,
            vmem_capacity_bytes=896,
        )
    )
    assert report.memory.peak_live_vmem_bytes_per_device == 896


def test_pallas_native_collective_scratch_is_charged_to_vmem_capacity() -> None:
    physical = lower_distributed_matmul(
        distributed_matmul_schedule(mesh_size=8, m=1024, k=128, n=1024),
        tile=MatmulTile(128, 128),
        collective_implementation=(CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING),
    )
    kernel = next(operation for operation in physical.walk() if isinstance(operation, KernelOp))
    kernel.properties["vmem_capacity_bytes"] = IntAttr(4_718_592)

    with pytest.raises(
        VerifyException,
        match="VMEM capacity exceeded at stage 3: 4980736 > 4718592",
    ):
        physical.verify()

    kernel.properties["vmem_capacity_bytes"] = IntAttr(4_980_736)
    kernel.properties["argument_modes"] = ArrayAttr(
        (StringAttr("input"), StringAttr("input"), StringAttr("output"))
    )
    physical.verify()
    report = analyze_physical_kernel(physical, hardware=tpu7x_tensorcore_rates())
    assert report.memory.allocated_vmem_bytes_per_device == 5_046_272
    assert report.memory.peak_live_vmem_bytes_per_device == 4_980_736
    assert "pallas_collective_hbm_scratch_bytes=1048576" in report.unpriced_work
    assert "pallas_collective_dma_semaphores=5" in report.unpriced_work
    assert "pallas_collective_startup_semaphores=1" in report.unpriced_work


def test_nested_einsum_cannot_skip_tpu_tile_legality() -> None:
    with pytest.raises(VerifyException, match="TPU Pallas tile M"):
        _pipeline_module(
            initiation_interval=2,
            accumulator_rotations=1,
            use_einsum=True,
            tile_m=4,
        )


def test_report_replay_and_atomic_writer_reject_mutation_and_overwrite(tmp_path: Path) -> None:
    module = _small_seqax_physical()
    report = _report(module)
    output = tmp_path / "physical-cost.json"
    written = write_physical_kernel_report(
        output,
        module=module,
        hardware=tpu7x_tensorcore_rates(),
    )

    assert PhysicalKernelResourceReport.model_validate_json(output.read_text()) == written
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        validate_physical_kernel_report(
            report.model_copy(
                update={"priced_hbm_time_floor_ns": report.priced_hbm_time_floor_ns + 1}
            ),
            module=module,
            hardware=tpu7x_tensorcore_rates(),
        )
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        write_physical_kernel_report(
            output,
            module=module,
            hardware=tpu7x_tensorcore_rates(),
        )


def test_physical_source_and_expected_schedule_hashes_fail_closed() -> None:
    module = _small_seqax_physical()
    source = physical_schedule_source(module)

    assert source == physical_schedule_source(module)
    with pytest.raises(UnsupportedPhysicalCostModelError, match="expected schedule"):
        analyze_physical_kernel(
            module,
            hardware=tpu7x_tensorcore_rates(),
            expected_schedule_sha256="0" * 64,
        )


def test_hardware_rates_are_bound_to_the_physical_target() -> None:
    module = _small_seqax_physical()
    kernel = next(operation for operation in module.walk() if isinstance(operation, KernelOp))
    kernel.properties["target"] = StringAttr("different-target")

    with pytest.raises(UnsupportedPhysicalCostModelError, match="TPU7x hardware rate"):
        _report(module)


def test_all_zero_device_work_has_no_derivable_imbalance_ratio() -> None:
    record = PhysicalImbalanceRecord(
        resource="unused",
        minimum=0,
        maximum=0,
        maximum_to_minimum_ratio=None,
        derivable=False,
        reason="No declared work uses this resource.",
    )

    assert record.derivable is False
    with pytest.raises(ValueError, match="IMBALANCE_RATIO_MISMATCH"):
        PhysicalImbalanceRecord.model_validate(
            {
                **record.model_dump(),
                "maximum_to_minimum_ratio": 1,
                "derivable": True,
            }
        )


def test_schedule_with_no_priced_work_has_no_limiting_resource() -> None:
    external = buffer(
        (8, 8),
        "X Y",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 0),
    )
    builder = KernelBuilder(
        "no_priced_work",
        "tpu7x",
        (external,),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        argument_modes=("input",),
    )

    report = _report(builder.module())

    assert report.priced_compute_time_floor_ns == 0
    assert report.priced_hbm_time_floor_ns == 0
    assert report.priced_ici_time_scenario_ns == 0
    assert report.predicted_limiting_priced_resource == "none"
    assert report.vector_work == ()


def test_remote_dma_routes_have_exact_per_link_traffic() -> None:
    external = buffer(
        (8, 8),
        "X Y",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 4),
    )
    builder = KernelBuilder(
        "remote_cost",
        "tpu7x",
        (external,),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        mesh={"t": 4},
        interconnect_bandwidth_bytes_per_second={"t": 600_000_000_000},
        argument_modes=("input",),
    )
    source = builder.alloc(
        buffer(
            (8, 8),
            "X Y",
            bf16,
            memory=MemorySpace.VMEM,
            lifetime=(0, 4),
        ),
        "source",
    )
    destination = builder.alloc(
        buffer(
            (8, 8),
            "X Y",
            bf16,
            memory=MemorySpace.VMEM,
            lifetime=(0, 4),
        ),
        "destination",
    )
    inbound = builder.dma_start(builder.inputs[0], source, builder.semaphore(), stage=0)
    builder.dma_wait(inbound, stage=1)
    remote = builder.remote_dma_start(
        source,
        destination,
        builder.semaphore(),
        stage=2,
        transfer_plan="shift:t:+1",
    )
    builder.remote_dma_wait(remote, stage=3)
    collective_destination = builder.alloc(
        buffer(
            (8, 8),
            "X Y",
            bf16,
            memory=MemorySpace.VMEM,
            lifetime=(4, 4),
        ),
        "collective_destination",
    )
    builder.collective(
        source,
        collective_destination,
        stage=4,
        kind=CollectiveKind.ALL_REDUCE,
        mesh_axis="t",
        group_size=4,
        reducer="sum",
    )
    report = _report(builder.module())

    assert len(report.remote_dmas) == 1
    assert report.remote_dmas[0].route_count == 3
    assert report.remote_dmas[0].aggregate_link_bytes == 384
    used_links = tuple(value for value in report.links if value.remote_dma_operation_ids)
    assert len(used_links) == 3
    assert {value.exact_remote_dma_link_bytes for value in used_links} == {128}
    assert {value.remote_dma_endpoint_bytes for value in report.devices} == {128, 256}
    assert (
        report.remote_dma_exact_endpoint_time_floor_ns > report.remote_dma_exact_link_time_floor_ns
    )
    assert report.combined_ici_injection_time_scenario_ns == (
        report.collective_ring_equivalent_time_scenario_ns
        + report.remote_dma_exact_endpoint_time_floor_ns
    )
    assert report.priced_ici_time_scenario_ns == report.combined_ici_injection_time_scenario_ns


def test_remote_dma_time_prices_the_most_congested_exact_link() -> None:
    base = rectilinear_topology(("t",), (4,), {"t": 100})
    links = tuple(
        LinkAttr(
            link.link_id,
            link.source_device,
            link.destination_device,
            link.bandwidth_bytes_per_second,
            IntAttr(2 if link.link_id.data == "link:1-2" else 1),
        )
        for link in base.links
    )
    shared = TransferPlanAttr(
        StringAttr("shared-middle-link"),
        ArrayAttr(
            (
                TransferRouteAttr(
                    StringAttr("route:0->2"),
                    IntAttr(0),
                    IntAttr(2),
                    ArrayAttr((StringAttr("link:0-1"), StringAttr("link:1-2"))),
                ),
                TransferRouteAttr(
                    StringAttr("route:3->1"),
                    IntAttr(3),
                    IntAttr(1),
                    ArrayAttr((StringAttr("link:2-3"), StringAttr("link:1-2"))),
                ),
            )
        ),
    )
    transfer_plans = tuple(
        sorted(
            (*base.transfer_plans, shared),
            key=lambda value: value.plan_id.data,
        )
    )
    topology = TopologyAttr(
        base.devices,
        ArrayAttr(links),
        base.collective_plans,
        ArrayAttr(transfer_plans),
    )
    external = buffer(
        (8, 8),
        "X Y",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 3),
    )
    builder = KernelBuilder(
        "shared_remote_cost",
        "tpu7x",
        (external,),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
        mesh={"t": 4},
        topology=topology,
        argument_modes=("input",),
    )
    source = builder.alloc(
        buffer(
            (8, 8),
            "X Y",
            bf16,
            memory=MemorySpace.VMEM,
            lifetime=(0, 3),
        ),
        "source",
    )
    destination = builder.alloc(
        buffer(
            (8, 8),
            "X Y",
            bf16,
            memory=MemorySpace.VMEM,
            lifetime=(0, 3),
        ),
        "destination",
    )
    inbound = builder.dma_start(builder.inputs[0], source, builder.semaphore(), stage=0)
    builder.dma_wait(inbound, stage=1)
    remote = builder.remote_dma_start(
        source,
        destination,
        builder.semaphore(),
        stage=2,
        transfer_plan="shared-middle-link",
    )
    builder.remote_dma_wait(remote, stage=3)
    report = _report(builder.module())

    assert tuple(value.remote_dma_endpoint_bytes for value in report.devices) == (
        128,
        128,
        128,
        128,
    )
    assert tuple(
        value.exact_remote_dma_link_bytes
        for value in report.links
        if value.exact_remote_dma_link_bytes
    ) == (128, 256, 128)
    assert report.remote_dma_exact_link_time_floor_ns == 2_560_000_000


def test_opaque_rpa_has_no_physical_work_convention() -> None:
    module = inkling_rpa_schedule()
    kernel = next(operation for operation in module.walk() if isinstance(operation, KernelOp))
    kernel.properties["argument_modes"] = ArrayAttr(
        StringAttr(value) for value in (*("input",) * 6, "output")
    )

    with pytest.raises(
        UnsupportedPhysicalCostModelError,
        match="no work convention for tpu_schedule.ragged_paged_attention",
    ):
        _report(module)


def test_public_cli_derives_and_replays_physical_cost(tmp_path: Path) -> None:
    module = _small_seqax_physical()
    schedule = tmp_path / "physical.xdsl"
    report = tmp_path / "physical-cost.json"
    schedule.write_text(canonical_module_text(module))

    assert _estimate_physical_cost(schedule, report) == 0
    assert _verify_physical_cost(report, schedule) == 0
    relocated = tmp_path / "relocated" / "renamed-schedule.xdsl"
    relocated.parent.mkdir()
    relocated.write_text(schedule.read_text())
    assert _verify_physical_cost(report, relocated) == 0

    saved = PhysicalKernelResourceReport.model_validate_json(report.read_text())
    assert saved.metrics[0].sources[0].artifact_path == (
        f"physical/{saved.physical_schedule_sha256}.xdsl"
    )
    report.write_text(
        saved.model_copy(
            update={"priced_compute_time_floor_ns": saved.priced_compute_time_floor_ns + 1}
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        _verify_physical_cost(report, schedule)
