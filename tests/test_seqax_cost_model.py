from decimal import Decimal

import pytest
from xdsl.dialects.builtin import bf16, f32, f64

from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.metrics import MeasurementKind, MetricSource
from tpu_cake.seqax_cost_model import (
    UnsupportedSeqaxCostModelError,
    estimate_seqax_forward,
)
from tpu_cake.workloads.seqax_forward import (
    SeqaxNormScalePlacement,
    SeqaxWeightDataPlacement,
    seqax_forward_schedule,
)


def _source() -> MetricSource:
    return MetricSource(
        artifact_sha256="a" * 64,
        artifact_path="seqax-forward.xdsl",
        tool="tpu-cake",
        field="canonical-distributed-tensor-ir",
    )


def test_sharded_elementwise_counts_bytes_work_and_liveness_exactly() -> None:
    value = tensor(bf16, (("B", 4),), sharding={"B": ("d",)})
    builder = DistributedProgramBuilder("small_add", {"d": 2}, (value, value))
    result = builder.elementwise(
        *builder.inputs,
        result=value,
        function="add",
    )

    report = estimate_seqax_forward(
        builder.module(result),
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
    )

    assert report.counts.useful_global_vector_flops == 4
    assert report.counts.vector_flops_per_device == 2
    assert report.counts.global_logical_read_bytes == 16
    assert report.counts.global_logical_write_bytes == 8
    assert report.counts.local_logical_read_bytes_per_device == 8
    assert report.counts.local_logical_write_bytes_per_device == 4
    assert report.counts.minimum_hbm_read_bytes_per_device == 8
    assert report.counts.minimum_hbm_write_bytes_per_device == 4
    assert report.counts.peak_global_logical_live_bytes == 24
    assert report.counts.peak_local_logical_live_bytes_per_device == 12
    assert report.predicted_limiting_resource == "hbm"
    assert all(metric.kind is MeasurementKind.ESTIMATED for metric in report.metrics)
    assert all(metric.formula is not None for metric in report.metrics)


def test_matmul_and_all_reduce_have_hand_checkable_compute_and_ici_counts() -> None:
    lhs = tensor(bf16, (("M", 2), ("K", 4)), sharding={"K": ("d",)})
    rhs = tensor(bf16, (("K", 4), ("N", 3)), sharding={"K": ("d",)})
    partial = tensor(
        f32,
        (("M", 2), ("N", 3)),
        pending_reductions={"d": "sum"},
    )
    output = tensor(f32, (("M", 2), ("N", 3)))
    builder = DistributedProgramBuilder("small_matmul", {"d": 2}, (lhs, rhs))
    value = builder.einsum(
        *builder.inputs,
        result=partial,
        contracting_dimensions=("K",),
    )
    value = builder.all_reduce(value, output, axes=("d",))

    report = estimate_seqax_forward(
        builder.module(value),
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
    )

    assert report.counts.useful_global_mxu_flops == 48
    assert report.counts.mxu_flops_per_device == 24
    assert report.counts.global_logical_read_bytes == 64
    assert report.counts.global_logical_write_bytes == 48
    assert report.counts.local_logical_read_bytes_per_device == 44
    assert report.counts.local_logical_write_bytes_per_device == 48
    assert report.counts.ici_bidirectional_bytes_per_device == Decimal(48)
    assert report.counts.peak_global_logical_live_bytes == 64
    assert report.counts.peak_local_logical_live_bytes_per_device == 48
    assert report.collectives[0].kind == "all_reduce"
    assert report.collectives[0].axes == ("d",)
    assert report.collectives[0].executions == 1
    assert report.collectives[0].bidirectional_bytes_per_device == Decimal(48)


def test_layer_scan_trip_count_multiplies_work_but_not_live_stacked_weights() -> None:
    carry = tensor(bf16, (("B", 4),), sharding={"B": ("d",)})
    stacked = tensor(
        bf16,
        (("Z", 3), ("B", 4)),
        sharding={"B": ("d",)},
    )
    builder = DistributedProgramBuilder("small_scan", {"d": 2}, (carry, stacked))

    def body(nested, arguments):
        result = nested.elementwise(
            *arguments,
            result=carry,
            function="add",
        )
        return (result,)

    (result,) = builder.layer_scan(
        builder.inputs,
        body,
        carry_count=1,
        stacked_count=1,
        layer_dimension="Z",
        trip_count=3,
    )
    report = estimate_seqax_forward(
        builder.module(result),
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
    )

    assert dict(report.counts.operation_executions) == {
        "dtensor.elementwise": 3,
        "dtensor.layer_scan": 1,
    }
    assert report.counts.vector_flops_per_device == 6
    assert report.counts.local_logical_read_bytes_per_device == 24
    assert report.counts.local_logical_write_bytes_per_device == 12
    assert report.counts.peak_local_logical_live_bytes_per_device == 24


def test_complete_seqax_forward_expands_layer_scan_and_emits_typed_bounds() -> None:
    module = seqax_forward_schedule()
    execution_plan = lower_distributed_program_to_jax_mesh(module)
    report = estimate_seqax_forward(
        module,
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
        expected_schedule_sha256=execution_plan.schedule_sha256,
    )
    executions = dict(report.counts.operation_executions)

    assert report.schedule_sha256
    assert report.program_name == "program"
    assert report.schedule_sha256 == execution_plan.schedule_sha256
    assert report.mesh_axes == execution_plan.mesh_axes
    assert report.canonical_operation_inventory == execution_plan.operation_inventory
    assert report.counts.mesh_devices == 8
    assert executions["dtensor.layer_scan"] == 1
    assert executions["dtensor.all_gather"] == 24
    assert executions["dtensor.reduce_scatter"] == 5
    assert executions["dtensor.einsum"] == 17
    assert executions["dtensor.cast"] == 28
    assert report.counts.mxu_flops_per_device > 0
    assert report.counts.minimum_hbm_read_bytes_per_device > 0
    assert report.counts.materialized_hbm_bytes_per_device > 0
    assert report.counts.ici_bidirectional_bytes_per_device > 0
    assert report.counts.peak_local_logical_live_bytes_per_device > 0
    metrics = {metric.name: metric for metric in report.metrics}
    assert (
        metrics["seqax_global_logical_tensor_bytes"].interval.scope
        == "one complete logical Seqax forward across the full device mesh"
    )
    assert (
        metrics["seqax_local_logical_tensor_bytes_per_device"].interval.scope
        == "one complete logical Seqax forward on one JAX TPU device"
    )
    assert report.balance.maximum_to_minimum_work_ratio == Decimal(1)
    assert report.predicted_limiting_resource in {"compute", "hbm", "ici"}
    assert any("none of its values are device measurements" in item for item in report.omissions)


def test_norm_scale_replication_accounts_for_memory_and_communication_tradeoff() -> None:
    parameters = {
        "batch": 2,
        "sequence": 1,
        "model": 256,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 4,
        "head": 4,
        "layers": 1,
        "data_mesh": 2,
        "tensor_mesh": 4,
        "rope_max_timescale": 256,
    }
    sharded = estimate_seqax_forward(
        seqax_forward_schedule(**parameters),
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
    )
    replicated = estimate_seqax_forward(
        seqax_forward_schedule(
            **parameters,
            norm_scale_placement=SeqaxNormScalePlacement.REPLICATED,
        ),
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
    )

    assert (
        replicated.counts.minimum_hbm_read_bytes_per_device
        - sharded.counts.minimum_hbm_read_bytes_per_device
        == 2_688
    )
    assert (
        sharded.counts.ici_bidirectional_bytes_per_device
        - replicated.counts.ici_bidirectional_bytes_per_device
        == 5_376
    )
    assert sharded.predicted_limiting_resource == "ici"
    assert replicated.predicted_limiting_resource == "ici"


def test_weight_data_replication_accounts_for_memory_and_communication_tradeoff() -> None:
    parameters = {
        "batch": 2,
        "sequence": 1,
        "model": 256,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 4,
        "head": 4,
        "layers": 1,
        "data_mesh": 2,
        "tensor_mesh": 4,
        "rope_max_timescale": 256,
    }
    sharded = estimate_seqax_forward(
        seqax_forward_schedule(**parameters),
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
    )
    replicated = estimate_seqax_forward(
        seqax_forward_schedule(
            **parameters,
            weight_data_placement=SeqaxWeightDataPlacement.REPLICATED,
        ),
        hardware=tpu7x_tensorcore_rates(),
        source=_source(),
    )

    assert (
        replicated.counts.minimum_hbm_read_bytes_per_device
        - sharded.counts.minimum_hbm_read_bytes_per_device
        == 22_528
    )
    assert (
        sharded.counts.ici_bidirectional_bytes_per_device
        - replicated.counts.ici_bidirectional_bytes_per_device
        == 22_528
    )
    assert sharded.predicted_limiting_resource == "ici"
    assert replicated.predicted_limiting_resource == "ici"


def test_elementwise_without_an_explicit_work_convention_fails_closed() -> None:
    value = tensor(bf16, (("B", 4),), sharding={"B": ("d",)})
    builder = DistributedProgramBuilder("unsupported_gelu", {"d": 2}, (value,))
    result = builder.elementwise(
        builder.inputs[0],
        result=value,
        function="gelu",
    )

    with pytest.raises(
        UnsupportedSeqaxCostModelError,
        match="no explicit scalar work convention for elementwise gelu",
    ):
        estimate_seqax_forward(
            builder.module(result),
            hardware=tpu7x_tensorcore_rates(),
            source=_source(),
        )


def test_unsupported_dtype_and_schedule_identity_fail_closed() -> None:
    unsupported = tensor(f64, (("B", 4),), sharding={"B": ("d",)})
    builder = DistributedProgramBuilder("unsupported_dtype", {"d": 2}, (unsupported,))
    module = builder.module(builder.inputs[0])

    with pytest.raises(
        UnsupportedSeqaxCostModelError,
        match="unsupported distributed tensor element type",
    ):
        estimate_seqax_forward(
            module,
            hardware=tpu7x_tensorcore_rates(),
            source=_source(),
        )

    supported = tensor(bf16, (("B", 4),), sharding={"B": ("d",)})
    builder = DistributedProgramBuilder("wrong_identity", {"d": 2}, (supported,))
    result = builder.elementwise(
        builder.inputs[0],
        result=supported,
        function="silu",
    )
    with pytest.raises(
        UnsupportedSeqaxCostModelError,
        match="distributed schedule hash mismatch",
    ):
        estimate_seqax_forward(
            builder.module(result),
            hardware=tpu7x_tensorcore_rates(),
            source=_source(),
            expected_schedule_sha256="0" * 64,
        )


def test_all_gather_over_multiple_tensor_dimensions_fails_closed() -> None:
    sharded = tensor(
        bf16,
        (("B", 4), ("M", 4)),
        sharding={"B": ("d",), "M": ("t",)},
    )
    replicated = tensor(bf16, (("B", 4), ("M", 4)))
    builder = DistributedProgramBuilder(
        "unsupported_all_gather",
        {"d": 2, "t": 2},
        (sharded,),
    )
    result = builder.all_gather(builder.inputs[0], replicated)

    with pytest.raises(
        UnsupportedSeqaxCostModelError,
        match="sharding removal from exactly one dimension",
    ):
        estimate_seqax_forward(
            builder.module(result),
            hardware=tpu7x_tensorcore_rates(),
            source=_source(),
        )
