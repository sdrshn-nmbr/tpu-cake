import numpy as np
import pytest

from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.rpa_lowering import (
    lower_inkling_rpa_to_pallas,
    lower_inkling_sharded_rpa_to_pallas,
)
from tpu_cake.workloads import inkling_rpa as rpa_workload
from tpu_cake.workloads.inkling_rpa import (
    inkling_fused_rpa_contract,
    inkling_fused_rpa_experiment,
    inkling_fused_rpa_schedule,
    inkling_rpa_inputs,
    inkling_rpa_reference,
    inkling_rpa_schedule,
    inkling_sharded_fused_rpa_contract,
    inkling_sharded_fused_rpa_experiment,
    inkling_sharded_fused_rpa_inputs,
    inkling_sharded_fused_rpa_reference,
    inkling_sharded_fused_rpa_schedule,
)
from tpu_cake.workloads.matmul import matmul_inputs, matmul_reference, matmul_schedule


def test_matmul_reference_has_expected_shape() -> None:
    lhs, rhs = matmul_inputs()
    actual = matmul_reference(lhs, rhs)
    np.testing.assert_allclose(actual, lhs @ rhs)
    assert actual.shape == (16, 16)


def test_rpa_reference_is_deterministic_and_finite() -> None:
    inputs = inkling_rpa_inputs()
    first = inkling_rpa_reference(*inputs)
    second = inkling_rpa_reference(*inputs)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 4, 32)
    assert np.isfinite(first).all()


def test_rpa_reference_supports_independent_key_and_value_heads() -> None:
    query = np.ones((1, 4, 8), dtype=np.float32)
    key_cache = np.ones((2, 4, 2, 8), dtype=np.float32)
    value_cache = np.arange(2 * 4 * 1 * 6, dtype=np.float32).reshape(2, 4, 1, 6)
    page_table = np.array([[0, 1]], dtype=np.int32)
    sequence_lengths = np.array([5], dtype=np.int32)
    bias = np.zeros((4, 8), dtype=np.float32)

    output = inkling_rpa_reference(
        query,
        key_cache,
        value_cache,
        page_table,
        sequence_lengths,
        bias,
    )

    assert output.shape == (1, 4, 6)
    np.testing.assert_allclose(output[0], np.broadcast_to(np.arange(6) + 12, (4, 6)))


def test_schedule_text_and_hash_are_stable() -> None:
    for build in (
        matmul_schedule,
        inkling_rpa_schedule,
        inkling_sharded_fused_rpa_schedule,
    ):
        first = build()
        second = build()
        assert canonical_module_text(first) == canonical_module_text(second)
        assert schedule_sha256(first) == schedule_sha256(second)


def test_fused_rpa_experiment_binds_oracle_preflight_and_backend_sources() -> None:
    contract = inkling_fused_rpa_contract()
    experiment = inkling_fused_rpa_experiment()
    plan = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())

    assert len(contract.inputs) == 11
    assert len(contract.outputs) == 2
    assert contract.execution.scope == "local-shard-caller-owned-sharding"
    assert contract.execution.preflight == "tpu_cake.rpa_lowering.FusedRpaPlan.preflight"
    assert len(contract.execution.source_manifest) == 4
    assert experiment.workload == contract
    assert experiment.schedule_sha256 == schedule_sha256(inkling_fused_rpa_schedule())
    assert tuple(tensor.shape for tensor in contract.inputs) == plan.input_shapes
    assert tuple(tensor.dtype for tensor in contract.inputs) == plan.input_dtypes
    assert tuple(tensor.shape for tensor in contract.outputs) == (
        plan.output_shape,
        plan.fused_cache_shape,
    )
    assert tuple(tensor.dtype for tensor in contract.outputs) == plan.output_dtypes
    assert (
        tuple((source.path, source.sha256) for source in contract.execution.source_manifest)
        == plan.backend_manifest
    )


def test_fused_rpa_experiment_materializes_decode_block_sizes() -> None:
    incumbent = inkling_fused_rpa_experiment()
    candidate_blocks = (8, 64, 8, 64)
    candidate_schedule = inkling_fused_rpa_schedule(candidate_blocks)
    candidate_plan = lower_inkling_rpa_to_pallas(candidate_schedule)
    candidate = inkling_fused_rpa_experiment(candidate_blocks)

    assert candidate_plan.decode_block_sizes == candidate_blocks
    assert candidate.schedule_sha256 == candidate_plan.schedule_sha256
    assert candidate.schedule_sha256 != incumbent.schedule_sha256


def test_sharded_fused_rpa_contract_binds_the_global_production_surface() -> None:
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    contract = inkling_sharded_fused_rpa_contract(plan)
    experiment = inkling_sharded_fused_rpa_experiment()

    assert tuple(tensor.shape for tensor in contract.inputs) == plan.global_input_shapes
    assert tuple(tensor.sharding for tensor in contract.inputs) == plan.input_partition_specs
    assert tuple(tensor.shape for tensor in contract.outputs) == plan.global_output_shapes
    assert tuple(tensor.sharding for tensor in contract.outputs) == plan.output_partition_specs
    assert contract.numerical.absolute_tolerance == 0.001
    assert contract.numerical.relative_tolerance == 0.006
    assert contract.execution.scope == "owned-data-tensor-shard-map"
    assert experiment.target.chip_count == 8
    assert experiment.schedule_sha256 == plan.schedule_sha256


def test_sharded_fused_rpa_reference_reconstructs_all_local_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = inkling_sharded_fused_rpa_inputs(1301)
    calls = []

    def local_reference(local_inputs):
        calls.append(tuple(value.shape for value in local_inputs))
        return local_inputs[0], local_inputs[3]

    monkeypatch.setattr(rpa_workload, "inkling_fused_rpa_reference", local_reference)
    output, cache = inkling_sharded_fused_rpa_reference(inputs)

    assert len(calls) == 8
    assert len(set(calls)) == 1
    np.testing.assert_array_equal(output, inputs[0])
    np.testing.assert_array_equal(cache, inputs[3])
