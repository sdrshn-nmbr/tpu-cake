import numpy as np

from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.workloads.inkling_rpa import (
    inkling_fused_rpa_contract,
    inkling_fused_rpa_experiment,
    inkling_fused_rpa_schedule,
    inkling_rpa_inputs,
    inkling_rpa_reference,
    inkling_rpa_schedule,
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
    for build in (matmul_schedule, inkling_rpa_schedule):
        first = build()
        second = build()
        assert canonical_module_text(first) == canonical_module_text(second)
        assert schedule_sha256(first) == schedule_sha256(second)


def test_fused_rpa_experiment_binds_oracle_preflight_and_backend_sources() -> None:
    contract = inkling_fused_rpa_contract()
    experiment = inkling_fused_rpa_experiment()

    assert len(contract.inputs) == 11
    assert len(contract.outputs) == 2
    assert contract.execution.scope == "local-shard-caller-owned-sharding"
    assert contract.execution.preflight == "tpu_cake.rpa_lowering.FusedRpaPlan.preflight"
    assert len(contract.execution.source_manifest) == 3
    assert experiment.workload == contract
    assert experiment.schedule_sha256 == schedule_sha256(inkling_fused_rpa_schedule())
