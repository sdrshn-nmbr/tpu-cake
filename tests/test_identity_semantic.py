import numpy as np

from tpu_cake.dialects.tpu_schedule import SemaphoreAllocOp
from tpu_cake.identity import array_sha256, candidate_rng, semantic_seed, workload_rng
from tpu_cake.semantic import batch_permutation_invariance, prefix_invariance
from tpu_cake.source import SourceLocation, attach_source


def test_semantic_identity_is_order_independent_of_other_candidates() -> None:
    parts = ("experiment", "scenario", "attempt-0", "lhs")
    first = workload_rng(*parts).normal(size=(8, 8))
    candidate_rng("other", "scenario", "candidate", "attempt-0", "tile-order").normal(size=(8, 8))
    second = workload_rng(*parts).normal(size=(8, 8))
    np.testing.assert_array_equal(first, second)
    assert semantic_seed(*parts) == semantic_seed(*parts)
    assert array_sha256(first) == array_sha256(second)


def test_workload_inputs_are_matched_across_candidates() -> None:
    first = workload_rng("experiment", "scenario", "attempt-0", "lhs").normal(size=(4, 4))
    candidate_rng("experiment", "scenario", "candidate-a", "attempt-0", "search")
    candidate_rng("experiment", "scenario", "candidate-b", "attempt-0", "search")
    second = workload_rng("experiment", "scenario", "attempt-0", "lhs").normal(size=(4, 4))
    np.testing.assert_array_equal(first, second)


def test_prefix_invariance_detects_future_dependency() -> None:
    tokens = np.arange(8, dtype=np.float32)
    replacement = tokens + 100
    causal = prefix_invariance(lambda value: np.cumsum(value), tokens, 4, replacement=replacement)
    leaky = prefix_invariance(
        lambda value: np.full_like(value, value.sum()),
        tokens,
        4,
        replacement=replacement,
    )
    assert causal.passed
    assert not leaky.passed


def test_batch_permutation_invariance_detects_slot_dependency() -> None:
    inputs = np.arange(12, dtype=np.float32).reshape(3, 4)
    permutation = np.array([2, 0, 1])
    invariant = batch_permutation_invariance(lambda value: value * 2, inputs, permutation)
    slot_dependent = batch_permutation_invariance(
        lambda value: value + np.arange(value.shape[0])[:, None],
        inputs,
        permutation,
    )
    assert invariant.passed
    assert not slot_dependent.passed


def test_source_location_is_attached_without_entering_schedule_identity() -> None:
    operation = SemaphoreAllocOp()
    attach_source(operation, SourceLocation("model.py", 12, 4))
    assert "model.py" in str(operation.location)
