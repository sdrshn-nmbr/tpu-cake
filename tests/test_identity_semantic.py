import numpy as np
import pytest
from xdsl.dialects.builtin import bf16, f32
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import MemorySpace, Ownership, SemaphoreAllocOp
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.frontend import KernelBuilder, buffer, schedule_sha256
from tpu_cake.identity import (
    array_sha256,
    candidate_rng,
    semantic_seed,
    semantic_sha256,
    workload_rng,
)
from tpu_cake.semantic import (
    batch_permutation_invariance,
    cache_equivalence,
    prefill_decode_equivalence,
    prefix_invariance,
    state_isolation,
    stepwise_equivalence,
)
from tpu_cake.source import SourceLocation, attach_source


def test_semantic_identity_is_order_independent_of_other_candidates() -> None:
    parts = ("experiment", "scenario", "attempt-0", "lhs")
    first = workload_rng(*parts).normal(size=(8, 8))
    candidate_rng("other", "scenario", "candidate", "attempt-0", "tile-order").normal(size=(8, 8))
    second = workload_rng(*parts).normal(size=(8, 8))
    np.testing.assert_array_equal(first, second)
    assert semantic_seed(*parts) == semantic_seed(*parts)
    assert array_sha256(first) == array_sha256(second)


def test_semantic_identity_is_unambiguous_across_part_boundaries() -> None:
    assert semantic_sha256("a", "b\x1fc") != semantic_sha256("a\x1fb", "c")


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


def test_execution_path_properties_detect_state_and_cache_errors() -> None:
    inputs = np.arange(12, dtype=np.float32).reshape(3, 4)
    identity = lambda value: value * 2
    shifted = lambda value: value * 2 + 1

    assert prefill_decode_equivalence(identity, identity, inputs).passed
    assert not prefill_decode_equivalence(identity, shifted, inputs).passed
    assert stepwise_equivalence(identity, identity, inputs).passed
    assert not cache_equivalence(identity, shifted, inputs).passed

    together = lambda value: value * 2
    isolated = lambda value: value * 2
    contaminated = lambda value: value * 2 + np.arange(len(value))[:, None]
    assert state_isolation(together, isolated, inputs).passed
    assert not state_isolation(contaminated, isolated, inputs).passed


def test_source_location_is_attached_without_entering_schedule_identity() -> None:
    operation = SemaphoreAllocOp()
    attach_source(operation, SourceLocation("model.py", 12, 4))
    assert "model.py" in str(operation.location)


def test_commutative_elementwise_operand_order_has_one_identity() -> None:
    value = tensor(f32, (("M", 16),))
    first = DistributedProgramBuilder("add", {"t": 1}, (value, value))
    second = DistributedProgramBuilder("add", {"t": 1}, (value, value))
    first_result = first.elementwise(
        first.inputs[0], first.inputs[1], result=value, function="add"
    )
    second_result = second.elementwise(
        second.inputs[1], second.inputs[0], result=value, function="add"
    )
    assert schedule_sha256(first.module(first_result)) == schedule_sha256(
        second.module(second_result)
    )


def test_alias_names_are_alpha_normalized_for_identity() -> None:
    spec = buffer(
        (16,),
        ("M",),
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
    )
    first = KernelBuilder(
        "views", "tpu", (spec,), vmem_capacity_bytes=1 << 20, smem_capacity_bytes=1 << 20
    )
    second = KernelBuilder(
        "views", "tpu", (spec,), vmem_capacity_bytes=1 << 20, smem_capacity_bytes=1 << 20
    )
    local_first = first.alloc(
        buffer((16,), ("M",), bf16, memory=MemorySpace.VMEM), "local"
    )
    local_second = second.alloc(
        buffer((16,), ("M",), bf16, memory=MemorySpace.VMEM), "local"
    )
    first.view(
        local_first,
        buffer((8,), ("8",), bf16, memory=MemorySpace.VMEM),
        offsets=(0,),
        alias_group="human-name",
    )
    second.view(
        local_second,
        buffer((8,), ("8",), bf16, memory=MemorySpace.VMEM),
        offsets=(0,),
        alias_group="different-name",
    )
    assert schedule_sha256(first.module()) == schedule_sha256(second.module())


def test_human_labels_are_alpha_normalized_for_schedule_identity() -> None:
    spec = buffer(
        (16,),
        ("M",),
        bf16,
        memory=MemorySpace.VMEM,
    )
    first = KernelBuilder(
        "first-human-name",
        "tpu",
        (),
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 20,
    )
    second = KernelBuilder(
        "second-human-name",
        "tpu",
        (),
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 20,
    )
    first.alloc(spec, "lhs-tile")
    second.alloc(spec, "a-label-with-no-execution-meaning")

    assert schedule_sha256(first.module()) == schedule_sha256(second.module())


def test_distributed_program_symbol_is_not_schedule_identity() -> None:
    value = tensor(f32, (("M", 16),))
    first = DistributedProgramBuilder("first-human-name", {"t": 1}, (value,))
    second = DistributedProgramBuilder("second-human-name", {"t": 1}, (value,))

    assert schedule_sha256(first.module(first.inputs[0])) == schedule_sha256(
        second.module(second.inputs[0])
    )


def test_verifier_failure_points_to_source_expression() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    invalid = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16))),
        contracting_dimension="K",
        source=SourceLocation("factory.py", 123, 7),
    )
    with pytest.raises(
        VerifyException,
        match=r'dtensor.einsum_local at loc\("factory.py":123:7\)',
    ):
        builder.module(invalid)
