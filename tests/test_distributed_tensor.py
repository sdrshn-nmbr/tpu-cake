import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, bf16, f16, f32, i1, i32
from xdsl.ir import SSAValue
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.distributed_tensor import MeshAttr, PendingReductionsAttr
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def test_distributed_matmul_verifies_and_hashes_stably() -> None:
    first = distributed_matmul_schedule()
    second = distributed_matmul_schedule()
    first.verify()
    assert canonical_module_text(first) == canonical_module_text(second)
    assert schedule_sha256(first) == schedule_sha256(second)


def test_missing_reduction_is_rejected() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    with pytest.raises(VerifyException, match="partially reduced"):
        builder.module(partial)


def test_incorrect_reduce_scatter_sharding_is_rejected() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    result = builder.reduce_scatter(
        partial,
        tensor(f32, (("M", 16), ("N", 16)), sharding={"M": ("t",)}),
        axes=("t",),
        scatter_dimensions=("N",),
    )
    with pytest.raises(VerifyException, match="incorrect sharding"):
        builder.module(result)


def test_unknown_mesh_axis_is_rejected() -> None:
    lhs = tensor(bf16, (("M", 16),), sharding={"M": ("missing",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs,))
    with pytest.raises(VerifyException, match="unknown mesh axis"):
        builder.module(builder.inputs[0])


def test_einsum_cannot_drop_noncontracted_sharding() -> None:
    lhs = tensor(
        bf16,
        (("M", 16), ("K", 32)),
        sharding={"M": ("d",), "K": ("t",)},
    )
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"d": 2, "t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    result = builder.reduce_scatter(
        partial,
        tensor(f32, (("M", 16), ("N", 16)), sharding={"N": ("t",)}),
        axes=("t",),
        scatter_dimensions=("N",),
    )
    with pytest.raises(VerifyException, match="preserve sharding"):
        builder.module(result)


def test_einsum_cannot_invent_noncontracted_sharding() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"d": 2, "t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(
            f32,
            (("M", 16), ("N", 16)),
            sharding={"M": ("d",)},
            pending_reductions={"t": "sum"},
        ),
        contracting_dimension="K",
    )
    result = builder.reduce_scatter(
        partial,
        tensor(
            f32,
            (("M", 16), ("N", 16)),
            sharding={"M": ("d",), "N": ("t",)},
        ),
        axes=("t",),
        scatter_dimensions=("N",),
    )
    with pytest.raises(VerifyException, match="preserve sharding"):
        builder.module(result)


def test_mesh_map_order_does_not_change_identity() -> None:
    input_spec = tensor(bf16, (("M", 16),), sharding={"M": ("d",)})
    first = DistributedProgramBuilder("identity", {"d": 2, "t": 4}, (input_spec,))
    second = DistributedProgramBuilder("identity", {"t": 4, "d": 2}, (input_spec,))
    assert schedule_sha256(first.module(first.inputs[0])) == schedule_sha256(
        second.module(second.inputs[0])
    )


def test_einsum_rejects_mismatched_local_contraction_extent() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)))
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    complete = builder.all_reduce(
        partial,
        tensor(f32, (("M", 16), ("N", 16))),
        axes=("t",),
    )
    with pytest.raises(VerifyException, match="identical local sharding"):
        builder.module(complete)


def test_einsum_rejects_unsupported_accumulation_type() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f16, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
        accumulation_type=f16,  # type: ignore[arg-type]
    )
    with pytest.raises(VerifyException, match="accumulation must be f32"):
        builder.module(partial)


def test_mesh_rejects_noncanonical_axis_order() -> None:
    with pytest.raises(VerifyException, match="canonical lexical order"):
        MeshAttr(
            ArrayAttr((StringAttr("t"), StringAttr("d"))),
            ArrayAttr((IntAttr(4), IntAttr(2))),
        )


def test_pending_reductions_reject_noncanonical_axis_order() -> None:
    with pytest.raises(VerifyException, match="canonical lexical order"):
        PendingReductionsAttr(
            ArrayAttr((StringAttr("t"), StringAttr("d"))),
            ArrayAttr((StringAttr("sum"), StringAttr("sum"))),
        )


def test_all_reduce_rejects_duplicate_axes() -> None:
    lhs = tensor(bf16, (("M", 16), ("K", 32)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", 32), ("N", 16)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("bad", {"t": 4}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
    )
    complete = builder.all_reduce(
        partial,
        tensor(f32, (("M", 16), ("N", 16))),
        axes=("t", "t"),
    )
    with pytest.raises(VerifyException, match="axes must be unique"):
        builder.module(complete)


def test_seqax_style_embedding_and_mlp_primitives_verify_together() -> None:
    table = tensor(bf16, (("V", 64), ("D", 32)), sharding={"V": ("t",)})
    indices = tensor(i32, (("B", 8), ("S", 4)), sharding={"B": ("d",)})
    builder = DistributedProgramBuilder("forward_fragment", {"d": 2, "t": 4}, (table, indices))
    partial = builder.embedding_lookup(
        builder.inputs[0],
        builder.inputs[1],
        tensor(
            bf16,
            (("B", 8), ("S", 4), ("D", 32)),
            sharding={"B": ("d",)},
            pending_reductions={"t": "sum"},
        ),
        vocabulary_dimension="V",
    )
    complete = builder.all_reduce(
        partial,
        tensor(
            bf16,
            (("B", 8), ("S", 4), ("D", 32)),
            sharding={"B": ("d",)},
        ),
        axes=("t",),
    )
    activated = builder.elementwise(
        complete,
        result=tensor(
            bf16,
            (("B", 8), ("S", 4), ("D", 32)),
            sharding={"B": ("d",)},
        ),
        function="silu",
    )
    reduced = builder.reduce_local(
        activated,
        tensor(bf16, (("B", 8), ("S", 4)), sharding={"B": ("d",)}),
        dimensions=("D",),
        reducer="sum",
    )
    broadcast = builder.broadcast(
        reduced,
        tensor(
            bf16,
            (("B", 8), ("S", 4), ("D", 32)),
            sharding={"B": ("d",)},
        ),
    )
    result = builder.transpose(
        broadcast,
        tensor(
            bf16,
            (("S", 4), ("B", 8), ("D", 32)),
            sharding={"B": ("d",)},
        ),
        permutation=(1, 0, 2),
    )

    builder.module(result).verify()


def test_sharded_embedding_cannot_hide_its_pending_sum() -> None:
    table = tensor(bf16, (("V", 64), ("D", 32)), sharding={"V": ("t",)})
    indices = tensor(i32, (("B", 8),))
    builder = DistributedProgramBuilder("bad_embedding", {"t": 4}, (table, indices))
    result = builder.embedding_lookup(
        builder.inputs[0],
        builder.inputs[1],
        tensor(bf16, (("B", 8), ("D", 32))),
        vocabulary_dimension="V",
    )

    with pytest.raises(VerifyException, match="cross-device sum as pending"):
        builder.module(result)


def test_local_reduction_over_sharded_dimension_must_remain_pending() -> None:
    value = tensor(bf16, (("B", 8), ("D", 32)), sharding={"D": ("t",)})
    builder = DistributedProgramBuilder("bad_reduction", {"t": 4}, (value,))
    result = builder.reduce_local(
        builder.inputs[0],
        tensor(bf16, (("B", 8),)),
        dimensions=("D",),
        reducer="sum",
    )

    with pytest.raises(VerifyException, match="as pending"):
        builder.module(result)


def test_broadcast_cannot_invent_sharding_for_a_new_dimension() -> None:
    value = tensor(bf16, (("B", 8),))
    builder = DistributedProgramBuilder("bad_broadcast", {"t": 4}, (value,))
    result = builder.broadcast(
        builder.inputs[0],
        tensor(bf16, (("B", 8), ("D", 32)), sharding={"D": ("t",)}),
    )

    with pytest.raises(VerifyException, match="new broadcast dimensions must be replicated"):
        builder.module(result)


def test_elementwise_requires_identical_distributed_types() -> None:
    lhs = tensor(bf16, (("B", 8),), sharding={"B": ("t",)})
    rhs = tensor(bf16, (("B", 8),))
    builder = DistributedProgramBuilder("bad_elementwise", {"t": 4}, (lhs, rhs))
    result = builder.elementwise(
        *builder.inputs,
        result=lhs,
        function="add",
    )

    with pytest.raises(VerifyException, match="identical distributed tensor types"):
        builder.module(result)


def test_nonlinear_elementwise_rejects_pending_cross_device_reduction() -> None:
    value = tensor(
        bf16,
        (("B", 8), ("D", 32)),
        pending_reductions={"t": "sum"},
    )
    builder = DistributedProgramBuilder("bad_nonlinear_partial", {"t": 4}, (value,))
    result = builder.elementwise(
        builder.inputs[0],
        result=value,
        function="silu",
    )

    with pytest.raises(VerifyException, match="partially reduced"):
        builder.module(result)


def test_pending_axis_cannot_also_shard_a_retained_dimension() -> None:
    value = tensor(
        bf16,
        (("B", 8),),
        sharding={"B": ("t",)},
        pending_reductions={"t": "sum"},
    )
    with pytest.raises(VerifyException, match="cannot also shard"):
        DistributedProgramBuilder("bad_axis_role", {"t": 4}, (value,))


@pytest.mark.parametrize(
    ("function", "arity"),
    (("add", 1), ("exp", 2), ("multiply", 3)),
)
def test_elementwise_signatures_are_strict(function: str, arity: int) -> None:
    value = tensor(bf16, (("B", 8),))
    builder = DistributedProgramBuilder("bad_arity", {}, (value,) * arity)
    result = builder.elementwise(
        *builder.inputs,
        result=value,
        function=function,
    )

    with pytest.raises(VerifyException, match="requires"):
        builder.module(result)


def test_broadcast_cannot_silently_transpose_existing_dimensions() -> None:
    value = tensor(bf16, (("B", 8), ("S", 4)))
    builder = DistributedProgramBuilder("bad_broadcast_order", {}, (value,))
    result = builder.broadcast(
        builder.inputs[0],
        tensor(bf16, (("S", 4), ("B", 8), ("D", 32))),
    )

    with pytest.raises(VerifyException, match="preserve input dimension order"):
        builder.module(result)


def test_attention_einsum_preserves_shared_batch_and_head_dimensions() -> None:
    query = tensor(
        bf16,
        (("B", 8), ("Qlen", 16), ("Q", 4), ("K", 4), ("D", 8)),
        sharding={"B": ("d",), "K": ("t",)},
    )
    key = tensor(
        bf16,
        (("B", 8), ("Klen", 16), ("K", 4), ("D", 8)),
        sharding={"B": ("d",), "K": ("t",)},
    )
    builder = DistributedProgramBuilder("attention", {"d": 2, "t": 4}, (query, key))
    logits = builder.einsum(
        builder.inputs[0],
        builder.inputs[1],
        tensor(
            f32,
            (("B", 8), ("Qlen", 16), ("Klen", 16), ("Q", 4), ("K", 4)),
            sharding={"B": ("d",), "K": ("t",)},
        ),
        contracting_dimensions=("D",),
    )

    builder.module(logits).verify()


def test_general_einsum_exposes_reduction_over_every_sharded_contraction() -> None:
    activation = tensor(
        bf16,
        (("B", 8), ("L", 16), ("Q", 4), ("K", 4), ("D", 8)),
        sharding={"B": ("d",), "K": ("t",)},
    )
    weight = tensor(
        bf16,
        (("M", 32), ("Q", 4), ("K", 4), ("D", 8)),
        sharding={"K": ("t",)},
    )
    builder = DistributedProgramBuilder(
        "attention_projection", {"d": 2, "t": 4}, (activation, weight)
    )
    partial = builder.einsum(
        builder.inputs[0],
        builder.inputs[1],
        tensor(
            f32,
            (("B", 8), ("L", 16), ("M", 32)),
            sharding={"B": ("d",)},
            pending_reductions={"t": "sum"},
        ),
        contracting_dimensions=("D", "K", "Q"),
    )
    result = builder.reduce_scatter(
        partial,
        tensor(
            f32,
            (("B", 8), ("L", 16), ("M", 32)),
            sharding={"B": ("d",), "M": ("t",)},
        ),
        axes=("t",),
        scatter_dimensions=("M",),
    )

    builder.module(result).verify()


def test_general_einsum_cannot_drop_a_shared_batch_dimension() -> None:
    lhs = tensor(bf16, (("B", 8), ("M", 16), ("K", 32)))
    rhs = tensor(bf16, (("B", 8), ("K", 32), ("N", 16)))
    builder = DistributedProgramBuilder("bad_einsum", {}, (lhs, rhs))
    result = builder.einsum(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", 16), ("N", 16))),
        contracting_dimensions=("K",),
    )

    with pytest.raises(VerifyException, match="every noncontracted dimension"):
        builder.module(result)


def test_seqax_normalization_rope_mask_and_softmax_semantics_verify() -> None:
    value = tensor(f32, (("B", 8), ("L", 16), ("M", 32)), sharding={"B": ("d",)})
    scale = tensor(f32, (("M", 32),))
    sequence_starts = tensor(i1, (("B", 8), ("L", 16)), sharding={"B": ("d",)})
    query = tensor(
        bf16,
        (("B", 8), ("Qlen", 16), ("Q", 4), ("K", 4), ("D", 8)),
        sharding={"B": ("d",), "K": ("t",)},
    )
    logits = tensor(
        f32,
        (("B", 8), ("Qlen", 16), ("Klen", 16), ("Q", 4), ("K", 4)),
        sharding={"B": ("d",), "K": ("t",)},
    )
    builder = DistributedProgramBuilder(
        "seqax_semantics",
        {"d": 2, "t": 4},
        (value, scale, sequence_starts, query, logits),
    )
    cast = builder.cast(
        builder.inputs[0],
        tensor(bf16, value.dimensions, sharding={"B": ("d",)}),
    )
    normalized = builder.rms_norm(
        cast,
        builder.inputs[1],
        tensor(bf16, value.dimensions, sharding={"B": ("d",)}),
        dimension="M",
    )
    rotated = builder.rotary_embedding(
        builder.inputs[3],
        query,
        sequence_dimension="Qlen",
        head_dimension="D",
        maximum_timescale=10_000,
    )
    mask = builder.packed_causal_mask(
        builder.inputs[2],
        tensor(
            i1,
            (("B", 8), ("Qlen", 16), ("Klen", 16)),
            sharding={"B": ("d",)},
        ),
        sequence_dimension="L",
        query_dimension="Qlen",
        key_dimension="Klen",
    )
    probabilities = builder.masked_softmax(
        builder.inputs[4],
        mask,
        tensor(bf16, logits.dimensions, sharding={"B": ("d",), "K": ("t",)}),
        dimension="Klen",
    )

    builder.module(normalized, rotated, probabilities).verify()


def test_slice_removes_only_one_unsharded_dimension() -> None:
    qkv = tensor(
        bf16,
        (("KV", 2), ("B", 8), ("L", 16), ("K", 4), ("D", 8)),
        sharding={"B": ("d",), "K": ("t",)},
    )
    builder = DistributedProgramBuilder("slice", {"d": 2, "t": 4}, (qkv,))
    key = builder.slice(
        builder.inputs[0],
        tensor(
            bf16,
            (("B", 8), ("L", 16), ("K", 4), ("D", 8)),
            sharding={"B": ("d",), "K": ("t",)},
        ),
        dimension="KV",
        index=0,
    )

    builder.module(key).verify()


def test_slice_cannot_index_a_sharded_dimension() -> None:
    value = tensor(bf16, (("B", 8), ("M", 32)), sharding={"B": ("d",)})
    builder = DistributedProgramBuilder("bad_slice", {"d": 2}, (value,))
    result = builder.slice(
        builder.inputs[0],
        tensor(bf16, (("M", 32),)),
        dimension="B",
        index=0,
    )

    with pytest.raises(VerifyException, match="cannot index a sharded dimension"):
        builder.module(result)


def test_rms_norm_rejects_a_sharded_scale() -> None:
    value = tensor(bf16, (("B", 8), ("M", 32)), sharding={"B": ("d",)})
    scale = tensor(bf16, (("M", 32),), sharding={"M": ("t",)})
    builder = DistributedProgramBuilder("bad_norm", {"d": 2, "t": 4}, (value, scale))
    result = builder.rms_norm(
        builder.inputs[0],
        builder.inputs[1],
        value,
        dimension="M",
    )

    with pytest.raises(VerifyException, match="locally replicated"):
        builder.module(result)


def test_rotary_embedding_rejects_an_odd_head_dimension() -> None:
    value = tensor(bf16, (("B", 8), ("L", 16), ("D", 7)))
    builder = DistributedProgramBuilder("bad_rope", {}, (value,))
    result = builder.rotary_embedding(
        builder.inputs[0],
        value,
        sequence_dimension="L",
        head_dimension="D",
        maximum_timescale=10_000,
    )

    with pytest.raises(VerifyException, match="even head dimension"):
        builder.module(result)


def test_packed_causal_mask_rejects_invented_query_sharding() -> None:
    starts = tensor(i1, (("B", 8), ("L", 16)), sharding={"B": ("d",)})
    builder = DistributedProgramBuilder("bad_mask", {"d": 2, "t": 4}, (starts,))
    result = builder.packed_causal_mask(
        builder.inputs[0],
        tensor(
            i1,
            (("B", 8), ("Qlen", 16), ("Klen", 16)),
            sharding={"B": ("d",), "Qlen": ("t",)},
        ),
        sequence_dimension="L",
        query_dimension="Qlen",
        key_dimension="Klen",
    )

    with pytest.raises(VerifyException, match="wrong shape or sharding"):
        builder.module(result)


def test_masked_softmax_rejects_mask_sharding_that_differs_from_logits() -> None:
    logits = tensor(
        f32,
        (("B", 8), ("Qlen", 16), ("Klen", 16)),
        sharding={"B": ("d",)},
    )
    mask = tensor(
        i1,
        (("B", 8), ("Qlen", 16), ("Klen", 16)),
        sharding={"B": ("d",), "Qlen": ("t",)},
    )
    builder = DistributedProgramBuilder("bad_softmax", {"d": 2, "t": 4}, (logits, mask))
    result = builder.masked_softmax(
        builder.inputs[0],
        builder.inputs[1],
        tensor(bf16, logits.dimensions, sharding={"B": ("d",)}),
        dimension="Klen",
    )

    with pytest.raises(VerifyException, match="named subset"):
        builder.module(result)


def test_cast_rejects_a_partially_reduced_value() -> None:
    partial = tensor(
        f32,
        (("B", 8), ("M", 32)),
        pending_reductions={"t": "sum"},
    )
    builder = DistributedProgramBuilder("bad_cast", {"t": 4}, (partial,))
    result = builder.cast(builder.inputs[0], tensor(bf16, partial.dimensions))

    with pytest.raises(VerifyException, match="partially reduced"):
        builder.module(result)


def test_layer_scan_types_carries_stacked_weights_and_invariants() -> None:
    carry = tensor(bf16, (("B", 8), ("L", 16), ("M", 32)), sharding={"B": ("d",)})
    stacked = tensor(
        bf16,
        (("Z", 2), ("M", 32), ("F", 64)),
        sharding={"F": ("t",)},
    )
    invariant = carry
    builder = DistributedProgramBuilder(
        "layer_scan",
        {"d": 2, "t": 4},
        (carry, stacked, invariant),
    )

    def body(
        nested: DistributedProgramBuilder, arguments: tuple[SSAValue, ...]
    ) -> tuple[SSAValue, ...]:
        carried, _weight, residual = arguments
        return (
            nested.elementwise(
                carried,
                residual,
                result=carry,
                function="add",
            ),
        )

    (result,) = builder.layer_scan(
        builder.inputs,
        body,
        carry_count=1,
        stacked_count=1,
        layer_dimension="Z",
        trip_count=2,
    )

    builder.module(result).verify()


def test_layer_scan_rejects_a_sharded_layer_dimension() -> None:
    carry = tensor(bf16, (("B", 8), ("M", 32)))
    stacked = tensor(
        bf16,
        (("Z", 2), ("M", 32)),
        sharding={"Z": ("d",)},
    )
    builder = DistributedProgramBuilder("bad_scan", {"d": 2}, (carry, stacked))

    (result,) = builder.layer_scan(
        builder.inputs,
        lambda _nested, arguments: (arguments[0],),
        carry_count=1,
        stacked_count=1,
        layer_dimension="Z",
        trip_count=2,
    )

    with pytest.raises(VerifyException, match="locally replicated"):
        builder.module(result)


def test_layer_scan_rejects_a_yield_type_that_changes_the_carry() -> None:
    carry = tensor(bf16, (("B", 8), ("M", 32)))
    stacked = tensor(bf16, (("Z", 2), ("M", 32)))
    invariant = tensor(bf16, (("B", 8), ("F", 64)))
    builder = DistributedProgramBuilder(
        "bad_yield",
        {},
        (carry, stacked, invariant),
    )

    (result,) = builder.layer_scan(
        builder.inputs,
        lambda _nested, arguments: (arguments[2],),
        carry_count=1,
        stacked_count=1,
        layer_dimension="Z",
        trip_count=2,
    )

    with pytest.raises(VerifyException, match="yield types"):
        builder.module(result)
