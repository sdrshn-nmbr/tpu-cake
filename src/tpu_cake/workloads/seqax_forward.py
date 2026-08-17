from __future__ import annotations

from xdsl.dialects.builtin import IntegerType, ModuleOp, Signedness, bf16, f32, i1
from xdsl.ir import SSAValue

from tpu_cake.distributed_frontend import (
    DistributedProgramBuilder,
    DistributedTensorSpec,
    tensor,
)
from tpu_cake.source import SourceLocation

SEQAX_REVISION = "b418a2d9059a1bfcff801d22b7088cc444257703"
SEQAX_FORWARD_SOURCE = "seqax/train.py"
U32 = IntegerType(32, Signedness.UNSIGNED)


def _source(line: int) -> SourceLocation:
    return SourceLocation(SEQAX_FORWARD_SOURCE, line, 1)


def seqax_forward_schedule(
    *,
    batch: int = 8,
    sequence: int = 16,
    model: int = 32,
    vocabulary: int = 64,
    feed_forward: int = 64,
    query_groups: int = 4,
    key_value_heads: int = 4,
    head: int = 8,
    layers: int = 2,
    data_mesh: int = 2,
    tensor_mesh: int = 4,
    rope_max_timescale: int = 10_000,
) -> ModuleOp:
    tokens = tensor(U32, (("B", batch), ("L", sequence)), sharding={"B": ("d",)})
    sequence_starts = tensor(
        i1,
        (("B", batch), ("L", sequence)),
        sharding={"B": ("d",)},
    )
    embedding = tensor(
        f32,
        (("V", vocabulary), ("M", model)),
        sharding={"V": ("t",), "M": ("d",)},
    )
    layer_norm = tensor(
        f32,
        (("Z", layers), ("M", model)),
        sharding={"M": ("t", "d")},
    )
    query_weights = tensor(
        f32,
        (
            ("Z", layers),
            ("M", model),
            ("Q", query_groups),
            ("K", key_value_heads),
            ("D", head),
        ),
        sharding={"M": ("d",), "K": ("t",)},
    )
    key_value_weights = tensor(
        f32,
        (
            ("Z", layers),
            ("KV", 2),
            ("M", model),
            ("K", key_value_heads),
            ("D", head),
        ),
        sharding={"M": ("d",), "K": ("t",)},
    )
    output_weights = query_weights
    feed_forward_weights = tensor(
        f32,
        (("Z", layers), ("M", model), ("F", feed_forward)),
        sharding={"M": ("d",), "F": ("t",)},
    )
    final_layer_norm = tensor(
        f32,
        (("M", model),),
        sharding={"M": ("t", "d")},
    )
    unembedding = embedding
    inputs = (
        tokens,
        sequence_starts,
        embedding,
        layer_norm,
        layer_norm,
        query_weights,
        key_value_weights,
        output_weights,
        feed_forward_weights,
        feed_forward_weights,
        feed_forward_weights,
        final_layer_norm,
        unembedding,
    )
    builder = DistributedProgramBuilder(
        "seqax_forward",
        {"d": data_mesh, "t": tensor_mesh},
        inputs,
    )
    (
        token_value,
        sequence_start_value,
        embedding_value,
        ln1,
        ln2,
        wq,
        wkv,
        wo,
        wgate,
        wup,
        wdown,
        final_ln,
        unembed,
    ) = builder.inputs

    embedding_bf16 = builder.cast(
        embedding_value,
        tensor(
            bf16,
            (("V", vocabulary), ("M", model)),
            sharding={"V": ("t",), "M": ("d",)},
        ),
        source=_source(137),
    )
    gathered_embedding = builder.all_gather(
        embedding_bf16,
        tensor(
            bf16,
            (("V", vocabulary), ("M", model)),
            sharding={"V": ("t",)},
        ),
        source=_source(137),
    )
    embedded = builder.embedding_lookup(
        gathered_embedding,
        token_value,
        tensor(
            bf16,
            (("B", batch), ("L", sequence), ("M", model)),
            sharding={"B": ("d",)},
            pending_reductions={"t": "sum"},
        ),
        vocabulary_dimension="V",
        source=_source(138),
    )
    x = builder.reduce_scatter(
        embedded,
        tensor(
            bf16,
            (("B", batch), ("L", sequence), ("M", model)),
            sharding={"B": ("d",), "M": ("t",)},
        ),
        axes=("t",),
        scatter_dimensions=("M",),
        source=_source(139),
    )
    causal_mask = builder.packed_causal_mask(
        sequence_start_value,
        tensor(
            i1,
            (("B", batch), ("Qlen", sequence), ("Klen", sequence)),
            sharding={"B": ("d",)},
        ),
        sequence_dimension="L",
        query_dimension="Qlen",
        key_dimension="Klen",
        source=_source(143),
    )

    activation = tensor(
        bf16,
        (("B", batch), ("L", sequence), ("M", model)),
        sharding={"B": ("d",), "M": ("t",)},
    )

    def layer_body(
        body: DistributedProgramBuilder,
        arguments: tuple[SSAValue, ...],
    ) -> tuple[SSAValue, ...]:
        (
            carry,
            layer_ln1,
            layer_ln2,
            layer_wq,
            layer_wkv,
            layer_wo,
            layer_wgate,
            layer_wup,
            layer_wdown,
            mask,
        ) = arguments
        full_model_activation = tensor(
            bf16,
            (("B", batch), ("L", sequence), ("M", model)),
            sharding={"B": ("d",)},
        )
        norm_scale = tensor(f32, (("M", model),))
        gathered_ln1 = body.all_gather(layer_ln1, norm_scale, source=_source(159))
        gathered_x = body.all_gather(carry, full_model_activation, source=_source(160))
        normalized = body.rms_norm(
            gathered_x,
            gathered_ln1,
            full_model_activation,
            dimension="M",
            source=_source(161),
        )

        layer_wq = body.cast(
            layer_wq,
            tensor(
                bf16,
                (("M", model), ("Q", query_groups), ("K", key_value_heads), ("D", head)),
                sharding={"M": ("d",), "K": ("t",)},
            ),
            source=_source(164),
        )
        gathered_wq = body.all_gather(
            layer_wq,
            tensor(
                bf16,
                (("M", model), ("Q", query_groups), ("K", key_value_heads), ("D", head)),
                sharding={"K": ("t",)},
            ),
            source=_source(164),
        )
        query_f32 = body.einsum(
            normalized,
            gathered_wq,
            tensor(
                f32,
                (
                    ("B", batch),
                    ("L", sequence),
                    ("Q", query_groups),
                    ("K", key_value_heads),
                    ("D", head),
                ),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            contracting_dimensions=("M",),
            source=_source(165),
        )
        query = body.cast(
            query_f32,
            tensor(
                bf16,
                (
                    ("B", batch),
                    ("L", sequence),
                    ("Q", query_groups),
                    ("K", key_value_heads),
                    ("D", head),
                ),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            source=_source(165),
        )
        query = body.rename_dimension(
            query,
            tensor(
                bf16,
                (
                    ("B", batch),
                    ("Qlen", sequence),
                    ("Q", query_groups),
                    ("K", key_value_heads),
                    ("D", head),
                ),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            source_dimension="L",
            destination_dimension="Qlen",
            source=_source(165),
        )
        query = body.rotary_embedding(
            query,
            DistributedTensorSpec(
                query.type.element_type,
                query.type.logical_shape(),
                query.type.sharding_axes(),
            ),
            sequence_dimension="Qlen",
            head_dimension="D",
            maximum_timescale=rope_max_timescale,
            source=_source(166),
        )

        layer_wkv = body.cast(
            layer_wkv,
            tensor(
                bf16,
                (("KV", 2), ("M", model), ("K", key_value_heads), ("D", head)),
                sharding={"M": ("d",), "K": ("t",)},
            ),
            source=_source(167),
        )
        gathered_wkv = body.all_gather(
            layer_wkv,
            tensor(
                bf16,
                (
                    ("KV", 2),
                    ("M", model),
                    ("K", key_value_heads),
                    ("D", head),
                ),
                sharding={"K": ("t",)},
            ),
            source=_source(167),
        )
        key_values_f32 = body.einsum(
            normalized,
            gathered_wkv,
            tensor(
                f32,
                (
                    ("KV", 2),
                    ("B", batch),
                    ("L", sequence),
                    ("K", key_value_heads),
                    ("D", head),
                ),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            contracting_dimensions=("M",),
            source=_source(168),
        )
        key_values = body.cast(
            key_values_f32,
            tensor(
                bf16,
                (
                    ("KV", 2),
                    ("B", batch),
                    ("L", sequence),
                    ("K", key_value_heads),
                    ("D", head),
                ),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            source=_source(168),
        )
        key_value = tensor(
            bf16,
            (("B", batch), ("L", sequence), ("K", key_value_heads), ("D", head)),
            sharding={"B": ("d",), "K": ("t",)},
        )
        key = body.slice(
            key_values, key_value, dimension="KV", index=0, source=_source(168)
        )
        value = body.slice(
            key_values, key_value, dimension="KV", index=1, source=_source(168)
        )
        renamed_key_value = tensor(
            bf16,
            (("B", batch), ("Klen", sequence), ("K", key_value_heads), ("D", head)),
            sharding={"B": ("d",), "K": ("t",)},
        )
        key = body.rename_dimension(
            key,
            renamed_key_value,
            source_dimension="L",
            destination_dimension="Klen",
            source=_source(168),
        )
        value = body.rename_dimension(
            value,
            renamed_key_value,
            source_dimension="L",
            destination_dimension="Klen",
            source=_source(168),
        )
        key = body.rotary_embedding(
            key,
            renamed_key_value,
            sequence_dimension="Klen",
            head_dimension="D",
            maximum_timescale=rope_max_timescale,
            source=_source(171),
        )
        logits = body.einsum(
            query,
            key,
            tensor(
                f32,
                (
                    ("B", batch),
                    ("Qlen", sequence),
                    ("Klen", sequence),
                    ("Q", query_groups),
                    ("K", key_value_heads),
                ),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            contracting_dimensions=("D",),
            source=_source(172),
        )
        probabilities = body.masked_softmax(
            logits,
            mask,
            tensor(
                bf16,
                logits.type.logical_shape(),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            dimension="Klen",
            source=_source(176),
        )
        attention_f32 = body.einsum(
            probabilities,
            value,
            tensor(
                f32,
                (
                    ("B", batch),
                    ("Qlen", sequence),
                    ("Q", query_groups),
                    ("K", key_value_heads),
                    ("D", head),
                ),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            contracting_dimensions=("Klen",),
            source=_source(177),
        )
        attention = body.cast(
            attention_f32,
            tensor(
                bf16,
                attention_f32.type.logical_shape(),
                sharding={"B": ("d",), "K": ("t",)},
            ),
            source=_source(177),
        )
        layer_wo = body.cast(
            layer_wo,
            tensor(
                bf16,
                (("M", model), ("Q", query_groups), ("K", key_value_heads), ("D", head)),
                sharding={"M": ("d",), "K": ("t",)},
            ),
            source=_source(178),
        )
        gathered_wo = body.all_gather(
            layer_wo,
            tensor(
                bf16,
                (("M", model), ("Q", query_groups), ("K", key_value_heads), ("D", head)),
                sharding={"K": ("t",)},
            ),
            source=_source(178),
        )
        attention_partial = body.einsum(
            attention,
            gathered_wo,
            tensor(
                f32,
                (("B", batch), ("Qlen", sequence), ("M", model)),
                sharding={"B": ("d",)},
                pending_reductions={"t": "sum"},
            ),
            contracting_dimensions=("D", "K", "Q"),
            source=_source(179),
        )
        attention_output = body.reduce_scatter(
            attention_partial,
            tensor(
                f32,
                (("B", batch), ("Qlen", sequence), ("M", model)),
                sharding={"B": ("d",), "M": ("t",)},
            ),
            axes=("t",),
            scatter_dimensions=("M",),
            source=_source(180),
        )
        attention_output = body.rename_dimension(
            attention_output,
            tensor(
                f32,
                (("B", batch), ("L", sequence), ("M", model)),
                sharding={"B": ("d",), "M": ("t",)},
            ),
            source_dimension="Qlen",
            destination_dimension="L",
            source=_source(180),
        )
        attention_output = body.cast(
            attention_output, activation, source=_source(180)
        )
        carry = body.elementwise(
            carry,
            attention_output,
            result=activation,
            function="add",
            source=_source(181),
        )

        gathered_ln2 = body.all_gather(layer_ln2, norm_scale, source=_source(184))
        gathered_x = body.all_gather(carry, full_model_activation, source=_source(185))
        normalized = body.rms_norm(
            gathered_x,
            gathered_ln2,
            full_model_activation,
            dimension="M",
            source=_source(186),
        )
        projected = tensor(
            f32,
            (("B", batch), ("L", sequence), ("F", feed_forward)),
            sharding={"B": ("d",), "F": ("t",)},
        )
        projected_bf16 = tensor(
            bf16,
            projected.dimensions,
            sharding={"B": ("d",), "F": ("t",)},
        )

        def project(weight: SSAValue, line: int) -> SSAValue:
            weight = body.cast(
                weight,
                tensor(
                    bf16,
                    (("M", model), ("F", feed_forward)),
                    sharding={"M": ("d",), "F": ("t",)},
                ),
                source=_source(line),
            )
            gathered = body.all_gather(
                weight,
                tensor(
                    bf16,
                    (("M", model), ("F", feed_forward)),
                    sharding={"F": ("t",)},
                ),
                source=_source(line),
            )
            result = body.einsum(
                normalized,
                gathered,
                projected,
                contracting_dimensions=("M",),
                source=_source(line + 1),
            )
            return body.cast(result, projected_bf16, source=_source(line + 1))

        gate = project(layer_wgate, 189)
        up = project(layer_wup, 191)
        gate = body.elementwise(
            gate,
            result=projected_bf16,
            function="silu",
            source=_source(193),
        )
        feed_forward_value = body.elementwise(
            gate,
            up,
            result=projected_bf16,
            function="multiply",
            source=_source(193),
        )
        layer_wdown = body.cast(
            layer_wdown,
            tensor(
                bf16,
                (("M", model), ("F", feed_forward)),
                sharding={"M": ("d",), "F": ("t",)},
            ),
            source=_source(194),
        )
        gathered_down = body.all_gather(
            layer_wdown,
            tensor(
                bf16,
                (("M", model), ("F", feed_forward)),
                sharding={"F": ("t",)},
            ),
            source=_source(194),
        )
        feed_forward_partial = body.einsum(
            feed_forward_value,
            gathered_down,
            tensor(
                f32,
                (("B", batch), ("L", sequence), ("M", model)),
                sharding={"B": ("d",)},
                pending_reductions={"t": "sum"},
            ),
            contracting_dimensions=("F",),
            source=_source(195),
        )
        feed_forward_output = body.reduce_scatter(
            feed_forward_partial,
            tensor(
                f32,
                (("B", batch), ("L", sequence), ("M", model)),
                sharding={"B": ("d",), "M": ("t",)},
            ),
            axes=("t",),
            scatter_dimensions=("M",),
            source=_source(196),
        )
        feed_forward_output = body.cast(
            feed_forward_output, activation, source=_source(196)
        )
        return (
            body.elementwise(
                carry,
                feed_forward_output,
                result=activation,
                function="add",
                source=_source(198),
            ),
        )

    (x,) = builder.layer_scan(
        (x, ln1, ln2, wq, wkv, wo, wgate, wup, wdown, causal_mask),
        layer_body,
        carry_count=1,
        stacked_count=8,
        layer_dimension="Z",
        trip_count=layers,
        source=_source(200),
    )
    gathered_x = builder.all_gather(
        x,
        tensor(
            bf16,
            (("B", batch), ("L", sequence), ("M", model)),
            sharding={"B": ("d",)},
        ),
        source=_source(203),
    )
    gathered_final_ln = builder.all_gather(
        final_ln,
        tensor(f32, (("M", model),)),
        source=_source(204),
    )
    normalized = builder.rms_norm(
        gathered_x,
        gathered_final_ln,
        tensor(
            bf16,
            (("B", batch), ("L", sequence), ("M", model)),
            sharding={"B": ("d",)},
        ),
        dimension="M",
        source=_source(205),
    )
    unembed = builder.cast(
        unembed,
        tensor(
            bf16,
            (("V", vocabulary), ("M", model)),
            sharding={"V": ("t",), "M": ("d",)},
        ),
        source=_source(206),
    )
    gathered_unembed = builder.all_gather(
        unembed,
        tensor(
            bf16,
            (("V", vocabulary), ("M", model)),
            sharding={"V": ("t",)},
        ),
        source=_source(206),
    )
    logits = builder.einsum(
        normalized,
        gathered_unembed,
        tensor(
            f32,
            (("B", batch), ("L", sequence), ("V", vocabulary)),
            sharding={"B": ("d",), "V": ("t",)},
        ),
        contracting_dimensions=("M",),
        source=_source(207),
    )
    return builder.module(logits)
