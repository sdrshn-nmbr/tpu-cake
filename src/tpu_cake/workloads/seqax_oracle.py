from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

SEQAX_REFERENCE_DECIMALS = 6


def seqax_forward_inputs(
    *,
    seed: int,
    batch: int,
    sequence: int,
    model: int,
    vocabulary: int,
    feed_forward: int,
    query_groups: int,
    key_value_heads: int,
    head: int,
    layers: int,
    **_unused: int,
) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)

    def weights(shape: tuple[int, ...]) -> np.ndarray:
        return generator.normal(scale=0.08, size=shape).astype(np.float32)

    tokens = generator.integers(0, vocabulary, size=(batch, sequence), dtype=np.uint32)
    sequence_starts = np.zeros((batch, sequence), dtype=np.bool_)
    sequence_starts[:, 0] = True
    if sequence >= 4:
        sequence_starts[1::2, sequence // 2] = True
    ln1 = generator.uniform(0.75, 1.25, size=(layers, model)).astype(np.float32)
    ln2 = generator.uniform(1.25, 1.75, size=(layers, model)).astype(np.float32)
    query_shape = (layers, model, query_groups, key_value_heads, head)
    feed_forward_shape = (layers, model, feed_forward)
    return (
        tokens,
        sequence_starts,
        weights((vocabulary, model)),
        ln1,
        ln2,
        weights(query_shape),
        weights((layers, 2, model, key_value_heads, head)),
        weights(query_shape),
        weights(feed_forward_shape),
        weights(feed_forward_shape),
        weights(feed_forward_shape),
        np.ones((model,), dtype=np.float32),
        weights((vocabulary, model)),
    )


def _rms_norm(value: jax.Array, scale: jax.Array) -> jax.Array:
    mean_square = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
    normalized = value * jax.lax.rsqrt(mean_square + 1e-6)
    return (normalized * scale).astype(jnp.bfloat16)


def _rope(value: jax.Array, maximum_timescale: int) -> jax.Array:
    sequence = value.shape[1]
    half_head = value.shape[-1] // 2
    timescale = jnp.logspace(
        0,
        jnp.log10(jnp.float32(maximum_timescale)),
        half_head,
        endpoint=False,
    )
    position = jnp.arange(sequence, dtype=jnp.int32)
    angle = position[:, None].astype(jnp.float32) / timescale[None, :]
    shape = (1, sequence, *(1 for _ in value.shape[2:-1]), half_head)
    sine = jnp.sin(angle).reshape(shape)
    cosine = jnp.cos(angle).reshape(shape)
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine),
        axis=-1,
    )


def _seqax_forward_reference_on_current_device(
    inputs: tuple[np.ndarray, ...],
    *,
    rope_max_timescale: int,
    **_parameters: int,
) -> np.ndarray:
    (
        token_ids,
        sequence_starts,
        embedding,
        ln1,
        ln2,
        query_weights,
        key_value_weights,
        output_weights,
        gate_weights,
        up_weights,
        down_weights,
        final_ln,
        unembedding,
    ) = (jnp.asarray(value) for value in inputs)
    x = embedding.astype(jnp.bfloat16)[token_ids]
    segment_ids = jnp.cumsum(sequence_starts, axis=1)
    segment_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = jnp.tril(jnp.ones(segment_mask.shape[1:], dtype=jnp.bool_))
    mask = (segment_mask & causal[None, :, :])[:, :, :, None, None]

    for layer in range(query_weights.shape[0]):
        normalized = _rms_norm(x, ln1[layer].astype(jnp.float32))
        query = jnp.einsum(
            "blm,mqkd->blqkd",
            normalized,
            query_weights[layer].astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        query = _rope(query, rope_max_timescale)
        key_value = jnp.einsum(
            "blm,vmkd->vblkd",
            normalized,
            key_value_weights[layer].astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        key = _rope(key_value[0], rope_max_timescale)
        value = key_value[1]
        logits = jnp.einsum(
            "blqkd,bskd->blsqk",
            query,
            key,
            preferred_element_type=jnp.float32,
        )
        probabilities = jax.nn.softmax(jnp.where(mask, logits, -1e10), axis=2).astype(
            jnp.bfloat16
        )
        attention = jnp.einsum(
            "blsqk,bskd->blqkd",
            probabilities,
            value,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        attention_output = jnp.einsum(
            "blqkd,mqkd->blm",
            attention,
            output_weights[layer].astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        x = (x + attention_output).astype(jnp.bfloat16)

        normalized = _rms_norm(x, ln2[layer].astype(jnp.float32))
        gate = jnp.einsum(
            "blm,mf->blf",
            normalized,
            gate_weights[layer].astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        up = jnp.einsum(
            "blm,mf->blf",
            normalized,
            up_weights[layer].astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        hidden = jax.nn.silu(gate).astype(jnp.bfloat16) * up
        feed_forward = jnp.einsum(
            "blf,mf->blm",
            hidden,
            down_weights[layer].astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        x = (x + feed_forward).astype(jnp.bfloat16)

    normalized = _rms_norm(x, final_ln.astype(jnp.float32))
    logits = jnp.einsum(
        "blm,vm->blv",
        normalized,
        unembedding.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    return np.asarray(logits, dtype=np.float32)


def seqax_forward_reference(
    inputs: tuple[np.ndarray, ...],
    *,
    rope_max_timescale: int,
    **parameters: int,
) -> np.ndarray:
    return _seqax_forward_reference_on_current_device(
        inputs,
        rope_max_timescale=rope_max_timescale,
        **parameters,
    )


def seqax_forward_canonical_reference(
    inputs: tuple[np.ndarray, ...],
    *,
    rope_max_timescale: int,
    quantization_decimals: int = SEQAX_REFERENCE_DECIMALS,
    **parameters: int,
) -> np.ndarray:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("SEQAX_REFERENCE_CPU_UNAVAILABLE")
    with jax.default_device(cpu_devices[0]):
        result = _seqax_forward_reference_on_current_device(
            inputs,
            rope_max_timescale=rope_max_timescale,
            **parameters,
        )
    return np.round(result, decimals=quantization_decimals).astype(np.float32)
