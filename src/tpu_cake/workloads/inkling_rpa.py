from __future__ import annotations

import math
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
from xdsl.dialects.builtin import ModuleOp, bf16, i32

from tpu_cake.contracts import (
    BenchmarkProtocol,
    ExecutionContract,
    KernelExperiment,
    NumericalContract,
    ProfileExpectation,
    SearchPolicy,
    SourceFileContract,
    TargetHardware,
    TensorContract,
    WorkloadContract,
)
from tpu_cake.dialects.tpu_schedule import MemorySpace, Ownership
from tpu_cake.frontend import KernelBuilder, buffer, schedule_sha256
from tpu_cake.rpa_lowering import (
    FusedRpaPlan,
    lower_inkling_rpa_to_pallas,
)
from tpu_cake.source import SourceLocation


def inkling_rpa_reference(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    page_table: np.ndarray,
    sequence_lengths: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    batch, query_heads, query_dimension = query.shape
    _, key_page_size, key_heads, key_dimension = key_cache.shape
    _, value_page_size, value_heads, value_dimension = value_cache.shape
    if key_page_size != value_page_size:
        raise ValueError("key and value caches must use the same page size")
    if query_dimension != key_dimension:
        raise ValueError("query and key head dimensions must match")
    if query_heads % key_heads or query_heads % value_heads:
        raise ValueError("query heads must divide evenly across key and value heads")
    if bias.shape[0] != query_heads:
        raise ValueError("relative bias must provide one row per query head")
    key_head_for_query = np.arange(query_heads) // (query_heads // key_heads)
    value_head_for_query = np.arange(query_heads) // (query_heads // value_heads)
    page_size = key_cache.shape[1]
    output = np.empty((batch, query_heads, value_dimension), dtype=np.float32)
    for request in range(batch):
        length = int(sequence_lengths[request])
        pages = page_table[request, : math.ceil(length / page_size)]
        keys = key_cache[pages].reshape(-1, key_heads, key_dimension)[:length]
        values = value_cache[pages].reshape(-1, value_heads, value_dimension)[:length]
        query_keys = keys[:, key_head_for_query, :]
        query_values = values[:, value_head_for_query, :]
        scores = np.einsum("hd,lhd->hl", query[request], query_keys, dtype=np.float32)
        scores = scores / query_dimension + bias[:, :length]
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        output[request] = np.einsum("hl,lhd->hd", probabilities, query_values, dtype=np.float32)
    return output


def inkling_rpa_inputs(seed: int = 0) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    query = generator.normal(size=(4, 4, 32)).astype(np.float32)
    key_cache = generator.normal(size=(32, 16, 4, 32)).astype(np.float32)
    value_cache = generator.normal(size=(32, 16, 4, 32)).astype(np.float32)
    page_table = np.arange(32, dtype=np.int32).reshape(4, 8)
    sequence_lengths = np.array([17, 33, 65, 127], dtype=np.int32)
    bias = generator.normal(scale=0.05, size=(4, 128)).astype(np.float32)
    return query, key_cache, value_cache, page_table, sequence_lengths, bias


def inkling_rpa_schedule() -> ModuleOp:
    external = {
        "memory": MemorySpace.HBM,
        "ownership": Ownership.EXTERNAL,
        "lifetime": (0, 4),
    }
    inputs = (
        buffer((4, 4, 32), "B H D", bf16, **external),
        buffer((32, 16, 4, 32), "P S H D", bf16, **external),
        buffer((32, 16, 4, 32), "P S H D", bf16, **external),
        buffer((4, 8), "B MP", i32, **external),
        buffer((4,), "B", i32, **external),
        buffer((4, 128), "H L", bf16, **external),
        buffer((4, 4, 32), "B H D", bf16, **external),
    )
    builder = KernelBuilder(
        "inkling_rpa_decode",
        "tpu7x",
        inputs,
        vmem_capacity_bytes=2 << 20,
        smem_capacity_bytes=1 << 16,
    )
    query = builder.alloc(
        buffer((4, 4, 32), "B H D", bf16, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        "query_tile",
    )
    bias = builder.alloc(
        buffer((4, 128), "H L", bf16, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        "relative_bias",
    )
    output = builder.alloc(
        buffer((4, 4, 32), "B H D", bf16, memory=MemorySpace.VMEM, lifetime=(2, 4)),
        "attention_output",
    )
    query_semaphore = builder.semaphore()
    bias_semaphore = builder.semaphore()
    output_semaphore = builder.semaphore()
    query_dma = builder.dma_start(builder.inputs[0], query, query_semaphore, stage=0)
    bias_dma = builder.dma_start(builder.inputs[5], bias, bias_semaphore, stage=0)
    builder.dma_wait(query_dma, stage=1)
    builder.dma_wait(bias_dma, stage=1)
    builder.ragged_paged_attention(
        query,
        builder.inputs[1],
        builder.inputs[2],
        builder.inputs[3],
        builder.inputs[4],
        bias,
        output,
        stage=2,
        query_block_size=1,
        kv_block_size=8,
    )
    output_dma = builder.dma_start(output, builder.inputs[6], output_semaphore, stage=3)
    builder.dma_wait(output_dma, stage=4)
    return builder.module()


def inkling_fused_rpa_schedule(
    decode_block_sizes: tuple[int, int, int, int] = (8, 128, 8, 128),
) -> ModuleOp:
    query_block_size, kv_block_size, query_cluster_size, kv_cluster_size = (
        decode_block_sizes
    )
    external = {
        "memory": MemorySpace.HBM,
        "ownership": Ownership.EXTERNAL,
        "lifetime": (0, 0),
    }
    inputs = (
        buffer((4, 4, 32), "T Hq D", bf16, **external),
        buffer((4, 2, 32), "T Hkv D", bf16, **external),
        buffer((4, 2, 32), "T Hkv D", bf16, **external),
        buffer((32, 16, 2, 2, 128), "P S Hkv2p Pack Dpad", bf16, **external),
        buffer((4,), "N", i32, **external),
        buffer((32,), "PI", i32, **external),
        buffer((5,), "N1", i32, **external),
        buffer((5,), "N1", i32, **external),
        buffer((3,), "R3", i32, **external),
        buffer((4, 4, 16), "T Hq R", bf16, **external),
        buffer((16, 128), "R E", bf16, **external),
    )
    builder = KernelBuilder(
        "inkling_fused_rpa_decode",
        "tpu7x",
        inputs,
        vmem_capacity_bytes=96 << 20,
        smem_capacity_bytes=1 << 20,
    )
    builder.fused_ragged_paged_attention(
        *builder.inputs,
        builder.inputs[0],
        builder.inputs[3],
        stage=0,
        causal=1,
        softmax_scale="0.03125",
        softmax_dtype="float32",
        sliding_window=0,
        query_block_size=query_block_size,
        kv_block_size=kv_block_size,
        query_cluster_size=query_cluster_size,
        kv_cluster_size=kv_cluster_size,
        vmem_limit_bytes=96 << 20,
        source_location=SourceLocation(
            "engine/sglang-jax/python/sgl_jax/srt/kernels/"
            "ragged_paged_attention/ragged_paged_attention_v3.py",
            1802,
            1,
        ),
    )
    return builder.module()


class FusedRpaOracleMutation(StrEnum):
    WRONG_SCALE = "wrong_scale"
    OMIT_RELATIVE_BIAS = "omit_relative_bias"
    WRONG_PAGE_INDEX = "wrong_page_index"
    IGNORE_PAGE_INDICES = "ignore_page_indices"
    SWAP_KV_INTERLEAVE = "swap_kv_interleave"
    SKIP_CACHE_UPDATE = "skip_cache_update"


def inkling_fused_rpa_inputs(seed: int = 0) -> tuple[jax.Array, ...]:
    generator = np.random.default_rng(seed)

    def bf16(shape: tuple[int, ...], scale: float = 1.0) -> jax.Array:
        values = generator.normal(scale=scale, size=shape).astype(np.float32)
        return jnp.asarray(values, dtype=jnp.bfloat16)

    return (
        bf16((4, 4, 32)),
        bf16((4, 2, 32)),
        bf16((4, 2, 32)),
        bf16((32, 16, 2, 2, 128), scale=0.25),
        jnp.asarray((1, 17, 33, 49), dtype=jnp.int32),
        jnp.asarray(
            (
                7,
                2,
                19,
                4,
                25,
                1,
                16,
                9,
                30,
                12,
                0,
                5,
                8,
                11,
                14,
                17,
                20,
                23,
                26,
                29,
                3,
                6,
                10,
                13,
                15,
                18,
                21,
                22,
                24,
                27,
                28,
                31,
            ),
            dtype=jnp.int32,
        ),
        jnp.arange(5, dtype=jnp.int32),
        jnp.asarray((0, 16, 48, 96, 160), dtype=jnp.int32),
        jnp.full((3,), 4, dtype=jnp.int32),
        bf16((4, 4, 16), scale=0.4),
        bf16((16, 128), scale=0.4),
    )


def inkling_fused_rpa_reference(
    inputs: tuple[jax.Array, ...],
    *,
    mutation: FusedRpaOracleMutation | None = None,
) -> tuple[jax.Array, jax.Array]:
    (
        queries_value,
        keys_value,
        values_value,
        cache_value,
        kv_lengths_value,
        page_indices_value,
        cumulative_query_lengths_value,
        cumulative_kv_lengths_value,
        distribution_value,
        relative_states_value,
        relative_projection_value,
    ) = inputs
    queries = np.asarray(queries_value, dtype=np.float32)
    keys = np.asarray(keys_value, dtype=np.float32)
    values = np.asarray(values_value, dtype=np.float32)
    cache = np.asarray(cache_value, dtype=np.float32).copy()
    kv_lengths = np.asarray(kv_lengths_value, dtype=np.int32)
    page_indices = np.asarray(page_indices_value, dtype=np.int32).copy()
    cumulative_query_lengths = np.asarray(cumulative_query_lengths_value, dtype=np.int32)
    cumulative_kv_lengths = np.asarray(cumulative_kv_lengths_value, dtype=np.int32)
    distribution = np.asarray(distribution_value, dtype=np.int32)
    relative_states = np.asarray(relative_states_value, dtype=np.float32)
    relative_projection = np.asarray(relative_projection_value, dtype=np.float32)
    sequence_count = kv_lengths.shape[0]
    if not np.array_equal(distribution, np.full((3,), sequence_count, dtype=np.int32)):
        raise ValueError("fused RPA oracle requires a decode-only distribution")
    if not np.array_equal(
        cumulative_query_lengths,
        np.arange(sequence_count + 1, dtype=np.int32),
    ):
        raise ValueError("fused RPA oracle requires one query token per sequence")
    if mutation is FusedRpaOracleMutation.IGNORE_PAGE_INDICES:
        page_indices = np.arange(page_indices.size, dtype=np.int32)
    elif mutation is FusedRpaOracleMutation.WRONG_PAGE_INDEX:
        page_indices[0] = (page_indices[0] + 1) % cache.shape[0]

    page_size = cache.shape[1]
    packing = cache.shape[3]
    key_slot, value_slot = (
        (1, 0) if mutation is FusedRpaOracleMutation.SWAP_KV_INTERLEAVE else (0, 1)
    )
    if mutation is not FusedRpaOracleMutation.SKIP_CACHE_UPDATE:
        for sequence in range(sequence_count):
            position = int(kv_lengths[sequence]) - 1
            page_offset = int(cumulative_kv_lengths[sequence]) // page_size
            page = page_indices[page_offset + position // page_size]
            offset = position % page_size
            for head in range(keys.shape[1]):
                for slot, source in ((key_slot, keys), (value_slot, values)):
                    interleaved = 2 * head + slot
                    destination = cache[
                        page,
                        offset,
                        interleaved // packing,
                        interleaved % packing,
                    ]
                    destination.fill(0)
                    destination[: source.shape[-1]] = source[sequence, head]

    unpacked = cache.reshape(*cache.shape[:2], cache.shape[2] * packing, cache.shape[-1])
    outputs = np.empty_like(queries, dtype=np.float32)
    scale = (
        1.0 / math.sqrt(queries.shape[-1])
        if mutation is FusedRpaOracleMutation.WRONG_SCALE
        else 1.0 / queries.shape[-1]
    )
    for sequence in range(sequence_count):
        length = int(kv_lengths[sequence])
        page_offset = int(cumulative_kv_lengths[sequence]) // page_size
        page_count = math.ceil(length / page_size)
        pages = page_indices[page_offset : page_offset + page_count]
        sequence_cache = unpacked[pages].reshape(-1, unpacked.shape[2], unpacked.shape[3])[:length]
        key_cache = sequence_cache[:, key_slot::2, : queries.shape[-1]][:, : keys.shape[1]]
        value_cache = sequence_cache[:, value_slot::2, : queries.shape[-1]][:, : values.shape[1]]
        query_heads_per_kv_head = queries.shape[1] // keys.shape[1]
        head_index = np.arange(queries.shape[1]) // query_heads_per_kv_head
        scores = (
            np.einsum(
                "hd,lhd->hl",
                queries[sequence],
                key_cache[:, head_index],
                dtype=np.float32,
            )
            * scale
        )
        if mutation is not FusedRpaOracleMutation.OMIT_RELATIVE_BIAS:
            distances = (length - 1) - np.arange(length, dtype=np.int32)
            clipped = np.clip(distances, 0, relative_projection.shape[1] - 1)
            projection = relative_projection[:, clipped]
            bias = np.einsum(
                "hr,rl->hl",
                relative_states[sequence],
                projection,
                dtype=np.float32,
            )
            scores += np.where(
                distances[None] < relative_projection.shape[1],
                bias,
                0.0,
            )
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores, dtype=np.float32)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        outputs[sequence] = np.einsum(
            "hl,lhd->hd",
            probabilities,
            value_cache[:, head_index],
            dtype=np.float32,
        )
    return (
        jnp.asarray(outputs, dtype=jnp.bfloat16),
        jnp.asarray(cache, dtype=jnp.bfloat16),
    )


def inkling_rpa_contract() -> WorkloadContract:
    def tensor(
        name: str, shape: tuple[int, ...], logical: tuple[str, ...], dtype: str
    ) -> TensorContract:
        return TensorContract(
            name=name, shape=shape, logical_shape=logical, dtype=dtype, sharding=("",) * len(shape)
        )

    return WorkloadContract(
        name="inkling-ragged-paged-relative-bias-attention",
        stage="steady_decode",
        inputs=(
            tensor("query", (4, 4, 32), ("B", "H", "D"), "bf16"),
            tensor("key_cache", (32, 16, 4, 32), ("P", "S", "H", "D"), "bf16"),
            tensor("value_cache", (32, 16, 4, 32), ("P", "S", "H", "D"), "bf16"),
            tensor("page_table", (4, 8), ("B", "MP"), "i32"),
            tensor("sequence_lengths", (4,), ("B",), "i32"),
            tensor("relative_bias", (4, 128), ("H", "L"), "bf16"),
        ),
        outputs=(tensor("output", (4, 4, 32), ("B", "H", "D"), "bf16"),),
        numerical=NumericalContract(
            reference="tpu_cake.workloads.inkling_rpa.inkling_rpa_reference",
            absolute_tolerance=0.02,
            relative_tolerance=0.02,
        ),
        execution=ExecutionContract(
            executor="tpu_cake.workloads.inkling_rpa.inkling_rpa_reference",
            scope="conceptual-separate-cache-prototype",
        ),
    )


def inkling_fused_rpa_contract(plan: FusedRpaPlan | None = None) -> WorkloadContract:
    if plan is None:
        plan = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())

    def tensor(
        name: str, shape: tuple[int, ...], logical: tuple[str, ...], dtype: str
    ) -> TensorContract:
        return TensorContract(
            name=name,
            shape=shape,
            logical_shape=logical,
            dtype=dtype,
            sharding=("",) * len(shape),
        )

    input_metadata = (
        ("queries", ("T", "Hq", "D")),
        ("keys", ("T", "Hkv", "D")),
        ("values", ("T", "Hkv", "D")),
        ("fused_cache", ("P", "S", "Hkv2p", "Pack", "Dpad")),
        ("kv_lengths", ("N",)),
        ("page_indices", ("PI",)),
        ("cumulative_query_lengths", ("N1",)),
        ("cumulative_kv_lengths", ("N1",)),
        ("distribution", ("R3",)),
        ("relative_states", ("T", "Hq", "R")),
        ("relative_projection", ("R", "E")),
    )
    inputs = tuple(
        tensor(name, shape, logical, dtype)
        for (name, logical), shape, dtype in zip(
            input_metadata,
            plan.input_shapes,
            plan.input_dtypes,
            strict=True,
        )
    )
    outputs = (
        tensor("output", plan.output_shape, ("T", "Hq", "D"), plan.output_dtypes[0]),
        tensor(
            "updated_fused_cache",
            plan.fused_cache_shape,
            ("P", "S", "Hkv2p", "Pack", "Dpad"),
            plan.output_dtypes[1],
        ),
    )
    return WorkloadContract(
        name="inkling-fused-ragged-paged-relative-bias-decode-local-shard",
        stage="steady_decode",
        inputs=inputs,
        outputs=outputs,
        numerical=NumericalContract(
            reference="tpu_cake.workloads.inkling_rpa.inkling_fused_rpa_reference",
            absolute_tolerance=0.02,
            relative_tolerance=0.02,
        ),
        execution=ExecutionContract(
            executor=(
                "sgl_jax.srt.kernels.ragged_paged_attention."
                "ragged_paged_attention_v3.ragged_paged_attention"
            ),
            scope="local-shard-caller-owned-sharding",
            preflight="tpu_cake.rpa_lowering.FusedRpaPlan.preflight",
            source_revision=plan.backend_repository_revision,
            source_manifest=tuple(
                SourceFileContract(path=path, sha256=sha256)
                for path, sha256 in plan.backend_manifest
            ),
        ),
    )


def inkling_fused_rpa_experiment(
    decode_block_sizes: tuple[int, int, int, int] = (8, 128, 8, 128),
) -> KernelExperiment:
    schedule = inkling_fused_rpa_schedule(decode_block_sizes)
    plan = lower_inkling_rpa_to_pallas(schedule)
    return KernelExperiment(
        workload=inkling_fused_rpa_contract(plan),
        target=TargetHardware(
            accelerator="TPU7x",
            topology="local shard; caller owns outer mesh",
            chip_count=1,
            vmem_budget_bytes_per_core=128 << 20,
            smem_budget_bytes_per_core=32 << 20,
            runtime_target="Pallas Mosaic TPU through pinned sglang-jax RPA v3",
        ),
        benchmark=BenchmarkProtocol(
            warmup_iterations=5,
            measured_iterations=50,
            synchronization="block until output and updated cache are ready",
            statistic="median synchronized local-shard invocation wall duration",
        ),
        search=SearchPolicy(objective_metric="median_synchronized_invocation_ns"),
        profile=ProfileExpectation(
            name="inkling-fused-rpa-local-shard-decode",
            stage="steady_decode",
            minimum_tpu_device_planes=1,
            require_tensor_core_activity=False,
            required_timed_hlo_markers=("ragged_paged_attention", "pallas_call"),
            forbidden_timed_hlo_fragments=("native_backend.py:500",),
        ),
        schedule_sha256=plan.schedule_sha256,
    )


def inkling_rpa_experiment() -> KernelExperiment:
    schedule = inkling_rpa_schedule()
    return KernelExperiment(
        workload=inkling_rpa_contract(),
        target=TargetHardware(
            accelerator="TPU7x",
            topology="4x1",
            chip_count=4,
            vmem_budget_bytes_per_core=2 << 20,
            smem_budget_bytes_per_core=1 << 16,
            runtime_target="Pallas Mosaic TPU",
        ),
        benchmark=BenchmarkProtocol(
            warmup_iterations=5,
            measured_iterations=50,
            synchronization="block until device result is ready",
            statistic="median device duration",
        ),
        search=SearchPolicy(objective_metric="median_device_duration_ns"),
        profile=ProfileExpectation(
            name="inkling-rpa-steady-decode",
            stage="steady_decode",
            minimum_tpu_device_planes=8,
            required_timed_hlo_markers=("ragged_paged_attention",),
            forbidden_timed_hlo_fragments=("12582912,16", "native_backend.py:500"),
        ),
        schedule_sha256=schedule_sha256(schedule),
    )
