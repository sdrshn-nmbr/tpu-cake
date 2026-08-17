from __future__ import annotations

import math

import numpy as np
from xdsl.dialects.builtin import ModuleOp, bf16, i32

from tpu_cake.contracts import (
    BenchmarkProtocol,
    KernelExperiment,
    NumericalContract,
    ProfileExpectation,
    SearchPolicy,
    TargetHardware,
    TensorContract,
    WorkloadContract,
)
from tpu_cake.dialects.tpu_schedule import MemorySpace, Ownership
from tpu_cake.frontend import KernelBuilder, buffer, schedule_sha256


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
        scores = scores / math.sqrt(query_dimension) + bias[:, :length]
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        output[request] = np.einsum(
            "hl,lhd->hd", probabilities, query_values, dtype=np.float32
        )
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
