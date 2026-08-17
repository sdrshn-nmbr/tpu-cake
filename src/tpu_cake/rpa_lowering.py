from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from xdsl.dialects.builtin import BFloat16Type, IntegerType, ModuleOp
from xdsl.ir import Operation

from tpu_cake.dialects.tpu_schedule import (
    BufferType,
    FusedRaggedPagedAttentionOp,
    KernelOp,
    YieldOp,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.lowering import UnsupportedLoweringError
from tpu_cake.source import verify_with_sources

INKLE_REPOSITORY_REVISION = "c3d805516e0f3bc0c6f621f2590489aff12f8a59"
INKLING_RPA_FILE_REVISION = "e65c204fe42692254b0f7d6f678415dfc6fca401"
INKLING_RPA_SOURCE_SHA256 = "56d00d027cf921def1908e4815ced12e79210e1ac3cf57bcd727c5e6c6168eaa"
INKLING_RPA_UTIL_SHA256 = "fe76b20b6791ce3e6a30ff22a0e7de11b9a809113facbedb79b52079191f9562"
INKLING_RPA_TUNING_SHA256 = "563d74378b319710c8a7c6cd89ec174563462695417f9167d65716d74a4a2d50"
INKLING_RPA_BASE_TUNING_SHA256 = (
    "4577624d0ef37726cbcceb5d570530399eb1f9113f8736e621a4e9aba9d3caef"
)
INKLING_RPA_MODULE = (
    "sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3"
)
RPA_EXECUTION_SCHEMA = "sglang-jax-rpa-v3-adapter-v1"


@dataclass(frozen=True)
class FusedRpaPlan:
    name: str
    schedule_sha256: str
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    fused_cache_shape: tuple[int, ...]
    kv_lengths_shape: tuple[int, ...]
    page_indices_shape: tuple[int, ...]
    cumulative_query_lengths_shape: tuple[int, ...]
    cumulative_kv_lengths_shape: tuple[int, ...]
    distribution_shape: tuple[int, ...]
    relative_states_shape: tuple[int, ...]
    relative_projection_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, str]
    decode_block_sizes: tuple[int, int, int, int]
    causal: int
    softmax_scale: float
    softmax_dtype: str
    sliding_window: int
    vmem_limit_bytes: int
    backend_module: str = INKLING_RPA_MODULE
    backend_repository_revision: str = INKLE_REPOSITORY_REVISION
    backend_file_revision: str = INKLING_RPA_FILE_REVISION
    backend_sha256: str = INKLING_RPA_SOURCE_SHA256
    backend_manifest: tuple[tuple[str, str], ...] = (
        ("ragged_paged_attention_v3.py", INKLING_RPA_SOURCE_SHA256),
        ("tuned_block_sizes.py", INKLING_RPA_BASE_TUNING_SHA256),
        ("tuned_block_sizes_v3.py", INKLING_RPA_TUNING_SHA256),
        ("util.py", INKLING_RPA_UTIL_SHA256),
    )
    execution_scope: str = "local-shard-caller-owned-sharding"

    @property
    def input_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            self.query_shape,
            self.key_shape,
            self.value_shape,
            self.fused_cache_shape,
            self.kv_lengths_shape,
            self.page_indices_shape,
            self.cumulative_query_lengths_shape,
            self.cumulative_kv_lengths_shape,
            self.distribution_shape,
            self.relative_states_shape,
            self.relative_projection_shape,
        )

    def _validate_signature(self, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != len(self.input_shapes):
            raise ValueError(
                f"fused RPA needs {len(self.input_shapes)} inputs, got {len(inputs)}"
            )
        for index, (value, shape, dtype) in enumerate(
            zip(inputs, self.input_shapes, self.input_dtypes, strict=True)
        ):
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"fused RPA input {index} has shape {tuple(value.shape)}, expected {shape}"
                )
            if str(value.dtype) != dtype:
                raise ValueError(
                    f"fused RPA input {index} has dtype {value.dtype}, expected {dtype}"
                )

    def preflight(self, *inputs: Any) -> None:
        self._validate_signature(inputs)
        kv_lengths = np.asarray(inputs[4], dtype=np.int64)
        page_indices = np.asarray(inputs[5], dtype=np.int64)
        cumulative_query_lengths = np.asarray(inputs[6], dtype=np.int64)
        cumulative_kv_lengths = np.asarray(inputs[7], dtype=np.int64)
        distribution = np.asarray(inputs[8], dtype=np.int64)
        sequence_count = self.kv_lengths_shape[0]
        expected_distribution = np.full((3,), sequence_count, dtype=np.int64)
        if not np.array_equal(distribution, expected_distribution):
            raise ValueError(
                "Inkling fused RPA adapter accepts decode-only distribution "
                f"{tuple(expected_distribution)}, got {tuple(distribution)}"
            )
        expected_query_lengths = np.arange(sequence_count + 1, dtype=np.int64)
        if not np.array_equal(cumulative_query_lengths, expected_query_lengths):
            raise ValueError(
                "Inkling fused RPA adapter requires one query token per sequence"
            )
        if np.any(kv_lengths <= 0):
            raise ValueError("Inkling fused RPA requires positive KV lengths")
        page_size = self.fused_cache_shape[1]
        maximum_sequence_capacity = self.fused_cache_shape[0] * page_size
        if np.any(kv_lengths > maximum_sequence_capacity):
            raise ValueError("fused RPA KV length exceeds total cache capacity")
        aligned_lengths = ((kv_lengths + page_size - 1) // page_size) * page_size
        expected_cumulative_kv = np.concatenate(
            (np.zeros((1,), dtype=np.int64), np.cumsum(aligned_lengths, dtype=np.int64))
        )
        if not np.array_equal(cumulative_kv_lengths, expected_cumulative_kv):
            raise ValueError("fused RPA cumulative KV lengths must be page-aligned")
        required_pages = int(expected_cumulative_kv[-1]) // page_size
        if required_pages > page_indices.size:
            raise ValueError("fused RPA page table cannot cover all KV lengths")
        referenced_pages = page_indices[:required_pages]
        total_pages = self.fused_cache_shape[0]
        if np.any(referenced_pages < 0) or np.any(referenced_pages >= total_pages):
            raise ValueError("fused RPA page index is outside the fused cache")
        sequence_page_sets: list[set[int]] = []
        for sequence in range(sequence_count):
            first_page = int(cumulative_kv_lengths[sequence]) // page_size
            last_page = int(cumulative_kv_lengths[sequence + 1]) // page_size
            sequence_pages = page_indices[first_page:last_page]
            if sequence_pages.size != np.unique(sequence_pages).size:
                raise ValueError(
                    "fused RPA page table aliases logical pages within one sequence"
                )
            sequence_page_sets.append({int(page) for page in sequence_pages})
        write_locations: list[tuple[int, int]] = []
        for sequence, length in enumerate(kv_lengths):
            position = int(length) - 1
            page_offset = int(cumulative_kv_lengths[sequence]) // page_size
            page = int(page_indices[page_offset + position // page_size])
            write_locations.append((page, position % page_size))
        if len(write_locations) != len(set(write_locations)):
            raise ValueError("fused RPA decode updates collide in the physical cache")
        for writer, (write_page, _write_offset) in enumerate(write_locations):
            for reader, used_pages in enumerate(sequence_page_sets):
                if writer != reader and write_page in used_pages:
                    raise ValueError(
                        "fused RPA active write page is shared across sequences"
                    )

    def invoke(
        self,
        kernel: Callable[..., tuple[Any, Any]],
        *inputs: Any,
        backend_manifest: tuple[tuple[str, str], ...],
        device_kind: str,
    ) -> tuple[Any, Any]:
        self._validate_signature(inputs)
        if backend_manifest != self.backend_manifest:
            raise ValueError("fused RPA backend source manifest does not match the plan")
        normalized_device_kind = device_kind.strip().lower()
        if re.fullmatch(r"tpu(?: v)?7x(?: lite)?", normalized_device_kind) is None:
            raise ValueError(f"fused RPA plan requires TPU7x, got {device_kind!r}")
        results = kernel(
            *inputs[:9],
            None,
            relative_states=inputs[9],
            relative_projection=inputs[10],
            causal=self.causal,
            sm_scale=self.softmax_scale,
            softmax_dtype=jnp.float32,
            sliding_window=self.sliding_window or None,
            d_block_sizes=self.decode_block_sizes,
            vmem_limit_bytes=self.vmem_limit_bytes,
        )
        if not isinstance(results, tuple) or len(results) != 2:
            raise ValueError("fused RPA kernel must return output and updated cache")
        for index, (value, shape, dtype) in enumerate(
            zip(results, (self.output_shape, self.fused_cache_shape), self.output_dtypes, strict=True)
        ):
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"fused RPA result {index} has shape/dtype "
                    f"{tuple(value.shape)}/{value.dtype}, expected {shape}/{dtype}"
                )
        return results

    def run_preflighted(
        self,
        kernel: Callable[..., tuple[Any, Any]],
        *inputs: Any,
        backend_manifest: tuple[tuple[str, str], ...],
        device_kind: str,
    ) -> tuple[Any, Any]:
        self.preflight(*inputs)
        return self.invoke(
            kernel,
            *inputs,
            backend_manifest=backend_manifest,
            device_kind=device_kind,
        )

    def render_executable_source(self) -> str:
        return f'''from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import jax
from {self.backend_module} import ragged_paged_attention
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes import get_simplified_key
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3 import get_tuned_block_sizes_v3
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from tpu_cake.rpa_lowering import FusedRpaPlan

RPA_EXECUTION_SCHEMA = {RPA_EXECUTION_SCHEMA!r}
NAME = {self.name!r}
SCHEDULE_SHA256 = {self.schedule_sha256!r}
QUERY_SHAPE = {self.query_shape!r}
KEY_SHAPE = {self.key_shape!r}
VALUE_SHAPE = {self.value_shape!r}
FUSED_CACHE_SHAPE = {self.fused_cache_shape!r}
KV_LENGTHS_SHAPE = {self.kv_lengths_shape!r}
PAGE_INDICES_SHAPE = {self.page_indices_shape!r}
CUMULATIVE_QUERY_LENGTHS_SHAPE = {self.cumulative_query_lengths_shape!r}
CUMULATIVE_KV_LENGTHS_SHAPE = {self.cumulative_kv_lengths_shape!r}
DISTRIBUTION_SHAPE = {self.distribution_shape!r}
RELATIVE_STATES_SHAPE = {self.relative_states_shape!r}
RELATIVE_PROJECTION_SHAPE = {self.relative_projection_shape!r}
OUTPUT_SHAPE = {self.output_shape!r}
INPUT_DTYPES = {self.input_dtypes!r}
OUTPUT_DTYPES = {self.output_dtypes!r}
DECODE_BLOCK_SIZES = {self.decode_block_sizes!r}
CAUSAL = {self.causal}
SOFTMAX_SCALE = {self.softmax_scale!r}
SOFTMAX_DTYPE = {self.softmax_dtype!r}
SLIDING_WINDOW = {self.sliding_window}
VMEM_LIMIT_BYTES = {self.vmem_limit_bytes}
BACKEND_REPOSITORY_REVISION = {self.backend_repository_revision!r}
BACKEND_FILE_REVISION = {self.backend_file_revision!r}
BACKEND_SHA256 = {self.backend_sha256!r}
BACKEND_MANIFEST = {self.backend_manifest!r}
EXECUTION_SCOPE = {self.execution_scope!r}

PLAN = FusedRpaPlan(
    name=NAME,
    schedule_sha256=SCHEDULE_SHA256,
    query_shape=QUERY_SHAPE,
    key_shape=KEY_SHAPE,
    value_shape=VALUE_SHAPE,
    fused_cache_shape=FUSED_CACHE_SHAPE,
    kv_lengths_shape=KV_LENGTHS_SHAPE,
    page_indices_shape=PAGE_INDICES_SHAPE,
    cumulative_query_lengths_shape=CUMULATIVE_QUERY_LENGTHS_SHAPE,
    cumulative_kv_lengths_shape=CUMULATIVE_KV_LENGTHS_SHAPE,
    distribution_shape=DISTRIBUTION_SHAPE,
    relative_states_shape=RELATIVE_STATES_SHAPE,
    relative_projection_shape=RELATIVE_PROJECTION_SHAPE,
    output_shape=OUTPUT_SHAPE,
    input_dtypes=INPUT_DTYPES,
    output_dtypes=OUTPUT_DTYPES,
    decode_block_sizes=DECODE_BLOCK_SIZES,
    causal=CAUSAL,
    softmax_scale=SOFTMAX_SCALE,
    softmax_dtype=SOFTMAX_DTYPE,
    sliding_window=SLIDING_WINDOW,
    vmem_limit_bytes=VMEM_LIMIT_BYTES,
    backend_repository_revision=BACKEND_REPOSITORY_REVISION,
    backend_file_revision=BACKEND_FILE_REVISION,
    backend_sha256=BACKEND_SHA256,
    backend_manifest=BACKEND_MANIFEST,
    execution_scope=EXECUTION_SCOPE,
)


def _source_sha256(value):
    source = inspect.getsourcefile(inspect.unwrap(value))
    if source is None:
        raise RuntimeError("cannot locate a fused RPA backend source")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _backend_manifest():
    return (
        ("ragged_paged_attention_v3.py", _source_sha256(ragged_paged_attention)),
        ("tuned_block_sizes.py", _source_sha256(get_simplified_key)),
        ("tuned_block_sizes_v3.py", _source_sha256(get_tuned_block_sizes_v3)),
        ("util.py", _source_sha256(get_dtype_packing)),
    )


def preflight(*inputs):
    PLAN.preflight(*inputs)


def run_preflighted(*inputs):
    return PLAN.run_preflighted(
        ragged_paged_attention,
        *inputs,
        backend_manifest=_backend_manifest(),
        device_kind=jax.devices()[0].device_kind,
    )
'''

    def source_sha256(self) -> str:
        return hashlib.sha256(self.render_executable_source().encode()).hexdigest()


def _shape(value) -> tuple[int, ...]:
    value_type = value.type
    if not isinstance(value_type, BufferType):
        raise UnsupportedLoweringError("fused RPA lowering expects physical buffers")
    return value_type.storage.get_shape()


def _dtype(value) -> str:
    value_type = value.type
    if not isinstance(value_type, BufferType):
        raise UnsupportedLoweringError("fused RPA lowering expects physical buffers")
    element_type = value_type.storage.element_type
    if isinstance(element_type, BFloat16Type):
        return "bfloat16"
    if isinstance(element_type, IntegerType) and element_type.width.data == 32:
        return "int32"
    raise UnsupportedLoweringError(f"unsupported fused RPA dtype {element_type}")


def _unsupported(message: str, operation: Operation) -> UnsupportedLoweringError:
    location = str(operation.location)
    if location == "loc(unknown)":
        return UnsupportedLoweringError(message)
    return UnsupportedLoweringError(
        f"{message}: relevant source site: {operation.name} at {location}"
    )


def _require_local_buffer_contract(value, operation: Operation) -> None:
    value_type = value.type
    if not isinstance(value_type, BufferType):
        raise UnsupportedLoweringError("fused RPA lowering expects physical buffers")
    rank = len(value_type.storage.get_shape())
    layout = tuple(index.data for index in value_type.layout.order)
    sharding = tuple(axis.data for axis in value_type.sharding.axes)
    if layout != tuple(range(rank)):
        raise _unsupported(
            "local-shard fused RPA adapter supports default physical layout only",
            operation,
        )
    if any(sharding):
        raise _unsupported(
            "local-shard fused RPA adapter leaves all outer sharding to its caller",
            operation,
        )


def lower_inkling_rpa_to_pallas(module: ModuleOp) -> FusedRpaPlan:
    verify_with_sources(module)
    kernels = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, KernelOp)
    )
    if len(kernels) != 1 or len(tuple(module.body.block.ops)) != 1:
        raise UnsupportedLoweringError("fused RPA lowering expects one physical kernel")
    kernel = kernels[0]
    if kernel.target.data != "tpu7x":
        raise UnsupportedLoweringError(
            f"fused RPA lowering does not support target {kernel.target.data!r}"
        )
    if tuple(kernel.mesh_axis_names) or tuple(kernel.mesh_axis_sizes):
        raise UnsupportedLoweringError(
            "local-shard fused RPA adapter does not own an outer device mesh"
        )
    operations = tuple(kernel.body.block.ops)
    if len(operations) != 2 or not isinstance(
        operations[0], FusedRaggedPagedAttentionOp
    ) or not isinstance(operations[1], YieldOp):
        raise UnsupportedLoweringError(
            "fused RPA lowering requires exactly one fused attention operation and yield"
        )
    attention = operations[0]
    for value in attention.operands:
        _require_local_buffer_contract(value, attention)
    arguments = tuple(kernel.body.block.args)
    if len(arguments) != 11 or tuple(attention.operands[:11]) != arguments:
        raise UnsupportedLoweringError(
            "fused RPA operation must bind all eleven inputs exactly"
        )
    if attention.output is not arguments[0] or attention.updated_cache is not arguments[3]:
        raise UnsupportedLoweringError(
            "fused RPA results must alias the donated query and fused-cache inputs"
        )
    if attention.vmem_limit_bytes.data > kernel.vmem_capacity_bytes.data:
        raise UnsupportedLoweringError(
            "fused RPA VMEM limit exceeds the physical kernel capacity"
        )
    return FusedRpaPlan(
        name="kernel",
        schedule_sha256=schedule_sha256(module),
        query_shape=_shape(attention.queries),
        key_shape=_shape(attention.keys),
        value_shape=_shape(attention.values),
        fused_cache_shape=_shape(attention.fused_cache),
        kv_lengths_shape=_shape(attention.kv_lengths),
        page_indices_shape=_shape(attention.page_indices),
        cumulative_query_lengths_shape=_shape(attention.cumulative_query_lengths),
        cumulative_kv_lengths_shape=_shape(attention.cumulative_kv_lengths),
        distribution_shape=_shape(attention.distribution),
        relative_states_shape=_shape(attention.relative_states),
        relative_projection_shape=_shape(attention.relative_projection),
        output_shape=_shape(attention.output),
        input_dtypes=tuple(_dtype(value) for value in attention.operands[:11]),
        output_dtypes=(_dtype(attention.output), _dtype(attention.updated_cache)),
        decode_block_sizes=(
            attention.query_block_size.data,
            attention.kv_block_size.data,
            attention.query_cluster_size.data,
            attention.kv_cluster_size.data,
        ),
        causal=attention.causal.data,
        softmax_scale=float(attention.softmax_scale.data),
        softmax_dtype=attention.softmax_dtype.data,
        sliding_window=attention.sliding_window.data,
        vmem_limit_bytes=attention.vmem_limit_bytes.data,
    )
