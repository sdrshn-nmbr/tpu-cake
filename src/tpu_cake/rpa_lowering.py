from __future__ import annotations

import hashlib
import inspect
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from xdsl.dialects.builtin import BFloat16Type, IntegerType, ModuleOp
from xdsl.ir import Operation

from tpu_cake.artifacts import file_sha256
from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    FusedRaggedPagedAttentionOp,
    KernelOp,
    RpaDecodeCoreOp,
    SemaphoreAllocOp,
    YieldOp,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import RenderedSourceIdentity
from tpu_cake.lowering import UnsupportedLoweringError
from tpu_cake.rpa_owned_kernel import (
    OwnedRpaCase,
    _owned_rpa_kernel,
    owned_rpa_has_bank_conflicts,
    semaphore_kwargs,
)
from tpu_cake.source import verify_with_sources

INKLE_REPOSITORY_REVISION = "c3d805516e0f3bc0c6f621f2590489aff12f8a59"
INKLING_RPA_FILE_REVISION = "e65c204fe42692254b0f7d6f678415dfc6fca401"
INKLING_RPA_SOURCE_SHA256 = "56d00d027cf921def1908e4815ced12e79210e1ac3cf57bcd727c5e6c6168eaa"
INKLING_RPA_UTIL_SHA256 = "fe76b20b6791ce3e6a30ff22a0e7de11b9a809113facbedb79b52079191f9562"
INKLING_RPA_TUNING_SHA256 = "563d74378b319710c8a7c6cd89ec174563462695417f9167d65716d74a4a2d50"
INKLING_RPA_BASE_TUNING_SHA256 = "4577624d0ef37726cbcceb5d570530399eb1f9113f8736e621a4e9aba9d3caef"
INKLING_RPA_MODULE = "sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3"
RPA_EXECUTION_SCHEMA = "sglang-jax-rpa-v3-adapter-v2"
SHARDED_RPA_EXECUTION_SCHEMA = "sglang-jax-rpa-v3-owned-shard-map-v2"
OWNED_RPA_DECODE_CORE_EXECUTION_SCHEMA = "tpu-cake-owned-rpa-decode-core-v1"
SHARDED_INKLE_REPOSITORY_REVISION = "9e1a7d39ccdcf9f396e024bfc45935f4f50f70c7"
SHARDED_INKLING_RPA_FILE_REVISION = "ac88a2ecfa905965b43edbbb5e6510eb272d09e5"
SHARDED_INKLING_RPA_SOURCE_SHA256 = (
    "12c6aeeade66538d3bb638f048850c3d69095ade4ec42559cd8b3566bfc68897"
)
INKLING_RPA_BACKEND_MANIFEST = (
    ("ragged_paged_attention_v3.py", INKLING_RPA_SOURCE_SHA256),
    ("tuned_block_sizes.py", INKLING_RPA_BASE_TUNING_SHA256),
    ("tuned_block_sizes_v3.py", INKLING_RPA_TUNING_SHA256),
    ("util.py", INKLING_RPA_UTIL_SHA256),
)
SHARDED_INKLING_RPA_BACKEND_MANIFEST = (
    ("ragged_paged_attention_v3.py", SHARDED_INKLING_RPA_SOURCE_SHA256),
    *INKLING_RPA_BACKEND_MANIFEST[1:],
)


def _preflight_decode_metadata(
    inputs: tuple[Any, ...],
    input_shapes: tuple[tuple[int, ...], ...],
) -> None:
    kv_lengths = np.asarray(inputs[4], dtype=np.int64)
    page_indices = np.asarray(inputs[5], dtype=np.int64)
    cumulative_query_lengths = np.asarray(inputs[6], dtype=np.int64)
    cumulative_kv_lengths = np.asarray(inputs[7], dtype=np.int64)
    distribution = np.asarray(inputs[8], dtype=np.int64)
    sequence_count = input_shapes[4][0]
    expected_distribution = np.full((3,), sequence_count, dtype=np.int64)
    if not np.array_equal(distribution, expected_distribution):
        raise ValueError(
            "Inkling fused RPA adapter accepts decode-only distribution "
            f"{tuple(expected_distribution)}, got {tuple(distribution)}"
        )
    expected_query_lengths = np.arange(sequence_count + 1, dtype=np.int64)
    if not np.array_equal(cumulative_query_lengths, expected_query_lengths):
        raise ValueError("Inkling fused RPA adapter requires one query token per sequence")
    if np.any(kv_lengths <= 0):
        raise ValueError("Inkling fused RPA requires positive KV lengths")
    page_size = input_shapes[3][1]
    maximum_sequence_capacity = input_shapes[3][0] * page_size
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
    total_pages = input_shapes[3][0]
    if np.any(referenced_pages < 0) or np.any(referenced_pages >= total_pages):
        raise ValueError("fused RPA page index is outside the fused cache")
    sequence_page_sets: list[set[int]] = []
    for sequence in range(sequence_count):
        first_page = int(cumulative_kv_lengths[sequence]) // page_size
        last_page = int(cumulative_kv_lengths[sequence + 1]) // page_size
        sequence_pages = page_indices[first_page:last_page]
        if sequence_pages.size != np.unique(sequence_pages).size:
            raise ValueError("fused RPA page table aliases logical pages within one sequence")
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
                raise ValueError("fused RPA active write page is shared across sequences")


@dataclass(frozen=True)
class FusedRpaPlan(RenderedSourceIdentity):
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
    backend_executor_qualname: str = "ragged_paged_attention"
    backend_repository_revision: str = INKLE_REPOSITORY_REVISION
    backend_file_revision: str = INKLING_RPA_FILE_REVISION
    backend_sha256: str = INKLING_RPA_SOURCE_SHA256
    backend_manifest: tuple[tuple[str, str], ...] = INKLING_RPA_BACKEND_MANIFEST
    execution_scope: str = "local-shard-caller-owned-sharding"

    def validate_backend_callable(self, kernel: Callable[..., tuple[Any, Any]]) -> tuple[str, str]:
        unwrapped = inspect.unwrap(kernel)
        module = getattr(kernel, "__module__", None)
        qualname = getattr(kernel, "__qualname__", None)
        source = inspect.getsourcefile(unwrapped)
        loaded_module = sys.modules.get(self.backend_module)
        resolved = (
            getattr(loaded_module, self.backend_executor_qualname, None)
            if loaded_module is not None
            else None
        )
        if (
            module != self.backend_module
            or qualname != self.backend_executor_qualname
            or source is None
            or kernel is not resolved
        ):
            raise ValueError("fused RPA executor does not match the pinned backend callable")
        source_path = Path(source)
        source_sha256 = file_sha256(source_path)
        live_manifest: list[tuple[str, str]] = []
        for name, _expected_sha256 in self.backend_manifest:
            dependency = source_path.parent / name
            if not dependency.is_file():
                raise ValueError(f"fused RPA backend dependency is missing: {name}")
            live_manifest.append((name, file_sha256(dependency)))
        if (
            source_sha256 != self.backend_sha256
            or (source_path.name, source_sha256) not in self.backend_manifest
            or tuple(live_manifest) != self.backend_manifest
        ):
            raise ValueError("fused RPA executor source does not match the pinned backend")
        return f"{module}.{qualname}", source_sha256

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
            raise ValueError(f"fused RPA needs {len(self.input_shapes)} inputs, got {len(inputs)}")
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
        _preflight_decode_metadata(inputs, self.input_shapes)

    def invoke(
        self,
        kernel: Callable[..., tuple[Any, Any]],
        *inputs: Any,
        backend_manifest: tuple[tuple[str, str], ...],
        device_kind: str,
    ) -> tuple[Any, Any]:
        self._validate_signature(inputs)
        self.validate_backend_callable(kernel)
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
            zip(
                results,
                (self.output_shape, self.fused_cache_shape),
                self.output_dtypes,
                strict=True,
            )
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
        return f"""from __future__ import annotations

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
BACKEND_EXECUTOR_QUALNAME = {self.backend_executor_qualname!r}
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
    backend_executor_qualname=BACKEND_EXECUTOR_QUALNAME,
    backend_manifest=BACKEND_MANIFEST,
    execution_scope=EXECUTION_SCOPE,
)


def _source_sha256(value):
    source = inspect.getsourcefile(inspect.unwrap(value))
    if source is None:
        raise RuntimeError("cannot locate a fused RPA backend source")
    return file_sha256(Path(source))


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
"""

@dataclass(frozen=True)
class ShardedFusedRpaPlan:
    local_plan: FusedRpaPlan
    mesh_axes: tuple[str, str]
    mesh_shape: tuple[int, int]
    input_partition_specs: tuple[tuple[str, ...], ...]
    output_partition_specs: tuple[tuple[str, ...], tuple[str, ...]]
    execution_scope: str = "owned-data-tensor-shard-map"

    def __post_init__(self) -> None:
        if self.mesh_axes != ("data", "tensor") or self.mesh_shape != (2, 4):
            raise ValueError("sharded Inkling RPA requires the exact 2x4 data/tensor mesh")
        if len(self.input_partition_specs) != len(self.local_plan.input_shapes):
            raise ValueError("sharded RPA needs one partition spec per input")
        for shape, spec in zip(
            self.local_plan.input_shapes,
            self.input_partition_specs,
            strict=True,
        ):
            if len(shape) != len(spec):
                raise ValueError("sharded RPA partition specs must match input ranks")
            if any(axis and axis not in self.mesh_axes for axis in spec):
                raise ValueError("sharded RPA partition specs use an unknown mesh axis")
        expected_outputs = (
            self.input_partition_specs[0],
            self.input_partition_specs[3],
        )
        if self.output_partition_specs != expected_outputs:
            raise ValueError("sharded RPA outputs must preserve query and cache sharding")

    @property
    def schedule_sha256(self) -> str:
        return self.local_plan.schedule_sha256

    @property
    def backend_manifest(self) -> tuple[tuple[str, str], ...]:
        return self.local_plan.backend_manifest

    @property
    def global_input_shapes(self) -> tuple[tuple[int, ...], ...]:
        axis_sizes = dict(zip(self.mesh_axes, self.mesh_shape, strict=True))
        return tuple(
            tuple(size * axis_sizes.get(axis, 1) for size, axis in zip(shape, spec, strict=True))
            for shape, spec in zip(
                self.local_plan.input_shapes,
                self.input_partition_specs,
                strict=True,
            )
        )

    @property
    def global_output_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.global_input_shapes[0], self.global_input_shapes[3]

    @property
    def external_donate_argnums(self) -> tuple[int, int]:
        return 0, 3

    def _validate_global_signature(self, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != len(self.global_input_shapes):
            raise ValueError(
                f"sharded fused RPA needs {len(self.global_input_shapes)} inputs, got {len(inputs)}"
            )
        for index, (value, shape, dtype) in enumerate(
            zip(
                inputs,
                self.global_input_shapes,
                self.local_plan.input_dtypes,
                strict=True,
            )
        ):
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"sharded fused RPA input {index} has shape/dtype "
                    f"{tuple(value.shape)}/{value.dtype}, expected {shape}/{dtype}"
                )

    def _local_host_shard(
        self,
        value: Any,
        spec: tuple[str, ...],
        coordinates: dict[str, int],
    ) -> np.ndarray:
        slices = []
        for dimension, axis in enumerate(spec):
            if not axis:
                slices.append(slice(None))
                continue
            axis_size = self.mesh_shape[self.mesh_axes.index(axis)]
            local_size = value.shape[dimension] // axis_size
            start = coordinates[axis] * local_size
            slices.append(slice(start, start + local_size))
        return np.asarray(value[tuple(slices)])

    def preflight(self, *inputs: Any) -> None:
        self._validate_global_signature(inputs)
        for data_index in range(self.mesh_shape[0]):
            for tensor_index in range(self.mesh_shape[1]):
                coordinates = {"data": data_index, "tensor": tensor_index}
                local_inputs = tuple(
                    self._local_host_shard(value, spec, coordinates)
                    for value, spec in zip(
                        inputs,
                        self.input_partition_specs,
                        strict=True,
                    )
                )
                self.local_plan.preflight(*local_inputs)

    def mesh(self, devices: tuple[Any, ...]) -> Mesh:
        expected_devices = self.mesh_shape[0] * self.mesh_shape[1]
        if len(devices) != expected_devices:
            raise ValueError(
                f"sharded Inkling RPA needs {expected_devices} devices, got {len(devices)}"
            )
        if any(
            re.fullmatch(r"tpu(?: v)?7x(?: lite)?", device.device_kind.strip().lower()) is None
            for device in devices
        ):
            raise ValueError("sharded Inkling RPA requires only TPU7x devices")
        return jax.make_mesh(self.mesh_shape, self.mesh_axes, devices=devices)

    def place_inputs(
        self,
        inputs: tuple[Any, ...],
        *,
        mesh: Mesh,
    ) -> tuple[jax.Array, ...]:
        self.preflight(*inputs)
        return tuple(
            jax.device_put(
                np.array(value, copy=True),
                NamedSharding(mesh, P(*spec)),
            )
            for value, spec in zip(inputs, self.input_partition_specs, strict=True)
        )

    def build_executable(
        self,
        kernel: Callable[..., tuple[Any, Any]],
        *,
        backend_manifest: tuple[tuple[str, str], ...],
        devices: tuple[Any, ...],
    ) -> tuple[Mesh, Callable[..., tuple[Any, Any]]]:
        mesh = self.mesh(devices)
        self.local_plan.validate_backend_callable(kernel)
        if backend_manifest != self.backend_manifest:
            raise ValueError("sharded RPA backend source manifest does not match the plan")

        def local(*inputs: Any) -> tuple[Any, Any]:
            return self.local_plan.invoke(
                kernel,
                *inputs,
                backend_manifest=backend_manifest,
                device_kind="TPU7x",
            )

        sharded = jax.shard_map(
            local,
            mesh=mesh,
            in_specs=tuple(P(*spec) for spec in self.input_partition_specs),
            out_specs=tuple(P(*spec) for spec in self.output_partition_specs),
        )
        return mesh, jax.jit(sharded, donate_argnums=self.external_donate_argnums)

    def source_sha256(self) -> str:
        payload = (
            SHARDED_RPA_EXECUTION_SCHEMA,
            self.local_plan.source_sha256(),
            self.mesh_axes,
            self.mesh_shape,
            self.input_partition_specs,
            self.output_partition_specs,
            self.external_donate_argnums,
            self.execution_scope,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()


@dataclass(frozen=True)
class OwnedRpaDecodeCorePlan:
    schedule_sha256: str
    core_input_shapes: tuple[tuple[int, ...], ...]
    core_output_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    external_input_shapes: tuple[tuple[int, ...], ...]
    external_input_dtypes: tuple[str, ...]
    external_output_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    block_sizes: tuple[int, int, int, int]
    relative_extent: int
    softmax_scale: float
    vmem_limit_bytes: int
    backend_repository_revision: str
    backend_file_revision: str
    backend_sha256: str
    implementation_sha256: str
    lowering_sha256: str
    execution_schema: str = OWNED_RPA_DECODE_CORE_EXECUTION_SCHEMA

    @property
    def mesh_axes(self) -> tuple[str, str]:
        return "data", "tensor"

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return 2, 4

    @property
    def input_partition_specs(self) -> tuple[tuple[str, ...], ...]:
        return (
            ("data", "tensor", ""),
            ("data", "tensor", ""),
            ("data", "tensor", ""),
            ("data", "", "tensor", "", ""),
            ("data",),
            ("data",),
            ("data",),
            ("data",),
            ("data",),
            ("data", "tensor", ""),
            ("", ""),
        )

    @property
    def output_partition_specs(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.input_partition_specs[0], self.input_partition_specs[3]

    @property
    def core_input_output_aliases(self) -> tuple[tuple[int, int], ...]:
        return (9, 0), (11, 1)

    @property
    def external_donate_argnums(self) -> tuple[int, ...]:
        return 0, 3

    @property
    def global_input_shapes(self) -> tuple[tuple[int, ...], ...]:
        axis_sizes = dict(zip(self.mesh_axes, self.mesh_shape, strict=True))
        return tuple(
            tuple(size * axis_sizes.get(axis, 1) for size, axis in zip(shape, spec, strict=True))
            for shape, spec in zip(
                self.external_input_shapes,
                self.input_partition_specs,
                strict=True,
            )
        )

    def __post_init__(self) -> None:
        if len(self.core_input_shapes) != 10:
            raise ValueError("owned RPA decode core requires ten physical inputs")
        if self.core_output_shapes != (
            self.core_input_shapes[0],
            self.core_input_shapes[2],
        ):
            raise ValueError("owned RPA decode outputs must alias query and cache shapes")
        if len(self.external_input_shapes) != 11 or len(self.external_input_dtypes) != 11:
            raise ValueError("owned RPA executable requires eleven external inputs")
        if self.external_output_shapes != (
            self.external_input_shapes[0],
            self.external_input_shapes[3],
        ):
            raise ValueError("owned RPA external outputs must match query and cache")
        if self.block_sizes != (8, 128, 8, 128):
            raise ValueError("owned RPA decode core requires the production block sizes")
        if self.relative_extent != 512 or self.softmax_scale != 1.0 / 128.0:
            raise ValueError("owned RPA decode core has the wrong relative-bias contract")
        for name, value in (
            ("implementation", self.implementation_sha256),
            ("lowering", self.lowering_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"owned RPA {name} source identity is not canonical")

    def _validate_signature(self, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != len(self.core_input_shapes):
            raise ValueError(
                f"owned RPA decode core needs {len(self.core_input_shapes)} inputs, "
                f"got {len(inputs)}"
            )
        for index, (value, shape, dtype) in enumerate(
            zip(
                inputs,
                self.core_input_shapes,
                ("bfloat16",) * 3 + ("int32",) * 5 + ("bfloat16",) * 2,
                strict=True,
            )
        ):
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"owned RPA input {index} has shape/dtype "
                    f"{tuple(value.shape)}/{value.dtype}, expected {shape}/{dtype}"
                )

    def _validate_global_signature(self, inputs: tuple[Any, ...]) -> None:
        shapes = self.global_input_shapes
        if len(inputs) != len(shapes):
            raise ValueError(f"owned RPA executable needs {len(shapes)} inputs, got {len(inputs)}")
        for index, (value, shape, dtype) in enumerate(
            zip(
                inputs,
                shapes,
                self.external_input_dtypes,
                strict=True,
            )
        ):
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"owned RPA external input {index} has shape/dtype "
                    f"{tuple(value.shape)}/{value.dtype}, expected {shape}/{dtype}"
                )

    def preflight(self, *inputs: Any) -> None:
        self._validate_global_signature(inputs)
        metadata = tuple(np.asarray(inputs[index]) for index in range(4, 9))
        for data_shard in range(self.mesh_shape[0]):
            local_inputs = list(inputs)
            for index, values in zip(range(4, 9), metadata, strict=True):
                local_inputs[index] = np.split(values, self.mesh_shape[0], axis=0)[data_shard]
            _preflight_decode_metadata(tuple(local_inputs), self.external_input_shapes)

    def _build_local(self, *, interpret: bool = False) -> Callable[..., tuple[Any, Any]]:
        if interpret:
            raise ValueError(
                "owned RPA decode uses Pallas DMA reshape/bitcast transforms that "
                "JAX 0.11 interpret mode does not support"
            )
        query_shape, _, cache_shape = self.core_input_shapes[:3]
        query_dtype = jnp.dtype("bfloat16")
        cache_dtype = jnp.dtype("bfloat16")
        query_fetch, kv_fetch, query_compute, kv_compute = self.block_sizes
        kv_heads, _, packed_query_heads, query_packing, head_dimension = query_shape
        packed_interleaved_heads = cache_shape[2]
        kv_packing = cache_shape[3]
        query_heads_per_kv_head = packed_query_heads * query_packing
        bkv_stride = packed_interleaved_heads + owned_rpa_has_bank_conflicts(
            packed_interleaved_heads
        )
        scalar_prefetch_count = 9
        input_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            None,
            pl.BlockSpec(memory_space=pltpu.HBM),
            None,
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        )
        output_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        )
        scratch_shapes = (
            None,
            pltpu.VMEM(
                (2, kv_fetch, bkv_stride, kv_packing, head_dimension),
                cache_dtype,
            ),
            pltpu.VMEM((2, kv_heads, query_fetch, *query_shape[2:]), query_dtype),
            pltpu.VMEM((2, kv_heads, query_fetch, *query_shape[2:]), query_dtype),
            pltpu.SemaphoreType.DMA((5, 2)),
            pltpu.VMEM(
                (kv_heads, query_fetch * query_heads_per_kv_head, 128),
                query_dtype,
            ),
            pltpu.VMEM(
                (kv_heads, query_fetch * query_heads_per_kv_head, 128),
                query_dtype,
            ),
            pltpu.VMEM(
                (kv_heads, query_fetch * query_heads_per_kv_head, head_dimension),
                query_dtype,
            ),
        )
        output_shapes = (
            pltpu.HBM(shape=query_shape, dtype=query_dtype),
            pltpu.HBM(shape=cache_shape, dtype=cache_dtype),
        )
        kernel = pl.pallas_call(
            partial(
                _owned_rpa_kernel,
                causal=True,
                sm_scale=self.softmax_scale,
                sliding_window=None,
                soft_cap=None,
                mask_value=-0.7 * float(jnp.finfo(jnp.float32).max),
                q_scale=None,
                k_scale=None,
                v_scale=None,
                xai_temperature_len=None,
                softmax_dtype=jnp.float32,
                relative_extent=self.relative_extent,
                static_q_len=None,
                bq_sz=query_fetch,
                bkv_sz=kv_fetch,
                bq_csz=query_compute,
                bkv_csz=kv_compute,
                case=OwnedRpaCase.DECODE,
                skip_kv_mask=False,
                tpu_version=7,
                debug_mode=False,
                mask_aligned_to_cu_kv=False,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=scalar_prefetch_count,
                in_specs=input_specs,
                out_specs=output_specs,
                grid=(1,),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=self.vmem_limit_bytes,
                disable_bounds_checks=True,
                **semaphore_kwargs(True),
            ),
            out_shape=output_shapes,
            input_output_aliases=dict(self.core_input_output_aliases),
            interpret=False,
            name="inkling_owned_rpa_decode_core",
            metadata={"schedule_sha256": self.schedule_sha256},
        )
        zero_mask = jnp.zeros((kv_fetch, head_dimension), dtype=jnp.int32)

        def run(*inputs: Any) -> tuple[Any, Any]:
            self._validate_signature(inputs)
            (
                queries,
                merged_kv,
                fused_cache,
                kv_lengths,
                page_indices,
                cumulative_query_lengths,
                cumulative_kv_lengths,
                distribution,
                relative_states,
                relative_projection,
            ) = inputs
            scalar_prefetches = (
                kv_lengths,
                page_indices,
                cumulative_query_lengths,
                cumulative_kv_lengths,
                jnp.zeros((1,), dtype=jnp.int32),
                distribution,
                jnp.zeros((3,), dtype=jnp.int32),
                jnp.full((4,), -1, dtype=jnp.int32),
                jnp.full((6,), -1, dtype=jnp.int32),
            )
            return kernel(
                *scalar_prefetches,
                pltpu.with_memory_space_constraint(queries, pltpu.HBM),
                pltpu.with_memory_space_constraint(merged_kv, pltpu.HBM),
                pltpu.with_memory_space_constraint(fused_cache, pltpu.HBM),
                None,
                zero_mask,
                None,
                relative_states,
                relative_projection,
            )

        return run

    def mesh(self, devices: tuple[Any, ...]) -> Mesh:
        expected_devices = self.mesh_shape[0] * self.mesh_shape[1]
        if len(devices) != expected_devices:
            raise ValueError(
                f"owned RPA decode core needs {expected_devices} devices, got {len(devices)}"
            )
        if any(
            re.fullmatch(r"tpu(?: v)?7x(?: lite)?", device.device_kind.strip().lower()) is None
            for device in devices
        ):
            raise ValueError("owned RPA decode core requires only TPU7x devices")
        return jax.make_mesh(self.mesh_shape, self.mesh_axes, devices=devices)

    def place_inputs(
        self,
        inputs: tuple[Any, ...],
        *,
        mesh: Mesh,
    ) -> tuple[jax.Array, ...]:
        self.preflight(*inputs)
        placed = []
        for index, (value, shape, dtype, spec) in enumerate(
            zip(
                inputs,
                self.global_input_shapes,
                self.external_input_dtypes,
                self.input_partition_specs,
                strict=True,
            )
        ):
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"owned RPA global input {index} has shape/dtype "
                    f"{tuple(value.shape)}/{value.dtype}, expected {shape}/{dtype}"
                )
            placed.append(
                jax.device_put(
                    np.array(value, copy=True),
                    NamedSharding(mesh, P(*spec)),
                )
            )
        return tuple(placed)

    def build(
        self,
        *,
        interpret: bool = False,
        devices: tuple[Any, ...] | None = None,
    ) -> tuple[Mesh, Callable[..., tuple[Any, Any]]]:
        selected_devices = tuple(devices or jax.devices())
        mesh = self.mesh(selected_devices)
        core_local = self._build_local(interpret=interpret)
        core_query_shape = self.core_input_shapes[0]
        core_cache_shape = self.core_input_shapes[2]

        def local(*inputs: Any) -> tuple[Any, Any]:
            queries, keys, values, fused_cache = inputs[:4]
            tokens, query_heads, head_dimension = queries.shape
            kv_heads = keys.shape[1]
            query_heads_per_kv_head = query_heads // kv_heads
            prepared_queries = (
                jnp.pad(
                    queries.reshape(
                        tokens,
                        kv_heads,
                        query_heads_per_kv_head,
                        head_dimension,
                    ),
                    ((0, 0), (0, 0), (0, 0), (0, 0)),
                    constant_values=0,
                )
                .reshape(tokens, kv_heads, *core_query_shape[2:])
                .swapaxes(0, 1)
            )
            merged_kv = jnp.concatenate((keys, values), axis=-1).reshape(self.core_input_shapes[1])
            prepared_cache = jnp.pad(
                fused_cache,
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
                constant_values=0,
            ).reshape(core_cache_shape)
            relative_states = jnp.pad(
                inputs[9].reshape(tokens, kv_heads, query_heads_per_kv_head, 16),
                ((0, 0), (0, 0), (0, 0), (0, 112)),
            ).swapaxes(0, 1)
            relative_projection = jnp.pad(
                inputs[10][:, ::-1],
                ((0, 112), (2048, 2048)),
            )
            output, updated_cache = core_local(
                prepared_queries,
                merged_kv,
                prepared_cache,
                *inputs[4:9],
                relative_states,
                relative_projection,
            )
            output = output.swapaxes(0, 1).reshape(self.external_output_shapes[0])
            return output, updated_cache.reshape(self.external_output_shapes[1])

        mapped = jax.shard_map(
            local,
            mesh=mesh,
            in_specs=tuple(P(*spec) for spec in self.input_partition_specs),
            out_specs=tuple(P(*spec) for spec in self.output_partition_specs),
            check_vma=False,
        )
        return mesh, jax.jit(mapped, donate_argnums=self.external_donate_argnums)

    def source_sha256(self) -> str:
        payload = (
            self.execution_schema,
            self.schedule_sha256,
            self.core_input_shapes,
            self.core_output_shapes,
            self.external_input_shapes,
            self.external_input_dtypes,
            self.external_output_shapes,
            self.block_sizes,
            self.relative_extent,
            self.softmax_scale,
            self.vmem_limit_bytes,
            self.backend_repository_revision,
            self.backend_file_revision,
            self.backend_sha256,
            self.implementation_sha256,
            self.lowering_sha256,
            self.core_input_output_aliases,
            self.external_donate_argnums,
            "tpu7x-hbm-output-v1",
            "decode-metadata-preflight-v1",
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()


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


def _fused_rpa_plan(
    module: ModuleOp,
    attention: FusedRaggedPagedAttentionOp,
    *,
    backend_repository_revision: str = INKLE_REPOSITORY_REVISION,
    backend_file_revision: str = INKLING_RPA_FILE_REVISION,
    backend_sha256: str = INKLING_RPA_SOURCE_SHA256,
    backend_manifest: tuple[tuple[str, str], ...] = INKLING_RPA_BACKEND_MANIFEST,
    execution_scope: str = "local-shard-caller-owned-sharding",
) -> FusedRpaPlan:
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
        backend_repository_revision=backend_repository_revision,
        backend_file_revision=backend_file_revision,
        backend_sha256=backend_sha256,
        backend_manifest=backend_manifest,
        execution_scope=execution_scope,
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
    if (
        len(operations) != 2
        or not isinstance(operations[0], FusedRaggedPagedAttentionOp)
        or not isinstance(operations[1], YieldOp)
    ):
        raise UnsupportedLoweringError(
            "fused RPA lowering requires exactly one fused attention operation and yield"
        )
    attention = operations[0]
    for value in attention.operands:
        _require_local_buffer_contract(value, attention)
    arguments = tuple(kernel.body.block.args)
    if len(arguments) != 11 or tuple(attention.operands[:11]) != arguments:
        raise UnsupportedLoweringError("fused RPA operation must bind all eleven inputs exactly")
    if attention.output is not arguments[0] or attention.updated_cache is not arguments[3]:
        raise UnsupportedLoweringError(
            "fused RPA results must alias the donated query and fused-cache inputs"
        )
    if attention.vmem_limit_bytes.data > kernel.vmem_capacity_bytes.data:
        raise UnsupportedLoweringError("fused RPA VMEM limit exceeds the physical kernel capacity")
    return _fused_rpa_plan(module, attention)


_SHARDED_INPUT_PARTITION_SPECS = (
    ("data", "tensor", ""),
    ("data", "tensor", ""),
    ("data", "tensor", ""),
    ("data", "", "tensor", "", ""),
    ("data",),
    ("data",),
    ("data",),
    ("data",),
    ("data",),
    ("data", "tensor", ""),
    ("", ""),
)


def lower_inkling_sharded_rpa_to_pallas(module: ModuleOp) -> ShardedFusedRpaPlan:
    verify_with_sources(module)
    kernels = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, KernelOp)
    )
    if len(kernels) != 1 or len(tuple(module.body.block.ops)) != 1:
        raise UnsupportedLoweringError("sharded fused RPA lowering expects one physical kernel")
    kernel = kernels[0]
    if kernel.target.data != "tpu7x":
        raise UnsupportedLoweringError(
            f"sharded fused RPA lowering does not support target {kernel.target.data!r}"
        )
    mesh_axes = tuple(value.data for value in kernel.mesh_axis_names)
    mesh_shape = tuple(value.data for value in kernel.mesh_axis_sizes)
    if mesh_axes != ("data", "tensor") or mesh_shape != (2, 4):
        raise UnsupportedLoweringError("sharded fused RPA requires the exact 2x4 data/tensor mesh")
    operations = tuple(kernel.body.block.ops)
    if (
        len(operations) != 2
        or not isinstance(operations[0], FusedRaggedPagedAttentionOp)
        or not isinstance(operations[1], YieldOp)
    ):
        raise UnsupportedLoweringError(
            "sharded fused RPA lowering requires exactly one fused attention operation and yield"
        )
    attention = operations[0]
    for value, expected_sharding in zip(
        attention.operands[:11],
        _SHARDED_INPUT_PARTITION_SPECS,
        strict=True,
    ):
        value_type = value.type
        if not isinstance(value_type, BufferType):
            raise UnsupportedLoweringError("sharded fused RPA lowering expects physical buffers")
        rank = len(value_type.storage.get_shape())
        layout = tuple(index.data for index in value_type.layout.order)
        sharding = tuple(axis.data for axis in value_type.sharding.axes)
        if layout != tuple(range(rank)):
            raise _unsupported(
                "sharded fused RPA supports default physical layout only",
                attention,
            )
        if sharding != expected_sharding:
            raise _unsupported(
                "sharded fused RPA input does not match the owned partition contract",
                attention,
            )
    if tuple(attention.operands[11:]) != (
        attention.operands[0],
        attention.operands[3],
    ):
        raise UnsupportedLoweringError(
            "sharded fused RPA results must alias query and fused-cache inputs"
        )
    arguments = tuple(kernel.body.block.args)
    if len(arguments) != 11 or tuple(attention.operands[:11]) != arguments:
        raise UnsupportedLoweringError(
            "sharded fused RPA operation must bind all eleven inputs exactly"
        )
    if attention.vmem_limit_bytes.data > kernel.vmem_capacity_bytes.data:
        raise UnsupportedLoweringError(
            "sharded fused RPA VMEM limit exceeds the physical kernel capacity"
        )
    local_plan = _fused_rpa_plan(
        module,
        attention,
        backend_repository_revision=SHARDED_INKLE_REPOSITORY_REVISION,
        backend_file_revision=SHARDED_INKLING_RPA_FILE_REVISION,
        backend_sha256=SHARDED_INKLING_RPA_SOURCE_SHA256,
        backend_manifest=SHARDED_INKLING_RPA_BACKEND_MANIFEST,
        execution_scope="local-device-body-of-owned-shard-map",
    )
    return ShardedFusedRpaPlan(
        local_plan=local_plan,
        mesh_axes=mesh_axes,
        mesh_shape=mesh_shape,
        input_partition_specs=_SHARDED_INPUT_PARTITION_SPECS,
        output_partition_specs=(
            _SHARDED_INPUT_PARTITION_SPECS[0],
            _SHARDED_INPUT_PARTITION_SPECS[3],
        ),
    )


def lower_inkling_owned_rpa_decode_core_to_pallas(
    module: ModuleOp,
) -> OwnedRpaDecodeCorePlan:
    verify_with_sources(module)
    kernels = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, KernelOp)
    )
    if len(kernels) != 1 or len(tuple(module.body.block.ops)) != 1:
        raise UnsupportedLoweringError("owned RPA lowering expects one physical kernel")
    kernel = kernels[0]
    if kernel.target.data != "tpu7x":
        raise UnsupportedLoweringError("owned RPA lowering requires TPU7x")
    operations = tuple(kernel.body.block.ops)
    cores = tuple(operation for operation in operations if isinstance(operation, RpaDecodeCoreOp))
    if len(cores) != 1 or any(
        not isinstance(operation, (AllocOp, SemaphoreAllocOp, RpaDecodeCoreOp, YieldOp))
        for operation in operations
    ):
        raise UnsupportedLoweringError(
            "owned RPA lowering requires allocations, one decode core, and yield"
        )
    core = cores[0]
    if tuple(core.operands[:10]) != tuple(kernel.body.block.args):
        raise UnsupportedLoweringError("owned RPA core must bind all ten inputs exactly")
    if (
        core.backend_repository_revision.data != SHARDED_INKLE_REPOSITORY_REVISION
        or core.backend_file_revision.data != SHARDED_INKLING_RPA_FILE_REVISION
        or core.backend_sha256.data != SHARDED_INKLING_RPA_SOURCE_SHA256
    ):
        raise UnsupportedLoweringError("owned RPA core backend identity is not pinned")
    implementation_path = Path(inspect.getsourcefile(_owned_rpa_kernel) or "")
    if not implementation_path.is_file():
        raise UnsupportedLoweringError("owned RPA implementation source is missing")
    lowering_path = Path(__file__)
    return OwnedRpaDecodeCorePlan(
        schedule_sha256=schedule_sha256(module),
        core_input_shapes=tuple(_shape(value) for value in core.operands[:10]),
        core_output_shapes=(_shape(core.output), _shape(core.updated_cache)),
        external_input_shapes=(
            (4, 8, 128),
            (4, 4, 128),
            (4, 4, 128),
            (3712, 1, 4, 2, 128),
            (4,),
            (8192,),
            (5,),
            (5,),
            (3,),
            (4, 8, 16),
            (16, 512),
        ),
        external_input_dtypes=("bfloat16",) * 4 + ("int32",) * 5 + ("bfloat16",) * 2,
        external_output_shapes=((4, 8, 128), (3712, 1, 4, 2, 128)),
        block_sizes=(
            core.query_fetch_size.data,
            core.kv_fetch_size.data,
            core.query_compute_size.data,
            core.kv_compute_size.data,
        ),
        relative_extent=core.relative_extent.data,
        softmax_scale=float(core.softmax_scale.data),
        vmem_limit_bytes=kernel.vmem_capacity_bytes.data,
        backend_repository_revision=core.backend_repository_revision.data,
        backend_file_revision=core.backend_file_revision.data,
        backend_sha256=core.backend_sha256.data,
        implementation_sha256=file_sha256(implementation_path),
        lowering_sha256=file_sha256(lowering_path),
    )
