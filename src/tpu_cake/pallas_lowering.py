from __future__ import annotations

import ast
import functools
import inspect
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec
from xdsl.dialects.builtin import BFloat16Type, Float32Type, ModuleOp

from tpu_cake.artifacts import file_sha256
from tpu_cake.canonical import parse_physical_module
from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    CollectiveImplementation,
    CollectiveImplementationResources,
    CollectiveKind,
    CollectiveOp,
    CollectiveReduceScatterOp,
    DmaStartOp,
    DmaWaitOp,
    KernelOp,
    MxuMatmulOp,
    SemaphoreAllocOp,
    YieldOp,
    pallas_bidirectional_ring_resources,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import RenderedSourceIdentity
from tpu_cake.lowering import UnsupportedLoweringError

LEGACY_PALLAS_EXECUTION_SCHEMA = "standalone-rendering-v1"
PALLAS_EXECUTION_SCHEMA = "delegated-plan-v2"
PALLAS_NATIVE_COLLECTIVE_EXECUTION_SCHEMA = "native-collective-plan-v3"


def _matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = lax.dot_general(
        lhs_ref[...],
        rhs_ref[...],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _ring_mod(value, modulus):
    return lax.rem(value + modulus, modulus)


def _ring_device_id(
    neighbor,
    *,
    axis_name: str,
    mesh_axis_names: tuple[str, ...],
):
    if mesh_axis_names.count(axis_name) != 1:
        raise ValueError("ring axis must appear exactly once in the mesh")
    return tuple(
        neighbor if name == axis_name else lax.axis_index(name) for name in mesh_axis_names
    )


def _ring_signal(
    direction,
    semaphore,
    *,
    axis_name: str,
    mesh_axis_names: tuple[str, ...],
    group_size: int,
) -> None:
    device_id = lax.axis_index(axis_name)
    neighbor = _ring_mod(device_id - 1, group_size)
    if direction == 1:
        neighbor = _ring_mod(device_id + 1, group_size)
    pl.semaphore_signal(
        semaphore,
        inc=1,
        device_id=_ring_device_id(
            neighbor,
            axis_name=axis_name,
            mesh_axis_names=mesh_axis_names,
        ),
        device_id_type=pl.DeviceIdType.MESH,
    )


def _ring_barrier(left_neighbor, right_neighbor) -> None:
    barrier = pltpu.get_barrier_semaphore()
    for neighbor in (left_neighbor, right_neighbor):
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id=neighbor,
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, 2)

    @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
    def _second_barrier(second):
        for neighbor in (left_neighbor, right_neighbor):
            pl.semaphore_signal(
                second,
                inc=1,
                device_id=neighbor,
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(second, 2)


def _bidirectional_reduce_scatter_kernel(
    input_ref,
    output_ref,
    hbm_scratch,
    local_copy_sem,
    left_recv_sem,
    left_send_sem,
    right_recv_sem,
    right_send_sem,
    left_capacity_sem,
    right_capacity_sem,
    accumulator,
    *,
    axis_name: str,
    mesh_axis_names: tuple[str, ...],
    group_size: int,
    block_shape: tuple[int, int],
) -> None:
    left = 0
    right = 1
    outer_step = pl.program_id(0)
    phase = pl.program_id(1)
    is_start = jnp.logical_and(outer_step == 0, phase == 0)
    is_last = outer_step == pl.num_programs(0) - 1
    working_slot = lax.rem(outer_step, 2)
    receiving_slot = 1 - working_slot
    device_id = lax.axis_index(axis_name)
    right_neighbor = _ring_mod(device_id + 1, group_size)
    left_neighbor = _ring_mod(device_id - 1, group_size)
    right_neighbor_id = _ring_device_id(
        right_neighbor,
        axis_name=axis_name,
        mesh_axis_names=mesh_axis_names,
    )
    left_neighbor_id = _ring_device_id(
        left_neighbor,
        axis_name=axis_name,
        mesh_axis_names=mesh_axis_names,
    )
    left_source = _ring_mod(device_id + outer_step + 1, group_size)
    right_source = _ring_mod(device_id - outer_step - 1, group_size)
    half_rows = block_shape[0] // 2
    left_slice = pl.ds(0, half_rows)
    right_slice = pl.ds(half_rows, half_rows)
    phase_slice = pl.ds(phase * half_rows, half_rows)
    signal = functools.partial(
        _ring_signal,
        axis_name=axis_name,
        mesh_axis_names=mesh_axis_names,
        group_size=group_size,
    )

    initial_left = pltpu.make_async_remote_copy(
        input_ref.at[device_id, left_slice],
        hbm_scratch.at[working_slot, left_slice],
        left_send_sem,
        left_recv_sem,
        device_id=left_neighbor_id,
        device_id_type=pl.DeviceIdType.MESH,
    )
    initial_right = pltpu.make_async_remote_copy(
        input_ref.at[device_id, right_slice],
        hbm_scratch.at[working_slot, right_slice],
        right_send_sem,
        right_recv_sem,
        device_id=right_neighbor_id,
        device_id_type=pl.DeviceIdType.MESH,
    )
    left_copy = pltpu.make_async_remote_copy(
        hbm_scratch.at[working_slot, left_slice],
        hbm_scratch.at[receiving_slot, left_slice],
        left_send_sem,
        left_recv_sem,
        device_id=left_neighbor_id,
        device_id_type=pl.DeviceIdType.MESH,
    )
    right_copy = pltpu.make_async_remote_copy(
        hbm_scratch.at[receiving_slot, right_slice],
        hbm_scratch.at[working_slot, right_slice],
        right_send_sem,
        right_recv_sem,
        device_id=right_neighbor_id,
        device_id_type=pl.DeviceIdType.MESH,
    )

    @pl.when(is_start)
    def _start():
        _ring_barrier(left_neighbor_id, right_neighbor_id)
        output_ref[...] = jnp.zeros_like(output_ref[...])
        accumulator[...] = jnp.zeros_like(accumulator[...])
        initial_left.start()
        initial_left.wait()
        initial_right.start()
        signal(left, right_capacity_sem)
        signal(right, left_capacity_sem)

    @pl.when(~is_start)
    def _send():
        @pl.when(phase == left)
        def _send_right():
            pl.semaphore_wait(right_capacity_sem, 1)
            right_copy.start()

        @pl.when(phase == right)
        def _send_left():
            pl.semaphore_wait(left_capacity_sem, 1)
            left_copy.start()

    local_copy = pltpu.make_async_copy(
        hbm_scratch.at[working_slot, phase_slice], accumulator, local_copy_sem
    )
    local_copy.start()
    local_copy.wait()

    @pl.when(~is_last)
    def _accumulate():
        @pl.when(phase == left)
        def _add_left():
            accumulator[...] += input_ref[left_source, left_slice]

        @pl.when(phase == right)
        def _add_right():
            accumulator[...] += input_ref[right_source, right_slice]

    local_store = pltpu.make_async_copy(
        accumulator, hbm_scratch.at[working_slot, phase_slice], local_copy_sem
    )
    local_store.start()
    local_store.wait()

    @pl.when(is_start)
    def _finish_initial_right():
        initial_right.wait()

    @pl.when(~is_start)
    def _finish_send():
        @pl.when(phase == left)
        def _finish_right():
            right_copy.wait()
            signal(left, right_capacity_sem)

        @pl.when(phase == right)
        def _finish_left():
            left_copy.wait()
            signal(right, left_capacity_sem)

    @pl.when(is_last)
    def _store_result():
        @pl.when(phase == left)
        def _store_left():
            output_ref[left_slice, ...] = accumulator[...]
            pl.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == right)
        def _store_right():
            output_ref[right_slice, ...] = accumulator[...]
            pl.semaphore_wait(left_capacity_sem, 1)


def _pallas_bidirectional_reduce_scatter(
    partial,
    *,
    axis_name: str,
    mesh_axis_names: tuple[str, ...],
    group_size: int,
    scatter_dimension: int,
    output_shape: tuple[int, int],
    name: str,
    interpret,
):
    reshaped = partial.reshape(
        *partial.shape[:scatter_dimension],
        group_size,
        output_shape[scatter_dimension],
        *partial.shape[scatter_dimension + 1 :],
    )
    chunks = jnp.moveaxis(reshaped, scatter_dimension, 0)
    output_spec = (
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
        jax.ShapeDtypeStruct((2, *output_shape), jnp.float32),
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        grid=(group_size, 2),
        scratch_shapes=(
            [pltpu.SemaphoreType.DMA] * 5
            + [pltpu.SemaphoreType.REGULAR] * 2
            + [pltpu.VMEM((output_shape[0] // 2, output_shape[1]), jnp.float32)]
        ),
    )
    kernel = functools.partial(
        _bidirectional_reduce_scatter_kernel,
        axis_name=axis_name,
        mesh_axis_names=mesh_axis_names,
        group_size=group_size,
        block_shape=output_shape,
    )
    return pl.pallas_call(
        kernel,
        out_shape=output_spec,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=0),
        name=name,
        interpret=interpret,
    )(chunks)[0]


def _partition_spec(sharding: tuple[str, ...]) -> PartitionSpec:
    return PartitionSpec(*(axis or None for axis in sharding))


@dataclass(frozen=True)
class PallasMatmulPlan(RenderedSourceIdentity):
    name: str
    schedule_sha256: str
    mesh_axis: str
    mesh_size: int
    lhs_local_shape: tuple[int, int]
    rhs_local_shape: tuple[int, int]
    partial_local_shape: tuple[int, int]
    output_local_shape: tuple[int, int]
    lhs_sharding: tuple[str, str]
    rhs_sharding: tuple[str, str]
    output_sharding: tuple[str, str]
    scatter_dimension: int
    tile_m: int
    tile_k: int
    tile_n: int
    collective_link_bandwidths: tuple[tuple[str, int], ...] = ()
    collective_implementation: CollectiveImplementation | None = None

    @property
    def global_lhs_shape(self) -> tuple[int, int]:
        return self._global_shape(self.lhs_local_shape, self.lhs_sharding)

    @property
    def global_rhs_shape(self) -> tuple[int, int]:
        return self._global_shape(self.rhs_local_shape, self.rhs_sharding)

    @property
    def global_output_shape(self) -> tuple[int, int]:
        return self._global_shape(self.output_local_shape, self.output_sharding)

    @property
    def collective_hbm_scratch_bytes(self) -> int:
        return self._collective_resources.hbm_scratch_bytes

    @property
    def collective_accumulator_vmem_bytes(self) -> int:
        return self._collective_resources.vmem_scratch_bytes

    @property
    def collective_dma_semaphore_count(self) -> int:
        return self._collective_resources.dma_semaphore_count

    @property
    def collective_capacity_semaphore_count(self) -> int:
        return self._collective_resources.capacity_semaphore_count

    @property
    def collective_startup_semaphore_count(self) -> int:
        return self._collective_resources.startup_semaphore_count

    @property
    def collective_startup_barrier_phases(self) -> int:
        return self._collective_resources.startup_barrier_phases

    @property
    def collective_remote_half_output_copy_count(self) -> int:
        return self._collective_resources.remote_half_output_copy_count

    @property
    def collective_remote_payload_bytes(self) -> int:
        return self._collective_resources.remote_payload_bytes

    @property
    def collective_remote_bidirectional_endpoint_bytes(self) -> int:
        return self._collective_resources.remote_bidirectional_endpoint_bytes

    @property
    def _collective_resources(self) -> CollectiveImplementationResources:
        if self.collective_implementation is not CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING:
            return CollectiveImplementationResources(0, 0, 0, 0, 0, 0, 0, 0, 0)
        rows, columns = self.output_local_shape
        return pallas_bidirectional_ring_resources(
            rows * columns * 4,
            self.mesh_size,
        )

    def _global_shape(self, shape: tuple[int, int], sharding: tuple[str, str]) -> tuple[int, int]:
        return tuple(
            size * (self.mesh_size if axis == self.mesh_axis else 1)
            for size, axis in zip(shape, sharding, strict=True)
        )

    def build(self, *, interpret: bool = False, devices=None):
        selected_devices = tuple(devices or jax.devices())
        if len(selected_devices) != self.mesh_size:
            raise ValueError(
                f"Pallas plan needs {self.mesh_size} devices, found {len(selected_devices)}"
            )
        mesh = Mesh(selected_devices, (self.mesh_axis,))
        interpret_setting = (
            pltpu.InterpretParams(detect_races=True, out_of_bounds_reads="raise")
            if interpret
            else False
        )
        local_call = pl.pallas_call(
            _matmul_kernel,
            out_shape=jax.ShapeDtypeStruct(self.partial_local_shape, jnp.float32),
            in_specs=(
                pl.BlockSpec((self.tile_m, self.tile_k), lambda i, _j: (i, 0)),
                pl.BlockSpec((self.tile_k, self.tile_n), lambda _i, j: (0, j)),
            ),
            out_specs=pl.BlockSpec((self.tile_m, self.tile_n), lambda i, j: (i, j)),
            grid=(
                self.partial_local_shape[0] // self.tile_m,
                self.partial_local_shape[1] // self.tile_n,
            ),
            interpret=interpret_setting,
            name=self.name,
            metadata={"schedule_sha256": self.schedule_sha256},
        )

        def distributed(lhs, rhs):
            partial = local_call(lhs, rhs)
            if self.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING:
                return _pallas_bidirectional_reduce_scatter(
                    partial,
                    axis_name=self.mesh_axis,
                    mesh_axis_names=(self.mesh_axis,),
                    group_size=self.mesh_size,
                    scatter_dimension=self.scatter_dimension,
                    output_shape=self.output_local_shape,
                    name=f"{self.name}_pallas_reduce_scatter",
                    interpret=interpret_setting,
                )
            return lax.psum_scatter(
                partial,
                self.mesh_axis,
                scatter_dimension=self.scatter_dimension,
                tiled=True,
            )

        mapped = jax.shard_map(
            distributed,
            mesh=mesh,
            in_specs=(
                _partition_spec(self.lhs_sharding),
                _partition_spec(self.rhs_sharding),
            ),
            out_specs=_partition_spec(self.output_sharding),
            check_vma=False,
        )
        return jax.jit(mapped), mesh

    def render_source(self) -> str:
        kernel_source = inspect.getsource(_matmul_kernel).rstrip()
        return f"""from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec

{kernel_source}

NAME = {self.name!r}
SCHEDULE_SHA256 = {self.schedule_sha256!r}
MESH_AXIS = {self.mesh_axis!r}
MESH_SIZE = {self.mesh_size}
LHS_LOCAL_SHAPE = {self.lhs_local_shape!r}
RHS_LOCAL_SHAPE = {self.rhs_local_shape!r}
PARTIAL_LOCAL_SHAPE = {self.partial_local_shape!r}
OUTPUT_SHARDING = {self.output_sharding!r}
LHS_SHARDING = {self.lhs_sharding!r}
RHS_SHARDING = {self.rhs_sharding!r}
SCATTER_DIMENSION = {self.scatter_dimension}
TILE_M = {self.tile_m}
TILE_K = {self.tile_k}
TILE_N = {self.tile_n}


def _partition_spec(sharding):
    return PartitionSpec(*(axis or None for axis in sharding))


def build(*, interpret=False, devices=None):
    selected_devices = tuple(devices or jax.devices())
    if len(selected_devices) != MESH_SIZE:
        raise ValueError(f"expected {{MESH_SIZE}} devices, found {{len(selected_devices)}}")
    mesh = Mesh(selected_devices, (MESH_AXIS,))
    interpret_setting = (
        pltpu.InterpretParams(detect_races=True, out_of_bounds_reads="raise")
        if interpret
        else False
    )
    local_call = pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(PARTIAL_LOCAL_SHAPE, jnp.float32),
        in_specs=(
            pl.BlockSpec((TILE_M, TILE_K), lambda i, _j: (i, 0)),
            pl.BlockSpec((TILE_K, TILE_N), lambda _i, j: (0, j)),
        ),
        out_specs=pl.BlockSpec((TILE_M, TILE_N), lambda i, j: (i, j)),
        grid=(PARTIAL_LOCAL_SHAPE[0] // TILE_M, PARTIAL_LOCAL_SHAPE[1] // TILE_N),
        interpret=interpret_setting,
        name=NAME,
        metadata={{"schedule_sha256": SCHEDULE_SHA256}},
    )

    def distributed(lhs, rhs):
        partial = local_call(lhs, rhs)
        return lax.psum_scatter(
            partial,
            MESH_AXIS,
            scatter_dimension=SCATTER_DIMENSION,
            tiled=True,
        )

    mapped = jax.shard_map(
        distributed,
        mesh=mesh,
        in_specs=(_partition_spec(LHS_SHARDING), _partition_spec(RHS_SHARDING)),
        out_specs=_partition_spec(OUTPUT_SHARDING),
        check_vma=False,
    )
    return jax.jit(mapped), mesh
"""

    def render_executable_source(self) -> str:
        if self.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING:
            return self._render_native_collective_source()
        topology_constants = ""
        topology_arguments = ""
        if self.collective_link_bandwidths:
            topology_constants = (
                f"\nCOLLECTIVE_LINK_BANDWIDTHS = {self.collective_link_bandwidths!r}"
            )
            topology_arguments = "    collective_link_bandwidths=COLLECTIVE_LINK_BANDWIDTHS,\n"
        return f"""from __future__ import annotations

from tpu_cake.pallas_lowering import PallasMatmulPlan

PALLAS_EXECUTION_SCHEMA = {PALLAS_EXECUTION_SCHEMA!r}
NAME = {self.name!r}
SCHEDULE_SHA256 = {self.schedule_sha256!r}
MESH_AXIS = {self.mesh_axis!r}
MESH_SIZE = {self.mesh_size}
LHS_LOCAL_SHAPE = {self.lhs_local_shape!r}
RHS_LOCAL_SHAPE = {self.rhs_local_shape!r}
PARTIAL_LOCAL_SHAPE = {self.partial_local_shape!r}
OUTPUT_LOCAL_SHAPE = {self.output_local_shape!r}
OUTPUT_SHARDING = {self.output_sharding!r}
LHS_SHARDING = {self.lhs_sharding!r}
RHS_SHARDING = {self.rhs_sharding!r}
SCATTER_DIMENSION = {self.scatter_dimension}
TILE_M = {self.tile_m}
TILE_K = {self.tile_k}
TILE_N = {self.tile_n}{topology_constants}

PLAN = PallasMatmulPlan(
    name=NAME,
    schedule_sha256=SCHEDULE_SHA256,
    mesh_axis=MESH_AXIS,
    mesh_size=MESH_SIZE,
    lhs_local_shape=LHS_LOCAL_SHAPE,
    rhs_local_shape=RHS_LOCAL_SHAPE,
    partial_local_shape=PARTIAL_LOCAL_SHAPE,
    output_local_shape=OUTPUT_LOCAL_SHAPE,
    lhs_sharding=LHS_SHARDING,
    rhs_sharding=RHS_SHARDING,
    output_sharding=OUTPUT_SHARDING,
    scatter_dimension=SCATTER_DIMENSION,
    tile_m=TILE_M,
    tile_k=TILE_K,
    tile_n=TILE_N,
{topology_arguments})


def build(*, interpret=False, devices=None):
    return PLAN.build(interpret=interpret, devices=devices)
"""

    def _render_native_collective_source(self) -> str:
        topology_constants = ""
        topology_arguments = ""
        if self.collective_link_bandwidths:
            topology_constants = (
                f"\nCOLLECTIVE_LINK_BANDWIDTHS = {self.collective_link_bandwidths!r}"
            )
            topology_arguments = "    collective_link_bandwidths=COLLECTIVE_LINK_BANDWIDTHS,\n"
        return f"""from __future__ import annotations

from tpu_cake.dialects.tpu_schedule import CollectiveImplementation
from tpu_cake.pallas_lowering import PallasMatmulPlan

PALLAS_EXECUTION_SCHEMA = {PALLAS_NATIVE_COLLECTIVE_EXECUTION_SCHEMA!r}
NAME = {self.name!r}
SCHEDULE_SHA256 = {self.schedule_sha256!r}
MESH_AXIS = {self.mesh_axis!r}
MESH_SIZE = {self.mesh_size}
LHS_LOCAL_SHAPE = {self.lhs_local_shape!r}
RHS_LOCAL_SHAPE = {self.rhs_local_shape!r}
PARTIAL_LOCAL_SHAPE = {self.partial_local_shape!r}
OUTPUT_LOCAL_SHAPE = {self.output_local_shape!r}
OUTPUT_SHARDING = {self.output_sharding!r}
LHS_SHARDING = {self.lhs_sharding!r}
RHS_SHARDING = {self.rhs_sharding!r}
SCATTER_DIMENSION = {self.scatter_dimension}
TILE_M = {self.tile_m}
TILE_K = {self.tile_k}
TILE_N = {self.tile_n}
COLLECTIVE_IMPLEMENTATION = {self.collective_implementation.value!r}{topology_constants}
COLLECTIVE_HBM_SCRATCH_BYTES = {self.collective_hbm_scratch_bytes}
COLLECTIVE_ACCUMULATOR_VMEM_BYTES = {self.collective_accumulator_vmem_bytes}
COLLECTIVE_DMA_SEMAPHORE_COUNT = {self.collective_dma_semaphore_count}
COLLECTIVE_CAPACITY_SEMAPHORE_COUNT = {self.collective_capacity_semaphore_count}
COLLECTIVE_STARTUP_SEMAPHORE_COUNT = {self.collective_startup_semaphore_count}
COLLECTIVE_STARTUP_BARRIER_PHASES = {self.collective_startup_barrier_phases}
COLLECTIVE_REMOTE_HALF_OUTPUT_COPY_COUNT = {self.collective_remote_half_output_copy_count}
COLLECTIVE_REMOTE_PAYLOAD_BYTES = {self.collective_remote_payload_bytes}
COLLECTIVE_REMOTE_BIDIRECTIONAL_ENDPOINT_BYTES = {self.collective_remote_bidirectional_endpoint_bytes}

PLAN = PallasMatmulPlan(
    name=NAME,
    schedule_sha256=SCHEDULE_SHA256,
    mesh_axis=MESH_AXIS,
    mesh_size=MESH_SIZE,
    lhs_local_shape=LHS_LOCAL_SHAPE,
    rhs_local_shape=RHS_LOCAL_SHAPE,
    partial_local_shape=PARTIAL_LOCAL_SHAPE,
    output_local_shape=OUTPUT_LOCAL_SHAPE,
    lhs_sharding=LHS_SHARDING,
    rhs_sharding=RHS_SHARDING,
    output_sharding=OUTPUT_SHARDING,
    scatter_dimension=SCATTER_DIMENSION,
    tile_m=TILE_M,
    tile_k=TILE_K,
    tile_n=TILE_N,
{topology_arguments}    collective_implementation=CollectiveImplementation(
        COLLECTIVE_IMPLEMENTATION
    ),
)


def build(*, interpret=False, devices=None):
    return PLAN.build(interpret=interpret, devices=devices)
"""


def _shape(buffer: BufferType) -> tuple[int, int]:
    shape = buffer.storage.get_shape()
    if len(shape) != 2:
        raise UnsupportedLoweringError("Pallas matmul buffers must be rank 2")
    return shape


def _sharding(buffer: BufferType) -> tuple[str, str]:
    sharding = tuple(value.data for value in buffer.sharding.axes)
    if len(sharding) != 2:
        raise UnsupportedLoweringError("Pallas matmul sharding must have rank 2")
    return sharding


def lower_physical_matmul_to_pallas(module: ModuleOp) -> PallasMatmulPlan:
    module.verify()
    kernels = [operation for operation in module.body.block.ops if isinstance(operation, KernelOp)]
    if len(kernels) != 1 or len(list(module.body.block.ops)) != 1:
        raise UnsupportedLoweringError("Pallas lowering expects exactly one physical kernel")
    kernel = kernels[0]
    if kernel.target.data != "tpu7x":
        raise UnsupportedLoweringError(
            f"Pallas matmul lowering does not support target {kernel.target.data!r}"
        )
    operations = list(kernel.body.block.ops)
    expected_types = (
        AllocOp,
        AllocOp,
        AllocOp,
        AllocOp,
        SemaphoreAllocOp,
        SemaphoreAllocOp,
        SemaphoreAllocOp,
        DmaStartOp,
        DmaStartOp,
        DmaWaitOp,
        DmaWaitOp,
        MxuMatmulOp,
        (CollectiveReduceScatterOp, CollectiveOp),
        DmaStartOp,
        DmaWaitOp,
        YieldOp,
    )
    if len(operations) != len(expected_types) or any(
        not isinstance(operation, expected)
        for operation, expected in zip(operations, expected_types, strict=True)
    ):
        mismatch = next(
            (
                operation
                for operation, expected in zip(operations, expected_types)
                if not isinstance(operation, expected)
            ),
            operations[len(expected_types)] if len(operations) > len(expected_types) else kernel,
        )
        raise UnsupportedLoweringError(
            "Pallas matmul lowering requires the exact supported load, matmul, "
            f"reduce-scatter, and store schedule; found {mismatch.name} "
            f"at {mismatch.location}"
        )
    matmuls = [operation for operation in operations if isinstance(operation, MxuMatmulOp)]
    collectives = [
        operation
        for operation in operations
        if isinstance(operation, CollectiveReduceScatterOp)
        or (
            isinstance(operation, CollectiveOp)
            and operation.kind.data is CollectiveKind.REDUCE_SCATTER
        )
    ]
    if len(matmuls) != 1 or len(collectives) != 1:
        raise UnsupportedLoweringError(
            "Pallas lowering supports one MXU matmul followed by one reduce-scatter"
        )
    matmul, collective = matmuls[0], collectives[0]
    if collective.source != matmul.accumulator:
        raise UnsupportedLoweringError("Pallas reduce-scatter must consume the MXU accumulator")
    if collective.reducer.data != "sum":
        raise UnsupportedLoweringError("Pallas matmul lowering supports sum reduction only")
    input_dmas = operations[7:9]
    input_waits = operations[9:11]
    output_dma = operations[13]
    output_wait = operations[14]
    assert all(isinstance(operation, DmaStartOp) for operation in input_dmas)
    assert all(isinstance(operation, DmaWaitOp) for operation in input_waits)
    assert isinstance(output_dma, DmaStartOp)
    assert isinstance(output_wait, DmaWaitOp)
    if len(kernel.body.block.args) != 3:
        raise UnsupportedLoweringError("Pallas matmul lowering expects two inputs and one output")
    if tuple(operation.source for operation in input_dmas) != tuple(kernel.body.block.args[:2]):
        raise UnsupportedLoweringError("Pallas matmul input DMAs must load the kernel inputs")
    if tuple(operation.destination for operation in input_dmas) != (matmul.lhs, matmul.rhs):
        raise UnsupportedLoweringError("Pallas matmul input DMAs must feed the MXU operands")
    if tuple(operation.token for operation in input_dmas) != tuple(
        operation.token for operation in input_waits
    ):
        raise UnsupportedLoweringError("Pallas matmul input DMA waits do not match their loads")
    if output_dma.source != collective.destination:
        raise UnsupportedLoweringError(
            "Pallas matmul output DMA must store the reduce-scatter destination"
        )
    if output_dma.destination != kernel.body.block.args[2]:
        raise UnsupportedLoweringError("Pallas matmul output DMA must store the kernel output")
    if output_wait.token != output_dma.token:
        raise UnsupportedLoweringError("Pallas matmul output DMA wait does not match its store")
    lhs, rhs, partial = matmul.lhs.type, matmul.rhs.type, matmul.accumulator.type
    output = collective.destination.type
    assert isinstance(lhs, BufferType)
    assert isinstance(rhs, BufferType)
    assert isinstance(partial, BufferType)
    assert isinstance(output, BufferType)
    if (
        not isinstance(lhs.storage.element_type, BFloat16Type)
        or lhs.storage.element_type != rhs.storage.element_type
    ):
        raise UnsupportedLoweringError("Pallas matmul lowering supports bf16 inputs only")
    if (
        not isinstance(partial.storage.element_type, Float32Type)
        or partial.storage.element_type != output.storage.element_type
    ):
        raise UnsupportedLoweringError("Pallas matmul lowering supports f32 outputs only")
    mesh_names = tuple(value.data for value in kernel.mesh_axis_names)
    mesh_sizes = tuple(value.data for value in kernel.mesh_axis_sizes)
    if len(mesh_names) != 1 or collective.mesh_axis.data != mesh_names[0]:
        raise UnsupportedLoweringError("Pallas matmul lowering supports one mesh axis")
    collective_link_bandwidths: tuple[tuple[str, int], ...] = ()
    if kernel.topology is not None:
        if collective.collective_plan is None:
            raise UnsupportedLoweringError("structured Pallas matmul needs a collective plan")
        topology_plan = kernel.topology.plans_by_id()[collective.collective_plan.data]
        links = kernel.topology.links_by_id()
        used_link_ids = sorted(
            {link_id.data for group in topology_plan.groups for link_id in group.route_link_ids}
        )
        collective_link_bandwidths = tuple(
            (link_id, links[link_id].bandwidth_bytes_per_second.data) for link_id in used_link_ids
        )
    collective_implementation = (
        collective.implementation.data
        if isinstance(collective, CollectiveOp) and collective.implementation is not None
        else None
    )
    output_shape = _shape(output)
    if collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING and (
        output_shape[0] % 16 or output_shape[1] % 128 or mesh_sizes[0] < 2
    ):
        raise UnsupportedLoweringError(
            "Pallas bidirectional reduce-scatter needs rank-2 output blocks "
            "with rows divisible by 16 and columns divisible by 128"
        )
    return PallasMatmulPlan(
        name=kernel.sym_name.data,
        schedule_sha256=schedule_sha256(module),
        mesh_axis=mesh_names[0],
        mesh_size=mesh_sizes[0],
        lhs_local_shape=_shape(lhs),
        rhs_local_shape=_shape(rhs),
        partial_local_shape=_shape(partial),
        output_local_shape=output_shape,
        lhs_sharding=_sharding(lhs),
        rhs_sharding=_sharding(rhs),
        output_sharding=_sharding(output),
        scatter_dimension=(
            collective.scatter_dimension.data
            if isinstance(collective, CollectiveReduceScatterOp)
            else collective.split_dimension.data
        ),
        tile_m=matmul.tile_m.data,
        tile_k=matmul.tile_k.data,
        tile_n=matmul.tile_n.data,
        collective_link_bandwidths=collective_link_bandwidths,
        collective_implementation=collective_implementation,
    )


def validate_saved_pallas_plan(
    physical_path: Path,
    pallas_path: Path,
    *,
    schedule_sha256: str,
    pallas_source_sha256: str,
) -> PallasMatmulPlan:
    if file_sha256(physical_path) != schedule_sha256:
        raise ValueError("SAVED_PHYSICAL_IR_HASH_MISMATCH")
    if file_sha256(pallas_path) != pallas_source_sha256:
        raise ValueError("SAVED_PALLAS_SOURCE_HASH_MISMATCH")
    try:
        module = parse_physical_module(physical_path.read_text(), name=str(physical_path))
        plan = replace(
            lower_physical_matmul_to_pallas(module),
            schedule_sha256=schedule_sha256,
        )
    except Exception as error:
        raise ValueError("SAVED_PHYSICAL_IR_INVALID") from error
    try:
        tree = ast.parse(pallas_path.read_text(), filename=str(pallas_path))
    except SyntaxError as error:
        raise ValueError("SAVED_PALLAS_SOURCE_INVALID") from error
    constants: dict[str, object] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
    execution_schema = constants.get("PALLAS_EXECUTION_SCHEMA", LEGACY_PALLAS_EXECUTION_SCHEMA)
    expected = {
        "SCHEDULE_SHA256": schedule_sha256,
        "MESH_AXIS": plan.mesh_axis,
        "MESH_SIZE": plan.mesh_size,
        "LHS_LOCAL_SHAPE": plan.lhs_local_shape,
        "RHS_LOCAL_SHAPE": plan.rhs_local_shape,
        "PARTIAL_LOCAL_SHAPE": plan.partial_local_shape,
        "OUTPUT_SHARDING": plan.output_sharding,
        "LHS_SHARDING": plan.lhs_sharding,
        "RHS_SHARDING": plan.rhs_sharding,
        "SCATTER_DIMENSION": plan.scatter_dimension,
        "TILE_M": plan.tile_m,
        "TILE_K": plan.tile_k,
        "TILE_N": plan.tile_n,
    }
    if plan.collective_link_bandwidths and execution_schema != LEGACY_PALLAS_EXECUTION_SCHEMA:
        expected["COLLECTIVE_LINK_BANDWIDTHS"] = plan.collective_link_bandwidths
    if plan.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING:
        expected.update(
            {
                "COLLECTIVE_IMPLEMENTATION": plan.collective_implementation.value,
                "COLLECTIVE_HBM_SCRATCH_BYTES": plan.collective_hbm_scratch_bytes,
                "COLLECTIVE_ACCUMULATOR_VMEM_BYTES": (plan.collective_accumulator_vmem_bytes),
                "COLLECTIVE_DMA_SEMAPHORE_COUNT": (plan.collective_dma_semaphore_count),
                "COLLECTIVE_CAPACITY_SEMAPHORE_COUNT": (plan.collective_capacity_semaphore_count),
                "COLLECTIVE_STARTUP_SEMAPHORE_COUNT": (plan.collective_startup_semaphore_count),
                "COLLECTIVE_STARTUP_BARRIER_PHASES": (plan.collective_startup_barrier_phases),
                "COLLECTIVE_REMOTE_HALF_OUTPUT_COPY_COUNT": (
                    plan.collective_remote_half_output_copy_count
                ),
                "COLLECTIVE_REMOTE_PAYLOAD_BYTES": (plan.collective_remote_payload_bytes),
                "COLLECTIVE_REMOTE_BIDIRECTIONAL_ENDPOINT_BYTES": (
                    plan.collective_remote_bidirectional_endpoint_bytes
                ),
            }
        )
    if any(constants.get(name) != value for name, value in expected.items()):
        raise ValueError("SAVED_PALLAS_SOURCE_PLAN_MISMATCH")
    name = constants.get("NAME")
    if not isinstance(name, str):
        raise TypeError("SAVED_PALLAS_SOURCE_PLAN_MISMATCH")
    rendering_plan = replace(plan, name=name)
    expected_execution_schema = (
        PALLAS_NATIVE_COLLECTIVE_EXECUTION_SCHEMA
        if plan.collective_implementation is CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING
        else PALLAS_EXECUTION_SCHEMA
    )
    if execution_schema == LEGACY_PALLAS_EXECUTION_SCHEMA:
        if plan.collective_implementation is not None:
            raise ValueError("SAVED_PALLAS_EXECUTION_SCHEMA_MISMATCH")
        expected_source = rendering_plan.render_source()
    elif execution_schema == expected_execution_schema:
        expected_source = rendering_plan.render_executable_source()
    elif execution_schema in {
        PALLAS_EXECUTION_SCHEMA,
        PALLAS_NATIVE_COLLECTIVE_EXECUTION_SCHEMA,
    }:
        raise ValueError("SAVED_PALLAS_EXECUTION_SCHEMA_MISMATCH")
    else:
        raise ValueError("SAVED_PALLAS_EXECUTION_SCHEMA_UNSUPPORTED")
    if expected_source != pallas_path.read_text():
        raise ValueError("SAVED_PALLAS_RENDERING_MISMATCH")
    return rendering_plan
