from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec
from xdsl.dialects.builtin import BFloat16Type, Float32Type, ModuleOp

from tpu_cake.dialects.tpu_schedule import (
    BufferType,
    CollectiveReduceScatterOp,
    KernelOp,
    MxuMatmulOp,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.lowering import UnsupportedLoweringError


def _matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = lax.dot_general(
        lhs_ref[...],
        rhs_ref[...],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _partition_spec(sharding: tuple[str, ...]) -> PartitionSpec:
    return PartitionSpec(*(axis or None for axis in sharding))


@dataclass(frozen=True)
class PallasMatmulPlan:
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

    @property
    def global_lhs_shape(self) -> tuple[int, int]:
        return self._global_shape(self.lhs_local_shape, self.lhs_sharding)

    @property
    def global_rhs_shape(self) -> tuple[int, int]:
        return self._global_shape(self.rhs_local_shape, self.rhs_sharding)

    @property
    def global_output_shape(self) -> tuple[int, int]:
        return self._global_shape(self.output_local_shape, self.output_sharding)

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
                pl.BlockSpec(self.lhs_local_shape, lambda: (0, 0)),
                pl.BlockSpec(self.rhs_local_shape, lambda: (0, 0)),
            ),
            out_specs=pl.BlockSpec(self.partial_local_shape, lambda: (0, 0)),
            grid=(),
            interpret=interpret_setting,
            name=self.name,
            metadata={"schedule_sha256": self.schedule_sha256},
        )

        def distributed(lhs, rhs):
            partial = local_call(lhs, rhs)
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
        fields = (
            f"name={self.name!r}",
            f"schedule_sha256={self.schedule_sha256!r}",
            f"mesh_axis={self.mesh_axis!r}",
            f"mesh_size={self.mesh_size}",
            f"lhs_local_shape={self.lhs_local_shape!r}",
            f"rhs_local_shape={self.rhs_local_shape!r}",
            f"partial_local_shape={self.partial_local_shape!r}",
            f"output_local_shape={self.output_local_shape!r}",
            f"scatter_dimension={self.scatter_dimension}",
        )
        return "\n".join((inspect.getsource(_matmul_kernel).rstrip(), *fields, ""))

    def source_sha256(self) -> str:
        return hashlib.sha256(self.render_source().encode()).hexdigest()


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
    matmuls = [
        operation for operation in kernel.body.block.ops if isinstance(operation, MxuMatmulOp)
    ]
    collectives = [
        operation
        for operation in kernel.body.block.ops
        if isinstance(operation, CollectiveReduceScatterOp)
    ]
    if len(matmuls) != 1 or len(collectives) != 1:
        raise UnsupportedLoweringError(
            "Pallas lowering supports one MXU matmul followed by one reduce-scatter"
        )
    matmul, collective = matmuls[0], collectives[0]
    if collective.source != matmul.accumulator:
        raise UnsupportedLoweringError("Pallas reduce-scatter must consume the MXU accumulator")
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
    return PallasMatmulPlan(
        name=kernel.sym_name.data,
        schedule_sha256=schedule_sha256(module),
        mesh_axis=mesh_names[0],
        mesh_size=mesh_sizes[0],
        lhs_local_shape=_shape(lhs),
        rhs_local_shape=_shape(rhs),
        partial_local_shape=_shape(partial),
        output_local_shape=_shape(output),
        lhs_sharding=_sharding(lhs),
        rhs_sharding=_sharding(rhs),
        output_sharding=_sharding(output),
        scatter_dimension=collective.scatter_dimension.data,
    )
