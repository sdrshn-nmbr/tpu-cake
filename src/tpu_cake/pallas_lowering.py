from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec
from xdsl.context import Context
from xdsl.dialects.builtin import BFloat16Type, Builtin, Float32Type, ModuleOp
from xdsl.parser import Parser

from tpu_cake.dialects.tpu_schedule import (
    BufferType,
    CollectiveKind,
    CollectiveOp,
    CollectiveReduceScatterOp,
    KernelOp,
    MxuMatmulOp,
    TPUSchedule,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.lowering import UnsupportedLoweringError

LEGACY_PALLAS_EXECUTION_SCHEMA = "standalone-rendering-v1"
PALLAS_EXECUTION_SCHEMA = "delegated-plan-v2"


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
    tile_m: int
    tile_k: int
    tile_n: int

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
        return f'''from __future__ import annotations

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
'''

    def render_executable_source(self) -> str:
        return f'''from __future__ import annotations

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
TILE_N = {self.tile_n}

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
)


def build(*, interpret=False, devices=None):
    return PLAN.build(interpret=interpret, devices=devices)
'''

    def source_sha256(self) -> str:
        return hashlib.sha256(self.render_executable_source().encode()).hexdigest()


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
        scatter_dimension=(
            collective.scatter_dimension.data
            if isinstance(collective, CollectiveReduceScatterOp)
            else collective.split_dimension.data
        ),
        tile_m=matmul.tile_m.data,
        tile_k=matmul.tile_k.data,
        tile_n=matmul.tile_n.data,
    )


def validate_saved_pallas_plan(
    physical_path: Path,
    pallas_path: Path,
    *,
    schedule_sha256: str,
    pallas_source_sha256: str,
) -> PallasMatmulPlan:
    if hashlib.sha256(physical_path.read_bytes()).hexdigest() != schedule_sha256:
        raise ValueError("SAVED_PHYSICAL_IR_HASH_MISMATCH")
    if hashlib.sha256(pallas_path.read_bytes()).hexdigest() != pallas_source_sha256:
        raise ValueError("SAVED_PALLAS_SOURCE_HASH_MISMATCH")
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(TPUSchedule)
    try:
        module = Parser(
            context, physical_path.read_text(), name=str(physical_path)
        ).parse_module()
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
    if any(constants.get(name) != value for name, value in expected.items()):
        raise ValueError("SAVED_PALLAS_SOURCE_PLAN_MISMATCH")
    name = constants.get("NAME")
    if not isinstance(name, str):
        raise TypeError("SAVED_PALLAS_SOURCE_PLAN_MISMATCH")
    rendering_plan = replace(plan, name=name)
    execution_schema = constants.get(
        "PALLAS_EXECUTION_SCHEMA", LEGACY_PALLAS_EXECUTION_SCHEMA
    )
    expected_source = (
        rendering_plan.render_source()
        if execution_schema == LEGACY_PALLAS_EXECUTION_SCHEMA
        else rendering_plan.render_executable_source()
        if execution_schema == PALLAS_EXECUTION_SCHEMA
        else None
    )
    if expected_source is None:
        raise ValueError("SAVED_PALLAS_EXECUTION_SCHEMA_UNSUPPORTED")
    if expected_source != pallas_path.read_text():
        raise ValueError("SAVED_PALLAS_RENDERING_MISMATCH")
    return rendering_plan
