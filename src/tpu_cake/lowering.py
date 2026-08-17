from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import BFloat16Type, Float32Type, ModuleOp
from xdsl.ir import Operation

from tpu_cake.dialects.distributed_tensor import (
    DTensorType,
    EinsumLocalOp,
    ProgramOp,
    ReduceScatterOp,
    ReturnOp,
)
from tpu_cake.dialects.tpu_schedule import MemorySpace, Ownership
from tpu_cake.frontend import KernelBuilder, buffer


class UnsupportedLoweringError(ValueError):
    pass


@dataclass(frozen=True)
class LoweringTarget:
    name: str
    vmem_capacity_bytes: int
    smem_capacity_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("lowering target needs a name")
        if self.vmem_capacity_bytes <= 0 or self.smem_capacity_bytes <= 0:
            raise ValueError("lowering target memory capacities must be positive")


TPU7X_TARGET = LoweringTarget(
    name="tpu7x",
    vmem_capacity_bytes=64 << 20,
    smem_capacity_bytes=16 << 20,
)


def _reject(operation: Operation, message: str) -> UnsupportedLoweringError:
    return UnsupportedLoweringError(f"{message}: {operation.name} at {operation.location}")


def _local_shape(tensor: DTensorType, mesh: dict[str, int]) -> tuple[int, ...]:
    shape: list[int] = []
    for (_, size), axes in zip(tensor.logical_shape(), tensor.sharding_axes(), strict=True):
        divisor = 1
        for axis in axes:
            divisor *= mesh[axis]
        shape.append(size // divisor)
    return tuple(shape)


def lower_distributed_matmul(
    module: ModuleOp,
    *,
    target: LoweringTarget = TPU7X_TARGET,
) -> ModuleOp:
    module.verify()
    top_level = list(module.body.block.ops)
    if len(top_level) != 1 or not isinstance(top_level[0], ProgramOp):
        operation = top_level[0] if top_level else module
        raise _reject(operation, "expected one distributed program")
    program = top_level[0]
    operations = list(program.body.block.ops)
    if len(operations) != 3:
        unexpected = next(
            (
                operation
                for operation in operations
                if not isinstance(operation, (EinsumLocalOp, ReduceScatterOp, ReturnOp))
            ),
            program,
        )
        raise _reject(unexpected, "only einsum, reduce-scatter, and return are supported")
    einsum, collective, terminator = operations
    if not isinstance(einsum, EinsumLocalOp):
        raise _reject(einsum, "the first operation must be a local einsum")
    if not isinstance(collective, ReduceScatterOp):
        raise _reject(collective, "the second operation must be a reduce-scatter")
    if not isinstance(terminator, ReturnOp):
        raise _reject(terminator, "the final operation must be a return")
    if collective.value.owner is not einsum:
        raise _reject(collective, "reduce-scatter must consume the einsum result")
    if len(terminator.values) != 1 or terminator.values[0].owner is not collective:
        raise _reject(terminator, "the program must return the reduce-scatter result")
    if len(program.body.block.args) != 2:
        raise _reject(program, "distributed matmul needs exactly two inputs")
    lhs_type, rhs_type = (value.type for value in program.body.block.args)
    partial_type = einsum.result.type
    result_type = collective.result.type
    assert isinstance(lhs_type, DTensorType)
    assert isinstance(rhs_type, DTensorType)
    assert isinstance(partial_type, DTensorType)
    assert isinstance(result_type, DTensorType)
    if (
        not isinstance(lhs_type.element_type, BFloat16Type)
        or lhs_type.element_type != rhs_type.element_type
    ):
        raise _reject(einsum, "the first lowerer supports bf16 inputs only")
    if (
        not isinstance(partial_type.element_type, Float32Type)
        or partial_type.element_type != result_type.element_type
    ):
        raise _reject(einsum, "the first lowerer supports f32 accumulation only")
    lhs_dimensions = [name for name, _ in lhs_type.logical_shape()]
    rhs_dimensions = [name for name, _ in rhs_type.logical_shape()]
    contraction = einsum.contracting_dimension.data
    if (
        len(lhs_dimensions) != 2
        or len(rhs_dimensions) != 2
        or lhs_dimensions[1] != contraction
        or rhs_dimensions[0] != contraction
    ):
        raise _reject(
            einsum,
            "the first lowerer requires rank-2 lhs[M,K] and rhs[K,N] orientation",
        )
    axes = tuple(value.data for value in collective.axes)
    dimensions = tuple(value.data for value in collective.scatter_dimensions)
    if len(axes) != 1 or len(dimensions) != 1 or collective.reducer.data != "sum":
        raise _reject(collective, "the first lowerer supports one sum reduce-scatter axis")
    mesh = program.mesh.sizes()
    mesh_axis = axes[0]
    scatter_dimension = dimensions[0]
    partial_dimensions = [name for name, _ in partial_type.logical_shape()]
    scatter_index = partial_dimensions.index(scatter_dimension)

    lhs_shape = _local_shape(lhs_type, mesh)
    rhs_shape = _local_shape(rhs_type, mesh)
    partial_shape = _local_shape(partial_type, mesh)
    result_shape = _local_shape(result_type, mesh)
    external = {
        "memory": MemorySpace.HBM,
        "ownership": Ownership.EXTERNAL,
        "lifetime": (0, 5),
    }
    inputs = (
        buffer(
            lhs_shape,
            tuple(name for name, _ in lhs_type.logical_shape()),
            lhs_type.element_type,
            sharding=tuple("/".join(axes) for axes in lhs_type.sharding_axes()),
            **external,
        ),
        buffer(
            rhs_shape,
            tuple(name for name, _ in rhs_type.logical_shape()),
            rhs_type.element_type,
            sharding=tuple("/".join(axes) for axes in rhs_type.sharding_axes()),
            **external,
        ),
        buffer(
            result_shape,
            tuple(name for name, _ in result_type.logical_shape()),
            result_type.element_type,
            sharding=tuple("/".join(axes) for axes in result_type.sharding_axes()),
            **external,
        ),
    )
    builder = KernelBuilder(
        f"{program.sym_name.data}_physical",
        target.name,
        inputs,
        vmem_capacity_bytes=target.vmem_capacity_bytes,
        smem_capacity_bytes=target.smem_capacity_bytes,
        mesh=mesh,
    )
    lhs = builder.alloc(
        buffer(
            lhs_shape,
            tuple(name for name, _ in lhs_type.logical_shape()),
            lhs_type.element_type,
            memory=MemorySpace.VMEM,
            sharding=tuple("/".join(axes) for axes in lhs_type.sharding_axes()),
            lifetime=(0, 2),
        ),
        "lhs_tile",
    )
    rhs = builder.alloc(
        buffer(
            rhs_shape,
            tuple(name for name, _ in rhs_type.logical_shape()),
            rhs_type.element_type,
            memory=MemorySpace.VMEM,
            sharding=tuple("/".join(axes) for axes in rhs_type.sharding_axes()),
            lifetime=(0, 2),
        ),
        "rhs_tile",
    )
    partial = builder.alloc(
        buffer(
            partial_shape,
            tuple(name for name, _ in partial_type.logical_shape()),
            partial_type.element_type,
            memory=MemorySpace.VMEM,
            lifetime=(2, 3),
        ),
        "partial_accumulator",
    )
    reduced = builder.alloc(
        buffer(
            result_shape,
            tuple(name for name, _ in result_type.logical_shape()),
            result_type.element_type,
            memory=MemorySpace.VMEM,
            sharding=tuple("/".join(axes) for axes in result_type.sharding_axes()),
            lifetime=(3, 5),
        ),
        "reduced_output",
    )
    lhs_semaphore = builder.semaphore()
    rhs_semaphore = builder.semaphore()
    output_semaphore = builder.semaphore()
    lhs_dma = builder.dma_start(builder.inputs[0], lhs, lhs_semaphore, stage=0)
    rhs_dma = builder.dma_start(builder.inputs[1], rhs, rhs_semaphore, stage=0)
    builder.dma_wait(lhs_dma, stage=1)
    builder.dma_wait(rhs_dma, stage=1)
    physical_matmul = builder.matmul(lhs, rhs, partial, stage=2)
    physical_matmul.location = einsum.location
    physical_collective = builder.collective_reduce_scatter(
        partial,
        reduced,
        stage=3,
        mesh_axis=mesh_axis,
        group_size=mesh[mesh_axis],
        scatter_dimension=scatter_index,
    )
    physical_collective.location = collective.location
    output_dma = builder.dma_start(reduced, builder.inputs[2], output_semaphore, stage=4)
    builder.dma_wait(output_dma, stage=5)
    return builder.module()
