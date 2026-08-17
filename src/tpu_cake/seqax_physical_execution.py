from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    IntegerType,
    ModuleOp,
    Signedness,
)
from xdsl.ir import SSAValue

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    CollectiveKind,
    CollectiveOp,
    DmaStartOp,
    DmaWaitOp,
    KernelOp,
    MxuEinsumOp,
    SemaphoreAllocOp,
    VectorComputeOp,
    YieldOp,
)


class UnsupportedPhysicalExecutionError(ValueError):
    pass


def _dtype(buffer: BufferType):
    element_type = buffer.storage.element_type
    if isinstance(element_type, BFloat16Type):
        return jnp.bfloat16
    if isinstance(element_type, Float16Type):
        return jnp.float16
    if isinstance(element_type, Float32Type):
        return jnp.float32
    if isinstance(element_type, IntegerType):
        width = element_type.width.data
        if width == 1:
            return jnp.bool_
        signed = element_type.signedness.data is not Signedness.UNSIGNED
        by_width = {
            (8, True): jnp.int8,
            (8, False): jnp.uint8,
            (16, True): jnp.int16,
            (16, False): jnp.uint16,
            (32, True): jnp.int32,
            (32, False): jnp.uint32,
            (64, True): jnp.int64,
            (64, False): jnp.uint64,
        }
        if (width, signed) in by_width:
            return by_width[(width, signed)]
    raise UnsupportedPhysicalExecutionError(f"unsupported physical dtype {element_type}")


def _names(buffer: BufferType) -> tuple[str, ...]:
    return tuple(value.data for value in buffer.shape.dimensions)


def _configuration(operation: VectorComputeOp) -> dict[str, str]:
    return dict(value.data.split("=", 1) for value in operation.configuration)


def _align_named(
    value: jax.Array,
    source_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> jax.Array:
    retained = tuple(name for name in target_names if name in source_names)
    if set(retained) != set(source_names):
        raise UnsupportedPhysicalExecutionError(
            "physical named broadcast cannot drop a source dimension"
        )
    permutation = tuple(source_names.index(name) for name in retained)
    if permutation != tuple(range(len(permutation))):
        value = jnp.transpose(value, permutation)
    shape = tuple(
        value.shape[retained.index(name)] if name in retained else 1 for name in target_names
    )
    return jnp.reshape(value, shape)


def _vector_compute(
    operation: VectorComputeOp,
    values: tuple[jax.Array, ...],
    mesh: dict[str, int],
) -> jax.Array:
    function = operation.function.data
    configuration = _configuration(operation)
    output_type = operation.output.type
    assert isinstance(output_type, BufferType)
    input_types = tuple(value.type for value in operation.inputs)
    assert all(isinstance(value, BufferType) for value in input_types)
    typed_inputs = tuple(value for value in input_types if isinstance(value, BufferType))

    if function in {"cast", "rename_dimension"}:
        result = values[0]
    elif function == "slice":
        axis = _names(typed_inputs[0]).index(configuration["dimension"])
        result = jnp.take(values[0], int(configuration["index"]), axis=axis)
    elif function == "rms_norm":
        value, scale = values
        value_type, scale_type = typed_inputs
        aligned_scale = _align_named(scale, _names(scale_type), _names(value_type))
        axis = _names(value_type).index(configuration["dimension"])
        mean_square = jnp.mean(
            jnp.square(value.astype(jnp.float32)),
            axis=axis,
            keepdims=True,
        )
        normalized = value * jax.lax.rsqrt(mean_square + float(configuration["epsilon"]))
        result = normalized * aligned_scale
    elif function == "rotary_embedding":
        value = values[0]
        names = _names(typed_inputs[0])
        sequence_axis = names.index(configuration["sequence_dimension"])
        head_axis = names.index(configuration["head_dimension"])
        if head_axis != value.ndim - 1:
            raise UnsupportedPhysicalExecutionError(
                "physical RoPE requires the head dimension last"
            )
        half_head = value.shape[-1] // 2
        timescale = jnp.logspace(
            0,
            jnp.log10(jnp.float32(configuration["maximum_timescale"])),
            half_head,
            endpoint=False,
        )
        position = jnp.arange(value.shape[sequence_axis], dtype=jnp.int32)
        angle = position[:, None].astype(jnp.float32) / timescale[None, :]
        shape = [1] * value.ndim
        shape[sequence_axis] = value.shape[sequence_axis]
        shape[-1] = half_head
        sine = jnp.sin(angle).reshape(shape)
        cosine = jnp.cos(angle).reshape(shape)
        first, second = jnp.split(value, 2, axis=-1)
        result = jnp.concatenate(
            (first * cosine - second * sine, second * cosine + first * sine),
            axis=-1,
        )
    elif function == "packed_causal_mask":
        starts = values[0]
        axis = _names(typed_inputs[0]).index(configuration["sequence_dimension"])
        segments = jnp.cumsum(starts, axis=axis)
        same_segment = segments[:, :, None] == segments[:, None, :]
        causal = jnp.tril(jnp.ones(same_segment.shape[1:], dtype=jnp.bool_))
        result = same_segment & causal[None, :, :]
    elif function == "masked_softmax":
        value, mask = values
        value_type, mask_type = typed_inputs
        aligned_mask = _align_named(mask, _names(mask_type), _names(value_type))
        axis = _names(value_type).index(configuration["dimension"])
        result = jax.nn.softmax(
            jnp.where(aligned_mask, value, -1e10),
            axis=axis,
        )
    elif function == "embedding_lookup":
        table, indices = values
        table_type, indices_type = typed_inputs
        table_names = _names(table_type)
        indices_names = _names(indices_type)
        axis = table_names.index(configuration["vocabulary_dimension"])
        sharding = tuple(
            filter(
                None,
                tuple(table_type.sharding.axes)[axis].data.split("/"),
            )
        )
        valid = None
        if sharding:
            shard_index = jnp.int32(0)
            for mesh_axis in sharding:
                shard_index = shard_index * mesh[mesh_axis] + jax.lax.axis_index(mesh_axis)
            local_vocabulary = table.shape[axis]
            offset = shard_index * local_vocabulary
            local_indices = indices.astype(jnp.int32) - offset
            valid = (local_indices >= 0) & (local_indices < local_vocabulary)
            local_indices = jnp.clip(local_indices, 0, local_vocabulary - 1)
            result = jnp.take(table, local_indices, axis=axis)
        else:
            result = jnp.take(table, indices, axis=axis)
        take_names = (*table_names[:axis], *indices_names, *table_names[axis + 1 :])
        output_names = _names(output_type)
        permutation = tuple(take_names.index(name) for name in output_names)
        if permutation != tuple(range(len(permutation))):
            result = jnp.transpose(result, permutation)
        if valid is not None:
            aligned_valid = _align_named(valid, indices_names, output_names)
            result = jnp.where(aligned_valid, result, 0)
    elif function == "add":
        result = values[0] + values[1]
    elif function == "multiply":
        result = values[0] * values[1]
    elif function == "silu":
        result = jax.nn.silu(values[0])
    elif function == "exp":
        result = jnp.exp(values[0])
    else:
        raise UnsupportedPhysicalExecutionError(
            f"unsupported physical vector function {function!r}"
        )
    result = jnp.asarray(result, dtype=_dtype(output_type))
    if tuple(result.shape) != output_type.storage.get_shape():
        raise UnsupportedPhysicalExecutionError(
            f"physical {function} produced {tuple(result.shape)}, "
            f"expected {output_type.storage.get_shape()}"
        )
    return result


def execute_seqax_physical_program_jax(
    module: ModuleOp,
    inputs: Sequence[jax.Array],
    *,
    einsum: Callable[[MxuEinsumOp, jax.Array, jax.Array], jax.Array],
) -> tuple[jax.Array, ...]:
    module.verify()
    kernels = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, KernelOp)
    )
    if len(kernels) != 1 or len(tuple(module.body.block.ops)) != 1:
        raise UnsupportedPhysicalExecutionError("physical execution expects exactly one TPU kernel")
    kernel = kernels[0]
    block = kernel.body.block
    if kernel.argument_modes is None:
        raise UnsupportedPhysicalExecutionError(
            "physical execution requires explicit kernel argument modes"
        )
    modes = tuple(value.data for value in kernel.argument_modes)
    input_arguments = tuple(
        argument
        for argument, mode in zip(block.args, modes, strict=True)
        if mode in {"input", "inout"}
    )
    output_arguments = tuple(
        argument
        for argument, mode in zip(block.args, modes, strict=True)
        if mode in {"output", "inout"}
    )
    if not output_arguments:
        raise UnsupportedPhysicalExecutionError(
            "physical execution requires at least one external output"
        )
    if len(inputs) != len(input_arguments):
        raise ValueError(
            f"physical execution expects {len(input_arguments)} inputs, got {len(inputs)}"
        )
    for argument, value in zip(input_arguments, inputs, strict=True):
        argument_type = argument.type
        assert isinstance(argument_type, BufferType)
        if tuple(value.shape) != argument_type.storage.get_shape():
            raise ValueError(
                f"physical input shape {tuple(value.shape)} does not match "
                f"{argument_type.storage.get_shape()}"
            )
        if jnp.dtype(value.dtype) != jnp.dtype(_dtype(argument_type)):
            raise ValueError(
                f"physical input dtype {value.dtype} does not match "
                f"{jnp.dtype(_dtype(argument_type))}"
            )
    environment: dict[SSAValue, jax.Array] = dict(zip(input_arguments, inputs, strict=True))
    pending_dma: dict[SSAValue, jax.Array] = {}
    external_outputs: dict[SSAValue, jax.Array] = {}
    mesh = dict(
        zip(
            (value.data for value in kernel.mesh_axis_names),
            (value.data for value in kernel.mesh_axis_sizes),
            strict=True,
        )
    )

    for operation in block.ops:
        if isinstance(operation, (AllocOp, SemaphoreAllocOp)):
            continue
        if isinstance(operation, DmaStartOp):
            pending_dma[operation.token] = environment[operation.source]
            continue
        if isinstance(operation, DmaWaitOp):
            start = operation.token.owner
            assert isinstance(start, DmaStartOp)
            value = pending_dma.pop(operation.token)
            environment[start.destination] = value
            if start.destination in output_arguments:
                external_outputs[start.destination] = value
            continue
        if isinstance(operation, VectorComputeOp):
            values = tuple(environment[value] for value in operation.inputs)
            environment[operation.output] = _vector_compute(operation, values, mesh)
            continue
        if isinstance(operation, MxuEinsumOp):
            environment[operation.accumulator] = einsum(
                operation,
                environment[operation.lhs],
                environment[operation.rhs],
            )
            continue
        if isinstance(operation, CollectiveOp):
            source = environment[operation.source]
            if operation.reducer.data not in {"none", "sum"}:
                raise UnsupportedPhysicalExecutionError(
                    "physical Seqax execution supports sum collectives only"
                )
            if operation.kind.data is CollectiveKind.ALL_GATHER:
                result = jax.lax.all_gather(
                    source,
                    operation.mesh_axis.data,
                    axis=operation.concat_dimension.data,
                    tiled=True,
                )
            elif operation.kind.data is CollectiveKind.REDUCE_SCATTER:
                result = jax.lax.psum_scatter(
                    source,
                    operation.mesh_axis.data,
                    scatter_dimension=operation.split_dimension.data,
                    tiled=True,
                )
            elif operation.kind.data is CollectiveKind.ALL_REDUCE:
                result = jax.lax.psum(source, operation.mesh_axis.data)
            elif operation.kind.data is CollectiveKind.ALL_TO_ALL:
                result = jax.lax.all_to_all(
                    source,
                    operation.mesh_axis.data,
                    split_axis=operation.split_dimension.data,
                    concat_axis=operation.concat_dimension.data,
                    tiled=True,
                )
            else:
                raise UnsupportedPhysicalExecutionError(
                    f"unsupported physical collective {operation.kind.data}"
                )
            environment[operation.destination] = result
            continue
        if isinstance(operation, YieldOp):
            if pending_dma:
                raise UnsupportedPhysicalExecutionError(
                    "physical execution reached yield with DMA operations in flight"
                )
            if any(value not in external_outputs for value in output_arguments):
                raise UnsupportedPhysicalExecutionError(
                    "physical execution did not initialize every external output"
                )
            return tuple(external_outputs[value] for value in output_arguments)
        raise UnsupportedPhysicalExecutionError(f"no Seqax physical execution for {operation.name}")
    raise UnsupportedPhysicalExecutionError("physical TPU kernel has no yield")
