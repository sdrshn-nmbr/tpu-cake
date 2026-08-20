from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    IntegerType,
    ModuleOp,
    Signedness,
)
from xdsl.ir import Block, SSAValue

from tpu_cake.dialects.distributed_tensor import (
    AllGatherOp,
    AllReduceOp,
    BroadcastOp,
    CastOp,
    DTensorType,
    EinsumLocalOp,
    EinsumOp,
    ElementwiseOp,
    EmbeddingLookupOp,
    LayerScanOp,
    MaskedSoftmaxOp,
    PackedCausalMaskOp,
    ProgramOp,
    ReduceLocalOp,
    ReduceScatterOp,
    RenameDimensionOp,
    ReturnOp,
    RmsNormOp,
    RotaryEmbeddingOp,
    ScanYieldOp,
    SliceOp,
    TransposeOp,
)


class UnsupportedInterpretationError(ValueError):
    pass


@dataclass(frozen=True)
class _ExecutionSemantics:
    mesh: dict[str, int] | None = None

    @property
    def physical(self) -> bool:
        return self.mesh is not None

    def shape(self, value_type: DTensorType) -> tuple[int, ...]:
        if self.mesh is None:
            return tuple(size for _, size in value_type.logical_shape())
        result = []
        for (_, size), axes in zip(
            value_type.logical_shape(), value_type.sharding_axes(), strict=True
        ):
            divisor = 1
            for axis in axes:
                divisor *= self.mesh[axis]
            result.append(size // divisor)
        return tuple(result)


_LOGICAL_SEMANTICS = _ExecutionSemantics()


def _strict_typed_silu(value: jax.Array) -> jax.Array:
    return jax.nn.silu(value.astype(jnp.float32)).astype(value.dtype)


def _axis_name(axes: tuple[str, ...]):
    if not axes:
        raise UnsupportedInterpretationError("collective needs at least one mesh axis")
    return axes[0] if len(axes) == 1 else axes


def _names(value_type: DTensorType) -> tuple[str, ...]:
    return tuple(name for name, _ in value_type.logical_shape())


def _dtype(value_type: DTensorType):
    element_type = value_type.element_type
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
    raise UnsupportedInterpretationError(f"unsupported element type {element_type}")


def _cast(value: jax.Array, value_type: DTensorType) -> jax.Array:
    return jnp.asarray(value, dtype=_dtype(value_type))


def _align_named(
    value: jax.Array,
    source_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> jax.Array:
    retained = tuple(name for name in target_names if name in source_names)
    if set(retained) != set(source_names):
        raise UnsupportedInterpretationError("named broadcast cannot drop a source dimension")
    permutation = tuple(source_names.index(name) for name in retained)
    if permutation != tuple(range(len(permutation))):
        value = jnp.transpose(value, permutation)
    shape = tuple(
        value.shape[retained.index(name)] if name in retained else 1 for name in target_names
    )
    return jnp.reshape(value, shape)


def _einsum(operation: EinsumOp | EinsumLocalOp, environment: dict[SSAValue, jax.Array]):
    lhs = environment[operation.lhs]
    rhs = environment[operation.rhs]
    lhs_type = operation.lhs.type
    rhs_type = operation.rhs.type
    result_type = operation.result.type
    assert isinstance(lhs_type, DTensorType)
    assert isinstance(rhs_type, DTensorType)
    assert isinstance(result_type, DTensorType)
    names = tuple(dict.fromkeys((*_names(lhs_type), *_names(rhs_type))))
    labels = {name: index for index, name in enumerate(names)}
    result = jnp.einsum(
        lhs,
        [labels[name] for name in _names(lhs_type)],
        rhs,
        [labels[name] for name in _names(rhs_type)],
        [labels[name] for name in _names(result_type)],
        preferred_element_type=jnp.float32,
    )
    return _cast(result, result_type)


def _rope(value: jax.Array, operation: RotaryEmbeddingOp) -> jax.Array:
    value_type = operation.value.type
    assert isinstance(value_type, DTensorType)
    names = _names(value_type)
    sequence_axis = names.index(operation.sequence_dimension.data)
    head_axis = names.index(operation.head_dimension.data)
    if head_axis != value.ndim - 1:
        raise UnsupportedInterpretationError("RoPE interpretation needs the head dimension last")
    half_head = value.shape[-1] // 2
    timescale = jnp.logspace(
        0,
        jnp.log10(jnp.float32(operation.maximum_timescale.data)),
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
    return jnp.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine),
        axis=-1,
    )


def _execute_block(
    block: Block,
    environment: dict[SSAValue, jax.Array],
    semantics: _ExecutionSemantics = _LOGICAL_SEMANTICS,
    strict_mlp_checkpoints: list[list[jax.Array]] | None = None,
) -> tuple[jax.Array, ...] | None:
    for operation in block.ops:
        if isinstance(operation, (ReturnOp, ScanYieldOp)):
            return tuple(environment[value] for value in operation.values)
        if isinstance(operation, AllGatherOp):
            result = environment[operation.value]
            if semantics.physical:
                before = operation.value.type
                after = operation.result.type
                assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
                for dimension, (old_axes, new_axes) in enumerate(
                    zip(before.sharding_axes(), after.sharding_axes(), strict=True)
                ):
                    removed = old_axes[len(new_axes) :]
                    if removed:
                        result = jax.lax.all_gather(
                            result,
                            _axis_name(removed),
                            axis=dimension,
                            tiled=True,
                        )
        elif isinstance(operation, AllReduceOp):
            result = environment[operation.value]
            if semantics.physical:
                axes = tuple(value.data for value in operation.axes)
                reducer = operation.reducer.data
                collective = {
                    "sum": jax.lax.psum,
                    "max": jax.lax.pmax,
                    "min": jax.lax.pmin,
                }[reducer]
                result = collective(result, _axis_name(axes))
        elif isinstance(operation, ReduceScatterOp):
            result = environment[operation.value]
            if semantics.physical:
                axes = tuple(value.data for value in operation.axes)
                dimensions = tuple(value.data for value in operation.scatter_dimensions)
                if operation.reducer.data != "sum":
                    raise UnsupportedInterpretationError(
                        "physical reduce-scatter currently supports sum only"
                    )
                before = operation.value.type
                assert isinstance(before, DTensorType)
                if len(dimensions) != 1:
                    raise UnsupportedInterpretationError(
                        "physical reduce-scatter currently supports one scatter dimension"
                    )
                scatter_dimension = _names(before).index(dimensions[0])
                result = jax.lax.psum_scatter(
                    result,
                    _axis_name(axes),
                    scatter_dimension=scatter_dimension,
                    tiled=True,
                )
            if strict_mlp_checkpoints and len(strict_mlp_checkpoints[-1]) == 4:
                producer = operation.value.owner
                if isinstance(producer, (EinsumOp, EinsumLocalOp)):
                    hidden = producer.lhs.owner
                    if (
                        isinstance(hidden, ElementwiseOp)
                        and hidden.function.data == "multiply"
                        and hidden.materialization is not None
                    ):
                        strict_mlp_checkpoints[-1].append(result)
        elif isinstance(operation, CastOp):
            result_type = operation.result.type
            assert isinstance(result_type, DTensorType)
            result = _cast(environment[operation.value], result_type)
            if (
                strict_mlp_checkpoints
                and len(strict_mlp_checkpoints[-1]) == 5
                and isinstance(operation.value.owner, ReduceScatterOp)
            ):
                strict_mlp_checkpoints[-1].append(result)
        elif isinstance(operation, RmsNormOp):
            value = environment[operation.value]
            scale = environment[operation.scale]
            value_type = operation.value.type
            scale_type = operation.scale.type
            result_type = operation.result.type
            assert isinstance(value_type, DTensorType)
            assert isinstance(scale_type, DTensorType)
            assert isinstance(result_type, DTensorType)
            aligned_scale = _align_named(scale, _names(scale_type), _names(value_type))
            axis = _names(value_type).index(operation.dimension.data)
            mean_square = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=axis, keepdims=True)
            normalized = value * jax.lax.rsqrt(mean_square + float(operation.epsilon.data))
            result = _cast(normalized * aligned_scale, result_type)
        elif isinstance(operation, RotaryEmbeddingOp):
            result_type = operation.result.type
            assert isinstance(result_type, DTensorType)
            result = _cast(_rope(environment[operation.value], operation), result_type)
        elif isinstance(operation, SliceOp):
            value_type = operation.value.type
            assert isinstance(value_type, DTensorType)
            axis = _names(value_type).index(operation.dimension.data)
            result = jnp.take(environment[operation.value], operation.index.data, axis=axis)
        elif isinstance(operation, RenameDimensionOp):
            result = environment[operation.value]
        elif isinstance(operation, PackedCausalMaskOp):
            starts = environment[operation.sequence_starts]
            starts_type = operation.sequence_starts.type
            assert isinstance(starts_type, DTensorType)
            axis = _names(starts_type).index(operation.sequence_dimension.data)
            if starts.ndim != 2 or axis != 1:
                raise UnsupportedInterpretationError(
                    "packed-mask interpretation needs [batch, sequence] inputs"
                )
            segments = jnp.cumsum(starts, axis=axis)
            same_segment = segments[:, :, None] == segments[:, None, :]
            causal = jnp.tril(jnp.ones(same_segment.shape[1:], dtype=jnp.bool_))
            result = same_segment & causal[None, :, :]
        elif isinstance(operation, MaskedSoftmaxOp):
            value = environment[operation.value]
            mask = environment[operation.mask]
            value_type = operation.value.type
            mask_type = operation.mask.type
            result_type = operation.result.type
            assert isinstance(value_type, DTensorType)
            assert isinstance(mask_type, DTensorType)
            assert isinstance(result_type, DTensorType)
            aligned_mask = _align_named(mask, _names(mask_type), _names(value_type))
            axis = _names(value_type).index(operation.dimension.data)
            result = _cast(
                jax.nn.softmax(jnp.where(aligned_mask, value, -1e10), axis=axis),
                result_type,
            )
        elif isinstance(operation, ReduceLocalOp):
            value = environment[operation.value]
            value_type = operation.value.type
            assert isinstance(value_type, DTensorType)
            axes = tuple(
                _names(value_type).index(dimension.data) for dimension in operation.dimensions
            )
            reducer = operation.reducer.data
            reduce = {"sum": jnp.sum, "max": jnp.max, "min": jnp.min}[reducer]
            result = reduce(value, axis=axes)
        elif isinstance(operation, TransposeOp):
            result = jnp.transpose(
                environment[operation.value],
                tuple(index.data for index in operation.permutation),
            )
        elif isinstance(operation, BroadcastOp):
            before = operation.value.type
            after = operation.result.type
            assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
            result = jnp.broadcast_to(
                _align_named(environment[operation.value], _names(before), _names(after)),
                tuple(size for _, size in after.logical_shape()),
            )
        elif isinstance(operation, EmbeddingLookupOp):
            table = environment[operation.table]
            indices = environment[operation.indices]
            table_type = operation.table.type
            assert isinstance(table_type, DTensorType)
            axis = _names(table_type).index(operation.vocabulary_dimension.data)
            vocabulary_axes = table_type.sharding_axes()[axis]
            if semantics.physical and vocabulary_axes:
                assert semantics.mesh is not None
                shard_index = jnp.int32(0)
                for mesh_axis in vocabulary_axes:
                    shard_index = shard_index * semantics.mesh[mesh_axis] + jax.lax.axis_index(
                        mesh_axis
                    )
                local_vocabulary = table.shape[axis]
                offset = shard_index * local_vocabulary
                local_indices = indices.astype(jnp.int32) - offset
                valid = (local_indices >= 0) & (local_indices < local_vocabulary)
                local_indices = jnp.clip(local_indices, 0, local_vocabulary - 1)
                result = jnp.take(table, local_indices, axis=axis)
                valid_shape = (*valid.shape, *((1,) * (table.ndim - 1)))
                result = jnp.where(valid.reshape(valid_shape), result, 0)
            else:
                result = jnp.take(table, indices, axis=axis)
        elif isinstance(operation, (EinsumOp, EinsumLocalOp)):
            result = _einsum(operation, environment)
        elif isinstance(operation, ElementwiseOp):
            values = tuple(environment[value] for value in operation.values)
            strict_materialization = operation.materialization is not None
            if strict_materialization:
                values = tuple(jax.lax.optimization_barrier(value) for value in values)
            function = operation.function.data
            if function == "add":
                result = values[0] + values[1]
            elif function == "multiply":
                result = values[0] * values[1]
            elif function == "silu":
                result = (
                    _strict_typed_silu(values[0])
                    if strict_materialization
                    else jax.nn.silu(values[0])
                )
            elif function == "silu_multiply":
                result = jax.nn.silu(values[0]) * values[1]
            elif function == "exp":
                result = jnp.exp(values[0])
            else:
                raise UnsupportedInterpretationError(f"unsupported elementwise function {function}")
            result_type = operation.result.type
            assert isinstance(result_type, DTensorType)
            result = _cast(result, result_type)
            if strict_materialization:
                result = jax.lax.optimization_barrier(result)
            if strict_materialization and strict_mlp_checkpoints is not None:
                if function == "silu":
                    strict_mlp_checkpoints.append([values[0], result])
                elif function == "multiply":
                    if not strict_mlp_checkpoints or len(strict_mlp_checkpoints[-1]) != 2:
                        raise UnsupportedInterpretationError(
                            "strict hidden multiply must follow its strict SiLU"
                        )
                    silu_indices = tuple(
                        index
                        for index, operand in enumerate(operation.values)
                        if isinstance(operand.owner, ElementwiseOp)
                        and operand.owner.function.data == "silu"
                        and operand.owner.materialization is not None
                    )
                    if len(silu_indices) != 1:
                        raise UnsupportedInterpretationError(
                            "strict hidden multiply must consume one strict SiLU"
                        )
                    up_index = 1 - silu_indices[0]
                    strict_mlp_checkpoints[-1].extend((values[up_index], result))
        elif isinstance(operation, LayerScanOp):
            captures = tuple(environment[value] for value in operation.captures)
            carries = captures[: operation.carry_count.data]
            stacked_end = operation.carry_count.data + operation.stacked_count.data
            stacked = captures[operation.carry_count.data : stacked_end]
            invariants = captures[stacked_end:]
            layer_dimension = operation.layer_dimension.data
            for layer in range(operation.trip_count.data):
                body_inputs = [*carries]
                for value, captured in zip(
                    stacked,
                    operation.captures[operation.carry_count.data : stacked_end],
                    strict=True,
                ):
                    captured_type = captured.type
                    assert isinstance(captured_type, DTensorType)
                    axis = _names(captured_type).index(layer_dimension)
                    body_inputs.append(jnp.take(value, layer, axis=axis))
                body_inputs.extend(invariants)
                nested_environment = dict(environment)
                nested_environment.update(zip(operation.body.block.args, body_inputs, strict=True))
                yielded = _execute_block(
                    operation.body.block,
                    nested_environment,
                    semantics,
                    strict_mlp_checkpoints,
                )
                if yielded is None:
                    raise UnsupportedInterpretationError("layer scan body did not yield")
                carries = yielded
            for output, value in zip(operation.outputs, carries, strict=True):
                environment[output] = value
            continue
        else:
            raise UnsupportedInterpretationError(
                f"no distributed interpretation for {operation.name}"
            )

        if len(operation.results) != 1:
            raise UnsupportedInterpretationError(
                f"single-result interpretation expected for {operation.name}"
            )
        result_type = operation.results[0].type
        assert isinstance(result_type, DTensorType)
        expected_shape = semantics.shape(result_type)
        if tuple(result.shape) != expected_shape:
            raise UnsupportedInterpretationError(
                f"{operation.name} produced shape {tuple(result.shape)}, expected {expected_shape}"
            )
        environment[operation.results[0]] = _cast(result, result_type)
    return None


def execute_distributed_program_jax(
    module: ModuleOp,
    inputs: Sequence[np.ndarray | jax.Array],
) -> tuple[jax.Array, ...]:
    """Execute global logical tensor semantics with JAX on one device.

    This function does not implement physical sharding or communication. Collective
    operations are identities because every value is represented at its declared
    global logical shape. Callers that need physical collectives must use a physical
    lowering instead.
    """
    module.verify()
    programs = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, ProgramOp)
    )
    if len(programs) != 1 or len(tuple(module.body.block.ops)) != 1:
        raise UnsupportedInterpretationError("interpretation expects one distributed program")
    program = programs[0]
    if len(inputs) != len(program.body.block.args):
        raise ValueError(
            f"distributed program expects {len(program.body.block.args)} inputs, got {len(inputs)}"
        )
    environment: dict[SSAValue, jax.Array] = {}
    for argument, value in zip(program.body.block.args, inputs, strict=True):
        value_type = argument.type
        assert isinstance(value_type, DTensorType)
        actual_dtype = jnp.dtype(value.dtype)
        expected_dtype = jnp.dtype(_dtype(value_type))
        if actual_dtype != expected_dtype:
            raise ValueError(f"input dtype {actual_dtype} does not match {expected_dtype}")
        array = jnp.asarray(value)
        expected_shape = tuple(size for _, size in value_type.logical_shape())
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"input shape {tuple(array.shape)} does not match {expected_shape}")
        environment[argument] = array
    outputs = _execute_block(program.body.block, environment, _LOGICAL_SEMANTICS)
    if outputs is None:
        raise UnsupportedInterpretationError("distributed program did not return")
    return outputs


def execute_distributed_program_jax_sharded(
    module: ModuleOp,
    inputs: Sequence[jax.Array],
) -> tuple[jax.Array, ...]:
    """Execute local shards with physical JAX collectives inside ``shard_map``."""
    module.verify()
    programs = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, ProgramOp)
    )
    if len(programs) != 1 or len(tuple(module.body.block.ops)) != 1:
        raise UnsupportedInterpretationError(
            "sharded interpretation expects one distributed program"
        )
    program = programs[0]
    if len(inputs) != len(program.body.block.args):
        raise ValueError(
            f"distributed program expects {len(program.body.block.args)} inputs, got {len(inputs)}"
        )
    semantics = _ExecutionSemantics(program.mesh.sizes())
    environment: dict[SSAValue, jax.Array] = {}
    for argument, value in zip(program.body.block.args, inputs, strict=True):
        value_type = argument.type
        assert isinstance(value_type, DTensorType)
        actual_dtype = jnp.dtype(value.dtype)
        expected_dtype = jnp.dtype(_dtype(value_type))
        if actual_dtype != expected_dtype:
            raise ValueError(f"input dtype {actual_dtype} does not match {expected_dtype}")
        if tuple(value.shape) != semantics.shape(value_type):
            raise ValueError(
                f"local input shape {tuple(value.shape)} does not match "
                f"{semantics.shape(value_type)}"
            )
        environment[argument] = value
    outputs = _execute_block(program.body.block, environment, semantics)
    if outputs is None:
        raise UnsupportedInterpretationError("distributed program did not return")
    return outputs


def execute_distributed_program_jax_sharded_with_strict_mlp(
    module: ModuleOp,
    inputs: Sequence[jax.Array],
) -> tuple[tuple[jax.Array, ...], tuple[tuple[jax.Array, ...], ...]]:
    module.verify()
    programs = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, ProgramOp)
    )
    if len(programs) != 1 or len(tuple(module.body.block.ops)) != 1:
        raise UnsupportedInterpretationError(
            "sharded interpretation expects one distributed program"
        )
    program = programs[0]
    if len(inputs) != len(program.body.block.args):
        raise ValueError(
            f"distributed program expects {len(program.body.block.args)} inputs, got {len(inputs)}"
        )
    semantics = _ExecutionSemantics(program.mesh.sizes())
    environment: dict[SSAValue, jax.Array] = {}
    for argument, value in zip(program.body.block.args, inputs, strict=True):
        value_type = argument.type
        assert isinstance(value_type, DTensorType)
        actual_dtype = jnp.dtype(value.dtype)
        expected_dtype = jnp.dtype(_dtype(value_type))
        if actual_dtype != expected_dtype:
            raise ValueError(f"input dtype {actual_dtype} does not match {expected_dtype}")
        if tuple(value.shape) != semantics.shape(value_type):
            raise ValueError(
                f"local input shape {tuple(value.shape)} does not match "
                f"{semantics.shape(value_type)}"
            )
        environment[argument] = value
    checkpoints: list[list[jax.Array]] = []
    outputs = _execute_block(
        program.body.block,
        environment,
        semantics,
        checkpoints,
    )
    if outputs is None:
        raise UnsupportedInterpretationError("distributed program did not return")
    if any(len(checkpoint) != 6 for checkpoint in checkpoints):
        raise UnsupportedInterpretationError("strict MLP checkpoint set is incomplete")
    return outputs, tuple(tuple(checkpoint) for checkpoint in checkpoints)


def interpret_distributed_program(
    module: ModuleOp,
    inputs: Sequence[np.ndarray | jax.Array],
) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(value) for value in execute_distributed_program_jax(module, inputs))
