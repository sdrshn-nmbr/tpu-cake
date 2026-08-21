from __future__ import annotations

import math
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, IntAttr, ModuleOp
from xdsl.ir import Block, Operation, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter

from tpu_cake.dialects.distributed_tensor import (
    AllGatherOp,
    AllReduceOp,
    CastOp,
    DistributedTensor,
    DTensorType,
    EinsumLocalOp,
    EinsumOp,
    ElementwiseMaterialization,
    ElementwiseOp,
    EmbeddingLookupOp,
    LayerScanOp,
    MaskedSoftmaxOp,
    PackedCausalMaskOp,
    ProgramOp,
    ReduceScatterOp,
    RenameDimensionOp,
    ResidualAllReduceOp,
    ReturnOp,
    RmsNormApplyOp,
    RmsNormOp,
    RmsNormPartialOp,
    RotaryEmbeddingOp,
    ScanYieldOp,
    SliceOp,
)
from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    CollectiveKind,
    DmaStartOp,
    DmaWaitOp,
    KernelOp,
    LifetimeAttr,
    MemorySpace,
    Ownership,
    VectorImplementation,
    VectorMaterialization,
)
from tpu_cake.frontend import (
    BufferSpec,
    KernelBuilder,
    buffer,
    canonical_module_text,
    schedule_sha256,
)
from tpu_cake.lowering import TPU7X_TARGET, LoweringTarget, UnsupportedLoweringError
from tpu_cake.source import verify_with_sources

SEQAX_PHYSICAL_SCHEMA = "seqax-unrolled-physical-v1"
_PROVISIONAL_LIFETIME_END = 1 << 30


@dataclass(frozen=True)
class SeqaxPhysicalLoweringResult:
    module: ModuleOp
    distributed_schedule_sha256: str
    physical_schedule_sha256: str
    operation_count: int
    unrolled_layer_count: int


def _reject(operation: Operation, message: str) -> UnsupportedLoweringError:
    return UnsupportedLoweringError(f"{message}: {operation.name} at {operation.location}")


def _program(module: ModuleOp) -> ProgramOp:
    module.verify()
    operations = tuple(module.body.block.ops)
    if len(operations) != 1 or not isinstance(operations[0], ProgramOp):
        operation = operations[0] if operations else module
        raise _reject(operation, "Seqax physical lowering expects one distributed program")
    return operations[0]


def _local_shape(value_type: DTensorType, mesh: dict[str, int]) -> tuple[int, ...]:
    result: list[int] = []
    for (_, size), axes in zip(
        value_type.logical_shape(),
        value_type.sharding_axes(),
        strict=True,
    ):
        divisor = math.prod(mesh[axis] for axis in axes)
        if size % divisor:
            raise UnsupportedLoweringError(
                f"logical extent {size} does not divide by mesh extent {divisor}"
            )
        result.append(size // divisor)
    return tuple(result)


def _buffer_spec(
    value_type: DTensorType,
    mesh: dict[str, int],
    *,
    memory: MemorySpace,
    ownership: Ownership,
) -> BufferSpec:
    return buffer(
        _local_shape(value_type, mesh),
        tuple(name for name, _ in value_type.logical_shape()),
        value_type.element_type,
        memory=memory,
        sharding=tuple("/".join(axes) for axes in value_type.sharding_axes()),
        ownership=ownership,
        lifetime=(0, _PROVISIONAL_LIFETIME_END),
    )


def _configuration(**values: object) -> tuple[str, ...]:
    return tuple(f"{key}={values[key]}" for key in sorted(values))


class _LoweringState:
    def __init__(
        self,
        builder: KernelBuilder,
        mesh: dict[str, int],
        environment: dict[SSAValue, SSAValue],
        *,
        stage: int,
        einsum_tiles: tuple[tuple[int, int, int], ...] | None,
    ) -> None:
        self.builder = builder
        self.mesh = mesh
        self.environment = environment
        self.stage = stage
        self.operation_count = 0
        self.unrolled_layer_count = 0
        self._role_index = 0
        self._einsum_tiles = einsum_tiles
        self._einsum_index = 0

    def require_all_einsum_tiles_consumed(self) -> None:
        if self._einsum_tiles is not None and self._einsum_index != len(self._einsum_tiles):
            raise UnsupportedLoweringError(
                "Seqax physical lowering received unused MXU einsum tile declarations"
            )

    def _next_role(self, operation: Operation) -> str:
        self._role_index += 1
        return f"{self._role_index:03d}_{operation.name.replace('.', '_')}"

    def allocate(self, value_type: DTensorType, operation: Operation):
        allocation = self.builder.alloc(
            _buffer_spec(
                value_type,
                self.mesh,
                memory=MemorySpace.VMEM,
                ownership=Ownership.KERNEL,
            ),
            self._next_role(operation),
        )
        allocation.location = operation.location
        return allocation.buffer

    def allocate_spec(self, spec: BufferSpec, operation: Operation):
        allocation = self.builder.alloc(spec, self._next_role(operation))
        allocation.location = operation.location
        return allocation.buffer

    def vector(
        self,
        operation: Operation,
        inputs: tuple[SSAValue, ...],
        output_type: DTensorType,
        function: str,
        configuration: tuple[str, ...] = (),
        materialization: VectorMaterialization | None = None,
        implementation: VectorImplementation | None = None,
    ) -> SSAValue:
        output = self.allocate(output_type, operation)
        scheduled = self.builder.vector_compute(
            inputs,
            output,
            stage=self.stage,
            function=function,
            configuration=configuration,
            pending_reduction_axes=tuple(output_type.pending_reductions()),
            materialization=materialization,
            implementation=implementation,
        )
        scheduled.location = operation.location
        self.stage += 1
        self.operation_count += 1
        return output

    def collective(self, operation: Operation) -> SSAValue:
        before = operation.value.type
        after = operation.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        source = self.environment[operation.value]
        if isinstance(operation, AllGatherOp):
            changes = [
                (index, old_axes, new_axes)
                for index, (old_axes, new_axes) in enumerate(
                    zip(before.sharding_axes(), after.sharding_axes(), strict=True)
                )
                if old_axes != new_axes
            ]
            if len(changes) != 1:
                raise _reject(operation, "physical all-gather supports one changed dimension")
            dimension, old_axes, new_axes = changes[0]
            removed = tuple(reversed(tuple(axis for axis in old_axes if axis not in new_axes)))
            if not removed or tuple(axis for axis in old_axes if axis in new_axes) != new_axes:
                raise _reject(operation, "physical all-gather can only remove mesh axes")
            current = source
            current_shape = list(current.type.storage.get_shape())
            current_sharding = [list(axes) for axes in before.sharding_axes()]
            for offset, axis in enumerate(removed):
                current_shape[dimension] *= self.mesh[axis]
                current_sharding[dimension].remove(axis)
                if offset == len(removed) - 1:
                    destination = self.allocate(after, operation)
                else:
                    destination = self.allocate_spec(
                        buffer(
                            current_shape,
                            tuple(name for name, _ in before.logical_shape()),
                            before.element_type,
                            memory=MemorySpace.VMEM,
                            sharding=tuple("/".join(axes) for axes in current_sharding),
                            ownership=Ownership.KERNEL,
                            lifetime=(0, _PROVISIONAL_LIFETIME_END),
                        ),
                        operation,
                    )
                scheduled = self.builder.collective(
                    current,
                    destination,
                    stage=self.stage,
                    kind=CollectiveKind.ALL_GATHER,
                    mesh_axis=axis,
                    group_size=self.mesh[axis],
                    concat_dimension=dimension,
                )
                scheduled.location = operation.location
                self.stage += 1
                self.operation_count += 1
                current = destination
            return current
        elif isinstance(operation, ReduceScatterOp):
            axes = tuple(value.data for value in operation.axes)
            dimensions = tuple(value.data for value in operation.scatter_dimensions)
            if len(axes) != 1 or len(dimensions) != 1:
                raise _reject(operation, "physical reduce-scatter supports one mesh axis")
            axis = axes[0]
            kind = CollectiveKind.REDUCE_SCATTER
            split_dimension = next(
                index
                for index, (name, _) in enumerate(before.logical_shape())
                if name == dimensions[0]
            )
            concat_dimension = -1
            reducer = operation.reducer.data
        elif isinstance(operation, AllReduceOp):
            axes = tuple(value.data for value in operation.axes)
            if not axes:
                raise _reject(operation, "physical all-reduce needs a mesh axis")
            current = source
            for offset, axis in enumerate(axes):
                destination = (
                    self.allocate(after, operation)
                    if offset == len(axes) - 1
                    else self.allocate(before, operation)
                )
                scheduled = self.builder.collective(
                    current,
                    destination,
                    stage=self.stage,
                    kind=CollectiveKind.ALL_REDUCE,
                    mesh_axis=axis,
                    group_size=self.mesh[axis],
                    reducer=operation.reducer.data,
                )
                scheduled.location = operation.location
                self.stage += 1
                self.operation_count += 1
                current = destination
            return current
        else:
            raise _reject(operation, "unsupported physical collective")
        destination = self.allocate(after, operation)
        scheduled = self.builder.collective(
            source,
            destination,
            stage=self.stage,
            kind=kind,
            mesh_axis=axis,
            group_size=self.mesh[axis],
            split_dimension=split_dimension,
            concat_dimension=concat_dimension,
            reducer=reducer,
        )
        scheduled.location = operation.location
        self.stage += 1
        self.operation_count += 1
        return destination

    def einsum(self, operation: EinsumOp | EinsumLocalOp) -> SSAValue:
        lhs_type = operation.lhs.type
        rhs_type = operation.rhs.type
        result_type = operation.result.type
        assert isinstance(lhs_type, DTensorType)
        assert isinstance(rhs_type, DTensorType)
        assert isinstance(result_type, DTensorType)
        lhs_names = dict(lhs_type.logical_shape())
        rhs_names = dict(rhs_type.logical_shape())
        contractions = tuple(value.data for value in operation.contracting_dimensions)
        shared = (set(lhs_names) & set(rhs_names)) - set(contractions)
        m = math.prod(
            size
            for name, size in zip(
                (name for name, _ in lhs_type.logical_shape()),
                _local_shape(lhs_type, self.mesh),
                strict=True,
            )
            if name not in shared and name not in contractions
        )
        k = math.prod(
            size
            for (name, _), size in zip(
                lhs_type.logical_shape(),
                _local_shape(lhs_type, self.mesh),
                strict=True,
            )
            if name in contractions
        )
        n = math.prod(
            size
            for (name, _), size in zip(
                rhs_type.logical_shape(),
                _local_shape(rhs_type, self.mesh),
                strict=True,
            )
            if name not in shared and name not in contractions
        )
        tile = (m, k, n)
        if self._einsum_tiles is not None:
            if self._einsum_index >= len(self._einsum_tiles):
                raise _reject(operation, "Seqax physical lowering is missing an MXU einsum tile")
            tile = self._einsum_tiles[self._einsum_index]
        self._einsum_index += 1
        output = self.allocate(result_type, operation)
        scheduled = self.builder.einsum(
            self.environment[operation.lhs],
            self.environment[operation.rhs],
            output,
            stage=self.stage,
            contracting_dimensions=contractions,
            pending_reduction_axes=tuple(result_type.pending_reductions()),
            tile_m=tile[0],
            tile_k=tile[1],
            tile_n=tile[2],
        )
        scheduled.location = operation.location
        self.stage += 1
        self.operation_count += 1
        return output

    def residual_all_reduce(self, operation: ResidualAllReduceOp) -> None:
        partial_type = operation.partial.type
        full_result_type = operation.full_result.type
        shard_result_type = operation.shard_result.type
        assert isinstance(partial_type, DTensorType)
        assert isinstance(full_result_type, DTensorType)
        assert isinstance(shard_result_type, DTensorType)
        axis = operation.mesh_axis.data
        configuration = _configuration(
            dimension=operation.dimension.data,
            group_size=self.mesh[axis],
            mesh_axis=axis,
        )
        contribution = self.vector(
            operation,
            (self.environment[operation.partial], self.environment[operation.residual]),
            partial_type,
            "residual_inject",
            configuration,
        )
        reduced = self.allocate(partial_type, operation)
        collective = self.builder.collective(
            contribution,
            reduced,
            stage=self.stage,
            kind=CollectiveKind.ALL_REDUCE,
            mesh_axis=axis,
            group_size=self.mesh[axis],
            reducer="sum",
        )
        collective.location = operation.location
        self.stage += 1
        self.operation_count += 1
        full_result = self.vector(
            operation,
            (reduced,),
            full_result_type,
            "cast",
            _configuration(dtype=str(full_result_type.element_type)),
        )
        shard_result = self.vector(
            operation,
            (full_result,),
            shard_result_type,
            "shard_extract",
            configuration,
        )
        self.environment[operation.full_result] = full_result
        self.environment[operation.shard_result] = shard_result

    def lower_operation(self, operation: Operation) -> None:
        if isinstance(operation, (ReturnOp, ScanYieldOp)):
            return
        if isinstance(operation, (AllGatherOp, ReduceScatterOp, AllReduceOp)):
            self.environment[operation.result] = self.collective(operation)
            return
        if isinstance(operation, ResidualAllReduceOp):
            self.residual_all_reduce(operation)
            return
        if isinstance(operation, (EinsumOp, EinsumLocalOp)):
            self.environment[operation.result] = self.einsum(operation)
            return
        if isinstance(operation, LayerScanOp):
            self.lower_scan(operation)
            return

        result = operation.results[0].type if len(operation.results) == 1 else None
        if not isinstance(result, DTensorType):
            raise _reject(operation, "physical vector lowering expects one tensor result")
        inputs = tuple(self.environment[value] for value in operation.operands)
        if isinstance(operation, CastOp):
            function = "cast"
            configuration = _configuration(dtype=str(result.element_type))
        elif isinstance(operation, RmsNormOp):
            function = "rms_norm"
            configuration = _configuration(
                dimension=operation.dimension.data,
                epsilon=operation.epsilon.data,
            )
        elif isinstance(operation, RmsNormPartialOp):
            function = "rms_norm_partial"
            configuration = _configuration(dimension=operation.dimension.data)
        elif isinstance(operation, RmsNormApplyOp):
            function = "rms_norm_apply"
            configuration = _configuration(
                dimension=operation.dimension.data,
                epsilon=operation.epsilon.data,
                normalized_size=operation.normalized_size.data,
            )
        elif isinstance(operation, RotaryEmbeddingOp):
            function = "rotary_embedding"
            configuration = _configuration(
                head_dimension=operation.head_dimension.data,
                maximum_timescale=operation.maximum_timescale.data,
                sequence_dimension=operation.sequence_dimension.data,
            )
        elif isinstance(operation, SliceOp):
            function = "slice"
            configuration = _configuration(
                dimension=operation.dimension.data,
                index=operation.index.data,
            )
        elif isinstance(operation, RenameDimensionOp):
            function = "rename_dimension"
            configuration = _configuration(
                destination=operation.destination_dimension.data,
                source=operation.source_dimension.data,
            )
        elif isinstance(operation, PackedCausalMaskOp):
            function = "packed_causal_mask"
            configuration = _configuration(
                key_dimension=operation.key_dimension.data,
                query_dimension=operation.query_dimension.data,
                sequence_dimension=operation.sequence_dimension.data,
            )
        elif isinstance(operation, MaskedSoftmaxOp):
            function = "masked_softmax"
            configuration = _configuration(dimension=operation.dimension.data)
        elif isinstance(operation, EmbeddingLookupOp):
            function = "embedding_lookup"
            configuration = _configuration(vocabulary_dimension=operation.vocabulary_dimension.data)
        elif isinstance(operation, ElementwiseOp):
            function = operation.function.data
            configuration = ()
        else:
            raise _reject(operation, "no physical Seqax lowering for operation")
        self.environment[operation.results[0]] = self.vector(
            operation,
            inputs,
            result,
            function,
            configuration,
            materialization=(
                VectorMaterialization.STRICT_TYPED
                if isinstance(operation, ElementwiseOp)
                and operation.materialization is not None
                and operation.materialization.data is ElementwiseMaterialization.STRICT_TYPED
                else None
            ),
            implementation=(
                VectorImplementation.PALLAS_FULL_LOCAL
                if isinstance(operation, ElementwiseOp)
                and operation.function.data == "silu_multiply"
                else None
            ),
        )

    def lower_block(self, block: Block) -> tuple[SSAValue, ...]:
        for operation in block.ops:
            self.lower_operation(operation)
            if isinstance(operation, (ReturnOp, ScanYieldOp)):
                return tuple(self.environment[value] for value in operation.values)
        raise UnsupportedLoweringError("physical Seqax block has no terminator")

    def lower_scan(self, operation: LayerScanOp) -> None:
        captures = tuple(self.environment[value] for value in operation.captures)
        carries = captures[: operation.carry_count.data]
        stacked_end = operation.carry_count.data + operation.stacked_count.data
        stacked = captures[operation.carry_count.data : stacked_end]
        invariants = captures[stacked_end:]
        captured_values = tuple(operation.captures)
        for layer in range(operation.trip_count.data):
            self.unrolled_layer_count += 1
            body_inputs: list[SSAValue] = [*carries]
            for value, captured in zip(
                stacked,
                captured_values[operation.carry_count.data : stacked_end],
                strict=True,
            ):
                captured_type = captured.type
                assert isinstance(captured_type, DTensorType)
                body_argument = operation.body.block.args[len(body_inputs)]
                body_type = body_argument.type
                assert isinstance(body_type, DTensorType)
                sliced = self.vector(
                    operation,
                    (value,),
                    body_type,
                    "slice",
                    _configuration(
                        dimension=operation.layer_dimension.data,
                        index=layer,
                    ),
                )
                body_inputs.append(sliced)
            body_inputs.extend(invariants)
            nested = dict(self.environment)
            nested.update(zip(operation.body.block.args, body_inputs, strict=True))
            prior = self.environment
            self.environment = nested
            carries = self.lower_block(operation.body.block)
            self.environment = prior
        for output, value in zip(operation.outputs, carries, strict=True):
            self.environment[output] = value


def _replace_lifetime(value: SSAValue, start: int, end: int) -> None:
    value_type = value.type
    assert isinstance(value_type, BufferType)
    replacement = BufferType(
        value_type.storage,
        value_type.shape,
        value_type.space,
        value_type.sharding,
        value_type.layout,
        value_type.ownership,
        LifetimeAttr(IntAttr(start), IntAttr(end)),
    )
    Rewriter.replace_value_with_new_type(value, replacement)


def _derive_exact_lifetimes(module: ModuleOp) -> None:
    kernels = tuple(
        operation for operation in module.body.block.ops if isinstance(operation, KernelOp)
    )
    if len(kernels) != 1:
        raise UnsupportedLoweringError("Seqax lifetime derivation expects one TPU kernel")
    block = kernels[0].body.block
    values = [
        *block.args,
        *(operation.buffer for operation in block.ops if isinstance(operation, AllocOp)),
    ]
    for value in values:
        stages: list[int] = []
        for use in value.uses:
            operation = use.operation
            stage = getattr(operation, "stage", None)
            if isinstance(stage, IntAttr):
                stages.append(stage.data)
            if isinstance(operation, DmaStartOp):
                for token_use in operation.token.uses:
                    if isinstance(token_use.operation, DmaWaitOp):
                        stages.append(token_use.operation.stage.data)
        if not stages:
            raise UnsupportedLoweringError(
                "Seqax physical buffer has no scheduled producer or consumer"
            )
        _replace_lifetime(value, min(stages), max(stages))


def lower_seqax_forward_to_physical(
    module: ModuleOp,
    *,
    target: LoweringTarget = TPU7X_TARGET,
    einsum_tiles: tuple[tuple[int, int, int], ...] | None = None,
) -> SeqaxPhysicalLoweringResult:
    distributed_schedule_hash = schedule_sha256(module)
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(DistributedTensor)
    canonical_module = Parser(context, canonical_module_text(module)).parse_module()
    program = _program(canonical_module)
    mesh = program.mesh.sizes()
    terminator = program.body.block.last_op
    if not isinstance(terminator, ReturnOp) or not terminator.values:
        raise _reject(program, "Seqax physical lowering requires returned tensors")
    argument_types = tuple(argument.type for argument in program.body.block.args)
    output_types = tuple(value.type for value in terminator.values)
    if not all(isinstance(value, DTensorType) for value in (*argument_types, *output_types)):
        raise _reject(program, "Seqax physical lowering requires distributed tensor ABI values")
    external = tuple(
        _buffer_spec(
            value,
            mesh,
            memory=MemorySpace.HBM,
            ownership=Ownership.EXTERNAL,
        )
        for value in (*argument_types, *output_types)
        if isinstance(value, DTensorType)
    )
    builder = KernelBuilder(
        f"{program.sym_name.data}_physical",
        target.name,
        external,
        vmem_capacity_bytes=target.vmem_capacity_bytes,
        smem_capacity_bytes=target.smem_capacity_bytes,
        mesh=mesh,
        interconnect_bandwidth_bytes_per_second={
            axis: target.ici_bandwidth_bytes_per_second for axis in mesh
        },
        argument_modes=(
            *("input" for _ in argument_types),
            *("output" for _ in output_types),
        ),
    )
    environment: dict[SSAValue, SSAValue] = {}
    input_count = len(argument_types)
    stage = 0
    for offset in range(0, input_count, 2):
        pending = []
        for index in range(offset, min(offset + 2, input_count)):
            value_type = argument_types[index]
            assert isinstance(value_type, DTensorType)
            local = builder.alloc(
                _buffer_spec(
                    value_type,
                    mesh,
                    memory=MemorySpace.VMEM,
                    ownership=Ownership.KERNEL,
                ),
                f"input_{index:02d}",
            )
            semaphore = builder.semaphore()
            pending.append(
                builder.dma_start(
                    builder.inputs[index],
                    local,
                    semaphore,
                    stage=stage,
                )
            )
            environment[program.body.block.args[index]] = local.buffer
        for token in pending:
            builder.dma_wait(token, stage=stage + 1)
        stage += 2

    state = _LoweringState(
        builder,
        mesh,
        environment,
        stage=stage,
        einsum_tiles=einsum_tiles,
    )
    outputs = state.lower_block(program.body.block)
    state.require_all_einsum_tiles_consumed()
    output_arguments = builder.inputs[input_count:]
    pending_outputs = []
    for source, destination in zip(outputs, output_arguments, strict=True):
        semaphore = builder.semaphore()
        pending_outputs.append(
            builder.dma_start(
                source,
                destination,
                semaphore,
                stage=state.stage,
            )
        )
    for token in pending_outputs:
        builder.dma_wait(token, stage=state.stage + 1)
    physical = builder.module(verify=False)
    _derive_exact_lifetimes(physical)
    verify_with_sources(physical)
    return SeqaxPhysicalLoweringResult(
        module=physical,
        distributed_schedule_sha256=distributed_schedule_hash,
        physical_schedule_sha256=schedule_sha256(physical),
        operation_count=state.operation_count,
        unrolled_layer_count=state.unrolled_layer_count,
    )
