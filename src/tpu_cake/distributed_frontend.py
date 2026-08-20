from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, Float32Type, IntAttr, ModuleOp, StringAttr, f32
from xdsl.ir import Attribute, Block, Region, SSAValue

from tpu_cake.dialects.distributed_tensor import (
    AllGatherOp,
    AllReduceOp,
    AxisListAttr,
    BroadcastOp,
    CastOp,
    DimensionAttr,
    DTensorType,
    EinsumLocalOp,
    EinsumOp,
    ElementwiseMaterialization,
    ElementwiseOp,
    EmbeddingLookupOp,
    LayerScanOp,
    MaskedSoftmaxOp,
    MeshAttr,
    PackedCausalMaskOp,
    PendingReductionsAttr,
    ProgramOp,
    ReduceLocalOp,
    ReduceScatterOp,
    RenameDimensionOp,
    ReturnOp,
    RmsNormApplyOp,
    RmsNormOp,
    RmsNormPartialOp,
    RotaryEmbeddingOp,
    ScanYieldOp,
    ShardingAttr,
    SliceOp,
    TransposeOp,
)
from tpu_cake.source import SourceLocation, attach_source, verify_with_sources


@dataclass(frozen=True)
class DistributedTensorSpec:
    element_type: Attribute
    dimensions: tuple[tuple[str, int], ...]
    sharding: tuple[tuple[str, ...], ...]
    pending_reductions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.dimensions) != len(self.sharding):
            raise ValueError("distributed tensor shape and sharding ranks must match")

    def to_type(self) -> DTensorType:
        return DTensorType(
            self.element_type,
            ArrayAttr(
                DimensionAttr(StringAttr(name), IntAttr(size)) for name, size in self.dimensions
            ),
            ShardingAttr(
                ArrayAttr(
                    AxisListAttr(ArrayAttr(StringAttr(axis) for axis in axes))
                    for axes in self.sharding
                )
            ),
            PendingReductionsAttr(
                ArrayAttr(StringAttr(axis) for axis, _ in self.pending_reductions),
                ArrayAttr(StringAttr(reducer) for _, reducer in self.pending_reductions),
            ),
        )


def tensor(
    element_type: Attribute,
    dimensions: Mapping[str, int] | Iterable[tuple[str, int]],
    *,
    sharding: Mapping[str, Iterable[str]] | None = None,
    pending_reductions: Mapping[str, str] | None = None,
) -> DistributedTensorSpec:
    dimension_items = tuple(dimensions.items() if isinstance(dimensions, Mapping) else dimensions)
    sharding = sharding or {}
    return DistributedTensorSpec(
        element_type=element_type,
        dimensions=dimension_items,
        sharding=tuple(tuple(sharding.get(name, ())) for name, _ in dimension_items),
        pending_reductions=tuple(sorted((pending_reductions or {}).items())),
    )


class DistributedProgramBuilder:
    def __init__(
        self,
        name: str,
        mesh: Mapping[str, int],
        inputs: Iterable[DistributedTensorSpec],
    ) -> None:
        self._name = name
        mesh = dict(sorted(mesh.items()))
        self._mesh = MeshAttr(
            ArrayAttr(StringAttr(axis) for axis in mesh),
            ArrayAttr(IntAttr(size) for size in mesh.values()),
        )
        self.block = Block(arg_types=[value.to_type() for value in inputs])

    @property
    def inputs(self) -> tuple[SSAValue, ...]:
        return tuple(self.block.args)

    def einsum_local(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        result: DistributedTensorSpec,
        *,
        contracting_dimension: str,
        accumulation_type: Float32Type = f32,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            EinsumLocalOp(
                lhs,
                rhs,
                result.to_type(),
                contracting_dimension,
                accumulation_type,
            ),
            source,
        )
        assert isinstance(operation, EinsumLocalOp)
        self.block.add_op(operation)
        return operation.result

    def einsum(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        result: DistributedTensorSpec,
        *,
        contracting_dimensions: tuple[str, ...],
        accumulation_type: Float32Type = f32,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            EinsumOp(
                lhs,
                rhs,
                result.to_type(),
                contracting_dimensions,
                accumulation_type,
            ),
            source,
        )
        assert isinstance(operation, EinsumOp)
        self.block.add_op(operation)
        return operation.result

    def elementwise(
        self,
        *values: SSAValue,
        result: DistributedTensorSpec,
        function: str,
        materialization: ElementwiseMaterialization | None = None,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            ElementwiseOp(
                tuple(values),
                result.to_type(),
                function,
                materialization,
            ),
            source,
        )
        assert isinstance(operation, ElementwiseOp)
        self.block.add_op(operation)
        return operation.result

    def cast(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(CastOp(value, result.to_type()), source)
        assert isinstance(operation, CastOp)
        self.block.add_op(operation)
        return operation.result

    def rms_norm(
        self,
        value: SSAValue,
        scale: SSAValue,
        result: DistributedTensorSpec,
        *,
        dimension: str,
        epsilon: str = "0.000001",
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            RmsNormOp(
                value,
                scale,
                result.to_type(),
                dimension=dimension,
                epsilon=epsilon,
            ),
            source,
        )
        assert isinstance(operation, RmsNormOp)
        self.block.add_op(operation)
        return operation.result

    def rms_norm_partial(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        dimension: str,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            RmsNormPartialOp(value, result.to_type(), dimension=dimension), source
        )
        assert isinstance(operation, RmsNormPartialOp)
        self.block.add_op(operation)
        return operation.result

    def rms_norm_apply(
        self,
        value: SSAValue,
        sum_squares: SSAValue,
        scale: SSAValue,
        result: DistributedTensorSpec,
        *,
        dimension: str,
        normalized_size: int,
        epsilon: str = "0.000001",
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            RmsNormApplyOp(
                value,
                sum_squares,
                scale,
                result.to_type(),
                dimension=dimension,
                normalized_size=normalized_size,
                epsilon=epsilon,
            ),
            source,
        )
        assert isinstance(operation, RmsNormApplyOp)
        self.block.add_op(operation)
        return operation.result

    def rotary_embedding(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        sequence_dimension: str,
        head_dimension: str,
        maximum_timescale: int,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            RotaryEmbeddingOp(
                value,
                result.to_type(),
                sequence_dimension=sequence_dimension,
                head_dimension=head_dimension,
                maximum_timescale=maximum_timescale,
            ),
            source,
        )
        assert isinstance(operation, RotaryEmbeddingOp)
        self.block.add_op(operation)
        return operation.result

    def slice(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        dimension: str,
        index: int,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            SliceOp(value, result.to_type(), dimension=dimension, index=index),
            source,
        )
        assert isinstance(operation, SliceOp)
        self.block.add_op(operation)
        return operation.result

    def rename_dimension(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        source_dimension: str,
        destination_dimension: str,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            RenameDimensionOp(
                value,
                result.to_type(),
                source_dimension=source_dimension,
                destination_dimension=destination_dimension,
            ),
            source,
        )
        assert isinstance(operation, RenameDimensionOp)
        self.block.add_op(operation)
        return operation.result

    def packed_causal_mask(
        self,
        sequence_starts: SSAValue,
        result: DistributedTensorSpec,
        *,
        sequence_dimension: str,
        query_dimension: str,
        key_dimension: str,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            PackedCausalMaskOp(
                sequence_starts,
                result.to_type(),
                sequence_dimension=sequence_dimension,
                query_dimension=query_dimension,
                key_dimension=key_dimension,
            ),
            source,
        )
        assert isinstance(operation, PackedCausalMaskOp)
        self.block.add_op(operation)
        return operation.result

    def masked_softmax(
        self,
        value: SSAValue,
        mask: SSAValue,
        result: DistributedTensorSpec,
        *,
        dimension: str,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            MaskedSoftmaxOp(value, mask, result.to_type(), dimension=dimension),
            source,
        )
        assert isinstance(operation, MaskedSoftmaxOp)
        self.block.add_op(operation)
        return operation.result

    def reduce_local(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        dimensions: tuple[str, ...],
        reducer: str,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            ReduceLocalOp(value, result.to_type(), dimensions, reducer), source
        )
        assert isinstance(operation, ReduceLocalOp)
        self.block.add_op(operation)
        return operation.result

    def transpose(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        permutation: tuple[int, ...],
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(TransposeOp(value, result.to_type(), permutation), source)
        assert isinstance(operation, TransposeOp)
        self.block.add_op(operation)
        return operation.result

    def broadcast(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(BroadcastOp(value, result.to_type()), source)
        assert isinstance(operation, BroadcastOp)
        self.block.add_op(operation)
        return operation.result

    def embedding_lookup(
        self,
        table: SSAValue,
        indices: SSAValue,
        result: DistributedTensorSpec,
        *,
        vocabulary_dimension: str,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            EmbeddingLookupOp(
                table,
                indices,
                result.to_type(),
                vocabulary_dimension,
            ),
            source,
        )
        assert isinstance(operation, EmbeddingLookupOp)
        self.block.add_op(operation)
        return operation.result

    def all_gather(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(AllGatherOp(value, result.to_type()), source)
        assert isinstance(operation, AllGatherOp)
        self.block.add_op(operation)
        return operation.result

    def reduce_scatter(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        axes: tuple[str, ...],
        scatter_dimensions: tuple[str, ...],
        reducer: str = "sum",
        source: SourceLocation | None = None,
    ) -> SSAValue:
        pairs = tuple(sorted(zip(axes, scatter_dimensions, strict=True)))
        operation = attach_source(
            ReduceScatterOp(
                value,
                result.to_type(),
                tuple(axis for axis, _ in pairs),
                tuple(dimension for _, dimension in pairs),
                reducer,
            ),
            source,
        )
        assert isinstance(operation, ReduceScatterOp)
        self.block.add_op(operation)
        return operation.result

    def all_reduce(
        self,
        value: SSAValue,
        result: DistributedTensorSpec,
        *,
        axes: tuple[str, ...],
        reducer: str = "sum",
        source: SourceLocation | None = None,
    ) -> SSAValue:
        operation = attach_source(
            AllReduceOp(value, result.to_type(), tuple(sorted(axes)), reducer), source
        )
        assert isinstance(operation, AllReduceOp)
        self.block.add_op(operation)
        return operation.result

    def layer_scan(
        self,
        captures: tuple[SSAValue, ...],
        body_builder: Callable[
            [DistributedProgramBuilder, tuple[SSAValue, ...]], tuple[SSAValue, ...]
        ],
        *,
        carry_count: int,
        stacked_count: int,
        layer_dimension: str,
        trip_count: int,
        source: SourceLocation | None = None,
    ) -> tuple[SSAValue, ...]:
        body_types: list[Attribute] = []
        for index, capture in enumerate(captures):
            capture_type = capture.type
            assert isinstance(capture_type, DTensorType)
            if carry_count <= index < carry_count + stacked_count:
                dimensions = tuple(
                    (name, size)
                    for name, size in capture_type.logical_shape()
                    if name != layer_dimension
                )
                sharding = tuple(
                    axes
                    for (name, _), axes in zip(
                        capture_type.logical_shape(),
                        capture_type.sharding_axes(),
                        strict=True,
                    )
                    if name != layer_dimension
                )
                capture_type = DistributedTensorSpec(
                    element_type=capture_type.element_type,
                    dimensions=dimensions,
                    sharding=sharding,
                    pending_reductions=tuple(capture_type.pending_reductions().items()),
                ).to_type()
            body_types.append(capture_type)
        body = Block(arg_types=body_types)
        nested = object.__new__(DistributedProgramBuilder)
        nested._name = self._name
        nested._mesh = self._mesh
        nested.block = body
        yielded = tuple(body_builder(nested, tuple(body.args)))
        body.add_op(ScanYieldOp(*yielded))
        operation = attach_source(
            LayerScanOp(
                captures,
                Region(body),
                carry_count=carry_count,
                stacked_count=stacked_count,
                layer_dimension=layer_dimension,
                trip_count=trip_count,
            ),
            source,
        )
        assert isinstance(operation, LayerScanOp)
        self.block.add_op(operation)
        return tuple(operation.outputs)

    def module(self, *results: SSAValue) -> ModuleOp:
        self.block.add_op(ReturnOp(*results))
        module = ModuleOp([ProgramOp(self._name, self._mesh, Region(self.block))])
        verify_with_sources(module)
        return module
