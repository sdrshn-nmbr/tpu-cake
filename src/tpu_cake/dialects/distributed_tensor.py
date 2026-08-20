from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    Float16Type,
    Float32Type,
    IntAttr,
    IntegerType,
    StringAttr,
    f32,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator
from xdsl.utils.exceptions import VerifyException


class ElementwiseMaterialization(StrEnum):
    STRICT_TYPED = "strict_typed"


@irdl_attr_definition
class ElementwiseMaterializationAttr(
    EnumAttribute[ElementwiseMaterialization], SpacedOpaqueSyntaxAttribute
):
    name = "dtensor.elementwise_materialization"


@irdl_attr_definition
class DimensionAttr(ParametrizedAttribute):
    name = "dtensor.dimension"
    symbol: StringAttr
    size: IntAttr

    def verify(self) -> None:
        if not self.symbol.data or self.size.data <= 0:
            raise VerifyException("distributed dimensions need a name and positive size")


@irdl_attr_definition
class AxisListAttr(ParametrizedAttribute):
    name = "dtensor.axis_list"
    axes: ArrayAttr[StringAttr]


@irdl_attr_definition
class MeshAttr(ParametrizedAttribute):
    name = "dtensor.mesh"
    axis_names: ArrayAttr[StringAttr]
    axis_sizes: ArrayAttr[IntAttr]

    def verify(self) -> None:
        names = [value.data for value in self.axis_names]
        sizes = [value.data for value in self.axis_sizes]
        if len(names) != len(sizes):
            raise VerifyException("mesh axis names and sizes must have equal length")
        if len(names) != len(set(names)) or any(not name for name in names):
            raise VerifyException("mesh axis names must be non-empty and unique")
        if names != sorted(names):
            raise VerifyException("mesh axes must be in canonical lexical order")
        if any(size <= 0 for size in sizes):
            raise VerifyException("mesh axis sizes must be positive")

    def sizes(self) -> dict[str, int]:
        return {
            name.data: size.data
            for name, size in zip(self.axis_names, self.axis_sizes, strict=True)
        }


@irdl_attr_definition
class ShardingAttr(ParametrizedAttribute):
    name = "dtensor.sharding"
    dimensions: ArrayAttr[AxisListAttr]


@irdl_attr_definition
class PendingReductionsAttr(ParametrizedAttribute):
    name = "dtensor.pending_reductions"
    axes: ArrayAttr[StringAttr]
    reducers: ArrayAttr[StringAttr]

    def verify(self) -> None:
        axes = [value.data for value in self.axes]
        reducers = [value.data for value in self.reducers]
        if len(axes) != len(reducers):
            raise VerifyException("pending reduction axes and reducers must have equal length")
        if len(axes) != len(set(axes)):
            raise VerifyException("a mesh axis can have only one pending reduction")
        if axes != sorted(axes):
            raise VerifyException("pending reduction axes must be in canonical lexical order")
        if any(reducer not in {"sum", "max", "min"} for reducer in reducers):
            raise VerifyException("unsupported pending reduction")


@irdl_attr_definition
class DTensorType(ParametrizedAttribute, TypeAttribute):
    name = "dtensor.tensor"
    element_type: TypeAttribute
    dimensions: ArrayAttr[DimensionAttr]
    sharding: ShardingAttr
    pending: PendingReductionsAttr

    def verify(self) -> None:
        if len(self.dimensions) != len(self.sharding.dimensions):
            raise VerifyException("distributed tensor shape and sharding ranks must match")
        names = [dimension.symbol.data for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise VerifyException("logical dimension names must be unique within a tensor")
        for dimension in self.dimensions:
            dimension.verify()
        self.pending.verify()
        sharding_axes = {axis for dimension in self.sharding_axes() for axis in dimension}
        overlap = sharding_axes & set(self.pending_reductions())
        if overlap:
            raise VerifyException(
                "pending reduction axes cannot also shard retained tensor dimensions"
            )

    def logical_shape(self) -> tuple[tuple[str, int], ...]:
        return tuple((value.symbol.data, value.size.data) for value in self.dimensions)

    def sharding_axes(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(axis.data for axis in dimension.axes) for dimension in self.sharding.dimensions
        )

    def pending_reductions(self) -> dict[str, str]:
        return {
            axis.data: reducer.data
            for axis, reducer in zip(self.pending.axes, self.pending.reducers, strict=True)
        }


def _same_value_shape(lhs: DTensorType, rhs: DTensorType) -> bool:
    return lhs.element_type == rhs.element_type and lhs.logical_shape() == rhs.logical_shape()


def _string_array(values: tuple[str, ...] | list[str]) -> ArrayAttr[StringAttr]:
    return ArrayAttr(StringAttr(value) for value in values)


def _dimension_index(tensor: DTensorType) -> dict[str, int]:
    return {name: index for index, (name, _) in enumerate(tensor.logical_shape())}


def _require_fully_reduced(*tensors: DTensorType) -> None:
    if any(tensor.pending_reductions() for tensor in tensors):
        raise VerifyException("operation cannot consume a partially reduced tensor")


def _is_float_type(value: Attribute) -> bool:
    return isinstance(value, (BFloat16Type, Float16Type, Float32Type))


def _is_boolean_type(value: Attribute) -> bool:
    return isinstance(value, IntegerType) and value.width.data == 1


@irdl_op_definition
class ElementwiseOp(IRDLOperation):
    name = "dtensor.elementwise"
    values = var_operand_def(DTensorType)
    result = result_def(DTensorType)
    function = prop_def(StringAttr)
    materialization = opt_prop_def(ElementwiseMaterializationAttr)

    def __init__(
        self,
        values: tuple[SSAValue | IRDLOperation, ...],
        result_type: DTensorType,
        function: str,
        materialization: ElementwiseMaterialization | None = None,
    ) -> None:
        properties: dict[str, Attribute] = {"function": StringAttr(function)}
        if materialization is not None:
            properties["materialization"] = ElementwiseMaterializationAttr(materialization)
        super().__init__(
            operands=[list(values)],
            result_types=[result_type],
            properties=properties,
        )

    def verify_(self) -> None:
        if not self.values:
            raise VerifyException("elementwise operation needs at least one input")
        result = self.result.type
        assert isinstance(result, DTensorType)
        if self.function.data not in {
            "add",
            "multiply",
            "silu",
            "silu_multiply",
            "gelu",
            "relu",
            "exp",
        }:
            raise VerifyException("unsupported elementwise function")
        unary = {"silu", "gelu", "relu", "exp"}
        expected_arity = 1 if self.function.data in unary else 2
        if len(self.values) != expected_arity:
            raise VerifyException(
                f"elementwise {self.function.data} requires {expected_arity} inputs"
            )
        for value in self.values:
            tensor = value.type
            assert isinstance(tensor, DTensorType)
            if tensor != result:
                raise VerifyException(
                    "elementwise inputs and result must have identical distributed tensor types"
                )
        if self.function.data != "add":
            _require_fully_reduced(result)
        if self.function.data in {*unary, "silu_multiply"} and not isinstance(
            result.element_type, (BFloat16Type, Float16Type, Float32Type)
        ):
            raise VerifyException("nonlinear elementwise functions require floating-point values")
        if self.materialization is not None:
            if self.function.data not in {"silu", "multiply"}:
                raise VerifyException(
                    "strict typed materialization is only supported for SiLU and multiply"
                )
            if not isinstance(result.element_type, BFloat16Type):
                raise VerifyException("strict typed materialization requires BF16")


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "dtensor.cast"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)

    def __init__(self, value: SSAValue | IRDLOperation, result_type: DTensorType) -> None:
        super().__init__(operands=[value], result_types=[result_type])

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        _require_fully_reduced(before)
        if before.logical_shape() != after.logical_shape():
            raise VerifyException("cast cannot change logical shape")
        if before.sharding_axes() != after.sharding_axes():
            raise VerifyException("cast cannot change sharding")
        if after.pending_reductions():
            raise VerifyException("cast cannot introduce pending reductions")
        supported = _is_float_type(before.element_type) or isinstance(
            before.element_type, IntegerType
        )
        supported = supported and (
            _is_float_type(after.element_type) or isinstance(after.element_type, IntegerType)
        )
        if not supported or before.element_type == after.element_type:
            raise VerifyException("cast needs distinct supported numeric types")


@irdl_op_definition
class RmsNormOp(IRDLOperation):
    name = "dtensor.rms_norm"
    value = operand_def(DTensorType)
    scale = operand_def(DTensorType)
    result = result_def(DTensorType)
    dimension = prop_def(StringAttr)
    epsilon = prop_def(StringAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        scale: SSAValue | IRDLOperation,
        result_type: DTensorType,
        *,
        dimension: str,
        epsilon: str = "0.000001",
    ) -> None:
        super().__init__(
            operands=[value, scale],
            result_types=[result_type],
            properties={
                "dimension": StringAttr(dimension),
                "epsilon": StringAttr(epsilon),
            },
        )

    def verify_(self) -> None:
        value, scale, result = self.value.type, self.scale.type, self.result.type
        assert isinstance(value, DTensorType)
        assert isinstance(scale, DTensorType)
        assert isinstance(result, DTensorType)
        _require_fully_reduced(value, scale, result)
        if not all(_is_float_type(tensor.element_type) for tensor in (value, scale, result)):
            raise VerifyException("RMSNorm requires floating-point tensors")
        if value.logical_shape() != result.logical_shape() or (
            value.sharding_axes() != result.sharding_axes()
        ):
            raise VerifyException("RMSNorm result must preserve value shape and sharding")
        dimension = self.dimension.data
        value_shape = dict(value.logical_shape())
        if scale.logical_shape() != ((dimension, value_shape.get(dimension, -1)),):
            raise VerifyException("RMSNorm scale must match exactly one normalized dimension")
        if dimension in value_shape and value.sharding_axes()[_dimension_index(value)[dimension]]:
            raise VerifyException("RMSNorm normalized dimension cannot be sharded")
        if scale.sharding_axes() != ((),):
            raise VerifyException("RMSNorm scale must be locally replicated")
        try:
            epsilon = Decimal(self.epsilon.data)
        except InvalidOperation as error:
            raise VerifyException("RMSNorm epsilon must be a finite decimal") from error
        if not epsilon.is_finite() or epsilon <= 0:
            raise VerifyException("RMSNorm epsilon must be positive and finite")


@irdl_op_definition
class RotaryEmbeddingOp(IRDLOperation):
    name = "dtensor.rotary_embedding"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)
    sequence_dimension = prop_def(StringAttr)
    head_dimension = prop_def(StringAttr)
    maximum_timescale = prop_def(IntAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        result_type: DTensorType,
        *,
        sequence_dimension: str,
        head_dimension: str,
        maximum_timescale: int,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            properties={
                "sequence_dimension": StringAttr(sequence_dimension),
                "head_dimension": StringAttr(head_dimension),
                "maximum_timescale": IntAttr(maximum_timescale),
            },
        )

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        _require_fully_reduced(before, after)
        if not _is_float_type(before.element_type) or not isinstance(
            after.element_type, Float32Type
        ):
            raise VerifyException("rotary embedding must promote a floating-point tensor to f32")
        if (
            before.logical_shape() != after.logical_shape()
            or before.sharding_axes() != after.sharding_axes()
        ):
            raise VerifyException("rotary embedding must preserve logical shape and sharding")
        shape = dict(before.logical_shape())
        sequence = self.sequence_dimension.data
        head = self.head_dimension.data
        if sequence not in shape or head not in shape or shape[head] % 2:
            raise VerifyException(
                "rotary embedding needs a sequence dimension and even head dimension"
            )
        sharding = before.sharding_axes()
        indexes = _dimension_index(before)
        if sharding[indexes[sequence]] or sharding[indexes[head]]:
            raise VerifyException("rotary embedding semantic dimensions cannot be sharded")
        if self.maximum_timescale.data <= 0:
            raise VerifyException("rotary embedding maximum timescale must be positive")


@irdl_op_definition
class SliceOp(IRDLOperation):
    name = "dtensor.slice"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)
    dimension = prop_def(StringAttr)
    index = prop_def(IntAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        result_type: DTensorType,
        *,
        dimension: str,
        index: int,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            properties={"dimension": StringAttr(dimension), "index": IntAttr(index)},
        )

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        dimension = self.dimension.data
        indexes = _dimension_index(before)
        if dimension not in indexes:
            raise VerifyException("slice references an unknown dimension")
        index = indexes[dimension]
        if before.sharding_axes()[index]:
            raise VerifyException("slice cannot index a sharded dimension")
        if not 0 <= self.index.data < before.logical_shape()[index][1]:
            raise VerifyException("slice index is out of bounds")
        expected_shape = tuple(
            value for offset, value in enumerate(before.logical_shape()) if offset != index
        )
        expected_sharding = tuple(
            value for offset, value in enumerate(before.sharding_axes()) if offset != index
        )
        if (
            after.logical_shape() != expected_shape
            or after.sharding_axes() != expected_sharding
            or before.element_type != after.element_type
            or before.pending_reductions() != after.pending_reductions()
        ):
            raise VerifyException("slice result must remove only its indexed unsharded dimension")


@irdl_op_definition
class RenameDimensionOp(IRDLOperation):
    name = "dtensor.rename_dimension"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)
    source_dimension = prop_def(StringAttr)
    destination_dimension = prop_def(StringAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        result_type: DTensorType,
        *,
        source_dimension: str,
        destination_dimension: str,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            properties={
                "source_dimension": StringAttr(source_dimension),
                "destination_dimension": StringAttr(destination_dimension),
            },
        )

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        source = self.source_dimension.data
        destination = self.destination_dimension.data
        before_shape = before.logical_shape()
        names = tuple(name for name, _ in before_shape)
        if not source or not destination or source not in names or destination in names:
            raise VerifyException(
                "dimension rename needs one present source and one fresh destination"
            )
        expected_shape = tuple(
            (destination if name == source else name, size) for name, size in before_shape
        )
        if (
            after.logical_shape() != expected_shape
            or before.element_type != after.element_type
            or before.sharding_axes() != after.sharding_axes()
            or before.pending_reductions() != after.pending_reductions()
        ):
            raise VerifyException("dimension rename may only change one logical name")


@irdl_op_definition
class PackedCausalMaskOp(IRDLOperation):
    name = "dtensor.packed_causal_mask"
    sequence_starts = operand_def(DTensorType)
    result = result_def(DTensorType)
    sequence_dimension = prop_def(StringAttr)
    query_dimension = prop_def(StringAttr)
    key_dimension = prop_def(StringAttr)

    def __init__(
        self,
        sequence_starts: SSAValue | IRDLOperation,
        result_type: DTensorType,
        *,
        sequence_dimension: str,
        query_dimension: str,
        key_dimension: str,
    ) -> None:
        super().__init__(
            operands=[sequence_starts],
            result_types=[result_type],
            properties={
                "sequence_dimension": StringAttr(sequence_dimension),
                "query_dimension": StringAttr(query_dimension),
                "key_dimension": StringAttr(key_dimension),
            },
        )

    def verify_(self) -> None:
        before, after = self.sequence_starts.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        _require_fully_reduced(before, after)
        if not _is_boolean_type(before.element_type) or not _is_boolean_type(after.element_type):
            raise VerifyException("packed causal masks require boolean tensors")
        sequence = self.sequence_dimension.data
        query = self.query_dimension.data
        key = self.key_dimension.data
        before_shape = dict(before.logical_shape())
        if sequence not in before_shape or query == key:
            raise VerifyException("packed causal mask dimensions are invalid")
        sequence_index = _dimension_index(before)[sequence]
        if before.sharding_axes()[sequence_index]:
            raise VerifyException("packed causal mask sequence dimension cannot be sharded")
        expected_shape = tuple(
            (name, size) for name, size in before.logical_shape() if name != sequence
        ) + ((query, before_shape[sequence]), (key, before_shape[sequence]))
        expected_sharding = tuple(
            axes for index, axes in enumerate(before.sharding_axes()) if index != sequence_index
        ) + ((), ())
        if after.logical_shape() != expected_shape or after.sharding_axes() != expected_sharding:
            raise VerifyException("packed causal mask result has the wrong shape or sharding")


@irdl_op_definition
class MaskedSoftmaxOp(IRDLOperation):
    name = "dtensor.masked_softmax"
    value = operand_def(DTensorType)
    mask = operand_def(DTensorType)
    result = result_def(DTensorType)
    dimension = prop_def(StringAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        mask: SSAValue | IRDLOperation,
        result_type: DTensorType,
        *,
        dimension: str,
    ) -> None:
        super().__init__(
            operands=[value, mask],
            result_types=[result_type],
            properties={"dimension": StringAttr(dimension)},
        )

    def verify_(self) -> None:
        value, mask, result = self.value.type, self.mask.type, self.result.type
        assert isinstance(value, DTensorType)
        assert isinstance(mask, DTensorType)
        assert isinstance(result, DTensorType)
        _require_fully_reduced(value, mask, result)
        if not _is_float_type(value.element_type) or not _is_float_type(result.element_type):
            raise VerifyException("masked softmax values must be floating point")
        if not _is_boolean_type(mask.element_type):
            raise VerifyException("masked softmax mask must be boolean")
        if (
            value.logical_shape() != result.logical_shape()
            or value.sharding_axes() != result.sharding_axes()
        ):
            raise VerifyException("masked softmax result must preserve shape and sharding")
        value_shape = dict(value.logical_shape())
        value_sharding = dict(zip(value_shape, value.sharding_axes(), strict=True))
        if self.dimension.data not in value_shape:
            raise VerifyException("masked softmax references an unknown dimension")
        if value_sharding[self.dimension.data]:
            raise VerifyException("masked softmax dimension cannot be sharded")
        for (name, size), axes in zip(mask.logical_shape(), mask.sharding_axes(), strict=True):
            if value_shape.get(name) != size or value_sharding[name] != axes:
                raise VerifyException(
                    "masked softmax mask dimensions must be a named subset of its values"
                )


@irdl_op_definition
class ReduceLocalOp(IRDLOperation):
    name = "dtensor.reduce_local"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)
    dimensions = prop_def(ArrayAttr[StringAttr])
    reducer = prop_def(StringAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        result_type: DTensorType,
        dimensions: tuple[str, ...],
        reducer: str,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            properties={
                "dimensions": _string_array(tuple(sorted(dimensions))),
                "reducer": StringAttr(reducer),
            },
        )

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        _require_fully_reduced(before)
        dimensions = tuple(value.data for value in self.dimensions)
        if not dimensions or dimensions != tuple(sorted(set(dimensions))):
            raise VerifyException(
                "local reduction dimensions must be non-empty, unique, and canonical"
            )
        if self.reducer.data not in {"sum", "max", "min"}:
            raise VerifyException("unsupported local reduction")
        indexes = _dimension_index(before)
        if any(dimension not in indexes for dimension in dimensions):
            raise VerifyException("local reduction references an unknown dimension")
        retained = tuple(
            (shape, sharding)
            for shape, sharding in zip(before.logical_shape(), before.sharding_axes(), strict=True)
            if shape[0] not in dimensions
        )
        if after.logical_shape() != tuple(shape for shape, _ in retained):
            raise VerifyException("local reduction result has the wrong logical shape")
        if after.sharding_axes() != tuple(sharding for _, sharding in retained):
            raise VerifyException("local reduction result has the wrong retained sharding")
        reduced_axes = sorted(
            {
                axis
                for dimension in dimensions
                for axis in before.sharding_axes()[indexes[dimension]]
            }
        )
        expected_pending = {axis: self.reducer.data for axis in reduced_axes}
        if after.pending_reductions() != expected_pending:
            raise VerifyException(
                "local reduction must expose reductions over sharded dimensions as pending"
            )
        if before.element_type != after.element_type:
            raise VerifyException("local reduction cannot change element type")


@irdl_op_definition
class TransposeOp(IRDLOperation):
    name = "dtensor.transpose"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)
    permutation = prop_def(ArrayAttr[IntAttr])

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        result_type: DTensorType,
        permutation: tuple[int, ...],
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            properties={"permutation": ArrayAttr(IntAttr(index) for index in permutation)},
        )

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        permutation = tuple(value.data for value in self.permutation)
        if sorted(permutation) != list(range(len(before.dimensions))):
            raise VerifyException("transpose permutation must cover every dimension exactly once")
        if after.logical_shape() != tuple(before.logical_shape()[index] for index in permutation):
            raise VerifyException("transpose result has the wrong logical shape")
        if after.sharding_axes() != tuple(before.sharding_axes()[index] for index in permutation):
            raise VerifyException("transpose result has the wrong sharding")
        if before.element_type != after.element_type:
            raise VerifyException("transpose cannot change element type")
        if before.pending_reductions() != after.pending_reductions():
            raise VerifyException("transpose cannot complete pending reductions")


@irdl_op_definition
class BroadcastOp(IRDLOperation):
    name = "dtensor.broadcast"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)

    def __init__(self, value: SSAValue | IRDLOperation, result_type: DTensorType) -> None:
        super().__init__(operands=[value], result_types=[result_type])

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        before_shape = dict(before.logical_shape())
        after_shape = dict(after.logical_shape())
        if any(after_shape.get(name) != size for name, size in before_shape.items()):
            raise VerifyException("broadcast must preserve every input dimension and size")
        before_sharding = dict(
            zip((name for name, _ in before.logical_shape()), before.sharding_axes(), strict=True)
        )
        after_sharding = dict(
            zip((name for name, _ in after.logical_shape()), after.sharding_axes(), strict=True)
        )
        if any(after_sharding[name] != axes for name, axes in before_sharding.items()):
            raise VerifyException("broadcast must preserve input sharding")
        new_dimensions = set(after_shape) - set(before_shape)
        if any(after_sharding[name] for name in new_dimensions):
            raise VerifyException("new broadcast dimensions must be replicated")
        retained_order = tuple(name for name, _ in after.logical_shape() if name in before_shape)
        if retained_order != tuple(name for name, _ in before.logical_shape()):
            raise VerifyException("broadcast must preserve input dimension order")
        if before.element_type != after.element_type:
            raise VerifyException("broadcast cannot change element type")
        if before.pending_reductions() != after.pending_reductions():
            raise VerifyException("broadcast cannot complete pending reductions")


@irdl_op_definition
class EmbeddingLookupOp(IRDLOperation):
    name = "dtensor.embedding_lookup"
    table = operand_def(DTensorType)
    indices = operand_def(DTensorType)
    result = result_def(DTensorType)
    vocabulary_dimension = prop_def(StringAttr)

    def __init__(
        self,
        table: SSAValue | IRDLOperation,
        indices: SSAValue | IRDLOperation,
        result_type: DTensorType,
        vocabulary_dimension: str,
    ) -> None:
        super().__init__(
            operands=[table, indices],
            result_types=[result_type],
            properties={"vocabulary_dimension": StringAttr(vocabulary_dimension)},
        )

    def verify_(self) -> None:
        table, indices, result = self.table.type, self.indices.type, self.result.type
        assert isinstance(table, DTensorType)
        assert isinstance(indices, DTensorType)
        assert isinstance(result, DTensorType)
        _require_fully_reduced(table, indices)
        if not isinstance(indices.element_type, IntegerType):
            raise VerifyException("embedding indices must have integer element type")
        vocabulary = self.vocabulary_dimension.data
        table_indexes = _dimension_index(table)
        if vocabulary not in table_indexes:
            raise VerifyException("embedding vocabulary dimension is absent from the table")
        retained_table = tuple(
            (shape, sharding)
            for shape, sharding in zip(table.logical_shape(), table.sharding_axes(), strict=True)
            if shape[0] != vocabulary
        )
        expected_shape = (*indices.logical_shape(), *(shape for shape, _ in retained_table))
        if len(dict(expected_shape)) != len(expected_shape):
            raise VerifyException("embedding result dimensions must be unique")
        if result.logical_shape() != expected_shape:
            raise VerifyException("embedding result has the wrong logical shape")
        expected_sharding = (
            *indices.sharding_axes(),
            *(sharding for _, sharding in retained_table),
        )
        if result.sharding_axes() != expected_sharding:
            raise VerifyException("embedding result has the wrong sharding")
        vocabulary_axes = table.sharding_axes()[table_indexes[vocabulary]]
        if result.pending_reductions() != {axis: "sum" for axis in vocabulary_axes}:
            raise VerifyException(
                "sharded embedding lookup must expose its cross-device sum as pending"
            )
        if result.element_type != table.element_type:
            raise VerifyException("embedding result must use the table element type")


@irdl_op_definition
class EinsumLocalOp(IRDLOperation):
    name = "dtensor.einsum_local"
    lhs = operand_def(DTensorType)
    rhs = operand_def(DTensorType)
    result = result_def(DTensorType)
    contracting_dimension = prop_def(StringAttr)
    accumulation_type = prop_def(Attribute)

    def __init__(
        self,
        lhs: SSAValue | IRDLOperation,
        rhs: SSAValue | IRDLOperation,
        result_type: DTensorType,
        contracting_dimension: str,
        accumulation_type: Attribute = f32,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
            properties={
                "contracting_dimension": StringAttr(contracting_dimension),
                "accumulation_type": accumulation_type,
            },
        )

    def verify_(self) -> None:
        lhs, rhs, result = self.lhs.type, self.rhs.type, self.result.type
        assert isinstance(lhs, DTensorType)
        assert isinstance(rhs, DTensorType)
        assert isinstance(result, DTensorType)
        if not isinstance(lhs.element_type, (BFloat16Type, Float16Type)):
            raise VerifyException("local einsum supports bf16 or f16 inputs")
        if lhs.element_type != rhs.element_type:
            raise VerifyException("local einsum inputs must have the same element type")
        if not isinstance(self.accumulation_type, Float32Type):
            raise VerifyException("local einsum accumulation must be f32")
        if result.element_type != self.accumulation_type:
            raise VerifyException("local einsum result must match its accumulation type")
        if lhs.pending_reductions() or rhs.pending_reductions():
            raise VerifyException("local einsum cannot consume a partially reduced tensor")
        contraction = self.contracting_dimension.data
        lhs_shape, rhs_shape = dict(lhs.logical_shape()), dict(rhs.logical_shape())
        if contraction not in lhs_shape or contraction not in rhs_shape:
            raise VerifyException("einsum contraction dimension must exist in both operands")
        if lhs_shape[contraction] != rhs_shape[contraction]:
            raise VerifyException("einsum contraction dimensions must have equal sizes")
        expected_dimensions = tuple(
            entry
            for entry in (*lhs.logical_shape(), *rhs.logical_shape())
            if entry[0] != contraction
        )
        if len(dict(expected_dimensions)) != len(expected_dimensions):
            raise VerifyException("non-contracting einsum dimensions must be unique")
        if result.logical_shape() != expected_dimensions:
            raise VerifyException("einsum result dimensions do not match non-contracted inputs")
        expected_sharding = tuple(
            axes
            for tensor in (lhs, rhs)
            for (name, _), axes in zip(tensor.logical_shape(), tensor.sharding_axes(), strict=True)
            if name != contraction
        )
        if result.sharding_axes() != expected_sharding:
            raise VerifyException(
                "einsum result must preserve sharding on non-contracted dimensions"
            )
        lhs_axes = lhs.sharding_axes()[list(lhs_shape).index(contraction)]
        rhs_axes = rhs.sharding_axes()[list(rhs_shape).index(contraction)]
        if lhs_axes != rhs_axes:
            raise VerifyException(
                "einsum contracting dimensions must have identical local sharding"
            )
        expected_pending = {axis: "sum" for axis in (*lhs_axes, *rhs_axes)}
        if result.pending_reductions() != expected_pending:
            raise VerifyException("einsum result has incorrect pending reductions")


@irdl_op_definition
class EinsumOp(IRDLOperation):
    name = "dtensor.einsum"
    lhs = operand_def(DTensorType)
    rhs = operand_def(DTensorType)
    result = result_def(DTensorType)
    contracting_dimensions = prop_def(ArrayAttr[StringAttr])
    accumulation_type = prop_def(Attribute)

    def __init__(
        self,
        lhs: SSAValue | IRDLOperation,
        rhs: SSAValue | IRDLOperation,
        result_type: DTensorType,
        contracting_dimensions: tuple[str, ...],
        accumulation_type: Attribute = f32,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
            properties={
                "contracting_dimensions": _string_array(tuple(sorted(contracting_dimensions))),
                "accumulation_type": accumulation_type,
            },
        )

    def verify_(self) -> None:
        lhs, rhs, result = self.lhs.type, self.rhs.type, self.result.type
        assert isinstance(lhs, DTensorType)
        assert isinstance(rhs, DTensorType)
        assert isinstance(result, DTensorType)
        if not isinstance(lhs.element_type, (BFloat16Type, Float16Type, Float32Type)):
            raise VerifyException("einsum supports bf16, f16, or f32 inputs")
        if lhs.element_type != rhs.element_type:
            raise VerifyException("einsum inputs must have the same element type")
        if not isinstance(self.accumulation_type, Float32Type):
            raise VerifyException("einsum accumulation must be f32")
        if result.element_type != self.accumulation_type:
            raise VerifyException("einsum result must match its accumulation type")
        if lhs.pending_reductions() or rhs.pending_reductions():
            raise VerifyException("einsum cannot consume a partially reduced tensor")

        contractions = tuple(value.data for value in self.contracting_dimensions)
        if not contractions or contractions != tuple(sorted(set(contractions))):
            raise VerifyException(
                "einsum contracting dimensions must be nonempty, unique, and canonical"
            )
        lhs_shape = dict(lhs.logical_shape())
        rhs_shape = dict(rhs.logical_shape())
        lhs_sharding = dict(zip(lhs_shape, lhs.sharding_axes(), strict=True))
        rhs_sharding = dict(zip(rhs_shape, rhs.sharding_axes(), strict=True))
        shared = set(lhs_shape) & set(rhs_shape)
        if not set(contractions) <= shared:
            raise VerifyException("every einsum contracting dimension must exist in both operands")
        for dimension in shared:
            if lhs_shape[dimension] != rhs_shape[dimension]:
                raise VerifyException(f"einsum dimension {dimension} has unequal sizes")
            if lhs_sharding[dimension] != rhs_sharding[dimension]:
                raise VerifyException(
                    f"einsum shared dimension {dimension} has unequal local sharding"
                )

        expected_dimensions = (set(lhs_shape) | set(rhs_shape)) - set(contractions)
        result_shape = dict(result.logical_shape())
        if set(result_shape) != expected_dimensions:
            raise VerifyException(
                "einsum result must contain every noncontracted dimension exactly once"
            )
        expected_sharding: list[tuple[str, ...]] = []
        for dimension, size in result.logical_shape():
            source_shape = lhs_shape if dimension in lhs_shape else rhs_shape
            source_sharding = lhs_sharding if dimension in lhs_sharding else rhs_sharding
            if source_shape[dimension] != size:
                raise VerifyException(f"einsum result dimension {dimension} has the wrong size")
            expected_sharding.append(source_sharding[dimension])
        if result.sharding_axes() != tuple(expected_sharding):
            raise VerifyException(
                "einsum result must preserve sharding on every retained dimension"
            )
        pending_axes = {axis for dimension in contractions for axis in lhs_sharding[dimension]}
        if result.pending_reductions() != {axis: "sum" for axis in sorted(pending_axes)}:
            raise VerifyException("einsum result has incorrect pending reductions")


@irdl_op_definition
class AllGatherOp(IRDLOperation):
    name = "dtensor.all_gather"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)

    def __init__(self, value: SSAValue | IRDLOperation, result_type: DTensorType) -> None:
        super().__init__(operands=[value], result_types=[result_type])

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        if not _same_value_shape(before, after):
            raise VerifyException("all-gather cannot change shape or dtype")
        if before.pending_reductions() != after.pending_reductions():
            raise VerifyException("all-gather cannot complete pending reductions")
        for old_axes, new_axes in zip(before.sharding_axes(), after.sharding_axes(), strict=True):
            if old_axes[: len(new_axes)] != new_axes:
                raise VerifyException("all-gather may only remove sharding suffixes")


@irdl_op_definition
class ReduceScatterOp(IRDLOperation):
    name = "dtensor.reduce_scatter"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)
    axes = prop_def(ArrayAttr[StringAttr])
    scatter_dimensions = prop_def(ArrayAttr[StringAttr])
    reducer = prop_def(StringAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        result_type: DTensorType,
        axes: tuple[str, ...],
        scatter_dimensions: tuple[str, ...],
        reducer: str = "sum",
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            properties={
                "axes": _string_array(axes),
                "scatter_dimensions": _string_array(scatter_dimensions),
                "reducer": StringAttr(reducer),
            },
        )

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        if not _same_value_shape(before, after):
            raise VerifyException("reduce-scatter cannot change global shape or dtype")
        axes = tuple(value.data for value in self.axes)
        dimensions = tuple(value.data for value in self.scatter_dimensions)
        if len(axes) != len(dimensions) or not axes:
            raise VerifyException("reduce-scatter needs matching axes and scatter dimensions")
        if len(axes) != len(set(axes)):
            raise VerifyException("reduce-scatter axes must be unique")
        if axes != tuple(sorted(axes)):
            raise VerifyException("reduce-scatter axes must be in canonical lexical order")
        pending = before.pending_reductions()
        if any(pending.get(axis) != self.reducer.data for axis in axes):
            raise VerifyException("reduce-scatter axes must complete matching pending reductions")
        expected_pending = {axis: reducer for axis, reducer in pending.items() if axis not in axes}
        if after.pending_reductions() != expected_pending:
            raise VerifyException("reduce-scatter result has incorrect pending reductions")
        dimension_index = {name: index for index, (name, _) in enumerate(before.logical_shape())}
        expected_sharding = [list(value) for value in before.sharding_axes()]
        for axis, dimension in zip(axes, dimensions, strict=True):
            if dimension not in dimension_index:
                raise VerifyException("reduce-scatter dimension is not present in the tensor")
            expected_sharding[dimension_index[dimension]].append(axis)
        if after.sharding_axes() != tuple(tuple(value) for value in expected_sharding):
            raise VerifyException("reduce-scatter result has incorrect sharding")


@irdl_op_definition
class AllReduceOp(IRDLOperation):
    name = "dtensor.all_reduce"
    value = operand_def(DTensorType)
    result = result_def(DTensorType)
    axes = prop_def(ArrayAttr[StringAttr])
    reducer = prop_def(StringAttr)

    def __init__(
        self,
        value: SSAValue | IRDLOperation,
        result_type: DTensorType,
        axes: tuple[str, ...],
        reducer: str = "sum",
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            properties={"axes": _string_array(axes), "reducer": StringAttr(reducer)},
        )

    def verify_(self) -> None:
        before, after = self.value.type, self.result.type
        assert isinstance(before, DTensorType) and isinstance(after, DTensorType)
        if not _same_value_shape(before, after) or before.sharding_axes() != after.sharding_axes():
            raise VerifyException("all-reduce cannot change shape, dtype, or sharding")
        axes = tuple(value.data for value in self.axes)
        if not axes or axes != tuple(sorted(axes)):
            raise VerifyException("all-reduce axes must be non-empty and canonically ordered")
        if len(axes) != len(set(axes)):
            raise VerifyException("all-reduce axes must be unique")
        pending = before.pending_reductions()
        if any(pending.get(axis) != self.reducer.data for axis in axes):
            raise VerifyException("all-reduce axes must complete matching pending reductions")
        expected = {axis: reducer for axis, reducer in pending.items() if axis not in axes}
        if after.pending_reductions() != expected:
            raise VerifyException("all-reduce result has incorrect pending reductions")


def _without_dimension(tensor: DTensorType, dimension: str) -> DTensorType:
    indexes = _dimension_index(tensor)
    if dimension not in indexes:
        raise VerifyException(f"stacked scan input is missing layer dimension {dimension}")
    index = indexes[dimension]
    if tensor.sharding_axes()[index]:
        raise VerifyException("scan layer dimension must be locally replicated")
    return DTensorType(
        tensor.element_type,
        ArrayAttr(value for offset, value in enumerate(tensor.dimensions) if offset != index),
        ShardingAttr(
            ArrayAttr(
                value for offset, value in enumerate(tensor.sharding.dimensions) if offset != index
            )
        ),
        tensor.pending,
    )


@irdl_op_definition
class ScanYieldOp(IRDLOperation):
    name = "dtensor.scan_yield"
    values = var_operand_def(DTensorType)
    traits = traits_def(IsTerminator())

    def __init__(self, *values: SSAValue | IRDLOperation) -> None:
        super().__init__(operands=[list(values)])

    def verify_(self) -> None:
        if not isinstance(self.parent_op(), LayerScanOp):
            raise VerifyException("dtensor.scan_yield must terminate a layer scan")


@irdl_op_definition
class LayerScanOp(IRDLOperation):
    name = "dtensor.layer_scan"
    captures = var_operand_def(DTensorType)
    outputs = var_result_def(DTensorType)
    body = region_def("single_block")
    carry_count = prop_def(IntAttr)
    stacked_count = prop_def(IntAttr)
    layer_dimension = prop_def(StringAttr)
    trip_count = prop_def(IntAttr)
    traits = traits_def(IsolatedFromAbove())

    def __init__(
        self,
        captures: tuple[SSAValue | IRDLOperation, ...],
        body: Region,
        *,
        carry_count: int,
        stacked_count: int,
        layer_dimension: str,
        trip_count: int,
    ) -> None:
        capture_types = [SSAValue.get(value).type for value in captures]
        super().__init__(
            operands=[list(captures)],
            result_types=[capture_types[:carry_count]],
            regions=[body],
            properties={
                "carry_count": IntAttr(carry_count),
                "stacked_count": IntAttr(stacked_count),
                "layer_dimension": StringAttr(layer_dimension),
                "trip_count": IntAttr(trip_count),
            },
        )

    def verify_(self) -> None:
        carry_count = self.carry_count.data
        stacked_count = self.stacked_count.data
        if carry_count <= 0 or stacked_count <= 0 or self.trip_count.data <= 0:
            raise VerifyException("layer scan needs positive carry, stacked-input, and trip counts")
        captures = tuple(self.captures)
        if carry_count + stacked_count > len(captures):
            raise VerifyException("layer scan capture segments exceed its inputs")
        if len(self.outputs) != carry_count:
            raise VerifyException("layer scan output count must match its carry count")
        body_arguments = tuple(self.body.block.args)
        if len(body_arguments) != len(captures):
            raise VerifyException("layer scan body arguments must match its captures")
        for capture, argument, output in zip(
            captures[:carry_count],
            body_arguments[:carry_count],
            self.outputs,
            strict=True,
        ):
            if capture.type != argument.type or capture.type != output.type:
                raise VerifyException("layer scan carries must preserve their exact types")
            assert isinstance(capture.type, DTensorType)
            _require_fully_reduced(capture.type)
        layer_dimension = self.layer_dimension.data
        for capture, argument in zip(
            captures[carry_count : carry_count + stacked_count],
            body_arguments[carry_count : carry_count + stacked_count],
            strict=True,
        ):
            assert isinstance(capture.type, DTensorType)
            shape = dict(capture.type.logical_shape())
            if shape.get(layer_dimension) != self.trip_count.data:
                raise VerifyException(
                    "stacked scan inputs must match the layer dimension and trip count"
                )
            if argument.type != _without_dimension(capture.type, layer_dimension):
                raise VerifyException(
                    "stacked scan body arguments must remove exactly the layer dimension"
                )
        for capture, argument in zip(
            captures[carry_count + stacked_count :],
            body_arguments[carry_count + stacked_count :],
            strict=True,
        ):
            if capture.type != argument.type:
                raise VerifyException("layer scan invariants must preserve their exact types")
        terminator = self.body.block.last_op
        if not isinstance(terminator, ScanYieldOp):
            raise VerifyException("layer scan body must end with dtensor.scan_yield")
        if len(terminator.values) != carry_count or any(
            yielded.type != output.type
            for yielded, output in zip(terminator.values, self.outputs, strict=True)
        ):
            raise VerifyException("layer scan yield types must match its carries")


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "dtensor.return"
    values = var_operand_def(DTensorType)
    traits = traits_def(IsTerminator())

    def __init__(self, *values: SSAValue | IRDLOperation) -> None:
        super().__init__(operands=[list(values)])

    def verify_(self) -> None:
        for value in self.values:
            assert isinstance(value.type, DTensorType)
            if value.type.pending_reductions():
                raise VerifyException("program cannot return a partially reduced tensor")


@irdl_op_definition
class ProgramOp(IRDLOperation):
    name = "dtensor.program"
    body = region_def("single_block")
    sym_name = prop_def(StringAttr)
    mesh = prop_def(MeshAttr)
    traits = traits_def(IsolatedFromAbove())

    def __init__(self, sym_name: str, mesh: MeshAttr, body: Region) -> None:
        super().__init__(
            properties={"sym_name": StringAttr(sym_name), "mesh": mesh}, regions=[body]
        )

    def verify_(self) -> None:
        self.mesh.verify()
        if not isinstance(self.body.block.last_op, ReturnOp):
            raise VerifyException("distributed program must end with dtensor.return")
        mesh = self.mesh.sizes()
        dimensions: dict[str, int] = {}
        values: list[SSAValue] = []
        for operation in self.walk():
            values.extend(operation.results)
            for region in operation.regions:
                for block in region.blocks:
                    values.extend(block.args)
        for value in values:
            if not isinstance(value.type, DTensorType):
                continue
            tensor = value.type
            tensor.verify()
            used_axes: list[str] = []
            for dimension, axes in zip(tensor.dimensions, tensor.sharding_axes(), strict=True):
                previous = dimensions.setdefault(dimension.symbol.data, dimension.size.data)
                if previous != dimension.size.data:
                    raise VerifyException(
                        f"logical dimension {dimension.symbol.data} has conflicting sizes"
                    )
                shard_count = 1
                for axis in axes:
                    if axis not in mesh:
                        raise VerifyException(f"unknown mesh axis {axis}")
                    shard_count *= mesh[axis]
                    used_axes.append(axis)
                if dimension.size.data % shard_count:
                    raise VerifyException(
                        f"dimension {dimension.symbol.data} is not divisible by its sharding"
                    )
            duplicates = [axis for axis, count in Counter(used_axes).items() if count > 1]
            if duplicates:
                raise VerifyException(f"mesh axes shard multiple tensor dimensions: {duplicates}")
            if any(axis not in mesh for axis in tensor.pending_reductions()):
                raise VerifyException("pending reduction references an unknown mesh axis")


DistributedTensor = Dialect(
    "dtensor",
    [
        ProgramOp,
        ElementwiseOp,
        CastOp,
        RmsNormOp,
        RotaryEmbeddingOp,
        SliceOp,
        RenameDimensionOp,
        PackedCausalMaskOp,
        MaskedSoftmaxOp,
        ReduceLocalOp,
        TransposeOp,
        BroadcastOp,
        EmbeddingLookupOp,
        EinsumLocalOp,
        EinsumOp,
        AllGatherOp,
        ReduceScatterOp,
        AllReduceOp,
        LayerScanOp,
        ScanYieldOp,
        ReturnOp,
    ],
    [
        ElementwiseMaterializationAttr,
        DimensionAttr,
        AxisListAttr,
        MeshAttr,
        ShardingAttr,
        PendingReductionsAttr,
        DTensorType,
    ],
)
