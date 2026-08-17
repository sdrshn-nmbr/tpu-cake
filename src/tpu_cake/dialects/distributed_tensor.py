from __future__ import annotations

from collections import Counter

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
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, Region, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator
from xdsl.utils.exceptions import VerifyException


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
    element_type: Attribute
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


@irdl_op_definition
class ElementwiseOp(IRDLOperation):
    name = "dtensor.elementwise"
    values = var_operand_def(DTensorType)
    result = result_def(DTensorType)
    function = prop_def(StringAttr)

    def __init__(
        self,
        values: tuple[SSAValue | IRDLOperation, ...],
        result_type: DTensorType,
        function: str,
    ) -> None:
        super().__init__(
            operands=[list(values)],
            result_types=[result_type],
            properties={"function": StringAttr(function)},
        )

    def verify_(self) -> None:
        if not self.values:
            raise VerifyException("elementwise operation needs at least one input")
        result = self.result.type
        assert isinstance(result, DTensorType)
        if self.function.data not in {"add", "multiply", "silu", "gelu", "relu", "exp"}:
            raise VerifyException("unsupported elementwise function")
        for value in self.values:
            tensor = value.type
            assert isinstance(tensor, DTensorType)
            if tensor != result:
                raise VerifyException(
                    "elementwise inputs and result must have identical distributed tensor types"
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
            for shape, sharding in zip(
                before.logical_shape(), before.sharding_axes(), strict=True
            )
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
        if after.sharding_axes() != tuple(
            before.sharding_axes()[index] for index in permutation
        ):
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


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "dtensor.return"
    values = var_operand_def(DTensorType)
    traits = traits_def(IsTerminator())

    def __init__(self, *values: SSAValue | IRDLOperation) -> None:
        super().__init__(operands=list(values))

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
        values = list(self.body.block.args)
        for operation in self.body.block.ops:
            values.extend(operation.results)
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
        ReduceLocalOp,
        TransposeOp,
        BroadcastOp,
        EmbeddingLookupOp,
        EinsumLocalOp,
        AllGatherOp,
        ReduceScatterOp,
        AllReduceOp,
        ReturnOp,
    ],
    [DimensionAttr, AxisListAttr, MeshAttr, ShardingAttr, PendingReductionsAttr, DTensorType],
)
