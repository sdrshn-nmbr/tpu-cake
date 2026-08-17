from __future__ import annotations

import math
from enum import StrEnum

from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    Float16Type,
    Float32Type,
    IntAttr,
    IntegerType,
    MemRefType,
    StringAttr,
)
from xdsl.ir import (
    Dialect,
    EnumAttribute,
    Operation,
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
    prop_def,
    region_def,
    result_def,
    traits_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator
from xdsl.utils.exceptions import VerifyException


class MemorySpace(StrEnum):
    HBM = "hbm"
    VMEM = "vmem"
    SMEM = "smem"


class Ownership(StrEnum):
    EXTERNAL = "external"
    KERNEL = "kernel"
    CORE = "core"
    DEVICE = "device"


@irdl_attr_definition
class MemorySpaceAttr(EnumAttribute[MemorySpace], SpacedOpaqueSyntaxAttribute):
    name = "tpu_schedule.memory_space"


@irdl_attr_definition
class OwnershipAttr(EnumAttribute[Ownership], SpacedOpaqueSyntaxAttribute):
    name = "tpu_schedule.ownership"


@irdl_attr_definition
class ShapeAttr(ParametrizedAttribute):
    name = "tpu_schedule.shape"
    dimensions: ArrayAttr[StringAttr]


@irdl_attr_definition
class ShardingAttr(ParametrizedAttribute):
    name = "tpu_schedule.sharding"
    axes: ArrayAttr[StringAttr]


@irdl_attr_definition
class LayoutAttr(ParametrizedAttribute):
    name = "tpu_schedule.layout"
    order: ArrayAttr[IntAttr]


@irdl_attr_definition
class LifetimeAttr(ParametrizedAttribute):
    name = "tpu_schedule.lifetime"
    start: IntAttr
    end: IntAttr

    def verify(self) -> None:
        if self.start.data < 0 or self.end.data < self.start.data:
            raise VerifyException("buffer lifetime must be a nonnegative closed stage interval")


@irdl_attr_definition
class TileRegionAttr(ParametrizedAttribute):
    name = "tpu_schedule.tile_region"
    offsets: ArrayAttr[IntAttr]
    sizes: ArrayAttr[IntAttr]
    strides: ArrayAttr[IntAttr]

    def verify(self) -> None:
        offsets = tuple(value.data for value in self.offsets)
        sizes = tuple(value.data for value in self.sizes)
        strides = tuple(value.data for value in self.strides)
        if not offsets or len(offsets) != len(sizes) or len(offsets) != len(strides):
            raise VerifyException("tile region offsets, sizes, and strides need equal rank")
        if any(offset < 0 for offset in offsets):
            raise VerifyException("tile region offsets must be nonnegative")
        if any(size <= 0 for size in sizes):
            raise VerifyException("tile region sizes must be positive")
        if any(stride <= 0 for stride in strides):
            raise VerifyException("tile region strides must be positive")

    def bounds(self) -> tuple[tuple[int, int], ...]:
        self.verify()
        return tuple(
            (offset, offset + (size - 1) * stride)
            for offset, size, stride in zip(
                (value.data for value in self.offsets),
                (value.data for value in self.sizes),
                (value.data for value in self.strides),
                strict=True,
            )
        )


@irdl_attr_definition
class InterconnectAttr(ParametrizedAttribute):
    name = "tpu_schedule.interconnect"
    mesh_axes: ArrayAttr[StringAttr]
    bandwidth_bytes_per_second: ArrayAttr[IntAttr]

    def verify(self) -> None:
        axes = tuple(value.data for value in self.mesh_axes)
        bandwidths = tuple(value.data for value in self.bandwidth_bytes_per_second)
        if len(axes) != len(bandwidths):
            raise VerifyException("interconnect axes and bandwidths must have equal length")
        if axes != tuple(sorted(set(axes))):
            raise VerifyException("interconnect axes must be unique and canonically ordered")
        if any(bandwidth <= 0 for bandwidth in bandwidths):
            raise VerifyException("interconnect bandwidths must be positive")

    def bandwidths(self) -> dict[str, int]:
        self.verify()
        return {
            axis.data: bandwidth.data
            for axis, bandwidth in zip(
                self.mesh_axes,
                self.bandwidth_bytes_per_second,
                strict=True,
            )
        }


@irdl_attr_definition
class BufferType(ParametrizedAttribute, TypeAttribute):
    name = "tpu_schedule.buffer"
    storage: MemRefType
    shape: ShapeAttr
    space: MemorySpaceAttr
    sharding: ShardingAttr
    layout: LayoutAttr
    ownership: OwnershipAttr
    lifetime: LifetimeAttr

    def verify(self) -> None:
        if not self.storage.has_static_shape():
            raise VerifyException("schedule buffers must have static shapes")
        physical_shape = self.storage.get_shape()
        logical_shape = tuple(value.data for value in self.shape.dimensions)
        sharding = tuple(value.data for value in self.sharding.axes)
        layout = tuple(value.data for value in self.layout.order)
        rank = len(physical_shape)
        if len(logical_shape) != rank:
            raise VerifyException("logical shape rank must match storage rank")
        if len(sharding) != rank:
            raise VerifyException("sharding rank must match storage rank")
        if sorted(layout) != list(range(rank)):
            raise VerifyException("layout must be a permutation of buffer dimensions")
        for name, size in zip(logical_shape, physical_shape, strict=True):
            if name.isdecimal() and int(name) != size:
                raise VerifyException(
                    f"numeric logical dimension {name} does not match physical size {size}"
                )
        self.lifetime.verify()


@irdl_attr_definition
class DmaTokenType(ParametrizedAttribute, TypeAttribute):
    name = "tpu_schedule.dma_token"


@irdl_attr_definition
class SemaphoreType(ParametrizedAttribute, TypeAttribute):
    name = "tpu_schedule.semaphore"


def _element_bytes(buffer: BufferType) -> int:
    element_type = buffer.storage.element_type
    if isinstance(element_type, BFloat16Type | Float16Type):
        return 2
    if isinstance(element_type, Float32Type):
        return 4
    if isinstance(element_type, IntegerType):
        return math.ceil(element_type.width.data / 8)
    raise VerifyException(f"unsupported buffer element type: {element_type}")


def buffer_bytes(buffer: BufferType) -> int:
    buffer.verify()
    return math.prod(buffer.storage.get_shape()) * _element_bytes(buffer)


def _stage(operation: Operation) -> int | None:
    value = getattr(operation, "stage", None)
    return value.data if isinstance(value, IntAttr) else None


def _check_live(buffer: BufferType, stage: int) -> None:
    if not buffer.lifetime.start.data <= stage <= buffer.lifetime.end.data:
        raise VerifyException(
            f"buffer used at stage {stage} outside lifetime "
            f"[{buffer.lifetime.start.data}, {buffer.lifetime.end.data}]"
        )


def _lifetimes_overlap(lhs: BufferType, rhs: BufferType) -> bool:
    return not (
        lhs.lifetime.end.data < rhs.lifetime.start.data
        or rhs.lifetime.end.data < lhs.lifetime.start.data
    )


def _regions_overlap(lhs: TileRegionAttr, rhs: TileRegionAttr) -> bool:
    def progression_intersects(
        lhs_offset: int,
        lhs_size: int,
        lhs_stride: int,
        rhs_offset: int,
        rhs_size: int,
        rhs_stride: int,
    ) -> bool:
        low = max(lhs_offset, rhs_offset)
        high = min(
            lhs_offset + (lhs_size - 1) * lhs_stride,
            rhs_offset + (rhs_size - 1) * rhs_stride,
        )
        if low > high:
            return False
        divisor = math.gcd(lhs_stride, rhs_stride)
        delta = rhs_offset - lhs_offset
        if delta % divisor:
            return False
        reduced_modulus = rhs_stride // divisor
        if reduced_modulus == 1:
            first = lhs_offset
        else:
            index = (
                (delta // divisor)
                * pow(lhs_stride // divisor, -1, reduced_modulus)
            ) % reduced_modulus
            first = lhs_offset + index * lhs_stride
        period = math.lcm(lhs_stride, rhs_stride)
        if first < low:
            first += math.ceil((low - first) / period) * period
        return first <= high

    return all(
        progression_intersects(*values)
        for values in zip(
            (value.data for value in lhs.offsets),
            (value.data for value in lhs.sizes),
            (value.data for value in lhs.strides),
            (value.data for value in rhs.offsets),
            (value.data for value in rhs.sizes),
            (value.data for value in rhs.strides),
            strict=True,
        )
    )


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "tpu_schedule.alloc"
    buffer = result_def(BufferType)
    role = prop_def(StringAttr)

    def __init__(self, result_type: BufferType, role: str | StringAttr):
        super().__init__(
            result_types=[result_type],
            properties={"role": StringAttr(role) if isinstance(role, str) else role},
        )

    def verify_(self) -> None:
        buffer = self.buffer.type
        assert isinstance(buffer, BufferType)
        if buffer.space.data is MemorySpace.HBM:
            raise VerifyException("HBM buffers are kernel inputs, not local allocations")
        if buffer.ownership.data is Ownership.EXTERNAL:
            raise VerifyException("local allocations cannot have external ownership")
        buffer_bytes(buffer)


@irdl_op_definition
class ViewOp(IRDLOperation):
    name = "tpu_schedule.view"
    base = operand_def(BufferType)
    view = result_def(BufferType)
    region = prop_def(TileRegionAttr)
    alias_group = prop_def(StringAttr)

    def __init__(
        self,
        base: SSAValue | Operation,
        result_type: BufferType,
        *,
        offsets: tuple[int, ...],
        sizes: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        alias_group: str,
    ) -> None:
        strides = strides or (1,) * len(offsets)
        super().__init__(
            operands=[base],
            result_types=[result_type],
            properties={
                "region": TileRegionAttr(
                    ArrayAttr(IntAttr(value) for value in offsets),
                    ArrayAttr(IntAttr(value) for value in sizes),
                    ArrayAttr(IntAttr(value) for value in strides),
                ),
                "alias_group": StringAttr(alias_group),
            },
        )

    def verify_(self) -> None:
        base, view = self.base.type, self.view.type
        assert isinstance(base, BufferType) and isinstance(view, BufferType)
        self.region.verify()
        if not self.alias_group.data:
            raise VerifyException("buffer view needs a non-empty alias group")
        base_shape = base.storage.get_shape()
        view_shape = view.storage.get_shape()
        bounds = self.region.bounds()
        if len(bounds) != len(base_shape):
            raise VerifyException("tile view rank must match its base buffer")
        sizes = tuple(value.data for value in self.region.sizes)
        if view_shape != sizes:
            raise VerifyException("tile view storage shape must match its region size")
        if any(high >= extent for (_, high), extent in zip(bounds, base_shape, strict=True)):
            raise VerifyException("tile view exceeds its base buffer bounds")
        if base.storage.element_type != view.storage.element_type:
            raise VerifyException("tile view cannot change element type")
        if base.space != view.space or base.ownership != view.ownership:
            raise VerifyException("tile view must preserve memory space and ownership")
        if base.sharding != view.sharding or base.layout != view.layout:
            raise VerifyException("tile view must preserve sharding and layout")
        if (
            view.lifetime.start.data < base.lifetime.start.data
            or view.lifetime.end.data > base.lifetime.end.data
        ):
            raise VerifyException("tile view lifetime must be contained by its base buffer")


@irdl_op_definition
class SemaphoreAllocOp(IRDLOperation):
    name = "tpu_schedule.semaphore_alloc"
    semaphore = result_def(SemaphoreType)

    def __init__(self):
        super().__init__(result_types=[SemaphoreType()])


@irdl_op_definition
class DmaStartOp(IRDLOperation):
    name = "tpu_schedule.dma_start"
    source = operand_def(BufferType)
    destination = operand_def(BufferType)
    semaphore = operand_def(SemaphoreType)
    token = result_def(DmaTokenType)
    stage = prop_def(IntAttr)

    def __init__(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        semaphore: SSAValue | Operation,
        stage: int | IntAttr,
    ):
        super().__init__(
            operands=[source, destination, semaphore],
            result_types=[DmaTokenType()],
            properties={"stage": IntAttr(stage) if isinstance(stage, int) else stage},
        )

    def verify_(self) -> None:
        source, destination = self.source.type, self.destination.type
        assert isinstance(source, BufferType) and isinstance(destination, BufferType)
        if source.space.data is destination.space.data:
            raise VerifyException("DMA source and destination must use different memory spaces")
        if source.storage.get_shape() != destination.storage.get_shape():
            raise VerifyException("DMA source and destination shapes must match")
        if source.storage.element_type != destination.storage.element_type:
            raise VerifyException("DMA source and destination element types must match")
        _check_live(source, self.stage.data)
        _check_live(destination, self.stage.data)


@irdl_op_definition
class DmaWaitOp(IRDLOperation):
    name = "tpu_schedule.dma_wait"
    token = operand_def(DmaTokenType)
    stage = prop_def(IntAttr)

    def __init__(self, token: SSAValue | Operation, stage: int | IntAttr):
        super().__init__(
            operands=[token],
            properties={"stage": IntAttr(stage) if isinstance(stage, int) else stage},
        )

    def verify_(self) -> None:
        owner = self.token.owner
        if not isinstance(owner, DmaStartOp):
            raise VerifyException("DMA wait token must come from tpu_schedule.dma_start")
        if self.stage.data < owner.stage.data:
            raise VerifyException("DMA wait cannot precede its start stage")
        source, destination = owner.source.type, owner.destination.type
        assert isinstance(source, BufferType) and isinstance(destination, BufferType)
        _check_live(source, self.stage.data)
        _check_live(destination, self.stage.data)


@irdl_op_definition
class MxuMatmulOp(IRDLOperation):
    name = "tpu_schedule.mxu_matmul"
    lhs = operand_def(BufferType)
    rhs = operand_def(BufferType)
    accumulator = operand_def(BufferType)
    stage = prop_def(IntAttr)
    tile_m = prop_def(IntAttr)
    tile_k = prop_def(IntAttr)
    tile_n = prop_def(IntAttr)

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        accumulator: SSAValue | Operation,
        stage: int | IntAttr,
        *,
        tile_m: int | None = None,
        tile_k: int | None = None,
        tile_n: int | None = None,
    ):
        lhs_type = SSAValue.get(lhs).type
        rhs_type = SSAValue.get(rhs).type
        assert isinstance(lhs_type, BufferType) and isinstance(rhs_type, BufferType)
        lhs_shape = lhs_type.storage.get_shape()
        rhs_shape = rhs_type.storage.get_shape()
        super().__init__(
            operands=[lhs, rhs, accumulator],
            properties={
                "stage": IntAttr(stage) if isinstance(stage, int) else stage,
                "tile_m": IntAttr(tile_m if tile_m is not None else lhs_shape[0]),
                "tile_k": IntAttr(tile_k if tile_k is not None else lhs_shape[1]),
                "tile_n": IntAttr(tile_n if tile_n is not None else rhs_shape[1]),
            },
        )

    def verify_(self) -> None:
        lhs, rhs, accumulator = self.lhs.type, self.rhs.type, self.accumulator.type
        assert isinstance(lhs, BufferType)
        assert isinstance(rhs, BufferType)
        assert isinstance(accumulator, BufferType)
        for buffer in (lhs, rhs, accumulator):
            if buffer.space.data is not MemorySpace.VMEM:
                raise VerifyException("MXU operands must be resident in VMEM")
            _check_live(buffer, self.stage.data)
        if any(len(buffer.storage.get_shape()) != 2 for buffer in (lhs, rhs, accumulator)):
            raise VerifyException("MXU matmul requires rank-2 buffers")
        m, k = lhs.storage.get_shape()
        rhs_k, n = rhs.storage.get_shape()
        if k != rhs_k or accumulator.storage.get_shape() != (m, n):
            raise VerifyException("MXU matmul shapes must be MxK, KxN, and MxN")
        if not isinstance(lhs.storage.element_type, BFloat16Type | Float16Type):
            raise VerifyException("MXU input buffers must use bf16 or f16")
        if rhs.storage.element_type != lhs.storage.element_type:
            raise VerifyException("MXU input element types must match")
        if not isinstance(accumulator.storage.element_type, Float32Type):
            raise VerifyException("MXU accumulation must use f32")
        tile = (self.tile_m.data, self.tile_k.data, self.tile_n.data)
        if any(size <= 0 for size in tile):
            raise VerifyException("MXU tile dimensions must be positive")
        if m % tile[0] or k % tile[1] or n % tile[2]:
            raise VerifyException("MXU tile dimensions must divide the operand dimensions")
        if tile[1] != k:
            raise VerifyException("the current MXU schedule requires a complete K tile")


@irdl_op_definition
class CollectiveReduceScatterOp(IRDLOperation):
    name = "tpu_schedule.collective_reduce_scatter"
    source = operand_def(BufferType)
    destination = operand_def(BufferType)
    stage = prop_def(IntAttr)
    mesh_axis = prop_def(StringAttr)
    group_size = prop_def(IntAttr)
    scatter_dimension = prop_def(IntAttr)
    reducer = prop_def(StringAttr)

    def __init__(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        *,
        stage: int,
        mesh_axis: str,
        group_size: int,
        scatter_dimension: int,
        reducer: str = "sum",
    ) -> None:
        super().__init__(
            operands=[source, destination],
            properties={
                "stage": IntAttr(stage),
                "mesh_axis": StringAttr(mesh_axis),
                "group_size": IntAttr(group_size),
                "scatter_dimension": IntAttr(scatter_dimension),
                "reducer": StringAttr(reducer),
            },
        )

    def verify_(self) -> None:
        source, destination = self.source.type, self.destination.type
        assert isinstance(source, BufferType) and isinstance(destination, BufferType)
        for buffer in (source, destination):
            if buffer.space.data is not MemorySpace.VMEM:
                raise VerifyException("reduce-scatter buffers must be resident in VMEM")
            _check_live(buffer, self.stage.data)
        if source.storage.element_type != destination.storage.element_type:
            raise VerifyException("reduce-scatter cannot change element type")
        source_shape = source.storage.get_shape()
        destination_shape = destination.storage.get_shape()
        if len(source_shape) != len(destination_shape):
            raise VerifyException("reduce-scatter cannot change rank")
        dimension = self.scatter_dimension.data
        if dimension < 0 or dimension >= len(source_shape):
            raise VerifyException("reduce-scatter dimension is out of range")
        if self.group_size.data <= 0:
            raise VerifyException("reduce-scatter group size must be positive")
        expected = list(source_shape)
        if expected[dimension] % self.group_size.data:
            raise VerifyException("reduce-scatter dimension must divide by the group size")
        expected[dimension] //= self.group_size.data
        if tuple(expected) != destination_shape:
            raise VerifyException("reduce-scatter destination has the wrong local shape")
        if self.reducer.data not in {"sum", "max", "min"}:
            raise VerifyException("unsupported reduce-scatter reducer")
        if not self.mesh_axis.data:
            raise VerifyException("reduce-scatter needs a mesh axis")


@irdl_op_definition
class RaggedPagedAttentionOp(IRDLOperation):
    name = "tpu_schedule.ragged_paged_attention"
    query = operand_def(BufferType)
    key_cache = operand_def(BufferType)
    value_cache = operand_def(BufferType)
    page_table = operand_def(BufferType)
    sequence_lengths = operand_def(BufferType)
    bias = operand_def(BufferType)
    output = operand_def(BufferType)
    stage = prop_def(IntAttr)
    query_block_size = prop_def(IntAttr)
    kv_block_size = prop_def(IntAttr)

    def __init__(
        self,
        query: SSAValue | Operation,
        key_cache: SSAValue | Operation,
        value_cache: SSAValue | Operation,
        page_table: SSAValue | Operation,
        sequence_lengths: SSAValue | Operation,
        bias: SSAValue | Operation,
        output: SSAValue | Operation,
        stage: int,
        query_block_size: int,
        kv_block_size: int,
    ):
        super().__init__(
            operands=[query, key_cache, value_cache, page_table, sequence_lengths, bias, output],
            properties={
                "stage": IntAttr(stage),
                "query_block_size": IntAttr(query_block_size),
                "kv_block_size": IntAttr(kv_block_size),
            },
        )

    def verify_(self) -> None:
        buffers = tuple(value.type for value in self.operands)
        assert all(isinstance(value, BufferType) for value in buffers)
        query, key_cache, value_cache, page_table, lengths, bias, output = buffers
        for buffer in buffers:
            assert isinstance(buffer, BufferType)
            _check_live(buffer, self.stage.data)
        assert isinstance(query, BufferType)
        assert isinstance(key_cache, BufferType)
        assert isinstance(value_cache, BufferType)
        assert isinstance(page_table, BufferType)
        assert isinstance(lengths, BufferType)
        assert isinstance(bias, BufferType)
        assert isinstance(output, BufferType)
        if query.space.data is not MemorySpace.VMEM or output.space.data is not MemorySpace.VMEM:
            raise VerifyException("RPA query and output must be resident in VMEM")
        if bias.space.data is not MemorySpace.VMEM:
            raise VerifyException("RPA relative-position bias must be resident in VMEM")
        if any(
            value.space.data is not MemorySpace.HBM
            for value in (key_cache, value_cache, page_table, lengths)
        ):
            raise VerifyException("RPA cache and page metadata must reside in HBM")
        if key_cache.storage.get_shape() != value_cache.storage.get_shape():
            raise VerifyException("RPA key and value cache shapes must match")
        if query.storage.get_shape() != output.storage.get_shape():
            raise VerifyException("RPA query and output shapes must match")
        if len(query.storage.get_shape()) != 3 or len(key_cache.storage.get_shape()) != 4:
            raise VerifyException(
                "RPA expects query [batch, heads, dim] and cache [pages, page, heads, dim]"
            )
        batch, heads, dimension = query.storage.get_shape()
        _, page_size, cache_heads, cache_dimension = key_cache.storage.get_shape()
        if heads != cache_heads or dimension != cache_dimension:
            raise VerifyException("RPA query and cache head dimensions must match")
        if page_table.storage.get_shape()[0] != batch or lengths.storage.get_shape() != (batch,):
            raise VerifyException("RPA page metadata batch dimensions must match query")
        if bias.storage.get_shape()[0] != heads:
            raise VerifyException("RPA bias head dimension must match query")
        if self.kv_block_size.data <= 0 or page_size % self.kv_block_size.data:
            raise VerifyException("RPA KV block size must be positive and divide page size")
        if self.query_block_size.data <= 0:
            raise VerifyException("RPA query block size must be positive")


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "tpu_schedule.yield"
    traits = traits_def(IsTerminator())

    def __init__(self):
        super().__init__()

    def verify_(self) -> None:
        if not isinstance(self.parent_op(), KernelOp):
            raise VerifyException("tpu_schedule.yield must terminate a TPU kernel")


@irdl_op_definition
class KernelOp(IRDLOperation):
    name = "tpu_schedule.kernel"
    body = region_def("single_block")
    sym_name = prop_def(StringAttr)
    target = prop_def(StringAttr)
    vmem_capacity_bytes = prop_def(IntAttr)
    smem_capacity_bytes = prop_def(IntAttr)
    mesh_axis_names = prop_def(ArrayAttr[StringAttr])
    mesh_axis_sizes = prop_def(ArrayAttr[IntAttr])
    interconnect = prop_def(InterconnectAttr)
    dma_engine_count = prop_def(IntAttr)
    mxu_count = prop_def(IntAttr)
    vector_unit_count = prop_def(IntAttr)
    ici_link_count = prop_def(IntAttr)
    traits = traits_def(IsolatedFromAbove())

    def __init__(
        self,
        sym_name: str | StringAttr,
        target: str | StringAttr,
        vmem_capacity_bytes: int | IntAttr,
        smem_capacity_bytes: int | IntAttr,
        mesh_axis_names: ArrayAttr[StringAttr],
        mesh_axis_sizes: ArrayAttr[IntAttr],
        body: Region,
        *,
        interconnect_bandwidth_bytes_per_second: dict[str, int] | None = None,
        dma_engine_count: int = 2,
        mxu_count: int = 1,
        vector_unit_count: int = 1,
        ici_link_count: int = 1,
    ):
        interconnect_bandwidth_bytes_per_second = dict(
            sorted((interconnect_bandwidth_bytes_per_second or {}).items())
        )
        super().__init__(
            properties={
                "sym_name": StringAttr(sym_name) if isinstance(sym_name, str) else sym_name,
                "target": StringAttr(target) if isinstance(target, str) else target,
                "vmem_capacity_bytes": IntAttr(vmem_capacity_bytes)
                if isinstance(vmem_capacity_bytes, int)
                else vmem_capacity_bytes,
                "smem_capacity_bytes": IntAttr(smem_capacity_bytes)
                if isinstance(smem_capacity_bytes, int)
                else smem_capacity_bytes,
                "mesh_axis_names": mesh_axis_names,
                "mesh_axis_sizes": mesh_axis_sizes,
                "interconnect": InterconnectAttr(
                    ArrayAttr(
                        StringAttr(axis)
                        for axis in interconnect_bandwidth_bytes_per_second
                    ),
                    ArrayAttr(
                        IntAttr(bandwidth)
                        for bandwidth in interconnect_bandwidth_bytes_per_second.values()
                    ),
                ),
                "dma_engine_count": IntAttr(dma_engine_count),
                "mxu_count": IntAttr(mxu_count),
                "vector_unit_count": IntAttr(vector_unit_count),
                "ici_link_count": IntAttr(ici_link_count),
            },
            regions=[body],
        )

    def verify_(self) -> None:
        block = self.body.block
        mesh_names = tuple(value.data for value in self.mesh_axis_names)
        mesh_sizes = tuple(value.data for value in self.mesh_axis_sizes)
        if len(mesh_names) != len(mesh_sizes):
            raise VerifyException("kernel mesh axis names and sizes must have equal length")
        if mesh_names != tuple(sorted(mesh_names)) or len(mesh_names) != len(set(mesh_names)):
            raise VerifyException("kernel mesh axes must be unique and canonically ordered")
        if any(size <= 0 for size in mesh_sizes):
            raise VerifyException("kernel mesh axis sizes must be positive")
        resource_capacities = {
            "DMA": self.dma_engine_count.data,
            "MXU": self.mxu_count.data,
            "vector": self.vector_unit_count.data,
            "ICI": self.ici_link_count.data,
        }
        if any(capacity <= 0 for capacity in resource_capacities.values()):
            raise VerifyException("kernel hardware resource capacities must be positive")
        mesh = dict(zip(mesh_names, mesh_sizes, strict=True))
        links = self.interconnect.bandwidths()
        if set(links) != set(mesh):
            raise VerifyException(
                "kernel interconnect must declare one bandwidth for every mesh axis"
            )
        if not isinstance(block.last_op, YieldOp):
            raise VerifyException("kernel must end with tpu_schedule.yield")
        operations = list(block.ops)
        positions = {operation: index for index, operation in enumerate(operations)}
        previous_stage = -1
        in_flight: dict[Operation, DmaStartOp] = {}
        symbols: dict[str, int] = {}
        buffers: list[BufferType] = []
        storage_buffers: list[BufferType] = []
        views_by_root: dict[SSAValue, list[tuple[ViewOp, TileRegionAttr]]] = {}
        initialized: set[SSAValue] = set(block.args)
        pending_dma_destinations: dict[Operation, tuple[SSAValue, TileRegionAttr]] = {}

        def root_region(value: SSAValue) -> tuple[SSAValue, TileRegionAttr]:
            chain: list[ViewOp] = []
            while isinstance(value.owner, ViewOp):
                chain.append(value.owner)
                value = value.owner.base
            shape = value.type.storage.get_shape()
            offsets = [0] * len(shape)
            sizes = list(shape)
            strides = [1] * len(shape)
            for view in reversed(chain):
                offsets = [
                    parent_offset + child_offset * parent_stride
                    for parent_offset, child_offset, parent_stride in zip(
                        offsets,
                        (item.data for item in view.region.offsets),
                        strides,
                        strict=True,
                    )
                ]
                strides = [
                    parent_stride * child_stride
                    for parent_stride, child_stride in zip(
                        strides,
                        (item.data for item in view.region.strides),
                        strict=True,
                    )
                ]
                sizes = [item.data for item in view.region.sizes]
            return value, TileRegionAttr(
                ArrayAttr(IntAttr(item) for item in offsets),
                ArrayAttr(IntAttr(item) for item in sizes),
                ArrayAttr(IntAttr(item) for item in strides),
            )

        def root(value: SSAValue) -> SSAValue:
            return root_region(value)[0]

        def require_initialized(value: SSAValue, operation: Operation) -> None:
            base = root(value)
            if base not in initialized:
                raise VerifyException(
                    f"{operation.name} reads a buffer before its producing operation completes"
                )
            if any(base == pending[0] for pending in pending_dma_destinations.values()):
                raise VerifyException(f"{operation.name} reads a buffer while DMA is in flight")

        for argument in block.args:
            if not isinstance(argument.type, BufferType):
                raise VerifyException("kernel arguments must be tpu_schedule buffers")
            if argument.type.space.data is not MemorySpace.HBM:
                raise VerifyException("kernel arguments must reside in HBM")
            if argument.type.ownership.data is not Ownership.EXTERNAL:
                raise VerifyException("kernel arguments must have external ownership")
            buffers.append(argument.type)
            storage_buffers.append(argument.type)

        for operation in operations:
            stage = _stage(operation)
            if stage is not None:
                if stage < previous_stage:
                    raise VerifyException("scheduled stages must be monotonic")
                previous_stage = stage
            if isinstance(operation, AllocOp):
                buffers.append(operation.buffer.type)
                storage_buffers.append(operation.buffer.type)
            if isinstance(operation, ViewOp):
                operation.verify_()
                buffers.append(operation.view.type)
                base, normalized_region = root_region(operation.view)
                for other, other_region in views_by_root.setdefault(base, []):
                    if (
                        _lifetimes_overlap(operation.view.type, other.view.type)
                        and _regions_overlap(normalized_region, other_region)
                        and operation.alias_group.data != other.alias_group.data
                    ):
                        raise VerifyException(
                            "overlapping live tile views must declare the same alias group"
                        )
                views_by_root[base].append((operation, normalized_region))
            if isinstance(operation, DmaStartOp):
                require_initialized(operation.source, operation)
                destination_root, destination_region = root_region(operation.destination)
                for other_root, other_region in pending_dma_destinations.values():
                    if destination_root == other_root and _regions_overlap(
                        destination_region, other_region
                    ):
                        raise VerifyException(
                            "concurrent DMA writes target overlapping buffer regions"
                        )
                semaphore_owner = operation.semaphore.owner
                if semaphore_owner in in_flight:
                    raise VerifyException("semaphore reused before its DMA was waited on")
                in_flight[semaphore_owner] = operation
                uses = list(operation.token.uses)
                if len(uses) != 1 or not isinstance(uses[0].operation, DmaWaitOp):
                    raise VerifyException("every DMA token must have exactly one DMA wait")
                if positions[uses[0].operation] <= positions[operation]:
                    raise VerifyException("DMA wait must occur after DMA start")
                pending_dma_destinations[semaphore_owner] = (
                    destination_root,
                    destination_region,
                )
            if isinstance(operation, DmaWaitOp):
                start = operation.token.owner
                assert isinstance(start, DmaStartOp)
                in_flight.pop(start.semaphore.owner, None)
                destination, _ = pending_dma_destinations.pop(start.semaphore.owner)
                initialized.add(destination)
            if isinstance(operation, MxuMatmulOp):
                require_initialized(operation.lhs, operation)
                require_initialized(operation.rhs, operation)
                initialized.add(root(operation.accumulator))
            if isinstance(operation, CollectiveReduceScatterOp):
                require_initialized(operation.source, operation)
                axis = operation.mesh_axis.data
                if axis not in mesh:
                    raise VerifyException(f"reduce-scatter references unknown mesh axis {axis}")
                if operation.group_size.data != mesh[axis]:
                    raise VerifyException(
                        "reduce-scatter group size must match its kernel mesh axis"
                    )
                if links[axis] <= 0:
                    raise VerifyException("reduce-scatter requires a usable interconnect link")
                initialized.add(root(operation.destination))
            if isinstance(operation, RaggedPagedAttentionOp):
                for value in (
                    operation.query,
                    operation.key_cache,
                    operation.value_cache,
                    operation.page_table,
                    operation.sequence_lengths,
                    operation.bias,
                ):
                    require_initialized(value, operation)
                initialized.add(root(operation.output))

        for buffer in buffers:
            buffer.verify()
            for symbol, size, sharding in zip(
                (value.data for value in buffer.shape.dimensions),
                buffer.storage.get_shape(),
                (value.data for value in buffer.sharding.axes),
                strict=True,
            ):
                if symbol.isdecimal():
                    continue
                shard_count = 1
                for axis in filter(None, sharding.split("/")):
                    if axis not in mesh:
                        raise VerifyException(f"buffer references unknown mesh axis {axis}")
                    shard_count *= mesh[axis]
                global_size = size * shard_count
                previous = symbols.setdefault(symbol, global_size)
                if previous != global_size:
                    raise VerifyException(
                        f"symbolic dimension {symbol} has conflicting global sizes "
                        f"{previous} and {global_size}"
                    )

        if in_flight:
            raise VerifyException("kernel ends with DMA operations still in flight")
        local_buffers = [
            buffer
            for buffer in storage_buffers
            if buffer.ownership.data is not Ownership.EXTERNAL
        ]
        max_operation_stage = max((_stage(operation) or 0 for operation in operations), default=0)
        max_lifetime_stage = max(
            (buffer.lifetime.end.data for buffer in local_buffers),
            default=0,
        )
        max_stage = max(max_operation_stage, max_lifetime_stage)
        for stage in range(max_stage + 1):
            active_dma = sum(
                start.stage.data <= stage <= wait.operation.stage.data
                for start in (op for op in operations if isinstance(op, DmaStartOp))
                for wait in start.token.uses
                if isinstance(wait.operation, DmaWaitOp)
            )
            mxu_uses = sum(
                isinstance(operation, (MxuMatmulOp, RaggedPagedAttentionOp))
                and operation.stage.data == stage
                for operation in operations
            )
            ici_uses = sum(
                isinstance(operation, CollectiveReduceScatterOp)
                and operation.stage.data == stage
                for operation in operations
            )
            if active_dma > self.dma_engine_count.data:
                raise VerifyException(
                    f"DMA engine capacity exceeded at stage {stage}: "
                    f"{active_dma} > {self.dma_engine_count.data}"
                )
            if mxu_uses > self.mxu_count.data:
                raise VerifyException(
                    f"MXU capacity exceeded at stage {stage}: {mxu_uses} > {self.mxu_count.data}"
                )
            if ici_uses > self.ici_link_count.data:
                raise VerifyException(
                    f"ICI link capacity exceeded at stage {stage}: "
                    f"{ici_uses} > {self.ici_link_count.data}"
                )
            for space, capacity, label in (
                (MemorySpace.VMEM, self.vmem_capacity_bytes.data, "VMEM"),
                (MemorySpace.SMEM, self.smem_capacity_bytes.data, "SMEM"),
            ):
                live_bytes = sum(
                    buffer_bytes(buffer)
                    for buffer in local_buffers
                    if buffer.space.data is space
                    and buffer.lifetime.start.data <= stage <= buffer.lifetime.end.data
                )
                if live_bytes > capacity:
                    raise VerifyException(
                        f"{label} capacity exceeded at stage {stage}: {live_bytes} > {capacity}"
                    )


TPUSchedule = Dialect(
    "tpu_schedule",
    [
        KernelOp,
        AllocOp,
        ViewOp,
        SemaphoreAllocOp,
        DmaStartOp,
        DmaWaitOp,
        MxuMatmulOp,
        CollectiveReduceScatterOp,
        RaggedPagedAttentionOp,
        YieldOp,
    ],
    [
        MemorySpaceAttr,
        OwnershipAttr,
        ShapeAttr,
        ShardingAttr,
        LayoutAttr,
        LifetimeAttr,
        TileRegionAttr,
        InterconnectAttr,
        BufferType,
        DmaTokenType,
        SemaphoreType,
    ],
)
