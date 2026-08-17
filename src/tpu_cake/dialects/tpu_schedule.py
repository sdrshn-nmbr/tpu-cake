from __future__ import annotations

import math
from enum import StrEnum
from itertools import pairwise, product

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
    Block,
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

from tpu_cake.source import source_aware_error


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
class DeviceAttr(ParametrizedAttribute):
    name = "tpu_schedule.device"
    device_id: IntAttr
    coordinates: ArrayAttr[IntAttr]
    core_count: IntAttr

    def verify(self) -> None:
        if self.device_id.data < 0:
            raise VerifyException("topology device IDs must be nonnegative")
        if self.core_count.data <= 0:
            raise VerifyException("topology device core counts must be positive")
        if any(value.data < 0 for value in self.coordinates):
            raise VerifyException("topology device coordinates must be nonnegative")


@irdl_attr_definition
class LinkAttr(ParametrizedAttribute):
    name = "tpu_schedule.link"
    link_id: StringAttr
    source_device: IntAttr
    destination_device: IntAttr
    bandwidth_bytes_per_second: IntAttr
    channel_count: IntAttr

    def verify(self) -> None:
        if not self.link_id.data:
            raise VerifyException("topology links need stable IDs")
        if self.source_device.data < 0 or self.destination_device.data < 0:
            raise VerifyException("topology link endpoints must be nonnegative")
        if self.source_device.data >= self.destination_device.data:
            raise VerifyException("topology link endpoints must use canonical ascending order")
        if self.bandwidth_bytes_per_second.data <= 0:
            raise VerifyException("topology link bandwidth must be positive")
        if self.channel_count.data <= 0:
            raise VerifyException("topology link channel count must be positive")


@irdl_attr_definition
class CollectiveGroupAttr(ParametrizedAttribute):
    name = "tpu_schedule.collective_group"
    group_id: StringAttr
    device_ids: ArrayAttr[IntAttr]
    route_link_ids: ArrayAttr[StringAttr]

    def verify(self) -> None:
        devices = tuple(value.data for value in self.device_ids)
        routes = tuple(value.data for value in self.route_link_ids)
        if not self.group_id.data:
            raise VerifyException("collective groups need stable IDs")
        if len(devices) < 2 or devices != tuple(sorted(set(devices))):
            raise VerifyException(
                "collective group devices must be unique, ordered, and contain at least two devices"
            )
        if not routes or routes != tuple(sorted(set(routes))):
            raise VerifyException("collective route links must be unique and ordered")


@irdl_attr_definition
class CollectivePlanAttr(ParametrizedAttribute):
    name = "tpu_schedule.collective_plan"
    plan_id: StringAttr
    mesh_axis: StringAttr
    groups: ArrayAttr[CollectiveGroupAttr]

    def verify(self) -> None:
        groups = tuple(self.groups)
        if not self.plan_id.data or not self.mesh_axis.data:
            raise VerifyException("collective plans need stable IDs and mesh axes")
        for group in groups:
            group.verify()
        group_ids = tuple(group.group_id.data for group in groups)
        if not groups or group_ids != tuple(sorted(set(group_ids))):
            raise VerifyException("collective plan groups must be unique and ordered")


@irdl_attr_definition
class TopologyAttr(ParametrizedAttribute):
    name = "tpu_schedule.topology"
    devices: ArrayAttr[DeviceAttr]
    links: ArrayAttr[LinkAttr]
    collective_plans: ArrayAttr[CollectivePlanAttr]

    def verify(self) -> None:
        devices = tuple(self.devices)
        links = tuple(self.links)
        plans = tuple(self.collective_plans)
        for device in devices:
            device.verify()
        for link in links:
            link.verify()
        for plan in plans:
            plan.verify()

        device_ids = tuple(device.device_id.data for device in devices)
        if device_ids != tuple(range(len(devices))):
            raise VerifyException("topology device IDs must be dense and canonically ordered")
        coordinate_rank = len(devices[0].coordinates) if devices else 0
        coordinates = tuple(
            tuple(value.data for value in device.coordinates) for device in devices
        )
        if any(len(device.coordinates) != coordinate_rank for device in devices):
            raise VerifyException("topology device coordinates must have equal rank")
        if len(coordinates) != len(set(coordinates)):
            raise VerifyException("topology device coordinates must be unique")

        link_ids = tuple(link.link_id.data for link in links)
        if link_ids != tuple(sorted(set(link_ids))):
            raise VerifyException("topology links must be unique and canonically ordered")
        known_devices = set(device_ids)
        endpoints: set[tuple[int, int]] = set()
        links_by_id: dict[str, LinkAttr] = {}
        for link in links:
            endpoint = (link.source_device.data, link.destination_device.data)
            if not set(endpoint) <= known_devices:
                raise VerifyException("topology link references an unknown device")
            if endpoint in endpoints:
                raise VerifyException("topology cannot declare duplicate physical links")
            endpoints.add(endpoint)
            links_by_id[link.link_id.data] = link

        plan_ids = tuple(plan.plan_id.data for plan in plans)
        if plan_ids != tuple(sorted(set(plan_ids))):
            raise VerifyException(
                "topology collective plans must be unique and canonically ordered"
            )
        for plan in plans:
            covered: set[int] = set()
            group_size: int | None = None
            for group in plan.groups:
                group_devices = {value.data for value in group.device_ids}
                if not group_devices <= known_devices or covered & group_devices:
                    raise VerifyException(
                        "collective plan groups must be disjoint and reference known devices"
                    )
                covered |= group_devices
                if group_size is None:
                    group_size = len(group_devices)
                elif len(group_devices) != group_size:
                    raise VerifyException("collective plan groups must have equal size")
                adjacency = {device: set() for device in group_devices}
                for route in group.route_link_ids:
                    link = links_by_id.get(route.data)
                    if link is None:
                        raise VerifyException("collective route references an unknown link")
                    source = link.source_device.data
                    destination = link.destination_device.data
                    if source not in group_devices or destination not in group_devices:
                        raise VerifyException(
                            "collective route link endpoints must stay inside their group"
                        )
                    adjacency[source].add(destination)
                    adjacency[destination].add(source)
                reached = {next(iter(group_devices))}
                frontier = list(reached)
                while frontier:
                    current = frontier.pop()
                    for neighbor in adjacency[current] - reached:
                        reached.add(neighbor)
                        frontier.append(neighbor)
                if reached != group_devices:
                    raise VerifyException("collective route must connect every group device")
            if covered != known_devices:
                raise VerifyException("collective plan groups must partition all devices")

    def plans_by_id(self) -> dict[str, CollectivePlanAttr]:
        self.verify()
        return {plan.plan_id.data: plan for plan in self.collective_plans}

    def links_by_id(self) -> dict[str, LinkAttr]:
        self.verify()
        return {link.link_id.data: link for link in self.links}


def rectilinear_topology(
    mesh_axis_names: tuple[str, ...],
    mesh_axis_sizes: tuple[int, ...],
    bandwidth_bytes_per_second: dict[str, int],
    *,
    cores_per_device: int = 2,
    channels_per_link: int = 1,
) -> TopologyAttr:
    coordinates = tuple(product(*(range(size) for size in mesh_axis_sizes)))
    coordinate_to_device = {
        coordinate: device_id for device_id, coordinate in enumerate(coordinates)
    }
    devices = tuple(
        DeviceAttr(
            IntAttr(device_id),
            ArrayAttr(IntAttr(value) for value in coordinate),
            IntAttr(cores_per_device),
        )
        for device_id, coordinate in enumerate(coordinates)
    )
    links: dict[tuple[int, int], LinkAttr] = {}
    plans: list[CollectivePlanAttr] = []
    for axis_index, axis in enumerate(mesh_axis_names):
        if mesh_axis_sizes[axis_index] == 1:
            continue
        groups: list[CollectiveGroupAttr] = []
        other_indices = tuple(
            index for index in range(len(mesh_axis_names)) if index != axis_index
        )
        fixed_coordinates = product(
            *(range(mesh_axis_sizes[index]) for index in other_indices)
        )
        for group_index, fixed in enumerate(fixed_coordinates):
            base = dict(zip(other_indices, fixed, strict=True))
            group_coordinates = tuple(
                tuple(
                    coordinate if index == axis_index else base[index]
                    for index in range(len(mesh_axis_names))
                )
                for coordinate in range(mesh_axis_sizes[axis_index])
            )
            device_ids = tuple(
                coordinate_to_device[coordinate] for coordinate in group_coordinates
            )
            route_ids: list[str] = []
            for source, destination in pairwise(device_ids):
                endpoint = (min(source, destination), max(source, destination))
                link_id = f"link:{endpoint[0]}-{endpoint[1]}"
                existing = links.get(endpoint)
                bandwidth = bandwidth_bytes_per_second[axis]
                if existing is not None and (
                    existing.bandwidth_bytes_per_second.data != bandwidth
                    or existing.channel_count.data != channels_per_link
                ):
                    raise ValueError("one physical link cannot have conflicting capacities")
                links[endpoint] = existing or LinkAttr(
                    StringAttr(link_id),
                    IntAttr(endpoint[0]),
                    IntAttr(endpoint[1]),
                    IntAttr(bandwidth),
                    IntAttr(channels_per_link),
                )
                route_ids.append(link_id)
            groups.append(
                CollectiveGroupAttr(
                    StringAttr(f"group:{axis}:{group_index}"),
                    ArrayAttr(IntAttr(device_id) for device_id in sorted(device_ids)),
                    ArrayAttr(StringAttr(link_id) for link_id in sorted(route_ids)),
                )
            )
        plans.append(
            CollectivePlanAttr(
                StringAttr(f"axis:{axis}"),
                StringAttr(axis),
                ArrayAttr(sorted(groups, key=lambda group: group.group_id.data)),
            )
        )
    topology = TopologyAttr(
        ArrayAttr(devices),
        ArrayAttr(sorted(links.values(), key=lambda link: link.link_id.data)),
        ArrayAttr(sorted(plans, key=lambda plan: plan.plan_id.data)),
    )
    topology.verify()
    return topology


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
    slots = opt_prop_def(IntAttr)

    def __init__(self, slots: int = 1):
        super().__init__(
            result_types=[SemaphoreType()],
            properties={"slots": IntAttr(slots)},
        )

    def verify_(self) -> None:
        if self.slot_count <= 0:
            raise VerifyException("semaphore slot count must be positive")

    @property
    def slot_count(self) -> int:
        return 1 if self.slots is None else self.slots.data


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
    collective_plan = opt_prop_def(StringAttr)

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
        collective_plan: str | None = None,
    ) -> None:
        super().__init__(
            operands=[source, destination],
            properties={
                "stage": IntAttr(stage),
                "mesh_axis": StringAttr(mesh_axis),
                "group_size": IntAttr(group_size),
                "scatter_dimension": IntAttr(scatter_dimension),
                "reducer": StringAttr(reducer),
                "collective_plan": StringAttr(
                    collective_plan or f"axis:{mesh_axis}"
                ),
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
class PipelineYieldOp(IRDLOperation):
    name = "tpu_schedule.pipeline_yield"
    values = var_operand_def(BufferType)
    traits = traits_def(IsTerminator())

    def __init__(self, *values: SSAValue | Operation):
        super().__init__(operands=[list(values)])

    def verify_(self) -> None:
        if not isinstance(self.parent_op(), PipelineLoopOp):
            raise VerifyException(
                "tpu_schedule.pipeline_yield must terminate a pipeline loop"
            )


@irdl_op_definition
class PipelineLoopOp(IRDLOperation):
    name = "tpu_schedule.pipeline_loop"
    captures = var_operand_def(BufferType)
    outputs = var_result_def(BufferType)
    body = region_def("single_block")
    trip_count = prop_def(IntAttr)
    initiation_interval = prop_def(IntAttr)
    pipeline_stages = prop_def(IntAttr)
    rotation_counts = prop_def(ArrayAttr[IntAttr])

    def __init__(
        self,
        captures: tuple[SSAValue | Operation, ...],
        body: Region | Block,
        *,
        trip_count: int,
        initiation_interval: int,
        pipeline_stages: int,
        rotation_counts: tuple[int, ...] | None = None,
    ) -> None:
        if isinstance(body, Block):
            body = Region(body)
        capture_types = [SSAValue.get(value).type for value in captures]
        rotations = rotation_counts or (1,) * len(captures)
        super().__init__(
            operands=[list(captures)],
            result_types=[capture_types],
            regions=[body],
            properties={
                "trip_count": IntAttr(trip_count),
                "initiation_interval": IntAttr(initiation_interval),
                "pipeline_stages": IntAttr(pipeline_stages),
                "rotation_counts": ArrayAttr(IntAttr(value) for value in rotations),
            },
        )

    def verify_(self) -> None:
        if (
            self.trip_count.data <= 0
            or self.initiation_interval.data <= 0
            or self.pipeline_stages.data <= 0
        ):
            raise VerifyException(
                "pipeline trip count, initiation interval, and stages must be positive"
            )
        block = self.body.block
        terminator = block.last_op
        if not isinstance(terminator, PipelineYieldOp):
            raise VerifyException("pipeline loop must end with tpu_schedule.pipeline_yield")
        rotations = tuple(value.data for value in self.rotation_counts)
        if len(rotations) != len(self.captures) or any(value <= 0 for value in rotations):
            raise VerifyException(
                "pipeline rotation counts must be positive and match captured buffers"
            )
        if len(block.args) != len(self.captures):
            raise VerifyException("pipeline block arguments must match captured buffers")
        if len(self.outputs) != len(self.captures) or len(terminator.values) != len(
            self.captures
        ):
            raise VerifyException("pipeline results and yields must match captured buffers")
        for capture, argument, result, yielded in zip(
            self.captures,
            block.args,
            self.outputs,
            terminator.values,
            strict=True,
        ):
            if not (
                capture.type == argument.type == result.type == yielded.type
            ):
                raise VerifyException(
                    "pipeline captures, block arguments, yields, and results must have equal types"
                )

        operations = list(block.ops)
        scheduled = [operation for operation in operations if _stage(operation) is not None]
        if any(operation.regions for operation in operations):
            raise VerifyException("nested pipeline regions are not supported")
        allowed_unscheduled = (ViewOp, SemaphoreAllocOp, PipelineYieldOp)
        if any(
            _stage(operation) is None and not isinstance(operation, allowed_unscheduled)
            for operation in operations
        ):
            raise VerifyException("pipeline body contains an unscheduled operation")
        stages = tuple(_stage(operation) for operation in scheduled)
        if stages != tuple(sorted(stages)):
            raise VerifyException("pipeline body stages must be monotonic")
        if any(
            stage is None or stage < 0 or stage >= self.pipeline_stages.data
            for stage in stages
        ):
            raise VerifyException("pipeline operation stage is outside the declared pipeline")

        positions = {operation: index for index, operation in enumerate(operations)}
        initialized: set[SSAValue] = set(block.args)
        pending_dma: dict[Operation, DmaStartOp] = {}

        def root(value: SSAValue) -> SSAValue:
            while isinstance(value.owner, ViewOp):
                value = value.owner.base
            return value

        written: set[SSAValue] = set()
        for operation in operations:
            if isinstance(operation, DmaStartOp):
                if root(operation.source) not in initialized:
                    raise VerifyException("pipeline DMA reads an uninitialized source")
                semaphore = operation.semaphore.owner
                if semaphore in pending_dma:
                    raise VerifyException("pipeline semaphore is reused before its wait")
                uses = list(operation.token.uses)
                if len(uses) != 1 or not isinstance(uses[0].operation, DmaWaitOp):
                    raise VerifyException("pipeline DMA token must have exactly one wait")
                if positions[uses[0].operation] <= positions[operation]:
                    raise VerifyException("pipeline DMA wait must follow its start")
                pending_dma[semaphore] = operation
                written.add(root(operation.destination))
            elif isinstance(operation, DmaWaitOp):
                start = operation.token.owner
                assert isinstance(start, DmaStartOp)
                pending_dma.pop(start.semaphore.owner, None)
                initialized.add(root(start.destination))
            elif isinstance(operation, MxuMatmulOp):
                if any(root(value) not in initialized for value in (operation.lhs, operation.rhs)):
                    raise VerifyException("pipeline MXU reads an uninitialized operand")
                initialized.add(root(operation.accumulator))
                written.add(root(operation.accumulator))
            elif isinstance(operation, CollectiveReduceScatterOp):
                if root(operation.source) not in initialized:
                    raise VerifyException("pipeline collective reads an uninitialized source")
                initialized.add(root(operation.destination))
                written.add(root(operation.destination))
            elif isinstance(operation, RaggedPagedAttentionOp):
                if any(root(value) not in initialized for value in operation.operands[:-1]):
                    raise VerifyException("pipeline RPA reads an uninitialized operand")
                initialized.add(root(operation.output))
                written.add(root(operation.output))
        if pending_dma:
            raise VerifyException("pipeline iteration ends with DMA operations in flight")
        if any(root(value) not in initialized for value in terminator.values):
            raise VerifyException("pipeline yields an uninitialized buffer")

        for index, (argument, rotation_count) in enumerate(
            zip(block.args, rotations, strict=True)
        ):
            if root(argument) not in written:
                continue
            buffer = argument.type
            assert isinstance(buffer, BufferType)
            live_stages = buffer.lifetime.end.data - buffer.lifetime.start.data + 1
            required = math.ceil(live_stages / self.initiation_interval.data)
            if rotation_count < required:
                raise VerifyException(
                    f"pipeline capture {index} needs {required} rotating buffers, "
                    f"but declares {rotation_count}"
                )

        kernel = self.parent_op()
        if not isinstance(kernel, KernelOp):
            raise VerifyException("pipeline loop must be directly contained by a TPU kernel")
        topology_plans: dict[str, CollectivePlanAttr] = {}
        topology_links: dict[str, LinkAttr] = {}
        if kernel.physical_schema is not None:
            if kernel.topology is None:
                raise VerifyException("structured pipeline kernel needs a topology")
            topology_plans = kernel.topology.plans_by_id()
            topology_links = kernel.topology.links_by_id()
            mesh = dict(
                zip(
                    (value.data for value in kernel.mesh_axis_names),
                    (value.data for value in kernel.mesh_axis_sizes),
                    strict=True,
                )
            )
            for operation in scheduled:
                if not isinstance(operation, CollectiveReduceScatterOp):
                    continue
                if operation.collective_plan is None:
                    raise VerifyException(
                        "structured pipeline collective needs a collective plan"
                    )
                plan = topology_plans.get(operation.collective_plan.data)
                if (
                    plan is None
                    or plan.mesh_axis.data != operation.mesh_axis.data
                    or operation.group_size.data != mesh.get(operation.mesh_axis.data)
                ):
                    raise VerifyException(
                        "pipeline collective references an incompatible collective plan"
                    )
        horizon = (
            (self.trip_count.data - 1) * self.initiation_interval.data
            + self.pipeline_stages.data
        )
        for absolute_stage in range(horizon):
            active_dma = 0
            mxu_uses = 0
            ici_uses = 0
            link_uses: dict[str, int] = {}
            semaphore_uses: dict[Operation, int] = {}
            for iteration in range(self.trip_count.data):
                logical_stage = absolute_stage - iteration * self.initiation_interval.data
                if logical_stage < 0 or logical_stage >= self.pipeline_stages.data:
                    continue
                for operation in scheduled:
                    if isinstance(operation, DmaStartOp):
                        wait = next(
                            use.operation
                            for use in operation.token.uses
                            if isinstance(use.operation, DmaWaitOp)
                        )
                        if operation.stage.data <= logical_stage <= wait.stage.data:
                            active_dma += 1
                            owner = operation.semaphore.owner
                            semaphore_uses[owner] = semaphore_uses.get(owner, 0) + 1
                    elif isinstance(operation, (MxuMatmulOp, RaggedPagedAttentionOp)):
                        mxu_uses += operation.stage.data == logical_stage
                    elif isinstance(operation, CollectiveReduceScatterOp):
                        if operation.stage.data == logical_stage:
                            ici_uses += 1
                            if operation.collective_plan is not None:
                                plan = topology_plans.get(
                                    operation.collective_plan.data
                                )
                                if plan is not None:
                                    for group in plan.groups:
                                        for link_id in group.route_link_ids:
                                            link_uses[link_id.data] = (
                                                link_uses.get(link_id.data, 0) + 1
                                            )
            if active_dma > kernel.dma_engine_count.data:
                raise VerifyException(
                    f"pipeline exceeds DMA capacity at absolute stage {absolute_stage}"
                )
            if mxu_uses > kernel.mxu_count.data:
                raise VerifyException(
                    f"pipeline exceeds MXU capacity at absolute stage {absolute_stage}"
                )
            if ici_uses > kernel.ici_link_count.data:
                raise VerifyException(
                    f"pipeline exceeds ICI capacity at absolute stage {absolute_stage}"
                )
            for link_id, uses in link_uses.items():
                capacity = topology_links[link_id].channel_count.data
                if uses > capacity:
                    raise VerifyException(
                        f"pipeline exceeds topology link {link_id} capacity "
                        f"at absolute stage {absolute_stage}"
                    )
            for owner, uses in semaphore_uses.items():
                if not isinstance(owner, SemaphoreAllocOp) or uses > owner.slot_count:
                    raise VerifyException(
                        f"pipeline exceeds semaphore slots at absolute stage {absolute_stage}"
                    )


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
    interconnect = opt_prop_def(InterconnectAttr)
    topology = opt_prop_def(TopologyAttr)
    physical_schema = opt_prop_def(StringAttr)
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
        topology: TopologyAttr | None = None,
        dma_engine_count: int = 2,
        mxu_count: int = 1,
        vector_unit_count: int = 1,
        ici_link_count: int = 1,
    ):
        interconnect_bandwidth_bytes_per_second = dict(
            sorted((interconnect_bandwidth_bytes_per_second or {}).items())
        )
        mesh_names = tuple(value.data for value in mesh_axis_names)
        mesh_sizes = tuple(value.data for value in mesh_axis_sizes)
        topology = topology or rectilinear_topology(
            mesh_names,
            mesh_sizes,
            interconnect_bandwidth_bytes_per_second,
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
                "topology": topology,
                "physical_schema": StringAttr("structured-topology-v2"),
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
        if self.physical_schema is None:
            if self.topology is not None or self.interconnect is None:
                raise VerifyException(
                    "legacy kernels require only their axis-bandwidth interconnect"
                )
            links = self.interconnect.bandwidths()
            if set(links) != set(mesh):
                raise VerifyException(
                    "kernel interconnect must declare one bandwidth for every mesh axis"
                )
            topology_plans: dict[str, CollectivePlanAttr] = {}
            topology_links: dict[str, LinkAttr] = {}
        else:
            if self.physical_schema.data != "structured-topology-v2":
                raise VerifyException("unsupported physical schedule schema")
            if self.topology is None or self.interconnect is not None:
                raise VerifyException(
                    "structured kernels require topology and must not duplicate legacy interconnect"
                )
            self.topology.verify()
            topology_plans = self.topology.plans_by_id()
            topology_links = self.topology.links_by_id()
            if len(self.topology.devices) != math.prod(mesh_sizes):
                raise VerifyException("topology device count must match the kernel mesh")
            coordinates = {
                tuple(value.data for value in device.coordinates)
                for device in self.topology.devices
            }
            expected_coordinates = set(
                product(*(range(size) for size in mesh_sizes))
            )
            if coordinates != expected_coordinates:
                raise VerifyException("topology coordinates must exactly cover the kernel mesh")
            plans_by_axis = {
                plan.mesh_axis.data: plan for plan in self.topology.collective_plans
            }
            expected_plan_axes = {
                axis for axis, size in mesh.items() if size > 1
            }
            if set(plans_by_axis) != expected_plan_axes:
                raise VerifyException(
                    "topology must declare one collective plan for every nontrivial mesh axis"
                )
            for axis, plan in plans_by_axis.items():
                if any(len(group.device_ids) != mesh[axis] for group in plan.groups):
                    raise VerifyException(
                        "collective plan group size must match its mesh axis"
                    )
                axis_index = mesh_names.index(axis)
                coordinates_by_id = {
                    device.device_id.data: tuple(
                        value.data for value in device.coordinates
                    )
                    for device in self.topology.devices
                }
                for group in plan.groups:
                    group_coordinates = [
                        coordinates_by_id[value.data] for value in group.device_ids
                    ]
                    fixed = {
                        tuple(
                            coordinate[index]
                            for index in range(len(mesh_names))
                            if index != axis_index
                        )
                        for coordinate in group_coordinates
                    }
                    varying = {
                        coordinate[axis_index] for coordinate in group_coordinates
                    }
                    if len(fixed) != 1 or varying != set(range(mesh[axis])):
                        raise VerifyException(
                            "collective plan groups must follow their declared mesh axis"
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
        rotation_copies: list[tuple[BufferType, int]] = []

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
                for semaphore_owner, (
                    other_root,
                    other_region,
                ) in pending_dma_destinations.items():
                    if destination_root == other_root and _regions_overlap(
                        destination_region, other_region
                    ):
                        raise source_aware_error(
                            "concurrent DMA writes target overlapping buffer regions",
                            in_flight[semaphore_owner],
                            operation,
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
                if self.physical_schema is None:
                    if links[axis] <= 0:
                        raise VerifyException(
                            "reduce-scatter requires a usable interconnect link"
                        )
                else:
                    if operation.collective_plan is None:
                        raise VerifyException(
                            "structured reduce-scatter needs a collective plan"
                        )
                    plan = topology_plans.get(operation.collective_plan.data)
                    if plan is None or plan.mesh_axis.data != axis:
                        raise VerifyException(
                            "reduce-scatter references an incompatible collective plan"
                        )
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
            if isinstance(operation, PipelineLoopOp):
                for value in operation.captures:
                    require_initialized(value, operation)
                capture_roots = [root(value) for value in operation.captures]
                if len(capture_roots) != len(set(capture_roots)):
                    raise VerifyException("pipeline captures must reference distinct buffers")
                for value, rotation in zip(
                    capture_roots, operation.rotation_counts, strict=True
                ):
                    buffer = value.type
                    assert isinstance(buffer, BufferType)
                    if (
                        buffer.ownership.data is not Ownership.EXTERNAL
                        and rotation.data > 1
                    ):
                        rotation_copies.append((buffer, rotation.data - 1))
                for value in operation.outputs:
                    initialized.add(root(value))

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
            if self.physical_schema is not None:
                link_uses: dict[str, int] = {}
                for operation in operations:
                    if not isinstance(operation, CollectiveReduceScatterOp):
                        continue
                    if operation.stage.data != stage or operation.collective_plan is None:
                        continue
                    plan = topology_plans[operation.collective_plan.data]
                    for group in plan.groups:
                        for link_id in group.route_link_ids:
                            link_uses[link_id.data] = link_uses.get(link_id.data, 0) + 1
                for link_id, uses in link_uses.items():
                    capacity = topology_links[link_id].channel_count.data
                    if uses > capacity:
                        raise VerifyException(
                            f"topology link {link_id} capacity exceeded at stage {stage}: "
                            f"{uses} > {capacity}"
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
                live_bytes += sum(
                    buffer_bytes(buffer) * copies
                    for buffer, copies in rotation_copies
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
        PipelineLoopOp,
        PipelineYieldOp,
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
        DeviceAttr,
        LinkAttr,
        CollectiveGroupAttr,
        CollectivePlanAttr,
        TopologyAttr,
        BufferType,
        DmaTokenType,
        SemaphoreType,
    ],
)
