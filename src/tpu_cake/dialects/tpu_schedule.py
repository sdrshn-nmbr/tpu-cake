from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
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
    Signedness,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
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


class CollectiveKind(StrEnum):
    ALL_GATHER = "all_gather"
    ALL_REDUCE = "all_reduce"
    ALL_TO_ALL = "all_to_all"
    REDUCE_SCATTER = "reduce_scatter"


class VectorMaterialization(StrEnum):
    STRICT_TYPED = "strict_typed"


@irdl_attr_definition
class MemorySpaceAttr(EnumAttribute[MemorySpace], SpacedOpaqueSyntaxAttribute):
    name = "tpu_schedule.memory_space"


@irdl_attr_definition
class OwnershipAttr(EnumAttribute[Ownership], SpacedOpaqueSyntaxAttribute):
    name = "tpu_schedule.ownership"


@irdl_attr_definition
class CollectiveKindAttr(EnumAttribute[CollectiveKind], SpacedOpaqueSyntaxAttribute):
    name = "tpu_schedule.collective_kind"


@irdl_attr_definition
class VectorMaterializationAttr(EnumAttribute[VectorMaterialization], SpacedOpaqueSyntaxAttribute):
    name = "tpu_schedule.vector_materialization"


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

    def verify(self) -> None:
        if self.device_id.data < 0:
            raise VerifyException("topology device IDs must be nonnegative")
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
class TransferRouteAttr(ParametrizedAttribute):
    name = "tpu_schedule.transfer_route"
    route_id: StringAttr
    source_device: IntAttr
    destination_device: IntAttr
    route_link_ids: ArrayAttr[StringAttr]

    def verify(self) -> None:
        routes = tuple(value.data for value in self.route_link_ids)
        if not self.route_id.data:
            raise VerifyException("transfer routes need stable IDs")
        if self.source_device.data < 0 or self.destination_device.data < 0:
            raise VerifyException("transfer route endpoints must be nonnegative")
        if self.source_device.data == self.destination_device.data:
            raise VerifyException("transfer routes must cross devices")
        if not routes or len(routes) != len(set(routes)):
            raise VerifyException("transfer routes need a nonempty simple link path")


@irdl_attr_definition
class TransferPlanAttr(ParametrizedAttribute):
    name = "tpu_schedule.transfer_plan"
    plan_id: StringAttr
    routes: ArrayAttr[TransferRouteAttr]

    def verify(self) -> None:
        routes = tuple(self.routes)
        if not self.plan_id.data:
            raise VerifyException("transfer plans need stable IDs")
        for route in routes:
            route.verify()
        route_ids = tuple(route.route_id.data for route in routes)
        if not routes or route_ids != tuple(sorted(set(route_ids))):
            raise VerifyException("transfer plan routes must be unique and ordered")
        sources = tuple(route.source_device.data for route in routes)
        destinations = tuple(route.destination_device.data for route in routes)
        if len(sources) != len(set(sources)) or len(destinations) != len(set(destinations)):
            raise VerifyException(
                "transfer plans cannot send from or write to one device more than once"
            )


@irdl_attr_definition
class TopologyAttr(ParametrizedAttribute):
    name = "tpu_schedule.topology"
    devices: ArrayAttr[DeviceAttr]
    links: ArrayAttr[LinkAttr]
    collective_plans: ArrayAttr[CollectivePlanAttr]
    transfer_plans: ArrayAttr[TransferPlanAttr]

    def verify(self) -> None:
        devices = tuple(self.devices)
        links = tuple(self.links)
        plans = tuple(self.collective_plans)
        transfer_plans = tuple(self.transfer_plans)
        for device in devices:
            device.verify()
        for link in links:
            link.verify()
        for plan in plans:
            plan.verify()
        for plan in transfer_plans:
            plan.verify()

        device_ids = tuple(device.device_id.data for device in devices)
        if device_ids != tuple(range(len(devices))):
            raise VerifyException("topology device IDs must be dense and canonically ordered")
        coordinate_rank = len(devices[0].coordinates) if devices else 0
        coordinates = tuple(tuple(value.data for value in device.coordinates) for device in devices)
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

        transfer_plan_ids = tuple(plan.plan_id.data for plan in transfer_plans)
        if transfer_plan_ids != tuple(sorted(set(transfer_plan_ids))):
            raise VerifyException("topology transfer plans must be unique and canonically ordered")
        for plan in transfer_plans:
            for route in plan.routes:
                source = route.source_device.data
                destination = route.destination_device.data
                if source not in known_devices or destination not in known_devices:
                    raise VerifyException("transfer route references an unknown device")
                current = source
                visited = {source}
                for link_id in route.route_link_ids:
                    link = links_by_id.get(link_id.data)
                    if link is None:
                        raise VerifyException("transfer route references an unknown link")
                    endpoints = {link.source_device.data, link.destination_device.data}
                    if current not in endpoints:
                        raise VerifyException("transfer route links do not form a contiguous path")
                    current = next(device for device in endpoints if device != current)
                    if current in visited:
                        raise VerifyException("transfer routes cannot revisit a device")
                    visited.add(current)
                if current != destination:
                    raise VerifyException("transfer route does not reach its destination")

    def plans_by_id(self) -> dict[str, CollectivePlanAttr]:
        self.verify()
        return {plan.plan_id.data: plan for plan in self.collective_plans}

    def links_by_id(self) -> dict[str, LinkAttr]:
        self.verify()
        return {link.link_id.data: link for link in self.links}

    def transfer_plans_by_id(self) -> dict[str, TransferPlanAttr]:
        self.verify()
        return {plan.plan_id.data: plan for plan in self.transfer_plans}


def rectilinear_topology(
    mesh_axis_names: tuple[str, ...],
    mesh_axis_sizes: tuple[int, ...],
    bandwidth_bytes_per_second: dict[str, int],
    *,
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
        )
        for device_id, coordinate in enumerate(coordinates)
    )
    links: dict[tuple[int, int], LinkAttr] = {}
    plans: list[CollectivePlanAttr] = []
    transfer_plans: list[TransferPlanAttr] = []
    for axis_index, axis in enumerate(mesh_axis_names):
        if mesh_axis_sizes[axis_index] == 1:
            continue
        groups: list[CollectiveGroupAttr] = []
        other_indices = tuple(index for index in range(len(mesh_axis_names)) if index != axis_index)
        fixed_coordinates = product(*(range(mesh_axis_sizes[index]) for index in other_indices))
        for group_index, fixed in enumerate(fixed_coordinates):
            base = dict(zip(other_indices, fixed, strict=True))
            group_coordinates = tuple(
                tuple(
                    coordinate if index == axis_index else base[index]
                    for index in range(len(mesh_axis_names))
                )
                for coordinate in range(mesh_axis_sizes[axis_index])
            )
            device_ids = tuple(coordinate_to_device[coordinate] for coordinate in group_coordinates)
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
        for direction in (-1, 1):
            routes: list[TransferRouteAttr] = []
            for coordinate, source in coordinate_to_device.items():
                peer = list(coordinate)
                peer[axis_index] += direction
                if not 0 <= peer[axis_index] < mesh_axis_sizes[axis_index]:
                    continue
                destination = coordinate_to_device[tuple(peer)]
                endpoint = (min(source, destination), max(source, destination))
                routes.append(
                    TransferRouteAttr(
                        StringAttr(f"route:{source}->{destination}"),
                        IntAttr(source),
                        IntAttr(destination),
                        ArrayAttr((StringAttr(f"link:{endpoint[0]}-{endpoint[1]}"),)),
                    )
                )
            transfer_plans.append(
                TransferPlanAttr(
                    StringAttr(f"shift:{axis}:{direction:+d}"),
                    ArrayAttr(sorted(routes, key=lambda route: route.route_id.data)),
                )
            )
    topology = TopologyAttr(
        ArrayAttr(devices),
        ArrayAttr(sorted(links.values(), key=lambda link: link.link_id.data)),
        ArrayAttr(sorted(plans, key=lambda plan: plan.plan_id.data)),
        ArrayAttr(sorted(transfer_plans, key=lambda plan: plan.plan_id.data)),
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
        if len(logical_shape) != len(set(logical_shape)):
            raise VerifyException("logical shape dimensions must be unique")
        if len(sharding) != rank:
            raise VerifyException("sharding rank must match storage rank")
        used_mesh_axes = [
            axis
            for dimension_sharding in sharding
            for axis in filter(None, dimension_sharding.split("/"))
        ]
        if len(used_mesh_axes) != len(set(used_mesh_axes)):
            raise VerifyException("one mesh axis cannot shard multiple buffer dimensions")
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
                (delta // divisor) * pow(lhs_stride // divisor, -1, reduced_modulus)
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
class RemoteDmaStartOp(IRDLOperation):
    name = "tpu_schedule.remote_dma_start"
    source = operand_def(BufferType)
    destination = operand_def(BufferType)
    semaphore = operand_def(SemaphoreType)
    token = result_def(DmaTokenType)
    stage = prop_def(IntAttr)
    transfer_plan = prop_def(StringAttr)

    def __init__(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        semaphore: SSAValue | Operation,
        *,
        stage: int,
        transfer_plan: str,
    ) -> None:
        super().__init__(
            operands=[source, destination, semaphore],
            result_types=[DmaTokenType()],
            properties={
                "stage": IntAttr(stage),
                "transfer_plan": StringAttr(transfer_plan),
            },
        )

    def verify_(self) -> None:
        source, destination = self.source.type, self.destination.type
        assert isinstance(source, BufferType) and isinstance(destination, BufferType)
        for buffer in (source, destination):
            if buffer.space.data is not MemorySpace.VMEM:
                raise VerifyException("remote DMA buffers must be resident in VMEM")
            _check_live(buffer, self.stage.data)
        if source.storage != destination.storage:
            raise VerifyException("remote DMA source and destination storage must match")
        if source.shape != destination.shape:
            raise VerifyException("remote DMA cannot rename logical dimensions")
        if source.sharding != destination.sharding or source.layout != destination.layout:
            raise VerifyException("remote DMA must preserve sharding and layout")
        if not self.transfer_plan.data:
            raise VerifyException("remote DMA needs a transfer plan")


@irdl_op_definition
class RemoteDmaWaitOp(IRDLOperation):
    name = "tpu_schedule.remote_dma_wait"
    token = operand_def(DmaTokenType)
    stage = prop_def(IntAttr)

    def __init__(self, token: SSAValue | Operation, *, stage: int) -> None:
        super().__init__(operands=[token], properties={"stage": IntAttr(stage)})

    def verify_(self) -> None:
        owner = self.token.owner
        if not isinstance(owner, RemoteDmaStartOp):
            raise VerifyException(
                "remote DMA wait token must come from tpu_schedule.remote_dma_start"
            )
        if self.stage.data < owner.stage.data:
            raise VerifyException("remote DMA wait cannot precede its start stage")
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


def _named_buffer_shape(buffer: BufferType) -> dict[str, int]:
    names = tuple(value.data for value in buffer.shape.dimensions)
    sizes = buffer.storage.get_shape()
    if len(names) != len(set(names)):
        raise VerifyException("physical tensor operations require unique logical dimensions")
    return dict(zip(names, sizes, strict=True))


def _same_physical_value_contract(lhs: BufferType, rhs: BufferType) -> bool:
    return (
        lhs.storage == rhs.storage
        and lhs.shape == rhs.shape
        and lhs.space == rhs.space
        and lhs.sharding == rhs.sharding
        and lhs.layout == rhs.layout
        and lhs.ownership == rhs.ownership
    )


def _same_physical_shape_and_placement(lhs: BufferType, rhs: BufferType) -> bool:
    return (
        lhs.storage.get_shape() == rhs.storage.get_shape()
        and lhs.shape == rhs.shape
        and lhs.space == rhs.space
        and lhs.sharding == rhs.sharding
        and lhs.layout == rhs.layout
        and lhs.ownership == rhs.ownership
    )


def _element_type_name(buffer: BufferType) -> str:
    element_type = buffer.storage.element_type
    if isinstance(element_type, BFloat16Type):
        return "bf16"
    if isinstance(element_type, Float16Type):
        return "f16"
    if isinstance(element_type, Float32Type):
        return "f32"
    if isinstance(element_type, IntegerType):
        if element_type.width.data == 1:
            return "bool"
        prefix = "u" if element_type.signedness.data is Signedness.UNSIGNED else "i"
        return f"{prefix}{element_type.width.data}"
    raise VerifyException(f"unsupported physical element type {element_type}")


def _is_float_buffer(buffer: BufferType) -> bool:
    return isinstance(
        buffer.storage.element_type,
        BFloat16Type | Float16Type | Float32Type,
    )


def _dimension_sharding(buffer: BufferType, dimension: str) -> str:
    names = tuple(value.data for value in buffer.shape.dimensions)
    if dimension not in names:
        raise VerifyException(f"physical tensor has no logical dimension {dimension!r}")
    return tuple(buffer.sharding.axes)[names.index(dimension)].data


@irdl_op_definition
class MxuEinsumOp(IRDLOperation):
    name = "tpu_schedule.mxu_einsum"
    lhs = operand_def(BufferType)
    rhs = operand_def(BufferType)
    accumulator = operand_def(BufferType)
    stage = prop_def(IntAttr)
    contracting_dimensions = prop_def(ArrayAttr[StringAttr])
    pending_reduction_axes = prop_def(ArrayAttr[StringAttr])
    tile_m = prop_def(IntAttr)
    tile_k = prop_def(IntAttr)
    tile_n = prop_def(IntAttr)

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        accumulator: SSAValue | Operation,
        *,
        stage: int,
        contracting_dimensions: tuple[str, ...],
        pending_reduction_axes: tuple[str, ...] = (),
        tile_m: int,
        tile_k: int,
        tile_n: int,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, accumulator],
            properties={
                "stage": IntAttr(stage),
                "contracting_dimensions": ArrayAttr(
                    StringAttr(value) for value in sorted(contracting_dimensions)
                ),
                "pending_reduction_axes": ArrayAttr(
                    StringAttr(value) for value in sorted(pending_reduction_axes)
                ),
                "tile_m": IntAttr(tile_m),
                "tile_k": IntAttr(tile_k),
                "tile_n": IntAttr(tile_n),
            },
        )

    def verify_(self) -> None:
        lhs, rhs, accumulator = self.lhs.type, self.rhs.type, self.accumulator.type
        assert isinstance(lhs, BufferType)
        assert isinstance(rhs, BufferType)
        assert isinstance(accumulator, BufferType)
        for buffer in (lhs, rhs, accumulator):
            if buffer.space.data is not MemorySpace.VMEM:
                raise VerifyException("MXU einsum operands must be resident in VMEM")
            _check_live(buffer, self.stage.data)
        if not isinstance(
            lhs.storage.element_type,
            BFloat16Type | Float16Type | Float32Type,
        ):
            raise VerifyException("MXU einsum inputs must use bf16, f16, or f32")
        if rhs.storage.element_type != lhs.storage.element_type:
            raise VerifyException("MXU einsum input element types must match")
        if not isinstance(accumulator.storage.element_type, Float32Type):
            raise VerifyException("MXU einsum accumulation must use f32")

        contractions = tuple(value.data for value in self.contracting_dimensions)
        if not contractions or contractions != tuple(sorted(set(contractions))):
            raise VerifyException(
                "MXU einsum contraction dimensions must be non-empty, unique, and canonical"
            )
        lhs_shape = _named_buffer_shape(lhs)
        rhs_shape = _named_buffer_shape(rhs)
        result_shape = _named_buffer_shape(accumulator)
        lhs_sharding = {
            name: axis.data for name, axis in zip(lhs_shape, lhs.sharding.axes, strict=True)
        }
        rhs_sharding = {
            name: axis.data for name, axis in zip(rhs_shape, rhs.sharding.axes, strict=True)
        }
        result_sharding = {
            name: axis.data
            for name, axis in zip(result_shape, accumulator.sharding.axes, strict=True)
        }
        if any(name not in lhs_shape or name not in rhs_shape for name in contractions):
            raise VerifyException("MXU einsum contraction dimensions must exist in both inputs")
        if any(lhs_shape[name] != rhs_shape[name] for name in contractions):
            raise VerifyException("MXU einsum local contraction extents must match")
        if any(lhs_sharding[name] != rhs_sharding[name] for name in contractions):
            raise VerifyException("MXU einsum contracted dimensions must have equal sharding")
        expected_pending = tuple(
            sorted(
                {
                    axis
                    for name in contractions
                    for axis in filter(None, lhs_sharding[name].split("/"))
                }
            )
        )
        pending = tuple(value.data for value in self.pending_reduction_axes)
        if pending != tuple(sorted(set(pending))) or pending != expected_pending:
            raise VerifyException(
                "MXU einsum pending reductions must match contracted-dimension sharding"
            )

        shared = (set(lhs_shape) & set(rhs_shape)) - set(contractions)
        if any(lhs_shape[name] != rhs_shape[name] for name in shared):
            raise VerifyException("MXU einsum shared local extents must match")
        if any(lhs_sharding[name] != rhs_sharding[name] for name in shared):
            raise VerifyException("MXU einsum shared dimensions must have equal sharding")
        expected_names = (set(lhs_shape) | set(rhs_shape)) - set(contractions)
        if set(result_shape) != expected_names:
            raise VerifyException("MXU einsum result has the wrong logical dimensions")
        for name, size in result_shape.items():
            expected_size = lhs_shape.get(name, rhs_shape.get(name))
            if expected_size != size:
                raise VerifyException("MXU einsum result has the wrong local extent")
            expected_sharding = lhs_sharding.get(name, rhs_sharding.get(name))
            if result_sharding[name] != expected_sharding:
                raise VerifyException("MXU einsum result must preserve retained-dimension sharding")

        batch = math.prod(lhs_shape[name] for name in shared)
        m = math.prod(
            size
            for name, size in lhs_shape.items()
            if name not in shared and name not in contractions
        )
        k = math.prod(lhs_shape[name] for name in contractions)
        n = math.prod(
            size
            for name, size in rhs_shape.items()
            if name not in shared and name not in contractions
        )
        tile = (self.tile_m.data, self.tile_k.data, self.tile_n.data)
        if any(size <= 0 for size in tile):
            raise VerifyException("MXU einsum tile dimensions must be positive")
        if m % tile[0] or k % tile[1] or n % tile[2]:
            raise VerifyException("MXU einsum tiles must divide flattened M, K, and N")
        kernel = self.parent_op()
        while kernel is not None and not isinstance(kernel, KernelOp):
            kernel = kernel.parent_op()
        if isinstance(kernel, KernelOp) and kernel.target.data == "tpu7x":
            if tile[0] != m and tile[0] % 8:
                raise VerifyException("TPU Pallas tile M must span M or be divisible by 8")
            if tile[1] != k and tile[1] % 128:
                raise VerifyException("TPU Pallas tile K must span K or be divisible by 128")
            if tile[2] != n and tile[2] % 128:
                raise VerifyException("TPU Pallas tile N must span N or be divisible by 128")
        if batch <= 0:
            raise VerifyException("MXU einsum batch extent must be positive")


@irdl_op_definition
class VectorComputeOp(IRDLOperation):
    name = "tpu_schedule.vector_compute"
    inputs = var_operand_def(BufferType)
    output = operand_def(BufferType)
    stage = prop_def(IntAttr)
    function = prop_def(StringAttr)
    configuration = prop_def(ArrayAttr[StringAttr])
    pending_reduction_axes = prop_def(ArrayAttr[StringAttr])
    materialization = opt_prop_def(VectorMaterializationAttr)

    def __init__(
        self,
        inputs: tuple[SSAValue | Operation, ...],
        output: SSAValue | Operation,
        *,
        stage: int,
        function: str,
        configuration: tuple[str, ...] = (),
        pending_reduction_axes: tuple[str, ...] = (),
        materialization: VectorMaterialization | None = None,
    ) -> None:
        properties: dict[str, Attribute] = {
            "stage": IntAttr(stage),
            "function": StringAttr(function),
            "configuration": ArrayAttr(StringAttr(value) for value in sorted(configuration)),
            "pending_reduction_axes": ArrayAttr(
                StringAttr(value) for value in sorted(pending_reduction_axes)
            ),
        }
        if materialization is not None:
            properties["materialization"] = VectorMaterializationAttr(materialization)
        super().__init__(
            operands=[list(inputs), output],
            properties=properties,
        )

    def verify_(self) -> None:
        supported_arity = {
            "cast": 1,
            "rename_dimension": 1,
            "slice": 1,
            "rms_norm": 2,
            "rotary_embedding": 1,
            "packed_causal_mask": 1,
            "masked_softmax": 2,
            "embedding_lookup": 2,
            "add": 2,
            "multiply": 2,
            "silu": 1,
            "exp": 1,
        }
        required_configuration = {
            "cast": {"dtype"},
            "rename_dimension": {"destination", "source"},
            "slice": {"dimension", "index"},
            "rms_norm": {"dimension", "epsilon"},
            "rotary_embedding": {
                "head_dimension",
                "maximum_timescale",
                "sequence_dimension",
            },
            "packed_causal_mask": {
                "key_dimension",
                "query_dimension",
                "sequence_dimension",
            },
            "masked_softmax": {"dimension"},
            "embedding_lookup": {"vocabulary_dimension"},
            "add": set(),
            "multiply": set(),
            "silu": set(),
            "exp": set(),
        }
        function = self.function.data
        if function not in supported_arity:
            raise VerifyException(f"unsupported physical vector function {function!r}")
        if len(self.inputs) != supported_arity[function]:
            raise VerifyException(
                f"physical vector function {function!r} requires {supported_arity[function]} inputs"
            )
        configuration = tuple(value.data for value in self.configuration)
        if configuration != tuple(sorted(set(configuration))):
            raise VerifyException("vector configuration must be unique and canonical")
        keys = tuple(value.split("=", 1)[0] for value in configuration)
        if any("=" not in value or not key for value, key in zip(configuration, keys, strict=True)):
            raise VerifyException("vector configuration entries must use key=value form")
        if len(keys) != len(set(keys)):
            raise VerifyException("vector configuration keys must be unique")
        if set(keys) != required_configuration[function]:
            raise VerifyException(
                f"physical vector function {function!r} has the wrong configuration keys"
            )
        parsed_configuration = dict(value.split("=", 1) for value in configuration)
        pending_reductions = tuple(value.data for value in self.pending_reduction_axes)
        if pending_reductions != tuple(sorted(set(pending_reductions))):
            raise VerifyException("vector pending reductions must be unique and canonical")

        buffers = tuple(value.type for value in (*self.inputs, self.output))
        assert all(isinstance(value, BufferType) for value in buffers)
        for buffer in buffers:
            assert isinstance(buffer, BufferType)
            if buffer.space.data is not MemorySpace.VMEM:
                raise VerifyException("physical vector operands must be resident in VMEM")
            _check_live(buffer, self.stage.data)
        output = self.output.type
        assert isinstance(output, BufferType)
        inputs = tuple(value.type for value in self.inputs)
        assert all(isinstance(value, BufferType) for value in inputs)

        if function in {"add", "multiply"} and any(
            not _same_physical_value_contract(value, output) for value in inputs
        ):
            raise VerifyException("binary vector operands must have identical buffer types")
        if function in {"silu", "exp"} and not _same_physical_value_contract(inputs[0], output):
            raise VerifyException("unary vector operations must preserve buffer type")
        if function in {"silu", "exp"} and not _is_float_buffer(output):
            raise VerifyException("nonlinear physical vector operations require floating point")
        if self.materialization is not None:
            if function != "silu":
                raise VerifyException("strict typed materialization is only supported for SiLU")
            if not isinstance(output.storage.element_type, BFloat16Type):
                raise VerifyException("strict typed SiLU materialization requires BF16")
        if function in {"cast", "rename_dimension"}:
            source = inputs[0]
            assert isinstance(source, BufferType)
            if (
                source.storage.get_shape() != output.storage.get_shape()
                or source.sharding != output.sharding
                or source.layout != output.layout
            ):
                raise VerifyException(f"{function} must preserve physical shape and placement")
            if function == "cast" and source.storage.element_type == output.storage.element_type:
                raise VerifyException("physical cast must change element type")
            if function == "cast" and parsed_configuration["dtype"] != _element_type_name(output):
                raise VerifyException("physical cast dtype does not match its output")
            if function == "cast" and source.shape != output.shape:
                raise VerifyException("physical cast cannot rename logical dimensions")
            if (
                function == "rename_dimension"
                and source.storage.element_type != output.storage.element_type
            ):
                raise VerifyException("physical dimension rename cannot change element type")
            if function == "rename_dimension":
                source_names = tuple(value.data for value in source.shape.dimensions)
                source_name = parsed_configuration["source"]
                destination_name = parsed_configuration["destination"]
                if source_name not in source_names or (
                    destination_name in source_names and destination_name != source_name
                ):
                    raise VerifyException("physical dimension rename has invalid names")
                expected_names = tuple(
                    destination_name if name == source_name else name for name in source_names
                )
                if tuple(value.data for value in output.shape.dimensions) != expected_names:
                    raise VerifyException("physical dimension rename has the wrong output names")
        if function == "slice":
            source = inputs[0]
            assert isinstance(source, BufferType)
            if len(source.storage.get_shape()) != len(output.storage.get_shape()) + 1:
                raise VerifyException("physical slice must remove exactly one dimension")
            source_names = tuple(value.data for value in source.shape.dimensions)
            dimension = parsed_configuration["dimension"]
            if dimension not in source_names:
                raise VerifyException("physical slice dimension does not exist")
            axis = source_names.index(dimension)
            if tuple(source.sharding.axes)[axis].data:
                raise VerifyException("physical slice cannot index a sharded dimension")
            try:
                index = int(parsed_configuration["index"])
            except ValueError as error:
                raise VerifyException("physical slice index must be an integer") from error
            if not 0 <= index < source.storage.get_shape()[axis]:
                raise VerifyException("physical slice index is out of bounds")
            expected_shape = (
                source.storage.get_shape()[:axis] + source.storage.get_shape()[axis + 1 :]
            )
            expected_names = source_names[:axis] + source_names[axis + 1 :]
            source_sharding = tuple(source.sharding.axes)
            expected_sharding = source_sharding[:axis] + source_sharding[axis + 1 :]
            if (
                output.storage.get_shape() != expected_shape
                or output.storage.element_type != source.storage.element_type
                or tuple(value.data for value in output.shape.dimensions) != expected_names
                or tuple(output.sharding.axes) != expected_sharding
            ):
                raise VerifyException("physical slice has the wrong output contract")
        if function == "rms_norm":
            value, scale = inputs
            assert isinstance(value, BufferType) and isinstance(scale, BufferType)
            if not all(_is_float_buffer(buffer) for buffer in (value, scale, output)):
                raise VerifyException("physical RMSNorm requires floating-point tensors")
            if not _same_physical_value_contract(value, output):
                raise VerifyException("physical RMSNorm must preserve value storage and sharding")
            if len(scale.storage.get_shape()) != 1:
                raise VerifyException("physical RMSNorm scale must have rank one")
            dimension = parsed_configuration["dimension"]
            value_shape = _named_buffer_shape(value)
            if dimension not in value_shape:
                raise VerifyException("physical RMSNorm dimension does not exist")
            if _named_buffer_shape(scale) != {dimension: value_shape[dimension]}:
                raise VerifyException("physical RMSNorm scale has the wrong named shape")
            if _dimension_sharding(value, dimension):
                raise VerifyException("physical RMSNorm dimension cannot be sharded")
            if any(value.data for value in scale.sharding.axes):
                raise VerifyException("physical RMSNorm scale must be locally replicated")
            try:
                epsilon = Decimal(parsed_configuration["epsilon"])
            except InvalidOperation as error:
                raise VerifyException("physical RMSNorm epsilon must be a decimal") from error
            if not epsilon.is_finite() or epsilon <= 0:
                raise VerifyException("physical RMSNorm epsilon must be positive and finite")
        if function == "rotary_embedding":
            value = inputs[0]
            assert isinstance(value, BufferType)
            if not _is_float_buffer(value):
                raise VerifyException("physical rotary embedding requires floating-point input")
            if not _same_physical_shape_and_placement(value, output):
                raise VerifyException("physical rotary embedding must preserve shape and sharding")
            if not isinstance(output.storage.element_type, Float32Type):
                raise VerifyException("physical rotary embedding must produce f32")
            names = tuple(value.data for value in value.shape.dimensions)
            sequence = parsed_configuration["sequence_dimension"]
            head = parsed_configuration["head_dimension"]
            if sequence not in names or not names or names[-1] != head:
                raise VerifyException("physical rotary embedding has invalid dimensions")
            if _dimension_sharding(value, sequence) or _dimension_sharding(value, head):
                raise VerifyException(
                    "physical rotary embedding semantic dimensions cannot be sharded"
                )
            if value.storage.get_shape()[-1] % 2:
                raise VerifyException("physical rotary head dimension must be even")
            try:
                maximum_timescale = Decimal(parsed_configuration["maximum_timescale"])
            except InvalidOperation as error:
                raise VerifyException("physical rotary timescale must be a decimal") from error
            if not maximum_timescale.is_finite() or maximum_timescale <= 0:
                raise VerifyException("physical rotary timescale must be positive and finite")
        if function == "packed_causal_mask":
            starts = inputs[0]
            assert isinstance(starts, BufferType)
            starts_names = tuple(value.data for value in starts.shape.dimensions)
            sequence = parsed_configuration["sequence_dimension"]
            query = parsed_configuration["query_dimension"]
            key = parsed_configuration["key_dimension"]
            if query == key:
                raise VerifyException("physical packed mask query and key dimensions must differ")
            if len(starts_names) != 2 or starts_names[-1] != sequence:
                raise VerifyException("physical packed mask needs [batch, sequence] input")
            batch = starts_names[0]
            sequence_extent = starts.storage.get_shape()[-1]
            starts_sharding = tuple(starts.sharding.axes)
            expected_sharding = (
                *starts_sharding[:-1],
                StringAttr(""),
                StringAttr(""),
            )
            if (
                not isinstance(starts.storage.element_type, IntegerType)
                or starts.storage.element_type.width.data != 1
                or not isinstance(output.storage.element_type, IntegerType)
                or output.storage.element_type.width.data != 1
                or tuple(value.data for value in output.shape.dimensions) != (batch, query, key)
                or output.storage.get_shape()
                != (starts.storage.get_shape()[0], sequence_extent, sequence_extent)
                or starts_sharding[-1].data
                or tuple(output.sharding.axes) != expected_sharding
            ):
                raise VerifyException("physical packed mask has the wrong output contract")
        if function == "masked_softmax":
            value, mask = inputs
            assert isinstance(value, BufferType) and isinstance(mask, BufferType)
            if not _is_float_buffer(value) or not _is_float_buffer(output):
                raise VerifyException("physical masked softmax values must be floating point")
            if not _same_physical_shape_and_placement(value, output):
                raise VerifyException("physical masked softmax must preserve shape and sharding")
            dimension = parsed_configuration["dimension"]
            value_shape = _named_buffer_shape(value)
            mask_shape = _named_buffer_shape(mask)
            if dimension not in value_shape:
                raise VerifyException("physical masked softmax dimension does not exist")
            if _dimension_sharding(value, dimension):
                raise VerifyException("physical masked softmax dimension cannot be sharded")
            value_sharding = {
                name: axis.data for name, axis in zip(value_shape, value.sharding.axes, strict=True)
            }
            mask_sharding = {
                name: axis.data for name, axis in zip(mask_shape, mask.sharding.axes, strict=True)
            }
            if (
                not isinstance(mask.storage.element_type, IntegerType)
                or mask.storage.element_type.width.data != 1
                or any(
                    name not in value_shape or value_shape[name] != size
                    for name, size in mask_shape.items()
                )
                or any(value_sharding[name] != axis for name, axis in mask_sharding.items())
            ):
                raise VerifyException("physical masked softmax mask is incompatible")
        if function == "embedding_lookup":
            table, indices = inputs
            assert isinstance(table, BufferType) and isinstance(indices, BufferType)
            if not isinstance(indices.storage.element_type, IntegerType):
                raise VerifyException("physical embedding indices must be integers")
            if len(output.storage.get_shape()) != (
                len(indices.storage.get_shape()) + len(table.storage.get_shape()) - 1
            ):
                raise VerifyException("physical embedding lookup has the wrong output rank")
            vocabulary = parsed_configuration["vocabulary_dimension"]
            table_shape = _named_buffer_shape(table)
            if vocabulary not in table_shape:
                raise VerifyException("physical embedding vocabulary dimension does not exist")
            expected_names = (
                *tuple(value.data for value in indices.shape.dimensions),
                *(name for name in table_shape if name != vocabulary),
            )
            expected_shape = (
                *indices.storage.get_shape(),
                *(size for name, size in table_shape.items() if name != vocabulary),
            )
            table_names = tuple(value.data for value in table.shape.dimensions)
            vocabulary_axis = table_names.index(vocabulary)
            expected_pending = tuple(
                sorted(
                    filter(
                        None,
                        tuple(table.sharding.axes)[vocabulary_axis].data.split("/"),
                    )
                )
            )
            if pending_reductions != expected_pending:
                raise VerifyException(
                    "physical embedding pending reductions must match vocabulary sharding"
                )
            expected_sharding = (
                *tuple(indices.sharding.axes),
                *(
                    axis
                    for offset, axis in enumerate(table.sharding.axes)
                    if offset != vocabulary_axis
                ),
            )
            if (
                tuple(value.data for value in output.shape.dimensions) != expected_names
                or output.storage.get_shape() != expected_shape
                or output.storage.element_type != table.storage.element_type
                or tuple(output.sharding.axes) != expected_sharding
            ):
                raise VerifyException("physical embedding lookup has the wrong output contract")


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
                "collective_plan": StringAttr(collective_plan or f"axis:{mesh_axis}"),
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


def _buffer_sharding(buffer: BufferType) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(filter(None, value.data.split("/"))) for value in buffer.sharding.axes)


@irdl_op_definition
class CollectiveOp(IRDLOperation):
    name = "tpu_schedule.collective"
    source = operand_def(BufferType)
    destination = operand_def(BufferType)
    stage = prop_def(IntAttr)
    kind = prop_def(CollectiveKindAttr)
    mesh_axis = prop_def(StringAttr)
    group_size = prop_def(IntAttr)
    split_dimension = prop_def(IntAttr)
    concat_dimension = prop_def(IntAttr)
    reducer = prop_def(StringAttr)
    collective_plan = prop_def(StringAttr)

    def __init__(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        *,
        stage: int,
        kind: CollectiveKind,
        mesh_axis: str,
        group_size: int,
        split_dimension: int = -1,
        concat_dimension: int = -1,
        reducer: str = "none",
        collective_plan: str | None = None,
    ) -> None:
        super().__init__(
            operands=[source, destination],
            properties={
                "stage": IntAttr(stage),
                "kind": CollectiveKindAttr(kind),
                "mesh_axis": StringAttr(mesh_axis),
                "group_size": IntAttr(group_size),
                "split_dimension": IntAttr(split_dimension),
                "concat_dimension": IntAttr(concat_dimension),
                "reducer": StringAttr(reducer),
                "collective_plan": StringAttr(collective_plan or f"axis:{mesh_axis}"),
            },
        )

    def verify_(self) -> None:
        source, destination = self.source.type, self.destination.type
        assert isinstance(source, BufferType) and isinstance(destination, BufferType)
        for buffer in (source, destination):
            if buffer.space.data is not MemorySpace.VMEM:
                raise VerifyException("collective buffers must be resident in VMEM")
            _check_live(buffer, self.stage.data)
        if source.storage.element_type != destination.storage.element_type:
            raise VerifyException("collectives cannot change element type")
        if source.shape != destination.shape:
            raise VerifyException("collectives cannot rename logical dimensions")
        if source.layout != destination.layout:
            raise VerifyException("collectives cannot change physical layout")
        source_shape = source.storage.get_shape()
        destination_shape = destination.storage.get_shape()
        if len(source_shape) != len(destination_shape):
            raise VerifyException("collectives cannot change rank")
        if self.group_size.data <= 1:
            raise VerifyException("collective group size must be greater than one")
        if not self.mesh_axis.data or not self.collective_plan.data:
            raise VerifyException("collectives need a mesh axis and collective plan")
        if self.reducer.data not in {"none", "sum", "max", "min"}:
            raise VerifyException("unsupported collective reducer")

        rank = len(source_shape)
        split = self.split_dimension.data
        concat = self.concat_dimension.data
        for dimension in (split, concat):
            if dimension < -1 or dimension >= rank:
                raise VerifyException("collective dimension is out of range")
        source_sharding = [list(value) for value in _buffer_sharding(source)]
        destination_sharding = [list(value) for value in _buffer_sharding(destination)]
        expected_shape = list(source_shape)
        expected_sharding = [list(value) for value in source_sharding]
        axis = self.mesh_axis.data
        group_size = self.group_size.data

        if self.kind.data is CollectiveKind.ALL_REDUCE:
            if split != -1 or concat != -1 or self.reducer.data == "none":
                raise VerifyException("all-reduce needs a reducer and no split or concat dimension")
        elif self.kind.data is CollectiveKind.REDUCE_SCATTER:
            if split < 0 or concat != -1 or self.reducer.data == "none":
                raise VerifyException("reduce-scatter needs a reducer and one split dimension")
            if expected_shape[split] % group_size:
                raise VerifyException(
                    "reduce-scatter split dimension must divide by the group size"
                )
            if axis in expected_sharding[split]:
                raise VerifyException("reduce-scatter cannot add an already-present sharding axis")
            expected_shape[split] //= group_size
            expected_sharding[split].append(axis)
        elif self.kind.data is CollectiveKind.ALL_GATHER:
            if split != -1 or concat < 0 or self.reducer.data != "none":
                raise VerifyException("all-gather needs one concat dimension and no reducer")
            if not expected_sharding[concat] or expected_sharding[concat][-1] != axis:
                raise VerifyException(
                    "all-gather must remove its mesh axis from the gathered dimension"
                )
            expected_shape[concat] *= group_size
            expected_sharding[concat].pop()
        elif self.kind.data is CollectiveKind.ALL_TO_ALL:
            if split < 0 or concat < 0 or split == concat or self.reducer.data != "none":
                raise VerifyException(
                    "all-to-all needs distinct split and concat dimensions and no reducer"
                )
            if expected_shape[split] % group_size:
                raise VerifyException("all-to-all split dimension must divide by the group size")
            if not expected_sharding[concat] or expected_sharding[concat][-1] != axis:
                raise VerifyException(
                    "all-to-all must move its mesh axis from concat to split dimension"
                )
            if axis in expected_sharding[split]:
                raise VerifyException("all-to-all split dimension already contains its mesh axis")
            expected_shape[split] //= group_size
            expected_shape[concat] *= group_size
            expected_sharding[concat].pop()
            expected_sharding[split].append(axis)

        if tuple(expected_shape) != destination_shape:
            raise VerifyException("collective destination has the wrong local shape")
        if tuple(tuple(value) for value in expected_sharding) != tuple(
            tuple(value) for value in destination_sharding
        ):
            raise VerifyException("collective destination has the wrong sharding")


PhysicalCollectiveOp = CollectiveReduceScatterOp | CollectiveOp


def _is_collective(operation: Operation) -> bool:
    return isinstance(operation, (CollectiveReduceScatterOp, CollectiveOp))


def _collective_plan_id(operation: PhysicalCollectiveOp) -> str | None:
    value = operation.collective_plan
    return None if value is None else value.data


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
        query_shape = query.storage.get_shape()
        key_shape = key_cache.storage.get_shape()
        value_shape = value_cache.storage.get_shape()
        output_shape = output.storage.get_shape()
        if (
            len(query_shape) != 3
            or len(key_shape) != 4
            or len(value_shape) != 4
            or len(output_shape) != 3
        ):
            raise VerifyException(
                "RPA expects query/output [batch, heads, dim] and caches [pages, page, heads, dim]"
            )
        batch, query_heads, query_dimension = query_shape
        key_pages, page_size, key_heads, key_dimension = key_shape
        value_pages, value_page_size, value_heads, value_dimension = value_shape
        output_batch, output_heads, output_dimension = output_shape
        if (key_pages, page_size) != (value_pages, value_page_size):
            raise VerifyException("RPA key and value caches must share pages and page size")
        if query_dimension != key_dimension:
            raise VerifyException("RPA query and key head dimensions must match")
        if key_heads <= 0 or value_heads <= 0:
            raise VerifyException("RPA key and value head counts must be positive")
        if query_heads % key_heads or query_heads % value_heads:
            raise VerifyException("RPA query heads must divide evenly across key and value heads")
        if (output_batch, output_heads, output_dimension) != (
            batch,
            query_heads,
            value_dimension,
        ):
            raise VerifyException("RPA output must use batch/query heads and value dimension")
        if len(page_table.storage.get_shape()) != 2 or len(bias.storage.get_shape()) != 2:
            raise VerifyException("RPA page table and bias must be rank 2")
        if page_table.storage.get_shape()[0] != batch or lengths.storage.get_shape() != (batch,):
            raise VerifyException("RPA page metadata batch dimensions must match query")
        if bias.storage.get_shape()[0] != query_heads:
            raise VerifyException("RPA bias head dimension must match query")
        if bias.storage.get_shape()[1] < page_table.storage.get_shape()[1] * page_size:
            raise VerifyException("RPA bias extent must cover every addressable cache position")
        if self.kv_block_size.data <= 0 or page_size % self.kv_block_size.data:
            raise VerifyException("RPA KV block size must be positive and divide page size")
        if self.query_block_size.data <= 0:
            raise VerifyException("RPA query block size must be positive")


@irdl_op_definition
class FusedRaggedPagedAttentionOp(IRDLOperation):
    name = "tpu_schedule.fused_ragged_paged_attention"
    queries = operand_def(BufferType)
    keys = operand_def(BufferType)
    values = operand_def(BufferType)
    fused_cache = operand_def(BufferType)
    kv_lengths = operand_def(BufferType)
    page_indices = operand_def(BufferType)
    cumulative_query_lengths = operand_def(BufferType)
    cumulative_kv_lengths = operand_def(BufferType)
    distribution = operand_def(BufferType)
    relative_states = operand_def(BufferType)
    relative_projection = operand_def(BufferType)
    output = operand_def(BufferType)
    updated_cache = operand_def(BufferType)
    stage = prop_def(IntAttr)
    causal = prop_def(IntAttr)
    softmax_scale = prop_def(StringAttr)
    softmax_dtype = prop_def(StringAttr)
    sliding_window = prop_def(IntAttr)
    query_block_size = prop_def(IntAttr)
    kv_block_size = prop_def(IntAttr)
    query_cluster_size = prop_def(IntAttr)
    kv_cluster_size = prop_def(IntAttr)
    vmem_limit_bytes = prop_def(IntAttr)
    execution_authority = prop_def(StringAttr)
    donated_operand_indices = prop_def(ArrayAttr[IntAttr])

    def __init__(
        self,
        queries: SSAValue | Operation,
        keys: SSAValue | Operation,
        values: SSAValue | Operation,
        fused_cache: SSAValue | Operation,
        kv_lengths: SSAValue | Operation,
        page_indices: SSAValue | Operation,
        cumulative_query_lengths: SSAValue | Operation,
        cumulative_kv_lengths: SSAValue | Operation,
        distribution: SSAValue | Operation,
        relative_states: SSAValue | Operation,
        relative_projection: SSAValue | Operation,
        output: SSAValue | Operation,
        updated_cache: SSAValue | Operation,
        *,
        stage: int,
        causal: int,
        softmax_scale: str,
        softmax_dtype: str,
        sliding_window: int,
        query_block_size: int,
        kv_block_size: int,
        query_cluster_size: int,
        kv_cluster_size: int,
        vmem_limit_bytes: int,
    ) -> None:
        super().__init__(
            operands=[
                queries,
                keys,
                values,
                fused_cache,
                kv_lengths,
                page_indices,
                cumulative_query_lengths,
                cumulative_kv_lengths,
                distribution,
                relative_states,
                relative_projection,
                output,
                updated_cache,
            ],
            properties={
                "stage": IntAttr(stage),
                "causal": IntAttr(causal),
                "softmax_scale": StringAttr(softmax_scale),
                "softmax_dtype": StringAttr(softmax_dtype),
                "sliding_window": IntAttr(sliding_window),
                "query_block_size": IntAttr(query_block_size),
                "kv_block_size": IntAttr(kv_block_size),
                "query_cluster_size": IntAttr(query_cluster_size),
                "kv_cluster_size": IntAttr(kv_cluster_size),
                "vmem_limit_bytes": IntAttr(vmem_limit_bytes),
                "execution_authority": StringAttr("opaque-upstream-wrapper"),
                "donated_operand_indices": ArrayAttr(IntAttr(index) for index in (0, 1, 2, 3)),
            },
        )

    def verify_(self) -> None:
        buffers = tuple(value.type for value in self.operands)
        assert all(isinstance(value, BufferType) for value in buffers)
        for buffer in buffers:
            assert isinstance(buffer, BufferType)
            _check_live(buffer, self.stage.data)
            if buffer.space.data is not MemorySpace.HBM:
                raise VerifyException("fused RPA wrapper operands must reside in HBM")
        (
            queries,
            keys,
            values,
            fused_cache,
            kv_lengths,
            page_indices,
            cumulative_query_lengths,
            cumulative_kv_lengths,
            distribution,
            relative_states,
            relative_projection,
            output,
            updated_cache,
        ) = buffers
        assert all(isinstance(value, BufferType) for value in buffers)
        query_shape = queries.storage.get_shape()
        key_shape = keys.storage.get_shape()
        value_shape = values.storage.get_shape()
        cache_shape = fused_cache.storage.get_shape()
        if len(query_shape) != 3 or len(key_shape) != 3 or len(value_shape) != 3:
            raise VerifyException("fused RPA Q/K/V must have [tokens, heads, dim] shape")
        if len(cache_shape) != 5:
            raise VerifyException(
                "fused RPA cache must have [pages, page, packed_heads, packing, dim] shape"
            )
        tokens, query_heads, dimension = query_shape
        key_tokens, key_heads, key_dimension = key_shape
        value_tokens, value_heads, value_dimension = value_shape
        _, _, packed_interleaved_heads, packing, cache_dimension = cache_shape
        if any(value <= 0 for value in (*query_shape, *key_shape, *value_shape, *cache_shape)):
            raise VerifyException("fused RPA Q/K/V/cache dimensions must be positive")
        data_buffers = (
            queries,
            keys,
            values,
            fused_cache,
            relative_states,
            relative_projection,
            output,
            updated_cache,
        )
        if any(
            not isinstance(buffer.storage.element_type, BFloat16Type) for buffer in data_buffers
        ):
            raise VerifyException("Inkling fused RPA data buffers must use bf16")
        metadata_buffers = (
            kv_lengths,
            page_indices,
            cumulative_query_lengths,
            cumulative_kv_lengths,
            distribution,
        )
        if any(
            not isinstance(buffer.storage.element_type, IntegerType)
            or buffer.storage.element_type.width.data != 32
            or buffer.storage.element_type.signedness.data is Signedness.UNSIGNED
            for buffer in metadata_buffers
        ):
            raise VerifyException("fused RPA metadata buffers must use i32")
        if (key_tokens, value_tokens) != (tokens, tokens):
            raise VerifyException("fused RPA Q/K/V token counts must match")
        if key_heads != value_heads or query_heads % key_heads:
            raise VerifyException("fused RPA requires equal K/V heads dividing query heads")
        if (key_dimension, value_dimension) != (dimension, dimension):
            raise VerifyException("fused RPA Q/K/V head dimensions must match")
        padded_dimension = ((dimension + 127) // 128) * 128
        if cache_dimension != padded_dimension:
            raise VerifyException(
                "fused RPA cache head dimension must be padded to a multiple of 128"
            )
        if packing != 2 or packed_interleaved_heads * packing != 2 * key_heads:
            raise VerifyException("fused RPA cache must interleave one K and V per KV head")
        if output.storage.get_shape() != query_shape:
            raise VerifyException("fused RPA output shape must match queries")
        if updated_cache.storage.get_shape() != cache_shape:
            raise VerifyException("fused RPA updated cache shape must match its input cache")
        sequence_count_shape = kv_lengths.storage.get_shape()
        page_indices_shape = page_indices.storage.get_shape()
        if len(sequence_count_shape) != 1 or not sequence_count_shape[0]:
            raise VerifyException("fused RPA KV lengths must be nonempty rank 1")
        if len(page_indices_shape) != 1:
            raise VerifyException("fused RPA page indices must be rank 1")
        sequence_count = sequence_count_shape[0]
        if tokens != sequence_count:
            raise VerifyException("Inkling fused decode RPA requires one query token per sequence")
        if page_indices_shape[0] % sequence_count:
            raise VerifyException("fused RPA flat page indices must divide across sequences")
        if cumulative_query_lengths.storage.get_shape() != (sequence_count + 1,) or (
            cumulative_kv_lengths.storage.get_shape() != (sequence_count + 1,)
        ):
            raise VerifyException("fused RPA cumulative lengths must have sequences + 1 entries")
        if distribution.storage.get_shape() != (3,):
            raise VerifyException("fused RPA distribution must contain three boundaries")
        if relative_states.storage.get_shape()[:2] != (tokens, query_heads):
            raise VerifyException("fused RPA relative states must match query tokens and heads")
        if (
            len(relative_states.storage.get_shape()) != 3
            or len(relative_projection.storage.get_shape()) != 2
        ):
            raise VerifyException("fused RPA relative states/projection must be rank 3/rank 2")
        if relative_states.storage.get_shape()[2] != relative_projection.storage.get_shape()[0]:
            raise VerifyException("fused RPA relative dimensions must match")
        if self.causal.data != 1:
            raise VerifyException(
                "Inkling fused RPA adapter requires causal attention without a custom mask"
            )
        try:
            scale = Decimal(self.softmax_scale.data)
        except InvalidOperation as error:
            raise VerifyException("fused RPA softmax scale must be decimal") from error
        if not scale.is_finite() or scale <= 0:
            raise VerifyException("fused RPA softmax scale must be positive and finite")
        if scale * dimension != Decimal(1):
            raise VerifyException("Inkling fused RPA softmax scale must equal 1 / head dimension")
        if self.softmax_dtype.data != "float32":
            raise VerifyException("Inkling fused RPA softmax must use float32")
        if self.sliding_window.data < 0:
            raise VerifyException("fused RPA sliding window cannot be negative")
        if any(
            value.data <= 0
            for value in (
                self.query_block_size,
                self.kv_block_size,
                self.query_cluster_size,
                self.kv_cluster_size,
                self.vmem_limit_bytes,
            )
        ):
            raise VerifyException("fused RPA block sizes and VMEM limit must be positive")
        if self.execution_authority.data != "opaque-upstream-wrapper":
            raise VerifyException("fused RPA must declare delegated execution authority")
        if tuple(value.data for value in self.donated_operand_indices) != (0, 1, 2, 3):
            raise VerifyException("fused RPA must donate Q, K, V, and fused cache")
        if self.output is not self.queries or self.updated_cache is not self.fused_cache:
            raise VerifyException(
                "fused RPA output/cache destinations must alias query/cache inputs"
            )
        page_size = cache_shape[1]
        if (
            self.query_block_size.data % self.query_cluster_size.data
            or self.kv_block_size.data % self.kv_cluster_size.data
            or self.kv_block_size.data % page_size
            or self.kv_cluster_size.data % page_size
        ):
            raise VerifyException(
                "fused RPA blocks must divide into clusters and KV blocks must align to pages"
            )


@irdl_op_definition
class PipelineYieldOp(IRDLOperation):
    name = "tpu_schedule.pipeline_yield"
    values = var_operand_def(BufferType)
    traits = traits_def(IsTerminator())

    def __init__(self, *values: SSAValue | Operation):
        super().__init__(operands=[list(values)])

    def verify_(self) -> None:
        if not isinstance(self.parent_op(), PipelineLoopOp):
            raise VerifyException("tpu_schedule.pipeline_yield must terminate a pipeline loop")


def mxu_accumulator_scratch_bytes(operation: MxuMatmulOp | MxuEinsumOp) -> int:
    if isinstance(operation, MxuEinsumOp):
        return operation.tile_m.data * operation.tile_n.data * 4
    return 0


def pipeline_resident_memory_bytes(
    kernel: KernelOp,
) -> dict[MemorySpace, int]:
    def root(value: SSAValue) -> SSAValue:
        while isinstance(value.owner, ViewOp):
            value = value.owner.base
        return value

    resident = {MemorySpace.VMEM: 0, MemorySpace.SMEM: 0}
    for operation in kernel.body.block.ops:
        if not isinstance(operation, AllocOp):
            continue
        buffer = operation.buffer.type
        assert isinstance(buffer, BufferType)
        resident[buffer.space.data] += buffer_bytes(buffer)
    for operation in kernel.body.block.ops:
        if not isinstance(operation, PipelineLoopOp):
            continue
        rotations = tuple(value.data for value in operation.rotation_counts)
        if len(rotations) != len(operation.captures) or any(value <= 0 for value in rotations):
            raise VerifyException(
                "pipeline rotation counts must be positive and match captured buffers"
            )
        for capture, rotation in zip(operation.captures, rotations, strict=True):
            buffer = root(capture).type
            assert isinstance(buffer, BufferType)
            if buffer.ownership.data is not Ownership.EXTERNAL:
                resident[buffer.space.data] += buffer_bytes(buffer) * (rotation - 1)
    return resident


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
        if len(self.outputs) != len(self.captures) or len(terminator.values) != len(self.captures):
            raise VerifyException("pipeline results and yields must match captured buffers")
        for capture, argument, result, yielded in zip(
            self.captures,
            block.args,
            self.outputs,
            terminator.values,
            strict=True,
        ):
            if not (capture.type == argument.type == result.type == yielded.type):
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
            stage is None or stage < 0 or stage >= self.pipeline_stages.data for stage in stages
        ):
            raise VerifyException("pipeline operation stage is outside the declared pipeline")

        positions = {operation: index for index, operation in enumerate(operations)}
        initialized: set[SSAValue] = set(block.args)
        pending_dma: dict[Operation, DmaStartOp] = {}
        pending_remote_dma: dict[Operation, RemoteDmaStartOp] = {}

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
            elif isinstance(operation, RemoteDmaStartOp):
                if root(operation.source) not in initialized:
                    raise VerifyException("pipeline remote DMA reads an uninitialized source")
                semaphore = operation.semaphore.owner
                if semaphore in pending_dma or semaphore in pending_remote_dma:
                    raise VerifyException("pipeline semaphore is reused before its wait")
                uses = list(operation.token.uses)
                if len(uses) != 1 or not isinstance(uses[0].operation, RemoteDmaWaitOp):
                    raise VerifyException("pipeline remote DMA token must have exactly one wait")
                if positions[uses[0].operation] <= positions[operation]:
                    raise VerifyException("pipeline remote DMA wait must follow its start")
                pending_remote_dma[semaphore] = operation
                written.add(root(operation.destination))
            elif isinstance(operation, DmaWaitOp):
                start = operation.token.owner
                assert isinstance(start, DmaStartOp)
                pending_dma.pop(start.semaphore.owner, None)
                initialized.add(root(start.destination))
            elif isinstance(operation, RemoteDmaWaitOp):
                start = operation.token.owner
                assert isinstance(start, RemoteDmaStartOp)
                pending_remote_dma.pop(start.semaphore.owner, None)
                initialized.add(root(start.destination))
            elif isinstance(operation, (MxuMatmulOp, MxuEinsumOp)):
                if any(root(value) not in initialized for value in (operation.lhs, operation.rhs)):
                    raise VerifyException("pipeline MXU reads an uninitialized operand")
                if isinstance(operation, MxuEinsumOp) and len(operation.pending_reduction_axes) > 0:
                    raise VerifyException(
                        "pipeline MXU einsum does not yet support partial reductions"
                    )
                initialized.add(root(operation.accumulator))
                written.add(root(operation.accumulator))
            elif isinstance(operation, VectorComputeOp):
                if any(root(value) not in initialized for value in operation.inputs):
                    raise VerifyException(
                        "pipeline vector operation reads an uninitialized operand"
                    )
                initialized.add(root(operation.output))
                written.add(root(operation.output))
            elif _is_collective(operation):
                assert isinstance(operation, (CollectiveReduceScatterOp, CollectiveOp))
                if root(operation.source) not in initialized:
                    raise VerifyException("pipeline collective reads an uninitialized source")
                initialized.add(root(operation.destination))
                written.add(root(operation.destination))
            elif isinstance(operation, RaggedPagedAttentionOp):
                if any(root(value) not in initialized for value in operation.operands[:-1]):
                    raise VerifyException("pipeline RPA reads an uninitialized operand")
                initialized.add(root(operation.output))
                written.add(root(operation.output))
            elif isinstance(operation, FusedRaggedPagedAttentionOp):
                if any(root(value) not in initialized for value in operation.operands[:-2]):
                    raise VerifyException("pipeline fused RPA reads an uninitialized operand")
                for value in (operation.output, operation.updated_cache):
                    initialized.add(root(value))
                    written.add(root(value))
        if pending_dma or pending_remote_dma:
            raise VerifyException("pipeline iteration ends with DMA operations in flight")
        if any(root(value) not in initialized for value in terminator.values):
            raise VerifyException("pipeline yields an uninitialized buffer")

        for index, (argument, rotation_count) in enumerate(zip(block.args, rotations, strict=True)):
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
        transfer_plans: dict[str, TransferPlanAttr] = {}
        if kernel.physical_schema is not None:
            if kernel.topology is None:
                raise VerifyException("structured pipeline kernel needs a topology")
            topology_plans = kernel.topology.plans_by_id()
            topology_links = kernel.topology.links_by_id()
            transfer_plans = kernel.topology.transfer_plans_by_id()
            mesh = dict(
                zip(
                    (value.data for value in kernel.mesh_axis_names),
                    (value.data for value in kernel.mesh_axis_sizes),
                    strict=True,
                )
            )
            for operation in scheduled:
                if not _is_collective(operation):
                    continue
                assert isinstance(operation, (CollectiveReduceScatterOp, CollectiveOp))
                if operation.collective_plan is None:
                    raise VerifyException("structured pipeline collective needs a collective plan")
                plan = topology_plans.get(operation.collective_plan.data)
                if (
                    plan is None
                    or plan.mesh_axis.data != operation.mesh_axis.data
                    or operation.group_size.data != mesh.get(operation.mesh_axis.data)
                ):
                    raise VerifyException(
                        "pipeline collective references an incompatible collective plan"
                    )
            for operation in scheduled:
                if isinstance(operation, RemoteDmaStartOp) and (
                    operation.transfer_plan.data not in transfer_plans
                ):
                    raise VerifyException("pipeline remote DMA references an unknown transfer plan")
        horizon = (
            self.trip_count.data - 1
        ) * self.initiation_interval.data + self.pipeline_stages.data
        resident_bytes = pipeline_resident_memory_bytes(kernel)
        for absolute_stage in range(horizon):
            active_dma = 0
            active_remote_dma = 0
            mxu_uses = 0
            vector_uses = 0
            ici_uses = 0
            accumulator_scratch_bytes = 0
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
                    elif isinstance(operation, RemoteDmaStartOp):
                        wait = next(
                            use.operation
                            for use in operation.token.uses
                            if isinstance(use.operation, RemoteDmaWaitOp)
                        )
                        if operation.stage.data <= logical_stage <= wait.stage.data:
                            active_remote_dma += 1
                            owner = operation.semaphore.owner
                            semaphore_uses[owner] = semaphore_uses.get(owner, 0) + 1
                            plan = transfer_plans.get(operation.transfer_plan.data)
                            if plan is not None:
                                for route in plan.routes:
                                    for link_id in route.route_link_ids:
                                        link_uses[link_id.data] = link_uses.get(link_id.data, 0) + 1
                    elif isinstance(
                        operation,
                        (MxuMatmulOp, MxuEinsumOp, RaggedPagedAttentionOp),
                    ):
                        mxu_uses += operation.stage.data == logical_stage
                        if operation.stage.data == logical_stage and isinstance(
                            operation, (MxuMatmulOp, MxuEinsumOp)
                        ):
                            accumulator_scratch_bytes += mxu_accumulator_scratch_bytes(operation)
                    elif isinstance(operation, VectorComputeOp):
                        vector_uses += operation.stage.data == logical_stage
                    elif _is_collective(operation):
                        assert isinstance(operation, (CollectiveReduceScatterOp, CollectiveOp))
                        if operation.stage.data == logical_stage:
                            ici_uses += 1
                            if operation.collective_plan is not None:
                                plan = topology_plans.get(operation.collective_plan.data)
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
            if (
                kernel.remote_dma_engine_count is None
                or active_remote_dma > kernel.remote_dma_engine_count.data
            ):
                raise VerifyException(
                    f"pipeline exceeds remote DMA capacity at absolute stage {absolute_stage}"
                )
            if mxu_uses > kernel.mxu_count.data:
                raise VerifyException(
                    f"pipeline exceeds MXU capacity at absolute stage {absolute_stage}"
                )
            if vector_uses > kernel.vector_unit_count.data:
                raise VerifyException(
                    f"pipeline exceeds vector capacity at absolute stage {absolute_stage}"
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
            live_vmem_bytes = resident_bytes[MemorySpace.VMEM] + accumulator_scratch_bytes
            if live_vmem_bytes > kernel.vmem_capacity_bytes.data:
                raise VerifyException(
                    "pipeline VMEM capacity exceeded at absolute stage "
                    f"{absolute_stage}: {live_vmem_bytes} > "
                    f"{kernel.vmem_capacity_bytes.data}"
                )
            live_smem_bytes = resident_bytes[MemorySpace.SMEM]
            if live_smem_bytes > kernel.smem_capacity_bytes.data:
                raise VerifyException(
                    "pipeline SMEM capacity exceeded at absolute stage "
                    f"{absolute_stage}: {live_smem_bytes} > "
                    f"{kernel.smem_capacity_bytes.data}"
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
    topology_authority = opt_prop_def(StringAttr)
    argument_modes = opt_prop_def(ArrayAttr[StringAttr])
    dma_engine_count = prop_def(IntAttr)
    mxu_count = prop_def(IntAttr)
    vector_unit_count = prop_def(IntAttr)
    ici_link_count = prop_def(IntAttr)
    remote_dma_engine_count = opt_prop_def(IntAttr)
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
        remote_dma_engine_count: int = 1,
        argument_modes: tuple[str, ...] | None = None,
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
        properties = {
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
            "physical_schema": StringAttr("static-topology-v3"),
            "topology_authority": StringAttr("static-cost-model-only"),
            "dma_engine_count": IntAttr(dma_engine_count),
            "mxu_count": IntAttr(mxu_count),
            "vector_unit_count": IntAttr(vector_unit_count),
            "ici_link_count": IntAttr(ici_link_count),
            "remote_dma_engine_count": IntAttr(remote_dma_engine_count),
        }
        if argument_modes is not None:
            properties["argument_modes"] = ArrayAttr(StringAttr(mode) for mode in argument_modes)
        super().__init__(
            properties=properties,
            regions=[body],
        )

    def verify_(self) -> None:
        block = self.body.block
        modes: tuple[str, ...] | None = None
        if self.argument_modes is not None:
            modes = tuple(value.data for value in self.argument_modes)
            if len(modes) != len(block.args):
                raise VerifyException("kernel argument modes must match its arguments")
            if any(mode not in {"input", "output", "inout"} for mode in modes):
                raise VerifyException("kernel argument modes must be input, output, or inout")
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
        if self.physical_schema is not None and (
            self.remote_dma_engine_count is None or self.remote_dma_engine_count.data <= 0
        ):
            raise VerifyException("structured kernels need a positive remote DMA engine capacity")
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
            topology_device_ids: set[int] = set()
        else:
            if self.physical_schema.data != "static-topology-v3":
                raise VerifyException("unsupported physical schedule schema")
            if (
                self.topology_authority is None
                or self.topology_authority.data != "static-cost-model-only"
            ):
                raise VerifyException("structured kernels must declare static topology authority")
            if self.topology is None or self.interconnect is not None:
                raise VerifyException(
                    "structured kernels require topology and must not duplicate legacy interconnect"
                )
            try:
                self.topology.verify()
            except VerifyException as error:
                raise source_aware_error(str(error), self) from error
            topology_plans = self.topology.plans_by_id()
            topology_links = self.topology.links_by_id()
            topology_device_ids = {device.device_id.data for device in self.topology.devices}
            if len(self.topology.devices) != math.prod(mesh_sizes):
                raise VerifyException("topology device count must match the kernel mesh")
            coordinates = {
                tuple(value.data for value in device.coordinates)
                for device in self.topology.devices
            }
            expected_coordinates = set(product(*(range(size) for size in mesh_sizes)))
            if coordinates != expected_coordinates:
                raise VerifyException("topology coordinates must exactly cover the kernel mesh")
            plans_by_axis = {plan.mesh_axis.data: plan for plan in self.topology.collective_plans}
            expected_plan_axes = {axis for axis, size in mesh.items() if size > 1}
            if set(plans_by_axis) != expected_plan_axes:
                raise VerifyException(
                    "topology must declare one collective plan for every nontrivial mesh axis"
                )
            for axis, plan in plans_by_axis.items():
                if any(len(group.device_ids) != mesh[axis] for group in plan.groups):
                    raise VerifyException("collective plan group size must match its mesh axis")
                axis_index = mesh_names.index(axis)
                coordinates_by_id = {
                    device.device_id.data: tuple(value.data for value in device.coordinates)
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
                    varying = {coordinate[axis_index] for coordinate in group_coordinates}
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
        remote_in_flight: dict[Operation, RemoteDmaStartOp] = {}
        symbols: dict[str, int] = {}
        buffers: list[BufferType] = []
        storage_buffers: list[BufferType] = []
        views_by_root: dict[SSAValue, list[tuple[ViewOp, TileRegionAttr]]] = {}
        initialized: set[SSAValue] = (
            set(block.args)
            if modes is None
            else {
                argument
                for argument, mode in zip(block.args, modes, strict=True)
                if mode in {"input", "inout"}
            }
        )
        required_output_writes = (
            set()
            if modes is None
            else {
                argument
                for argument, mode in zip(block.args, modes, strict=True)
                if mode in {"output", "inout"}
            }
        )
        mode_by_argument = {} if modes is None else dict(zip(block.args, modes, strict=True))
        written_external_outputs: set[SSAValue] = set()
        partial_reductions: dict[SSAValue, frozenset[str]] = {}
        pending_dma_destinations: dict[Operation, tuple[SSAValue, TileRegionAttr]] = {}
        pending_remote_destinations: dict[Operation, tuple[SSAValue, TileRegionAttr]] = {}
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
            if any(base == pending[0] for pending in pending_remote_destinations.values()):
                raise VerifyException(
                    f"{operation.name} reads a buffer while remote DMA is in flight"
                )

        def require_fully_reduced(value: SSAValue, operation: Operation) -> None:
            pending = partial_reductions.get(root(value), frozenset())
            if pending:
                raise VerifyException(
                    f"{operation.name} consumes a partial reduction over {sorted(pending)}"
                )

        def mark_written(value: SSAValue) -> None:
            base = root(value)
            if mode_by_argument.get(base) == "input":
                raise VerifyException("kernel cannot write an input-only argument")
            if base in required_output_writes:
                written_external_outputs.add(base)

        def reject_overlapping_pending_write(
            destination_root: SSAValue,
            destination_region: TileRegionAttr,
            operation: Operation,
        ) -> None:
            pending = (
                *(
                    (root_value, region, in_flight[semaphore])
                    for semaphore, (root_value, region) in pending_dma_destinations.items()
                ),
                *(
                    (root_value, region, remote_in_flight[semaphore])
                    for semaphore, (root_value, region) in pending_remote_destinations.items()
                ),
            )
            for other_root, other_region, start in pending:
                if destination_root == other_root and _regions_overlap(
                    destination_region, other_region
                ):
                    raise source_aware_error(
                        "concurrent DMA writes target overlapping buffer regions",
                        start,
                        operation,
                    )

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
                require_fully_reduced(operation.source, operation)
                destination_root, destination_region = root_region(operation.destination)
                reject_overlapping_pending_write(destination_root, destination_region, operation)
                semaphore_owner = operation.semaphore.owner
                if semaphore_owner in in_flight or semaphore_owner in remote_in_flight:
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
            if isinstance(operation, RemoteDmaStartOp):
                require_initialized(operation.source, operation)
                require_fully_reduced(operation.source, operation)
                destination_root, destination_region = root_region(operation.destination)
                reject_overlapping_pending_write(destination_root, destination_region, operation)
                semaphore_owner = operation.semaphore.owner
                if semaphore_owner in in_flight or semaphore_owner in remote_in_flight:
                    raise VerifyException("semaphore reused before its DMA was waited on")
                if self.physical_schema is None or self.topology is None:
                    raise VerifyException("remote DMA requires a structured topology")
                transfer_plan = self.topology.transfer_plans_by_id().get(
                    operation.transfer_plan.data
                )
                if transfer_plan is None:
                    raise VerifyException("remote DMA references an unknown transfer plan")
                remote_in_flight[semaphore_owner] = operation
                uses = list(operation.token.uses)
                if len(uses) != 1 or not isinstance(uses[0].operation, RemoteDmaWaitOp):
                    raise VerifyException("every remote DMA token must have exactly one wait")
                if positions[uses[0].operation] <= positions[operation]:
                    raise VerifyException("remote DMA wait must occur after its start")
                pending_remote_destinations[semaphore_owner] = (
                    destination_root,
                    destination_region,
                )
            if isinstance(operation, DmaWaitOp):
                start = operation.token.owner
                assert isinstance(start, DmaStartOp)
                in_flight.pop(start.semaphore.owner, None)
                destination, _ = pending_dma_destinations.pop(start.semaphore.owner)
                initialized.add(destination)
                mark_written(destination)
                partial_reductions[destination] = partial_reductions.get(
                    root(start.source), frozenset()
                )
            if isinstance(operation, RemoteDmaWaitOp):
                start = operation.token.owner
                assert isinstance(start, RemoteDmaStartOp)
                remote_in_flight.pop(start.semaphore.owner, None)
                destination, _ = pending_remote_destinations.pop(start.semaphore.owner)
                assert self.topology is not None
                transfer_plan = self.topology.transfer_plans_by_id()[start.transfer_plan.data]
                covered_destinations = {
                    route.destination_device.data for route in transfer_plan.routes
                }
                if covered_destinations == topology_device_ids:
                    initialized.add(destination)
                    mark_written(destination)
                    partial_reductions[destination] = partial_reductions.get(
                        root(start.source), frozenset()
                    )
            if isinstance(operation, MxuMatmulOp):
                require_initialized(operation.lhs, operation)
                require_initialized(operation.rhs, operation)
                require_fully_reduced(operation.lhs, operation)
                require_fully_reduced(operation.rhs, operation)
                accumulator = root(operation.accumulator)
                initialized.add(accumulator)
                partial_reductions.pop(accumulator, None)
            if isinstance(operation, MxuEinsumOp):
                require_initialized(operation.lhs, operation)
                require_initialized(operation.rhs, operation)
                require_fully_reduced(operation.lhs, operation)
                require_fully_reduced(operation.rhs, operation)
                accumulator = root(operation.accumulator)
                initialized.add(accumulator)
                partial_reductions[accumulator] = frozenset(
                    value.data for value in operation.pending_reduction_axes
                )
            if isinstance(operation, VectorComputeOp):
                for value in operation.inputs:
                    require_initialized(value, operation)
                input_pending = tuple(
                    partial_reductions.get(root(value), frozenset()) for value in operation.inputs
                )
                declared_pending = frozenset(
                    value.data for value in operation.pending_reduction_axes
                )
                function = operation.function.data
                if function == "embedding_lookup":
                    if any(input_pending):
                        raise VerifyException(
                            "physical embedding cannot consume a partial reduction"
                        )
                elif function == "add":
                    if len(set(input_pending)) != 1 or input_pending[0] != declared_pending:
                        raise VerifyException(
                            "physical add must preserve matching pending reductions"
                        )
                elif function in {"rename_dimension", "slice"}:
                    if input_pending[0] != declared_pending:
                        raise VerifyException(
                            f"physical {function} must preserve pending reductions"
                        )
                else:
                    for value in operation.inputs:
                        require_fully_reduced(value, operation)
                    if declared_pending:
                        raise VerifyException(
                            f"physical {function} cannot introduce pending reductions"
                        )
                output = root(operation.output)
                initialized.add(output)
                partial_reductions[output] = declared_pending
            if _is_collective(operation):
                assert isinstance(operation, (CollectiveReduceScatterOp, CollectiveOp))
                require_initialized(operation.source, operation)
                axis = operation.mesh_axis.data
                if axis not in mesh:
                    raise VerifyException(f"collective references unknown mesh axis {axis}")
                if operation.group_size.data != mesh[axis]:
                    raise VerifyException("collective group size must match its kernel mesh axis")
                if self.physical_schema is None:
                    if links[axis] <= 0:
                        raise VerifyException("collective requires a usable interconnect link")
                else:
                    if operation.collective_plan is None:
                        raise VerifyException("structured collective needs a collective plan")
                    plan = topology_plans.get(operation.collective_plan.data)
                    if plan is None or plan.mesh_axis.data != axis:
                        raise VerifyException(
                            "collective references an incompatible collective plan"
                        )
                destination = root(operation.destination)
                remaining = set(partial_reductions.get(root(operation.source), frozenset()))
                reducing_collective = isinstance(operation, CollectiveReduceScatterOp) or (
                    isinstance(operation, CollectiveOp)
                    and operation.kind.data
                    in {CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER}
                )
                if reducing_collective and operation.reducer.data == "sum":
                    remaining.discard(axis)
                initialized.add(destination)
                partial_reductions[destination] = frozenset(remaining)
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
                    require_fully_reduced(value, operation)
                output = root(operation.output)
                initialized.add(output)
                partial_reductions.pop(output, None)
            if isinstance(operation, FusedRaggedPagedAttentionOp):
                for value in operation.operands[:-2]:
                    require_initialized(value, operation)
                    require_fully_reduced(value, operation)
                for value in (operation.output, operation.updated_cache):
                    output = root(value)
                    initialized.add(output)
                    mark_written(output)
                    partial_reductions.pop(output, None)
            if isinstance(operation, PipelineLoopOp):
                for value in operation.captures:
                    require_initialized(value, operation)
                    require_fully_reduced(value, operation)
                capture_roots = [root(value) for value in operation.captures]
                if len(capture_roots) != len(set(capture_roots)):
                    raise VerifyException("pipeline captures must reference distinct buffers")
                for value, rotation in zip(capture_roots, operation.rotation_counts, strict=True):
                    buffer = value.type
                    assert isinstance(buffer, BufferType)
                    if buffer.ownership.data is not Ownership.EXTERNAL and rotation.data > 1:
                        rotation_copies.append((buffer, rotation.data - 1))
                for value in operation.outputs:
                    output = root(value)
                    initialized.add(output)
                    mark_written(output)
                    partial_reductions.pop(output, None)

        missing_output_writes = required_output_writes - written_external_outputs
        if missing_output_writes:
            raise VerifyException("kernel does not write every output or inout argument")

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

        if in_flight or remote_in_flight:
            raise VerifyException("kernel ends with DMA operations still in flight")
        local_buffers = [
            buffer for buffer in storage_buffers if buffer.ownership.data is not Ownership.EXTERNAL
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
            active_remote_dma = sum(
                start.stage.data <= stage <= wait.operation.stage.data
                for start in (op for op in operations if isinstance(op, RemoteDmaStartOp))
                for wait in start.token.uses
                if isinstance(wait.operation, RemoteDmaWaitOp)
            )
            mxu_uses = sum(
                isinstance(
                    operation,
                    (MxuMatmulOp, MxuEinsumOp, RaggedPagedAttentionOp),
                )
                and operation.stage.data == stage
                for operation in operations
            )
            vector_uses = sum(
                isinstance(operation, VectorComputeOp) and operation.stage.data == stage
                for operation in operations
            )
            ici_uses = sum(
                _is_collective(operation) and operation.stage.data == stage
                for operation in operations
            )
            if active_dma > self.dma_engine_count.data:
                raise VerifyException(
                    f"DMA engine capacity exceeded at stage {stage}: "
                    f"{active_dma} > {self.dma_engine_count.data}"
                )
            if (
                self.physical_schema is not None
                and self.remote_dma_engine_count is not None
                and active_remote_dma > self.remote_dma_engine_count.data
            ):
                raise VerifyException(
                    f"remote DMA engine capacity exceeded at stage {stage}: "
                    f"{active_remote_dma} > {self.remote_dma_engine_count.data}"
                )
            if mxu_uses > self.mxu_count.data:
                raise VerifyException(
                    f"MXU capacity exceeded at stage {stage}: {mxu_uses} > {self.mxu_count.data}"
                )
            if vector_uses > self.vector_unit_count.data:
                raise VerifyException(
                    f"vector capacity exceeded at stage {stage}: "
                    f"{vector_uses} > {self.vector_unit_count.data}"
                )
            if ici_uses > self.ici_link_count.data:
                raise VerifyException(
                    f"ICI link capacity exceeded at stage {stage}: "
                    f"{ici_uses} > {self.ici_link_count.data}"
                )
            if self.physical_schema is not None:
                link_uses: dict[str, int] = {}
                for operation in operations:
                    if not _is_collective(operation):
                        continue
                    assert isinstance(operation, (CollectiveReduceScatterOp, CollectiveOp))
                    if operation.stage.data != stage or operation.collective_plan is None:
                        continue
                    plan = topology_plans[operation.collective_plan.data]
                    for group in plan.groups:
                        for link_id in group.route_link_ids:
                            link_uses[link_id.data] = link_uses.get(link_id.data, 0) + 1
                assert self.topology is not None
                transfer_plans = self.topology.transfer_plans_by_id()
                for operation in operations:
                    if not isinstance(operation, RemoteDmaStartOp):
                        continue
                    wait = next(
                        use.operation
                        for use in operation.token.uses
                        if isinstance(use.operation, RemoteDmaWaitOp)
                    )
                    if not operation.stage.data <= stage <= wait.stage.data:
                        continue
                    plan = transfer_plans[operation.transfer_plan.data]
                    for route in plan.routes:
                        for link_id in route.route_link_ids:
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
                if space is MemorySpace.VMEM:
                    live_bytes += sum(
                        mxu_accumulator_scratch_bytes(operation)
                        for operation in operations
                        if isinstance(operation, (MxuMatmulOp, MxuEinsumOp))
                        and operation.stage.data == stage
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
        RemoteDmaStartOp,
        RemoteDmaWaitOp,
        MxuMatmulOp,
        MxuEinsumOp,
        VectorComputeOp,
        CollectiveReduceScatterOp,
        CollectiveOp,
        RaggedPagedAttentionOp,
        FusedRaggedPagedAttentionOp,
        PipelineLoopOp,
        PipelineYieldOp,
        YieldOp,
    ],
    [
        MemorySpaceAttr,
        OwnershipAttr,
        CollectiveKindAttr,
        VectorMaterializationAttr,
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
        TransferRouteAttr,
        TransferPlanAttr,
        TopologyAttr,
        BufferType,
        DmaTokenType,
        SemaphoreType,
    ],
)
