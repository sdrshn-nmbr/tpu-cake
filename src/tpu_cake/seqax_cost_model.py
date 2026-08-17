from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    IntegerType,
    ModuleOp,
)
from xdsl.ir import Block, Operation, SSAValue

from tpu_cake.cost_model import HardwareRateModel
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
from tpu_cake.frontend import schedule_sha256
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)


class UnsupportedSeqaxCostModelError(ValueError):
    pass


class SeqaxPhysicalCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mesh_devices: int = Field(gt=0)
    operation_executions: tuple[tuple[str, int], ...]
    useful_global_mxu_flops: int = Field(ge=0)
    mxu_flops_per_device: int = Field(ge=0)
    useful_global_vector_flops: int = Field(ge=0)
    vector_flops_per_device: int = Field(ge=0)
    special_function_ops_per_device: int = Field(ge=0)
    index_and_compare_ops_per_device: int = Field(ge=0)
    global_logical_read_bytes: int = Field(ge=0)
    global_logical_write_bytes: int = Field(ge=0)
    local_logical_read_bytes_per_device: int = Field(ge=0)
    local_logical_write_bytes_per_device: int = Field(ge=0)
    aggregate_logical_tensor_bytes_across_mesh: int = Field(ge=0)
    minimum_hbm_read_bytes_per_device: int = Field(ge=0)
    minimum_hbm_write_bytes_per_device: int = Field(ge=0)
    materialized_hbm_bytes_per_device: int = Field(ge=0)
    ici_bidirectional_bytes_per_device: Decimal = Field(ge=0)
    peak_global_logical_live_bytes: int = Field(ge=0)
    peak_local_logical_live_bytes_per_device: int = Field(ge=0)


class CollectiveTraffic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    axes: tuple[str, ...]
    executions: int = Field(gt=0)
    bidirectional_bytes_per_device: Decimal = Field(gt=0)


class DeviceBalance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    derivable: bool
    devices: int = Field(gt=0)
    maximum_to_minimum_work_ratio: Decimal = Field(ge=1)
    reason: str


class SeqaxCostModelReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    program_name: str = Field(min_length=1)
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_scope: str = "multi-device-local-shards"
    mesh_axes: tuple[tuple[str, int], ...]
    canonical_operation_inventory: tuple[tuple[str, int], ...]
    hardware: HardwareRateModel
    counts: SeqaxPhysicalCounts
    collectives: tuple[CollectiveTraffic, ...]
    balance: DeviceBalance
    predicted_limiting_resource: str
    approximations: tuple[str, ...]
    compiler_materialization_assumptions: tuple[str, ...]
    omissions: tuple[str, ...]
    metrics: tuple[Metric, ...]


class _Work:
    __slots__ = (
        "global_mxu",
        "global_vector",
        "local_index",
        "local_mxu",
        "local_special",
        "local_vector",
    )

    def __init__(self) -> None:
        self.global_mxu = 0
        self.local_mxu = 0
        self.global_vector = 0
        self.local_vector = 0
        self.local_special = 0
        self.local_index = 0

    def add(self, other: _Work, multiplier: int = 1) -> None:
        self.global_mxu += other.global_mxu * multiplier
        self.local_mxu += other.local_mxu * multiplier
        self.global_vector += other.global_vector * multiplier
        self.local_vector += other.local_vector * multiplier
        self.local_special += other.local_special * multiplier
        self.local_index += other.local_index * multiplier


class _Traffic:
    __slots__ = (
        "collectives",
        "global_read",
        "global_write",
        "ici",
        "local_read",
        "local_write",
    )

    def __init__(self) -> None:
        self.global_read = 0
        self.global_write = 0
        self.local_read = 0
        self.local_write = 0
        self.ici = Decimal(0)
        self.collectives: dict[tuple[str, tuple[str, ...]], list[Decimal | int]] = {}

    def add(self, other: _Traffic, multiplier: int = 1) -> None:
        self.global_read += other.global_read * multiplier
        self.global_write += other.global_write * multiplier
        self.local_read += other.local_read * multiplier
        self.local_write += other.local_write * multiplier
        self.ici += other.ici * multiplier
        for key, (executions, byte_count) in other.collectives.items():
            current = self.collectives.setdefault(key, [0, Decimal(0)])
            current[0] += int(executions) * multiplier
            current[1] += Decimal(byte_count) * multiplier


class _Analysis:
    __slots__ = ("executions", "traffic", "work")

    def __init__(self) -> None:
        self.work = _Work()
        self.traffic = _Traffic()
        self.executions: Counter[str] = Counter()

    def add(self, other: _Analysis, multiplier: int = 1) -> None:
        self.work.add(other.work, multiplier)
        self.traffic.add(other.traffic, multiplier)
        self.executions.update(
            {name: count * multiplier for name, count in other.executions.items()}
        )


_CONTROL_OPERATIONS = (ProgramOp, ReturnOp, ScanYieldOp)
_DATA_OPERATIONS = (
    AllGatherOp,
    AllReduceOp,
    BroadcastOp,
    CastOp,
    EinsumLocalOp,
    EinsumOp,
    ElementwiseOp,
    EmbeddingLookupOp,
    MaskedSoftmaxOp,
    PackedCausalMaskOp,
    ReduceLocalOp,
    ReduceScatterOp,
    RenameDimensionOp,
    RmsNormOp,
    RotaryEmbeddingOp,
    SliceOp,
    TransposeOp,
)


def _dtype_bytes(value_type: DTensorType) -> int:
    element_type = value_type.element_type
    if isinstance(element_type, (BFloat16Type, Float16Type)):
        return 2
    if isinstance(element_type, Float32Type):
        return 4
    if isinstance(element_type, IntegerType) and element_type.width.data in {
        1,
        8,
        16,
        32,
        64,
    }:
        return max(1, element_type.width.data // 8)
    raise UnsupportedSeqaxCostModelError(
        f"unsupported distributed tensor element type {element_type}"
    )


def _tensor_type(value: SSAValue) -> DTensorType:
    if not isinstance(value.type, DTensorType):
        raise UnsupportedSeqaxCostModelError(
            f"cost model expected a distributed tensor, got {value.type}"
        )
    return value.type


def _global_elements(value_type: DTensorType) -> int:
    return math.prod(size for _, size in value_type.logical_shape())


def _sharding_factor(value_type: DTensorType, mesh: dict[str, int]) -> int:
    axes = tuple(axis for dimension in value_type.sharding_axes() for axis in dimension)
    if len(axes) != len(set(axes)):
        raise UnsupportedSeqaxCostModelError(
            "a tensor uses one mesh axis on more than one dimension"
        )
    unknown = set(axes) - set(mesh)
    if unknown:
        raise UnsupportedSeqaxCostModelError(
            f"tensor references unknown mesh axes {sorted(unknown)}"
        )
    factor = math.prod(mesh[axis] for axis in axes)
    elements = _global_elements(value_type)
    if elements % factor:
        raise UnsupportedSeqaxCostModelError(
            "logical tensor elements do not divide their sharding factor"
        )
    return factor


def _global_bytes(value_type: DTensorType) -> int:
    return _global_elements(value_type) * _dtype_bytes(value_type)


def _local_bytes(value_type: DTensorType, mesh: dict[str, int]) -> int:
    return _global_bytes(value_type) // _sharding_factor(value_type, mesh)


def _local_elements(value_type: DTensorType, mesh: dict[str, int]) -> int:
    return _global_elements(value_type) // _sharding_factor(value_type, mesh)


def _einsum_counts(
    operation: EinsumOp | EinsumLocalOp,
    mesh: dict[str, int],
) -> tuple[int, int]:
    lhs_type = _tensor_type(operation.lhs)
    rhs_type = _tensor_type(operation.rhs)
    dimensions: dict[str, tuple[int, tuple[str, ...]]] = {}
    for value_type in (lhs_type, rhs_type):
        for (name, size), axes in zip(
            value_type.logical_shape(), value_type.sharding_axes(), strict=True
        ):
            previous = dimensions.setdefault(name, (size, axes))
            if previous != (size, axes):
                raise UnsupportedSeqaxCostModelError(
                    f"einsum dimension {name} has inconsistent shape or sharding"
                )
    global_flops = 2 * math.prod(size for size, _ in dimensions.values())
    axes = {axis for _, sharding in dimensions.values() for axis in sharding}
    shard_factor = math.prod(mesh[axis] for axis in axes)
    if global_flops % shard_factor:
        raise UnsupportedSeqaxCostModelError(
            "einsum work does not divide evenly over the declared mesh"
        )
    return global_flops, global_flops // shard_factor


def _operation_work(operation: Operation, mesh: dict[str, int]) -> _Work:
    work = _Work()
    if isinstance(operation, (EinsumOp, EinsumLocalOp)):
        work.global_mxu, work.local_mxu = _einsum_counts(operation, mesh)
        return work
    if not isinstance(operation, _DATA_OPERATIONS):
        raise UnsupportedSeqaxCostModelError(
            f"no physical work model for operation {operation.name}"
        )
    if not operation.results:
        return work
    result_type = _tensor_type(operation.results[0])
    global_elements = _global_elements(result_type)
    local_elements = _local_elements(result_type, mesh)
    if isinstance(operation, ElementwiseOp):
        function = operation.function.data
        if function in {"add", "multiply"}:
            global_factor = local_factor = 1
        elif function == "silu":
            global_factor = local_factor = 4
            work.local_special += local_elements
        elif function == "exp":
            global_factor = local_factor = 0
            work.local_special += local_elements
        elif function == "relu":
            global_factor = local_factor = 0
            work.local_index += local_elements
        else:
            raise UnsupportedSeqaxCostModelError(
                f"no explicit scalar work convention for elementwise {function}"
            )
        work.global_vector += global_elements * global_factor
        work.local_vector += local_elements * local_factor
    elif isinstance(operation, RmsNormOp):
        dimension_size = dict(result_type.logical_shape())[operation.dimension.data]
        global_rows = global_elements // dimension_size
        local_rows = local_elements // dimension_size
        work.global_vector += 4 * global_elements
        work.local_vector += 4 * local_elements
        work.local_special += local_rows
        if global_rows <= 0:
            raise UnsupportedSeqaxCostModelError("RMSNorm has no normalization rows")
    elif isinstance(operation, RotaryEmbeddingOp):
        work.global_vector += 3 * global_elements
        work.local_vector += 3 * local_elements
        work.local_special += local_elements
    elif isinstance(operation, MaskedSoftmaxOp):
        dimension_size = dict(result_type.logical_shape())[operation.dimension.data]
        global_rows = global_elements // dimension_size
        local_rows = local_elements // dimension_size
        work.global_vector += 3 * global_elements - global_rows
        work.local_vector += 3 * local_elements - local_rows
        work.local_special += local_elements
        work.local_index += local_elements
    elif isinstance(operation, ReduceLocalOp):
        input_type = _tensor_type(operation.value)
        work.global_vector += _global_elements(input_type) - global_elements
        work.local_vector += _local_elements(input_type, mesh) - local_elements
    elif isinstance(
        operation,
        (
            BroadcastOp,
            CastOp,
            EmbeddingLookupOp,
            PackedCausalMaskOp,
            SliceOp,
            TransposeOp,
        ),
    ):
        work.local_index += local_elements
    elif isinstance(
        operation,
        (AllGatherOp, AllReduceOp, ReduceScatterOp, RenameDimensionOp),
    ):
        pass
    else:
        raise UnsupportedSeqaxCostModelError(
            f"no physical work model for operation {operation.name}"
        )
    return work


def _collective_axes(operation: Operation) -> tuple[str, ...]:
    if isinstance(operation, AllGatherOp):
        before = _tensor_type(operation.value)
        after = _tensor_type(operation.result)
        removed: list[str] = []
        removed_dimensions = 0
        for old_axes, new_axes in zip(
            before.sharding_axes(), after.sharding_axes(), strict=True
        ):
            if old_axes[: len(new_axes)] != new_axes:
                raise UnsupportedSeqaxCostModelError(
                    "all-gather result does not remove a sharding suffix"
                )
            suffix = old_axes[len(new_axes) :]
            if suffix:
                removed_dimensions += 1
                removed.extend(suffix)
        if removed_dimensions != 1:
            raise UnsupportedSeqaxCostModelError(
                "all-gather cost model requires sharding removal from exactly one dimension"
            )
        axes = tuple(sorted(removed))
    elif isinstance(operation, (AllReduceOp, ReduceScatterOp)):
        axes = tuple(value.data for value in operation.axes)
    else:
        raise UnsupportedSeqaxCostModelError(
            f"cannot derive collective axes for {operation.name}"
        )
    if not axes or len(axes) != len(set(axes)):
        raise UnsupportedSeqaxCostModelError(
            f"collective {operation.name} needs nonempty unique mesh axes"
        )
    return axes


def _collective_traffic(
    operation: AllGatherOp | AllReduceOp | ReduceScatterOp,
    mesh: dict[str, int],
) -> tuple[tuple[str, ...], Decimal]:
    axes = _collective_axes(operation)
    group_size = math.prod(mesh[axis] for axis in axes)
    if group_size <= 1:
        raise UnsupportedSeqaxCostModelError(
            f"collective {operation.name} has a one-device group"
        )
    before_bytes = _local_bytes(_tensor_type(operation.value), mesh)
    after_bytes = _local_bytes(_tensor_type(operation.result), mesh)
    if isinstance(operation, AllGatherOp):
        if after_bytes != before_bytes * group_size:
            raise UnsupportedSeqaxCostModelError(
                "all-gather local result bytes do not match its group size"
            )
        byte_count = Decimal(2 * after_bytes * (group_size - 1)) / Decimal(group_size)
    elif isinstance(operation, ReduceScatterOp):
        if before_bytes != after_bytes * group_size:
            raise UnsupportedSeqaxCostModelError(
                "reduce-scatter local input bytes do not match its group size"
            )
        byte_count = Decimal(2 * before_bytes * (group_size - 1)) / Decimal(group_size)
    else:
        if before_bytes != after_bytes:
            raise UnsupportedSeqaxCostModelError(
                "all-reduce must preserve local tensor bytes"
            )
        byte_count = Decimal(4 * before_bytes * (group_size - 1)) / Decimal(group_size)
    return axes, byte_count


def _operation_traffic(operation: Operation, mesh: dict[str, int]) -> _Traffic:
    traffic = _Traffic()
    if not isinstance(operation, _DATA_OPERATIONS):
        raise UnsupportedSeqaxCostModelError(
            f"no tensor traffic model for operation {operation.name}"
        )
    for operand in operation.operands:
        value_type = _tensor_type(operand)
        traffic.global_read += _global_bytes(value_type)
        traffic.local_read += _local_bytes(value_type, mesh)
    for result in operation.results:
        value_type = _tensor_type(result)
        traffic.global_write += _global_bytes(value_type)
        traffic.local_write += _local_bytes(value_type, mesh)
    if isinstance(operation, (AllGatherOp, AllReduceOp, ReduceScatterOp)):
        axes, byte_count = _collective_traffic(operation, mesh)
        key = (operation.name.removeprefix("dtensor."), axes)
        traffic.ici += byte_count
        traffic.collectives[key] = [1, byte_count]
    return traffic


def _analyze_block(block: Block, mesh: dict[str, int]) -> _Analysis:
    analysis = _Analysis()
    for operation in block.ops:
        if isinstance(operation, (ReturnOp, ScanYieldOp)):
            continue
        if isinstance(operation, LayerScanOp):
            analysis.executions[operation.name] += 1
            nested = _analyze_block(operation.body.block, mesh)
            analysis.add(nested, operation.trip_count.data)
            continue
        if isinstance(operation, ProgramOp):
            raise UnsupportedSeqaxCostModelError("nested distributed programs are unsupported")
        if not isinstance(operation, _DATA_OPERATIONS):
            raise UnsupportedSeqaxCostModelError(
                f"no Seqax physical cost model for operation {operation.name}"
            )
        analysis.executions[operation.name] += 1
        analysis.work.add(_operation_work(operation, mesh))
        analysis.traffic.add(_operation_traffic(operation, mesh))
    return analysis


def _remaining_uses(block: Block) -> Counter[SSAValue]:
    uses: Counter[SSAValue] = Counter()
    for operation in block.ops:
        uses.update(operation.operands)
    return uses


def _block_peak_live_bytes(
    block: Block,
    mesh: dict[str, int],
    *,
    count_arguments: bool,
) -> tuple[int, int]:
    remaining = _remaining_uses(block)
    global_live = 0
    local_live = 0
    if count_arguments:
        for argument in block.args:
            value_type = _tensor_type(argument)
            global_live += _global_bytes(value_type)
            local_live += _local_bytes(value_type, mesh)
    peak_global = global_live
    peak_local = local_live
    for operation in block.ops:
        result_global = sum(_global_bytes(_tensor_type(value)) for value in operation.results)
        result_local = sum(
            _local_bytes(_tensor_type(value), mesh) for value in operation.results
        )
        nested_global = nested_local = 0
        if isinstance(operation, LayerScanOp):
            nested_global, nested_local = _block_peak_live_bytes(
                operation.body.block,
                mesh,
                count_arguments=False,
            )
        peak_global = max(peak_global, global_live + result_global + nested_global)
        peak_local = max(peak_local, local_live + result_local + nested_local)
        global_live += result_global
        local_live += result_local
        for operand in operation.operands:
            remaining[operand] -= 1
            if remaining[operand] == 0 and (
                operand.owner is not block or count_arguments
            ):
                value_type = _tensor_type(operand)
                global_live -= _global_bytes(value_type)
                local_live -= _local_bytes(value_type, mesh)
        for result in operation.results:
            if remaining[result] == 0:
                value_type = _tensor_type(result)
                global_live -= _global_bytes(value_type)
                local_live -= _local_bytes(value_type, mesh)
    if global_live < 0 or local_live < 0:
        raise UnsupportedSeqaxCostModelError("logical liveness accounting became negative")
    return peak_global, peak_local


def _metric(
    name: str,
    value: Decimal | int,
    unit: Unit,
    source: MetricSource,
    formula_name: str,
    expression: str,
    *,
    scope: str = "one complete logical Seqax forward on one JAX TPU device",
    numerator: Quantity | None = None,
    denominator: Quantity | None = None,
) -> Metric:
    return Metric(
        name=name,
        quantity=Quantity(value=Decimal(value), unit=unit),
        kind=MeasurementKind.ESTIMATED,
        interval=MeasurementInterval(scope=scope),
        sources=(source,),
        formula=FormulaIdentity(name=formula_name, version="1", expression=expression),
        numerator=numerator,
        denominator=denominator,
    )


def estimate_seqax_forward(
    module: ModuleOp,
    *,
    hardware: HardwareRateModel,
    source: MetricSource,
    expected_schedule_sha256: str | None = None,
) -> SeqaxCostModelReport:
    module.verify()
    top_level = tuple(module.body.block.ops)
    if len(top_level) != 1 or not isinstance(top_level[0], ProgramOp):
        raise UnsupportedSeqaxCostModelError(
            "Seqax cost model expects one top-level distributed program"
        )
    program = top_level[0]
    actual_schedule_sha256 = schedule_sha256(module)
    if (
        expected_schedule_sha256 is not None
        and actual_schedule_sha256 != expected_schedule_sha256
    ):
        raise UnsupportedSeqaxCostModelError(
            "distributed schedule hash mismatch: "
            f"expected {expected_schedule_sha256}, got {actual_schedule_sha256}"
        )
    mesh = program.mesh.sizes()
    if not mesh:
        raise UnsupportedSeqaxCostModelError("Seqax physical cost model needs a device mesh")
    mesh_devices = math.prod(mesh.values())
    if mesh_devices <= 1:
        raise UnsupportedSeqaxCostModelError(
            "Seqax physical cost model needs more than one mesh device"
        )
    for operation in program.walk():
        for value in (*operation.operands, *operation.results):
            if isinstance(value.type, DTensorType):
                _dtype_bytes(value.type)
    analysis = _analyze_block(program.body.block, mesh)
    peak_global, peak_local = _block_peak_live_bytes(
        program.body.block,
        mesh,
        count_arguments=True,
    )
    terminator = program.body.block.last_op
    if not isinstance(terminator, ReturnOp):
        raise UnsupportedSeqaxCostModelError("distributed program has no return")
    minimum_hbm_read = sum(
        _local_bytes(_tensor_type(argument), mesh) for argument in program.body.block.args
    )
    minimum_hbm_write = sum(
        _local_bytes(_tensor_type(value), mesh) for value in terminator.values
    )
    materialized_hbm = analysis.traffic.local_read + analysis.traffic.local_write
    minimum_hbm = minimum_hbm_read + minimum_hbm_write
    compute_ns = (
        Decimal(analysis.work.local_mxu)
        * Decimal(1_000_000_000)
        / Decimal(hardware.compute_flops_per_second)
    )
    hbm_ns = (
        Decimal(minimum_hbm)
        * Decimal(1_000_000_000)
        / Decimal(hardware.hbm_bytes_per_second)
    )
    materialized_hbm_ns = (
        Decimal(materialized_hbm)
        * Decimal(1_000_000_000)
        / Decimal(hardware.hbm_bytes_per_second)
    )
    ici_ns = (
        analysis.traffic.ici
        * Decimal(1_000_000_000)
        / Decimal(hardware.ici_bytes_per_second)
    )
    lower_bound_ns = max(compute_ns, hbm_ns, ici_ns)
    limiting = max(
        (("compute", compute_ns), ("hbm", hbm_ns), ("ici", ici_ns)),
        key=lambda item: item[1],
    )[0]
    total_local_work = (
        analysis.work.local_mxu
        + analysis.work.local_vector
        + analysis.work.local_special
        + analysis.work.local_index
    )
    logical_local_bytes = analysis.traffic.local_read + analysis.traffic.local_write
    logical_global_bytes = analysis.traffic.global_read + analysis.traffic.global_write
    balance_basis = total_local_work or logical_local_bytes
    if balance_basis <= 0:
        raise UnsupportedSeqaxCostModelError("distributed program has no modeled work or traffic")
    imbalance_numerator = Quantity(value=Decimal(balance_basis), unit=Unit.COUNT)
    imbalance_denominator = Quantity(value=Decimal(balance_basis), unit=Unit.COUNT)
    counts = SeqaxPhysicalCounts(
        mesh_devices=mesh_devices,
        operation_executions=tuple(sorted(analysis.executions.items())),
        useful_global_mxu_flops=analysis.work.global_mxu,
        mxu_flops_per_device=analysis.work.local_mxu,
        useful_global_vector_flops=analysis.work.global_vector,
        vector_flops_per_device=analysis.work.local_vector,
        special_function_ops_per_device=analysis.work.local_special,
        index_and_compare_ops_per_device=analysis.work.local_index,
        global_logical_read_bytes=analysis.traffic.global_read,
        global_logical_write_bytes=analysis.traffic.global_write,
        local_logical_read_bytes_per_device=analysis.traffic.local_read,
        local_logical_write_bytes_per_device=analysis.traffic.local_write,
        aggregate_logical_tensor_bytes_across_mesh=logical_local_bytes * mesh_devices,
        minimum_hbm_read_bytes_per_device=minimum_hbm_read,
        minimum_hbm_write_bytes_per_device=minimum_hbm_write,
        materialized_hbm_bytes_per_device=materialized_hbm,
        ici_bidirectional_bytes_per_device=analysis.traffic.ici,
        peak_global_logical_live_bytes=peak_global,
        peak_local_logical_live_bytes_per_device=peak_local,
    )
    collectives = tuple(
        CollectiveTraffic(
            kind=kind,
            axes=axes,
            executions=int(values[0]),
            bidirectional_bytes_per_device=Decimal(values[1]),
        )
        for (kind, axes), values in sorted(analysis.traffic.collectives.items())
    )
    metrics = (
        _metric(
            "seqax_mxu_flops_per_device",
            analysis.work.local_mxu,
            Unit.FLOP,
            source,
            "seqax_einsum_flops",
            "sum(2*product(unique_einsum_dimensions)/sharding_factor)*scan_trip_count",
        ),
        _metric(
            "seqax_vector_flops_per_device",
            analysis.work.local_vector,
            Unit.FLOP,
            source,
            "seqax_declared_vector_flops",
            "sum(operation_vector_flop_convention*local_elements)*scan_trip_count",
        ),
        _metric(
            "seqax_global_logical_tensor_bytes",
            logical_global_bytes,
            Unit.BYTE,
            source,
            "seqax_global_logical_traffic",
            "sum(global_operand_bytes+global_result_bytes)*scan_trip_count",
            scope="one complete logical Seqax forward across the full device mesh",
        ),
        _metric(
            "seqax_local_logical_tensor_bytes_per_device",
            logical_local_bytes,
            Unit.BYTE,
            source,
            "seqax_local_logical_traffic",
            "sum(local_operand_bytes+local_result_bytes)*scan_trip_count",
        ),
        _metric(
            "seqax_minimum_hbm_bytes_per_device",
            minimum_hbm,
            Unit.BYTE,
            source,
            "seqax_program_io_hbm_floor",
            "sum(local_program_input_bytes)+sum(local_return_bytes)",
        ),
        _metric(
            "seqax_materialized_hbm_bytes_per_device",
            materialized_hbm,
            Unit.BYTE,
            source,
            "seqax_unfused_materialization_traffic",
            "sum(local_operand_bytes+local_result_bytes)*scan_trip_count",
        ),
        _metric(
            "seqax_ici_bidirectional_bytes_per_device",
            analysis.traffic.ici,
            Unit.BYTE,
            source,
            "seqax_ring_equivalent_collective_traffic",
            "sum(collective_ring_equivalent_send_plus_receive_bytes)*scan_trip_count",
        ),
        _metric(
            "seqax_peak_local_logical_live_bytes_per_device",
            peak_local,
            Unit.BYTE,
            source,
            "seqax_ssa_liveness",
            "max(sum(bytes_of_live_local_ssa_values))",
        ),
        _metric(
            "seqax_compute_time_floor",
            compute_ns,
            Unit.NANOSECOND,
            source,
            "seqax_mxu_compute_floor",
            "mxu_flops_per_device/advertised_mxu_flops_per_second",
        ),
        _metric(
            "seqax_hbm_time_floor",
            hbm_ns,
            Unit.NANOSECOND,
            source,
            "seqax_program_io_hbm_floor",
            "minimum_hbm_bytes_per_device/advertised_hbm_bytes_per_second",
        ),
        _metric(
            "seqax_materialized_hbm_time",
            materialized_hbm_ns,
            Unit.NANOSECOND,
            source,
            "seqax_unfused_hbm_scenario",
            "materialized_hbm_bytes_per_device/advertised_hbm_bytes_per_second",
        ),
        _metric(
            "seqax_ici_time_floor",
            ici_ns,
            Unit.NANOSECOND,
            source,
            "seqax_ring_equivalent_ici_floor",
            "ici_bidirectional_bytes_per_device/advertised_ici_bytes_per_second",
        ),
        _metric(
            "seqax_idealized_time_floor",
            lower_bound_ns,
            Unit.NANOSECOND,
            source,
            "seqax_overlapped_resource_floor",
            "max(compute_time_floor,hbm_time_floor,ici_time_floor)",
        ),
        _metric(
            "seqax_device_work_imbalance_ratio",
            Decimal(1),
            Unit.RATIO,
            source,
            "seqax_uniform_static_sharding_balance",
            "maximum_per_device_declared_work/minimum_per_device_declared_work",
            scope="one complete logical Seqax forward across the full device mesh",
            numerator=imbalance_numerator,
            denominator=imbalance_denominator,
        ),
    )
    return SeqaxCostModelReport(
        program_name=program.sym_name.data,
        schedule_sha256=actual_schedule_sha256,
        mesh_axes=tuple(mesh.items()),
        canonical_operation_inventory=tuple(
            sorted(Counter(operation.name for operation in module.walk()).items())
        ),
        hardware=hardware,
        counts=counts,
        collectives=collectives,
        balance=DeviceBalance(
            derivable=True,
            devices=mesh_devices,
            maximum_to_minimum_work_ratio=Decimal(1),
            reason=(
                "All represented dimensions divide static mesh axes exactly; the IR has no "
                "ragged or data-dependent routing operation."
            ),
        ),
        predicted_limiting_resource=limiting,
        approximations=(
            "MXU work counts multiply-add as two floating-point operations.",
            (
                "RMSNorm counts four ordinary floating-point operations per element and one "
                "reciprocal-square-root per normalization row."
            ),
            "SiLU counts four ordinary floating-point operations and one exponential per element.",
            (
                "Masked softmax counts subtract, reduction, divide, compare, and exponential work; "
                "rotary embedding counts three ordinary operations and one special function per "
                "element."
            ),
            "Collective traffic uses ring-equivalent send-plus-receive bytes and ignores startup.",
            (
                "Uniform static sharding implies a 1.0 logical work imbalance; runtime and topology "
                "imbalance remain unmeasured."
            ),
        ),
        compiler_materialization_assumptions=(
            (
                "The strict HBM floor reads every local program input once and writes every returned "
                "value once; it assumes intermediates can remain on chip."
            ),
            (
                "The materialized HBM scenario reads every operation operand and writes every "
                "result, including view-like operations; compiler fusion, aliasing, reloads, "
                "padding, and layout conversions can move real traffic below or above that scenario."
            ),
            (
                "Peak live bytes count logical SSA values, not allocated HBM, VMEM, or compiler "
                "buffer reuse. Layer-scan body arguments alias their captured values in this "
                "accounting."
            ),
        ),
        omissions=(
            (
                "Vector, special-function, index, launch, synchronization, and "
                "collective-startup time are not assigned hardware rates and are absent from the "
                "idealized time floor."
            ),
            "The model does not choose collective algorithms or physical routes.",
            (
                "The report is calculated from verified IR and advertised rates; none of its "
                "values are device measurements."
            ),
        ),
        metrics=metrics,
    )
