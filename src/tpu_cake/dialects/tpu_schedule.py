from __future__ import annotations

import math
from enum import StrEnum

from xdsl.dialects.builtin import (
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


@irdl_attr_definition
class MemorySpaceAttr(EnumAttribute[MemorySpace], SpacedOpaqueSyntaxAttribute):
    name = "tpu_schedule.memory_space"


@irdl_attr_definition
class DmaTokenType(ParametrizedAttribute, TypeAttribute):
    name = "tpu_schedule.dma_token"


@irdl_attr_definition
class SemaphoreType(ParametrizedAttribute, TypeAttribute):
    name = "tpu_schedule.semaphore"


def _space(memref: MemRefType) -> MemorySpace:
    if not isinstance(memref.memory_space, MemorySpaceAttr):
        raise VerifyException("buffer must use #tpu_schedule.memory_space")
    return memref.memory_space.data


def _element_bytes(memref: MemRefType) -> int:
    element_type = memref.element_type
    if isinstance(element_type, BFloat16Type | Float16Type):
        return 2
    if isinstance(element_type, Float32Type):
        return 4
    if isinstance(element_type, IntegerType):
        return math.ceil(element_type.width.data / 8)
    raise VerifyException(f"unsupported buffer element type: {element_type}")


def buffer_bytes(memref: MemRefType) -> int:
    if not memref.has_static_shape():
        raise VerifyException("schedule buffers must have static shapes")
    return math.prod(memref.get_shape()) * _element_bytes(memref)


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "tpu_schedule.alloc"

    buffer = result_def(MemRefType)
    role = prop_def(StringAttr)

    def __init__(self, result_type: MemRefType, role: str | StringAttr):
        if isinstance(role, str):
            role = StringAttr(role)
        super().__init__(result_types=[result_type], properties={"role": role})

    def verify_(self) -> None:
        space = _space(self.buffer.type)
        if space is MemorySpace.HBM:
            raise VerifyException("HBM buffers are kernel inputs, not local allocations")
        buffer_bytes(self.buffer.type)


@irdl_op_definition
class SemaphoreAllocOp(IRDLOperation):
    name = "tpu_schedule.semaphore_alloc"

    semaphore = result_def(SemaphoreType)

    def __init__(self):
        super().__init__(result_types=[SemaphoreType()])


@irdl_op_definition
class DmaStartOp(IRDLOperation):
    name = "tpu_schedule.dma_start"

    source = operand_def(MemRefType)
    destination = operand_def(MemRefType)
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
        if isinstance(stage, int):
            stage = IntAttr(stage)
        super().__init__(
            operands=[source, destination, semaphore],
            result_types=[DmaTokenType()],
            properties={"stage": stage},
        )

    def verify_(self) -> None:
        source = self.source.type
        destination = self.destination.type
        assert isinstance(source, MemRefType)
        assert isinstance(destination, MemRefType)
        if _space(source) is _space(destination):
            raise VerifyException("DMA source and destination must use different memory spaces")
        if source.get_shape() != destination.get_shape():
            raise VerifyException("DMA source and destination shapes must match")
        if source.element_type != destination.element_type:
            raise VerifyException("DMA source and destination element types must match")
        buffer_bytes(source)


@irdl_op_definition
class DmaWaitOp(IRDLOperation):
    name = "tpu_schedule.dma_wait"

    token = operand_def(DmaTokenType)
    stage = prop_def(IntAttr)

    def __init__(self, token: SSAValue | Operation, stage: int | IntAttr):
        if isinstance(stage, int):
            stage = IntAttr(stage)
        super().__init__(operands=[token], properties={"stage": stage})

    def verify_(self) -> None:
        owner = self.token.owner
        if not isinstance(owner, DmaStartOp):
            raise VerifyException("DMA wait token must come from tpu_schedule.dma_start")
        if self.stage.data < owner.stage.data:
            raise VerifyException("DMA wait cannot precede its start stage")


@irdl_op_definition
class MxuMatmulOp(IRDLOperation):
    name = "tpu_schedule.mxu_matmul"

    lhs = operand_def(MemRefType)
    rhs = operand_def(MemRefType)
    accumulator = operand_def(MemRefType)
    stage = prop_def(IntAttr)

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        accumulator: SSAValue | Operation,
        stage: int | IntAttr,
    ):
        if isinstance(stage, int):
            stage = IntAttr(stage)
        super().__init__(operands=[lhs, rhs, accumulator], properties={"stage": stage})

    def verify_(self) -> None:
        lhs = self.lhs.type
        rhs = self.rhs.type
        accumulator = self.accumulator.type
        assert isinstance(lhs, MemRefType)
        assert isinstance(rhs, MemRefType)
        assert isinstance(accumulator, MemRefType)
        if any(_space(buffer) is not MemorySpace.VMEM for buffer in (lhs, rhs, accumulator)):
            raise VerifyException("MXU operands must be resident in VMEM")
        if any(len(buffer.get_shape()) != 2 for buffer in (lhs, rhs, accumulator)):
            raise VerifyException("MXU matmul requires rank-2 buffers")
        m, k = lhs.get_shape()
        rhs_k, n = rhs.get_shape()
        if k != rhs_k or accumulator.get_shape() != (m, n):
            raise VerifyException("MXU matmul shapes must be MxK, KxN, and MxN")
        if not isinstance(lhs.element_type, BFloat16Type | Float16Type):
            raise VerifyException("MXU input buffers must use bf16 or f16")
        if rhs.element_type != lhs.element_type:
            raise VerifyException("MXU input element types must match")
        if not isinstance(accumulator.element_type, Float32Type):
            raise VerifyException("MXU accumulation must use f32")


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

    traits = traits_def(IsolatedFromAbove())

    def __init__(
        self,
        sym_name: str | StringAttr,
        target: str | StringAttr,
        vmem_capacity_bytes: int | IntAttr,
        smem_capacity_bytes: int | IntAttr,
        body: Region,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(target, str):
            target = StringAttr(target)
        if isinstance(vmem_capacity_bytes, int):
            vmem_capacity_bytes = IntAttr(vmem_capacity_bytes)
        if isinstance(smem_capacity_bytes, int):
            smem_capacity_bytes = IntAttr(smem_capacity_bytes)
        super().__init__(
            properties={
                "sym_name": sym_name,
                "target": target,
                "vmem_capacity_bytes": vmem_capacity_bytes,
                "smem_capacity_bytes": smem_capacity_bytes,
            },
            regions=[body],
        )

    def verify_(self) -> None:
        block = self.body.block
        if not isinstance(block.last_op, YieldOp):
            raise VerifyException("kernel must end with tpu_schedule.yield")

        operations = list(block.ops)
        positions = {operation: index for index, operation in enumerate(operations)}
        previous_stage = -1
        in_flight: dict[Operation, DmaStartOp] = {}
        vmem_bytes = 0
        smem_bytes = 0

        for argument in block.args:
            if not isinstance(argument.type, MemRefType):
                raise VerifyException("kernel arguments must be memrefs")
            if _space(argument.type) is not MemorySpace.HBM:
                raise VerifyException("kernel arguments must reside in HBM")

        for operation in operations:
            stage = getattr(operation, "stage", None)
            if isinstance(stage, IntAttr):
                if stage.data < previous_stage:
                    raise VerifyException("scheduled stages must be monotonic")
                previous_stage = stage.data

            if isinstance(operation, AllocOp):
                size = buffer_bytes(operation.buffer.type)
                space = _space(operation.buffer.type)
                if space is MemorySpace.VMEM:
                    vmem_bytes += size
                elif space is MemorySpace.SMEM:
                    smem_bytes += size

            if isinstance(operation, DmaStartOp):
                semaphore_owner = operation.semaphore.owner
                if semaphore_owner in in_flight:
                    raise VerifyException("semaphore reused before its DMA was waited on")
                in_flight[semaphore_owner] = operation
                uses = list(operation.token.uses)
                if len(uses) != 1 or not isinstance(uses[0].operation, DmaWaitOp):
                    raise VerifyException("every DMA token must have exactly one DMA wait")
                if positions[uses[0].operation] <= positions[operation]:
                    raise VerifyException("DMA wait must occur after DMA start")

            if isinstance(operation, DmaWaitOp):
                start = operation.token.owner
                assert isinstance(start, DmaStartOp)
                in_flight.pop(start.semaphore.owner, None)

        if in_flight:
            raise VerifyException("kernel ends with DMA operations still in flight")
        if vmem_bytes > self.vmem_capacity_bytes.data:
            raise VerifyException(
                f"VMEM capacity exceeded: {vmem_bytes} > {self.vmem_capacity_bytes.data}"
            )
        if smem_bytes > self.smem_capacity_bytes.data:
            raise VerifyException(
                f"SMEM capacity exceeded: {smem_bytes} > {self.smem_capacity_bytes.data}"
            )


TPUSchedule = Dialect(
    "tpu_schedule",
    [
        KernelOp,
        AllocOp,
        SemaphoreAllocOp,
        DmaStartOp,
        DmaWaitOp,
        MxuMatmulOp,
        YieldOp,
    ],
    [MemorySpaceAttr, DmaTokenType, SemaphoreType],
)
