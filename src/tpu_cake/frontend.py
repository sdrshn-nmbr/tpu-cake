from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO

from xdsl.dialects.builtin import ArrayAttr, IntAttr, MemRefType, ModuleOp, StringAttr
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.printer import Printer

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    DmaStartOp,
    DmaWaitOp,
    KernelOp,
    LayoutAttr,
    LifetimeAttr,
    MemorySpace,
    MemorySpaceAttr,
    MxuMatmulOp,
    Ownership,
    OwnershipAttr,
    RaggedPagedAttentionOp,
    SemaphoreAllocOp,
    ShapeAttr,
    ShardingAttr,
    YieldOp,
)


@dataclass(frozen=True)
class BufferSpec:
    physical_shape: tuple[int, ...]
    logical_shape: tuple[str, ...]
    element_type: Attribute
    memory_space: MemorySpace
    sharding: tuple[str, ...]
    layout: tuple[int, ...]
    ownership: Ownership
    lifetime: tuple[int, int]

    def __post_init__(self) -> None:
        rank = len(self.physical_shape)
        if len(self.logical_shape) != rank or len(self.sharding) != rank:
            raise ValueError("physical shape, logical shape, and sharding must have equal rank")
        if sorted(self.layout) != list(range(rank)):
            raise ValueError("layout must be a rank permutation")

    def to_type(self) -> BufferType:
        return BufferType(
            MemRefType(self.element_type, self.physical_shape),
            ShapeAttr(ArrayAttr(StringAttr(value) for value in self.logical_shape)),
            MemorySpaceAttr(self.memory_space),
            ShardingAttr(ArrayAttr(StringAttr(value) for value in self.sharding)),
            LayoutAttr(ArrayAttr(IntAttr(value) for value in self.layout)),
            OwnershipAttr(self.ownership),
            LifetimeAttr(IntAttr(self.lifetime[0]), IntAttr(self.lifetime[1])),
        )


def buffer(
    shape: Iterable[int],
    logical: str | Iterable[str],
    element_type: Attribute,
    *,
    memory: MemorySpace,
    sharding: Iterable[str] | None = None,
    layout: Iterable[int] | None = None,
    ownership: Ownership = Ownership.KERNEL,
    lifetime: tuple[int, int] = (0, 0),
) -> BufferSpec:
    physical_shape = tuple(shape)
    logical_shape = tuple(logical.split()) if isinstance(logical, str) else tuple(logical)
    rank = len(physical_shape)
    return BufferSpec(
        physical_shape=physical_shape,
        logical_shape=logical_shape,
        element_type=element_type,
        memory_space=memory,
        sharding=tuple(sharding or ("",) * rank),
        layout=tuple(layout if layout is not None else range(rank)),
        ownership=ownership,
        lifetime=lifetime,
    )


class KernelBuilder:
    def __init__(
        self,
        name: str,
        target: str,
        inputs: Iterable[BufferSpec],
        *,
        vmem_capacity_bytes: int,
        smem_capacity_bytes: int,
    ) -> None:
        self._name = name
        self._target = target
        self._vmem_capacity_bytes = vmem_capacity_bytes
        self._smem_capacity_bytes = smem_capacity_bytes
        self._input_specs = tuple(inputs)
        self.block = Block(arg_types=[spec.to_type() for spec in self._input_specs])

    @property
    def inputs(self) -> tuple[SSAValue, ...]:
        return tuple(self.block.args)

    def _add(self, operation: Operation) -> Operation:
        self.block.add_op(operation)
        return operation

    def alloc(self, spec: BufferSpec, role: str) -> AllocOp:
        operation = AllocOp(spec.to_type(), role)
        self._add(operation)
        return operation

    def semaphore(self) -> SemaphoreAllocOp:
        operation = SemaphoreAllocOp()
        self._add(operation)
        return operation

    def dma(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        semaphore: SSAValue | Operation,
        *,
        start_stage: int,
        wait_stage: int,
    ) -> DmaStartOp:
        start = self.dma_start(source, destination, semaphore, stage=start_stage)
        self.dma_wait(start, stage=wait_stage)
        return start

    def dma_start(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        semaphore: SSAValue | Operation,
        *,
        stage: int,
    ) -> DmaStartOp:
        operation = DmaStartOp(source, destination, semaphore, stage)
        self._add(operation)
        return operation

    def dma_wait(self, token: SSAValue | Operation, *, stage: int) -> DmaWaitOp:
        operation = DmaWaitOp(token, stage)
        self._add(operation)
        return operation

    def matmul(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        accumulator: SSAValue | Operation,
        *,
        stage: int,
    ) -> MxuMatmulOp:
        operation = MxuMatmulOp(lhs, rhs, accumulator, stage)
        self._add(operation)
        return operation

    def ragged_paged_attention(
        self,
        query: SSAValue | Operation,
        key_cache: SSAValue | Operation,
        value_cache: SSAValue | Operation,
        page_table: SSAValue | Operation,
        sequence_lengths: SSAValue | Operation,
        bias: SSAValue | Operation,
        output: SSAValue | Operation,
        *,
        stage: int,
        query_block_size: int,
        kv_block_size: int,
    ) -> RaggedPagedAttentionOp:
        operation = RaggedPagedAttentionOp(
            query,
            key_cache,
            value_cache,
            page_table,
            sequence_lengths,
            bias,
            output,
            stage,
            query_block_size,
            kv_block_size,
        )
        self._add(operation)
        return operation

    def module(self) -> ModuleOp:
        self.block.add_op(YieldOp())
        kernel = KernelOp(
            self._name,
            self._target,
            self._vmem_capacity_bytes,
            self._smem_capacity_bytes,
            Region(self.block),
        )
        module = ModuleOp([kernel])
        module.verify()
        return module


def canonical_module_text(module: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue() + ("" if stream.getvalue().endswith("\n") else "\n")


def schedule_sha256(module: ModuleOp) -> str:
    return hashlib.sha256(canonical_module_text(module).encode()).hexdigest()
