from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntAttr, MemRefType, ModuleOp, StringAttr
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from tpu_cake.canonical import canonical_sha256, canonical_text
from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    CollectiveKind,
    CollectiveOp,
    DmaStartOp,
    DmaWaitOp,
    FusedRaggedPagedAttentionOp,
    KernelOp,
    LayoutAttr,
    LifetimeAttr,
    MemorySpace,
    MemorySpaceAttr,
    MxuMatmulOp,
    Ownership,
    OwnershipAttr,
    RaggedPagedAttentionOp,
    RemoteDmaStartOp,
    RemoteDmaWaitOp,
    SemaphoreAllocOp,
    ShapeAttr,
    ShardingAttr,
    TopologyAttr,
    ViewOp,
    YieldOp,
)
from tpu_cake.source import SourceLocation, attach_source, verify_with_sources


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
        mesh: Mapping[str, int] | None = None,
        interconnect_bandwidth_bytes_per_second: Mapping[str, int] | None = None,
        topology: TopologyAttr | None = None,
        dma_engine_count: int = 2,
        mxu_count: int = 1,
        vector_unit_count: int = 1,
        ici_link_count: int = 1,
        remote_dma_engine_count: int = 1,
    ) -> None:
        self._name = name
        self._target = target
        self._vmem_capacity_bytes = vmem_capacity_bytes
        self._smem_capacity_bytes = smem_capacity_bytes
        self._input_specs = tuple(inputs)
        self._mesh = dict(sorted((mesh or {}).items()))
        self._interconnect = dict(
            sorted((interconnect_bandwidth_bytes_per_second or {}).items())
        )
        if topology is not None and self._interconnect:
            raise ValueError(
                "provide either a structured topology or axis bandwidths, not both"
            )
        if topology is None and set(self._interconnect) != set(self._mesh):
            raise ValueError(
                "interconnect must declare one bandwidth for every kernel mesh axis"
            )
        self._dma_engine_count = dma_engine_count
        self._mxu_count = mxu_count
        self._vector_unit_count = vector_unit_count
        self._ici_link_count = ici_link_count
        self._remote_dma_engine_count = remote_dma_engine_count
        self._topology = topology
        self.block = Block(arg_types=[spec.to_type() for spec in self._input_specs])

    @property
    def inputs(self) -> tuple[SSAValue, ...]:
        return tuple(self.block.args)

    def _add(self, operation: Operation) -> Operation:
        self.block.add_op(operation)
        return operation

    def alloc(
        self, spec: BufferSpec, role: str, *, source: SourceLocation | None = None
    ) -> AllocOp:
        operation = attach_source(AllocOp(spec.to_type(), role), source)
        assert isinstance(operation, AllocOp)
        self._add(operation)
        return operation

    def semaphore(
        self, *, slots: int = 1, source: SourceLocation | None = None
    ) -> SemaphoreAllocOp:
        operation = attach_source(SemaphoreAllocOp(slots), source)
        assert isinstance(operation, SemaphoreAllocOp)
        self._add(operation)
        return operation

    def view(
        self,
        base: SSAValue | Operation,
        spec: BufferSpec,
        *,
        offsets: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        alias_group: str,
        source: SourceLocation | None = None,
    ) -> ViewOp:
        operation = attach_source(
            ViewOp(
                base,
                spec.to_type(),
                offsets=offsets,
                sizes=spec.physical_shape,
                strides=strides,
                alias_group=alias_group,
            ),
            source,
        )
        assert isinstance(operation, ViewOp)
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
        source_location: SourceLocation | None = None,
    ) -> DmaStartOp:
        start = self.dma_start(
            source,
            destination,
            semaphore,
            stage=start_stage,
            source_location=source_location,
        )
        self.dma_wait(start, stage=wait_stage, source=source_location)
        return start

    def dma_start(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        semaphore: SSAValue | Operation,
        *,
        stage: int,
        source_location: SourceLocation | None = None,
    ) -> DmaStartOp:
        operation = attach_source(
            DmaStartOp(source, destination, semaphore, stage), source_location
        )
        assert isinstance(operation, DmaStartOp)
        self._add(operation)
        return operation

    def dma_wait(
        self,
        token: SSAValue | Operation,
        *,
        stage: int,
        source: SourceLocation | None = None,
    ) -> DmaWaitOp:
        operation = attach_source(DmaWaitOp(token, stage), source)
        assert isinstance(operation, DmaWaitOp)
        self._add(operation)
        return operation

    def remote_dma_start(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        semaphore: SSAValue | Operation,
        *,
        stage: int,
        transfer_plan: str,
        source_location: SourceLocation | None = None,
    ) -> RemoteDmaStartOp:
        operation = attach_source(
            RemoteDmaStartOp(
                source,
                destination,
                semaphore,
                stage=stage,
                transfer_plan=transfer_plan,
            ),
            source_location,
        )
        assert isinstance(operation, RemoteDmaStartOp)
        self._add(operation)
        return operation

    def remote_dma_wait(
        self,
        token: SSAValue | Operation,
        *,
        stage: int,
        source: SourceLocation | None = None,
    ) -> RemoteDmaWaitOp:
        operation = attach_source(RemoteDmaWaitOp(token, stage=stage), source)
        assert isinstance(operation, RemoteDmaWaitOp)
        self._add(operation)
        return operation

    def matmul(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        accumulator: SSAValue | Operation,
        *,
        stage: int,
        tile_m: int | None = None,
        tile_k: int | None = None,
        tile_n: int | None = None,
        source: SourceLocation | None = None,
    ) -> MxuMatmulOp:
        operation = attach_source(
            MxuMatmulOp(
                lhs,
                rhs,
                accumulator,
                stage,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
            ),
            source,
        )
        assert isinstance(operation, MxuMatmulOp)
        self._add(operation)
        return operation

    def collective_reduce_scatter(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        *,
        stage: int,
        mesh_axis: str,
        group_size: int,
        scatter_dimension: int,
        reducer: str = "sum",
        source_location: SourceLocation | None = None,
    ) -> CollectiveOp:
        return self.collective(
            source,
            destination,
            stage=stage,
            kind=CollectiveKind.REDUCE_SCATTER,
            mesh_axis=mesh_axis,
            group_size=group_size,
            split_dimension=scatter_dimension,
            reducer=reducer,
            source_location=source_location,
        )

    def collective(
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
        source_location: SourceLocation | None = None,
    ) -> CollectiveOp:
        operation = attach_source(
            CollectiveOp(
                source,
                destination,
                stage=stage,
                kind=kind,
                mesh_axis=mesh_axis,
                group_size=group_size,
                split_dimension=split_dimension,
                concat_dimension=concat_dimension,
                reducer=reducer,
            ),
            source_location,
        )
        assert isinstance(operation, CollectiveOp)
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

    def fused_ragged_paged_attention(
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
        source_location: SourceLocation | None = None,
    ) -> FusedRaggedPagedAttentionOp:
        operation = attach_source(
            FusedRaggedPagedAttentionOp(
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
                stage=stage,
                causal=causal,
                softmax_scale=softmax_scale,
                softmax_dtype=softmax_dtype,
                sliding_window=sliding_window,
                query_block_size=query_block_size,
                kv_block_size=kv_block_size,
                query_cluster_size=query_cluster_size,
                kv_cluster_size=kv_cluster_size,
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            source_location,
        )
        assert isinstance(operation, FusedRaggedPagedAttentionOp)
        self._add(operation)
        return operation

    def module(self) -> ModuleOp:
        self.block.add_op(YieldOp())
        kernel = KernelOp(
            self._name,
            self._target,
            self._vmem_capacity_bytes,
            self._smem_capacity_bytes,
            ArrayAttr(StringAttr(axis) for axis in self._mesh),
            ArrayAttr(IntAttr(size) for size in self._mesh.values()),
            Region(self.block),
            interconnect_bandwidth_bytes_per_second=self._interconnect,
            topology=self._topology,
            dma_engine_count=self._dma_engine_count,
            mxu_count=self._mxu_count,
            vector_unit_count=self._vector_unit_count,
            ici_link_count=self._ici_link_count,
            remote_dma_engine_count=self._remote_dma_engine_count,
        )
        module = ModuleOp([kernel])
        verify_with_sources(module)
        return module


def canonical_module_text(module: ModuleOp) -> str:
    return canonical_text(module)


def schedule_sha256(module: ModuleOp) -> str:
    return canonical_sha256(module)
