from __future__ import annotations

import math
from dataclasses import dataclass

from tpu_cake.dialects.tpu_schedule import BufferType, MxuEinsumOp, MxuMatmulOp


class UnsupportedPhysicalGeometryError(ValueError):
    pass


def _names(buffer: BufferType) -> tuple[str, ...]:
    return tuple(value.data for value in buffer.shape.dimensions)


@dataclass(frozen=True)
class MxuGeometry:
    batch_names: tuple[str, ...]
    lhs_free_names: tuple[str, ...]
    rhs_free_names: tuple[str, ...]
    contraction_names: tuple[str, ...]
    batch_shape: tuple[int, ...]
    lhs_free_shape: tuple[int, ...]
    rhs_free_shape: tuple[int, ...]
    contraction_shape: tuple[int, ...]
    lhs_permutation: tuple[int, ...]
    rhs_permutation: tuple[int, ...]
    result_permutation: tuple[int, ...]
    batch: int
    m: int
    k: int
    n: int
    tile_m: int
    tile_k: int
    tile_n: int

    @property
    def grid(self) -> tuple[int, int, int, int]:
        return (
            self.batch,
            self.m // self.tile_m,
            self.n // self.tile_n,
            self.k // self.tile_k,
        )

    @property
    def tile_program_count(self) -> int:
        return math.prod(self.grid)

    @property
    def flops(self) -> int:
        return 2 * self.batch * self.m * self.k * self.n


def named_einsum_geometry(operation: MxuEinsumOp) -> MxuGeometry:
    lhs_type = operation.lhs.type
    rhs_type = operation.rhs.type
    result_type = operation.accumulator.type
    assert isinstance(lhs_type, BufferType)
    assert isinstance(rhs_type, BufferType)
    assert isinstance(result_type, BufferType)
    lhs_names = _names(lhs_type)
    rhs_names = _names(rhs_type)
    result_names = _names(result_type)
    contractions = tuple(value.data for value in operation.contracting_dimensions)
    batch_names = tuple(
        name for name in lhs_names if name in rhs_names and name not in contractions
    )
    lhs_free_names = tuple(
        name for name in lhs_names if name not in batch_names and name not in contractions
    )
    rhs_free_names = tuple(
        name for name in rhs_names if name not in batch_names and name not in contractions
    )
    dot_names = (*batch_names, *lhs_free_names, *rhs_free_names)
    if set(dot_names) != set(result_names) or len(dot_names) != len(result_names):
        raise UnsupportedPhysicalGeometryError(
            "physical MXU einsum cannot reconstruct result dimensions"
        )
    lhs_contract = tuple(lhs_names.index(name) for name in contractions)
    rhs_contract = tuple(rhs_names.index(name) for name in contractions)
    lhs_batch = tuple(lhs_names.index(name) for name in batch_names)
    rhs_batch = tuple(rhs_names.index(name) for name in batch_names)
    lhs_free = tuple(lhs_names.index(name) for name in lhs_free_names)
    rhs_free = tuple(rhs_names.index(name) for name in rhs_free_names)
    expected_lhs = lhs_type.storage.get_shape()
    expected_rhs = rhs_type.storage.get_shape()
    batch_shape = tuple(expected_lhs[index] for index in lhs_batch)
    lhs_free_shape = tuple(expected_lhs[index] for index in lhs_free)
    rhs_free_shape = tuple(expected_rhs[index] for index in rhs_free)
    contraction_shape = tuple(expected_lhs[index] for index in lhs_contract)
    if batch_shape != tuple(expected_rhs[index] for index in rhs_batch):
        raise UnsupportedPhysicalGeometryError("physical MXU batch dimensions do not match")
    if contraction_shape != tuple(expected_rhs[index] for index in rhs_contract):
        raise UnsupportedPhysicalGeometryError("physical MXU contraction dimensions do not match")
    batch = math.prod(batch_shape)
    m = math.prod(lhs_free_shape)
    k = math.prod(contraction_shape)
    n = math.prod(rhs_free_shape)
    tile_m = operation.tile_m.data
    tile_k = operation.tile_k.data
    tile_n = operation.tile_n.data
    if m % tile_m or k % tile_k or n % tile_n:
        raise UnsupportedPhysicalGeometryError(
            "physical MXU tiles must divide the flattened local contraction"
        )
    return MxuGeometry(
        batch_names=batch_names,
        lhs_free_names=lhs_free_names,
        rhs_free_names=rhs_free_names,
        contraction_names=contractions,
        batch_shape=batch_shape,
        lhs_free_shape=lhs_free_shape,
        rhs_free_shape=rhs_free_shape,
        contraction_shape=contraction_shape,
        lhs_permutation=(*lhs_batch, *lhs_free, *lhs_contract),
        rhs_permutation=(*rhs_batch, *rhs_contract, *rhs_free),
        result_permutation=tuple(dot_names.index(name) for name in result_names),
        batch=batch,
        m=m,
        k=k,
        n=n,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )


def matmul_geometry(operation: MxuMatmulOp) -> MxuGeometry:
    lhs_type = operation.lhs.type
    rhs_type = operation.rhs.type
    assert isinstance(lhs_type, BufferType)
    assert isinstance(rhs_type, BufferType)
    m, k = lhs_type.storage.get_shape()
    rhs_k, n = rhs_type.storage.get_shape()
    if k != rhs_k:
        raise UnsupportedPhysicalGeometryError("physical MXU matmul K dimensions do not match")
    return MxuGeometry(
        batch_names=(),
        lhs_free_names=("M",),
        rhs_free_names=("N",),
        contraction_names=("K",),
        batch_shape=(),
        lhs_free_shape=(m,),
        rhs_free_shape=(n,),
        contraction_shape=(k,),
        lhs_permutation=(0, 1),
        rhs_permutation=(0, 1),
        result_permutation=(0, 1),
        batch=1,
        m=m,
        k=k,
        n=n,
        tile_m=operation.tile_m.data,
        tile_k=operation.tile_k.data,
        tile_n=operation.tile_n.data,
    )


def mxu_geometry(operation: MxuEinsumOp | MxuMatmulOp) -> MxuGeometry:
    if isinstance(operation, MxuEinsumOp):
        return named_einsum_geometry(operation)
    return matmul_geometry(operation)
