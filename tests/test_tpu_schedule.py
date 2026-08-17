import pytest
from xdsl.dialects.builtin import MemRefType, bf16, f32
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    DmaStartOp,
    DmaWaitOp,
    KernelOp,
    MemorySpace,
    MemorySpaceAttr,
    MxuMatmulOp,
    SemaphoreAllocOp,
    YieldOp,
)


def _memref(shape: tuple[int, ...], element_type, space: MemorySpace) -> MemRefType:
    return MemRefType(element_type, shape, memory_space=MemorySpaceAttr(space))


def _kernel(*, wait_for_rhs: bool = True, vmem_capacity: int = 1 << 20) -> KernelOp:
    hbm_lhs = _memref((16, 32), bf16, MemorySpace.HBM)
    hbm_rhs = _memref((32, 16), bf16, MemorySpace.HBM)
    hbm_out = _memref((16, 16), f32, MemorySpace.HBM)
    block = Block(arg_types=[hbm_lhs, hbm_rhs, hbm_out])
    lhs = AllocOp(_memref((16, 32), bf16, MemorySpace.VMEM), "lhs_tile")
    rhs = AllocOp(_memref((32, 16), bf16, MemorySpace.VMEM), "rhs_tile")
    acc = AllocOp(_memref((16, 16), f32, MemorySpace.VMEM), "accumulator")
    lhs_sem = SemaphoreAllocOp()
    rhs_sem = SemaphoreAllocOp()
    lhs_dma = DmaStartOp(block.args[0], lhs, lhs_sem, 0)
    rhs_dma = DmaStartOp(block.args[1], rhs, rhs_sem, 0)
    operations = [lhs, rhs, acc, lhs_sem, rhs_sem, lhs_dma, rhs_dma, DmaWaitOp(lhs_dma, 1)]
    if wait_for_rhs:
        operations.append(DmaWaitOp(rhs_dma, 1))
    operations.extend([MxuMatmulOp(lhs, rhs, acc, 2), YieldOp()])
    block.add_ops(operations)
    return KernelOp(
        "matmul",
        "tpu7x",
        vmem_capacity,
        1 << 16,
        Region(block),
    )


def test_valid_schedule_verifies() -> None:
    _kernel().verify()


def test_unwaited_dma_is_rejected() -> None:
    with pytest.raises(VerifyException, match="exactly one DMA wait"):
        _kernel(wait_for_rhs=False).verify()


def test_vmem_capacity_is_checked() -> None:
    with pytest.raises(VerifyException, match="VMEM capacity exceeded"):
        _kernel(vmem_capacity=1).verify()
