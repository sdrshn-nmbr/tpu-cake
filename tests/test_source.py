import pytest
from xdsl.dialects.builtin import bf16
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import MemorySpace, Ownership
from tpu_cake.frontend import KernelBuilder, buffer
from tpu_cake.source import SourceLocation


def test_cross_operation_failure_reports_all_relevant_source_sites() -> None:
    external = buffer(
        (8, 8),
        "X Y",
        bf16,
        memory=MemorySpace.HBM,
        ownership=Ownership.EXTERNAL,
        lifetime=(0, 1),
    )
    builder = KernelBuilder(
        "write_hazard",
        "tpu7x",
        (external, external),
        vmem_capacity_bytes=1024,
        smem_capacity_bytes=1024,
    )
    destination = builder.alloc(
        buffer((8, 8), "X Y", bf16, memory=MemorySpace.VMEM, lifetime=(0, 1)),
        "destination",
    )
    first = builder.dma_start(
        builder.inputs[0],
        destination,
        builder.semaphore(),
        stage=0,
        source_location=SourceLocation("factory.py", 10, 3),
    )
    second = builder.dma_start(
        builder.inputs[1],
        destination,
        builder.semaphore(),
        stage=0,
        source_location=SourceLocation("factory.py", 20, 3),
    )
    builder.dma_wait(first, stage=1)
    builder.dma_wait(second, stage=1)

    with pytest.raises(VerifyException) as failure:
        builder.module()
    message = str(failure.value)
    assert "concurrent DMA writes" in message
    assert 'loc("factory.py":10:3)' in message
    assert 'loc("factory.py":20:3)' in message
