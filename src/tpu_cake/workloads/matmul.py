from __future__ import annotations

import numpy as np
from xdsl.dialects.builtin import ModuleOp, bf16, f32

from tpu_cake.contracts import (
    BenchmarkProtocol,
    KernelExperiment,
    NumericalContract,
    ProfileExpectation,
    SearchPolicy,
    TargetHardware,
    TensorContract,
    WorkloadContract,
)
from tpu_cake.dialects.tpu_schedule import MemorySpace, Ownership
from tpu_cake.frontend import KernelBuilder, buffer, schedule_sha256


def matmul_reference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs.astype(np.float32) @ rhs.astype(np.float32)


def matmul_inputs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    return (
        generator.normal(size=(16, 32)).astype(np.float32),
        generator.normal(size=(32, 16)).astype(np.float32),
    )


def matmul_schedule() -> ModuleOp:
    external = {
        "memory": MemorySpace.HBM,
        "ownership": Ownership.EXTERNAL,
        "lifetime": (0, 4),
    }
    inputs = (
        buffer((16, 32), "M K", bf16, **external),
        buffer((32, 16), "K N", bf16, **external),
        buffer((16, 16), "M N", f32, **external),
    )
    builder = KernelBuilder(
        "matmul_16x32x16",
        "tpu7x",
        inputs,
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 16,
    )
    lhs = builder.alloc(
        buffer((16, 32), "M K", bf16, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        "lhs_tile",
    )
    rhs = builder.alloc(
        buffer((32, 16), "K N", bf16, memory=MemorySpace.VMEM, lifetime=(0, 2)),
        "rhs_tile",
    )
    accumulator = builder.alloc(
        buffer((16, 16), "M N", f32, memory=MemorySpace.VMEM, lifetime=(2, 4)),
        "accumulator",
    )
    lhs_semaphore = builder.semaphore()
    rhs_semaphore = builder.semaphore()
    output_semaphore = builder.semaphore()
    lhs_dma = builder.dma_start(builder.inputs[0], lhs, lhs_semaphore, stage=0)
    rhs_dma = builder.dma_start(builder.inputs[1], rhs, rhs_semaphore, stage=0)
    builder.dma_wait(lhs_dma, stage=1)
    builder.dma_wait(rhs_dma, stage=1)
    builder.matmul(lhs, rhs, accumulator, stage=2)
    output_dma = builder.dma_start(accumulator, builder.inputs[2], output_semaphore, stage=3)
    builder.dma_wait(output_dma, stage=4)
    return builder.module()


def matmul_contract() -> WorkloadContract:
    tensor = lambda name, shape, logical, dtype: TensorContract(
        name=name, shape=shape, logical_shape=logical, dtype=dtype, sharding=("",) * len(shape)
    )
    return WorkloadContract(
        name="dense-matmul-16x32x16",
        stage="control",
        inputs=(
            tensor("lhs", (16, 32), ("M", "K"), "bf16"),
            tensor("rhs", (32, 16), ("K", "N"), "bf16"),
        ),
        outputs=(tensor("output", (16, 16), ("M", "N"), "f32"),),
        numerical=NumericalContract(
            reference="tpu_cake.workloads.matmul.matmul_reference",
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        ),
    )


def matmul_experiment() -> KernelExperiment:
    schedule = matmul_schedule()
    return KernelExperiment(
        workload=matmul_contract(),
        target=TargetHardware(
            accelerator="TPU7x",
            topology="4x1",
            chip_count=4,
            vmem_budget_bytes_per_core=1 << 20,
            smem_budget_bytes_per_core=1 << 16,
            runtime_target="Pallas Mosaic TPU",
        ),
        benchmark=BenchmarkProtocol(
            warmup_iterations=5,
            measured_iterations=50,
            synchronization="block until device result is ready",
            statistic="median device duration",
        ),
        search=SearchPolicy(objective_metric="median_device_duration_ns"),
        profile=ProfileExpectation(
            name="matmul-control",
            stage="control",
            required_timed_hlo_markers=("pallas_call",),
        ),
        schedule_sha256=schedule_sha256(schedule),
    )
