from __future__ import annotations

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
    WorkloadStage,
)
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.source import SourceLocation


def distributed_matmul_schedule(
    *,
    mesh_size: int = 4,
    m: int = 16,
    k: int = 32,
    n: int = 16,
) -> ModuleOp:
    lhs = tensor(bf16, (("M", m), ("K", k)), sharding={"K": ("t",)})
    rhs = tensor(bf16, (("K", k), ("N", n)), sharding={"K": ("t",)})
    builder = DistributedProgramBuilder("distributed_matmul", {"t": mesh_size}, (lhs, rhs))
    partial = builder.einsum_local(
        builder.inputs[0],
        builder.inputs[1],
        tensor(f32, (("M", m), ("N", n)), pending_reductions={"t": "sum"}),
        contracting_dimension="K",
        source=SourceLocation("tpu_cake/workloads/distributed_matmul.py", 18, 15),
    )
    result = builder.reduce_scatter(
        partial,
        tensor(f32, (("M", m), ("N", n)), sharding={"N": ("t",)}),
        axes=("t",),
        scatter_dimensions=("N",),
        source=SourceLocation("tpu_cake/workloads/distributed_matmul.py", 26, 14),
    )
    return builder.module(result)


def distributed_matmul_experiment(
    *,
    schedule_sha256: str,
    mesh_size: int,
    m: int,
    k: int,
    n: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> KernelExperiment:
    return KernelExperiment(
        workload=WorkloadContract(
            name=f"distributed-matmul-{m}x{k}x{n}-mesh{mesh_size}",
            stage=WorkloadStage.CONTROL,
            inputs=(
                TensorContract(
                    name="lhs",
                    shape=(m, k),
                    logical_shape=("M", "K"),
                    dtype="bf16",
                    sharding=("", "t"),
                ),
                TensorContract(
                    name="rhs",
                    shape=(k, n),
                    logical_shape=("K", "N"),
                    dtype="bf16",
                    sharding=("t", ""),
                ),
            ),
            outputs=(
                TensorContract(
                    name="output",
                    shape=(m, n),
                    logical_shape=("M", "N"),
                    dtype="f32",
                    sharding=("", "t"),
                ),
            ),
            numerical=NumericalContract(
                reference="numpy.matmul on the exact BF16 inputs",
                absolute_tolerance=1e-3,
                relative_tolerance=1e-3,
            ),
        ),
        target=TargetHardware(
            accelerator="TPU7x",
            topology=f"mesh(t={mesh_size})",
            chip_count=(mesh_size + 1) // 2,
            vmem_budget_bytes_per_core=128 << 20,
            smem_budget_bytes_per_core=32 << 20,
            runtime_target="Pallas Mosaic TPU",
        ),
        benchmark=BenchmarkProtocol(
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            synchronization="block until all output shards are ready",
            statistic="median end-to-end distributed device duration",
        ),
        search=SearchPolicy(objective_metric="median_device_duration_ns"),
        profile=ProfileExpectation(
            name="distributed-matmul",
            stage=WorkloadStage.CONTROL,
            minimum_tpu_device_planes=mesh_size,
            require_tensor_core_activity=False,
            required_timed_hlo_markers=(
                "distributed_matmul_physical",
                "reduce-scatter",
                "pallas_call",
                schedule_sha256,
            ),
        ),
        schedule_sha256=schedule_sha256,
    )
