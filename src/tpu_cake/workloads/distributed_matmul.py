from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp, bf16, f32

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
