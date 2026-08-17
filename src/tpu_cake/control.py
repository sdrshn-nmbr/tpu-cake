from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_cake.identity import array_sha256, workload_rng
from tpu_cake.lowering import lower_distributed_matmul
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


def run_interpret_control() -> dict[str, object]:
    distributed = distributed_matmul_schedule()
    physical = lower_distributed_matmul(distributed)
    plan = lower_physical_matmul_to_pallas(physical)
    executable, mesh = plan.build(interpret=True)
    generator = workload_rng("matmul-control", "distributed", "attempt-0", "inputs")
    lhs_host = generator.normal(size=plan.global_lhs_shape).astype(np.float32)
    rhs_host = generator.normal(size=plan.global_rhs_shape).astype(np.float32)
    lhs = jax.device_put(
        jnp.asarray(lhs_host, dtype=jnp.bfloat16),
        NamedSharding(mesh, PartitionSpec(None, plan.mesh_axis)),
    )
    rhs = jax.device_put(
        jnp.asarray(rhs_host, dtype=jnp.bfloat16),
        NamedSharding(mesh, PartitionSpec(plan.mesh_axis, None)),
    )
    actual = executable(lhs, rhs)
    actual.block_until_ready()
    expected = jnp.asarray(lhs, jnp.float32) @ jnp.asarray(rhs, jnp.float32)
    maximum_absolute_error = float(jnp.max(jnp.abs(actual - expected)))
    return {
        "passed": bool(jnp.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "device_count": len(jax.devices()),
        "schedule_sha256": plan.schedule_sha256,
        "pallas_source_sha256": plan.source_sha256(),
        "lhs_sha256": array_sha256(lhs_host),
        "rhs_sha256": array_sha256(rhs_host),
        "output_sha256": array_sha256(np.asarray(actual)),
        "maximum_absolute_error": maximum_absolute_error,
        "output_shape": list(actual.shape),
        "output_dtype": str(actual.dtype),
    }


def main() -> None:
    print(json.dumps(run_interpret_control(), sort_keys=True))


if __name__ == "__main__":
    main()
