from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_cake.dialects.tpu_schedule import CollectiveImplementation
from tpu_cake.identity import array_sha256, workload_rng
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
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


def run_native_collective_interpret_control() -> dict[str, object]:
    distributed = distributed_matmul_schedule(mesh_size=8, m=128, k=1024, n=1024)
    physical = lower_distributed_matmul(
        distributed,
        tile=MatmulTile(128, 128),
        collective_implementation=(CollectiveImplementation.PALLAS_BIDIRECTIONAL_RING),
    )
    plan = lower_physical_matmul_to_pallas(physical)
    executable, mesh = plan.build(interpret=True)
    generator = workload_rng(
        "matmul-control", "pallas-native-reduce-scatter", "attempt-0", "inputs"
    )
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
        "collective_implementation": plan.collective_implementation.value,
        "lhs_sha256": array_sha256(lhs_host),
        "rhs_sha256": array_sha256(rhs_host),
        "output_sha256": array_sha256(np.asarray(actual)),
        "maximum_absolute_error": maximum_absolute_error,
        "output_shape": list(actual.shape),
        "output_dtype": str(actual.dtype),
    }


def main() -> None:
    result = (
        run_native_collective_interpret_control()
        if os.environ.get("TPU_CAKE_CONTROL_NATIVE_COLLECTIVE") == "1"
        else run_interpret_control()
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
