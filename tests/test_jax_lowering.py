import hashlib
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xdsl.dialects.builtin import bf16

from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.frontend import schedule_sha256
from tpu_cake.jax_lowering import (
    JAX_DISTRIBUTED_EXECUTION_SCHEMA,
    JAX_LOGICAL_EXECUTION_SCHEMA,
    UnsupportedJaxLoweringError,
    load_jax_distributed_plan,
    lower_distributed_program_to_jax,
    lower_distributed_program_to_jax_mesh,
)
from tpu_cake.source import SourceLocation
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_inputs,
    seqax_forward_reference,
)

SMALL_SEQAX = {
    "batch": 2,
    "sequence": 4,
    "model": 8,
    "vocabulary": 16,
    "feed_forward": 16,
    "query_groups": 2,
    "key_value_heads": 4,
    "head": 4,
    "layers": 2,
    "data_mesh": 2,
    "tensor_mesh": 4,
    "rope_max_timescale": 256,
}


def test_complete_seqax_forward_lowers_to_replayable_logical_jax() -> None:
    module = seqax_forward_schedule(**SMALL_SEQAX)
    plan = lower_distributed_program_to_jax(module)

    assert plan.schema == JAX_LOGICAL_EXECUTION_SCHEMA
    assert plan.schedule_sha256 == schedule_sha256(module)
    assert plan.execution_scope == "one-device-global-logical-tensors"
    assert plan.collective_semantics == "global-value-identity"
    assert plan.uses_physical_collectives is False
    assert plan.uses_pallas is False
    assert plan.operation_counts["dtensor.layer_scan"] == 1
    assert plan.operation_counts["dtensor.all_gather"] == 14
    assert plan.operation_counts["dtensor.reduce_scatter"] == 3

    inputs = seqax_forward_inputs(seed=9173, **SMALL_SEQAX)
    expected = seqax_forward_reference(inputs, **SMALL_SEQAX)
    executable = plan.build(device=jax.devices("cpu")[0])
    (actual,) = executable(*(jnp.asarray(value) for value in inputs))

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=5e-2, atol=6e-3)


def test_rendered_logical_jax_source_reconstructs_the_same_plan() -> None:
    plan = lower_distributed_program_to_jax(seqax_forward_schedule(**SMALL_SEQAX))
    source = plan.render_executable_source()
    namespace: dict[str, object] = {}

    exec(compile(source, "<seqax-logical-jax>", "exec"), namespace)  # noqa: S102
    replayed = namespace["PLAN"]

    assert replayed.schedule_sha256 == plan.schedule_sha256
    assert replayed.source_sha256() == hashlib.sha256(source.encode()).hexdigest()
    assert replayed.manifest() == plan.manifest()


def test_seqax_forward_mesh_plan_binds_exact_partition_specs_and_source() -> None:
    plan = lower_distributed_program_to_jax_mesh(seqax_forward_schedule(**SMALL_SEQAX))

    assert plan.schema == JAX_DISTRIBUTED_EXECUTION_SCHEMA
    assert plan.mesh == {"d": 2, "t": 4}
    assert plan.device_count == 8
    assert plan.execution_scope == "multi-device-local-shards"
    assert plan.collective_semantics == "jax-lax-physical-collectives"
    assert plan.uses_physical_collectives is True
    assert plan.uses_pallas is False
    assert repr(plan.input_partition_specs[0]) == "P('d', None)"
    assert repr(plan.input_partition_specs[2]) == "P('t', 'd')"
    assert repr(plan.output_partition_specs[0]) == "P('d', None, 't')"
    assert plan.manifest()["input_contracts"][2]["local_shape"] == [4, 4]

    namespace: dict[str, object] = {}
    source = plan.render_executable_source()
    exec(compile(source, "<seqax-distributed-jax>", "exec"), namespace)  # noqa: S102
    assert namespace["PLAN"].manifest() == plan.manifest()

    with pytest.raises(ValueError, match="needs exactly 8 devices"):
        plan.build(devices=jax.devices("cpu"))


def test_complete_seqax_forward_executes_with_real_collectives_on_eight_devices() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs, seqax_forward_reference

parameters = {
    "batch": 2,
    "sequence": 4,
    "model": 8,
    "vocabulary": 16,
    "feed_forward": 16,
    "query_groups": 2,
    "key_value_heads": 4,
    "head": 4,
    "layers": 2,
    "data_mesh": 2,
    "tensor_mesh": 4,
    "rope_max_timescale": 256,
}
devices = jax.devices("cpu")
assert len(devices) == 8, len(devices)
plan = lower_distributed_program_to_jax_mesh(seqax_forward_schedule(**parameters))
executable, mesh = plan.build(devices=devices)
inputs = seqax_forward_inputs(seed=9173, **parameters)
arrays = tuple(jnp.asarray(value) for value in inputs)
stablehlo = str(executable.lower(*arrays).compiler_ir("stablehlo"))
assert "stablehlo.all_gather" in stablehlo
assert "stablehlo.reduce_scatter" in stablehlo
(actual,) = executable(*arrays)
actual.block_until_ready()
expected = seqax_forward_reference(inputs, **parameters)
np.testing.assert_allclose(np.asarray(actual), expected, rtol=5e-2, atol=6e-3)
assert mesh.shape == {"d": 2, "t": 4}
print(float(np.max(np.abs(np.asarray(actual) - expected))))
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_replay_rejects_canonical_program_hash_mismatch() -> None:
    plan = lower_distributed_program_to_jax(seqax_forward_schedule(**SMALL_SEQAX))

    with pytest.raises(UnsupportedJaxLoweringError, match="schedule hash mismatch"):
        load_jax_distributed_plan(
            plan.canonical_xdsl,
            expected_schedule_sha256="0" * 64,
        )


def test_jax_lowering_fails_closed_with_source_linked_unsupported_operation() -> None:
    value_type = tensor(bf16, (("B", 2), ("M", 8)))
    builder = DistributedProgramBuilder("unsupported_relu", {}, (value_type,))
    result = builder.elementwise(
        builder.inputs[0],
        result=value_type,
        function="relu",
        source=SourceLocation("unsupported_seqax.py", 17, 5),
    )

    with pytest.raises(
        UnsupportedJaxLoweringError,
        match=r"elementwise function 'relu'.*unsupported_seqax.py.*17:5",
    ):
        lower_distributed_program_to_jax(builder.module(result))
