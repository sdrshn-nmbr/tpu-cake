import hashlib
import os
import subprocess
import sys
from dataclasses import replace

import jax.numpy as jnp
import pytest

from tpu_cake.dialects.tpu_schedule import MxuEinsumOp
from tpu_cake.frontend import canonical_module_text
from tpu_cake.seqax_pallas_lowering import (
    SEQAX_PALLAS_EXECUTION_SCHEMA,
    UnsupportedSeqaxPallasLoweringError,
    lower_seqax_physical_to_pallas,
)
from tpu_cake.seqax_physical_execution import execute_seqax_physical_program_jax
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule

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


def _plan():
    distributed = seqax_forward_schedule(**SMALL_SEQAX)
    physical = lower_seqax_forward_to_physical(distributed).module
    return lower_seqax_physical_to_pallas(distributed, physical)


def test_complete_seqax_physical_schedule_lowers_to_replayable_pallas_plan() -> None:
    plan = _plan()

    assert plan.schema == SEQAX_PALLAS_EXECUTION_SCHEMA
    assert plan.mesh == (("d", 2), ("t", 4))
    assert plan.device_count == 8
    assert plan.uses_pallas is True
    assert plan.execution_scope == "multi-device-local-shards-with-pallas-einsums"
    assert plan.pallas_region_count == 17

    source = plan.render_executable_source()
    namespace: dict[str, object] = {}
    exec(compile(source, "<seqax-physical-pallas>", "exec"), namespace)  # noqa: S102
    replayed = namespace["PLAN"]
    assert replayed.manifest() == plan.manifest()
    assert replayed.source_sha256() == hashlib.sha256(source.encode()).hexdigest()


def test_pallas_plan_rejects_a_noncanonical_physical_schedule() -> None:
    plan = _plan()
    mutated = plan.canonical_physical_xdsl.replace(
        'role = "alloc0"',
        'role = "mutated_input"',
        1,
    )
    assert mutated != plan.canonical_physical_xdsl

    with pytest.raises(
        UnsupportedSeqaxPallasLoweringError,
        match="not the canonical lowering",
    ):
        replace(plan, canonical_physical_xdsl=mutated)._validated_modules()


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda plan: replace(plan, mesh=(("d", 4), ("t", 2))),
            "mesh does not match",
        ),
        (
            lambda plan: replace(
                plan,
                input_contracts=(
                    replace(plan.input_contracts[0], declared_sharding=((), ())),
                    *plan.input_contracts[1:],
                ),
            ),
            "inputs do not match",
        ),
        (
            lambda plan: replace(
                plan,
                output_contracts=(replace(plan.output_contracts[0], dtype="bfloat16"),),
            ),
            "outputs do not match",
        ),
    ),
)
def test_pallas_plan_rejects_forged_execution_contracts(mutate, message: str) -> None:
    with pytest.raises(UnsupportedSeqaxPallasLoweringError, match=message):
        mutate(_plan())._validated_modules()


def test_physical_schedule_contains_one_pallas_region_per_distributed_einsum() -> None:
    distributed = seqax_forward_schedule(**SMALL_SEQAX)
    physical = lower_seqax_forward_to_physical(distributed).module

    einsums = tuple(
        operation for operation in physical.walk() if isinstance(operation, MxuEinsumOp)
    )
    assert len(einsums) == 17
    assert all(operation.tile_k.data > 0 for operation in einsums)
    assert canonical_module_text(physical) == _plan().canonical_physical_xdsl


def _local_inputs(plan):
    dtypes = {
        "bfloat16": jnp.bfloat16,
        "bool": jnp.bool_,
        "float32": jnp.float32,
        "uint32": jnp.uint32,
    }
    mesh = dict(plan.mesh)
    return tuple(
        jnp.zeros(contract.local_shape(mesh), dtype=dtypes[contract.dtype])
        for contract in plan.input_contracts
    )


def _unexpected_einsum(*_args):
    raise AssertionError("invalid physical input reached execution")


def test_physical_executor_rejects_wrong_input_count() -> None:
    plan = _plan()
    physical = plan._validated_modules()[1]

    with pytest.raises(ValueError, match="expects .* inputs"):
        execute_seqax_physical_program_jax(
            physical,
            _local_inputs(plan)[:-1],
            einsum=_unexpected_einsum,
        )


def test_physical_executor_rejects_wrong_input_shape() -> None:
    plan = _plan()
    physical = plan._validated_modules()[1]
    inputs = _local_inputs(plan)
    corrupted = (jnp.zeros((9, 9), dtype=inputs[0].dtype), *inputs[1:])

    with pytest.raises(ValueError, match="input shape"):
        execute_seqax_physical_program_jax(
            physical,
            corrupted,
            einsum=_unexpected_einsum,
        )


def test_physical_executor_rejects_wrong_input_dtype() -> None:
    plan = _plan()
    physical = plan._validated_modules()[1]
    inputs = _local_inputs(plan)
    corrupted = (inputs[0].astype(jnp.int32), *inputs[1:])

    with pytest.raises(ValueError, match="input dtype"):
        execute_seqax_physical_program_jax(
            physical,
            corrupted,
            einsum=_unexpected_einsum,
        )


def test_complete_seqax_pallas_plan_executes_on_eight_devices_in_interpret_mode() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
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
distributed = seqax_forward_schedule(**parameters)
physical = lower_seqax_forward_to_physical(distributed).module
plan = lower_seqax_physical_to_pallas(distributed, physical)
source = plan.render_executable_source()
namespace = {}
exec(compile(source, "<seqax-physical-pallas>", "exec"), namespace)
assert namespace["PLAN"].manifest() == plan.manifest()
executable, mesh = namespace["build"](interpret=True, devices=devices)
inputs = seqax_forward_inputs(seed=9173, **parameters)
arrays = tuple(jnp.asarray(value) for value in inputs)
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
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
