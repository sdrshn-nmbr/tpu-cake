import hashlib
import os
import subprocess
import sys
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from xdsl.dialects.builtin import bf16, f32

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    CollectiveOp,
    MemorySpace,
    MxuEinsumOp,
    VectorComputeOp,
    VectorMaterialization,
)
from tpu_cake.frontend import buffer, canonical_module_text
from tpu_cake.seqax_pallas_lowering import (
    SEQAX_PALLAS_EXECUTION_SCHEMA,
    UnsupportedSeqaxPallasLoweringError,
    _einsum_tiles,
    _pallas_einsum,
    lower_seqax_physical_to_pallas,
)
from tpu_cake.seqax_physical_execution import execute_seqax_physical_program_jax
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import (
    REPLICATED_ATTENTION_WEIGHT_DATA,
    REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA,
    REPLICATED_EMBEDDING_WEIGHT_DATA,
    REPLICATED_FEED_FORWARD_WEIGHT_DATA,
    REPLICATED_WEIGHT_DATA,
    SeqaxNormScalePlacement,
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
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
TILED_SEQAX = {**SMALL_SEQAX, "model": 256}


def _partial_tpu_tiles(
    tiles: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            m,
            128 if k > 128 and k % 128 == 0 else k,
            128 if n > 128 and n % 128 == 0 else n,
        )
        for m, k, n in tiles
    )


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
    assert "tpu_cake/physical_geometry.py" in dict(plan.implementation_manifest)

    source = plan.render_executable_source()
    namespace: dict[str, object] = {}
    exec(compile(source, "<seqax-physical-pallas>", "exec"), namespace)  # noqa: S102
    replayed = namespace["PLAN"]
    assert replayed.manifest() == plan.manifest()
    assert replayed.source_sha256() == hashlib.sha256(source.encode()).hexdigest()


def test_pallas_plan_rejects_missing_geometry_source_authority() -> None:
    plan = _plan()
    without_geometry = tuple(
        value
        for value in plan.implementation_manifest
        if value[0] != "tpu_cake/physical_geometry.py"
    )

    with pytest.raises(UnsupportedSeqaxPallasLoweringError, match="source manifest"):
        replace(plan, implementation_manifest=without_geometry)._validated_modules()


def test_typed_bf16_materialization_lowers_into_the_physical_schedule() -> None:
    legacy = seqax_forward_schedule(**SMALL_SEQAX)
    strict = seqax_forward_schedule(
        **SMALL_SEQAX,
        numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_V1,
    )
    legacy_physical = lower_seqax_forward_to_physical(legacy).module
    strict_physical = lower_seqax_forward_to_physical(strict).module
    legacy_silu = tuple(
        operation
        for operation in legacy_physical.walk()
        if isinstance(operation, VectorComputeOp) and operation.function.data == "silu"
    )
    strict_silu = tuple(
        operation
        for operation in strict_physical.walk()
        if isinstance(operation, VectorComputeOp) and operation.function.data == "silu"
    )

    assert all(operation.materialization is None for operation in legacy_silu)
    assert len(strict_silu) == SMALL_SEQAX["layers"]
    assert all(operation.materialization is not None for operation in strict_silu)
    assert all(
        operation.materialization.data is VectorMaterialization.STRICT_TYPED
        for operation in strict_silu
        if operation.materialization is not None
    )
    assert canonical_module_text(legacy_physical) != canonical_module_text(strict_physical)


def test_replicated_norm_scales_remove_exact_physical_gather_chains() -> None:
    parameters = {**TILED_SEQAX, "sequence": 1, "layers": 1}
    sharded = seqax_forward_schedule(**parameters)
    replicated = seqax_forward_schedule(
        **parameters,
        norm_scale_placement=SeqaxNormScalePlacement.REPLICATED,
    )
    sharded_physical = lower_seqax_forward_to_physical(sharded).module
    replicated_physical = lower_seqax_forward_to_physical(replicated).module

    assert sum(isinstance(operation, CollectiveOp) for operation in sharded_physical.walk()) == 20
    assert (
        sum(isinstance(operation, CollectiveOp) for operation in replicated_physical.walk()) == 14
    )
    assert lower_seqax_physical_to_pallas(sharded, sharded_physical).pallas_region_count == 9
    assert lower_seqax_physical_to_pallas(replicated, replicated_physical).pallas_region_count == 9


def test_replicated_weight_data_removes_exact_physical_gather_chains() -> None:
    parameters = {**TILED_SEQAX, "sequence": 1, "layers": 1}
    sharded = seqax_forward_schedule(**parameters)
    replicated = seqax_forward_schedule(
        **parameters,
        weight_data_placement=REPLICATED_WEIGHT_DATA,
    )
    sharded_physical = lower_seqax_forward_to_physical(sharded).module
    replicated_physical = lower_seqax_forward_to_physical(replicated).module

    assert sum(isinstance(operation, CollectiveOp) for operation in sharded_physical.walk()) == 20
    assert (
        sum(isinstance(operation, CollectiveOp) for operation in replicated_physical.walk()) == 12
    )
    assert lower_seqax_physical_to_pallas(sharded, sharded_physical).pallas_region_count == 9
    assert lower_seqax_physical_to_pallas(replicated, replicated_physical).pallas_region_count == 9
    assert {
        placement: sum(
            isinstance(operation, CollectiveOp)
            for operation in lower_seqax_forward_to_physical(
                seqax_forward_schedule(
                    **parameters,
                    weight_data_placement=placement,
                )
            ).module.walk()
        )
        for placement in (
            REPLICATED_EMBEDDING_WEIGHT_DATA,
            REPLICATED_ATTENTION_WEIGHT_DATA,
            REPLICATED_FEED_FORWARD_WEIGHT_DATA,
        )
    } == {
        REPLICATED_EMBEDDING_WEIGHT_DATA: 18,
        REPLICATED_ATTENTION_WEIGHT_DATA: 17,
        REPLICATED_FEED_FORWARD_WEIGHT_DATA: 17,
    }
    combined = lower_seqax_forward_to_physical(
        seqax_forward_schedule(
            **parameters,
            weight_data_placement=REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA,
        )
    ).module
    assert sum(isinstance(operation, CollectiveOp) for operation in combined.walk()) == 15


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


def test_pallas_plan_rejects_a_bypassed_typed_materialization_boundary() -> None:
    distributed = seqax_forward_schedule(
        **SMALL_SEQAX,
        numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_V1,
    )
    physical = lower_seqax_forward_to_physical(distributed).module
    plan = lower_seqax_physical_to_pallas(distributed, physical)
    property_text = ", materialization = #tpu_schedule<vector_materialization strict_typed>"
    mutated = plan.canonical_physical_xdsl.replace(property_text, "", 1)
    assert mutated != plan.canonical_physical_xdsl

    with pytest.raises(
        UnsupportedSeqaxPallasLoweringError,
        match="physical schedule hash mismatch",
    ):
        replace(plan, canonical_physical_xdsl=mutated)._validated_modules()


def test_explicit_tiled_physical_schedule_is_canonical_and_replayable() -> None:
    distributed = seqax_forward_schedule(**TILED_SEQAX)
    default = lower_seqax_forward_to_physical(distributed).module
    tiles = _partial_tpu_tiles(_einsum_tiles(default))
    tiled = lower_seqax_forward_to_physical(distributed, einsum_tiles=tiles).module
    plan = lower_seqax_physical_to_pallas(distributed, tiled)

    assert _einsum_tiles(tiled) == tiles
    assert sum(left != right for left, right in zip(_einsum_tiles(default), tiles)) == 13
    assert plan.physical_schedule_sha256 != _plan().physical_schedule_sha256
    replayed_distributed, replayed_physical = plan._validated_modules()
    assert canonical_module_text(replayed_distributed) == plan.canonical_distributed_xdsl
    assert canonical_module_text(replayed_physical) == plan.canonical_physical_xdsl


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


def test_pallas_einsum_executes_declared_m_k_n_tiles() -> None:
    lhs_alloc = AllocOp(
        buffer((2, 16, 256), ("B", "M", "K"), bf16, memory=MemorySpace.VMEM).to_type(),
        "lhs",
    )
    rhs_alloc = AllocOp(
        buffer((2, 256, 256), ("B", "K", "N"), bf16, memory=MemorySpace.VMEM).to_type(),
        "rhs",
    )
    output_alloc = AllocOp(
        buffer((2, 16, 256), ("B", "M", "N"), f32, memory=MemorySpace.VMEM).to_type(),
        "output",
    )
    operation = MxuEinsumOp(
        lhs_alloc,
        rhs_alloc,
        output_alloc,
        stage=0,
        contracting_dimensions=("K",),
        tile_m=8,
        tile_k=128,
        tile_n=128,
    )
    operation.verify()
    lhs = jnp.arange(8192, dtype=jnp.bfloat16).reshape(2, 16, 256) / 1024
    rhs = jnp.arange(131072, dtype=jnp.bfloat16).reshape(2, 256, 256) / 4096

    actual = _pallas_einsum(
        operation,
        lhs,
        rhs,
        interpret=True,
        schedule_sha256_value="0" * 64,
    )
    expected = jnp.einsum("bmk,bkn->bmn", lhs, rhs, preferred_element_type=jnp.float32)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)


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

from tpu_cake.seqax_pallas_lowering import _einsum_tiles, lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import SeqaxNumericalSemantics, seqax_forward_schedule
from tpu_cake.seqax_numerical import _validate_strict_silu_stablehlo
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs, seqax_forward_reference

parameters = {
    "batch": 2,
    "sequence": 4,
    "model": 256,
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
distributed = seqax_forward_schedule(
    **parameters,
    numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_V1,
)
default_physical = lower_seqax_forward_to_physical(distributed).module
tiles = tuple(
    (
        m,
        128 if k > 128 and k % 128 == 0 else k,
        128 if n > 128 and n % 128 == 0 else n,
    )
    for m, k, n in _einsum_tiles(default_physical)
)
physical = lower_seqax_forward_to_physical(distributed, einsum_tiles=tiles).module
plan = lower_seqax_physical_to_pallas(distributed, physical)
assert _einsum_tiles(physical) == tiles
assert sum(
    left != right
    for left, right in zip(_einsum_tiles(default_physical), tiles)
) == 13
source = plan.render_executable_source()
namespace = {}
exec(compile(source, "<seqax-physical-pallas>", "exec"), namespace)
assert namespace["PLAN"].manifest() == plan.manifest()
executable, mesh = namespace["build"](interpret=True, devices=devices)
inputs = seqax_forward_inputs(seed=9173, **parameters)
arrays = tuple(jnp.asarray(value) for value in inputs)
stablehlo = str(executable.lower(*arrays).compiler_ir("stablehlo"))
_validate_strict_silu_stablehlo(
    stablehlo,
    expected_count=parameters["layers"],
    instrumented=False,
    allow_callbacks=True,
)
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
