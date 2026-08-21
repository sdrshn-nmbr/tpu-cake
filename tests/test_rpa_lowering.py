import hashlib
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from tpu_cake import rpa_lowering
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import (
    BufferType,
    FusedRaggedPagedAttentionOp,
    KernelOp,
    LayoutAttr,
    RpaDecodeCoreOp,
    SemaphoreAllocOp,
    ShardingAttr,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.lowering import UnsupportedLoweringError
from tpu_cake.physical_cost_model import (
    UnsupportedPhysicalCostModelError,
    analyze_physical_kernel,
)
from tpu_cake.rpa_lowering import (
    OwnedRpaDecodeCorePlan,
    ShardedFusedRpaPlan,
    lower_inkling_owned_rpa_decode_core_to_pallas,
    lower_inkling_rpa_to_pallas,
    lower_inkling_sharded_rpa_to_pallas,
)
from tpu_cake.workloads.inkling_rpa import (
    inkling_fused_rpa_schedule,
    inkling_owned_rpa_decode_core_schedule,
    inkling_sharded_fused_rpa_schedule,
)

_OBSERVED: dict[str, object] = {}


def _successful_fake_kernel(*args, **kwargs):
    _OBSERVED["args"] = args
    _OBSERVED["kwargs"] = kwargs
    return (
        jnp.ones(args[0].shape, dtype=args[0].dtype),
        jnp.ones(args[3].shape, dtype=args[3].dtype),
    )


def _bad_result_fake_kernel(*args, **_kwargs):
    return np.zeros((1,), dtype=np.float32), jnp.zeros(args[3].shape, dtype=args[3].dtype)


@wraps(_successful_fake_kernel)
def _forged_fake_kernel(*_args, **_kwargs):
    raise RuntimeError("forged executor ran")


def _trust_test_callable(plan, kernel):
    source = Path(__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = ((source.name, digest),)
    return replace(
        plan,
        backend_module=kernel.__module__,
        backend_executor_qualname=kernel.__qualname__,
        backend_sha256=digest,
        backend_manifest=manifest,
    )


def _valid_inputs(plan):
    inputs = tuple(
        jnp.zeros(shape, dtype=dtype)
        for shape, dtype in zip(plan.input_shapes, plan.input_dtypes, strict=True)
    )
    return (
        *inputs[:4],
        jnp.asarray((1, 17, 33, 49), dtype=jnp.int32),
        jnp.arange(32, dtype=jnp.int32),
        jnp.arange(5, dtype=jnp.int32),
        jnp.asarray((0, 16, 48, 96, 160), dtype=jnp.int32),
        jnp.full((3,), 4, dtype=jnp.int32),
        *inputs[9:],
    )


def _valid_owned_inputs(plan: OwnedRpaDecodeCorePlan):
    inputs = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(
            plan.global_input_shapes,
            plan.external_input_dtypes,
            strict=True,
        )
    )
    local_kv_lengths = jnp.asarray((1, 17, 33, 49), dtype=jnp.int32)
    local_cumulative_kv_lengths = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(local_kv_lengths, dtype=jnp.int32),
        )
    )
    return (
        *inputs[:4],
        jnp.tile(local_kv_lengths, 2),
        jnp.tile(jnp.arange(plan.external_input_shapes[5][0], dtype=jnp.int32), 2),
        jnp.tile(jnp.arange(5, dtype=jnp.int32), 2),
        jnp.tile(local_cumulative_kv_lengths, 2),
        jnp.full((6,), 4, dtype=jnp.int32),
        *inputs[9:],
    )


def test_fused_rpa_lowers_to_a_stable_upstream_plan() -> None:
    first = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())
    second = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())

    assert first == second
    assert first.query_shape == (4, 4, 32)
    assert first.fused_cache_shape == (32, 16, 2, 2, 128)
    assert first.decode_block_sizes == (8, 128, 8, 128)
    assert first.softmax_scale == 0.03125
    assert first.softmax_dtype == "float32"
    assert (
        first.backend_sha256 == "56d00d027cf921def1908e4815ced12e79210e1ac3cf57bcd727c5e6c6168eaa"
    )
    source = first.render_executable_source()
    compile(source, "lowered_rpa.py", "exec")
    assert "ragged_paged_attention_v3" in source
    assert first.source_sha256() == second.source_sha256()


def test_legacy_fused_rpa_identities_remain_unchanged() -> None:
    module = inkling_fused_rpa_schedule()
    plan = lower_inkling_rpa_to_pallas(module)

    assert schedule_sha256(module) == (
        "e1de39a3dfa8f4c930207fbe5218aa1f956b19b0b933e993fe35ae916e8ecad7"
    )
    assert plan.source_sha256() == (
        "d441631f10bb58ca3efe8c03230508e33eeab13efd511b1dde416c8fd9a4428a"
    )


def test_sharded_fused_rpa_lowers_to_the_production_mesh_contract() -> None:
    first = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    second = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())

    assert isinstance(first, ShardedFusedRpaPlan)
    assert first == second
    assert first.mesh_axes == ("data", "tensor")
    assert first.mesh_shape == (2, 4)
    assert first.local_plan.query_shape == (4, 8, 128)
    assert first.local_plan.fused_cache_shape == (3712, 1, 4, 2, 128)
    assert first.local_plan.decode_block_sizes == (8, 128, 8, 128)
    assert first.global_input_shapes == (
        (8, 32, 128),
        (8, 16, 128),
        (8, 16, 128),
        (7424, 1, 16, 2, 128),
        (8,),
        (16384,),
        (10,),
        (10,),
        (6,),
        (8, 32, 16),
        (16, 512),
    )
    assert first.global_output_shapes == (
        (8, 32, 128),
        (7424, 1, 16, 2, 128),
    )
    assert first.local_plan.backend_repository_revision == (
        "9e1a7d39ccdcf9f396e024bfc45935f4f50f70c7"
    )
    assert first.local_plan.backend_sha256 == (
        "12c6aeeade66538d3bb638f048850c3d69095ade4ec42559cd8b3566bfc68897"
    )
    assert first.source_sha256() == second.source_sha256()


def test_owned_rpa_decode_core_exposes_the_static_physical_resources() -> None:
    module = inkling_owned_rpa_decode_core_schedule()
    module.verify()
    assert schedule_sha256(module) == (
        "41946eb01074fd2ecd2831190321631f09d9c25fb7c3ee68a28ebc272f6d576a"
    )
    operation = next(
        operation for operation in module.walk() if isinstance(operation, RpaDecodeCoreOp)
    )

    assert operation.execution_authority.data == "tpu-cake-owned-pallas-decode-core-v1"
    assert operation.attention_case.data == "decode"
    assert operation.masking_authority.data == "decode-kv-length-no-custom-mask-v1"
    assert operation.relative_dimension.data == 16
    assert tuple(value.data for value in operation.cumulative_mask_initial) == (0,)
    assert tuple(value.data for value in operation.semaphore_ids_initial) == (0, 0, 0)
    assert tuple(value.data for value in operation.output_ids_initial) == (-1,) * 4
    assert tuple(value.data for value in operation.cache_update_ids_initial) == (-1,) * 6
    assert operation.backend_repository_revision.data == (
        "9e1a7d39ccdcf9f396e024bfc45935f4f50f70c7"
    )
    assert operation.backend_file_revision.data == ("ac88a2ecfa905965b43edbbb5e6510eb272d09e5")
    assert operation.backend_sha256.data == (
        "12c6aeeade66538d3bb638f048850c3d69095ade4ec42559cd8b3566bfc68897"
    )
    assert (
        operation.query_fetch_size.data,
        operation.kv_fetch_size.data,
        operation.query_compute_size.data,
        operation.kv_compute_size.data,
    ) == (8, 128, 8, 128)
    assert operation.buffer_slots.data == 2
    assert operation.dma_channels.data == 5
    assert operation.prefetch_distance.data == 1
    assert operation.relative_extent.data == 512
    assert operation.relative_states_resident.type.storage.get_shape() == (4, 4, 2, 128)
    assert operation.relative_projection_resident.type.storage.get_shape() == (128, 4608)
    assert operation.kv_double_buffer.type.storage.get_shape() == (2, 128, 5, 2, 128)
    assert operation.query_double_buffer.type.storage.get_shape() == (2, 4, 8, 1, 2, 128)
    assert operation.output_double_buffer.type.storage.get_shape() == (2, 4, 8, 1, 2, 128)
    assert operation.online_l.type.storage.get_shape() == (4, 16, 128)
    assert operation.online_m.type.storage.get_shape() == (4, 16, 128)
    assert operation.accumulator.type.storage.get_shape() == (4, 16, 128)
    assert operation.compute_intermediates.type.storage.get_shape() == (4, 8, 2, 256, 4)
    assert operation.cumulative_mask_lengths.type.storage.get_shape() == (1,)
    assert operation.semaphore_ids.type.storage.get_shape() == (3,)
    assert operation.output_ids.type.storage.get_shape() == (4,)
    assert operation.cache_update_ids.type.storage.get_shape() == (6,)
    assert isinstance(operation.dma_semaphores.owner, SemaphoreAllocOp)
    assert operation.dma_semaphores.owner.slot_count == 10
    assert operation.output is operation.queries
    assert operation.updated_cache is operation.fused_cache


def test_owned_rpa_decode_core_rejects_resource_contract_drift() -> None:
    wrong_block = inkling_owned_rpa_decode_core_schedule()
    operation = next(
        operation for operation in wrong_block.walk() if isinstance(operation, RpaDecodeCoreOp)
    )
    operation.properties["kv_fetch_size"] = IntAttr(256)
    with pytest.raises(VerifyException, match="scratch shapes"):
        wrong_block.verify()

    wrong_semaphore = inkling_owned_rpa_decode_core_schedule()
    operation = next(
        operation for operation in wrong_semaphore.walk() if isinstance(operation, RpaDecodeCoreOp)
    )
    assert isinstance(operation.dma_semaphores.owner, SemaphoreAllocOp)
    operation.dma_semaphores.owner.properties["slots"] = IntAttr(8)
    with pytest.raises(VerifyException, match="two buffers, five DMA channels"):
        wrong_semaphore.verify()

    undersized = inkling_owned_rpa_decode_core_schedule()
    kernel = next(operation for operation in undersized.walk() if isinstance(operation, KernelOp))
    kernel.properties["vmem_capacity_bytes"] = IntAttr(1_957_888)
    with pytest.raises(VerifyException, match="VMEM capacity exceeded.*2220032 > 1957888"):
        undersized.verify()

    wrong_source = inkling_owned_rpa_decode_core_schedule()
    operation = next(
        operation for operation in wrong_source.walk() if isinstance(operation, RpaDecodeCoreOp)
    )
    operation.properties["backend_sha256"] = StringAttr("not-a-sha")
    with pytest.raises(VerifyException, match="backend identity is not canonical"):
        wrong_source.verify()

    wrong_target = inkling_owned_rpa_decode_core_schedule()
    kernel = next(operation for operation in wrong_target.walk() if isinstance(operation, KernelOp))
    kernel.properties["target"] = StringAttr("tpu6e")
    with pytest.raises(VerifyException, match="requires a TPU7x kernel"):
        wrong_target.verify()

    wrong_semantics = inkling_owned_rpa_decode_core_schedule()
    operation = next(
        operation for operation in wrong_semantics.walk() if isinstance(operation, RpaDecodeCoreOp)
    )
    operation.properties["attention_case"] = StringAttr("prefill")
    with pytest.raises(VerifyException, match="execution semantics are not canonical"):
        wrong_semantics.verify()


def test_owned_rpa_decode_core_has_no_unearned_cost_model_claim() -> None:
    with pytest.raises(
        UnsupportedPhysicalCostModelError,
        match="no work convention for tpu_schedule.rpa_decode_core",
    ):
        analyze_physical_kernel(
            inkling_owned_rpa_decode_core_schedule(),
            hardware=tpu7x_tensorcore_rates(),
        )


def test_owned_rpa_decode_core_lowers_to_its_owned_pallas_plan() -> None:
    module = inkling_owned_rpa_decode_core_schedule()
    first = lower_inkling_owned_rpa_decode_core_to_pallas(module)
    second = lower_inkling_owned_rpa_decode_core_to_pallas(inkling_owned_rpa_decode_core_schedule())

    assert isinstance(first, OwnedRpaDecodeCorePlan)
    assert first == second
    assert first.schedule_sha256 == (
        "41946eb01074fd2ecd2831190321631f09d9c25fb7c3ee68a28ebc272f6d576a"
    )
    assert first.core_input_shapes == (
        (4, 4, 1, 2, 128),
        (4, 4, 2, 128),
        (3712, 1, 4, 2, 128),
        (4,),
        (8192,),
        (5,),
        (5,),
        (3,),
        (4, 4, 2, 128),
        (128, 4608),
    )
    assert first.external_input_shapes == (
        (4, 8, 128),
        (4, 4, 128),
        (4, 4, 128),
        (3712, 1, 4, 2, 128),
        (4,),
        (8192,),
        (5,),
        (5,),
        (3,),
        (4, 8, 16),
        (16, 512),
    )
    assert first.block_sizes == (8, 128, 8, 128)
    assert first.core_input_output_aliases == ((9, 0), (11, 1))
    assert first.external_donate_argnums == (0, 3)
    assert first.implementation_sha256 == (
        "17b2c99bd4cd76fef29ed17e4e4dd9238a9a1442015083b3995aa251b6342e88"
    )
    assert first.lowering_sha256 == (
        "067aaa964bb096a9cce4012daa69a616a016ce6dd3a74d84906393bee2fba698"
    )
    assert first.source_sha256() == (
        "331e31dd32985b8818e1b376ca390339a7aa0023a367b407466a59de2d153b94"
    )
    assert first.source_sha256() == second.source_sha256()
    with pytest.raises(ValueError, match="interpret mode does not support"):
        first._build_local(interpret=True)
    assert callable(first._build_local(interpret=False))

    forged = inkling_owned_rpa_decode_core_schedule()
    operation = next(
        operation for operation in forged.walk() if isinstance(operation, RpaDecodeCoreOp)
    )
    operation.properties["backend_sha256"] = StringAttr("0" * 64)
    with pytest.raises(UnsupportedLoweringError, match="backend identity is not pinned"):
        lower_inkling_owned_rpa_decode_core_to_pallas(forged)


def test_owned_rpa_decode_core_preflight_rejects_bad_decode_metadata() -> None:
    plan = lower_inkling_owned_rpa_decode_core_to_pallas(inkling_owned_rpa_decode_core_schedule())
    inputs = _valid_owned_inputs(plan)
    plan.preflight(*inputs)

    wrong_distribution = (*inputs[:8], jnp.zeros((6,), dtype=jnp.int32), *inputs[9:])
    with pytest.raises(ValueError, match="decode-only distribution"):
        plan.preflight(*wrong_distribution)

    bad_pages = (
        *inputs[:5],
        jnp.full(plan.global_input_shapes[5], 99_999, dtype=jnp.int32),
        *inputs[6:],
    )
    with pytest.raises(ValueError, match="outside the fused cache"):
        plan.preflight(*bad_pages)

    colliding_pages = inputs[5].at[17].set(inputs[5][0])
    collisions = (*inputs[:5], colliding_pages, *inputs[6:])
    with pytest.raises(ValueError, match="updates collide"):
        plan.preflight(*collisions)


def test_owned_rpa_decode_core_places_only_preflighted_global_inputs(monkeypatch) -> None:
    plan = lower_inkling_owned_rpa_decode_core_to_pallas(inkling_owned_rpa_decode_core_schedule())
    inputs = _valid_owned_inputs(plan)
    monkeypatch.setattr(rpa_lowering, "NamedSharding", lambda mesh, spec: (mesh, spec))
    monkeypatch.setattr(rpa_lowering.np, "array", lambda value, copy: value)
    monkeypatch.setattr(rpa_lowering.jax, "device_put", lambda value, _sharding: value)

    assert plan.place_inputs(inputs, mesh=object()) == inputs

    wrong_distribution = (*inputs[:8], jnp.zeros((6,), dtype=jnp.int32), *inputs[9:])
    with pytest.raises(ValueError, match="decode-only distribution"):
        plan.place_inputs(wrong_distribution, mesh=object())


def test_owned_rpa_decode_core_uses_the_exact_tpu7_output_abi(monkeypatch) -> None:
    plan = lower_inkling_owned_rpa_decode_core_to_pallas(inkling_owned_rpa_decode_core_schedule())
    captured = {}

    def fake_pallas_call(*_args, **kwargs):
        captured.update(kwargs)
        return lambda *_inputs: ()

    monkeypatch.setattr(rpa_lowering.pl, "pallas_call", fake_pallas_call)
    plan._build_local(interpret=False)

    assert captured["input_output_aliases"] == {9: 0, 11: 1}
    assert all(output.memory_space == rpa_lowering.pltpu.HBM for output in captured["out_shape"])


def test_sharded_fused_rpa_rejects_the_wrong_device_inventory() -> None:
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    tpu_devices = tuple(SimpleNamespace(device_kind="TPU7x") for _ in range(8))

    with pytest.raises(ValueError, match="needs 8 devices"):
        plan.mesh(tpu_devices[:-1])
    with pytest.raises(ValueError, match="only TPU7x"):
        plan.mesh((*tpu_devices[:-1], SimpleNamespace(device_kind="TPU v6e")))


def test_sharded_fused_rpa_rejects_a_wrong_mesh_or_partition() -> None:
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    with pytest.raises(ValueError, match="exact 2x4 data/tensor mesh"):
        replace(plan, mesh_shape=(1, 8))

    module = inkling_sharded_fused_rpa_schedule()
    kernel = next(operation for operation in module.walk() if isinstance(operation, KernelOp))
    query = kernel.body.block.args[0]
    query_type = query.type
    assert isinstance(query_type, BufferType)
    query._type = BufferType(
        query_type.storage,
        query_type.shape,
        query_type.space,
        ShardingAttr(ArrayAttr(StringAttr(axis) for axis in ("", "tensor", ""))),
        query_type.layout,
        query_type.ownership,
        query_type.lifetime,
    )
    with pytest.raises(VerifyException, match="conflicting global sizes"):
        module.verify()
    with pytest.raises(VerifyException, match="conflicting global sizes"):
        lower_inkling_sharded_rpa_to_pallas(module)


def test_fused_rpa_plan_invokes_the_exact_serving_contract() -> None:
    plan = _trust_test_callable(
        lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule()),
        _successful_fake_kernel,
    )
    inputs = _valid_inputs(plan)
    _OBSERVED.clear()

    output, cache = plan.invoke(
        _successful_fake_kernel,
        *inputs,
        backend_manifest=plan.backend_manifest,
        device_kind="TPU7x",
    )

    assert output.shape == plan.output_shape
    assert cache.shape == plan.fused_cache_shape
    assert len(_OBSERVED["args"]) == 10
    assert _OBSERVED["args"][-1] is None
    assert _OBSERVED["kwargs"] | {
        "relative_states": None,
        "relative_projection": None,
    } == {
        "causal": 1,
        "sm_scale": pytest.approx(0.03125),
        "softmax_dtype": jnp.float32,
        "sliding_window": None,
        "d_block_sizes": (8, 128, 8, 128),
        "vmem_limit_bytes": 96 << 20,
        "relative_states": None,
        "relative_projection": None,
    }
    assert _OBSERVED["kwargs"]["relative_states"] is inputs[9]
    assert _OBSERVED["kwargs"]["relative_projection"] is inputs[10]


def test_fused_rpa_plan_rejects_unverified_backend_and_bad_results() -> None:
    production_plan = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())
    plan = _trust_test_callable(production_plan, _bad_result_fake_kernel)
    inputs = _valid_inputs(plan)

    with pytest.raises(ValueError, match="pinned backend callable"):
        production_plan.invoke(
            _successful_fake_kernel,
            *inputs,
            backend_manifest=production_plan.backend_manifest,
            device_kind="TPU7x",
        )

    trusted_test_plan = _trust_test_callable(production_plan, _successful_fake_kernel)
    wrong_auxiliary_plan = replace(
        trusted_test_plan,
        backend_manifest=(
            *trusted_test_plan.backend_manifest,
            ("test_evidence.py", "0" * 64),
        ),
    )
    with pytest.raises(ValueError, match="executor source"):
        wrong_auxiliary_plan.validate_backend_callable(_successful_fake_kernel)

    with pytest.raises(ValueError, match="pinned backend callable"):
        trusted_test_plan.invoke(
            _forged_fake_kernel,
            *inputs,
            backend_manifest=trusted_test_plan.backend_manifest,
            device_kind="TPU7x",
        )

    with pytest.raises(ValueError, match="source manifest"):
        plan.invoke(
            _bad_result_fake_kernel,
            *inputs,
            backend_manifest=(("wrong.py", "0" * 64),),
            device_kind="TPU7x",
        )

    with pytest.raises(ValueError, match="result 0"):
        plan.invoke(
            _bad_result_fake_kernel,
            *inputs,
            backend_manifest=plan.backend_manifest,
            device_kind="TPU7x",
        )


@pytest.mark.parametrize("device_kind", ("TPU7x", "TPU v7x"))
def test_fused_rpa_plan_accepts_exact_tpu7_device_names(device_kind: str) -> None:
    plan = _trust_test_callable(
        lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule()),
        _successful_fake_kernel,
    )
    inputs = _valid_inputs(plan)

    output, cache = plan.invoke(
        _successful_fake_kernel,
        *inputs,
        backend_manifest=plan.backend_manifest,
        device_kind=device_kind,
    )

    assert output.shape == plan.output_shape
    assert cache.shape == plan.fused_cache_shape


@pytest.mark.parametrize("device_kind", ("not-TPU7-emulator", "TPU v6e", "gpu"))
def test_fused_rpa_plan_rejects_non_tpu7_device_names(device_kind: str) -> None:
    plan = _trust_test_callable(
        lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule()),
        _successful_fake_kernel,
    )
    inputs = _valid_inputs(plan)

    with pytest.raises(ValueError, match="requires TPU7x"):
        plan.invoke(
            _successful_fake_kernel,
            *inputs,
            backend_manifest=plan.backend_manifest,
            device_kind=device_kind,
        )


def test_fused_rpa_preflight_rejects_bad_decode_metadata() -> None:
    plan = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())
    inputs = _valid_inputs(plan)
    plan.preflight(*inputs)

    wrong_distribution = (*inputs[:8], jnp.zeros((3,), dtype=jnp.int32), *inputs[9:])
    with pytest.raises(ValueError, match="decode-only distribution"):
        plan.preflight(*wrong_distribution)

    bad_pages = (*inputs[:5], jnp.full((32,), 99, dtype=jnp.int32), *inputs[6:])
    with pytest.raises(ValueError, match="outside the fused cache"):
        plan.preflight(*bad_pages)

    colliding_pages = inputs[5].at[2].set(inputs[5][0])
    collisions = (*inputs[:5], colliding_pages, *inputs[6:])
    with pytest.raises(ValueError, match="updates collide"):
        plan.preflight(*collisions)

    duplicate_sequence_page = inputs[5].at[2].set(inputs[5][1])
    aliased_sequence = (*inputs[:5], duplicate_sequence_page, *inputs[6:])
    with pytest.raises(ValueError, match="aliases logical pages"):
        plan.preflight(*aliased_sequence)

    shared_write_page = inputs[5].at[1].set(inputs[5][0])
    cross_sequence_alias = (
        *inputs[:4],
        jnp.asarray((1, 2, 33, 49), dtype=jnp.int32),
        shared_write_page,
        inputs[6],
        jnp.asarray((0, 16, 32, 80, 144), dtype=jnp.int32),
        *inputs[8:],
    )
    with pytest.raises(ValueError, match="active write page is shared"):
        plan.preflight(*cross_sequence_alias)

    huge_lengths = jnp.full((4,), np.iinfo(np.int32).max, dtype=jnp.int32)
    overflowed_cumulative = jnp.asarray(
        (0, np.iinfo(np.int32).min, 0, np.iinfo(np.int32).min, 0),
        dtype=jnp.int32,
    )
    overflow = (
        *inputs[:4],
        huge_lengths,
        inputs[5],
        inputs[6],
        overflowed_cumulative,
        *inputs[8:],
    )
    with pytest.raises(ValueError, match="exceeds total cache capacity"):
        plan.preflight(*overflow)


def test_fused_rpa_preflight_is_enforced_before_kernel_execution() -> None:
    plan = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())
    inputs = _valid_inputs(plan)
    colliding_pages = inputs[5].at[2].set(inputs[5][0])
    collisions = (*inputs[:5], colliding_pages, *inputs[6:])
    kernel_called = False

    def fake_kernel(*_args, **_kwargs):
        nonlocal kernel_called
        kernel_called = True
        raise AssertionError("invalid metadata reached the kernel")

    with pytest.raises(ValueError, match="updates collide"):
        plan.run_preflighted(
            fake_kernel,
            *collisions,
            backend_manifest=plan.backend_manifest,
            device_kind="TPU7x",
        )
    assert not kernel_called


def test_fused_rpa_traced_invocation_has_no_host_array_conversion() -> None:
    plan = _trust_test_callable(
        lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule()),
        _successful_fake_kernel,
    )
    inputs = _valid_inputs(plan)

    invoked = jax.jit(
        lambda *values: plan.invoke(
            _successful_fake_kernel,
            *values,
            backend_manifest=plan.backend_manifest,
            device_kind="TPU7x",
        )
    )
    output, cache = invoked(*inputs)

    assert output.shape == plan.output_shape
    assert cache.shape == plan.fused_cache_shape


def test_fused_rpa_lowering_rejects_a_non_tpu_target() -> None:
    module = inkling_fused_rpa_schedule()
    kernel = next(operation for operation in module.walk() if isinstance(operation, KernelOp))
    kernel.properties["target"] = StringAttr("gpu")
    module.verify()

    with pytest.raises(UnsupportedLoweringError, match="does not support target 'gpu'"):
        lower_inkling_rpa_to_pallas(module)


def test_fused_rpa_plan_uses_canonical_identity_and_source_location() -> None:
    first_module = inkling_fused_rpa_schedule()
    second_module = inkling_fused_rpa_schedule()
    second_kernel = next(
        operation for operation in second_module.walk() if isinstance(operation, KernelOp)
    )
    second_kernel.properties["sym_name"] = StringAttr("decorative_name")
    first = lower_inkling_rpa_to_pallas(first_module)
    second = lower_inkling_rpa_to_pallas(second_module)
    attention = next(
        operation
        for operation in first_module.walk()
        if isinstance(operation, FusedRaggedPagedAttentionOp)
    )

    assert schedule_sha256(first_module) == schedule_sha256(second_module)
    assert first == second
    assert first.source_sha256() == second.source_sha256()
    assert str(attention.location) != "loc(unknown)"


def test_fused_rpa_verifier_failure_names_the_pinned_source() -> None:
    module = inkling_fused_rpa_schedule()
    attention = next(
        operation
        for operation in module.walk()
        if isinstance(operation, FusedRaggedPagedAttentionOp)
    )
    attention.properties["softmax_scale"] = StringAttr("1")

    with pytest.raises(VerifyException, match=r"ragged_paged_attention_v3\.py.*1802.*1"):
        lower_inkling_rpa_to_pallas(module)


def test_fused_rpa_lowering_failure_names_the_pinned_source() -> None:
    module = inkling_fused_rpa_schedule()
    attention = next(
        operation
        for operation in module.walk()
        if isinstance(operation, FusedRaggedPagedAttentionOp)
    )
    query_type = attention.queries.type
    assert isinstance(query_type, BufferType)
    attention.queries._type = BufferType(
        query_type.storage,
        query_type.shape,
        query_type.space,
        query_type.sharding,
        LayoutAttr(ArrayAttr((IntAttr(1), IntAttr(0), IntAttr(2)))),
        query_type.ownership,
        query_type.lifetime,
    )
    module.verify()

    with pytest.raises(
        UnsupportedLoweringError,
        match=r"ragged_paged_attention_v3\.py.*1802.*1",
    ):
        lower_inkling_rpa_to_pallas(module)


def test_fused_rpa_adapter_rejects_custom_mask_mode() -> None:
    module = inkling_fused_rpa_schedule()
    attention = next(
        operation
        for operation in module.walk()
        if isinstance(operation, FusedRaggedPagedAttentionOp)
    )
    attention.properties["causal"] = IntAttr(0)

    with pytest.raises(VerifyException, match="requires causal attention"):
        module.verify()


def test_fused_rpa_adapter_rejects_wrong_inkling_scale() -> None:
    module = inkling_fused_rpa_schedule()
    attention = next(
        operation
        for operation in module.walk()
        if isinstance(operation, FusedRaggedPagedAttentionOp)
    )
    attention.properties["softmax_scale"] = StringAttr("0.1767766952966369")

    with pytest.raises(VerifyException, match="1 / head dimension"):
        module.verify()


def test_fused_rpa_lowering_rejects_unrepresented_physical_layout() -> None:
    module = inkling_fused_rpa_schedule()
    kernel = next(operation for operation in module.walk() if isinstance(operation, KernelOp))
    argument = kernel.body.block.args[0]
    value_type = argument.type
    assert isinstance(value_type, BufferType)
    argument._type = BufferType(
        value_type.storage,
        value_type.shape,
        value_type.space,
        value_type.sharding,
        LayoutAttr(ArrayAttr(IntAttr(index) for index in (2, 1, 0))),
        value_type.ownership,
        value_type.lifetime,
    )
    module.verify()

    with pytest.raises(UnsupportedLoweringError, match="default physical layout"):
        lower_inkling_rpa_to_pallas(module)
