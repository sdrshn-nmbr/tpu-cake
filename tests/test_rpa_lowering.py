import hashlib
from dataclasses import replace
from functools import wraps
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import (
    BufferType,
    FusedRaggedPagedAttentionOp,
    KernelOp,
    LayoutAttr,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.lowering import UnsupportedLoweringError
from tpu_cake.rpa_lowering import lower_inkling_rpa_to_pallas
from tpu_cake.workloads.inkling_rpa import inkling_fused_rpa_schedule

_OBSERVED: dict[str, object] = {}


def _successful_fake_kernel(*args, **kwargs):
    _OBSERVED["args"] = args
    _OBSERVED["kwargs"] = kwargs
    return (
        jnp.ones(args[0].shape, dtype=args[0].dtype),
        jnp.ones(args[3].shape, dtype=args[3].dtype),
    )


def _bad_result_fake_kernel(*args, **_kwargs):
    return np.zeros((1,), dtype=np.float32), jnp.zeros(
        args[3].shape, dtype=args[3].dtype
    )


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


def test_fused_rpa_lowers_to_a_stable_upstream_plan() -> None:
    first = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())
    second = lower_inkling_rpa_to_pallas(inkling_fused_rpa_schedule())

    assert first == second
    assert first.query_shape == (4, 4, 32)
    assert first.fused_cache_shape == (32, 16, 2, 2, 128)
    assert first.decode_block_sizes == (8, 128, 8, 128)
    assert first.softmax_scale == 0.03125
    assert first.softmax_dtype == "float32"
    assert first.backend_sha256 == "56d00d027cf921def1908e4815ced12e79210e1ac3cf57bcd727c5e6c6168eaa"
    source = first.render_executable_source()
    compile(source, "lowered_rpa.py", "exec")
    assert "ragged_paged_attention_v3" in source
    assert first.source_sha256() == second.source_sha256()


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
