import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xdsl.dialects.builtin import bf16, f16, f32, i1, i32
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    MemorySpace,
    MxuEinsumOp,
    Ownership,
    VectorComputeOp,
    VectorImplementation,
    VectorMaterialization,
)
from tpu_cake.frontend import KernelBuilder, buffer
from tpu_cake.seqax_physical_execution import _vector_compute


def _spec(
    shape: tuple[int, ...],
    logical: tuple[str, ...],
    *,
    dtype=bf16,
    memory: MemorySpace = MemorySpace.VMEM,
    sharding: tuple[str, ...] | None = None,
):
    return buffer(
        shape,
        logical,
        dtype,
        memory=memory,
        sharding=sharding,
        ownership=Ownership.EXTERNAL if memory is MemorySpace.HBM else Ownership.KERNEL,
        lifetime=(0, 4),
    )


def test_named_mxu_einsum_verifies_inside_a_complete_physical_kernel() -> None:
    external = (
        _spec((2, 3, 4), ("B", "M", "K"), memory=MemorySpace.HBM),
        _spec((2, 4, 5), ("B", "K", "N"), memory=MemorySpace.HBM),
        _spec((2, 3, 5), ("B", "M", "N"), dtype=f32, memory=MemorySpace.HBM),
    )
    builder = KernelBuilder(
        "batched_einsum",
        "tpu7x",
        external,
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 20,
    )
    lhs = builder.alloc(_spec((2, 3, 4), ("B", "M", "K")), "lhs")
    rhs = builder.alloc(_spec((2, 4, 5), ("B", "K", "N")), "rhs")
    accumulator = builder.alloc(
        _spec((2, 3, 5), ("B", "M", "N"), dtype=f32),
        "accumulator",
    )
    lhs_semaphore = builder.semaphore()
    rhs_semaphore = builder.semaphore()
    output_semaphore = builder.semaphore()
    lhs_dma = builder.dma_start(builder.inputs[0], lhs, lhs_semaphore, stage=0)
    rhs_dma = builder.dma_start(builder.inputs[1], rhs, rhs_semaphore, stage=0)
    builder.dma_wait(lhs_dma, stage=1)
    builder.dma_wait(rhs_dma, stage=1)
    builder.einsum(
        lhs,
        rhs,
        accumulator,
        stage=2,
        contracting_dimensions=("K",),
        tile_m=3,
        tile_k=4,
        tile_n=5,
    )
    builder.dma(
        accumulator,
        builder.inputs[2],
        output_semaphore,
        start_stage=3,
        wait_stage=4,
    )

    builder.module().verify()


def test_named_mxu_einsum_rejects_mismatched_local_contraction_extents() -> None:
    lhs = AllocOp(_spec((2, 3, 4), ("B", "M", "K")).to_type(), "lhs")
    rhs = AllocOp(_spec((2, 5, 6), ("B", "K", "N")).to_type(), "rhs")
    output = AllocOp(
        _spec((2, 3, 6), ("B", "M", "N"), dtype=f32).to_type(),
        "output",
    )
    operation = MxuEinsumOp(
        lhs,
        rhs,
        output,
        stage=2,
        contracting_dimensions=("K",),
        tile_m=3,
        tile_k=4,
        tile_n=6,
    )

    with pytest.raises(VerifyException, match="local contraction extents"):
        operation.verify()


def test_named_mxu_einsum_rejects_dropped_free_dimension_sharding() -> None:
    lhs = AllocOp(
        _spec((2, 4), ("B", "K"), sharding=("d", "")).to_type(),
        "lhs",
    )
    rhs = AllocOp(_spec((4, 6), ("K", "N")).to_type(), "rhs")
    output = AllocOp(_spec((2, 6), ("B", "N"), dtype=f32).to_type(), "output")
    operation = MxuEinsumOp(
        lhs,
        rhs,
        output,
        stage=2,
        contracting_dimensions=("K",),
        tile_m=2,
        tile_k=4,
        tile_n=6,
    )

    with pytest.raises(VerifyException, match="preserve retained-dimension sharding"):
        operation.verify()


def test_named_mxu_einsum_rejects_mismatched_contracted_sharding() -> None:
    lhs = AllocOp(
        _spec((2, 4), ("B", "K"), sharding=("", "t")).to_type(),
        "lhs",
    )
    rhs = AllocOp(
        _spec((4, 6), ("K", "N"), sharding=("d", "")).to_type(),
        "rhs",
    )
    output = AllocOp(_spec((2, 6), ("B", "N"), dtype=f32).to_type(), "output")
    operation = MxuEinsumOp(
        lhs,
        rhs,
        output,
        stage=2,
        contracting_dimensions=("K",),
        pending_reduction_axes=("t",),
        tile_m=2,
        tile_k=4,
        tile_n=6,
    )

    with pytest.raises(VerifyException, match="contracted dimensions must have equal sharding"):
        operation.verify()


def test_named_mxu_einsum_requires_pending_contracted_reductions() -> None:
    lhs = AllocOp(
        _spec((2, 4), ("B", "K"), sharding=("", "t")).to_type(),
        "lhs",
    )
    rhs = AllocOp(
        _spec((4, 6), ("K", "N"), sharding=("t", "")).to_type(),
        "rhs",
    )
    output = AllocOp(_spec((2, 6), ("B", "N"), dtype=f32).to_type(), "output")
    operation = MxuEinsumOp(
        lhs,
        rhs,
        output,
        stage=2,
        contracting_dimensions=("K",),
        tile_m=2,
        tile_k=4,
        tile_n=6,
    )

    with pytest.raises(VerifyException, match="pending reductions must match"):
        operation.verify()


def test_kernel_rejects_vector_use_before_partial_reduction() -> None:
    external = (
        _spec(
            (2, 4),
            ("B", "K"),
            memory=MemorySpace.HBM,
            sharding=("", "t"),
        ),
        _spec(
            (4, 6),
            ("K", "N"),
            memory=MemorySpace.HBM,
            sharding=("t", ""),
        ),
    )
    builder = KernelBuilder(
        "partial_reduction",
        "tpu7x",
        external,
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 20,
        mesh={"t": 2},
        interconnect_bandwidth_bytes_per_second={"t": 600_000_000_000},
    )
    lhs = builder.alloc(
        _spec((2, 4), ("B", "K"), sharding=("", "t")),
        "lhs",
    )
    rhs = builder.alloc(
        _spec((4, 6), ("K", "N"), sharding=("t", "")),
        "rhs",
    )
    partial = builder.alloc(_spec((2, 6), ("B", "N"), dtype=f32), "partial")
    consumed = builder.alloc(_spec((2, 6), ("B", "N"), dtype=f32), "consumed")
    lhs_dma = builder.dma_start(builder.inputs[0], lhs, builder.semaphore(), stage=0)
    rhs_dma = builder.dma_start(builder.inputs[1], rhs, builder.semaphore(), stage=0)
    builder.dma_wait(lhs_dma, stage=1)
    builder.dma_wait(rhs_dma, stage=1)
    builder.einsum(
        lhs,
        rhs,
        partial,
        stage=2,
        contracting_dimensions=("K",),
        pending_reduction_axes=("t",),
        tile_m=2,
        tile_k=4,
        tile_n=6,
    )
    builder.vector_compute((partial,), consumed, stage=3, function="silu")

    with pytest.raises(VerifyException, match="consumes a partial reduction"):
        builder.module()


def test_vector_compute_rejects_duplicate_configuration_keys() -> None:
    source = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "source")
    output = AllocOp(_spec((2, 4), ("B", "M"), dtype=f32).to_type(), "output")
    operation = VectorComputeOp(
        (source,),
        output,
        stage=1,
        function="cast",
        configuration=("dtype=f32", "dtype=bf16"),
    )

    with pytest.raises(VerifyException, match="keys must be unique"):
        operation.verify()


def test_vector_compute_strict_materialization_requires_bf16_silu() -> None:
    bf16_source = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "bf16_source")
    bf16_output = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "bf16_output")
    strict_silu = VectorComputeOp(
        (bf16_source,),
        bf16_output,
        stage=1,
        function="silu",
        materialization=VectorMaterialization.STRICT_TYPED,
    )
    strict_silu.verify()

    f32_source = AllocOp(_spec((2, 4), ("B", "M"), dtype=f32).to_type(), "f32_source")
    f32_output = AllocOp(_spec((2, 4), ("B", "M"), dtype=f32).to_type(), "f32_output")
    wrong_dtype = VectorComputeOp(
        (f32_source,),
        f32_output,
        stage=1,
        function="silu",
        materialization=VectorMaterialization.STRICT_TYPED,
    )
    wrong_function = VectorComputeOp(
        (bf16_source, bf16_source),
        bf16_output,
        stage=1,
        function="add",
        materialization=VectorMaterialization.STRICT_TYPED,
    )

    with pytest.raises(VerifyException, match="requires BF16"):
        wrong_dtype.verify()
    with pytest.raises(VerifyException, match="only supported for SiLU"):
        wrong_function.verify()


def test_vector_compute_rejects_missing_slice_contract() -> None:
    source = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "source")
    output = AllocOp(_spec((2,), ("B",)).to_type(), "output")
    operation = VectorComputeOp(
        (source,),
        output,
        stage=1,
        function="slice",
        configuration=(),
    )

    with pytest.raises(VerifyException, match="wrong configuration keys"):
        operation.verify()


def test_vector_compute_rejects_unrelated_packed_mask_output() -> None:
    starts = AllocOp(_spec((2, 4), ("B", "L"), dtype=i1).to_type(), "starts")
    output = AllocOp(_spec((7,), ("X",), dtype=f32).to_type(), "output")
    operation = VectorComputeOp(
        (starts,),
        output,
        stage=1,
        function="packed_causal_mask",
        configuration=(
            "key_dimension=Klen",
            "query_dimension=Qlen",
            "sequence_dimension=L",
        ),
    )

    with pytest.raises(VerifyException, match="wrong output contract"):
        operation.verify()


def test_vector_compute_rejects_unrelated_softmax_mask() -> None:
    value = AllocOp(_spec((2, 4), ("B", "K"), dtype=f32).to_type(), "value")
    mask = AllocOp(_spec((7,), ("X",), dtype=i1).to_type(), "mask")
    output = AllocOp(_spec((2, 4), ("B", "K"), dtype=f32).to_type(), "output")
    operation = VectorComputeOp(
        (value, mask),
        output,
        stage=1,
        function="masked_softmax",
        configuration=("dimension=K",),
    )

    with pytest.raises(VerifyException, match="mask is incompatible"):
        operation.verify()


def test_vector_compute_rejects_rms_norm_over_a_sharded_dimension() -> None:
    value = AllocOp(
        _spec((2, 4), ("B", "M"), sharding=("", "t")).to_type(),
        "value",
    )
    scale = AllocOp(_spec((4,), ("M",), dtype=f32).to_type(), "scale")
    output = AllocOp(
        _spec((2, 4), ("B", "M"), sharding=("", "t")).to_type(),
        "output",
    )
    operation = VectorComputeOp(
        (value, scale),
        output,
        stage=1,
        function="rms_norm",
        configuration=("dimension=M", "epsilon=0.000001"),
    )

    with pytest.raises(VerifyException, match="dimension cannot be sharded"):
        operation.verify()


def test_vector_compute_rejects_softmax_over_a_sharded_dimension() -> None:
    value = AllocOp(
        _spec((2, 4), ("B", "K"), dtype=f32, sharding=("", "t")).to_type(),
        "value",
    )
    mask = AllocOp(
        _spec((2, 4), ("B", "K"), dtype=i1, sharding=("", "t")).to_type(),
        "mask",
    )
    output = AllocOp(
        _spec((2, 4), ("B", "K"), dtype=f32, sharding=("", "t")).to_type(),
        "output",
    )
    operation = VectorComputeOp(
        (value, mask),
        output,
        stage=1,
        function="masked_softmax",
        configuration=("dimension=K",),
    )

    with pytest.raises(VerifyException, match="dimension cannot be sharded"):
        operation.verify()


def test_vector_compute_rejects_slice_over_a_sharded_dimension() -> None:
    source = AllocOp(
        _spec((2, 4), ("Z", "M"), sharding=("t", "")).to_type(),
        "source",
    )
    output = AllocOp(_spec((4,), ("M",)).to_type(), "output")
    operation = VectorComputeOp(
        (source,),
        output,
        stage=1,
        function="slice",
        configuration=("dimension=Z", "index=0"),
    )

    with pytest.raises(VerifyException, match="cannot index a sharded dimension"):
        operation.verify()


def test_vector_compute_rejects_cast_that_renames_dimensions() -> None:
    source = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "source")
    output = AllocOp(_spec((2, 4), ("X", "Y"), dtype=f32).to_type(), "output")
    operation = VectorComputeOp(
        (source,),
        output,
        stage=1,
        function="cast",
        configuration=("dtype=f32",),
    )

    with pytest.raises(VerifyException, match="cannot rename logical dimensions"):
        operation.verify()


def test_vector_compute_rejects_noninteger_embedding_indices() -> None:
    table = AllocOp(_spec((8, 4), ("V", "M")).to_type(), "table")
    indices = AllocOp(_spec((2,), ("B",), dtype=f32).to_type(), "indices")
    output = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "output")
    operation = VectorComputeOp(
        (table, indices),
        output,
        stage=1,
        function="embedding_lookup",
        configuration=("vocabulary_dimension=V",),
    )

    with pytest.raises(VerifyException, match="indices must be integers"):
        operation.verify()


@pytest.mark.parametrize(
    ("function", "input_count"),
    (("silu", 1), ("exp", 1), ("silu_multiply", 2)),
)
def test_vector_compute_rejects_integer_nonlinear_operations(
    function: str,
    input_count: int,
) -> None:
    source = AllocOp(_spec((2, 4), ("B", "M"), dtype=i32).to_type(), "source")
    output = AllocOp(_spec((2, 4), ("B", "M"), dtype=i32).to_type(), "output")
    operation = VectorComputeOp(
        (source,) * input_count,
        output,
        stage=1,
        function=function,
        implementation=(
            VectorImplementation.PALLAS_FULL_LOCAL if function == "silu_multiply" else None
        ),
    )

    with pytest.raises(VerifyException, match="require floating point"):
        operation.verify()


def test_fused_vector_requires_its_declared_pallas_implementation() -> None:
    gate = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "gate")
    up = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "up")
    output = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "output")
    missing = VectorComputeOp(
        (gate, up),
        output,
        stage=1,
        function="silu_multiply",
    )
    misplaced = VectorComputeOp(
        (gate, up),
        output,
        stage=1,
        function="multiply",
        implementation=VectorImplementation.PALLAS_FULL_LOCAL,
    )

    with pytest.raises(VerifyException, match="requires the full-local Pallas"):
        missing.verify()
    with pytest.raises(VerifyException, match="only supported for fused SiLU"):
        misplaced.verify()


@pytest.mark.parametrize("dtype", (f16, f32))
def test_fused_pallas_vector_requires_bf16(dtype) -> None:
    gate = AllocOp(_spec((2, 4), ("B", "M"), dtype=dtype).to_type(), "gate")
    up = AllocOp(_spec((2, 4), ("B", "M"), dtype=dtype).to_type(), "up")
    output = AllocOp(_spec((2, 4), ("B", "M"), dtype=dtype).to_type(), "output")
    operation = VectorComputeOp(
        (gate, up),
        output,
        stage=1,
        function="silu_multiply",
        implementation=VectorImplementation.PALLAS_FULL_LOCAL,
    )

    with pytest.raises(VerifyException, match="requires BF16 buffers"):
        operation.verify()


def test_fused_vector_execution_requires_the_owned_pallas_callback() -> None:
    gate = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "gate")
    up = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "up")
    output = AllocOp(_spec((2, 4), ("B", "M")).to_type(), "output")
    operation = VectorComputeOp(
        (gate, up),
        output,
        stage=1,
        function="silu_multiply",
        implementation=VectorImplementation.PALLAS_FULL_LOCAL,
    )
    operation.verify()

    with pytest.raises(ValueError, match="declared Pallas implementation"):
        _vector_compute(
            operation,
            (
                jnp.ones((2, 4), dtype=jnp.bfloat16),
                jnp.ones((2, 4), dtype=jnp.bfloat16),
            ),
            {},
        )


@pytest.mark.parametrize("maximum_timescale", ("nan", "inf"))
def test_vector_compute_rejects_nonfinite_rotary_timescale(
    maximum_timescale: str,
) -> None:
    source = AllocOp(_spec((2, 4), ("L", "D")).to_type(), "source")
    output = AllocOp(_spec((2, 4), ("L", "D"), dtype=f32).to_type(), "output")
    operation = VectorComputeOp(
        (source,),
        output,
        stage=1,
        function="rotary_embedding",
        configuration=(
            "head_dimension=D",
            f"maximum_timescale={maximum_timescale}",
            "sequence_dimension=L",
        ),
    )

    with pytest.raises(VerifyException, match="positive and finite"):
        operation.verify()


def test_buffer_type_rejects_duplicate_logical_dimensions() -> None:
    with pytest.raises(VerifyException, match="dimensions must be unique"):
        _spec((2, 2), ("K", "K")).to_type()


def test_buffer_type_rejects_one_mesh_axis_on_two_dimensions() -> None:
    with pytest.raises(VerifyException, match="cannot shard multiple buffer dimensions"):
        _spec((2, 4), ("B", "M"), sharding=("d", "d")).to_type()


def test_embedding_lookup_executes_in_declared_named_dimension_order() -> None:
    table = AllocOp(_spec((2, 4, 3), ("M", "V", "N")).to_type(), "table")
    indices = AllocOp(_spec((2,), ("B",), dtype=i32).to_type(), "indices")
    output = AllocOp(_spec((2, 2, 3), ("B", "M", "N")).to_type(), "output")
    operation = VectorComputeOp(
        (table, indices),
        output,
        stage=1,
        function="embedding_lookup",
        configuration=("vocabulary_dimension=V",),
    )
    table_value = jnp.arange(24, dtype=jnp.bfloat16).reshape(2, 4, 3)
    index_value = jnp.asarray([3, 1], dtype=jnp.int32)

    actual = _vector_compute(operation, (table_value, index_value), {})
    expected = np.stack(
        (
            np.asarray(table_value)[:, 3, :],
            np.asarray(table_value)[:, 1, :],
        )
    )

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_sharded_embedding_masks_indices_in_declared_output_order(monkeypatch) -> None:
    table = AllocOp(
        _spec((2, 2, 3), ("M", "V", "N"), sharding=("", "t", "")).to_type(),
        "table",
    )
    indices = AllocOp(_spec((2,), ("B",), dtype=i32).to_type(), "indices")
    output = AllocOp(_spec((2, 2, 3), ("B", "M", "N")).to_type(), "output")
    operation = VectorComputeOp(
        (table, indices),
        output,
        stage=1,
        function="embedding_lookup",
        configuration=("vocabulary_dimension=V",),
        pending_reduction_axes=("t",),
    )
    table_value = jnp.arange(12, dtype=jnp.bfloat16).reshape(2, 2, 3)
    index_value = jnp.asarray([0, 3], dtype=jnp.int32)
    monkeypatch.setattr(jax.lax, "axis_index", lambda _axis: jnp.int32(0))

    actual = _vector_compute(operation, (table_value, index_value), {"t": 2})
    expected = np.stack(
        (
            np.asarray(table_value)[:, 0, :],
            np.zeros((2, 3), dtype=np.asarray(table_value).dtype),
        )
    )

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_kernel_rejects_unwritten_output_argument() -> None:
    external = _spec((2, 4), ("B", "M"), memory=MemorySpace.HBM)
    builder = KernelBuilder(
        "unwritten_output",
        "tpu7x",
        (external,),
        argument_modes=("output",),
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 20,
    )

    with pytest.raises(VerifyException, match="does not write every output"):
        builder.module()


def test_kernel_rejects_read_from_output_before_write() -> None:
    external = _spec((2, 4), ("B", "M"), memory=MemorySpace.HBM)
    builder = KernelBuilder(
        "read_output",
        "tpu7x",
        (external,),
        argument_modes=("output",),
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 20,
    )
    local = builder.alloc(_spec((2, 4), ("B", "M")), "local")
    transfer = builder.dma_start(builder.inputs[0], local, builder.semaphore(), stage=0)
    builder.dma_wait(transfer, stage=1)

    with pytest.raises(VerifyException, match="reads a buffer before"):
        builder.module()


def test_kernel_rejects_write_to_input_only_argument() -> None:
    external = _spec((2, 4), ("B", "M"), memory=MemorySpace.HBM)
    builder = KernelBuilder(
        "write_input",
        "tpu7x",
        (external, external),
        argument_modes=("input", "input"),
        vmem_capacity_bytes=1 << 20,
        smem_capacity_bytes=1 << 20,
    )
    local = builder.alloc(_spec((2, 4), ("B", "M")), "local")
    inbound = builder.dma_start(builder.inputs[0], local, builder.semaphore(), stage=0)
    builder.dma_wait(inbound, stage=1)
    outbound = builder.dma_start(local, builder.inputs[1], builder.semaphore(), stage=2)
    builder.dma_wait(outbound, stage=3)

    with pytest.raises(VerifyException, match="cannot write an input-only argument"):
        builder.module()
