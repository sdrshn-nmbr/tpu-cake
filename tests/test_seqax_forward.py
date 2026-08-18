from collections import Counter

import pytest
from xdsl.dialects.builtin import IntegerType, Signedness, bf16, f32
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.distributed_tensor import (
    AllGatherOp,
    CastOp,
    ProgramOp,
    RenameDimensionOp,
    RotaryEmbeddingOp,
)
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.dtensor_interpreter import interpret_distributed_program
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.lowering import UnsupportedLoweringError, lower_distributed_matmul
from tpu_cake.seqax_pallas_search import (
    SEQAX_PALLAS_CORRECTNESS_SEEDS,
    SEQAX_PALLAS_SEARCH_PARAMETERS,
)
from tpu_cake.workloads.seqax_forward import (
    REPLICATED_ATTENTION_WEIGHT_DATA,
    REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA,
    REPLICATED_EMBEDDING_WEIGHT_DATA,
    REPLICATED_FEED_FORWARD_WEIGHT_DATA,
    REPLICATED_WEIGHT_DATA,
    SEQAX_REVISION,
    SeqaxNormScalePlacement,
    SeqaxWeightDataPlacement,
    seqax_forward_schedule,
)
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


def test_complete_seqax_forward_algebra_verifies_and_hashes_stably() -> None:
    first = seqax_forward_schedule()
    second = seqax_forward_schedule()

    first.verify()
    assert SEQAX_REVISION == "b418a2d9059a1bfcff801d22b7088cc444257703"
    assert canonical_module_text(first) == canonical_module_text(second)
    assert schedule_sha256(first) == schedule_sha256(second)
    assert Counter(operation.name for operation in first.walk()) == Counter(
        {
            "builtin.module": 1,
            "dtensor.program": 1,
            "dtensor.embedding_lookup": 1,
            "dtensor.packed_causal_mask": 1,
            "dtensor.layer_scan": 1,
            "dtensor.scan_yield": 1,
            "dtensor.masked_softmax": 1,
            "dtensor.return": 1,
            "dtensor.all_gather": 14,
            "dtensor.einsum": 9,
            "dtensor.cast": 15,
            "dtensor.rename_dimension": 4,
            "dtensor.elementwise": 4,
            "dtensor.reduce_scatter": 3,
            "dtensor.rms_norm": 3,
            "dtensor.rotary_embedding": 2,
            "dtensor.slice": 2,
        }
    )


def test_seqax_forward_returns_sharded_vocabulary_logits() -> None:
    module = seqax_forward_schedule()
    program = next(operation for operation in module.walk() if isinstance(operation, ProgramOp))
    result = program.body.block.last_op.values[0].type

    assert result.logical_shape() == (("B", 8), ("L", 16), ("V", 64))
    assert result.sharding_axes() == (("d",), (), ("t",))
    assert result.pending_reductions() == {}


def test_seqax_forward_preserves_source_weight_types_and_casts() -> None:
    module = seqax_forward_schedule()
    program = next(operation for operation in module.walk() if isinstance(operation, ProgramOp))
    inputs = tuple(argument.type for argument in program.body.block.args)

    assert isinstance(inputs[0].element_type, IntegerType)
    assert inputs[0].element_type.signedness.data is Signedness.UNSIGNED
    assert all(value.element_type == f32 for value in inputs[2:])
    assert len(tuple(operation for operation in module.walk() if isinstance(operation, CastOp))) == 15


def test_norm_scale_replication_is_a_typed_communication_resource_choice() -> None:
    parameters = {
        "batch": 2,
        "sequence": 1,
        "model": 8,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 4,
        "head": 4,
        "layers": 1,
        "data_mesh": 2,
        "tensor_mesh": 4,
        "rope_max_timescale": 256,
    }
    sharded = seqax_forward_schedule(**parameters)
    replicated = seqax_forward_schedule(
        **parameters,
        norm_scale_placement=SeqaxNormScalePlacement.REPLICATED,
    )
    replicated_again = seqax_forward_schedule(
        **parameters,
        norm_scale_placement=SeqaxNormScalePlacement.REPLICATED,
    )
    sharded_program = next(
        operation for operation in sharded.walk() if isinstance(operation, ProgramOp)
    )
    replicated_program = next(
        operation for operation in replicated.walk() if isinstance(operation, ProgramOp)
    )

    assert len(tuple(operation for operation in sharded.walk() if isinstance(operation, AllGatherOp))) == 14
    assert len(tuple(operation for operation in replicated.walk() if isinstance(operation, AllGatherOp))) == 11
    assert sharded_program.body.block.args[3].type.sharding_axes() == ((), ("t", "d"))
    assert replicated_program.body.block.args[3].type.sharding_axes() == ((), ())
    assert sharded_program.body.block.args[2].type == replicated_program.body.block.args[2].type
    assert schedule_sha256(sharded) != schedule_sha256(replicated)
    assert schedule_sha256(replicated) == schedule_sha256(replicated_again)

    inputs = seqax_forward_inputs(seed=9173, **parameters)
    (sharded_output,) = interpret_distributed_program(sharded, inputs)
    (replicated_output,) = interpret_distributed_program(replicated, inputs)
    assert sharded_output.dtype == replicated_output.dtype
    assert sharded_output.shape == replicated_output.shape
    assert (sharded_output == replicated_output).all()

    with pytest.raises(TypeError, match="SeqaxNormScalePlacement"):
        seqax_forward_schedule(**parameters, norm_scale_placement="replicated")


def test_weight_data_replication_is_a_typed_communication_resource_choice() -> None:
    parameters = {
        "batch": 2,
        "sequence": 1,
        "model": 8,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 4,
        "head": 4,
        "layers": 1,
        "data_mesh": 2,
        "tensor_mesh": 4,
        "rope_max_timescale": 256,
    }
    sharded = seqax_forward_schedule(**parameters)
    replicated = seqax_forward_schedule(
        **parameters,
        weight_data_placement=REPLICATED_WEIGHT_DATA,
    )
    replicated_again = seqax_forward_schedule(
        **parameters,
        weight_data_placement=REPLICATED_WEIGHT_DATA,
    )
    sharded_program = next(
        operation for operation in sharded.walk() if isinstance(operation, ProgramOp)
    )
    replicated_program = next(
        operation for operation in replicated.walk() if isinstance(operation, ProgramOp)
    )

    assert len(tuple(operation for operation in sharded.walk() if isinstance(operation, AllGatherOp))) == 14
    assert len(tuple(operation for operation in replicated.walk() if isinstance(operation, AllGatherOp))) == 6
    assert {
        placement: sum(
            isinstance(operation, AllGatherOp)
            for operation in seqax_forward_schedule(
                **parameters,
                weight_data_placement=placement,
            ).walk()
        )
        for placement in (
            REPLICATED_EMBEDDING_WEIGHT_DATA,
            REPLICATED_ATTENTION_WEIGHT_DATA,
            REPLICATED_FEED_FORWARD_WEIGHT_DATA,
        )
    } == {
        REPLICATED_EMBEDDING_WEIGHT_DATA: 12,
        REPLICATED_ATTENTION_WEIGHT_DATA: 11,
        REPLICATED_FEED_FORWARD_WEIGHT_DATA: 11,
    }
    combined = seqax_forward_schedule(
        **parameters,
        weight_data_placement=REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA,
    )
    assert sum(isinstance(operation, AllGatherOp) for operation in combined.walk()) == 9
    assert sharded_program.body.block.args[2].type.sharding_axes() == (("t",), ("d",))
    assert replicated_program.body.block.args[2].type.sharding_axes() == (("t",), ())
    assert sharded_program.body.block.args[3].type == replicated_program.body.block.args[3].type
    assert schedule_sha256(sharded) != schedule_sha256(replicated)
    assert schedule_sha256(replicated) == schedule_sha256(replicated_again)

    inputs = seqax_forward_inputs(seed=9173, **parameters)
    (sharded_output,) = interpret_distributed_program(sharded, inputs)
    (replicated_output,) = interpret_distributed_program(replicated, inputs)
    assert sharded_output.dtype == replicated_output.dtype
    assert sharded_output.shape == replicated_output.shape
    assert (sharded_output == replicated_output).all()

    with pytest.raises(TypeError, match="SeqaxWeightDataPlacement"):
        seqax_forward_schedule(**parameters, weight_data_placement="replicated")
    for malformed in ("false", 1, None):
        with pytest.raises(TypeError, match="embedding must be a SeqaxDataAxisPlacement"):
            SeqaxWeightDataPlacement(embedding=malformed)


def test_weight_data_groups_preserve_five_seed_incumbent_semantics() -> None:
    placements = (
        REPLICATED_EMBEDDING_WEIGHT_DATA,
        REPLICATED_ATTENTION_WEIGHT_DATA,
        REPLICATED_FEED_FORWARD_WEIGHT_DATA,
        REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA,
        REPLICATED_WEIGHT_DATA,
    )
    baseline = seqax_forward_schedule(**SEQAX_PALLAS_SEARCH_PARAMETERS)
    candidates = tuple(
        seqax_forward_schedule(
            **SEQAX_PALLAS_SEARCH_PARAMETERS,
            weight_data_placement=placement,
        )
        for placement in placements
    )
    for seed in SEQAX_PALLAS_CORRECTNESS_SEEDS:
        inputs = seqax_forward_inputs(seed=seed, **SEQAX_PALLAS_SEARCH_PARAMETERS)
        (expected,) = interpret_distributed_program(baseline, inputs)
        for candidate in candidates:
            (actual,) = interpret_distributed_program(candidate, inputs)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert (actual == expected).all()


def test_seqax_forward_uses_configured_rope_timescale_and_source_locations() -> None:
    module = seqax_forward_schedule(rope_max_timescale=256)
    rotations = tuple(
        operation for operation in module.walk() if isinstance(operation, RotaryEmbeddingOp)
    )
    renames = tuple(
        operation for operation in module.walk() if isinstance(operation, RenameDimensionOp)
    )

    assert {operation.maximum_timescale.data for operation in rotations} == {256}
    assert all(operation.result.type.element_type == f32 for operation in rotations)
    assert all(str(operation.location) != "loc(unknown)" for operation in renames)


def test_dimension_rename_rejects_a_non_fresh_destination() -> None:
    value = tensor(bf16, (("B", 8), ("L", 16)))
    builder = DistributedProgramBuilder("bad_rename", {}, (value,))
    result = builder.rename_dimension(
        builder.inputs[0],
        value,
        source_dimension="L",
        destination_dimension="B",
    )

    with pytest.raises(VerifyException, match="fresh destination"):
        builder.module(result)


def test_seqax_forward_does_not_enter_the_narrow_matmul_lowerer() -> None:
    with pytest.raises(UnsupportedLoweringError, match="only einsum"):
        lower_distributed_matmul(seqax_forward_schedule())
