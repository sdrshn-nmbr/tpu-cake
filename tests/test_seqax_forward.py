from collections import Counter

import pytest
from xdsl.dialects.builtin import bf16
from xdsl.utils.exceptions import VerifyException

from tpu_cake.dialects.distributed_tensor import ProgramOp
from tpu_cake.distributed_frontend import DistributedProgramBuilder, tensor
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.lowering import UnsupportedLoweringError, lower_distributed_matmul
from tpu_cake.workloads.seqax_forward import SEQAX_REVISION, seqax_forward_schedule


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
            "dtensor.cast": 7,
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
