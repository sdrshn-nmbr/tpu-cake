from __future__ import annotations

import pytest
from xdsl.utils.exceptions import ParseError

from tpu_cake.canonical import (
    canonical_text,
    parse_distributed_module,
    parse_physical_module,
    parse_tpu_cake_module,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule


def test_dialect_specific_parsers_preserve_their_acceptance_boundaries() -> None:
    distributed_text = canonical_text(seqax_forward_schedule())
    physical_text = canonical_text(lower_seqax_forward_to_physical(seqax_forward_schedule()).module)

    assert canonical_text(parse_distributed_module(distributed_text)) == distributed_text
    assert canonical_text(parse_physical_module(physical_text)) == physical_text
    assert canonical_text(parse_tpu_cake_module(distributed_text)) == distributed_text
    assert canonical_text(parse_tpu_cake_module(physical_text)) == physical_text

    with pytest.raises(ParseError):
        parse_distributed_module(physical_text)
    with pytest.raises(ParseError):
        parse_physical_module(distributed_text)
