from __future__ import annotations

import hashlib
from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer

from tpu_cake.dialects.distributed_tensor import DistributedTensor
from tpu_cake.dialects.tpu_schedule import TPUSchedule


def canonical_text(module: ModuleOp) -> str:
    first = StringIO()
    Printer(stream=first, print_generic_format=True).print_op(module)
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(DistributedTensor)
    context.load_dialect(TPUSchedule)
    reparsed = Parser(context, first.getvalue()).parse_module()
    reparsed.verify()
    output = StringIO()
    Printer(stream=output, print_generic_format=True).print_op(reparsed)
    value = output.getvalue()
    return value + ("" if value.endswith("\n") else "\n")


def canonical_sha256(module: ModuleOp) -> str:
    return hashlib.sha256(canonical_text(module).encode()).hexdigest()
