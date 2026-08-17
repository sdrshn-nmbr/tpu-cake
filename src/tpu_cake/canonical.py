from __future__ import annotations

import hashlib
from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer

from tpu_cake.dialects.distributed_tensor import (
    DistributedTensor,
    ElementwiseOp,
    ProgramOp,
)
from tpu_cake.dialects.tpu_schedule import AllocOp, KernelOp, TPUSchedule, ViewOp


def _normalize_semantics(module: ModuleOp) -> None:
    value_order = {}
    next_value = 0
    for operation in module.walk():
        for region in operation.regions:
            for block in region.blocks:
                for argument in block.args:
                    value_order[argument] = next_value
                    next_value += 1
        for result in operation.results:
            value_order[result] = next_value
            next_value += 1

    alias_names: dict[str, str] = {}
    allocation_index = 0
    for operation in module.walk():
        if isinstance(operation, ElementwiseOp) and operation.function.data in {
            "add",
            "multiply",
        }:
            operation.operands = tuple(
                sorted(operation.operands, key=value_order.__getitem__)
            )
        if isinstance(operation, ViewOp):
            declared = operation.alias_group.data
            normalized = alias_names.setdefault(declared, f"alias{len(alias_names)}")
            operation.properties["alias_group"] = type(operation.alias_group)(normalized)
        if isinstance(operation, AllocOp):
            operation.properties["role"] = type(operation.role)(f"alloc{allocation_index}")
            allocation_index += 1
        if isinstance(operation, KernelOp):
            operation.properties["sym_name"] = type(operation.sym_name)("kernel")
        if isinstance(operation, ProgramOp):
            operation.properties["sym_name"] = type(operation.sym_name)("program")
        operation.properties = dict(sorted(operation.properties.items()))
        operation.attributes = dict(sorted(operation.attributes.items()))


def canonical_text(module: ModuleOp) -> str:
    first = StringIO()
    Printer(stream=first, print_generic_format=True).print_op(module)
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(DistributedTensor)
    context.load_dialect(TPUSchedule)
    reparsed = Parser(context, first.getvalue()).parse_module()
    reparsed.verify()
    _normalize_semantics(reparsed)
    reparsed.verify()
    output = StringIO()
    Printer(stream=output, print_generic_format=True).print_op(reparsed)
    value = output.getvalue()
    return value + ("" if value.endswith("\n") else "\n")


def canonical_sha256(module: ModuleOp) -> str:
    return hashlib.sha256(canonical_text(module).encode()).hexdigest()
