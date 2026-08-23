from __future__ import annotations

import re
from collections import deque

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tpu_cake.identity import json_sha256
from tpu_cake.workloads.seqax_forward import SeqaxFeedForwardFusion

_VECTOR_SHAPE = "bf16[128,1,1024]"
_STRICT_KERNELS = {
    "seqax_strict_bf16_silu",
    "seqax_strict_bf16_multiply",
    "seqax_strict_bf16_silu_multiply",
}
_COMPUTATION_REFERENCES = (
    "to_apply",
    "calls",
    "condition",
    "body",
    "fused_computation",
)


class SeqaxSiluFusionCompilerCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ordinal: int = Field(ge=0, le=1)
    kernel: str
    output_shape: str
    operand_count: int = Field(ge=1, le=2)
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector_region_index: int = Field(ge=0, le=1)
    implementation: str
    instruction_name: str = Field(min_length=1)
    operand_names: tuple[str, ...] = Field(min_length=1, max_length=2)


class SeqaxSiluFusionCompilerAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate: SeqaxFeedForwardFusion
    strict_vector_call_count: int = Field(ge=1, le=2)
    calls: tuple[SeqaxSiluFusionCompilerCall, ...] = Field(min_length=1, max_length=2)
    all_strict_vector_calls_are_live: bool
    gate_and_up_projection_lineages_are_distinct: bool
    silu_output_feeds_multiply: bool
    vector_output_feeds_one_down_projection: bool

    @model_validator(mode="after")
    def boundary_is_exact(self) -> SeqaxSiluFusionCompilerAnalysis:
        kernels = tuple(call.kernel for call in self.calls)
        expected = (
            ("seqax_strict_bf16_silu", "seqax_strict_bf16_multiply")
            if self.candidate is SeqaxFeedForwardFusion.SEPARATE
            else ("seqax_strict_bf16_silu_multiply",)
        )
        if (
            kernels != expected
            or self.strict_vector_call_count != len(expected)
            or tuple(call.ordinal for call in self.calls) != tuple(range(len(expected)))
            or any(call.output_shape != _VECTOR_SHAPE for call in self.calls)
            or not self.all_strict_vector_calls_are_live
            or not self.gate_and_up_projection_lineages_are_distinct
            or not self.vector_output_feeds_one_down_projection
        ):
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_BOUNDARY_MISMATCH")
        if (self.candidate is SeqaxFeedForwardFusion.SEPARATE) != self.silu_output_feeds_multiply:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_DEPENDENCY_MISMATCH")
        return self

    @computed_field
    @property
    def semantic_id(self) -> str:
        return json_sha256(
            {
                "candidate": self.candidate,
                "kernels": [
                    {
                        "kernel": call.kernel,
                        "output_shape": call.output_shape,
                        "operand_count": call.operand_count,
                        "schedule_sha256": call.schedule_sha256,
                        "vector_region_index": call.vector_region_index,
                        "implementation": call.implementation,
                    }
                    for call in self.calls
                ],
                "all_strict_vector_calls_are_live": self.all_strict_vector_calls_are_live,
                "gate_and_up_projection_lineages_are_distinct": (
                    self.gate_and_up_projection_lineages_are_distinct
                ),
                "silu_output_feeds_multiply": self.silu_output_feeds_multiply,
                "vector_output_feeds_one_down_projection": (
                    self.vector_output_feeds_one_down_projection
                ),
            }
        )


class _Instruction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    body: str
    root: bool

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(re.findall(r"%([A-Za-z0-9_.$-]+)", self.body))


class _Computation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    entry: bool
    instructions: tuple[_Instruction, ...]

    @property
    def by_name(self) -> dict[str, _Instruction]:
        return {instruction.name: instruction for instruction in self.instructions}

    @property
    def root(self) -> _Instruction:
        roots = tuple(instruction for instruction in self.instructions if instruction.root)
        if len(roots) != 1:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_ROOT_MISMATCH")
        return roots[0]


def _computations(compiler_hlo: str) -> dict[str, _Computation]:
    lines = compiler_hlo.splitlines()
    header_pattern = re.compile(
        r"\s*(?P<entry>ENTRY\s+)?%?(?P<name>[A-Za-z0-9_.$-]+)"
        r"(?:\s+\([^\n]*\)\s*->\s*[^\n{]+)?\s*\{\s*"
    )
    headers = tuple(
        (index, match.group("name"), match.group("entry") is not None)
        for index, line in enumerate(lines)
        if (match := header_pattern.fullmatch(line)) is not None
    )
    if len(tuple(value for value in headers if value[2])) != 1:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_ENTRY_MISMATCH")
    computations = {}
    instruction_pattern = re.compile(r"\s*(?P<root>ROOT\s+)?%(?P<name>[^\s=]+)\s*=\s*(?P<body>.+)$")
    for header_index, (start, name, entry) in enumerate(headers):
        limit = headers[header_index + 1][0] if header_index + 1 < len(headers) else len(lines)
        endings = tuple(
            index
            for index in range(start + 1, limit)
            if re.fullmatch(r'\s*}(?:,\s*execution_thread="[^"]+")?\s*', lines[index])
        )
        if not endings:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_COMPUTATION_INVALID")
        end = endings[-1]
        starts = tuple(
            (index, match)
            for index in range(start + 1, end)
            if (match := instruction_pattern.match(lines[index])) is not None
        )
        instructions = []
        for instruction_index, (line_index, match) in enumerate(starts):
            next_line = (
                starts[instruction_index + 1][0] if instruction_index + 1 < len(starts) else end
            )
            body = "\n".join([match.group("body"), *lines[line_index + 1 : next_line]])
            instructions.append(
                _Instruction(
                    name=match.group("name"),
                    body=body,
                    root=match.group("root") is not None,
                )
            )
        computations[name] = _Computation(
            name=name,
            entry=entry,
            instructions=tuple(instructions),
        )
    return computations


def _called_computations(body: str) -> tuple[str, ...]:
    called = []
    for attribute in _COMPUTATION_REFERENCES:
        called.extend(re.findall(rf"\b{attribute}=%?([A-Za-z0-9_.$-]+)", body))
    for attribute in ("branch_computations", "called_computations"):
        for values in re.findall(rf"\b{attribute}=\{{([^}}]*)\}}", body):
            called.extend(re.findall(r"%?([A-Za-z0-9_.$-]+)", values))
    return tuple(called)


def _reachable_and_live(
    compiler_hlo: str,
) -> tuple[dict[str, _Computation], set[tuple[str, str]]]:
    computations = _computations(compiler_hlo)
    entry = next(computation for computation in computations.values() if computation.entry)
    live: set[tuple[str, str]] = set()
    pending = deque([(entry.name, entry.root.name)])
    while pending:
        computation_name, instruction_name = pending.popleft()
        key = (computation_name, instruction_name)
        if key in live:
            continue
        computation = computations.get(computation_name)
        if computation is None:
            raise ValueError(
                f"SEQAX_SILU_FUSION_COMPILER_REFERENCE_MISSING computation={computation_name}"
            )
        instruction = computation.by_name.get(instruction_name)
        if instruction is None:
            continue
        live.add(key)
        pending.extend(
            (computation_name, reference)
            for reference in instruction.references
            if reference in computation.by_name
        )
        for called in _called_computations(instruction.body):
            called_computation = computations.get(called)
            if called_computation is None:
                raise ValueError(
                    f"SEQAX_SILU_FUSION_COMPILER_REFERENCE_MISSING computation={called}"
                )
            pending.append((called, called_computation.root.name))
    return computations, live


def _kernel_name(body: str) -> str | None:
    if "custom-call(" not in body or 'custom_call_target="tpu_custom_call"' not in body:
        return None
    matches = tuple(
        dict.fromkeys(
            re.findall(
                r'op_name="[^"]*/(seqax_[A-Za-z0-9_]+?)/pallas_call"',
                body,
            )
        )
    )
    if len(matches) > 1:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_KERNEL_METADATA_AMBIGUOUS")
    return matches[0] if matches else None


def _output_shape(body: str) -> str:
    match = re.match(r"\s*([^\s]+)\s+custom-call\(", body)
    if match is None:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_CUSTOM_CALL_INVALID")
    return match.group(1).split("{")[0]


def _custom_call_operands(body: str) -> tuple[str, ...]:
    match = re.search(r"\bcustom-call\((?P<operands>[^)]*)\)", body)
    if match is None:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_CUSTOM_CALL_INVALID")
    return tuple(re.findall(r"%([A-Za-z0-9_.$-]+)", match.group("operands")))


def _kernel_metadata(body: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}":(?:")?([^",\n}}]+)', body)
    if match is None:
        raise ValueError(f"SEQAX_SILU_FUSION_COMPILER_METADATA_MISSING key={key}")
    return match.group(1)


def _depends_on(
    computation: _Computation,
    instruction_name: str,
    ancestor_name: str,
) -> bool:
    pending = [instruction_name]
    visited = set()
    while pending:
        name = pending.pop()
        if name == ancestor_name:
            return True
        if name in visited:
            continue
        visited.add(name)
        instruction = computation.by_name.get(name)
        if instruction is not None:
            pending.extend(instruction.references)
    return False


def _nearest_einsum_ancestors(
    computation: _Computation,
    instruction_name: str,
) -> frozenset[str]:
    found = set()
    visited = set()
    pending = deque([instruction_name])
    while pending:
        name = pending.popleft()
        if name in visited:
            continue
        visited.add(name)
        instruction = computation.by_name.get(name)
        if instruction is None:
            continue
        if _kernel_name(instruction.body) == "seqax_named_einsum":
            found.add(name)
            continue
        pending.extend(instruction.references)
    return frozenset(found)


def analyze_seqax_silu_fusion_compiler_hlo(
    compiler_hlo: str,
    candidate: SeqaxFeedForwardFusion,
    *,
    expected_schedule_sha256: str,
) -> SeqaxSiluFusionCompilerAnalysis:
    computations, live = _reachable_and_live(compiler_hlo)
    all_strict = []
    live_strict = []
    unknown_strict = []
    for computation_name, computation in computations.items():
        for instruction in computation.instructions:
            kernel = _kernel_name(instruction.body)
            if kernel is None:
                continue
            if kernel.startswith("seqax_strict_bf16_") and kernel not in _STRICT_KERNELS:
                unknown_strict.append(kernel)
            if kernel in _STRICT_KERNELS:
                value = (computation_name, instruction, kernel)
                all_strict.append(value)
                if (computation_name, instruction.name) in live:
                    live_strict.append(value)
    if unknown_strict:
        raise ValueError(
            f"SEQAX_SILU_FUSION_COMPILER_UNKNOWN_STRICT_KERNEL kernels={unknown_strict}"
        )
    if len(all_strict) != len(live_strict):
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_DEAD_STRICT_VECTOR_CALL")
    expected_kernels = (
        ("seqax_strict_bf16_silu", "seqax_strict_bf16_multiply")
        if candidate is SeqaxFeedForwardFusion.SEPARATE
        else ("seqax_strict_bf16_silu_multiply",)
    )
    by_kernel = {
        kernel: (computation_name, instruction)
        for computation_name, instruction, kernel in live_strict
    }
    if set(by_kernel) != set(expected_kernels) or len(live_strict) != len(expected_kernels):
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_STRICT_VECTOR_CALL_MISMATCH")
    ordered = tuple((*by_kernel[kernel], kernel) for kernel in expected_kernels)
    if len({computation_name for computation_name, _instruction, _kernel in ordered}) != 1:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_BOUNDARY_COMPUTATION_MISMATCH")
    computation = computations[ordered[0][0]]
    calls = tuple(
        SeqaxSiluFusionCompilerCall(
            ordinal=ordinal,
            kernel=kernel,
            output_shape=_output_shape(instruction.body),
            operand_count=len(_custom_call_operands(instruction.body)),
            schedule_sha256=_kernel_metadata(instruction.body, "schedule_sha256"),
            vector_region_index=int(_kernel_metadata(instruction.body, "vector_region_index")),
            implementation=_kernel_metadata(instruction.body, "implementation"),
            instruction_name=instruction.name,
            operand_names=_custom_call_operands(instruction.body),
        )
        for ordinal, (_computation_name, instruction, kernel) in enumerate(ordered)
    )
    if any(
        call.schedule_sha256 != expected_schedule_sha256
        or call.vector_region_index != call.ordinal
        or call.implementation != "pallas_full_local"
        for call in calls
    ):
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_VECTOR_METADATA_MISMATCH")
    if candidate is SeqaxFeedForwardFusion.SEPARATE:
        silu, multiply = calls
        silu_feeds_multiply = _depends_on(
            computation,
            multiply.instruction_name,
            silu.instruction_name,
        )
        gate_ancestors = _nearest_einsum_ancestors(computation, silu.operand_names[0])
        silu_operands = tuple(
            operand
            for operand in multiply.operand_names
            if _depends_on(computation, operand, silu.instruction_name)
        )
        if len(silu_operands) != 1:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_SILU_MULTIPLY_EDGE_MISMATCH")
        up_operand = next(
            (operand for operand in multiply.operand_names if operand not in silu_operands),
            None,
        )
        up_ancestors = (
            frozenset()
            if up_operand is None
            else _nearest_einsum_ancestors(computation, up_operand)
        )
        output_name = multiply.instruction_name
    else:
        (fused,) = calls
        silu_feeds_multiply = False
        if len(fused.operand_names) != 2:
            raise ValueError("SEQAX_SILU_FUSION_COMPILER_FUSED_OPERAND_MISMATCH")
        gate_ancestors = _nearest_einsum_ancestors(computation, fused.operand_names[0])
        up_ancestors = _nearest_einsum_ancestors(computation, fused.operand_names[1])
        output_name = fused.instruction_name
    gate_regions = {
        int(_kernel_metadata(computation.by_name[name].body, "region_index"))
        for name in gate_ancestors
    }
    up_regions = {
        int(_kernel_metadata(computation.by_name[name].body, "region_index"))
        for name in up_ancestors
    }
    distinct_lineages = gate_regions == {5} and up_regions == {6}
    down_projections = tuple(
        instruction
        for instruction in computation.instructions
        if (computation.name, instruction.name) in live
        and _kernel_name(instruction.body) == "seqax_named_einsum"
        and _depends_on(computation, instruction.name, output_name)
        and int(_kernel_metadata(instruction.body, "region_index")) == 7
    )
    return SeqaxSiluFusionCompilerAnalysis(
        candidate=candidate,
        strict_vector_call_count=len(calls),
        calls=calls,
        all_strict_vector_calls_are_live=True,
        gate_and_up_projection_lineages_are_distinct=distinct_lineages,
        silu_output_feeds_multiply=silu_feeds_multiply,
        vector_output_feeds_one_down_projection=len(down_projections) == 1,
    )
