from __future__ import annotations

import pytest

from tpu_cake.seqax_silu_fusion_compiler import (
    analyze_seqax_silu_fusion_compiler_hlo,
)
from tpu_cake.workloads.seqax_forward import SeqaxFeedForwardFusion

_SCHEDULE = "1" * 64


def _call(
    name: str,
    kernel: str,
    operands: tuple[str, ...],
    *,
    region_index: int,
    vector_region_index: int | None = None,
    root: bool = False,
) -> str:
    metadata = [
        f'"region_index":{region_index}',
        f'"schedule_sha256":"{_SCHEDULE}"',
    ]
    if vector_region_index is not None:
        metadata.extend(
            (
                '"implementation":"pallas_full_local"',
                f'"vector_region_index":{vector_region_index}',
            )
        )
    output = "f32[128,1,32]" if region_index == 7 else "bf16[128,1,1024]"
    prefix = "ROOT " if root else ""
    operand_text = ", ".join(f"%{operand}" for operand in operands)
    return (
        f"  {prefix}%{name} = {output} custom-call({operand_text}), "
        'custom_call_target="tpu_custom_call", frontend_attributes={kernel_metadata={\n'
        + "\n".join(metadata)
        + f'\n}}, metadata={{op_name="jit(physical_call)/shard_map/{kernel}/pallas_call"}}'
    )


def _compiler_hlo(
    candidate: SeqaxFeedForwardFusion,
    *,
    gate_name: str = "gate_projection",
    up_name: str = "up_projection",
    vector_name: str = "vector_boundary",
    extra_entry_instruction: str = "",
    down_root: bool = True,
) -> str:
    lines = [
        "HloModule fusion",
        "",
        "ENTRY %main (%input: bf16[128,1,32]) -> f32[128,1,32] {",
        _call(gate_name, "seqax_named_einsum", ("input", "gate_weight"), region_index=5),
        _call(up_name, "seqax_named_einsum", ("input", "up_weight"), region_index=6),
    ]
    if candidate is SeqaxFeedForwardFusion.SEPARATE:
        lines.extend(
            (
                _call(
                    "activation",
                    "seqax_strict_bf16_silu",
                    (gate_name,),
                    region_index=-1,
                    vector_region_index=0,
                ),
                _call(
                    vector_name,
                    "seqax_strict_bf16_multiply",
                    ("activation", up_name),
                    region_index=-1,
                    vector_region_index=1,
                ),
            )
        )
    else:
        lines.append(
            _call(
                vector_name,
                "seqax_strict_bf16_silu_multiply",
                (gate_name, up_name),
                region_index=-1,
                vector_region_index=0,
            )
        )
    if extra_entry_instruction:
        lines.append(extra_entry_instruction)
    lines.append(
        _call(
            "down_projection",
            "seqax_named_einsum",
            (vector_name, "down_weight"),
            region_index=7,
            root=down_root,
        )
    )
    if not down_root:
        lines.append(f"  ROOT %result = bf16[128,1,1024] copy(%{vector_name})")
    lines.append("}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "candidate,expected_kernels",
    (
        (
            SeqaxFeedForwardFusion.SEPARATE,
            ("seqax_strict_bf16_silu", "seqax_strict_bf16_multiply"),
        ),
        (
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            ("seqax_strict_bf16_silu_multiply",),
        ),
    ),
)
def test_compiler_analysis_accepts_exact_live_boundary(candidate, expected_kernels) -> None:
    analysis = analyze_seqax_silu_fusion_compiler_hlo(
        _compiler_hlo(candidate),
        candidate,
        expected_schedule_sha256=_SCHEDULE,
    )

    assert tuple(call.kernel for call in analysis.calls) == expected_kernels
    assert analysis.gate_and_up_projection_lineages_are_distinct
    assert analysis.vector_output_feeds_one_down_projection


def test_compiler_analysis_is_independent_of_generated_instruction_names() -> None:
    original = analyze_seqax_silu_fusion_compiler_hlo(
        _compiler_hlo(SeqaxFeedForwardFusion.SILU_MULTIPLY),
        SeqaxFeedForwardFusion.SILU_MULTIPLY,
        expected_schedule_sha256=_SCHEDULE,
    )
    renamed = analyze_seqax_silu_fusion_compiler_hlo(
        _compiler_hlo(
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            gate_name="compiler_generated_41",
            up_name="compiler_generated_87",
            vector_name="compiler_generated_103",
        ),
        SeqaxFeedForwardFusion.SILU_MULTIPLY,
        expected_schedule_sha256=_SCHEDULE,
    )

    assert original.semantic_id == renamed.semantic_id


def test_compiler_analysis_rejects_dead_strict_call_in_entry() -> None:
    decoy = _call(
        "dead_decoy",
        "seqax_strict_bf16_silu",
        ("gate_projection",),
        region_index=-1,
        vector_region_index=1,
    )

    with pytest.raises(ValueError, match="DEAD_STRICT_VECTOR_CALL"):
        analyze_seqax_silu_fusion_compiler_hlo(
            _compiler_hlo(
                SeqaxFeedForwardFusion.SILU_MULTIPLY,
                extra_entry_instruction=decoy,
            ),
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            expected_schedule_sha256=_SCHEDULE,
        )


def test_compiler_analysis_rejects_dead_strict_call_computation() -> None:
    decoy = "\n".join(
        (
            "%dead () -> bf16[128,1,1024] {",
            _call(
                "dead_vector",
                "seqax_strict_bf16_silu",
                ("dead_input",),
                region_index=-1,
                vector_region_index=0,
                root=True,
            ),
            "}",
            "",
        )
    )

    with pytest.raises(ValueError, match="DEAD_STRICT_VECTOR_CALL"):
        analyze_seqax_silu_fusion_compiler_hlo(
            _compiler_hlo(SeqaxFeedForwardFusion.SILU_MULTIPLY).replace(
                "ENTRY %main",
                decoy + "ENTRY %main",
            ),
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            expected_schedule_sha256=_SCHEDULE,
        )


def test_compiler_analysis_rejects_baseline_without_silu_multiply_edge() -> None:
    hlo = _compiler_hlo(SeqaxFeedForwardFusion.SEPARATE).replace(
        "custom-call(%activation, %up_projection)",
        "custom-call(%gate_projection, %up_projection)",
    )
    hlo = hlo.replace("ROOT %down_projection", "%down_projection").replace(
        "\n}\n",
        "\n  ROOT %result = (f32[128,1,32], bf16[128,1,1024]) tuple(%down_projection, %activation)\n}\n",
    )

    with pytest.raises(ValueError, match="SILU_MULTIPLY_EDGE_MISMATCH"):
        analyze_seqax_silu_fusion_compiler_hlo(
            hlo,
            SeqaxFeedForwardFusion.SEPARATE,
            expected_schedule_sha256=_SCHEDULE,
        )


@pytest.mark.parametrize(
    "operands",
    (
        "%up_projection, %gate_projection",
        "%gate_projection, %gate_projection",
        "%up_projection, %up_projection",
    ),
)
def test_compiler_analysis_rejects_swapped_or_duplicate_fused_inputs(operands: str) -> None:
    hlo = _compiler_hlo(SeqaxFeedForwardFusion.SILU_MULTIPLY).replace(
        "custom-call(%gate_projection, %up_projection)",
        f"custom-call({operands})",
    )

    with pytest.raises(ValueError, match="COMPILER_BOUNDARY_MISMATCH"):
        analyze_seqax_silu_fusion_compiler_hlo(
            hlo,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            expected_schedule_sha256=_SCHEDULE,
        )


def test_compiler_analysis_rejects_vector_output_that_bypasses_down_projection() -> None:
    with pytest.raises(ValueError, match="COMPILER_BOUNDARY_MISMATCH"):
        analyze_seqax_silu_fusion_compiler_hlo(
            _compiler_hlo(SeqaxFeedForwardFusion.SILU_MULTIPLY, down_root=False),
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            expected_schedule_sha256=_SCHEDULE,
        )


def test_compiler_analysis_rejects_wrong_schedule_or_implementation_metadata() -> None:
    hlo = _compiler_hlo(SeqaxFeedForwardFusion.SILU_MULTIPLY)
    with pytest.raises(ValueError, match="VECTOR_METADATA_MISMATCH"):
        analyze_seqax_silu_fusion_compiler_hlo(
            hlo,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            expected_schedule_sha256="2" * 64,
        )
    with pytest.raises(ValueError, match="VECTOR_METADATA_MISMATCH"):
        analyze_seqax_silu_fusion_compiler_hlo(
            hlo.replace('"implementation":"pallas_full_local"', '"implementation":"xla"'),
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            expected_schedule_sha256=_SCHEDULE,
        )


def test_compiler_analysis_rejects_unknown_strict_kernel() -> None:
    hlo = _compiler_hlo(SeqaxFeedForwardFusion.SILU_MULTIPLY).replace(
        "seqax_strict_bf16_silu_multiply",
        "seqax_strict_bf16_relu_multiply",
    )

    with pytest.raises(ValueError, match="UNKNOWN_STRICT_KERNEL"):
        analyze_seqax_silu_fusion_compiler_hlo(
            hlo,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
            expected_schedule_sha256=_SCHEDULE,
        )
