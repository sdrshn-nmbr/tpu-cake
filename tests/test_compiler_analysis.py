from __future__ import annotations

from types import SimpleNamespace

import pytest

from tpu_cake.compiler_analysis import (
    CompilerExecutableAnalysis,
    capture_compiler_analysis,
)


class _Executable:
    def __init__(self, cost: object, memory: object) -> None:
        self._cost = cost
        self._memory = memory

    def cost_analysis(self) -> object:
        return self._cost

    def memory_analysis(self) -> object:
        return self._memory


def _memory(**updates: object) -> SimpleNamespace:
    values = {
        "generated_code_size_in_bytes": 100,
        "argument_size_in_bytes": 200,
        "output_size_in_bytes": 80,
        "alias_size_in_bytes": 16,
        "temp_size_in_bytes": 40,
        "host_generated_code_size_in_bytes": 0,
        "host_argument_size_in_bytes": 0,
        "host_output_size_in_bytes": 0,
        "host_alias_size_in_bytes": 0,
        "host_temp_size_in_bytes": 0,
        "peak_memory_in_bytes": 320,
        "serialized_buffer_assignment_proto": b"buffer-assignment",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_compiler_analysis_is_canonical_and_replayable() -> None:
    analysis = capture_compiler_analysis(
        _Executable(
            {"flops": 1024.0, "bytes accessed": 512.0},
            _memory(),
        ),
        stablehlo="module @main {}",
        compiler_hlo="HloModule main",
    )

    assert tuple(metric.name for metric in analysis.cost_metrics) == (
        "bytes accessed",
        "flops",
    )
    assert analysis.memory.buffer_assignment_size_bytes == len(b"buffer-assignment")
    assert CompilerExecutableAnalysis.model_validate_json(analysis.model_dump_json()) == analysis


@pytest.mark.parametrize(
    ("cost", "memory", "message"),
    (
        (None, _memory(), "COST_UNAVAILABLE"),
        ({}, _memory(), "COST_EMPTY"),
        ({"flops": float("nan")}, _memory(), "COST_VALUE_INVALID"),
        ({"flops": -1.0}, _memory(), "COST_VALUE_INVALID"),
        ({"flops": True}, _memory(), "COST_VALUE_INVALID"),
        ({"flops": 1.0}, None, "MEMORY_UNAVAILABLE"),
        ({"flops": 1.0}, _memory(peak_memory_in_bytes=-1), "MEMORY_FIELD_INVALID"),
        (
            {"flops": 1.0},
            _memory(serialized_buffer_assignment_proto=b""),
            "BUFFER_ASSIGNMENT_INVALID",
        ),
    ),
)
def test_compiler_analysis_fails_closed(
    cost: object,
    memory: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        capture_compiler_analysis(
            _Executable(cost, memory),
            stablehlo="module @main {}",
            compiler_hlo="HloModule main",
        )


def test_compiler_analysis_rejects_mutated_metric_order() -> None:
    analysis = capture_compiler_analysis(
        _Executable({"flops": 2.0, "bytes accessed": 1.0}, _memory()),
        stablehlo="module @main {}",
        compiler_hlo="HloModule main",
    )
    payload = analysis.model_dump(mode="json")
    payload["cost_metrics"] = list(reversed(payload["cost_metrics"]))

    with pytest.raises(ValueError, match="metrics must be sorted"):
        CompilerExecutableAnalysis.model_validate(payload)
