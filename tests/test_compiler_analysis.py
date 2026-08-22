from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tpu_cake.compiler_analysis import (
    CompilerCollectiveStrategyPoint,
    CompilerCollectiveStrategySurface,
    CompilerExecutableAnalysis,
    analyze_compiler_collectives,
    capture_compiler_analysis,
    validate_compiler_analysis,
)


class _Executable:
    def __init__(
        self,
        cost: object,
        memory: object,
        compiler_hlo: str = "HloModule main",
    ) -> None:
        self._cost = cost
        self._memory = memory
        self._compiler_hlo = compiler_hlo

    def cost_analysis(self) -> object:
        return self._cost

    def memory_analysis(self) -> object:
        return self._memory

    def as_text(self) -> str:
        return self._compiler_hlo


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
    assert analysis.memory.buffer_assignment_available is True
    assert analysis.cost_metrics[0].available is True
    assert CompilerExecutableAnalysis.model_validate_json(analysis.model_dump_json()) == analysis


def test_compiler_analysis_preserves_backend_unavailable_states() -> None:
    analysis = capture_compiler_analysis(
        _Executable(
            {"flops": 1024.0, "optimal_seconds": -3.99999737739563},
            _memory(serialized_buffer_assignment_proto=b""),
        ),
        stablehlo="module @main {}",
        compiler_hlo="HloModule main",
    )

    optimal_seconds = next(
        metric for metric in analysis.cost_metrics if metric.name == "optimal_seconds"
    )
    assert optimal_seconds.available is False
    assert optimal_seconds.value is None
    assert optimal_seconds.raw_value == -3.99999737739563
    assert analysis.memory.buffer_assignment_available is False
    assert analysis.memory.buffer_assignment_size_bytes == 0
    assert analysis.memory.buffer_assignment_sha256 is None


def test_compiler_analysis_rejects_hlo_not_bound_to_executable() -> None:
    with pytest.raises(ValueError, match="COMPILER_ANALYSIS_EXECUTABLE_HLO_MISMATCH"):
        capture_compiler_analysis(
            _Executable({"flops": 1.0}, _memory(), "HloModule executed"),
            stablehlo="module @main {}",
            compiler_hlo="HloModule pre_optimization",
        )


def test_compiler_collectives_report_semantic_and_emitted_strategy() -> None:
    analysis = analyze_compiler_collectives(
        stablehlo="""
          %0 = "stablehlo.reduce_scatter"(%arg0) {scatter_dimension = 1 : i64}
          %1 = "stablehlo.all_gather"(%0) {all_gather_dim = 1 : i64}
        """,
        compiler_hlo="""
          %reduce-scatter.1 = bf16[4,2] reduce-scatter(%arg0), backend_config={"device_type":"DEVICE_TYPE_SPARSECORE","reduce_scatter_offload_config":{}}
          ROOT %all-gather.1 = bf16[4,8] all-gather(%reduce-scatter.1), backend_config={"device_type":"DEVICE_TYPE_SPARSECORE","all_gather_offload_config":{}}
        """,
    )

    assert analysis.stablehlo_reduce_scatter_count == 1
    assert analysis.stablehlo_all_gather_count == 1
    assert analysis.compiler_reduce_scatter_count == 1
    assert analysis.compiler_all_reduce_count == 0
    assert analysis.compiler_all_gather_count == 1
    assert analysis.sparse_core_reduce_scatter_count == 1
    assert analysis.sparse_core_all_gather_count == 1


def test_compiler_collectives_do_not_count_metadata_names_as_operations() -> None:
    analysis = analyze_compiler_collectives(
        stablehlo="%0 = stablehlo.reduce_scatter %arg0",
        compiler_hlo="""
          %all-reduce.1 = bf16[4,8] all-reduce(%arg0), metadata={op_name="reduce-scatter/all-gather"}
          ROOT %slice.1 = bf16[4,2] dynamic-slice(%all-reduce.1)
        """,
    )

    assert analysis.compiler_reduce_scatter_count == 0
    assert analysis.compiler_all_reduce_count == 1
    assert analysis.compiler_all_gather_count == 0
    assert analysis.sparse_core_reduce_scatter_count == 0


@pytest.mark.parametrize(
    ("cost", "memory", "message"),
    (
        (None, _memory(), "COST_UNAVAILABLE"),
        ({}, _memory(), "COST_EMPTY"),
        ({"flops": float("nan")}, _memory(), "COST_VALUE_INVALID"),
        ({"flops": -1.0}, _memory(), "COST_VALUE_INVALID"),
        ({"optimal_seconds": float("-inf")}, _memory(), "COST_VALUE_INVALID"),
        ({"flops": True}, _memory(), "COST_VALUE_INVALID"),
        ({"flops": 1.0}, None, "MEMORY_UNAVAILABLE"),
        ({"flops": 1.0}, _memory(peak_memory_in_bytes=-1), "MEMORY_FIELD_INVALID"),
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


def test_compiler_analysis_replay_rejects_forged_collective_strategy(
    tmp_path: Path,
) -> None:
    stablehlo = '"stablehlo.reduce_scatter"(%arg0)'
    compiler_hlo = "%rs = bf16[8] reduce-scatter(%arg0)"
    analysis = capture_compiler_analysis(
        _Executable({"flops": 2.0}, _memory(), compiler_hlo=compiler_hlo),
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
    )
    payload = analysis.model_dump(mode="json")
    payload["collectives"]["compiler_reduce_scatter_count"] = 0
    stablehlo_path = tmp_path / "stablehlo.txt"
    compiler_hlo_path = tmp_path / "compiler_hlo.txt"
    analysis_path = tmp_path / "analysis.json"
    stablehlo_path.write_text(stablehlo + "\n")
    compiler_hlo_path.write_text(compiler_hlo + "\n")
    analysis_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="COLLECTIVE_MISMATCH"):
        validate_compiler_analysis(
            analysis_path,
            stablehlo_path=stablehlo_path,
            compiler_hlo_path=compiler_hlo_path,
        )


def test_compiler_collective_surface_rejects_unsorted_or_false_bf16_shapes() -> None:
    collectives = analyze_compiler_collectives(
        stablehlo='"stablehlo.reduce_scatter"(%arg0)',
        compiler_hlo="%rs = bf16[8] reduce-scatter(%arg0)",
    )

    def point(columns: int, *, payload_bytes: int | None = None):
        return CompilerCollectiveStrategyPoint(
            rows=128,
            columns=columns,
            payload_bytes_per_device=payload_bytes or 128 * columns * 2,
            stablehlo_sha256="1" * 64,
            compiler_hlo_sha256="2" * 64,
            compiler_analysis_sha256="3" * 64,
            output_sha256="4" * 64,
            collectives=collectives,
        )

    with pytest.raises(ValueError, match="payload does not match shape"):
        point(96, payload_bytes=1)
    with pytest.raises(ValueError, match="sorted and unique"):
        CompilerCollectiveStrategySurface(
            mesh_axes=(("d", 2), ("t", 4)),
            dtype="bfloat16",
            points=(point(128), point(96), point(100)),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["cost_metrics"][0].update(
                available=False,
                value=None,
            ),
            "unsupported compiler metric unavailable state",
        ),
        (
            lambda payload: payload["memory"].update(
                buffer_assignment_available=False,
            ),
            "unavailable buffer assignment cannot claim bytes",
        ),
        (
            lambda payload: payload["collectives"].update(
                sparse_core_all_gather_count=1,
            ),
            "sparse-core all-gather count exceeds emitted count",
        ),
    ),
)
def test_compiler_analysis_rejects_availability_and_strategy_lies(
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    analysis = capture_compiler_analysis(
        _Executable({"flops": 2.0}, _memory()),
        stablehlo="module @main {}",
        compiler_hlo="HloModule main",
    )
    payload = analysis.model_dump(mode="json")
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        CompilerExecutableAnalysis.model_validate(payload)
