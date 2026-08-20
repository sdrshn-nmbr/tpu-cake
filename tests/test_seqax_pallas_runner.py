from pathlib import Path

import pytest

from tpu_cake.cli import _parser
from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import SEMANTIC_IDENTITY_SCHEMA
from tpu_cake.runner import RunMode
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_pallas_runner import (
    SeqaxPallasInvocation,
    _validate_compiled_program,
    run_seqax_physical_pallas,
    seqax_physical_pallas_experiment,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_runner import (
    SEQAX_EVIDENCE_MEASURED_ITERATIONS,
    SEQAX_EVIDENCE_PARAMETERS,
    SEQAX_EVIDENCE_SEED,
    SEQAX_EVIDENCE_WARMUP_ITERATIONS,
)
from tpu_cake.workloads.seqax_forward import (
    SEQAX_FORWARD_INPUT_NAMES,
    seqax_forward_schedule,
)


def _plan():
    distributed = seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    physical = lower_seqax_forward_to_physical(distributed).module
    return lower_seqax_physical_to_pallas(distributed, physical)


def _stablehlo_pallas_chain(
    *,
    einsum_count: int,
    vector_count: int,
    vectors_in_dead_function: bool = False,
) -> str:
    def chain(
        *,
        start: int,
        count: int,
        kernel_name: str,
        input_value: str | None,
    ) -> tuple[list[str], str | None]:
        lines: list[str] = []
        previous = input_value
        for index in range(start, start + count):
            operands = "" if previous is None else previous
            operand_type = "" if previous is None else "tensor<f32>"
            lines.append(
                f"    %{index} = stablehlo.custom_call @tpu_custom_call({operands}) "
                f'{{kernel_name = "{kernel_name}"}} : ({operand_type}) -> tensor<f32>'
            )
            previous = f"%{index}"
        return lines, previous

    main_einsums, main_result = chain(
        start=0,
        count=einsum_count,
        kernel_name="seqax_named_einsum",
        input_value=None,
    )
    main_vectors: list[str] = []
    if not vectors_in_dead_function:
        main_vectors, main_result = chain(
            start=einsum_count,
            count=vector_count,
            kernel_name="seqax_silu_multiply",
            input_value=main_result,
        )
    assert main_result is not None
    functions = [
        "module @jit_physical_call {",
        "  func.func public @main() -> tensor<f32> {",
        *main_einsums,
        *main_vectors,
        f"    return {main_result} : tensor<f32>",
        "  }",
    ]
    if vectors_in_dead_function:
        dead_vectors, dead_result = chain(
            start=einsum_count,
            count=vector_count,
            kernel_name="seqax_silu_multiply",
            input_value=None,
        )
        assert dead_result is not None
        functions.extend(
            (
                "  func.func private @dead() -> tensor<f32> {",
                *dead_vectors,
                f"    return {dead_result} : tensor<f32>",
                "  }",
            )
        )
    functions.append("}")
    return "\n".join(functions)


def test_seqax_pallas_invocation_binds_both_ir_levels_and_source() -> None:
    plan = _plan()
    invocation = SeqaxPallasInvocation(
        identity_schema=SEMANTIC_IDENTITY_SCHEMA,
        execution_schema=plan.schema,
        mode=RunMode.TIMING,
        seed=SEQAX_EVIDENCE_SEED,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        parameters=SEQAX_EVIDENCE_PARAMETERS,
        distributed_schedule_sha256=plan.distributed_schedule_sha256,
        physical_schedule_sha256=plan.physical_schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
        pallas_region_count=plan.pallas_region_count,
        execution_scope=plan.execution_scope,
    )

    assert invocation.distributed_schedule_sha256 == schedule_sha256(
        seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    )
    assert invocation.pallas_region_count == 17


def test_seqax_pallas_invocation_rejects_protocol_drift() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="SEQAX_PALLAS_EVIDENCE_PROTOCOL_MISMATCH"):
        SeqaxPallasInvocation(
            identity_schema=SEMANTIC_IDENTITY_SCHEMA,
            execution_schema=plan.schema,
            mode=RunMode.TIMING,
            seed=SEQAX_EVIDENCE_SEED + 1,
            warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
            measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
            parameters=SEQAX_EVIDENCE_PARAMETERS,
            distributed_schedule_sha256=plan.distributed_schedule_sha256,
            physical_schedule_sha256=plan.physical_schedule_sha256,
            pallas_source_sha256=plan.source_sha256(),
            pallas_region_count=plan.pallas_region_count,
            execution_scope=plan.execution_scope,
        )


def test_seqax_pallas_experiment_binds_physical_execution() -> None:
    plan = _plan()
    experiment = seqax_physical_pallas_experiment(plan)

    assert tuple(value.name for value in experiment.workload.inputs) == (SEQAX_FORWARD_INPUT_NAMES)
    assert experiment.schedule_sha256 == plan.physical_schedule_sha256
    assert experiment.workload.execution is not None
    assert experiment.workload.execution.scope == plan.execution_scope
    assert experiment.workload.execution.source_manifest
    assert experiment.profile.required_timed_hlo_markers == (
        "pallas_call",
        "all-gather",
        "reduce_scatter",
    )


def test_seqax_pallas_runner_rejects_a_non_tpu_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "tpu_cake.seqax_pallas_runner.jax.default_backend",
        lambda: "cpu",
    )

    with pytest.raises(ValueError, match="requires a TPU backend"):
        run_seqax_physical_pallas(tmp_path / "run", mode=RunMode.TIMING)


def test_seqax_pallas_runner_is_available_through_the_cli() -> None:
    arguments = _parser().parse_args(
        (
            "run-seqax-physical-pallas",
            "--output-dir",
            "run",
            "--mode",
            "trace",
        )
    )

    assert arguments.command == "run-seqax-physical-pallas"
    assert arguments.mode == "trace"


def test_seqax_pallas_compiled_program_requires_exact_regions_and_collectives() -> None:
    stablehlo = _stablehlo_pallas_chain(einsum_count=17, vector_count=0)
    compiler_hlo = "\n".join(
        (
            "HloModule jit_physical_call",
            "ENTRY main.1 {",
            *(
                f'pallas_call.{index} = f32[] custom-call(), custom_call_target="tpu_custom_call"'
                for index in range(17)
            ),
            "all_gather.0 = f32[] all-gather(pallas_call.0)",
            "ROOT reduce_scatter.0 = f32[] reduce-scatter(all_gather.0)",
            "}",
        )
    )

    _validate_compiled_program(
        stablehlo,
        compiler_hlo,
        pallas_region_count=17,
        pallas_vector_region_count=0,
        all_gather_count=1,
        reduce_scatter_count=1,
    )

    with pytest.raises(ValueError, match="COMPILED_REGION_COUNT_MISMATCH"):
        _validate_compiled_program(
            stablehlo,
            compiler_hlo.replace(
                'pallas_call.0 = f32[] custom-call(), custom_call_target="tpu_custom_call"',
                "pallas_call.0 = f32[] add()",
            ),
            pallas_region_count=17,
            pallas_vector_region_count=0,
            all_gather_count=1,
            reduce_scatter_count=1,
        )

    with pytest.raises(ValueError, match="COLLECTIVE_COUNT_MISMATCH"):
        _validate_compiled_program(
            stablehlo,
            compiler_hlo.replace("reduce-scatter", "reduce_scatter"),
            pallas_region_count=17,
            pallas_vector_region_count=0,
            all_gather_count=1,
            reduce_scatter_count=1,
        )


def test_seqax_pallas_compiled_program_binds_owned_vector_regions() -> None:
    stablehlo = _stablehlo_pallas_chain(einsum_count=17, vector_count=2)
    compiler_hlo = "\n".join(
        (
            "HloModule jit_physical_call",
            "ENTRY main.1 {",
            *(
                f'pallas_call.{index} = f32[] custom-call(), custom_call_target="tpu_custom_call"'
                for index in range(19)
            ),
            "all_gather.0 = f32[] all-gather(pallas_call.0)",
            "ROOT reduce_scatter.0 = f32[] reduce-scatter(all_gather.0)",
            "}",
        )
    )

    _validate_compiled_program(
        stablehlo,
        compiler_hlo,
        pallas_region_count=17,
        pallas_vector_region_count=2,
        all_gather_count=1,
        reduce_scatter_count=1,
    )

    with pytest.raises(ValueError, match="STABLEHLO_UNKNOWN_TPU_CUSTOM_CALL"):
        _validate_compiled_program(
            stablehlo.replace(
                'kernel_name = "seqax_silu_multiply"',
                'kernel_name = "decoy_silu_multiply"',
                1,
            ),
            compiler_hlo,
            pallas_region_count=17,
            pallas_vector_region_count=2,
            all_gather_count=1,
            reduce_scatter_count=1,
        )

    with pytest.raises(ValueError, match="STABLEHLO_PALLAS_OUTSIDE_MAIN"):
        _validate_compiled_program(
            _stablehlo_pallas_chain(
                einsum_count=17,
                vector_count=2,
                vectors_in_dead_function=True,
            ),
            compiler_hlo,
            pallas_region_count=17,
            pallas_vector_region_count=2,
            all_gather_count=1,
            reduce_scatter_count=1,
        )

    compiler_with_dead_calls = "\n".join(
        (
            "HloModule jit_physical_call",
            "dead.0 {",
            'pallas_call.17 = f32[] custom-call(), custom_call_target="tpu_custom_call"',
            'ROOT pallas_call.18 = f32[] custom-call(), custom_call_target="tpu_custom_call"',
            "}",
            "ENTRY main.1 {",
            *(
                f'pallas_call.{index} = f32[] custom-call(), custom_call_target="tpu_custom_call"'
                for index in range(17)
            ),
            "all_gather.0 = f32[] all-gather(pallas_call.0)",
            "ROOT reduce_scatter.0 = f32[] reduce-scatter(all_gather.0)",
            "}",
        )
    )
    with pytest.raises(ValueError, match="COMPILED_REGION_COUNT_MISMATCH"):
        _validate_compiled_program(
            stablehlo,
            compiler_with_dead_calls,
            pallas_region_count=17,
            pallas_vector_region_count=2,
            all_gather_count=1,
            reduce_scatter_count=1,
        )


def test_seqax_pallas_compiled_program_rejects_marker_decoys() -> None:
    stablehlo = "\n".join(
        (
            "module @fake {",
            "func.func public @main() {",
            *("not hlo seqax_named_einsum" for _ in range(17)),
            "}",
            "}",
        )
    )
    compiler_hlo = "\n".join(
        (
            "HloModule fake",
            "ENTRY main {",
            *(f'not hlo custom_call_target="tpu_custom_call" {index}' for index in range(17)),
            "not hlo all-gather",
            "not hlo reduce-scatter",
            "}",
        )
    )

    with pytest.raises(ValueError, match="STABLEHLO_INVALID"):
        _validate_compiled_program(
            stablehlo,
            compiler_hlo,
            pallas_region_count=17,
            pallas_vector_region_count=0,
            all_gather_count=1,
            reduce_scatter_count=1,
        )
