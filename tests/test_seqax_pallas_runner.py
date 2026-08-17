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

    assert tuple(value.name for value in experiment.workload.inputs) == (
        SEQAX_FORWARD_INPUT_NAMES
    )
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
    stablehlo = "\n".join("seqax_named_einsum" for _ in range(17))
    compiler_hlo = "\n".join(
        (
            *(f'pallas_call.{index} custom_call_target="tpu_custom_call"' for index in range(17)),
            "all-gather",
            "reduce-scatter",
        )
    )

    _validate_compiled_program(stablehlo, compiler_hlo, pallas_region_count=17)

    with pytest.raises(ValueError, match="COMPILED_REGION_COUNT_MISMATCH"):
        _validate_compiled_program(
            stablehlo,
            compiler_hlo.replace(
                'pallas_call.0 custom_call_target="tpu_custom_call"',
                "pallas_call.0",
            ),
            pallas_region_count=17,
        )

    with pytest.raises(ValueError, match="COMPILER_HLO_MISSING"):
        _validate_compiled_program(
            stablehlo,
            compiler_hlo.replace("reduce-scatter", "reduce_scatter"),
            pallas_region_count=17,
        )
