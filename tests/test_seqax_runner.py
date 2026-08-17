from pathlib import Path

import pytest

from tpu_cake.frontend import schedule_sha256
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.runner import RunMode
from tpu_cake.seqax_runner import (
    SEQAX_EVIDENCE_MEASURED_ITERATIONS,
    SEQAX_EVIDENCE_PARAMETERS,
    SEQAX_EVIDENCE_SEED,
    SEQAX_EVIDENCE_WARMUP_ITERATIONS,
    SeqaxForwardInvocation,
    run_seqax_forward,
)
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule


def test_seqax_evidence_invocation_binds_the_complete_plan() -> None:
    module = seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    plan = lower_distributed_program_to_jax_mesh(module)
    invocation = SeqaxForwardInvocation(
        identity_schema="length-prefixed-v2",
        execution_schema=plan.schema,
        mode=RunMode.TIMING,
        seed=SEQAX_EVIDENCE_SEED,
        warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
        parameters=SEQAX_EVIDENCE_PARAMETERS,
        schedule_sha256=plan.schedule_sha256,
        jax_source_sha256=plan.source_sha256(),
        execution_scope=plan.execution_scope,
    )

    assert invocation.schedule_sha256 == schedule_sha256(module)
    assert invocation.execution_scope == "multi-device-local-shards"


def test_seqax_evidence_invocation_rejects_protocol_drift() -> None:
    module = seqax_forward_schedule(**SEQAX_EVIDENCE_PARAMETERS)
    plan = lower_distributed_program_to_jax_mesh(module)

    with pytest.raises(ValueError, match="SEQAX_EVIDENCE_PROTOCOL_MISMATCH"):
        SeqaxForwardInvocation(
            identity_schema="length-prefixed-v2",
            execution_schema=plan.schema,
            mode=RunMode.TIMING,
            seed=SEQAX_EVIDENCE_SEED + 1,
            warmup_iterations=SEQAX_EVIDENCE_WARMUP_ITERATIONS,
            measured_iterations=SEQAX_EVIDENCE_MEASURED_ITERATIONS,
            parameters=SEQAX_EVIDENCE_PARAMETERS,
            schedule_sha256=plan.schedule_sha256,
            jax_source_sha256=plan.source_sha256(),
            execution_scope=plan.execution_scope,
        )


def test_seqax_device_runner_rejects_a_non_tpu_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("tpu_cake.seqax_runner.jax.default_backend", lambda: "cpu")

    with pytest.raises(ValueError, match="requires a TPU backend"):
        run_seqax_forward(tmp_path / "run", mode=RunMode.TIMING)
