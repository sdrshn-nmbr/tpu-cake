from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tpu_cake.cli import _compare_physical_fusion, _verify_physical_fusion
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import VectorComputeOp
from tpu_cake.physical_fusion import (
    SeqaxSiluMultiplyFusionContract,
    SeqaxSiluMultiplyFusionReport,
    UnsupportedPhysicalFusionError,
    compare_physical_silu_multiply_fusion,
    default_seqax_silu_multiply_fusion_contract,
    derive_seqax_silu_multiply_fusion_report,
    validate_seqax_silu_multiply_fusion_report,
)
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import SeqaxFeedForwardFusion, seqax_forward_schedule

CONTRACT = Path("contracts/seqax-silu-multiply-fusion-v1.json")
SMALL_SEQAX = {
    "batch": 2,
    "sequence": 4,
    "model": 8,
    "vocabulary": 16,
    "feed_forward": 16,
    "query_groups": 2,
    "key_value_heads": 4,
    "head": 4,
    "layers": 2,
    "data_mesh": 2,
    "tensor_mesh": 4,
    "rope_max_timescale": 256,
}


def test_external_fusion_contract_is_canonical() -> None:
    saved = SeqaxSiluMultiplyFusionContract.model_validate_json(CONTRACT.read_text())

    assert saved == default_seqax_silu_multiply_fusion_contract()


def test_seqax_fusion_report_replays_all_declared_surface_scenarios() -> None:
    contract = default_seqax_silu_multiply_fusion_contract()
    report = derive_seqax_silu_multiply_fusion_report(contract)

    assert tuple(value.scenario_name for value in report.scenarios) == (
        "tiny",
        "wider",
        "deeper",
    )
    assert tuple(len(value.comparison.rewrites) for value in report.scenarios) == (2, 2, 4)
    assert tuple(
        value.comparison.allocated_vmem_savings_bytes_per_device for value in report.scenarios
    ) == (64, 512, 1024)
    assert tuple(
        value.comparison.peak_live_vmem_savings_bytes_per_device for value in report.scenarios
    ) == (0, 0, 0)
    assert all(
        value.baseline_distributed_schedule_sha256 != value.candidate_distributed_schedule_sha256
        for value in report.scenarios
    )
    assert all(value.comparison.declared_work_is_equal for value in report.scenarios)
    assert report.measured_performance_winner is None
    assert not report.predictive_validation

    validate_seqax_silu_multiply_fusion_report(report, contract=contract)


def test_fusion_report_rejects_coordinated_saved_report_change() -> None:
    contract = default_seqax_silu_multiply_fusion_contract()
    report = derive_seqax_silu_multiply_fusion_report(contract)
    scenario = report.scenarios[0]
    comparison = scenario.comparison.model_copy(
        update={
            "allocated_vmem_savings_bytes_per_device": (
                scenario.comparison.allocated_vmem_savings_bytes_per_device + 1
            )
        }
    )
    changed = report.model_copy(
        update={
            "scenarios": (
                scenario.model_copy(update={"comparison": comparison}),
                *report.scenarios[1:],
            )
        }
    )

    with pytest.raises(ValueError, match="REPORT_REPLAY_MISMATCH"):
        validate_seqax_silu_multiply_fusion_report(changed, contract=contract)


def test_fusion_contract_rejects_changed_surface_authority() -> None:
    payload = json.loads(CONTRACT.read_text())
    payload["surface_id"] = "0" * 64

    with pytest.raises(ValueError, match="CONTRACT_NOT_CANONICAL"):
        SeqaxSiluMultiplyFusionContract.model_validate(payload)


def test_fusion_comparison_rejects_swapped_gate_and_up_lineage() -> None:
    baseline = lower_seqax_forward_to_physical(seqax_forward_schedule(**SMALL_SEQAX)).module
    candidate = lower_seqax_forward_to_physical(
        seqax_forward_schedule(
            **SMALL_SEQAX,
            feed_forward_fusion=SeqaxFeedForwardFusion.SILU_MULTIPLY,
        )
    ).module
    for operation in candidate.walk():
        if isinstance(operation, VectorComputeOp) and operation.function.data == "silu_multiply":
            operation.operands = (operation.inputs[1], operation.inputs[0], operation.output)
    candidate.verify()

    with pytest.raises(UnsupportedPhysicalFusionError, match="producer lineage"):
        compare_physical_silu_multiply_fusion(
            baseline,
            candidate,
            hardware=tpu7x_tensorcore_rates(),
        )


def test_fused_schedule_binds_owned_pallas_vector_regions() -> None:
    baseline_distributed = seqax_forward_schedule(
        **SMALL_SEQAX,
        feed_forward_fusion=SeqaxFeedForwardFusion.SEPARATE,
    )
    baseline_physical = lower_seqax_forward_to_physical(baseline_distributed).module
    baseline_plan = lower_seqax_physical_to_pallas(baseline_distributed, baseline_physical)
    distributed = seqax_forward_schedule(
        **SMALL_SEQAX,
        feed_forward_fusion=SeqaxFeedForwardFusion.SILU_MULTIPLY,
    )
    physical = lower_seqax_forward_to_physical(distributed).module
    plan = lower_seqax_physical_to_pallas(distributed, physical)

    assert plan.pallas_region_count == 17
    assert plan.pallas_vector_region_count == 2
    assert plan.execution_scope.endswith("pallas-einsums-and-fused-vectors")
    assert plan.manifest()["pallas_vector_region_count"] == 2
    assert baseline_plan.pallas_vector_region_count == 0
    assert "pallas_vector_region_count" not in baseline_plan.manifest()
    assert "pallas_vector_region_count=" not in baseline_plan.render_executable_source()
    assert plan.source_sha256() != baseline_plan.source_sha256()
    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        compile(plan.render_executable_source(), "<fused-pallas>", "exec"), namespace
    )
    assert namespace["PLAN"].manifest() == plan.manifest()  # type: ignore[union-attr]


def test_public_fusion_cli_derives_and_replays_without_schedule_inputs(tmp_path: Path) -> None:
    report_path = tmp_path / "fusion-report.json"

    assert _compare_physical_fusion(CONTRACT, report_path) == 0
    assert _verify_physical_fusion(report_path, CONTRACT) == 0

    report = SeqaxSiluMultiplyFusionReport.model_validate_json(report_path.read_text())
    report_path.write_text(report.model_copy(update={"surface_id": "0" * 64}).model_dump_json())
    with pytest.raises(ValueError, match="REPORT_REPLAY_MISMATCH"):
        _verify_physical_fusion(report_path, CONTRACT)


def test_fusion_report_writer_refuses_overwrite(tmp_path: Path) -> None:
    report_path = tmp_path / "fusion-report.json"

    assert _compare_physical_fusion(CONTRACT, report_path) == 0
    before = report_path.read_bytes()
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        _compare_physical_fusion(CONTRACT, report_path)
    assert report_path.read_bytes() == before


def test_fused_seqax_executes_through_the_physical_pallas_plan_on_eight_devices() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import SeqaxFeedForwardFusion, seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs, seqax_forward_reference

parameters = {
    "batch": 2,
    "sequence": 4,
    "model": 8,
    "vocabulary": 16,
    "feed_forward": 16,
    "query_groups": 2,
    "key_value_heads": 4,
    "head": 4,
    "layers": 2,
    "data_mesh": 2,
    "tensor_mesh": 4,
    "rope_max_timescale": 256,
}
devices = jax.devices("cpu")
assert len(devices) == 8
inputs = seqax_forward_inputs(seed=9173, **parameters)
arrays = tuple(jnp.asarray(value) for value in inputs)
outputs = []
for fusion in (SeqaxFeedForwardFusion.SEPARATE, SeqaxFeedForwardFusion.SILU_MULTIPLY):
    distributed = seqax_forward_schedule(**parameters, feed_forward_fusion=fusion)
    physical = lower_seqax_forward_to_physical(distributed).module
    plan = lower_seqax_physical_to_pallas(distributed, physical)
    executable, mesh = plan.build(interpret=True, devices=devices)
    (output,) = executable(*arrays)
    output.block_until_ready()
    outputs.append(np.asarray(output))
    assert mesh.shape == {"d": 2, "t": 4}
np.testing.assert_array_equal(outputs[0], outputs[1])
expected = seqax_forward_reference(inputs, **parameters)
np.testing.assert_allclose(outputs[1], expected, rtol=5e-2, atol=6e-3)
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
