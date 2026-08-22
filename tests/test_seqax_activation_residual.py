from pathlib import Path

from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import json_sha256
from tpu_cake.physical_cost_model import analyze_physical_kernel
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_activation_residual import (
    SeqaxActivationResidualDesignContract,
    default_seqax_activation_residual_design_contract,
)
from tpu_cake.seqax_large_residual_qualification import (
    default_seqax_large_residual_qualification_failure_record,
)
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import (
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
)


def test_external_activation_residual_design_is_canonical() -> None:
    saved = SeqaxActivationResidualDesignContract.model_validate_json(
        Path("contracts/seqax-activation-residual-v1.json").read_text()
    )

    assert saved == default_seqax_activation_residual_design_contract(saved.runtime)
    assert saved.full_activation_bf16_bytes_per_data_shard == 1_048_576
    assert saved.full_activation_bf16_bytes_per_data_shard > (
        saved.illustrative_latency_crossover_bytes
    )
    assert saved.latency_crossover_role == ("illustrative-hardware-specific-not-a-tpu7x-threshold")
    assert saved.predicted_ring_savings_bytes_per_device == 3_145_728
    assert saved.predicted_ring_savings_fraction > 0.19
    assert saved.parameters["model"] == 256
    assert saved.parameters["batch"] == 64
    assert saved.parameters["sequence"] == 64
    assert saved.failed_workload_boundary_relation == (
        "same-one-mib-boundary-with-sixteen-times-shorter-model-axis-reductions"
    )
    assert saved.boundary_reduce_scatter_input == "f32[32,64,256]"
    assert saved.boundary_reduce_scatter_output == "f32[32,64,64]"
    assert saved.boundary_residual_output == "bf16[32,64,64]"
    assert saved.boundary_all_gather_output == "bf16[32,64,256]"
    assert saved.top1_role == "diagnostic-only"
    assert (
        saved.parent_failure_record_id
        == default_seqax_large_residual_qualification_failure_record().record_id
    )
    assert saved.correctness_policy_status == "pending-calibration"
    assert not saved.timing_authorized


def test_activation_residual_static_plans_replay() -> None:
    contract = default_seqax_activation_residual_design_contract(_runtime_identity())
    parameters = dict(contract.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])

    for expected in contract.candidates:
        distributed = seqax_forward_schedule(
            **parameters,
            residual_norm_strategy=expected.candidate,
        )
        physical = lower_seqax_forward_to_physical(distributed).module
        plan = lower_seqax_physical_to_pallas(distributed, physical)
        report = analyze_physical_kernel(physical, hardware=tpu7x_tensorcore_rates())

        assert plan.distributed_schedule_sha256 == expected.distributed_schedule_sha256
        assert plan.physical_schedule_sha256 == expected.physical_schedule_sha256
        assert plan.source_sha256() == expected.pallas_source_sha256
        assert json_sha256(plan.manifest()) == expected.pallas_manifest_sha256
        assert plan.pallas_region_count == expected.expected_pallas_regions
        assert (
            report.devices[0].collective_ring_equivalent_bytes
            == expected.expected_ring_equivalent_ici_bytes_per_device
        )
        assert (
            report.memory.peak_live_vmem_bytes_per_device
            == expected.expected_peak_vmem_bytes_per_device
        )
