from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import CollectiveOp, MxuEinsumOp, VectorComputeOp
from tpu_cake.identity import json_sha256
from tpu_cake.physical_cost_model import analyze_physical_kernel
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_silu_fusion import (
    SeqaxSiluFusionDesignContract,
    default_seqax_silu_fusion_design_contract,
)
from tpu_cake.workloads.seqax_forward import (
    SeqaxFeedForwardFusion,
    SeqaxFeedForwardVectorExecution,
    SeqaxNumericalSemantics,
    SeqaxResidualNormStrategy,
    seqax_forward_schedule,
)


def _plans(contract: SeqaxSiluFusionDesignContract):
    parameters = dict(contract.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    return tuple(
        (
            expected,
            distributed,
            physical,
            lower_seqax_physical_to_pallas(distributed, physical),
        )
        for expected in contract.candidates
        for distributed in (
            seqax_forward_schedule(
                **parameters,
                feed_forward_fusion=expected.candidate,
                feed_forward_vector_execution=contract.vector_execution,
                residual_norm_strategy=contract.residual_norm_strategy,
            ),
        )
        for physical in (lower_seqax_forward_to_physical(distributed).module,)
    )


def test_seqax_silu_fusion_design_is_canonical() -> None:
    path = Path("contracts/seqax-silu-fusion-design-v1.json")
    saved = SeqaxSiluFusionDesignContract.model_validate_json(path.read_text())

    assert saved == default_seqax_silu_fusion_design_contract(saved.runtime)
    assert saved.design_id == "ec6ca7194ce8430035e94210d9fb2a75c99bf5749c50ab579dedf4fd1914996b"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "e6c7fd18da0edba925b2bc6e0725c25b51f233f06013a9147c5eca0c977d0bff"
    )
    assert saved.baseline is SeqaxFeedForwardFusion.SEPARATE
    assert saved.candidate is SeqaxFeedForwardFusion.SILU_MULTIPLY
    assert saved.vector_execution is SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL
    assert saved.residual_norm_strategy is SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE
    assert saved.compiler_capture_status == "pending"
    assert not saved.timing_authorized


def test_seqax_silu_fusion_design_binds_one_ring_byte_failure() -> None:
    design = SeqaxSiluFusionDesignContract.model_validate_json(
        Path("contracts/seqax-silu-fusion-design-v1.json").read_text()
    )
    ledger = json.loads(Path("contracts/seqax-silu-fusion-compiler-failure-v1.json").read_text())
    matches = tuple(
        attempt
        for attempt in ledger["attempts"]
        if attempt["failed_design_id"] == design.compiler_ring_equivalent_bytes_source_design_id
    )

    assert len(matches) == 1
    assert matches[0]["failure_receipt_id"] == (
        design.compiler_ring_equivalent_bytes_source_failure_receipt_id
    )
    assert matches[0]["archive_sha256"] == (
        design.compiler_ring_equivalent_bytes_source_archive_sha256
    )


def test_seqax_silu_fusion_static_plans_replay() -> None:
    contract = SeqaxSiluFusionDesignContract.model_validate_json(
        Path("contracts/seqax-silu-fusion-design-v1.json").read_text()
    )

    for expected, _distributed, physical, plan in _plans(contract):
        vectors = tuple(
            operation for operation in physical.walk() if isinstance(operation, VectorComputeOp)
        )
        owned = tuple(
            operation.function.data for operation in vectors if operation.implementation is not None
        )
        collectives = Counter(
            operation.kind.data
            for operation in physical.walk()
            if isinstance(operation, CollectiveOp)
        )
        report = analyze_physical_kernel(physical, hardware=tpu7x_tensorcore_rates())

        assert plan.distributed_schedule_sha256 == expected.distributed_schedule_sha256
        assert plan.physical_schedule_sha256 == expected.physical_schedule_sha256
        assert plan.source_sha256() == expected.pallas_source_sha256
        assert json_sha256(plan.manifest()) == expected.pallas_manifest_sha256
        assert owned == tuple(
            name.removeprefix("seqax_strict_bf16_")
            for name in expected.expected_strict_vector_kernels
        )
        assert len(owned) == expected.expected_strict_vector_regions
        assert len(vectors) == expected.expected_physical_vector_operations
        assert plan.pallas_region_count == expected.expected_pallas_regions
        assert collectives == {
            "all_gather": expected.expected_all_gathers,
            "all_reduce": expected.expected_all_reduces,
            "reduce_scatter": expected.expected_reduce_scatters,
        }
        assert (
            report.memory.allocated_vmem_bytes_per_device
            == expected.allocated_vmem_bytes_per_device
        )
        assert (
            report.memory.peak_live_vmem_bytes_per_device
            == expected.peak_live_vmem_bytes_per_device
        )
        assert (
            report.devices[0].collective_ring_equivalent_bytes
            == expected.ring_equivalent_ici_bytes_per_device
        )


def test_seqax_silu_fusion_changes_only_the_declared_static_boundary() -> None:
    contract = default_seqax_silu_fusion_design_contract(
        SeqaxSiluFusionDesignContract.model_validate_json(
            Path("contracts/seqax-silu-fusion-design-v1.json").read_text()
        ).runtime
    )
    baseline, candidate = _plans(contract)

    def non_boundary_vectors(physical):
        return tuple(
            operation.function.data
            for operation in physical.walk()
            if isinstance(operation, VectorComputeOp)
            and operation.function.data not in {"silu", "multiply", "silu_multiply"}
        )

    assert non_boundary_vectors(baseline[2]) == non_boundary_vectors(candidate[2])
    assert sum(isinstance(operation, MxuEinsumOp) for operation in baseline[2].walk()) == sum(
        isinstance(operation, MxuEinsumOp) for operation in candidate[2].walk()
    )
    assert baseline[3].input_contracts == candidate[3].input_contracts
    assert baseline[3].output_contracts == candidate[3].output_contracts


def test_seqax_silu_fusion_contract_rejects_protocol_mutation() -> None:
    contract = SeqaxSiluFusionDesignContract.model_validate_json(
        Path("contracts/seqax-silu-fusion-design-v1.json").read_text()
    )
    payload = contract.model_dump(exclude_computed_fields=True)
    payload["parameters"]["batch"] = 128

    with pytest.raises(ValidationError, match="protocol mismatch"):
        SeqaxSiluFusionDesignContract.model_validate(payload)
