from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_large_residual import (
    SeqaxLargeResidualContract,
    default_seqax_large_residual_contract,
    pending_seqax_large_residual_contract,
)
from tpu_cake.seqax_large_residual_runner import (
    SeqaxLargeResidualCompilerCapture,
    SeqaxLargeResidualCompilerCaptureCandidate,
    SeqaxLargeResidualCompilerCaptureRecord,
)
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_residual_profile_runner import _prepare_candidates
from tpu_cake.workloads.seqax_forward import (
    SeqaxNumericalSemantics,
    SeqaxResidualNormStrategy,
    seqax_forward_schedule,
)


def _collectives(reduce_scatters: int) -> CompilerCollectiveAnalysis:
    return CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=reduce_scatters,
        stablehlo_all_gather_count=8,
        compiler_reduce_scatter_count=reduce_scatters,
        compiler_all_reduce_count=5,
        compiler_all_gather_count=8,
        sparse_core_reduce_scatter_count=reduce_scatters,
        sparse_core_all_gather_count=8,
    )


def _capture_candidate(
    candidate: SeqaxResidualNormStrategy,
    reduce_scatters: int,
) -> SeqaxLargeResidualCompilerCaptureCandidate:
    digest = "1" * 64
    return SeqaxLargeResidualCompilerCaptureCandidate(
        candidate=candidate,
        distributed_schedule_sha256=digest,
        physical_schedule_sha256=digest,
        pallas_source_sha256=digest,
        pallas_manifest_sha256=digest,
        pallas_stablehlo_sha256=digest,
        pallas_compiler_hlo_sha256=digest,
        control_stablehlo_sha256=digest,
        control_compiler_hlo_sha256=digest,
        pallas_compiler_collectives=_collectives(reduce_scatters),
        control_compiler_collectives=_collectives(reduce_scatters),
        pallas_peak_memory_bytes=1,
        control_peak_memory_bytes=1,
        physical_peak_vmem_bytes_per_device=1,
        ring_equivalent_ici_bytes_per_device=1,
    )


def test_large_residual_contract_round_trips_and_binds_larger_regime() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    saved = SeqaxLargeResidualContract.model_validate_json(
        contract.model_dump_json(exclude_computed_fields=True)
    )

    assert saved == contract
    assert saved.full_activation_bf16_bytes_per_data_shard == 1_048_576
    assert saved.full_activation_bf16_bytes_per_data_shard == (
        2048 * saved.prior_small_activation_bf16_bytes_per_data_shard
    )
    assert saved.full_activation_bf16_bytes_per_data_shard > (
        saved.illustrative_latency_crossover_bytes
    )
    assert (
        saved.candidates[1].expected_ring_equivalent_ici_bytes_per_device
        - (saved.candidates[0].expected_ring_equivalent_ici_bytes_per_device)
        == 3_145_728
    )
    assert saved.allow_early_stopping is False
    assert saved.allow_further_retry is False


def test_large_residual_external_contract_matches_factory() -> None:
    saved = SeqaxLargeResidualContract.model_validate_json(
        Path("contracts/seqax-large-residual-v1.json").read_text()
    )

    assert saved == default_seqax_large_residual_contract(saved.runtime)


def test_large_residual_compiler_capture_record_binds_two_clean_processes() -> None:
    record = SeqaxLargeResidualCompilerCaptureRecord.model_validate_json(
        Path("contracts/seqax-large-residual-compiler-captures-v1.json").read_text()
    )

    assert record.capture_log_sha256[0] == record.capture_log_sha256[1]
    assert record.capture.contract_id == record.pending_contract_id
    assert len(record.record_id) == 64


def test_large_residual_static_schedules_fit_and_match_contract() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    parameters = dict(contract.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    plans = tuple(
        lower_seqax_physical_to_pallas(
            distributed,
            lower_seqax_forward_to_physical(distributed).module,
        )
        for expected in contract.candidates
        for distributed in (
            seqax_forward_schedule(
                **parameters,
                residual_norm_strategy=expected.candidate,
            ),
        )
    )

    assert tuple(value.candidate for value in contract.candidates) == (
        SeqaxResidualNormStrategy.STANDARD,
        SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
    )
    assert tuple(plan.pallas_region_count for plan in plans) == (9, 9)
    assert tuple(plan.physical_schedule_sha256 for plan in plans) == tuple(
        value.physical_schedule_sha256 for value in contract.candidates
    )


def test_large_residual_frozen_contract_rejects_current_source_identity() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())

    with pytest.raises(ValueError, match="SEQAX_RESIDUAL_PROFILE_PLAN_IDENTITY_MISMATCH"):
        _prepare_candidates(contract)


def test_large_residual_contract_rejects_parameter_mutation() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    payload = contract.model_dump(exclude_computed_fields=True)
    payload["parameters"]["sequence"] = 127

    with pytest.raises(ValidationError, match="protocol mismatch"):
        SeqaxLargeResidualContract.model_validate(payload)


def test_large_residual_contract_rejects_nonzero_pending_identity() -> None:
    contract = pending_seqax_large_residual_contract(_runtime_identity())
    payload = contract.model_dump(exclude_computed_fields=True)
    payload["candidates"][0]["pallas_stablehlo_sha256"] = "1" * 64

    with pytest.raises(ValidationError, match="must be zero"):
        SeqaxLargeResidualContract.model_validate(payload)


def test_large_residual_pending_contract_identity_matches_capture_record() -> None:
    record = SeqaxLargeResidualCompilerCaptureRecord.model_validate_json(
        Path("contracts/seqax-large-residual-compiler-captures-v1.json").read_text()
    )

    assert pending_seqax_large_residual_contract(record.capture.runtime).contract_id == (
        record.pending_contract_id
    )


def test_large_residual_capture_rejects_rewritten_standard_reduce_scatter() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    common = {
        "contract_id": contract.contract_id,
        "source_commit": "1" * 40,
        "uv_lock_sha256": "1" * 64,
        "runtime": contract.runtime,
        "device_ids": tuple(range(8)),
    }

    with pytest.raises(ValidationError, match="was rewritten"):
        SeqaxLargeResidualCompilerCapture(
            **common,
            candidates=(
                _capture_candidate(SeqaxResidualNormStrategy.STANDARD, 1),
                _capture_candidate(SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE, 1),
            ),
        )


def test_large_residual_capture_accepts_native_standard_reduce_scatter() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    capture = SeqaxLargeResidualCompilerCapture(
        contract_id=contract.contract_id,
        source_commit="1" * 40,
        uv_lock_sha256="1" * 64,
        runtime=contract.runtime,
        device_ids=tuple(range(8)),
        candidates=(
            _capture_candidate(SeqaxResidualNormStrategy.STANDARD, 3),
            _capture_candidate(SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE, 1),
        ),
    )

    assert len(capture.capture_id) == 64
