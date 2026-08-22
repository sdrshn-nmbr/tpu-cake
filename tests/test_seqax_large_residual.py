from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_large_residual import (
    SeqaxLargeResidualContract,
    default_seqax_large_residual_contract,
)
from tpu_cake.seqax_large_residual_runner import (
    SeqaxLargeResidualCompilerCapture,
    SeqaxLargeResidualCompilerCaptureCandidate,
)
from tpu_cake.seqax_residual_profile_runner import _prepare_candidates
from tpu_cake.workloads.seqax_forward import SeqaxResidualNormStrategy


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


def test_large_residual_static_schedules_fit_and_match_contract() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    prepared = _prepare_candidates(contract)

    assert tuple(value.expected.candidate for value in prepared) == (
        SeqaxResidualNormStrategy.STANDARD,
        SeqaxResidualNormStrategy.RESIDUAL_ALL_REDUCE,
    )
    assert tuple(value.plan.pallas_region_count for value in prepared) == (9, 9)
    assert tuple(value.plan.physical_schedule_sha256 for value in prepared) == tuple(
        value.expected.physical_schedule_sha256 for value in prepared
    )


def test_large_residual_contract_rejects_parameter_mutation() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    payload = contract.model_dump(exclude_computed_fields=True)
    payload["parameters"]["sequence"] = 127

    with pytest.raises(ValidationError, match="protocol mismatch"):
        SeqaxLargeResidualContract.model_validate(payload)


def test_large_residual_contract_rejects_nonzero_pending_identity() -> None:
    contract = default_seqax_large_residual_contract(_runtime_identity())
    payload = contract.model_dump(exclude_computed_fields=True)
    payload["candidates"][0]["pallas_stablehlo_sha256"] = "1" * 64

    with pytest.raises(ValidationError, match="must be zero"):
        SeqaxLargeResidualContract.model_validate(payload)


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
