from __future__ import annotations

from pathlib import Path

import pytest

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.runner import RunMode
from tpu_cake.seqax_weight_placement import SeqaxWeightPlacementName
from tpu_cake.seqax_weight_placement_diagnostic import (
    SeqaxWeightPlacementCandidateProfiles,
    SeqaxWeightPlacementDiagnosticContract,
    SeqaxWeightPlacementProfileSummary,
    compare_weight_placement_profiles,
    default_seqax_weight_placement_diagnostic_contract,
)


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python="3.12.3",
        jax="0.11.0",
        jaxlib="0.11.0",
        libtpu="0.0.44.1",
        xla=" --xla_tpu_use_enhanced_launch_barrier=true",
    )


def _summary(
    candidate: SeqaxWeightPlacementName,
    mode: RunMode,
) -> SeqaxWeightPlacementProfileSummary:
    if candidate is SeqaxWeightPlacementName.SHARDED:
        return SeqaxWeightPlacementProfileSummary(
            candidate=candidate,
            mode=mode,
            module_execution_count=50,
            module_median_duration_ns=150_000,
            module_p90_duration_ns=160_000,
            pallas_average_self_time_sum_ns_per_device=1_800,
            collective_completion_average_self_time_sum_ns_per_device=34_000,
            semantic_all_gather_rows=17,
            semantic_reduce_scatter_rows=3,
            async_collective_completion_rows=20,
            high_level_all_gathers=14,
            physical_collectives=20,
            stablehlo_all_gathers=17,
            pallas_regions=9,
            parameter_bytes_per_device=22_912,
            ring_equivalent_ici_bytes_per_device=34_048,
        )
    return SeqaxWeightPlacementProfileSummary(
        candidate=candidate,
        mode=mode,
        module_execution_count=50,
        module_median_duration_ns=145_500,
        module_p90_duration_ns=155_000,
        pallas_average_self_time_sum_ns_per_device=1_890,
        collective_completion_average_self_time_sum_ns_per_device=27_200,
        semantic_all_gather_rows=12,
        semantic_reduce_scatter_rows=3,
        async_collective_completion_rows=15,
        high_level_all_gathers=9,
        physical_collectives=15,
        stablehlo_all_gathers=12,
        pallas_regions=9,
        parameter_bytes_per_device=33_152,
        ring_equivalent_ici_bytes_per_device=23_808,
    )


def _profiles() -> tuple[SeqaxWeightPlacementCandidateProfiles, ...]:
    return tuple(
        SeqaxWeightPlacementCandidateProfiles(
            candidate=candidate,
            trace=_summary(candidate, RunMode.TRACE),
            counters=_summary(candidate, RunMode.COUNTERS),
        )
        for candidate in (
            SeqaxWeightPlacementName.SHARDED,
            SeqaxWeightPlacementName.EMBEDDING_MLP,
        )
    )


def test_weight_placement_diagnostic_contract_is_stable_and_external() -> None:
    contract = default_seqax_weight_placement_diagnostic_contract(_runtime())
    replayed = SeqaxWeightPlacementDiagnosticContract.model_validate_json(
        contract.model_dump_json(exclude_computed_fields=True)
    )
    path = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "seqax-weight-placement-diagnostic-tpu7x.json"
    )
    saved = SeqaxWeightPlacementDiagnosticContract.model_validate_json(path.read_text())

    assert replayed == contract
    assert replayed.diagnostic_id == contract.diagnostic_id
    assert saved == contract
    assert tuple(value.candidate for value in contract.candidates) == (
        SeqaxWeightPlacementName.SHARDED,
        SeqaxWeightPlacementName.EMBEDDING_MLP,
    )


def test_weight_placement_diagnostic_contract_rejects_authority_and_profiler_drift() -> None:
    contract = default_seqax_weight_placement_diagnostic_contract(_runtime())
    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["search_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="search authority"):
        SeqaxWeightPlacementDiagnosticContract.model_validate(payload)

    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["trace_profiler_config"]["host_tracer_level"] = 99
    with pytest.raises(ValueError, match="trace profiler"):
        SeqaxWeightPlacementDiagnosticContract.model_validate(payload)

    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["candidates"][1]["expected_stablehlo_all_gathers"] = 13
    with pytest.raises(ValueError, match="candidates"):
        SeqaxWeightPlacementDiagnosticContract.model_validate(payload)


def test_weight_placement_diagnostic_comparison_is_explanatory_only() -> None:
    comparison = compare_weight_placement_profiles(
        default_seqax_weight_placement_diagnostic_contract(_runtime()),
        _profiles(),
    )

    assert comparison.trace_module_median_change == pytest.approx(-0.03)
    assert comparison.counter_module_median_change == pytest.approx(-0.03)
    assert comparison.trace_collective_completion_change == pytest.approx(-0.2)
    assert comparison.trace_pallas_self_time_change == pytest.approx(0.05)
    assert comparison.high_level_all_gathers_eliminated == 5
    assert comparison.physical_collectives_eliminated == 5
    assert comparison.stablehlo_all_gathers_eliminated == 5
    assert comparison.ring_equivalent_ici_bytes_eliminated_per_device == 10_240
    assert comparison.parameter_bytes_added_per_device == 10_240
    assert any("cannot promote" in value for value in comparison.interpretation)


def test_weight_placement_diagnostic_rejects_swapped_or_mutated_evidence() -> None:
    contract = default_seqax_weight_placement_diagnostic_contract(_runtime())
    profiles = _profiles()
    with pytest.raises(ValueError, match="profile order"):
        compare_weight_placement_profiles(contract, tuple(reversed(profiles)))

    payload = profiles[1].model_dump(mode="python")
    payload["trace"]["stablehlo_all_gathers"] = 11
    payload["counters"]["stablehlo_all_gathers"] = 11
    mutated = SeqaxWeightPlacementCandidateProfiles.model_validate(payload)
    with pytest.raises(ValueError, match="static evidence"):
        compare_weight_placement_profiles(contract, (profiles[0], mutated))

    payload = profiles[1].model_dump(mode="python")
    payload["trace"]["module_execution_count"] = 49
    payload["counters"]["module_execution_count"] = 49
    mutated = SeqaxWeightPlacementCandidateProfiles.model_validate(payload)
    with pytest.raises(ValueError, match="execution count"):
        compare_weight_placement_profiles(contract, (profiles[0], mutated))


def test_weight_placement_diagnostic_rejects_incomplete_collectives_and_nonfinite_values() -> None:
    payload = _summary(
        SeqaxWeightPlacementName.SHARDED,
        RunMode.TRACE,
    ).model_dump(mode="python")
    payload["async_collective_completion_rows"] = 19
    with pytest.raises(ValueError, match="collective inventory"):
        SeqaxWeightPlacementProfileSummary.model_validate(payload)

    payload = _summary(
        SeqaxWeightPlacementName.SHARDED,
        RunMode.TRACE,
    ).model_dump(mode="python")
    payload["module_median_duration_ns"] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        SeqaxWeightPlacementProfileSummary.model_validate(payload)
