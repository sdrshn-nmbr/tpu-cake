from __future__ import annotations

from pathlib import Path

import pytest

from tpu_cake.cli import _parser
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.runner import RunMode
from tpu_cake.seqax_pallas_diagnostic import _gviz_rows
from tpu_cake.seqax_weight_placement import SeqaxWeightPlacementName
from tpu_cake.seqax_weight_placement_diagnostic import (
    SeqaxWeightPlacementCandidateProfiles,
    SeqaxWeightPlacementDiagnosticCandidateResult,
    SeqaxWeightPlacementDiagnosticCapture,
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


def _capture(
    candidate: SeqaxWeightPlacementName,
    mode: RunMode,
) -> SeqaxWeightPlacementDiagnosticCapture:
    counters = mode is RunMode.COUNTERS
    return SeqaxWeightPlacementDiagnosticCapture(
        candidate=candidate,
        mode=mode,
        step_event=f"event.{candidate}.{mode}",
        profiler_config_sha256="1" * 64,
        xplane_sha256="2" * 64,
        assessment_sha256="3" * 64,
        attribution_sha256="4" * 64,
        program_id="42",
        summary=_summary(candidate, mode),
        periodic_counter_names=("COUNT_MXU_BUSY_0",) if counters else (),
        periodic_counter_samples_per_core=({"0": 2, "2": 2, "4": 2, "6": 2} if counters else {}),
        hbm_read_counter_names=1 if counters else 0,
        hbm_write_counter_names=1 if counters else 0,
        cycle_counter_names=1 if counters else 0,
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


def test_weight_placement_diagnostic_cli_requires_both_external_contracts() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "diagnose-seqax-weight-placement",
            "--search-root",
            "search",
            "--search-contract",
            "search.json",
            "--contract",
            "diagnostic.json",
            "--output-dir",
            "output",
        ]
    )
    assert args.search_contract == Path("search.json")
    assert args.contract == Path("diagnostic.json")

    verify = parser.parse_args(
        [
            "verify-seqax-weight-placement-diagnostic",
            "output",
            "--search-contract",
            "search.json",
            "--contract",
            "diagnostic.json",
        ]
    )
    assert verify.search_contract == Path("search.json")
    assert verify.contract == Path("diagnostic.json")


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
    assert comparison.trace_semantic_all_gather_rows_eliminated == 5
    assert comparison.trace_async_completion_rows_eliminated == 5
    assert any("cannot promote" in value for value in comparison.interpretation)


def test_weight_placement_diagnostic_separates_static_counts_from_aggregated_xprof_rows() -> None:
    accepted = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "imported"
        / "seqax-pallas-incumbent-diagnostic-8ce697a"
        / "trace"
        / "xprof"
        / "hlo_stats.json"
    )
    observed = (8, 3, 11)
    if accepted.is_file():
        rows = _gviz_rows(accepted)
        program_ids = {
            str(row["program_id"])
            for row in rows
            if str(row.get("hlo_op_name", "")).startswith("seqax_named_einsum")
        }
        assert len(program_ids) == 1
        program_rows = tuple(row for row in rows if str(row["program_id"]) in program_ids)
        observed = (
            sum(row.get("category") == "all-gather" for row in program_rows),
            sum(row.get("category") == "reduce-scatter" for row in program_rows),
            sum(
                row.get("category") == "async-done"
                and "call-done" in str(row.get("hlo_op_name", ""))
                for row in program_rows
            ),
        )
    assert observed == (8, 3, 11)
    profiles = list(_profiles())
    sharded_payload = profiles[0].model_dump(mode="python")
    for mode in ("trace", "counters"):
        sharded_payload[mode]["semantic_all_gather_rows"] = observed[0]
        sharded_payload[mode]["semantic_reduce_scatter_rows"] = observed[1]
        sharded_payload[mode]["async_collective_completion_rows"] = observed[2]
    candidate_payload = profiles[1].model_dump(mode="python")
    for mode in ("trace", "counters"):
        candidate_payload[mode]["semantic_all_gather_rows"] = 6
        candidate_payload[mode]["semantic_reduce_scatter_rows"] = 3
        candidate_payload[mode]["async_collective_completion_rows"] = 9
    comparison = compare_weight_placement_profiles(
        default_seqax_weight_placement_diagnostic_contract(_runtime()),
        (
            SeqaxWeightPlacementCandidateProfiles.model_validate(sharded_payload),
            SeqaxWeightPlacementCandidateProfiles.model_validate(candidate_payload),
        ),
    )

    assert comparison.stablehlo_all_gathers_eliminated == 5
    assert comparison.trace_semantic_all_gather_rows_eliminated == 2
    assert comparison.trace_async_completion_rows_eliminated == 2


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


def test_weight_placement_diagnostic_capture_rejects_counter_and_program_forgery() -> None:
    trace = _capture(SeqaxWeightPlacementName.SHARDED, RunMode.TRACE)
    counters = _capture(SeqaxWeightPlacementName.SHARDED, RunMode.COUNTERS)
    payload = trace.model_dump(mode="python")
    payload["periodic_counter_names"] = ("COUNT_MXU_BUSY_0",)
    with pytest.raises(ValueError, match="trace carries counter claims"):
        SeqaxWeightPlacementDiagnosticCapture.model_validate(payload)

    with pytest.raises(ValueError, match="candidate evidence"):
        SeqaxWeightPlacementDiagnosticCandidateResult(
            candidate=SeqaxWeightPlacementName.SHARDED,
            distributed_schedule_sha256="a" * 64,
            physical_schedule_sha256="b" * 64,
            pallas_source_sha256="c" * 64,
            stablehlo_sha256="d" * 64,
            compiler_hlo_sha256="e" * 64,
            cost_model_sha256="f" * 64,
            input_sha256=("1" * 64,),
            output_sha256="2" * 64,
            expected_output_sha256="2" * 64,
            exact_search_output_parity=True,
            trace=trace,
            counters=counters.model_copy(update={"program_id": "different"}),
        )
