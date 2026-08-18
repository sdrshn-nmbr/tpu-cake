from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tpu_cake.seqax_weight_placement_diagnostic_runner as runner
from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import ExperimentLedger, RunState
from tpu_cake.runner import RunMode
from tpu_cake.seqax_pallas_search import SeqaxPallasDevice
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementName,
    SeqaxWeightPlacementPlan,
)
from tpu_cake.seqax_weight_placement_diagnostic import (
    SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA,
    SeqaxWeightPlacementCandidateProfiles,
    SeqaxWeightPlacementDiagnosticCandidateResult,
    SeqaxWeightPlacementDiagnosticCapture,
    SeqaxWeightPlacementDiagnosticContract,
    SeqaxWeightPlacementDiagnosticReceipt,
    SeqaxWeightPlacementDiagnosticResult,
    SeqaxWeightPlacementProfileSummary,
    compare_weight_placement_profiles,
)
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


def _contracts() -> tuple[
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementDiagnosticContract,
]:
    repository = Path(__file__).resolve().parents[1]
    search = SeqaxWeightPlacementContract.model_validate_json(
        (repository / "contracts" / "seqax-weight-placement-tpu7x.json").read_text()
    )
    diagnostic = SeqaxWeightPlacementDiagnosticContract.model_validate_json(
        (repository / "contracts" / "seqax-weight-placement-diagnostic-tpu7x.json").read_text()
    )
    return search, diagnostic


def _devices() -> tuple[SeqaxPallasDevice, ...]:
    return tuple(
        SeqaxPallasDevice(
            id=index,
            process_index=0,
            platform="tpu",
            device_kind="TPU7x",
        )
        for index in range(8)
    )


def _summary(
    candidate: SeqaxWeightPlacementName,
    mode: RunMode,
) -> SeqaxWeightPlacementProfileSummary:
    sharded = candidate is SeqaxWeightPlacementName.SHARDED
    return SeqaxWeightPlacementProfileSummary(
        candidate=candidate,
        mode=mode,
        module_execution_count=50,
        module_median_duration_ns=150_000 if sharded else 145_000,
        module_p90_duration_ns=160_000 if sharded else 155_000,
        pallas_average_self_time_sum_ns_per_device=1_800 if sharded else 1_900,
        collective_completion_average_self_time_sum_ns_per_device=(34_000 if sharded else 27_000),
        semantic_all_gather_rows=17 if sharded else 12,
        semantic_reduce_scatter_rows=3,
        async_collective_completion_rows=20 if sharded else 15,
        high_level_all_gathers=14 if sharded else 9,
        physical_collectives=20 if sharded else 15,
        stablehlo_all_gathers=17 if sharded else 12,
        pallas_regions=9,
        parameter_bytes_per_device=22_912 if sharded else 33_152,
        ring_equivalent_ici_bytes_per_device=34_048 if sharded else 23_808,
    )


def _capture(
    candidate: SeqaxWeightPlacementName,
    mode: RunMode,
    event: str,
) -> SeqaxWeightPlacementDiagnosticCapture:
    counters = mode is RunMode.COUNTERS
    return SeqaxWeightPlacementDiagnosticCapture(
        candidate=candidate,
        mode=mode,
        step_event=event,
        profiler_config_sha256="1" * 64,
        xplane_sha256=("2" if mode is RunMode.TRACE else "3") * 64,
        assessment_sha256="4" * 64,
        attribution_sha256=("5" if mode is RunMode.TRACE else "6") * 64,
        program_id=f"program-{candidate}",
        summary=_summary(candidate, mode),
        periodic_counter_names=("COUNT_MXU_BUSY_0",) if counters else (),
        periodic_counter_samples_per_core=({"0": 2, "2": 2, "4": 2, "6": 2} if counters else {}),
        hbm_read_counter_names=1 if counters else 0,
        hbm_write_counter_names=1 if counters else 0,
        cycle_counter_names=1 if counters else 0,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _make_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, object, object]:
    search_contract, contract = _contracts()
    root = tmp_path / "diagnostic"
    root.mkdir(parents=True)
    _write_json(
        root / "contract.json", contract.model_dump(mode="json", exclude_computed_fields=True)
    )
    _write_json(root / "source_state.json", {"test": True})
    (root / "source_diff.patch").write_bytes(b"")
    source_manifest = (SourceFileContract(path="tpu_cake/test.py", sha256="a" * 64),)
    _write_json(
        root / "source_manifest.json",
        [value.model_dump(mode="json") for value in source_manifest],
    )
    prepared = runner.prepare_weight_placement_candidates(search_contract)
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **search_contract.parameters,
        )
    )
    for index, value in enumerate(host_inputs):
        path = root / "inputs" / f"{index:02d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value, allow_pickle=False)
    output = np.zeros((2, 1, 16), dtype=np.float32)
    plan_records = []
    candidate_results = []
    profiles = []
    for expected, value in zip(contract.candidates, prepared, strict=True):
        candidate_root = root / "candidates" / expected.candidate
        search_plan = root / "search" / "plans" / expected.candidate
        texts = {
            "distributed.xdsl": canonical_text(value.distributed),
            "physical.xdsl": canonical_text(value.physical),
            "lowered_pallas.py": value.plan.render_executable_source(),
            "stablehlo.txt": f"stablehlo-{expected.candidate}\n",
            "compiler_hlo.txt": f"compiler-hlo-{expected.candidate}\n",
        }
        for name, text in texts.items():
            candidate_path = candidate_root / name
            search_path = search_plan / name
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            search_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(text)
            search_path.write_text(text)
        _write_json(candidate_root / "plan_manifest.json", value.plan.manifest())
        _write_json(search_plan / "plan_manifest.json", value.plan.manifest())
        cost = runner._cost_report(root, value)
        _write_json(candidate_root / "cost_model.json", cost.model_dump(mode="json"))
        np.save(candidate_root / "output.npy", output, allow_pickle=False)
        expected_output = (
            root
            / "search"
            / "correctness"
            / str(contract.timing_seed)
            / "outputs"
            / f"{expected.candidate}.npy"
        )
        expected_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(expected_output, output, allow_pickle=False)
        stablehlo_sha = runner._sha256(candidate_root / "stablehlo.txt")
        compiler_hlo_sha = runner._sha256(candidate_root / "compiler_hlo.txt")
        plan_records.append(
            SeqaxWeightPlacementPlan(
                candidate=expected.candidate,
                policy=value.candidate.policy,
                distributed_schedule_sha256=value.plan.distributed_schedule_sha256,
                physical_schedule_sha256=value.plan.physical_schedule_sha256,
                pallas_source_sha256=value.plan.source_sha256(),
                stablehlo_sha256=stablehlo_sha,
                compiler_hlo_sha256=compiler_hlo_sha,
                high_level_all_gathers=expected.expected_high_level_all_gathers,
                physical_collectives=expected.expected_physical_collectives,
                stablehlo_all_gathers=expected.expected_stablehlo_all_gathers,
                pallas_regions=9,
                parameter_bytes_per_device=expected.expected_parameter_bytes_per_device,
            )
        )
        trace = _capture(expected.candidate, RunMode.TRACE, expected.trace_step_event)
        counters = _capture(expected.candidate, RunMode.COUNTERS, expected.counter_step_event)
        candidate_results.append(
            SeqaxWeightPlacementDiagnosticCandidateResult(
                candidate=expected.candidate,
                distributed_schedule_sha256=value.plan.distributed_schedule_sha256,
                physical_schedule_sha256=value.plan.physical_schedule_sha256,
                pallas_source_sha256=value.plan.source_sha256(),
                stablehlo_sha256=stablehlo_sha,
                compiler_hlo_sha256=compiler_hlo_sha,
                cost_model_sha256=runner._sha256(candidate_root / "cost_model.json"),
                input_sha256=arrays_sha256(host_inputs),
                output_sha256=array_sha256(output),
                expected_output_sha256=array_sha256(output),
                exact_search_output_parity=True,
                trace=trace,
                counters=counters,
            )
        )
        profiles.append(
            SeqaxWeightPlacementCandidateProfiles(
                candidate=expected.candidate,
                trace=trace.summary,
                counters=counters.summary,
            )
        )
    comparison = compare_weight_placement_profiles(contract, tuple(profiles))
    _write_json(root / "comparison.json", comparison.model_dump(mode="json"))
    run_id = semantic_sha256(
        SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA,
        contract.diagnostic_id,
        runner._sha256(root / "source_state.json"),
        runner._sha256(root / "source_manifest.json"),
    )
    result = SeqaxWeightPlacementDiagnosticResult(
        diagnostic_id=contract.diagnostic_id,
        run_id=run_id,
        search_id=contract.search_id,
        search_receipt_sha256=contract.search_receipt_sha256,
        runtime=contract.runtime,
        devices=_devices(),
        source_state_sha256=runner._sha256(root / "source_state.json"),
        source_manifest_sha256=runner._sha256(root / "source_manifest.json"),
        source_manifest=source_manifest,
        candidates=tuple(candidate_results),
        comparison=comparison,
        correctness_scope="incumbent-bit-exact-diagnostic",
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    search_result = SimpleNamespace(devices=_devices(), plans=tuple(plan_records))
    monkeypatch.setattr(runner, "_validate_search_snapshot", lambda *args, **kwargs: search_result)
    monkeypatch.setattr(runner, "_validate_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_validate_compiled_program", lambda *args, **kwargs: None)
    captures = {value.candidate: (value.trace, value.counters) for value in candidate_results}
    monkeypatch.setattr(
        runner,
        "_replay_candidate_profiles",
        lambda **kwargs: captures[kwargs["expected"].candidate],
    )
    ledger = root / "ledger.sqlite"
    with ExperimentLedger(ledger) as database:
        database.create(run_id, {"diagnostic_id": contract.diagnostic_id})
        database.transition(
            run_id,
            RunState.VERIFIED,
            {
                "search_receipt_sha256": contract.search_receipt_sha256,
                "distributed_schedules": {
                    value.candidate.name: value.plan.distributed_schedule_sha256
                    for value in prepared
                },
            },
        )
        database.transition(
            run_id,
            RunState.LOWERED,
            {
                "pallas_sources": {
                    value.candidate.name: value.plan.source_sha256() for value in prepared
                }
            },
        )
        database.transition(
            run_id,
            RunState.COMPILED,
            {
                "compiled_hlo": {
                    value.candidate: {
                        "stablehlo_sha256": value.stablehlo_sha256,
                        "compiler_hlo_sha256": value.compiler_hlo_sha256,
                    }
                    for value in candidate_results
                }
            },
        )
        database.transition(
            run_id,
            RunState.CORRECT,
            {
                "input_sha256": arrays_sha256(host_inputs),
                "output_sha256": {
                    value.candidate: value.output_sha256 for value in candidate_results
                },
            },
        )
        database.transition(
            run_id,
            RunState.COUNTERED,
            {
                "captures": {
                    value.candidate: {
                        "trace_xplane_sha256": value.trace.xplane_sha256,
                        "counter_xplane_sha256": value.counters.xplane_sha256,
                        "trace_attribution_sha256": value.trace.attribution_sha256,
                        "counter_attribution_sha256": value.counters.attribution_sha256,
                    }
                    for value in candidate_results
                },
                "comparison": comparison.model_dump(mode="json"),
            },
        )
        database.transition(
            run_id,
            RunState.ACCEPTED,
            {"result_sha256": runner._sha256(root / "result.json")},
        )
    runner._close_ledger(ledger)
    receipt = SeqaxWeightPlacementDiagnosticReceipt(
        status="passed",
        diagnostic_id=contract.diagnostic_id,
        run_id=run_id,
        search_id=contract.search_id,
        result_sha256=runner._sha256(root / "result.json"),
        ledger_sha256=runner._sha256(ledger),
        artifacts=runner._artifact_manifest(root),
    )
    _write_json(root / "receipt.json", receipt.model_dump(mode="json"))
    return root, search_contract, contract


def test_diagnostic_public_replay_reaches_the_complete_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, search_contract, contract = _make_tree(tmp_path, monkeypatch)

    result = runner.validate_seqax_weight_placement_diagnostic(
        root,
        search_contract,
        contract,
    )

    assert result.comparison.stablehlo_all_gathers_eliminated == 5
    assert result.comparison.ring_equivalent_ici_bytes_eliminated_per_device == 10_240


def test_diagnostic_rejects_repaired_result_or_swapped_hlo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, search_contract, contract = _make_tree(tmp_path, monkeypatch)
    result_path = root / "result.json"
    payload = json.loads(result_path.read_text())
    payload["comparison"]["trace_module_median_change"] = -0.9
    _write_json(result_path, payload)
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["result_sha256"] = runner._sha256(result_path)
    for artifact in receipt["artifacts"]:
        if artifact["path"] == "result.json":
            artifact["size_bytes"] = result_path.stat().st_size
            artifact["sha256"] = runner._sha256(result_path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="COMPARISON_REPLAY_MISMATCH"):
        runner.validate_seqax_weight_placement_diagnostic(root, search_contract, contract)

    root, search_contract, contract = _make_tree(tmp_path / "second", monkeypatch)
    sharded = root / "candidates" / "sharded" / "stablehlo.txt"
    candidate = root / "candidates" / "embedding-mlp" / "stablehlo.txt"
    sharded_bytes, candidate_bytes = sharded.read_bytes(), candidate.read_bytes()
    sharded.write_bytes(candidate_bytes)
    candidate.write_bytes(sharded_bytes)
    with pytest.raises(ValueError, match="COMPILED_PLAN_MISMATCH"):
        runner.validate_seqax_weight_placement_diagnostic(root, search_contract, contract)


def test_diagnostic_rejects_unsafe_or_overlapping_roots(tmp_path: Path) -> None:
    search = tmp_path / "search"
    search.mkdir()
    with pytest.raises(ValueError, match="ROOT_OVERLAP"):
        runner._require_safe_new_root(search / "diagnostic", search)
    with pytest.raises(ValueError, match="UNSAFE_ROOT"):
        runner._require_safe_new_root(Path.home().resolve(), search)


def test_diagnostic_archives_only_an_owned_incomplete_root(tmp_path: Path) -> None:
    _search, contract = _contracts()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "valuable.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="ROOT_NOT_OWNED"):
        runner._prepare_output_root(unrelated, contract)
    assert sentinel.read_text() == "preserve"

    owned = tmp_path / "owned"
    owned.mkdir()
    _write_json(
        owned / "contract.json",
        contract.model_dump(mode="json", exclude_computed_fields=True),
    )
    (owned / "partial.txt").write_text("negative evidence")
    runner._prepare_output_root(owned, contract)
    archives = tuple(tmp_path.glob("owned.incomplete-*"))
    assert len(archives) == 1
    assert (archives[0] / "partial.txt").read_text() == "negative evidence"
    assert owned.is_dir() and not any(owned.iterdir())


def test_diagnostic_rejects_root_and_nested_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("outside")
    (root / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="SYMLINK"):
        runner._preflight_root(root)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(bundle, target_is_directory=True)
    search, contract = _contracts()
    with pytest.raises(ValueError, match="ROOT_SYMLINK"):
        runner.validate_seqax_weight_placement_diagnostic(alias, search, contract)
