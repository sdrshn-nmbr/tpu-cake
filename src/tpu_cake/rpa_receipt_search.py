from __future__ import annotations

import json
import shutil
from pathlib import Path

from tpu_cake.artifacts import file_sha256 as _sha256
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    EvidencePhaseName,
    RpaSearchProvenance,
    RpaSearchSelection,
    RunReceipt,
)
from tpu_cake.rpa_bundle import build_fused_rpa_receipt, validate_fused_rpa_receipt
from tpu_cake.rpa_runner import FusedRpaRunResult
from tpu_cake.rpa_search import (
    RpaSearchContract,
    RpaSearchResult,
    validate_rpa_search_result,
)
from tpu_cake.workloads.inkling_rpa import inkling_fused_rpa_experiment


def _search_selection(
    result: RpaSearchResult,
) -> tuple[str, RpaSearchSelection]:
    if result.winner is not None:
        return result.winner, RpaSearchSelection.CHALLENGER_PROMOTED
    if result.provisional_winner is not None:
        return (
            result.baseline,
            RpaSearchSelection.BASELINE_RETAINED_CONFIRMATION_FAILED,
        )
    return (
        result.baseline,
        RpaSearchSelection.BASELINE_RETAINED_NO_PROMOTABLE_CHALLENGER,
    )


def _search_provenance(
    search_root: Path,
    contract: RpaSearchContract,
    result: RpaSearchResult,
) -> RpaSearchProvenance:
    selected, selection = _search_selection(result)
    candidate = next(item for item in contract.candidates if item.name == selected)
    selected_run = next(item for item in result.runs if item.candidate == selected)
    profiler_config = search_root / selected_run.path / "profiler_config.json"
    selected_schedule = inkling_fused_rpa_experiment(candidate.block_sizes).schedule_sha256
    return RpaSearchProvenance(
        search_id=contract.search_id,
        selection=selection,
        baseline=result.baseline,
        selected_candidate=selected,
        selected_block_sizes=candidate.block_sizes,
        selected_schedule_sha256=selected_schedule,
        selected_run_path=selected_run.path,
        selected_run_result_sha256=selected_run.result_sha256,
        selected_profiler_config_sha256=_sha256(profiler_config),
        contract_sha256=_sha256(search_root / "contract.json"),
        result_sha256=_sha256(search_root / "result.json"),
        run_count=len(result.runs),
    )


def _search_artifacts(bundle_root: Path, search_root: Path) -> tuple[ArtifactReference, ...]:
    artifacts = []
    for path in sorted(item for item in search_root.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle_root).as_posix()
        role = {
            "search/contract.json": ArtifactRole.SEARCH_CONTRACT,
            "search/result.json": ArtifactRole.SEARCH_RESULT,
        }.get(relative, ArtifactRole.SEARCH_EVIDENCE)
        artifacts.append(
            ArtifactReference(
                path=relative,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
                role=role,
            )
        )
    return tuple(artifacts)


def _copy_search_evidence(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"RPA_SEARCH_EVIDENCE_ALREADY_EXISTS path={destination}")
    shutil.copytree(source, destination)


def _run_identity(result: FusedRpaRunResult) -> tuple[object, ...]:
    return (
        result.schedule_sha256,
        result.pallas_source_sha256,
        result.stablehlo_sha256,
        result.compiler_hlo_sha256,
        result.input_sha256,
        result.output_sha256,
        result.oracle_sha256,
        result.runtime,
        result.backend,
        result.device_kind,
        result.device_count,
        result.execution_scope,
        result.backend_executor,
        result.backend_executor_sha256,
        result.backend_manifest,
    )


def build_search_bound_fused_rpa_receipt(
    root: Path,
    search_root: Path,
    expected_contract: RpaSearchContract,
) -> RunReceipt:
    root = root.resolve()
    if (root / "receipt.json").exists():
        raise ValueError("RPA_RECEIPT_ALREADY_EXISTS")
    source_search_root = search_root.resolve()
    if root.is_relative_to(source_search_root) or source_search_root.is_relative_to(root):
        raise ValueError("RPA_SEARCH_AND_FINAL_ROOTS_MUST_BE_DISJOINT")
    search_result = validate_rpa_search_result(source_search_root, expected_contract)
    selected, _ = _search_selection(search_result)
    candidate = next(item for item in expected_contract.candidates if item.name == selected)
    destination_search_root = root / "search"
    _copy_search_evidence(source_search_root, destination_search_root)
    copied_result = validate_rpa_search_result(destination_search_root, expected_contract)
    if copied_result != search_result:
        raise ValueError("RPA_COPIED_SEARCH_RESULT_MISMATCH")
    receipt = build_fused_rpa_receipt(
        root,
        candidate.block_sizes,
        write_receipt=False,
    )
    search_artifacts = _search_artifacts(root, destination_search_root)
    artifacts = tuple(sorted((*receipt.artifacts, *search_artifacts), key=lambda item: item.path))
    phases = tuple(
        phase.model_copy(
            update={
                "artifact_paths": (
                    phase.artifact_paths + tuple(item.path for item in search_artifacts)
                    if phase.name is EvidencePhaseName.AGGREGATE
                    else phase.artifact_paths
                )
            }
        )
        for phase in receipt.phases
    )
    provenance = _search_provenance(
        destination_search_root,
        expected_contract,
        copied_result,
    )
    bound = RunReceipt.model_validate(
        receipt.model_dump(mode="json")
        | {
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
            "phases": [item.model_dump(mode="json") for item in phases],
            "rpa_search_provenance": provenance.model_dump(mode="json"),
        }
    )
    validate_search_bound_fused_rpa_receipt(
        bound,
        root=root,
        expected_contract=expected_contract,
    )
    receipt_path = root / "receipt.json"
    temporary_path = root / "receipt.json.tmp"
    temporary_path.write_text(bound.model_dump_json(indent=2) + "\n")
    temporary_path.replace(receipt_path)
    return bound


def validate_search_bound_fused_rpa_receipt(
    receipt: RunReceipt,
    *,
    root: Path,
    expected_contract: RpaSearchContract,
) -> None:
    provenance = receipt.rpa_search_provenance
    if provenance is None:
        raise ValueError("RPA_SEARCH_PROVENANCE_MISSING")
    search_root = root.resolve() / "search"
    result = validate_rpa_search_result(search_root, expected_contract)
    expected_search_artifacts = _search_artifacts(root.resolve(), search_root)
    declared_search_artifacts = tuple(
        artifact for artifact in receipt.artifacts if Path(artifact.path).parts[0] == "search"
    )
    if declared_search_artifacts != expected_search_artifacts:
        raise ValueError("RPA_SEARCH_ARTIFACT_MANIFEST_MISMATCH")
    expected_provenance = _search_provenance(search_root, expected_contract, result)
    if provenance != expected_provenance:
        raise ValueError("RPA_SEARCH_PROVENANCE_MISMATCH")
    selected = next(
        item for item in expected_contract.candidates if item.name == provenance.selected_candidate
    )
    experiment = inkling_fused_rpa_experiment(selected.block_sizes)
    if receipt.schedule_sha256 != provenance.selected_schedule_sha256:
        raise ValueError("RPA_SEARCH_SELECTION_SCHEDULE_MISMATCH")
    selected_run_root = search_root / provenance.selected_run_path
    selected_run_result_path = selected_run_root / "result.json"
    selected_profiler_path = selected_run_root / "profiler_config.json"
    if (
        _sha256(selected_run_result_path) != provenance.selected_run_result_sha256
        or _sha256(selected_profiler_path) != provenance.selected_profiler_config_sha256
    ):
        raise ValueError("RPA_SEARCH_SELECTED_RUN_ARTIFACT_MISMATCH")
    selected_result = FusedRpaRunResult.model_validate_json(selected_run_result_path.read_text())
    final_trace_result = FusedRpaRunResult.model_validate_json(
        (root / "trace/result.json").read_text()
    )
    if _run_identity(selected_result) != _run_identity(final_trace_result):
        raise ValueError("RPA_SEARCH_FINALIST_EXECUTION_IDENTITY_MISMATCH")
    final_source_state = json.loads((root / "trace/source_state.json").read_text())
    if final_source_state["uv_lock_sha256"] != result.execution_identity.uv_lock_sha256:
        raise ValueError("RPA_SEARCH_FINALIST_LOCK_IDENTITY_MISMATCH")
    final_profiler_path = root / "trace/profiler_config.json"
    if (
        final_profiler_path.read_bytes() != selected_profiler_path.read_bytes()
        or _sha256(final_profiler_path) != provenance.selected_profiler_config_sha256
    ):
        raise ValueError("RPA_SEARCH_FINALIST_PROFILER_CONFIG_MISMATCH")
    validate_fused_rpa_receipt(receipt, experiment, root=root)
