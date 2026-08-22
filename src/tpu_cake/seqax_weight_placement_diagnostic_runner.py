from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

from tpu_cake.artifacts import (
    build_artifact_manifest,
    validate_artifact_manifest,
)
from tpu_cake.artifacts import (
    file_sha256 as _sha256,
)
from tpu_cake.artifacts import save_array as _save_array
from tpu_cake.artifacts import (
    write_json as _write_json,
)
from tpu_cake.artifacts import (
    write_text as _write_text,
)
from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, SourceFileContract
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import (
    EvidenceRun,
    RunState,
    finalize_ledger,
    payload_sha256,
    read_ledger_history,
)
from tpu_cake.metrics import MetricSource
from tpu_cake.runner import RunMode, _runtime_identity, _source_state
from tpu_cake.seqax_cost_model import SeqaxCostModelReport, estimate_seqax_forward
from tpu_cake.seqax_pallas_diagnostic import (
    SeqaxPallasDiagnosticAttribution,
    _attribution,
    _bound_program,
    _canonical_assessment,
    _capture_phase,
    _cost_metric,
    _expected_profile,
    _export_xprof,
    _gviz_rows,
    _profile_files,
    _profile_replay,
    _validate_counter_evidence,
    _validate_xprof,
    _validate_xprof_replay,
)
from tpu_cake.seqax_pallas_runner import (
    _physical_collective_counts,
    _validate_compiled_program,
)
from tpu_cake.seqax_runner import expected_seqax_profiler_contract
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementName,
    SeqaxWeightPlacementReceipt,
    SeqaxWeightPlacementResult,
    default_seqax_weight_placement_contract,
)
from tpu_cake.seqax_weight_placement_diagnostic import (
    SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS,
    SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA,
    SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_WARMUPS,
    SEQAX_WEIGHT_PLACEMENT_SEARCH_RECEIPT_SHA256,
    SeqaxWeightPlacementCandidateProfiles,
    SeqaxWeightPlacementDiagnosticCandidateContract,
    SeqaxWeightPlacementDiagnosticCandidateResult,
    SeqaxWeightPlacementDiagnosticCapture,
    SeqaxWeightPlacementDiagnosticContract,
    SeqaxWeightPlacementDiagnosticReceipt,
    SeqaxWeightPlacementDiagnosticResult,
    SeqaxWeightPlacementProfileSummary,
    compare_weight_placement_profiles,
    default_seqax_weight_placement_diagnostic_contract,
)
from tpu_cake.seqax_weight_placement_runner import (
    CompiledPlacement,
    PreparedPlacement,
    _compile,
    _device_inventory,
    _execute,
    _resident_inputs,
    _validate_devices,
    prepare_weight_placement_candidates,
)
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs
from tpu_cake.xprof_evidence import assess_capture

SEQAX_WEIGHT_PLACEMENT_SEARCH_COMMIT = "6a83e59e1591db09387adbafe90dd0f8af850f76"


def _source_manifest() -> tuple[SourceFileContract, ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "canonical.py",
        package / "cli.py",
        package / "contracts.py",
        package / "cost_model.py",
        package / "dtensor_interpreter.py",
        package / "evidence.py",
        package / "frontend.py",
        package / "identity.py",
        package / "jax_lowering.py",
        package / "ledger.py",
        package / "lowering.py",
        package / "metrics.py",
        package / "physical_geometry.py",
        package / "runner.py",
        package / "seqax_cost_model.py",
        package / "seqax_pallas_diagnostic.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_pallas_runner.py",
        package / "stablehlo.py",
        package / "seqax_pallas_search.py",
        package / "seqax_pallas_search_runner.py",
        package / "seqax_physical_execution.py",
        package / "seqax_physical_lowering.py",
        package / "seqax_runner.py",
        package / "seqax_weight_placement.py",
        package / "seqax_weight_placement_diagnostic.py",
        package / "seqax_weight_placement_diagnostic_runner.py",
        package / "seqax_weight_placement_runner.py",
        package / "xprof_evidence.py",
        package / "xprof_export.py",
        package / "dialects" / "distributed_tensor.py",
        package / "dialects" / "tpu_schedule.py",
        package / "workloads" / "seqax_forward.py",
        package / "workloads" / "seqax_oracle.py",
    )
    return tuple(
        SourceFileContract(
            path=path.relative_to(package.parent).as_posix(),
            sha256=_sha256(path),
        )
        for path in paths
    )


def _require_clean_repository(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SOURCE_DIRTY status={status}")


def _require_safe_new_root(root: Path, search_root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(root == value or root in value.parents for value in protected):
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_UNSAFE_ROOT path={root}")
    if root == search_root or root in search_root.parents or search_root in root.parents:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ROOT_OVERLAP")


def _prepare_output_root(
    root: Path,
    contract: SeqaxWeightPlacementDiagnosticContract,
) -> None:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ROOT_INVALID path={root}")
    if not any(root.iterdir()):
        return
    contract_path = root / "contract.json"
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ROOT_NOT_OWNED path={root}")
    saved = SeqaxWeightPlacementDiagnosticContract.model_validate_json(contract_path.read_text())
    if saved != contract:
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ROOT_NOT_OWNED path={root}")
    archived = root.with_name(f"{root.name}.incomplete-{time.time_ns()}")
    root.rename(archived)
    root.mkdir(parents=True, exist_ok=False)
    print(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ARCHIVED_INCOMPLETE source={root} archive={archived}")


def _preflight_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SYMLINK path={path}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_HARDLINK path={path}")


def _close_ledger(path: Path) -> None:
    if finalize_ledger(path):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_LEDGER_SIDECAR")


def _replay_search_with_recorded_validator(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="tpu-cake-weight-validator-") as directory:
        source_root = Path(directory)
        archive = subprocess.run(
            ["git", "archive", "--format=tar", SEQAX_WEIGHT_PLACEMENT_SEARCH_COMMIT],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        archive_path = source_root / "source.tar"
        archive_path.write_bytes(archive)
        with tarfile.open(archive_path) as stream:
            stream.extractall(source_root, filter="data")
        archive_path.unlink()
        script = """
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from tpu_cake.seqax_weight_placement import SeqaxWeightPlacementContract
import tpu_cake.seqax_weight_placement_runner as runner

root = Path(sys.argv[1])
commit = sys.argv[2]
contract = SeqaxWeightPlacementContract.model_validate_json((root / "contract.json").read_text())
original_run = subprocess.run
def recorded_commit_run(args, *positional, **keywords):
    if tuple(args) == ("git", "rev-parse", "HEAD"):
        return SimpleNamespace(stdout=commit + "\\n", stderr="", returncode=0)
    return original_run(args, *positional, **keywords)
runner.subprocess.run = recorded_commit_run
runner.validate_seqax_weight_placement(root, contract)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root / "src")
        environment["JAX_PLATFORMS"] = "cpu"
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(root),
                    SEQAX_WEIGHT_PLACEMENT_SEARCH_COMMIT,
                ],
                cwd=source_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "no validator output").strip()
            raise ValueError(
                "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_RECORDED_SEARCH_REPLAY_FAILED "
                f"detail={detail[-2000:]}"
            ) from error


def _validate_search_snapshot(
    root: Path,
    trusted_contract: SeqaxWeightPlacementContract,
    *,
    replay_recorded: bool,
) -> SeqaxWeightPlacementResult:
    root = root.resolve()
    _preflight_root(root)
    saved_contract = SeqaxWeightPlacementContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    receipt = SeqaxWeightPlacementReceipt.model_validate_json((root / "receipt.json").read_text())
    result = SeqaxWeightPlacementResult.model_validate_json((root / "result.json").read_text())
    source_state = json.loads((root / "source_state.json").read_text())
    if (
        saved_contract != trusted_contract
        or _sha256(root / "receipt.json") != SEQAX_WEIGHT_PLACEMENT_SEARCH_RECEIPT_SHA256
        or receipt.search_id != trusted_contract.search_id
        or result.search_id != trusted_contract.search_id
        or result.runtime != trusted_contract.runtime
        or result.winner is not None
        or result.provisional_winner != SeqaxWeightPlacementName.EMBEDDING_MLP
        or result.confirmation is None
        or result.confirmation.confirmed
        or source_state.get("git_commit") != SEQAX_WEIGHT_PLACEMENT_SEARCH_COMMIT
        or source_state.get("git_dirty") is not False
        or source_state.get("git_status") != []
        or (root / "source_diff.patch").read_bytes() != b""
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SEARCH_BINDING_MISMATCH")
    declared = tuple(value.path for value in receipt.artifacts)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "receipt.json"
    }
    if len(declared) != len(set(declared)) or set(declared) != observed:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SEARCH_CLOSED_WORLD_MISMATCH")
    for artifact in receipt.artifacts:
        path = root / artifact.path
        if path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SEARCH_ARTIFACT_MISMATCH path={artifact.path}"
            )
    if replay_recorded:
        _replay_search_with_recorded_validator(root)
    return result


def _cost_report(
    root: Path,
    prepared: PreparedPlacement,
) -> SeqaxCostModelReport:
    return estimate_seqax_forward(
        prepared.distributed,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=prepared.plan.distributed_schedule_sha256,
            artifact_path=(f"candidates/{prepared.candidate.name}/distributed.xdsl"),
            tool="tpu-cake",
            field="canonical distributed tensor program",
        ),
        expected_schedule_sha256=prepared.plan.distributed_schedule_sha256,
    )


def _profile_summary(
    *,
    expected: SeqaxWeightPlacementDiagnosticCandidateContract,
    mode: RunMode,
    attribution: SeqaxPallasDiagnosticAttribution,
    hlo_stats: Path,
) -> SeqaxWeightPlacementProfileSummary:
    rows = tuple(
        row for row in _gviz_rows(hlo_stats) if str(row.get("program_id")) == attribution.program_id
    )
    semantic_all_gathers = sum(row.get("category") == "all-gather" for row in rows)
    semantic_reduce_scatters = sum(row.get("category") == "reduce-scatter" for row in rows)
    completion_rows = sum(
        row.get("category") == "async-done"
        and str(row.get("hlo_op_name", "")).startswith(("all-gather", "reduce-scatter"))
        and "call-done" in str(row.get("hlo_op_name", ""))
        for row in rows
    )
    return SeqaxWeightPlacementProfileSummary(
        candidate=expected.candidate,
        mode=mode,
        module_execution_count=attribution.module_execution_count,
        module_median_duration_ns=attribution.module_median_duration_ns,
        module_p90_duration_ns=attribution.module_p90_duration_ns,
        pallas_average_self_time_sum_ns_per_device=(
            attribution.pallas_average_self_time_sum_ns_per_device
        ),
        collective_completion_average_self_time_sum_ns_per_device=(
            attribution.collective_completion_average_self_time_sum_ns_per_device
        ),
        semantic_all_gather_rows=semantic_all_gathers,
        semantic_reduce_scatter_rows=semantic_reduce_scatters,
        async_collective_completion_rows=completion_rows,
        high_level_all_gathers=expected.expected_high_level_all_gathers,
        physical_collectives=expected.expected_physical_collectives,
        stablehlo_all_gathers=expected.expected_stablehlo_all_gathers,
        pallas_regions=expected.expected_pallas_regions,
        parameter_bytes_per_device=expected.expected_parameter_bytes_per_device,
        ring_equivalent_ici_bytes_per_device=(
            expected.expected_ring_equivalent_ici_bytes_per_device
        ),
    )


def _capture_record(
    *,
    candidate_root: Path,
    expected: SeqaxWeightPlacementDiagnosticCandidateContract,
    mode: RunMode,
    xplane: Path,
    assessment: Any,
    attribution: SeqaxPallasDiagnosticAttribution,
) -> SeqaxWeightPlacementDiagnosticCapture:
    phase_root = candidate_root / mode.value
    program_id, _program_name = _bound_program(assessment)
    summary = _profile_summary(
        expected=expected,
        mode=mode,
        attribution=attribution,
        hlo_stats=phase_root / "xprof" / "hlo_stats.json",
    )
    counters = assessment.capture.counters
    if mode is RunMode.TRACE:
        periodic_names: tuple[str, ...] = ()
        samples: dict[str, int] = {}
        hbm_read = hbm_write = cycles = 0
    else:
        periodic_names = counters.periodic_counter_names
        samples = counters.periodic_samples_per_tpu_core
        hbm_read = counters.hbm_read_names
        hbm_write = counters.hbm_write_names
        cycles = counters.cycle_names
    return SeqaxWeightPlacementDiagnosticCapture(
        candidate=expected.candidate,
        mode=mode,
        step_event=(
            expected.trace_step_event if mode is RunMode.TRACE else expected.counter_step_event
        ),
        profiler_config_sha256=_sha256(phase_root / "profiler_config.json"),
        xplane_sha256=_sha256(xplane),
        assessment_sha256=_sha256(phase_root / "profile_assessment.json"),
        attribution_sha256=_sha256(phase_root / "attribution.json"),
        program_id=program_id,
        summary=summary,
        periodic_counter_names=periodic_names,
        periodic_counter_samples_per_core=samples,
        hbm_read_counter_names=hbm_read,
        hbm_write_counter_names=hbm_write,
        cycle_counter_names=cycles,
    )


def _capture_candidate_phase(
    *,
    candidate_root: Path,
    expected: SeqaxWeightPlacementDiagnosticCandidateContract,
    compiled: CompiledPlacement,
    resident: tuple[jax.Array, ...],
    mode: RunMode,
    cost_report: SeqaxCostModelReport,
) -> SeqaxWeightPlacementDiagnosticCapture:
    for _ in range(SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_WARMUPS):
        jax.block_until_ready(compiled.executable(*resident))
    step_event = expected.trace_step_event if mode is RunMode.TRACE else expected.counter_step_event
    phase_root = candidate_root / mode.value
    xplane, assessment, _steps, durations = _capture_phase(
        phase_root,
        SimpleNamespace(compiled=compiled.executable),
        resident,
        mode,
        step_event=step_event,
        iterations=SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS,
    )
    program_id, _program_name = _bound_program(assessment)
    attribution = _attribution(
        physical=compiled.prepared.physical,
        program_id=program_id,
        durations=durations,
        hlo_stats=phase_root / "xprof" / "hlo_stats.json",
        cost_report=cost_report,
        iterations=SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS,
    )
    _write_json(phase_root / "attribution.json", attribution.model_dump(mode="json"))
    return _capture_record(
        candidate_root=candidate_root,
        expected=expected,
        mode=mode,
        xplane=xplane,
        assessment=assessment,
        attribution=attribution,
    )


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    if relative.startswith("search/"):
        return ArtifactRole.SEARCH_EVIDENCE
    fixed = {
        "contract.json": ArtifactRole.EXPERIMENT,
        "source_state.json": ArtifactRole.SOURCE_STATE,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "source_manifest.json": ArtifactRole.BACKEND_MANIFEST,
        "comparison.json": ArtifactRole.SEARCH_EVIDENCE,
        "result.json": ArtifactRole.TRACE_RESULT,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
    }
    if relative in fixed:
        return fixed[relative]
    if relative.startswith("inputs/"):
        return ArtifactRole.CORRECTNESS_INPUT
    if not relative.startswith("candidates/"):
        raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ARTIFACT_UNRECOGNIZED path={relative}")
    name = path.name
    roles = {
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "physical.xdsl": ArtifactRole.PHYSICAL_IR,
        "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
        "stablehlo.txt": ArtifactRole.STABLEHLO,
        "compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "cost_model.json": ArtifactRole.COST_MODEL,
        "output.npy": ArtifactRole.CORRECTNESS_OUTPUT,
        "profiler_config.json": ArtifactRole.PROFILER_CONFIG,
        "profile_assessment.json": ArtifactRole.PROFILE_ASSESSMENT,
        "attribution.json": ArtifactRole.SEARCH_EVIDENCE,
    }
    if name in roles:
        return roles[name]
    if "/trace/profile/" in relative and relative.endswith(".xplane.pb"):
        return ArtifactRole.TIMING_TRACE
    if "/counters/profile/" in relative and relative.endswith(".xplane.pb"):
        return ArtifactRole.COUNTER_TRACE
    if "/profile/" in relative and relative.endswith(".trace.json.gz"):
        return ArtifactRole.PROFILE_AUXILIARY
    if "/xprof/" in relative:
        return ArtifactRole.XPROF_EXPORT
    raise ValueError(f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ARTIFACT_UNRECOGNIZED path={relative}")


def _artifact_manifest(root: Path) -> tuple[ArtifactReference, ...]:
    return build_artifact_manifest(
        root,
        role_for_path=_artifact_role,
    )


def _validate_manifest(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    validate_artifact_manifest(
        root,
        artifacts,
        role_for_path=_artifact_role,
        duplicate_error="SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_CLOSED_WORLD_MISMATCH",
        closed_world_error="SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_CLOSED_WORLD_MISMATCH",
        mismatch_error=lambda path: (
            f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ARTIFACT_MISMATCH path={path}"
        ),
    )


def _validate_source(root: Path, result: SeqaxWeightPlacementDiagnosticResult) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = tuple(
        SourceFileContract.model_validate(value)
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    state = json.loads((root / "source_state.json").read_text())
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        manifest != result.source_manifest
        or manifest != _source_manifest()
        or result.source_state_sha256 != _sha256(root / "source_state.json")
        or result.source_manifest_sha256 != _sha256(root / "source_manifest.json")
        or state.get("git_commit") != current_commit
        or state.get("git_dirty") is not False
        or state.get("git_status") != []
        or state.get("uv_lock_sha256") != _sha256(repository_root / "uv.lock")
        or (root / "source_diff.patch").read_bytes() != b""
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SOURCE_MISMATCH")
    for source in manifest:
        blob = subprocess.run(
            ["git", "show", f"{current_commit}:src/{source.path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != source.sha256:
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SOURCE_BLOB_MISMATCH path={source.path}"
            )


def _expected_plan_files(
    root: Path,
    prepared: PreparedPlacement,
    search_root: Path,
) -> None:
    candidate = prepared.candidate.name
    candidate_root = root / "candidates" / candidate
    search_plan = search_root / "plans" / candidate
    current = {
        "distributed.xdsl": canonical_text(prepared.distributed),
        "physical.xdsl": canonical_text(prepared.physical),
        "lowered_pallas.py": prepared.plan.render_executable_source(),
    }
    for name, text in current.items():
        if (candidate_root / name).read_text() != text or (search_plan / name).read_text() != text:
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_PLAN_REPLAY_MISMATCH candidate={candidate}"
            )
    if (
        json.loads((candidate_root / "plan_manifest.json").read_text()) != prepared.plan.manifest()
        or (candidate_root / "plan_manifest.json").read_bytes()
        != (search_plan / "plan_manifest.json").read_bytes()
        or (candidate_root / "stablehlo.txt").read_bytes()
        != (search_plan / "stablehlo.txt").read_bytes()
        or (candidate_root / "compiler_hlo.txt").read_bytes()
        != (search_plan / "compiler_hlo.txt").read_bytes()
    ):
        raise ValueError(
            f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_COMPILED_PLAN_MISMATCH candidate={candidate}"
        )


def _replay_candidate_profiles(
    *,
    root: Path,
    prepared: PreparedPlacement,
    expected: SeqaxWeightPlacementDiagnosticCandidateContract,
    cost_report: SeqaxCostModelReport,
) -> tuple[SeqaxWeightPlacementDiagnosticCapture, SeqaxWeightPlacementDiagnosticCapture]:
    candidate_root = root / "candidates" / expected.candidate
    records = []
    with tempfile.TemporaryDirectory(prefix="tpu-cake-placement-profile-") as directory:
        replay_parent = Path(directory)
        for mode in (RunMode.TRACE, RunMode.COUNTERS):
            phase_root = candidate_root / mode.value
            if json.loads((phase_root / "profiler_config.json").read_text()) != (
                expected_seqax_profiler_contract(mode)
            ):
                raise ValueError(
                    "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_PROFILER_CONTRACT_MISMATCH "
                    f"candidate={expected.candidate} mode={mode.value}"
                )
            xplane, _trace_files = _profile_files(phase_root / "profile")
            _validate_xprof(phase_root, xplane)
            replay_root = replay_parent / mode.value
            replay_profile = replay_root / "profile"
            replay_xplane = replay_profile / xplane.relative_to(phase_root / "profile")
            replay_xplane.parent.mkdir(parents=True)
            shutil.copy2(xplane, replay_xplane)
            for trace_file in (phase_root / "profile").rglob("*.trace.json.gz"):
                destination = replay_profile / trace_file.relative_to(phase_root / "profile")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(trace_file, destination)
            _export_xprof(replay_profile, replay_root / "xprof")
            _validate_xprof(replay_root, replay_xplane)
            _validate_xprof_replay(phase_root, replay_root)
            expectation = _expected_profile(counters=mode is RunMode.COUNTERS)
            saved_assessment = assess_capture(phase_root, expectation)
            assessment = assess_capture(replay_root, expectation)
            if (
                not saved_assessment.accepted
                or not assessment.accepted
                or _canonical_assessment(saved_assessment.model_dump(mode="json"))
                != _canonical_assessment(assessment.model_dump(mode="json"))
            ):
                raise ValueError(
                    "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ASSESSMENT_REPLAY_MISMATCH "
                    f"candidate={expected.candidate} mode={mode.value}"
                )
            if mode is RunMode.COUNTERS:
                _validate_counter_evidence(saved_assessment)
                _validate_counter_evidence(assessment)
            canonical = _canonical_assessment(assessment.model_dump(mode="json"))
            if json.loads((phase_root / "profile_assessment.json").read_text()) != canonical:
                raise ValueError(
                    "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SAVED_ASSESSMENT_MISMATCH "
                    f"candidate={expected.candidate} mode={mode.value}"
                )
            program_id, program_name = _bound_program(assessment)
            step_event = (
                expected.trace_step_event if mode is RunMode.TRACE else expected.counter_step_event
            )
            _steps, durations = _profile_replay(
                xplane,
                program_name,
                step_event=step_event,
                iterations=SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS,
            )
            attribution = _attribution(
                physical=prepared.physical,
                program_id=program_id,
                durations=durations,
                hlo_stats=replay_root / "xprof" / "hlo_stats.json",
                cost_report=cost_report,
                iterations=SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ITERATIONS,
            )
            saved = SeqaxPallasDiagnosticAttribution.model_validate_json(
                (phase_root / "attribution.json").read_text()
            )
            if saved != attribution:
                raise ValueError(
                    "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ATTRIBUTION_REPLAY_MISMATCH "
                    f"candidate={expected.candidate} mode={mode.value}"
                )
            records.append(
                _capture_record(
                    candidate_root=candidate_root,
                    expected=expected,
                    mode=mode,
                    xplane=xplane,
                    assessment=assessment,
                    attribution=attribution,
                )
            )
    return records[0], records[1]


def _validate(
    root: Path,
    trusted_search_contract: SeqaxWeightPlacementContract,
    trusted_contract: SeqaxWeightPlacementDiagnosticContract,
    *,
    require_accepted: bool,
) -> SeqaxWeightPlacementDiagnosticResult:
    _preflight_root(root)
    saved_contract = SeqaxWeightPlacementDiagnosticContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if saved_contract != trusted_contract:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_CONTRACT_MISMATCH")
    search_result = _validate_search_snapshot(
        root / "search",
        trusted_search_contract,
        replay_recorded=require_accepted,
    )
    result = SeqaxWeightPlacementDiagnosticResult.model_validate_json(
        (root / "result.json").read_text()
    )
    if (
        result.diagnostic_id != trusted_contract.diagnostic_id
        or result.search_id != trusted_contract.search_id
        or result.search_receipt_sha256 != trusted_contract.search_receipt_sha256
        or result.runtime != trusted_contract.runtime
        or result.devices != search_result.devices
        or tuple(value.id for value in result.devices) != tuple(range(8))
        or any(value.platform != "tpu" for value in result.devices)
        or any(value.device_kind not in {"TPU7x", "TPU v7x"} for value in result.devices)
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_RESULT_IDENTITY_MISMATCH")
    _validate_source(root, result)
    prepared = prepare_weight_placement_candidates(trusted_search_contract)
    prepared_by_name = {value.candidate.name: value for value in prepared}
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=trusted_contract.timing_seed,
            **trusted_search_contract.parameters,
        )
    )
    saved_inputs = tuple(
        np.load(root / "inputs" / f"{index:02d}.npy", allow_pickle=False)
        for index in range(len(host_inputs))
    )
    if any(
        saved.shape != expected.shape
        or saved.dtype != expected.dtype
        or not np.array_equal(saved, expected)
        for saved, expected in zip(saved_inputs, host_inputs, strict=True)
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_INPUT_REPLAY_MISMATCH")
    candidate_results = []
    profiles = []
    for expected, saved_result in zip(
        trusted_contract.candidates,
        result.candidates,
        strict=True,
    ):
        prepared_value = prepared_by_name[expected.candidate]
        candidate_root = root / "candidates" / expected.candidate
        _expected_plan_files(root, prepared_value, root / "search")
        stablehlo = (candidate_root / "stablehlo.txt").read_text()
        compiler_hlo = (candidate_root / "compiler_hlo.txt").read_text()
        all_gathers, reduce_scatters = _physical_collective_counts(prepared_value.physical)
        _validate_compiled_program(
            stablehlo,
            compiler_hlo,
            pallas_region_count=prepared_value.plan.pallas_region_count,
            pallas_vector_region_count=prepared_value.plan.pallas_vector_region_count,
            all_gather_count=all_gathers,
            reduce_scatter_count=reduce_scatters,
        )
        cost_report = _cost_report(root, prepared_value)
        saved_cost = SeqaxCostModelReport.model_validate_json(
            (candidate_root / "cost_model.json").read_text()
        )
        if (
            saved_cost != cost_report
            or int(_cost_metric(cost_report, "seqax_ici_bidirectional_bytes_per_device"))
            != expected.expected_ring_equivalent_ici_bytes_per_device
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_COST_REPLAY_MISMATCH candidate={expected.candidate}"
            )
        search_plan = next(
            value for value in search_result.plans if value.candidate is expected.candidate
        )
        expected_output = np.load(
            root
            / "search"
            / "correctness"
            / str(trusted_contract.timing_seed)
            / "outputs"
            / f"{expected.candidate}.npy",
            allow_pickle=False,
        )
        actual = np.load(candidate_root / "output.npy", allow_pickle=False)
        trace, counters = _replay_candidate_profiles(
            root=root,
            prepared=prepared_value,
            expected=expected,
            cost_report=cost_report,
        )
        replayed = SeqaxWeightPlacementDiagnosticCandidateResult(
            candidate=expected.candidate,
            distributed_schedule_sha256=prepared_value.plan.distributed_schedule_sha256,
            physical_schedule_sha256=prepared_value.plan.physical_schedule_sha256,
            pallas_source_sha256=prepared_value.plan.source_sha256(),
            stablehlo_sha256=_sha256(candidate_root / "stablehlo.txt"),
            compiler_hlo_sha256=_sha256(candidate_root / "compiler_hlo.txt"),
            cost_model_sha256=_sha256(candidate_root / "cost_model.json"),
            input_sha256=arrays_sha256(saved_inputs),
            output_sha256=array_sha256(actual),
            expected_output_sha256=array_sha256(expected_output),
            exact_search_output_parity=np.array_equal(actual, expected_output),
            trace=trace,
            counters=counters,
        )
        if (
            replayed != saved_result
            or search_plan.distributed_schedule_sha256 != replayed.distributed_schedule_sha256
            or search_plan.physical_schedule_sha256 != replayed.physical_schedule_sha256
            or search_plan.pallas_source_sha256 != replayed.pallas_source_sha256
            or search_plan.stablehlo_sha256 != replayed.stablehlo_sha256
            or search_plan.compiler_hlo_sha256 != replayed.compiler_hlo_sha256
            or actual.shape != expected_output.shape
            or actual.dtype != expected_output.dtype
            or not np.array_equal(actual, expected_output)
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_CANDIDATE_REPLAY_MISMATCH candidate={expected.candidate}"
            )
        candidate_results.append(replayed)
        profiles.append(
            SeqaxWeightPlacementCandidateProfiles(
                candidate=expected.candidate,
                trace=trace.summary,
                counters=counters.summary,
            )
        )
    comparison = compare_weight_placement_profiles(trusted_contract, tuple(profiles))
    if (
        result.comparison != comparison
        or json.loads((root / "comparison.json").read_text()) != comparison.model_dump(mode="json")
        or result.candidates != tuple(candidate_results)
        or result.correctness_scope != "incumbent-bit-exact-diagnostic"
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_COMPARISON_REPLAY_MISMATCH")
    run_id = semantic_sha256(
        SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA,
        trusted_contract.diagnostic_id,
        _sha256(root / "source_state.json"),
        _sha256(root / "source_manifest.json"),
    )
    if result.run_id != run_id:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_RUN_ID_MISMATCH")
    payloads = (
        (RunState.CREATED, {"diagnostic_id": trusted_contract.diagnostic_id}),
        (
            RunState.VERIFIED,
            {
                "search_receipt_sha256": trusted_contract.search_receipt_sha256,
                "distributed_schedules": {
                    value.candidate.name: value.plan.distributed_schedule_sha256
                    for value in prepared
                },
            },
        ),
        (
            RunState.LOWERED,
            {
                "pallas_sources": {
                    value.candidate.name: value.plan.source_sha256() for value in prepared
                }
            },
        ),
        (
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
        ),
        (
            RunState.CORRECT,
            {
                "input_sha256": arrays_sha256(saved_inputs),
                "output_sha256": {
                    value.candidate: value.output_sha256 for value in candidate_results
                },
            },
        ),
        (
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
        ),
    )
    if require_accepted:
        payloads += ((RunState.ACCEPTED, {"result_sha256": _sha256(root / "result.json")}),)
    history = read_ledger_history(root / "ledger.sqlite", run_id)
    if tuple(value.state for value in history) != tuple(value[0] for value in payloads):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_LEDGER_STATE_MISMATCH")
    if tuple(value.payload_sha256 for value in history) != tuple(
        payload_sha256(payload) for _state, payload in payloads
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_LEDGER_PAYLOAD_MISMATCH")
    if require_accepted:
        receipt = SeqaxWeightPlacementDiagnosticReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        _validate_manifest(root, receipt.artifacts)
        expected_receipt = SeqaxWeightPlacementDiagnosticReceipt(
            status="passed",
            diagnostic_id=trusted_contract.diagnostic_id,
            run_id=run_id,
            search_id=trusted_contract.search_id,
            result_sha256=_sha256(root / "result.json"),
            ledger_sha256=_sha256(root / "ledger.sqlite"),
            artifacts=_artifact_manifest(root),
        )
        if receipt != expected_receipt:
            raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_RECEIPT_MISMATCH")
    else:
        _validate_manifest(root, _artifact_manifest(root))
    return result


def run_seqax_weight_placement_diagnostic(
    root: Path,
    search_root: Path,
    trusted_search_contract: SeqaxWeightPlacementContract,
    contract: SeqaxWeightPlacementDiagnosticContract,
) -> SeqaxWeightPlacementDiagnosticResult:
    if root.is_symlink() or search_root.is_symlink():
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ROOT_SYMLINK")
    root = root.resolve()
    search_root = search_root.resolve()
    _require_safe_new_root(root, search_root)
    repository_root = Path(__file__).resolve().parents[2]
    _require_clean_repository(repository_root)
    runtime = _runtime_identity()
    if (
        runtime != contract.runtime
        or runtime != trusted_search_contract.runtime
        or contract != default_seqax_weight_placement_diagnostic_contract(runtime)
        or trusted_search_contract
        != default_seqax_weight_placement_contract(trusted_search_contract.runtime)
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_EXTERNAL_CONTRACT_MISMATCH")
    if (root / "receipt.json").is_file():
        return validate_seqax_weight_placement_diagnostic(
            root,
            trusted_search_contract,
            contract,
        )
    search_result = _validate_search_snapshot(
        search_root,
        trusted_search_contract,
        replay_recorded=True,
    )
    devices = tuple(jax.devices())
    _validate_devices(devices, trusted_search_contract)
    if _device_inventory(devices) != search_result.devices:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_DEVICE_INVENTORY_MISMATCH")
    _prepare_output_root(root, contract)
    _write_json(
        root / "contract.json", contract.model_dump(mode="json", exclude_computed_fields=True)
    )
    shutil.copytree(search_root, root / "search", copy_function=shutil.copy2)
    _source_state(repository_root, root)
    manifest = _source_manifest()
    _write_json(
        root / "source_manifest.json", [value.model_dump(mode="json") for value in manifest]
    )
    run_id = semantic_sha256(
        SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SCHEMA,
        contract.diagnostic_id,
        _sha256(root / "source_state.json"),
        _sha256(root / "source_manifest.json"),
    )
    ledger_path = root / "ledger.sqlite"
    evidence_run = EvidenceRun(ledger_path, run_id)
    evidence_run.create({"diagnostic_id": contract.diagnostic_id})
    prepared = prepare_weight_placement_candidates(trusted_search_contract)
    evidence_run.transition(
        RunState.VERIFIED,
        {
            "search_receipt_sha256": contract.search_receipt_sha256,
            "distributed_schedules": {
                value.candidate.name: value.plan.distributed_schedule_sha256 for value in prepared
            },
        },
    )
    for value in prepared:
        candidate_root = root / "candidates" / value.candidate.name
        _write_text(candidate_root / "distributed.xdsl", canonical_text(value.distributed))
        _write_text(candidate_root / "physical.xdsl", canonical_text(value.physical))
        _write_text(candidate_root / "lowered_pallas.py", value.plan.render_executable_source())
        _write_json(candidate_root / "plan_manifest.json", value.plan.manifest())
    evidence_run.transition(
        RunState.LOWERED,
        {
            "pallas_sources": {
                value.candidate.name: value.plan.source_sha256() for value in prepared
            }
        },
    )
    host_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=contract.timing_seed,
            **trusted_search_contract.parameters,
        )
    )
    for index, value in enumerate(host_inputs):
        _save_array(root / "inputs" / f"{index:02d}.npy", value)
    compiled = tuple(_compile(value, host_inputs, devices) for value in prepared)
    for value in compiled:
        candidate_root = root / "candidates" / value.prepared.candidate.name
        _write_text(candidate_root / "stablehlo.txt", value.stablehlo + "\n")
        _write_text(candidate_root / "compiler_hlo.txt", value.compiler_hlo + "\n")
        _expected_plan_files(root, value.prepared, root / "search")
    evidence_run.transition(
        RunState.COMPILED,
        {
            "compiled_hlo": {
                value.prepared.candidate.name: {
                    "stablehlo_sha256": _sha256(
                        root / "candidates" / value.prepared.candidate.name / "stablehlo.txt"
                    ),
                    "compiler_hlo_sha256": _sha256(
                        root / "candidates" / value.prepared.candidate.name / "compiler_hlo.txt"
                    ),
                }
                for value in compiled
            }
        },
    )
    executions = []
    outputs: dict[SeqaxWeightPlacementName, str] = {}
    for expected, value in zip(contract.candidates, compiled, strict=True):
        candidate_root = root / "candidates" / expected.candidate
        resident = _resident_inputs(host_inputs, value.prepared, value.mesh)
        actual = _execute(value, resident)
        expected_output = np.load(
            root
            / "search"
            / "correctness"
            / str(contract.timing_seed)
            / "outputs"
            / f"{expected.candidate}.npy",
            allow_pickle=False,
        )
        if (
            actual.shape != expected_output.shape
            or actual.dtype != expected_output.dtype
            or not np.array_equal(actual, expected_output)
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_OUTPUT_MISMATCH candidate={expected.candidate}"
            )
        _save_array(candidate_root / "output.npy", actual)
        outputs[expected.candidate] = array_sha256(actual)
        cost_report = _cost_report(root, value.prepared)
        if int(_cost_metric(cost_report, "seqax_ici_bidirectional_bytes_per_device")) != (
            expected.expected_ring_equivalent_ici_bytes_per_device
        ):
            raise ValueError(
                f"SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_COST_MISMATCH candidate={expected.candidate}"
            )
        _write_json(candidate_root / "cost_model.json", cost_report.model_dump(mode="json"))
        executions.append((expected, value, resident, actual, expected_output, cost_report))
    evidence_run.transition(
        RunState.CORRECT,
        {"input_sha256": arrays_sha256(host_inputs), "output_sha256": outputs},
    )
    candidate_results = []
    profiles = []
    for expected, value, resident, actual, expected_output, cost_report in executions:
        candidate_root = root / "candidates" / expected.candidate
        trace = _capture_candidate_phase(
            candidate_root=candidate_root,
            expected=expected,
            compiled=value,
            resident=resident,
            mode=RunMode.TRACE,
            cost_report=cost_report,
        )
        trace_output = _execute(value, resident)
        if not np.array_equal(trace_output, expected_output):
            raise ValueError(
                "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_POST_TRACE_OUTPUT_MISMATCH "
                f"candidate={expected.candidate}"
            )
        counters = _capture_candidate_phase(
            candidate_root=candidate_root,
            expected=expected,
            compiled=value,
            resident=resident,
            mode=RunMode.COUNTERS,
            cost_report=cost_report,
        )
        counter_output = _execute(value, resident)
        if not np.array_equal(counter_output, expected_output):
            raise ValueError(
                "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_POST_COUNTER_OUTPUT_MISMATCH "
                f"candidate={expected.candidate}"
            )
        result_value = SeqaxWeightPlacementDiagnosticCandidateResult(
            candidate=expected.candidate,
            distributed_schedule_sha256=value.prepared.plan.distributed_schedule_sha256,
            physical_schedule_sha256=value.prepared.plan.physical_schedule_sha256,
            pallas_source_sha256=value.prepared.plan.source_sha256(),
            stablehlo_sha256=_sha256(candidate_root / "stablehlo.txt"),
            compiler_hlo_sha256=_sha256(candidate_root / "compiler_hlo.txt"),
            cost_model_sha256=_sha256(candidate_root / "cost_model.json"),
            input_sha256=arrays_sha256(host_inputs),
            output_sha256=array_sha256(actual),
            expected_output_sha256=array_sha256(expected_output),
            exact_search_output_parity=True,
            trace=trace,
            counters=counters,
        )
        candidate_results.append(result_value)
        profiles.append(
            SeqaxWeightPlacementCandidateProfiles(
                candidate=expected.candidate,
                trace=trace.summary,
                counters=counters.summary,
            )
        )
    comparison = compare_weight_placement_profiles(contract, tuple(profiles))
    _write_json(root / "comparison.json", comparison.model_dump(mode="json"))
    result = SeqaxWeightPlacementDiagnosticResult(
        diagnostic_id=contract.diagnostic_id,
        run_id=run_id,
        search_id=contract.search_id,
        search_receipt_sha256=contract.search_receipt_sha256,
        runtime=runtime,
        devices=_device_inventory(devices),
        source_state_sha256=_sha256(root / "source_state.json"),
        source_manifest_sha256=_sha256(root / "source_manifest.json"),
        source_manifest=manifest,
        candidates=tuple(candidate_results),
        comparison=comparison,
        correctness_scope="incumbent-bit-exact-diagnostic",
    )
    _write_json(root / "result.json", result.model_dump(mode="json"))
    evidence_run.transition(
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
    _close_ledger(ledger_path)
    _validate(
        root,
        trusted_search_contract,
        contract,
        require_accepted=False,
    )
    evidence_run.transition(
        RunState.ACCEPTED,
        {"result_sha256": _sha256(root / "result.json")},
    )
    _close_ledger(ledger_path)
    receipt = SeqaxWeightPlacementDiagnosticReceipt(
        status="passed",
        diagnostic_id=contract.diagnostic_id,
        run_id=run_id,
        search_id=contract.search_id,
        result_sha256=_sha256(root / "result.json"),
        ledger_sha256=_sha256(ledger_path),
        artifacts=_artifact_manifest(root),
    )
    _write_json(root / "receipt.json", receipt.model_dump(mode="json"))
    return validate_seqax_weight_placement_diagnostic(
        root,
        trusted_search_contract,
        contract,
    )


def validate_seqax_weight_placement_diagnostic(
    root: Path,
    trusted_search_contract: SeqaxWeightPlacementContract,
    trusted_contract: SeqaxWeightPlacementDiagnosticContract,
) -> SeqaxWeightPlacementDiagnosticResult:
    if root.is_symlink():
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ROOT_SYMLINK")
    if trusted_search_contract != default_seqax_weight_placement_contract(
        trusted_search_contract.runtime
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_SEARCH_CONTRACT_INVALID")
    if trusted_contract != default_seqax_weight_placement_diagnostic_contract(
        trusted_contract.runtime
    ):
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_EXTERNAL_CONTRACT_INVALID")
    if trusted_search_contract.runtime != trusted_contract.runtime:
        raise ValueError("SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_RUNTIME_CONTRACT_MISMATCH")
    return _validate(
        root.resolve(),
        trusted_search_contract,
        trusted_contract,
        require_accepted=True,
    )
