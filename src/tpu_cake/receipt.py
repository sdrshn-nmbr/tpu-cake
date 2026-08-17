from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

from tpu_cake.artifacts import resolve_bundle_artifact, resolve_recorded_artifact
from tpu_cake.contracts import (
    ArtifactReference,
    ArtifactRole,
    KernelExperiment,
    ProfileExpectation,
    RunReceipt,
    RunStatus,
)
from tpu_cake.cost_model import (
    CostModelReport,
    MatmulCostModelInput,
    estimate_distributed_matmul_input,
    tpu7x_tensorcore_rates,
)
from tpu_cake.identity import (
    LEGACY_SEMANTIC_IDENTITY_SCHEMA,
    SEMANTIC_IDENTITY_SCHEMA,
    array_sha256,
    semantic_sha256,
)
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.pallas_lowering import (
    PALLAS_EXECUTION_SCHEMA,
    PallasMatmulPlan,
    validate_saved_pallas_plan,
)
from tpu_cake.receipt_metrics import build_receipt_metrics
from tpu_cake.runner import MatmulRunResult, RunMode, validate_profiler_contract
from tpu_cake.search import (
    MatmulSearchContract,
    MatmulSearchResult,
    validate_matmul_search_result,
)
from tpu_cake.workloads.distributed_matmul import distributed_matmul_experiment
from tpu_cake.xprof_evidence import assess_capture


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counter_expectation(experiment: KernelExperiment) -> ProfileExpectation:
    contract = experiment.profile.model_dump()
    contract.update(
        require_tensor_core_activity=False,
        require_hbm_read_counters=True,
        require_hbm_write_counters=True,
        require_cycle_counters=True,
        minimum_counter_device_planes=experiment.target.chip_count,
        required_timed_hlo_markers=tuple(
            marker
            for marker in experiment.profile.required_timed_hlo_markers
            if marker != "pallas_call"
        ),
    )
    return ProfileExpectation.model_validate(contract)


def _relative_json(value: object, root: Path) -> object:
    if isinstance(value, dict):
        return {key: _relative_json(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_json(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), ".")
    return value


def _phase_artifact(
    receipt: RunReceipt,
    role: ArtifactRole,
    phase: str,
    *,
    name: str | None = None,
) -> ArtifactReference:
    matches = [
        artifact
        for artifact in receipt.artifacts
        if artifact.role is role
        and Path(artifact.path).parts[0] == phase
        and (name is None or Path(artifact.path).name == name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"RECEIPT_PHASE_ARTIFACT_MISMATCH phase={phase} role={role.value} name={name}"
        )
    return matches[0]


def _resolve_result_artifact(
    root: Path,
    phase: str,
    artifact: ArtifactReference,
) -> Path:
    declared = Path(artifact.path)
    if declared.parts[0] == phase:
        return resolve_recorded_artifact(
            root,
            declared.as_posix(),
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
    return resolve_recorded_artifact(
        root / phase,
        declared.as_posix(),
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
    )


def _source_identity(
    state_path: Path, diff_path: Path, *, require_clean: bool = True
) -> tuple[str, str]:
    state = json.loads(state_path.read_text())
    commit = state.get("git_commit")
    lock = state.get("uv_lock_sha256")
    if (
        not isinstance(state.get("git_dirty"), bool)
        or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(lock, str)
        or re.fullmatch(r"[0-9a-f]{64}", lock) is None
        or state.get("source_diff_sha256") != _sha256(diff_path)
        or not isinstance(state.get("git_status"), list)
        or (state.get("git_dirty") is False and bool(state["git_status"]))
        or (state.get("git_dirty") is False and bool(diff_path.read_bytes()))
        or (require_clean and state.get("git_dirty") is not False)
    ):
        raise ValueError(f"SOURCE_STATE_INVALID path={state_path}")
    return commit, lock


def _validate_invocation_schemas(invocation: dict[str, object], phase: str) -> str:
    identity_schema = invocation.get(
        "identity_schema", LEGACY_SEMANTIC_IDENTITY_SCHEMA
    )
    if not isinstance(identity_schema, str):
        raise TypeError(f"RUN_IDENTITY_SCHEMA_INVALID phase={phase}")
    if "identity_schema" in invocation and identity_schema != SEMANTIC_IDENTITY_SCHEMA:
        raise ValueError(f"RUN_IDENTITY_SCHEMA_UNSUPPORTED phase={phase}")
    if (
        "pallas_execution_schema" in invocation
        and invocation["pallas_execution_schema"] != PALLAS_EXECUTION_SCHEMA
    ):
        raise ValueError(f"RUN_PALLAS_EXECUTION_SCHEMA_UNSUPPORTED phase={phase}")
    return identity_schema


def _validate_result_artifact_bindings(
    root: Path,
    receipt: RunReceipt,
    phase: str,
    result: MatmulRunResult,
) -> dict[str, Path]:
    receipt_artifacts = {artifact.path: artifact for artifact in receipt.artifacts}
    resolved: dict[str, Path] = {}
    for artifact in result.artifacts:
        path = _resolve_result_artifact(root, phase, artifact)
        relative = str(path.resolve().relative_to(root.resolve()))
        receipt_artifact = receipt_artifacts.get(relative)
        if receipt_artifact is None:
            raise ValueError(
                f"RESULT_ARTIFACT_NOT_BOUND_BY_RECEIPT phase={phase} path={relative}"
            )
        if (
            receipt_artifact.role is not artifact.role
            or receipt_artifact.size_bytes != artifact.size_bytes
            or receipt_artifact.sha256 != artifact.sha256
        ):
            raise ValueError(
                f"RESULT_ARTIFACT_RECEIPT_MISMATCH phase={phase} path={relative}"
            )
        if path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise ValueError(f"RESULT_ARTIFACT_CONTENT_MISMATCH phase={phase} path={relative}")
        resolved[path.name] = path
    return resolved


def _validate_saved_matmul_phase(
    root: Path,
    receipt: RunReceipt,
    experiment: KernelExperiment,
    phase: str,
    result: MatmulRunResult,
) -> tuple[float, float, PallasMatmulPlan]:
    artifacts = _validate_result_artifact_bindings(root, receipt, phase, result)
    required = {
        "invocation.json",
        "profiler_config.json",
        "source_state.json",
        "source_diff.patch",
        "experiment.json",
        "distributed.xdsl",
        "physical.xdsl",
        "lowered_pallas.py",
        "stablehlo.txt",
        "compiler_hlo.txt",
        "lhs.npy",
        "rhs.npy",
        "output.npy",
        "oracle.npy",
        "ledger.sqlite",
    }
    if not required <= artifacts.keys():
        raise ValueError(
            f"RESULT_ARTIFACT_SET_INCOMPLETE phase={phase} missing={sorted(required-artifacts.keys())}"
        )
    if _sha256(artifacts["physical.xdsl"]) != result.schedule_sha256:
        raise ValueError(f"PHYSICAL_IR_SCHEDULE_MISMATCH phase={phase}")
    if _sha256(artifacts["lowered_pallas.py"]) != result.pallas_source_sha256:
        raise ValueError(f"PALLAS_SOURCE_IDENTITY_MISMATCH phase={phase}")

    saved_plan = validate_saved_pallas_plan(
        artifacts["physical.xdsl"],
        artifacts["lowered_pallas.py"],
        schedule_sha256=result.schedule_sha256,
        pallas_source_sha256=result.pallas_source_sha256,
    )

    invocation = json.loads(artifacts["invocation.json"].read_text())
    identity_schema = _validate_invocation_schemas(invocation, phase)
    if invocation.get("mode") != result.mode.value:
        raise ValueError(f"RUN_INVOCATION_MODE_MISMATCH phase={phase}")
    expected_fields = {
        "mesh_size": result.device_count,
        "warmup_iterations": result.warmup_iterations,
        "measured_iterations": result.measured_iterations,
    }
    if any(invocation.get(name) != value for name, value in expected_fields.items()):
        raise ValueError(f"RUN_INVOCATION_RESULT_MISMATCH phase={phase}")
    invocation_tile_m = invocation["tile_m"] or saved_plan.partial_local_shape[0]
    invocation_tile_n = invocation["tile_n"] or saved_plan.partial_local_shape[1]
    if (
        saved_plan.mesh_size != invocation["mesh_size"]
        or saved_plan.tile_m != invocation_tile_m
        or saved_plan.tile_n != invocation_tile_n
    ):
        raise ValueError(f"RUN_INVOCATION_PLAN_MISMATCH phase={phase}")
    validate_profiler_contract(
        result.mode, json.loads(artifacts["profiler_config.json"].read_text())
    )
    _source_identity(
        artifacts["source_state.json"],
        artifacts["source_diff.patch"],
        require_clean=receipt.status is RunStatus.PASSED,
    )

    expected_experiment = distributed_matmul_experiment(
        schedule_sha256=result.schedule_sha256,
        mesh_size=int(invocation["mesh_size"]),
        m=int(invocation["m"]),
        k=int(invocation["k"]),
        n=int(invocation["n"]),
        warmup_iterations=result.warmup_iterations,
        measured_iterations=result.measured_iterations,
    )
    saved_experiment = KernelExperiment.model_validate_json(
        artifacts["experiment.json"].read_text()
    )
    if saved_experiment != expected_experiment:
        raise ValueError(f"RUN_EXPERIMENT_MISMATCH phase={phase}")
    if phase == "timing" and saved_experiment != experiment:
        raise ValueError("TIMING_EXPERIMENT_DOES_NOT_MATCH_RECEIPT_EXPERIMENT")

    lhs = np.load(artifacts["lhs.npy"], allow_pickle=False)
    rhs = np.load(artifacts["rhs.npy"], allow_pickle=False)
    output = np.load(artifacts["output.npy"], allow_pickle=False)
    oracle = np.load(artifacts["oracle.npy"], allow_pickle=False)
    if lhs.shape != (invocation["m"], invocation["k"]) or rhs.shape != (
        invocation["k"],
        invocation["n"],
    ):
        raise ValueError(f"RUN_INPUT_SHAPE_MISMATCH phase={phase}")
    if output.shape != (invocation["m"], invocation["n"]) or oracle.shape != output.shape:
        raise ValueError(f"RUN_OUTPUT_SHAPE_MISMATCH phase={phase}")
    if (
        array_sha256(lhs) != result.lhs_sha256
        or array_sha256(rhs) != result.rhs_sha256
        or array_sha256(output) != result.output_sha256
    ):
        raise ValueError(f"RUN_ARRAY_IDENTITY_MISMATCH phase={phase}")
    absolute = np.abs(output - oracle)
    denominator = np.maximum(np.abs(oracle), np.finfo(np.float32).tiny)
    maximum_absolute_error = float(absolute.max())
    maximum_relative_error = float((absolute / denominator).max())
    if not math.isclose(
        maximum_absolute_error, result.maximum_absolute_error, rel_tol=0, abs_tol=1e-12
    ) or not math.isclose(
        maximum_relative_error, result.maximum_relative_error, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"RUN_REPORTED_ERROR_MISMATCH phase={phase}")
    passed = bool(
        np.allclose(
            output,
            oracle,
            atol=experiment.workload.numerical.absolute_tolerance,
            rtol=experiment.workload.numerical.relative_tolerance,
        )
    )
    if result.passed is not passed:
        raise ValueError(f"RUN_CORRECTNESS_VERDICT_MISMATCH phase={phase}")

    expected_run_id = semantic_sha256(
        "distributed-matmul-run",
        result.mode.value,
        str(invocation["mesh_size"]),
        str(invocation["m"]),
        str(invocation["k"]),
        str(invocation["n"]),
        str(invocation["tile_m"]),
        str(invocation["tile_n"]),
        schema=identity_schema,
    )
    if result.run_id != expected_run_id:
        raise ValueError(f"RUN_ID_MISMATCH phase={phase}")

    expected_states = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        {
            RunMode.TIMING: RunState.TIMED,
            RunMode.TRACE: RunState.TRACED,
            RunMode.COUNTERS: RunState.COUNTERED,
        }[result.mode],
    )
    terminal_payload: dict[str, object] = {
        "measured_iterations": result.measured_iterations,
        "warmup_iterations": result.warmup_iterations,
    }
    if result.mode is RunMode.TIMING:
        samples = list(result.samples_ns)
        median = int(statistics.median(samples))
        ordered = sorted(samples)
        p90 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))]
        coefficient = (
            statistics.pstdev(samples) / statistics.mean(samples)
            if len(samples) > 1 and statistics.mean(samples)
            else None
        )
        if (
            result.median_ns != median
            or result.p90_ns != p90
            or result.coefficient_of_variation is None
            or coefficient is None
            or not math.isclose(
                result.coefficient_of_variation,
                coefficient,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("RUN_TIMING_STATISTIC_MISMATCH phase=timing")
        terminal_payload.update(
            median_ns=result.median_ns,
            p90_ns=result.p90_ns,
            sample_count=len(samples),
        )
    else:
        if result.samples_ns or any(
            value is not None
            for value in (result.median_ns, result.p90_ns, result.coefficient_of_variation)
        ):
            raise ValueError(f"PROFILE_RUN_CONTAINS_TIMING_CLAIMS phase={phase}")
        xplanes = sorted((root / phase / "profile").rglob("*.xplane.pb"))
        if len(xplanes) != 1:
            raise ValueError(f"PROFILE_XPLANE_COUNT_MISMATCH phase={phase}")
        terminal_payload.update(
            xplane_sha256=_sha256(xplanes[0]),
            xplane_size_bytes=xplanes[0].stat().st_size,
        )
    created_payload = {
            "mode": result.mode.value,
            "mesh_size": invocation["mesh_size"],
            "m": invocation["m"],
            "k": invocation["k"],
            "n": invocation["n"],
            "tile_m": invocation["tile_m"],
            "tile_n": invocation["tile_n"],
        }
    if "identity_schema" in invocation:
        created_payload["identity_schema"] = identity_schema
    if "pallas_execution_schema" in invocation:
        created_payload["pallas_execution_schema"] = invocation[
            "pallas_execution_schema"
        ]
    expected_payloads = (
        created_payload,
        {"distributed_ir_sha256": _sha256(artifacts["distributed.xdsl"])},
        {
            "physical_ir_sha256": result.schedule_sha256,
            "schedule_sha256": result.schedule_sha256,
            "pallas_source_sha256": result.pallas_source_sha256,
            **(
                {"pallas_execution_schema": invocation["pallas_execution_schema"]}
                if "pallas_execution_schema" in invocation
                else {}
            ),
        },
        {
            "stablehlo_sha256": _sha256(artifacts["stablehlo.txt"]),
            "compiler_hlo_sha256": _sha256(artifacts["compiler_hlo.txt"]),
            "compile_duration_ns": result.compile_duration_ns,
        },
        {
            "lhs_sha256": result.lhs_sha256,
            "rhs_sha256": result.rhs_sha256,
            "output_sha256": result.output_sha256,
            "oracle_sha256": array_sha256(oracle),
        },
        terminal_payload,
    )
    history = read_ledger_history(artifacts["ledger.sqlite"], result.run_id)
    if tuple(event.state for event in history) != expected_states or tuple(
        event.payload_sha256 for event in history
    ) != tuple(ExperimentLedger.payload_sha256(payload) for payload in expected_payloads):
        raise ValueError(f"RUN_LEDGER_EVIDENCE_MISMATCH phase={phase}")
    return maximum_absolute_error, maximum_relative_error, saved_plan


def _validate_cost_model(
    root: Path,
    receipt: RunReceipt,
    saved_plan: PallasMatmulPlan,
) -> CostModelReport:
    input_artifact = _phase_artifact(
        receipt, ArtifactRole.COST_MODEL_INPUT, "timing", name="cost_model_input.json"
    )
    report_artifact = _phase_artifact(
        receipt, ArtifactRole.COST_MODEL, "timing", name="cost_model.json"
    )
    input_path = root / input_artifact.path
    report_path = root / report_artifact.path
    model_input = MatmulCostModelInput.model_validate_json(input_path.read_text())
    expected_input = MatmulCostModelInput(
        schedule_sha256=saved_plan.schedule_sha256,
        mesh_size=saved_plan.mesh_size,
        m=saved_plan.global_lhs_shape[0],
        k=saved_plan.global_lhs_shape[1],
        n=saved_plan.global_rhs_shape[1],
        tile_m=saved_plan.tile_m,
        tile_k=saved_plan.tile_k,
        tile_n=saved_plan.tile_n,
        collective_link_bandwidths=saved_plan.collective_link_bandwidths,
        hardware=tpu7x_tensorcore_rates(),
    )
    if model_input != expected_input:
        raise ValueError("COST_MODEL_INPUT_DOES_NOT_MATCH_SAVED_PLAN")
    report = CostModelReport.model_validate_json(report_path.read_text())
    sources = {source for metric in report.metrics for source in metric.sources}
    if len(sources) != 1:
        raise ValueError("COST_MODEL_NEEDS_ONE_CANONICAL_INPUT_SOURCE")
    source = next(iter(sources))
    if (
        source.artifact_sha256 != input_artifact.sha256
        or Path(source.artifact_path).name != input_path.name
        or source.tool != "tpu-cake"
        or source.field != "distributed-matmul-v1"
    ):
        raise ValueError("COST_MODEL_SOURCE_MISMATCH")
    expected = estimate_distributed_matmul_input(model_input, source=source)
    if report != expected:
        raise ValueError("COST_MODEL_REPORT_DOES_NOT_MATCH_INPUT")
    return report


def _validate_roofline(
    root: Path,
    receipt: RunReceipt,
    timing: MatmulRunResult,
    cost_report: CostModelReport,
) -> None:
    input_artifact = next(
        artifact for artifact in receipt.artifacts if artifact.role is ArtifactRole.ROOFLINE_INPUT
    )
    metrics_artifact = next(
        artifact for artifact in receipt.artifacts if artifact.role is ArtifactRole.ROOFLINE_METRICS
    )
    model_input = json.loads((root / input_artifact.path).read_text())
    metrics = json.loads((root / metrics_artifact.path).read_text())
    hardware = model_input["hardware"]
    workload = model_input["workload"]
    operations = float(workload["operations"])
    total_bytes = float(workload["bytes_read"] + workload["bytes_written"])
    if operations != cost_report.counts.operations_per_device or total_bytes != (
        cost_report.counts.hbm_read_bytes_per_device
        + cost_report.counts.hbm_write_bytes_per_device
    ):
        raise ValueError("ROOFLINE_WORKLOAD_DOES_NOT_MATCH_COST_MODEL")
    if (
        float(hardware["compute_peak_ops_s"])
        != cost_report.hardware.compute_flops_per_second
        or float(hardware["memory_bandwidth_bytes_s"])
        != cost_report.hardware.hbm_bytes_per_second
    ):
        raise ValueError("ROOFLINE_HARDWARE_DOES_NOT_MATCH_COST_MODEL")
    compute_peak = float(hardware["compute_peak_ops_s"]) * float(
        hardware["effective_compute_fraction"]
    )
    memory_peak = float(hardware["memory_bandwidth_bytes_s"]) * float(
        hardware["effective_memory_fraction"]
    )
    measured = float(workload["measured_time_s"])
    expected = {
        "operations": operations,
        "total_bytes": total_bytes,
        "arithmetic_intensity_ops_per_byte": operations / total_bytes,
        "effective_compute_peak_ops_s": compute_peak,
        "effective_memory_bandwidth_bytes_s": memory_peak,
        "compute_time_s": operations / compute_peak,
        "memory_time_s": total_bytes / memory_peak,
        "lower_bound_time_s": max(operations / compute_peak, total_bytes / memory_peak),
        "measured_time_s": measured,
        "measured_throughput_ops_s": operations / measured,
    }
    expected["attainable_throughput_ops_s"] = min(
        compute_peak, memory_peak * expected["arithmetic_intensity_ops_per_byte"]
    )
    expected["attainable_efficiency"] = (
        expected["measured_throughput_ops_s"] / expected["attainable_throughput_ops_s"]
    )
    expected["peak_efficiency"] = expected["measured_throughput_ops_s"] / compute_peak
    expected["headroom_to_attainable"] = (
        expected["attainable_throughput_ops_s"] / expected["measured_throughput_ops_s"]
    )
    for name, value in expected.items():
        if name not in metrics or not math.isclose(
            float(metrics[name]), value, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"ROOFLINE_METRIC_MISMATCH field={name}")
    if not math.isclose(measured, timing.median_ns / 1_000_000_000, rel_tol=0, abs_tol=1e-15):
        raise ValueError("ROOFLINE_TIMING_SOURCE_MISMATCH")


def _validate_distributed_matmul_receipt(
    receipt: RunReceipt,
    experiment: KernelExperiment,
    root: Path,
) -> None:
    result_specs = (
        ("timing", ArtifactRole.TIMING_SAMPLES, RunMode.TIMING),
        ("trace", ArtifactRole.TRACE_RESULT, RunMode.TRACE),
        ("counters", ArtifactRole.COUNTER_RESULT, RunMode.COUNTERS),
    )
    results: list[MatmulRunResult] = []
    errors: list[tuple[float, float]] = []
    saved_plans: list[PallasMatmulPlan] = []
    for phase, role, mode in result_specs:
        artifact = _phase_artifact(receipt, role, phase, name="result.json")
        result = MatmulRunResult.model_validate_json((root / artifact.path).read_text())
        if result.mode is not mode:
            raise ValueError(f"RUN_MODE_MISMATCH phase={phase}")
        maximum_absolute, maximum_relative, saved_plan = _validate_saved_matmul_phase(
            root, receipt, experiment, phase, result
        )
        errors.append((maximum_absolute, maximum_relative))
        saved_plans.append(saved_plan)
        results.append(result)
    if any(plan != saved_plans[0] for plan in saved_plans[1:]):
        raise ValueError("RUN_PHASE_LOWERED_PLANS_DO_NOT_MATCH")
    source_identities = {
        _source_identity(
            root
            / _phase_artifact(
                receipt, ArtifactRole.SOURCE_STATE, phase, name="source_state.json"
            ).path,
            root
            / _phase_artifact(
                receipt, ArtifactRole.SOURCE_DIFF, phase, name="source_diff.patch"
            ).path,
        )
        for phase, _, _ in result_specs
    }
    if len(source_identities) != 1:
        raise ValueError("RUN_PHASE_SOURCE_IDENTITIES_DO_NOT_MATCH")
    finalizer_identity = _source_identity(
        root
        / _phase_artifact(
            receipt, ArtifactRole.SOURCE_STATE, "finalizer", name="source_state.json"
        ).path,
        root
        / _phase_artifact(
            receipt, ArtifactRole.SOURCE_DIFF, "finalizer", name="source_diff.patch"
        ).path,
    )
    if finalizer_identity[1] != next(iter(source_identities))[1]:
        raise ValueError("FINALIZER_LOCK_IDENTITY_DOES_NOT_MATCH_RUNS")
    if len(
        {
            (
                result.schedule_sha256,
                result.pallas_source_sha256,
                result.lhs_sha256,
                result.rhs_sha256,
                result.output_sha256,
                result.runtime,
            )
            for result in results
        }
    ) != 1:
        raise ValueError("RUN_PHASE_IDENTITIES_DO_NOT_MATCH")
    maximum_absolute = max(value[0] for value in errors)
    maximum_relative = max(value[1] for value in errors)
    if (
        receipt.correctness.oracle != experiment.workload.numerical.reference
        or receipt.correctness.maximum_absolute_error is None
        or receipt.correctness.maximum_relative_error is None
        or not math.isclose(
            receipt.correctness.maximum_absolute_error,
            maximum_absolute,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            receipt.correctness.maximum_relative_error,
            maximum_relative,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("RECEIPT_CORRECTNESS_CLAIM_MISMATCH")

    trace_assessment = assess_capture(root / "trace", experiment.profile)
    counter_assessment = assess_capture(root / "counters", counter_expectation(experiment))
    if not trace_assessment.accepted or not counter_assessment.accepted:
        raise ValueError("RECEIPT_PROFILE_EVIDENCE_REJECTED")
    assessment_artifact = next(
        artifact
        for artifact in receipt.artifacts
        if artifact.role is ArtifactRole.PROFILE_ASSESSMENT
    )
    saved_assessment = json.loads((root / assessment_artifact.path).read_text())
    expected_assessment = _relative_json(
        {
            "timing_trace": trace_assessment.model_dump(mode="json"),
            "counter_trace": counter_assessment.model_dump(mode="json"),
        },
        root,
    )
    if saved_assessment != expected_assessment:
        raise ValueError("PROFILE_ASSESSMENT_DOES_NOT_MATCH_RAW_CAPTURE")

    cost_report = _validate_cost_model(root, receipt, saved_plans[0])
    _validate_roofline(root, receipt, results[0], cost_report)
    expected_metrics = build_receipt_metrics(
        root,
        results[0],
        cost_report,
        trace_assessment,
        counter_assessment,
    )
    if receipt.metrics != expected_metrics:
        raise ValueError("RECEIPT_METRICS_DO_NOT_MATCH_EVIDENCE")


def validate_receipt(
    receipt: RunReceipt,
    experiment: KernelExperiment,
    *,
    root: Path | None = None,
) -> None:
    if receipt.experiment_id != experiment.experiment_id:
        raise ValueError("receipt experiment identity does not match the experiment")
    if receipt.schedule_sha256 != experiment.schedule_sha256:
        raise ValueError("receipt schedule identity does not match the experiment")
    required_properties = tuple(experiment.workload.numerical.semantic_properties)
    if receipt.required_semantic_properties != required_properties:
        raise ValueError("receipt semantic requirements do not match the experiment")
    if receipt.status is not RunStatus.PASSED:
        return
    if root is None:
        raise ValueError("passed receipt validation requires its bundle root")
    root = root.resolve()
    for artifact in receipt.artifacts:
        path = resolve_bundle_artifact(root, artifact.path)
        if not path.is_file():
            raise ValueError(f"receipt artifact is missing: {path}")
        if path.stat().st_size != artifact.size_bytes:
            raise ValueError(f"receipt artifact size changed: {path}")
        if _sha256(path) != artifact.sha256:
            raise ValueError(f"receipt artifact hash changed: {path}")
    artifacts_by_path = {artifact.path: artifact for artifact in receipt.artifacts}
    for metric in receipt.metrics:
        for source in metric.sources:
            artifact = artifacts_by_path.get(source.artifact_path)
            if artifact is None or artifact.sha256 != source.artifact_sha256:
                raise ValueError(
                    f"metric source is not bound to a receipt artifact: {source.artifact_path}"
                )
    if experiment.workload.name.startswith("distributed-matmul-"):
        _validate_distributed_matmul_receipt(receipt, experiment, root)
    provenance = receipt.search_provenance
    if provenance is None:
        return
    by_role = {
        role: [artifact for artifact in receipt.artifacts if artifact.role is role]
        for role in (ArtifactRole.SEARCH_CONTRACT, ArtifactRole.SEARCH_RESULT)
    }
    if any(len(artifacts) != 1 for artifacts in by_role.values()):
        raise ValueError("search provenance requires one contract and one result artifact")
    contract_artifact = by_role[ArtifactRole.SEARCH_CONTRACT][0]
    result_artifact = by_role[ArtifactRole.SEARCH_RESULT][0]

    contract_path = resolve_bundle_artifact(root, contract_artifact.path)
    result_path = resolve_bundle_artifact(root, result_artifact.path)
    if provenance.contract_sha256 != contract_artifact.sha256:
        raise ValueError("search contract identity does not match provenance")
    if provenance.result_sha256 != result_artifact.sha256:
        raise ValueError("search result identity does not match provenance")
    contract = MatmulSearchContract.model_validate_json(contract_path.read_text())
    result = MatmulSearchResult.model_validate_json(result_path.read_text())
    if provenance.search_id != contract.search_id or result.search_id != contract.search_id:
        raise ValueError("search identity does not match provenance")
    validate_matmul_search_result(
        contract_path.parent,
        contract,
        result,
        recompute_schedules=False,
    )
    if result.winner != provenance.winner or len(result.run_results) != provenance.run_count:
        raise ValueError("search winner or run count does not match provenance")
    candidate = next(
        (candidate for candidate in contract.candidates if candidate.name == result.winner),
        None,
    )
    if candidate is None or (candidate.tile_m, candidate.tile_n) != (
        provenance.tile_m,
        provenance.tile_n,
    ):
        raise ValueError("search winner tile does not match provenance")
    winner_schedules = {
        MatmulRunResult.model_validate_json(
            (contract_path.parent / run_path / "result.json").read_text()
        ).schedule_sha256
        for run_path in result.run_results
        if Path(run_path).name == result.winner
    }
    if winner_schedules != {provenance.winner_schedule_sha256}:
        raise ValueError("search winner schedule does not match provenance")
    if provenance.winner_schedule_sha256 != receipt.schedule_sha256:
        raise ValueError("search winner schedule does not match finalist receipt")
