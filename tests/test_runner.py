from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tpu_cake.contracts import (
    ArtifactReference,
    CorrectnessResult,
    KernelExperiment,
    RunReceipt,
    RuntimeIdentity,
)
from tpu_cake.ledger import ExperimentLedger, RunState
from tpu_cake.receipt import (
    _source_identity,
    _validate_cost_model,
    _validate_invocation_schemas,
    _validate_saved_matmul_phase,
)
from tpu_cake.runner import (
    MatmulCollectiveStrategy,
    MatmulRunResult,
    RunMode,
    _profiler_contract,
    validate_profiler_contract,
)
from tpu_cake.workloads.distributed_matmul import distributed_matmul_experiment


def test_counter_profiler_contract_retains_counter_only_options() -> None:
    timing = _profiler_contract(RunMode.TRACE)["advanced_configuration"]
    counters = _profiler_contract(RunMode.COUNTERS)["advanced_configuration"]

    assert "tpu_enable_periodic_counter_sampling" not in timing
    assert counters["tpu_enable_periodic_counter_sampling"] is True
    assert counters["num_tensor_cores_to_trace_per_device"] == 1
    assert "tpu_tc_perf_counter_sampling_options" in counters


def test_source_identity_rejects_a_clean_claim_with_a_nonempty_diff(tmp_path) -> None:
    diff = tmp_path / "source_diff.patch"
    diff.write_text("modified source\n")
    state = tmp_path / "source_state.json"
    state.write_text(
        json.dumps(
            {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "uv_lock_sha256": "b" * 64,
                "source_diff_sha256": hashlib.sha256(diff.read_bytes()).hexdigest(),
            }
        )
    )

    with pytest.raises(ValueError, match="SOURCE_STATE_INVALID"):
        _source_identity(state, diff)


def test_source_identity_rejects_a_clean_claim_with_nonempty_status(tmp_path) -> None:
    diff = tmp_path / "source_diff.patch"
    diff.write_text("")
    state = tmp_path / "source_state.json"
    state.write_text(
        json.dumps(
            {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "git_status": ["?? injected.py"],
                "uv_lock_sha256": "b" * 64,
                "source_diff_sha256": hashlib.sha256(diff.read_bytes()).hexdigest(),
            }
        )
    )

    with pytest.raises(ValueError, match="SOURCE_STATE_INVALID"):
        _source_identity(state, diff)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("identity_schema", "separator-v1", "RUN_IDENTITY_SCHEMA_UNSUPPORTED"),
        (
            "pallas_execution_schema",
            "standalone-rendering-v1",
            "RUN_PALLAS_EXECUTION_SCHEMA_UNSUPPORTED",
        ),
    ),
)
def test_explicit_legacy_schemas_are_rejected(field: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_invocation_schemas({field: value}, "timing")


def test_absent_schema_fields_select_the_legacy_reader() -> None:
    assert _validate_invocation_schemas({}, "timing") == "separator-v1"


@pytest.mark.parametrize(
    "field",
    (
        "tpu_enable_periodic_counter_sampling",
        "tpu_tc_perf_counter_sampling_options",
        "num_tensor_cores_to_trace_per_device",
    ),
)
def test_counter_profiler_contract_mutations_fail_closed(field: str) -> None:
    contract = copy.deepcopy(_profiler_contract(RunMode.COUNTERS))
    del contract["advanced_configuration"][field]
    with pytest.raises(ValueError, match="COUNTER_PROFILER_FIELDS_MISSING"):
        validate_profiler_contract(RunMode.COUNTERS, contract)


def test_trace_contract_rejects_counter_configuration_leakage() -> None:
    contract = copy.deepcopy(_profiler_contract(RunMode.TRACE))
    contract["advanced_configuration"]["tpu_enable_periodic_counter_sampling"] = True
    with pytest.raises(ValueError, match="MUST_NOT_ENABLE_PERIODIC_COUNTERS"):
        validate_profiler_contract(RunMode.TRACE, contract)


def test_owned_collective_profile_requires_its_kernel_and_forbids_xla_reduce_scatter() -> None:
    experiment = distributed_matmul_experiment(
        schedule_sha256="a" * 64,
        mesh_size=8,
        m=1024,
        k=65536,
        n=1024,
        warmup_iterations=5,
        measured_iterations=100,
        collective_strategy="pallas_bidirectional_ring",
    )

    assert "distributed_matmul_physical_pallas_reduce_scatter" in (
        experiment.profile.required_timed_hlo_markers
    )
    assert "reduce-scatter" not in experiment.profile.required_timed_hlo_markers
    assert experiment.profile.forbidden_timed_hlo_fragments == ("reduce-scatter(",)


def test_timing_runner_writes_replayable_artifacts(tmp_path) -> None:
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    output = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tpu_cake.cli",
            "run-matmul",
            "--output-dir",
            str(output),
            "--mode",
            "timing",
            "--mesh-size",
            "4",
            "--m",
            "16",
            "--k",
            "32",
            "--n",
            "16",
            "--warmup-iterations",
            "1",
            "--measured-iterations",
            "3",
            "--interpret",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] is True
    assert len(result["samples_ns"]) == 3
    assert result["median_ns"] > 0
    assert (output / "distributed.xdsl").exists()
    assert (output / "physical.xdsl").exists()
    assert (output / "lowered_pallas.py").exists()
    assert (output / "stablehlo.txt").exists()
    assert (output / "compiler_hlo.txt").exists()
    assert (output / "cost_model.json").exists()
    assert (output / "experiment.json").exists()
    assert (output / "invocation.json").exists()
    assert (output / "profiler_config.json").exists()
    assert (output / "source_state.json").exists()
    assert all(not Path(artifact["path"]).is_absolute() for artifact in result["artifacts"])
    assert all((output / artifact["path"]).is_file() for artifact in result["artifacts"])
    with ExperimentLedger(output / "ledger.sqlite") as ledger:
        history = tuple(event.state for event in ledger.history(result["run_id"]))
    assert history == (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.TIMED,
    )


def test_timing_runner_selects_the_owned_collective_explicitly(tmp_path) -> None:
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    output = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tpu_cake.cli",
            "run-matmul",
            "--output-dir",
            str(output),
            "--mode",
            "timing",
            "--mesh-size",
            "8",
            "--m",
            "16",
            "--k",
            "32",
            "--n",
            "1024",
            "--warmup-iterations",
            "0",
            "--measured-iterations",
            "1",
            "--collective-strategy",
            "pallas_bidirectional_ring",
            "--interpret",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    invocation = json.loads((output / "invocation.json").read_text())
    result = MatmulRunResult.model_validate_json((output / "result.json").read_text())

    assert invocation["collective_strategy"] == "pallas_bidirectional_ring"
    assert result.collective_strategy is MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING
    assert "pallas_bidirectional_ring" in (output / "physical.xdsl").read_text()
    assert "native-collective-plan-v3" in (output / "lowered_pallas.py").read_text()


def test_saved_run_replay_recomputes_correctness_and_binds_every_artifact(tmp_path) -> None:
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    root = tmp_path / "bundle"
    output = root / "timing"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tpu_cake.cli",
            "run-matmul",
            "--output-dir",
            str(output),
            "--mode",
            "timing",
            "--mesh-size",
            "4",
            "--m",
            "16",
            "--k",
            "32",
            "--n",
            "16",
            "--warmup-iterations",
            "1",
            "--measured-iterations",
            "3",
            "--tile-m",
            "8",
            "--tile-n",
            "8",
            "--interpret",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = MatmulRunResult.model_validate_json((output / "result.json").read_text())
    experiment = KernelExperiment.model_validate_json((output / "experiment.json").read_text())
    artifacts = tuple(
        ArtifactReference(
            path=str(Path("timing") / artifact.path),
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            role=artifact.role,
        )
        for artifact in result.artifacts
    )
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        schedule_sha256=experiment.schedule_sha256,
        status="rejected",
        runtime=RuntimeIdentity(python="3.13"),
        correctness=CorrectnessResult(passed=True, oracle="test"),
        required_semantic_properties=(),
        metrics=(),
        artifacts=artifacts,
        phases=(),
    )

    _, _, saved_plan = _validate_saved_matmul_phase(
        root, receipt, experiment, "timing", result
    )
    _validate_cost_model(root, receipt, saved_plan)
    cost_input = output / "cost_model_input.json"
    original_cost_input = cost_input.read_text()
    forged_cost_input = json.loads(original_cost_input)
    forged_cost_input["schedule_sha256"] = "0" * 64
    cost_input.write_text(json.dumps(forged_cost_input))
    with pytest.raises(ValueError, match="INPUT_DOES_NOT_MATCH_SAVED_PLAN"):
        _validate_cost_model(root, receipt, saved_plan)
    cost_input.write_text(original_cost_input)
    forged = result.model_copy(update={"maximum_absolute_error": 0.5})
    with pytest.raises(ValueError, match="REPORTED_ERROR_MISMATCH"):
        _validate_saved_matmul_phase(root, receipt, experiment, "timing", forged)
    forged_strategy = result.model_copy(
        update={"collective_strategy": MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING}
    )
    with pytest.raises(ValueError, match="RUN_INVOCATION_PLAN_MISMATCH"):
        _validate_saved_matmul_phase(root, receipt, experiment, "timing", forged_strategy)
    incomplete = receipt.model_copy(update={"artifacts": receipt.artifacts[1:]})
    with pytest.raises(ValueError, match="NOT_BOUND_BY_RECEIPT"):
        _validate_saved_matmul_phase(root, incomplete, experiment, "timing", result)


def test_candidates_use_identical_workload_inputs(tmp_path) -> None:
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    hashes = []
    for name, tile in (("whole", None), ("tiled", "8")):
        command = [
            sys.executable,
            "-m",
            "tpu_cake.cli",
            "run-matmul",
            "--output-dir",
            str(tmp_path / name),
            "--mode",
            "timing",
            "--mesh-size",
            "4",
            "--m",
            "16",
            "--k",
            "32",
            "--n",
            "16",
            "--warmup-iterations",
            "0",
            "--measured-iterations",
            "1",
            "--interpret",
        ]
        if tile is not None:
            command.extend(("--tile-m", tile, "--tile-n", tile))
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        result = json.loads((tmp_path / name / "result.json").read_text())
        hashes.append((result["lhs_sha256"], result["rhs_sha256"]))
    assert hashes[0] == hashes[1]
