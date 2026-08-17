from __future__ import annotations

import copy
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
from tpu_cake.receipt import _validate_saved_matmul_phase
from tpu_cake.runner import (
    MatmulRunResult,
    RunMode,
    _profiler_contract,
    validate_profiler_contract,
)


def test_counter_profiler_contract_retains_counter_only_options() -> None:
    timing = _profiler_contract(RunMode.TRACE)["advanced_configuration"]
    counters = _profiler_contract(RunMode.COUNTERS)["advanced_configuration"]

    assert "tpu_enable_periodic_counter_sampling" not in timing
    assert counters["tpu_enable_periodic_counter_sampling"] is True
    assert counters["num_tensor_cores_to_trace_per_device"] == 1
    assert "tpu_tc_perf_counter_sampling_options" in counters


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

    _validate_saved_matmul_phase(root, receipt, experiment, "timing", result)
    forged = result.model_copy(update={"maximum_absolute_error": 0.5})
    with pytest.raises(ValueError, match="REPORTED_ERROR_MISMATCH"):
        _validate_saved_matmul_phase(root, receipt, experiment, "timing", forged)
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
