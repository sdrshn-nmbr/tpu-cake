from __future__ import annotations

import json
import os
import subprocess
import sys

from tpu_cake.ledger import ExperimentLedger, RunState
from tpu_cake.runner import RunMode, _profiler_contract


def test_counter_profiler_contract_retains_counter_only_options() -> None:
    timing = _profiler_contract(RunMode.TRACE)["advanced_configuration"]
    counters = _profiler_contract(RunMode.COUNTERS)["advanced_configuration"]

    assert "tpu_enable_periodic_counter_sampling" not in timing
    assert counters["tpu_enable_periodic_counter_sampling"] is True
    assert counters["num_tensor_cores_to_trace_per_device"] == 1
    assert "tpu_tc_perf_counter_sampling_options" in counters


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
