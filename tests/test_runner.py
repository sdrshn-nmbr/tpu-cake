from __future__ import annotations

import json
import os
import subprocess
import sys


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
