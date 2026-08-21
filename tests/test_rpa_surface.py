import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from pydantic import ValidationError

from tpu_cake import rpa_surface_runner as surface_runner
from tpu_cake.canonical import canonical_text
from tpu_cake.rpa_lowering import lower_inkling_sharded_rpa_to_pallas
from tpu_cake.rpa_surface import (
    INKLING_SHARDED_RPA_CORRECTNESS_SEEDS,
    InklingShardedRpaSurfaceContract,
    InklingShardedRpaSurfaceResult,
    default_inkling_sharded_rpa_surface_contract,
)
from tpu_cake.workloads.inkling_rpa import inkling_sharded_fused_rpa_schedule

_CALIBRATION_SEEDS = {20260820, 20260821, 20260822, 20260823, 20260824, 20260825}


def _payload() -> dict:
    return json.loads(Path("contracts/inkling-sharded-rpa-surface.json").read_text())


def test_sharded_rpa_surface_contract_is_external_and_canonical() -> None:
    saved = InklingShardedRpaSurfaceContract.model_validate_json(
        Path("contracts/inkling-sharded-rpa-surface.json").read_text()
    )
    generated = default_inkling_sharded_rpa_surface_contract()

    assert saved == generated
    assert saved.surface_id == ("297debf7e0aea1106fbd7bce984eade52d7d5f9d7659c58de439360b75c874b9")
    assert not set(INKLING_SHARDED_RPA_CORRECTNESS_SEEDS) & _CALIBRATION_SEEDS
    assert saved.plan.compiler_hlo_authority == (
        "receipt-bound-raw-bytes-not-reproducible-identity"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("hlo_identity_status",), "pending"),
        (("output_relative_l2_error",), 0.007),
        (("plan", "stablehlo_sha256"), "0" * 64),
        (("plan", "mesh_shape"), [1, 8]),
        (("runtime", "jax"), "0.11.1"),
        (("backend_import_packages",), ["psutil==7.0.0"] * 4),
    ),
)
def test_sharded_rpa_surface_contract_rejects_coordinated_policy_drift(
    path: tuple[str, ...],
    value: object,
) -> None:
    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((ValidationError, ValueError)):
        InklingShardedRpaSurfaceContract.model_validate_json(json.dumps(payload))


def test_sharded_rpa_backend_runtime_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = default_inkling_sharded_rpa_surface_contract()
    surface_runner._require_backend_runtime(contract)
    observed_version = surface_runner.importlib.metadata.version
    monkeypatch.setattr(
        surface_runner.importlib.metadata,
        "version",
        lambda name: "0.0.0" if name == "psutil" else observed_version(name),
    )

    with pytest.raises(ValueError, match="BACKEND_RUNTIME_MISMATCH"):
        surface_runner._require_backend_runtime(contract)


def _repair_receipt(root: Path, *relative_paths: str) -> None:
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in relative_paths:
        path = root / relative
        artifact = next(value for value in receipt["artifacts"] if value["path"] == relative)
        artifact["size_bytes"] = path.stat().st_size
        artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative == "result.json":
            receipt["result_sha256"] = artifact["sha256"]
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_sharded_rpa_surface_runner_builds_and_replays_a_closed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_inkling_sharded_rpa_surface_contract()
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    devices = tuple(
        SimpleNamespace(id=index, process_index=0, platform="tpu", device_kind="TPU7x")
        for index in range(8)
    )
    output_shape = contract.plan.global_output_shapes[0]
    cache_shape = (2, 2)

    def reference(_inputs):
        return (
            np.zeros(output_shape, dtype=jax.numpy.bfloat16),
            np.zeros(cache_shape, dtype=jax.numpy.bfloat16),
        )

    def compile_surface(_contract, _kernel, _devices, _inputs):
        return surface_runner._CompiledSurface(
            plan=plan,
            mesh=None,
            executable=lambda *_values: reference(_values),
            stablehlo="fake-stablehlo\n",
            compiler_hlo="RPAd tpu_custom_call\n",
        )

    def validate_plan(root, _contract, result):
        if (
            (root / "physical.xdsl").read_text()
            != canonical_text(inkling_sharded_fused_rpa_schedule())
            or json.loads((root / "plan.json").read_text()) != contract.plan.model_dump(mode="json")
            or (root / "stablehlo.txt").read_text() != "fake-stablehlo\n"
        ):
            raise ValueError("INKLING_SHARDED_RPA_PLAN_REPLAY_MISMATCH")
        compiler_hlo = root / "compiler_hlo.txt"
        compiler_text = compiler_hlo.read_text()
        if (
            surface_runner._sha256(compiler_hlo) != result.compiler_hlo_sha256
            or "RPAd" not in compiler_text
            or "tpu_custom_call" not in compiler_text
        ):
            raise ValueError("INKLING_SHARDED_RPA_PLAN_REPLAY_MISMATCH")

    def git_show(arguments, *, cwd, check, capture_output):
        assert arguments[:2] == ["git", "show"]
        assert Path(cwd) == repository
        assert check and capture_output
        relative = arguments[2].split(":", 1)[1]
        return SimpleNamespace(stdout=(repository / relative).read_bytes())

    monkeypatch.setattr(surface_runner, "_require_compilation_root", lambda *_args: None)
    monkeypatch.setattr(surface_runner, "_require_backend_source", lambda *_args: None)
    monkeypatch.setattr(surface_runner, "_require_backend_runtime", lambda *_args: None)
    monkeypatch.setattr(surface_runner, "_require_clean_repository", lambda *_args: None)
    monkeypatch.setattr(surface_runner, "_git_output", lambda *_args: commit)
    monkeypatch.setattr(surface_runner.subprocess, "run", git_show)
    monkeypatch.setattr(surface_runner, "_runtime_identity", lambda: contract.runtime)
    monkeypatch.setattr(surface_runner.jax, "devices", lambda: devices)
    monkeypatch.setattr(surface_runner, "_compile_surface", compile_surface)
    monkeypatch.setattr(surface_runner, "_place_inputs", lambda _compiled, inputs: inputs)
    monkeypatch.setattr(surface_runner, "_validate_output_abi", lambda *_args: None)
    monkeypatch.setattr(
        surface_runner,
        "_execute",
        lambda _executable, inputs: tuple(value.copy() for value in reference(inputs)),
    )
    monkeypatch.setattr(surface_runner, "inkling_sharded_fused_rpa_reference", reference)
    monkeypatch.setattr(surface_runner, "_validate_plan_artifacts", validate_plan)

    root = tmp_path / "run"
    result = surface_runner.run_inkling_sharded_rpa_surface(root, contract, lambda: None)
    assert result.accepted
    assert tuple(value.seed for value in result.correctness) == contract.correctness_seeds
    assert all(value.input_sha256 for value in result.correctness)
    assert surface_runner.validate_inkling_sharded_rpa_surface(root, contract) == result

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert surface_runner.validate_inkling_sharded_rpa_surface(relocated, contract) == result

    changed_output = tmp_path / "changed-output"
    shutil.copytree(root, changed_output)
    output_path = (
        changed_output / "correctness" / f"seed-{contract.correctness_seeds[0]}" / "output.npy"
    )
    storage = np.load(output_path, allow_pickle=False)
    storage.reshape(-1)[0] ^= np.uint16(1)
    np.save(output_path, storage, allow_pickle=False)
    _repair_receipt(changed_output, output_path.relative_to(changed_output).as_posix())
    with pytest.raises(ValueError, match="CORRECTNESS_REPLAY_MISMATCH"):
        surface_runner.validate_inkling_sharded_rpa_surface(changed_output, contract)

    for name, relative_path, error in (
        (
            "changed-repeat-cache",
            f"correctness/seed-{contract.correctness_seeds[0]}/repeat_cache.npy",
            "CORRECTNESS_REPLAY_MISMATCH",
        ),
        (
            "changed-cache",
            f"correctness/seed-{contract.correctness_seeds[0]}/cache.npy",
            "CORRECTNESS_REPLAY_MISMATCH",
        ),
        (
            "changed-timing-cache",
            "timing/post_cache.npy",
            "TIMING_OUTPUT_MISMATCH",
        ),
    ):
        changed_array = tmp_path / name
        shutil.copytree(root, changed_array)
        array_path = changed_array / relative_path
        storage = np.load(array_path, allow_pickle=False)
        storage.reshape(-1)[0] ^= np.uint16(1)
        np.save(array_path, storage, allow_pickle=False)
        _repair_receipt(changed_array, relative_path)
        with pytest.raises(ValueError, match=error):
            surface_runner.validate_inkling_sharded_rpa_surface(changed_array, contract)

    changed_result = tmp_path / "changed-result"
    shutil.copytree(root, changed_result)
    result_payload = json.loads((changed_result / "result.json").read_text())
    result_payload["median_round_duration_ns"] += 1
    (changed_result / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_result, "result.json")
    with pytest.raises((ValidationError, ValueError)):
        surface_runner.validate_inkling_sharded_rpa_surface(changed_result, contract)

    changed_ledger = tmp_path / "changed-ledger"
    shutil.copytree(root, changed_ledger)
    with sqlite3.connect(changed_ledger / "ledger.sqlite") as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = 'timed'",
            ("0" * 64,),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for suffix in ("-shm", "-wal"):
        (changed_ledger / f"ledger.sqlite{suffix}").unlink(missing_ok=True)
    _repair_receipt(changed_ledger, "ledger.sqlite")
    with pytest.raises(ValueError, match="LEDGER_REPLAY_MISMATCH"):
        surface_runner.validate_inkling_sharded_rpa_surface(changed_ledger, contract)

    changed_hlo = tmp_path / "changed-hlo"
    shutil.copytree(root, changed_hlo)
    (changed_hlo / "compiler_hlo.txt").write_text("RPAd tpu_custom_call forged\n")
    _repair_receipt(changed_hlo, "compiler_hlo.txt")
    with pytest.raises(ValueError, match="PLAN_REPLAY_MISMATCH"):
        surface_runner.validate_inkling_sharded_rpa_surface(changed_hlo, contract)

    changed_source = tmp_path / "changed-source"
    shutil.copytree(root, changed_source)
    manifest_payload = json.loads((changed_source / "source_manifest.json").read_text())[:-1]
    (changed_source / "source_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    )
    result_payload = json.loads((changed_source / "result.json").read_text())
    result_payload["source_manifest"] = manifest_payload
    result_payload["source_manifest_sha256"] = surface_runner._sha256(
        changed_source / "source_manifest.json"
    )
    (changed_source / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_source, "source_manifest.json", "result.json")
    with pytest.raises(ValueError, match="SOURCE_EVIDENCE_MISMATCH"):
        surface_runner.validate_inkling_sharded_rpa_surface(changed_source, contract)

    extra_artifact = tmp_path / "extra-artifact"
    shutil.copytree(root, extra_artifact)
    (extra_artifact / "extra.txt").write_text("forged")
    with pytest.raises(ValueError, match="RECEIPT_IDENTITY_MISMATCH"):
        surface_runner.validate_inkling_sharded_rpa_surface(extra_artifact, contract)


def test_sharded_rpa_surface_result_rejects_failed_correctness() -> None:
    payload = {
        "surface_id": "0" * 64,
        "run_id": "1" * 64,
        "source_commit": "2" * 40,
        "uv_lock_sha256": "3" * 64,
        "source_state_sha256": "4" * 64,
        "source_manifest_sha256": "5" * 64,
        "source_manifest": [{"path": "a.py", "sha256": "6" * 64}],
        "runtime": default_inkling_sharded_rpa_surface_contract().runtime.model_dump(mode="json"),
        "devices": [
            {"id": index, "process_index": 0, "platform": "tpu", "device_kind": "TPU7x"}
            for index in range(8)
        ],
        "plan": default_inkling_sharded_rpa_surface_contract().plan.model_dump(mode="json"),
        "compiler_hlo_sha256": "7" * 64,
        "correctness": [
            {
                "seed": index,
                "input_sha256": ["8" * 64] * 11,
                "output_sha256": "9" * 64,
                "repeat_output_sha256": "9" * 64,
                "oracle_output_sha256": "a" * 64,
                "cache_sha256": "b" * 64,
                "repeat_cache_sha256": "b" * 64,
                "oracle_cache_sha256": "b" * 64,
                "repeated_output_exact": True,
                "repeated_cache_exact": True,
                "maximum_absolute_error": 0.0,
                "relative_l2_error": 0.0,
                "passed": index != 0,
            }
            for index in range(5)
        ],
        "timing_input_sha256": ["c" * 64] * 11,
        "pre_timing_output_sha256": ["d" * 64, "e" * 64],
        "rounds": [
            {"round_index": index, "samples_ns": [100, 100, 100], "median_ns": 100.0}
            for index in range(12)
        ],
        "post_timing_output_sha256": ["d" * 64, "e" * 64],
        "median_round_duration_ns": 100.0,
        "p90_round_duration_ns": 100.0,
        "coefficient_of_variation": 0.0,
        "accepted": True,
        "claim_scope": default_inkling_sharded_rpa_surface_contract().claim_scope,
    }
    with pytest.raises(ValidationError, match="correctness evidence must pass exactly"):
        InklingShardedRpaSurfaceResult.model_validate_json(json.dumps(payload))


def test_sharded_rpa_timing_does_not_materialize_device_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = object()
    observed = []
    monkeypatch.setattr(surface_runner.jax, "block_until_ready", observed.append)

    surface_runner._execute_timed(lambda *_inputs: output, ())

    assert observed == [output]


def test_sharded_rpa_error_metric_requires_exact_output_abi() -> None:
    with pytest.raises(ValueError, match="OUTPUT_ABI_MISMATCH"):
        surface_runner._errors(
            np.zeros((2,), dtype=jax.numpy.bfloat16),
            np.zeros((1, 2), dtype=jax.numpy.bfloat16),
        )


def test_sharded_rpa_publication_never_replaces_a_foreign_root(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    (staging / "ours").write_text("ours")

    with pytest.raises(FileExistsError):
        surface_runner._rename_directory_noreplace(staging, destination)

    assert (staging / "ours").read_text() == "ours"
    assert destination.is_dir() and not tuple(destination.iterdir())
