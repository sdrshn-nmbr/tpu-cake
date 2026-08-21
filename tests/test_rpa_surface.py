import hashlib
import json
import shutil
import sqlite3
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from pydantic import ValidationError

from tpu_cake import rpa_surface as surface
from tpu_cake import rpa_surface_runner as surface_runner
from tpu_cake.canonical import canonical_text
from tpu_cake.rpa_lowering import lower_inkling_sharded_rpa_to_pallas
from tpu_cake.rpa_surface import (
    INKLING_SHARDED_RPA_CORRECTNESS_SEEDS,
    InklingShardedRpaRelocationRuntime,
    InklingShardedRpaSurfaceContract,
    InklingShardedRpaSurfaceResult,
    default_inkling_sharded_rpa_surface_contract,
)
from tpu_cake.workloads.inkling_rpa import inkling_sharded_fused_rpa_schedule

_CALIBRATION_SEEDS = {
    20260820,
    20260821,
    20260822,
    20260823,
    20260824,
    20260825,
    8861363933501065961,
    1269528214265211801,
    4209644372387580568,
    15603344423790358252,
    7026367813976238475,
    9096428414533206234,
    12145031094770005217,
    622934234548142313,
    15696904859270974668,
    17038534533205655854,
}


def _payload() -> dict:
    return json.loads(Path("contracts/inkling-sharded-rpa-surface.json").read_text())


def test_sharded_rpa_surface_contract_is_external_and_canonical() -> None:
    saved = InklingShardedRpaSurfaceContract.model_validate_json(
        Path("contracts/inkling-sharded-rpa-surface.json").read_text()
    )
    generated = default_inkling_sharded_rpa_surface_contract()

    assert saved == generated
    assert saved.surface_id == ("1ce597bd87b25a45d6a95ab57c674babdaa6a18a6f52cc480cc9939792afb585")
    assert not set(INKLING_SHARDED_RPA_CORRECTNESS_SEEDS) & _CALIBRATION_SEEDS
    assert saved.plan.external_donate_argnums == (0, 3)
    assert saved.plan.compiler_hlo_authority == (
        "receipt-bound-raw-bytes-not-reproducible-identity"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("hlo_identity_status",), "pinned"),
        (("output_relative_l2_error",), 0.007),
        (("cpu_reference_replay_relative_l2_error",), 0.007),
        (("plan", "stablehlo_sha256"), "1" * 64),
        (("plan", "mesh_shape"), [1, 8]),
        (("plan", "external_donate_argnums"), [0, 2]),
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


def test_sharded_rpa_surface_runner_refuses_pending_hlo_before_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="HLO_IDENTITY_PENDING"):
        surface_runner.run_inkling_sharded_rpa_surface(
            root,
            default_inkling_sharded_rpa_surface_contract(),
            lambda: None,
        )
    assert not root.exists()


def test_sharded_rpa_attestation_refuses_pending_hlo_before_paths(
    tmp_path: Path,
) -> None:
    contract = default_inkling_sharded_rpa_surface_contract()
    archive = tmp_path / "missing.tar.zst"
    attestation = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="HLO_IDENTITY_PENDING"):
        surface_runner.write_inkling_sharded_rpa_relocation_attestation(
            attestation,
            archive=archive,
            contract=contract,
        )
    with pytest.raises(ValueError, match="HLO_IDENTITY_PENDING"):
        surface_runner.validate_inkling_sharded_rpa_relocation_attestation(
            attestation,
            archive=archive,
            contract=contract,
        )
    assert not archive.exists() and not attestation.exists()


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
    pending_contract = default_inkling_sharded_rpa_surface_contract()
    contract = pending_contract.model_copy(
        update={
            "hlo_identity_status": "pinned",
            "plan": pending_contract.plan.model_copy(
                update={"stablehlo_sha256": hashlib.sha256(b"fake-stablehlo\n").hexdigest()}
            ),
        }
    )
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_command = subprocess.run
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    devices = tuple(
        SimpleNamespace(id=index, process_index=0, platform="tpu", device_kind="TPU7x")
        for index in range(8)
    )
    output_shape = contract.plan.global_output_shapes[0]
    cache_shape = (2, 2)

    def reference(_inputs):
        return (
            np.full(output_shape, 0.1, dtype=jax.numpy.bfloat16),
            np.zeros(cache_shape, dtype=jax.numpy.bfloat16),
        )

    def compile_surface(_contract, _kernel, _devices, _inputs):
        return surface_runner._CompiledSurface(
            plan=plan,
            mesh=None,
            executable=lambda *_values: reference(_values),
            stablehlo="fake-stablehlo\n",
            compiler_hlo=f"RPAd tpu_custom_call {surface_runner._QUERY_CACHE_ALIAS}\n",
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
            or surface_runner._QUERY_CACHE_ALIAS not in compiler_text
        ):
            raise ValueError("INKLING_SHARDED_RPA_PLAN_REPLAY_MISMATCH")

    def git_show(arguments, **kwargs):
        if arguments[:2] != ["git", "show"]:
            return run_command(arguments, **kwargs)
        assert kwargs["check"] and kwargs["capture_output"]
        cwd = kwargs["cwd"]
        assert Path(cwd) == repository
        relative = arguments[2].split(":", 1)[1]
        return SimpleNamespace(stdout=(repository / relative).read_bytes())

    monkeypatch.setattr(surface_runner, "_require_compilation_root", lambda *_args: None)
    monkeypatch.setattr(
        surface_runner,
        "default_inkling_sharded_rpa_surface_contract",
        lambda: contract,
    )
    monkeypatch.setattr(surface, "_plan_contract", lambda: contract.plan)
    monkeypatch.setattr(surface_runner, "_require_backend_source", lambda *_args: None)
    monkeypatch.setattr(surface_runner, "_require_backend_runtime", lambda *_args: None)
    monkeypatch.setattr(surface_runner, "_require_clean_repository", lambda *_args: None)
    monkeypatch.setattr(surface_runner, "_git_output", lambda *_args: commit)
    monkeypatch.setattr(surface_runner.subprocess, "run", git_show)
    monkeypatch.setattr(surface_runner, "_runtime_identity", lambda: contract.runtime)
    monkeypatch.setattr(surface_runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(surface_runner.platform, "machine", lambda: "x86_64")
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

    def portable_reference(inputs):
        output, cache = reference(inputs)
        output.view(np.uint16).reshape(-1)[0] += np.uint16(1)
        return output, cache

    monkeypatch.setattr(
        surface_runner,
        "inkling_sharded_fused_rpa_reference",
        portable_reference,
    )
    assert surface_runner.validate_inkling_sharded_rpa_surface(relocated, contract) == result

    def incompatible_reference(inputs):
        output, cache = reference(inputs)
        output.fill(0.2)
        return output, cache

    monkeypatch.setattr(
        surface_runner,
        "inkling_sharded_fused_rpa_reference",
        incompatible_reference,
    )
    with pytest.raises(ValueError, match="ORACLE_REPLAY_MISMATCH"):
        surface_runner.validate_inkling_sharded_rpa_surface(relocated, contract)

    for nonfinite in (np.nan, np.inf):

        def nonfinite_reference(inputs, value=nonfinite):
            output, cache = reference(inputs)
            output.reshape(-1)[0] = value
            return output, cache

        monkeypatch.setattr(
            surface_runner,
            "inkling_sharded_fused_rpa_reference",
            nonfinite_reference,
        )
        with pytest.raises(ValueError, match="OUTPUT_NONFINITE"):
            surface_runner.validate_inkling_sharded_rpa_surface(relocated, contract)
    monkeypatch.setattr(surface_runner, "inkling_sharded_fused_rpa_reference", reference)

    archive_tar = tmp_path / "bundle.tar"
    with tarfile.open(archive_tar, "w") as bundle:
        bundle.add(root, arcname="bundle")
    archive = tmp_path / "bundle.tar.zst"
    with archive.open("wb") as output:
        subprocess.run(
            ["zstd", "--compress", "--stdout", str(archive_tar)],
            check=True,
            stdout=output,
        )
    monkeypatch.setattr(
        surface_runner,
        "_relocation_runtime",
        lambda _contract: InklingShardedRpaRelocationRuntime(
            python="3.12.11",
            jax="0.11.0",
            jaxlib="0.11.0",
            ml_dtypes="0.6.0",
            numpy="2.5.2",
            system="Darwin",
            machine="arm64",
        ),
    )
    attestation_path = tmp_path / "relocation-attestation.json"
    attestation = surface_runner.write_inkling_sharded_rpa_relocation_attestation(
        attestation_path,
        archive=archive,
        contract=contract,
    )
    assert attestation.status == "portable_accepted"
    assert len(attestation.observations) == len(contract.correctness_seeds)
    assert (
        surface_runner.validate_inkling_sharded_rpa_relocation_attestation(
            attestation_path,
            archive=archive,
            contract=contract,
        )
        == attestation
    )
    mutated_attestation_path = tmp_path / "mutated-attestation.json"
    mutated_attestation = json.loads(attestation_path.read_text())
    mutated_attestation["observations"][0]["verifier_reference_sha256"] = "0" * 64
    mutated_attestation_path.write_text(
        json.dumps(mutated_attestation, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="ATTESTATION_MISMATCH"):
        surface_runner.validate_inkling_sharded_rpa_relocation_attestation(
            mutated_attestation_path,
            archive=archive,
            contract=contract,
        )

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
            "ORACLE_REPLAY_MISMATCH",
        ),
        (
            "changed-timing-cache",
            "timing/post_cache.npy",
            "TIMING_OUTPUT_MISMATCH",
        ),
        (
            "changed-producer-oracle",
            f"correctness/seed-{contract.correctness_seeds[0]}/oracle_output.npy",
            "ORACLE_REPLAY_MISMATCH",
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
        "producer_system": "Linux",
        "producer_machine": "x86_64",
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

    assert surface_runner._execute_timed(lambda *_inputs: output, ()) is output

    assert observed == [output]


def test_sharded_rpa_timing_starts_each_round_from_fresh_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_inkling_sharded_rpa_surface_contract().model_copy(
        update={"warmup_iterations": 2, "timing_rounds": 3, "samples_per_round": 3}
    )
    compiled = SimpleNamespace(executable=object())
    placed = []
    observed = []
    events = []

    def place_inputs(_compiled, _host_inputs):
        values = (object(), object(), object(), object(), object())
        placed.append(values)
        events.append(("place", values))
        return values

    def execute_timed(_executable, inputs):
        observed.append(inputs)
        events.append(("execute", inputs))
        return object(), object()

    monkeypatch.setattr(surface_runner, "_place_inputs", place_inputs)
    monkeypatch.setattr(surface_runner, "_execute_timed", execute_timed)
    monkeypatch.setattr(
        surface_runner.jax,
        "block_until_ready",
        lambda inputs: events.append(("resident", inputs)),
    )

    rounds = surface_runner._timing_rounds(contract, compiled, ())

    assert len(placed) == 1 + contract.timing_rounds
    assert len(rounds) == contract.timing_rounds
    assert observed[0] is placed[0]
    assert observed[1][0] is not placed[0][0]
    for index, (event, inputs) in enumerate(events):
        if event == "place":
            assert events[index + 1] == ("resident", inputs)
            assert events[index + 2] == ("execute", inputs)
    for round_index in range(contract.timing_rounds):
        start = contract.warmup_iterations + round_index * contract.samples_per_round
        assert observed[start] is placed[round_index + 1]
        assert observed[start + 1][0] is not placed[round_index + 1][0]


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
