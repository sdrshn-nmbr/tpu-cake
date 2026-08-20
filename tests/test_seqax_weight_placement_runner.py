from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tpu_cake.seqax_weight_placement_runner as placement_runner
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.dtensor_interpreter import interpret_distributed_program
from tpu_cake.identity import array_sha256, arrays_sha256
from tpu_cake.seqax_pallas_runner import _errors
from tpu_cake.seqax_pallas_search import SeqaxPallasCandidateCorrectness, SeqaxPallasDevice
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementName,
    SeqaxWeightResidencyObservation,
    default_seqax_weight_placement_contract,
)
from tpu_cake.seqax_weight_placement_runner import (
    CompiledPlacement,
    _cpu_oracle_verdicts,
    _isolated_memory_observations,
    _preflight_existing_root,
    _require_safe_new_root,
    _source_manifest,
    _validate_closed_world,
    _validate_correctness,
    prepare_weight_placement_candidates,
    run_seqax_weight_placement,
    validate_seqax_weight_placement,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python="3.12.3",
        jax="0.11.0",
        jaxlib="0.11.0",
        libtpu="0.0.44.1",
        xla="--xla_tpu_use_enhanced_launch_barrier=true",
    )


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


def test_weight_placement_uses_the_canonical_seqax_oracle_tolerance() -> None:
    oracle = np.zeros((1,), dtype=np.float32)
    output = np.full((1,), 0.01, dtype=np.float32)

    assert np.allclose(output, oracle, atol=0.016, rtol=0.05)
    assert _cpu_oracle_verdicts([output], [oracle]) == (False,)


def _memory_payload(candidate: str, parameter_bytes: int) -> dict[str, object]:
    return {
        "candidate": candidate,
        "runtime": _runtime(),
        "devices": _devices(),
        "distributed_schedule_sha256": "1" * 64,
        "physical_schedule_sha256": "2" * 64,
        "pallas_source_sha256": "3" * 64,
        "stablehlo_sha256": "4" * 64,
        "compiler_hlo_sha256": "5" * 64,
        "source_commit": "6" * 40,
        "source_manifest": _source_manifest(),
        "timing_input_sha256": ("7" * 64,),
        "output_sha256": "8" * 64,
        "parameter_bytes_per_device": parameter_bytes,
        "device_bytes_limit": (100_000,) * 8,
        "peak_bytes_in_use": (50_000,) * 8,
        "largest_allocation_bytes": (10_000,) * 8,
        "isolated_process": True,
        "fits_observed_device_memory": True,
    }


def test_weight_placement_preparation_binds_exact_schedule_surface() -> None:
    contract = default_seqax_weight_placement_contract(_runtime())
    prepared = prepare_weight_placement_candidates(contract)

    assert tuple(value.candidate.name for value in prepared) == (
        SeqaxWeightPlacementName.SHARDED,
        SeqaxWeightPlacementName.EMBEDDING_MLP,
    )
    assert tuple(value.plan.pallas_region_count for value in prepared) == (9, 9)
    assert len({value.plan.distributed_schedule_sha256 for value in prepared}) == 2
    assert len({value.plan.physical_schedule_sha256 for value in prepared}) == 2
    assert len({value.plan.source_sha256() for value in prepared}) == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("peak_bytes_in_use", (0,) * 8, "positive peak"),
        ("largest_allocation_bytes", (0,) * 8, "positive allocation"),
        ("isolated_process", False, "isolated process"),
    ),
)
def test_weight_residency_rejects_invalid_observations(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _memory_payload("sharded", 22_912)
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        SeqaxWeightResidencyObservation.model_validate(payload)


def test_isolated_memory_probe_requires_one_typed_record_per_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_seqax_weight_placement_contract(_runtime())

    def run(command, **_kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="6" * 40 + "\n", stderr="")
        candidate = command[-1]
        parameter_bytes = 22_912 if candidate == "sharded" else 33_152
        record = SeqaxWeightResidencyObservation.model_validate(
            _memory_payload(candidate, parameter_bytes)
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"SEQAX_WEIGHT_PLACEMENT_MEMORY_JSON={record.model_dump_json()}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    observations = _isolated_memory_observations(contract, tmp_path)

    assert tuple(value.candidate for value in observations) == (
        SeqaxWeightPlacementName.SHARDED,
        SeqaxWeightPlacementName.EMBEDDING_MLP,
    )


def test_isolated_memory_probe_rejects_ambiguous_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_seqax_weight_placement_contract(_runtime())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="not a typed memory record\n",
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="MEMORY_SUBPROCESS_OUTPUT"):
        _isolated_memory_observations(contract, tmp_path)


def test_isolated_memory_probe_reports_child_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_seqax_weight_placement_contract(_runtime())

    def run(command, **_kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="6" * 40 + "\n", stderr="")
        return subprocess.CompletedProcess(command, 7, stdout="partial", stderr="compile failed")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="MEMORY_SUBPROCESS_FAILED.*returncode=7"):
        _isolated_memory_observations(contract, tmp_path)


def test_weight_placement_source_manifest_binds_runtime_authority() -> None:
    paths = {value.path for value in _source_manifest()}
    assert "tpu_cake/cli.py" in paths
    assert "tpu_cake/seqax_weight_confirmation.py" in paths
    assert "tpu_cake/seqax_weight_confirmation_runner.py" in paths
    assert "tpu_cake/seqax_weight_placement.py" in paths
    assert "tpu_cake/seqax_weight_placement_runner.py" in paths
    assert "tpu_cake/seqax_pallas_search.py" in paths
    assert "tpu_cake/physical_geometry.py" in paths
    assert "tpu_cake/dialects/distributed_tensor.py" in paths
    assert "tpu_cake/dialects/tpu_schedule.py" in paths


def test_weight_placement_root_checks_happen_before_writes(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="UNSAFE_ROOT"):
        _require_safe_new_root(repository_root)
    assert not (tmp_path / "new-run").exists()
    _require_safe_new_root((tmp_path / "new-run").resolve())


def test_closed_world_rejects_an_extra_artifact(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    expected.write_text("{}\n")
    extra = tmp_path / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="CLOSED_WORLD_MISMATCH"):
        _validate_closed_world(tmp_path, {expected.resolve()})


def test_existing_root_rejects_symlinked_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("evidence")
    root = tmp_path / "run"
    root.mkdir()
    (root / "alias.txt").symlink_to(target)
    with pytest.raises(ValueError, match="SYMLINK"):
        _preflight_existing_root(root)


def test_existing_root_rejects_hardlinked_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("evidence")
    root = tmp_path / "run"
    root.mkdir()
    os.link(target, root / "alias.txt")
    with pytest.raises(ValueError, match="HARDLINK"):
        _preflight_existing_root(root)


def _correctness_fixture(
    root: Path,
) -> tuple[object, tuple[object, ...], tuple[SeqaxPallasCandidateCorrectness, ...]]:
    contract = default_seqax_weight_placement_contract(_runtime())
    prepared = prepare_weight_placement_candidates(contract)
    inputs_by_seed = []
    oracles = []
    outputs = {value.candidate.name: [] for value in prepared}
    for seed in contract.correctness_seeds:
        inputs = tuple(
            np.asarray(value) for value in seqax_forward_inputs(seed=seed, **contract.parameters)
        )
        oracle = np.asarray(seqax_forward_canonical_reference(inputs, **contract.parameters))
        seed_root = root / str(seed)
        for index, value in enumerate(inputs):
            (seed_root / "inputs").mkdir(parents=True, exist_ok=True)
            np.save(seed_root / "inputs" / f"{index:02d}.npy", value, allow_pickle=False)
        np.save(seed_root / "cpu_oracle.npy", oracle, allow_pickle=False)
        inputs_by_seed.append(inputs)
        oracles.append(oracle)
        for value in prepared:
            output = interpret_distributed_program(value.distributed, inputs)[0]
            if seed in contract.correctness_seeds[3:]:
                output = output + np.float32(0.1)
            (seed_root / "outputs").mkdir(parents=True, exist_ok=True)
            np.save(
                seed_root / "outputs" / f"{value.candidate.name}.npy",
                output,
                allow_pickle=False,
            )
            outputs[value.candidate.name].append(output)
    baseline = outputs[contract.baseline]
    baseline_hashes = tuple(array_sha256(value) for value in baseline)
    records = []
    for value in prepared:
        candidate_outputs = outputs[value.candidate.name]
        errors = tuple(
            _errors(output, oracle)
            for output, oracle in zip(candidate_outputs, oracles, strict=True)
        )
        records.append(
            SeqaxPallasCandidateCorrectness(
                name=value.candidate.name,
                input_sha256=tuple(arrays_sha256(values) for values in inputs_by_seed),
                output_sha256=tuple(array_sha256(output) for output in candidate_outputs),
                baseline_output_sha256=baseline_hashes,
                exact_baseline_parity=True,
                cpu_oracle_sha256=tuple(array_sha256(oracle) for oracle in oracles),
                cpu_oracle_maximum_absolute_error=tuple(error[0] for error in errors),
                cpu_oracle_maximum_relative_error=tuple(error[1] for error in errors),
                cpu_oracle_passed=_cpu_oracle_verdicts(candidate_outputs, oracles),
            )
        )
    return contract, prepared, tuple(records)


def test_correctness_replay_requires_exact_five_seed_incumbent_parity(
    tmp_path: Path,
) -> None:
    contract, prepared, records = _correctness_fixture(tmp_path)
    _validate_correctness(tmp_path, contract, prepared, records)

    output_path = tmp_path / str(contract.correctness_seeds[0]) / "outputs" / "embedding-mlp.npy"
    output = np.load(output_path, allow_pickle=False)
    output.flat[0] = output.flat[0] + np.float32(1e-4)
    np.save(output_path, output, allow_pickle=False)
    with pytest.raises(ValueError, match="CORRECTNESS_REPLAY_MISMATCH"):
        _validate_correctness(tmp_path, contract, prepared, records)


def test_correctness_replay_rejects_dtype_only_output_mutation(tmp_path: Path) -> None:
    contract, prepared, records = _correctness_fixture(tmp_path)
    output_path = tmp_path / str(contract.correctness_seeds[0]) / "outputs" / "embedding-mlp.npy"
    output = np.load(output_path, allow_pickle=False).astype(np.float64)
    np.save(output_path, output, allow_pickle=False)
    with pytest.raises(ValueError, match="OUTPUT_ABI_MISMATCH"):
        _validate_correctness(tmp_path, contract, prepared, records)


def _synthetic_hlo(prepared) -> tuple[str, str]:
    tiles = tuple(
        (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
        for operation in prepared.physical.walk()
        if operation.name == "tpu_schedule.mxu_einsum"
    )
    stablehlo = "module @fixture {\n  func.func public @main() -> tensor<1xf32> {\n"
    stablehlo += (
        "    %p0 = stablehlo.custom_call @tpu_custom_call() "
        '{kernel_name = "seqax_named_einsum"} : () -> tensor<1xf32>\n'
    )
    stablehlo += "".join(
        f"    %p{index} = stablehlo.custom_call @tpu_custom_call(%p{index - 1}) "
        '{kernel_name = "seqax_named_einsum"} : (tensor<1xf32>) -> tensor<1xf32>\n'
        for index in range(1, 9)
    )
    stablehlo += "".join(
        f'    %g{index} = "stablehlo.all_gather"('
        f"%{'p8' if index == 0 else f'g{index - 1}'}) "
        "<{all_gather_dim = 0 : i64, "
        "replica_groups = dense<[[0]]> : tensor<1x1xi64>, "
        "channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, "
        "use_global_device_ids}> : (tensor<1xf32>) -> tensor<1xf32>\n"
        for index in range(prepared.candidate.expected_stablehlo_all_gathers)
    )
    last_value = f"%g{prepared.candidate.expected_stablehlo_all_gathers - 1}"
    stablehlo += f"    return {last_value} : tensor<1xf32>\n  }}\n}}\n"
    compiler_lines = ["HloModule fixture", "ENTRY main {"]
    for index, (tile_m, tile_k, tile_n) in enumerate(tiles):
        compiler_lines.extend(
            (
                (
                    f"  pallas_call.{index} = f32[] custom-call(), "
                    'custom_call_target="tpu_custom_call", '
                    "frontend_attributes={kernel_metadata={"
                ),
                f'"region_index":{index},',
                f'"schedule_sha256":"{prepared.plan.physical_schedule_sha256}",',
                f'"tile_k":{tile_k},',
                f'"tile_m":{tile_m},',
                f'"tile_n":{tile_n}',
                "}}, backend_config={}",
            )
        )
    all_gathers, reduce_scatters = placement_runner._physical_collective_counts(prepared.physical)
    compiler_lines.extend(f"  ag.{index} = f32[] all-gather()" for index in range(all_gathers))
    compiler_lines.extend(
        f"  rs.{index} = f32[] reduce-scatter()" for index in range(reduce_scatters)
    )
    compiler_lines.append("}")
    return stablehlo, "\n".join(compiler_lines) + "\n"


def _fake_devices() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            id=index,
            process_index=0,
            platform="tpu",
            device_kind="TPU7x",
        )
        for index in range(8)
    )


def _repair_receipt(root: Path, *relative_paths: str) -> None:
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in relative_paths:
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = next(value for value in receipt["artifacts"] if value["path"] == relative)
        artifact["size_bytes"] = path.stat().st_size
        artifact["sha256"] = digest
        if relative == "result.json":
            receipt["result_sha256"] = digest
        if relative == "ledger.sqlite":
            receipt["ledger_sha256"] = digest
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_weight_placement_runner_builds_and_replays_a_closed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_seqax_weight_placement_contract(_runtime())
    prepared = prepare_weight_placement_candidates(contract)
    devices = _fake_devices()
    original_jax_devices = placement_runner.jax.devices
    repository_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def source_state(_repository_root: Path, output_root: Path):
        (output_root / "source_diff.patch").write_bytes(b"")
        placement_runner._write_json(
            output_root / "source_state.json",
            {
                "git_commit": commit,
                "git_dirty": False,
                "git_status": [],
                "source_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "uv_lock_sha256": placement_runner._sha256(repository_root / "uv.lock"),
                "python_executable": "/test/python",
            },
        )
        return ()

    def compile_candidate(value, _host_inputs, _devices):
        stablehlo, compiler_hlo = _synthetic_hlo(value)

        def execute(*_inputs):
            return (np.zeros((2, 1, 16), dtype=np.float32),)

        return CompiledPlacement(
            prepared=value,
            executable=execute,
            mesh=None,
            stablehlo=stablehlo,
            compiler_hlo=compiler_hlo,
        )

    def memory_observations(_contract, _repository_root):
        host_inputs = tuple(
            np.asarray(value)
            for value in seqax_forward_inputs(seed=contract.timing_seed, **contract.parameters)
        )
        observations = []
        for value in prepared:
            stablehlo, compiler_hlo = _synthetic_hlo(value)
            output = interpret_distributed_program(value.distributed, host_inputs)[0]
            observations.append(
                SeqaxWeightResidencyObservation(
                    candidate=value.candidate.name,
                    runtime=contract.runtime,
                    devices=_devices(),
                    distributed_schedule_sha256=value.plan.distributed_schedule_sha256,
                    physical_schedule_sha256=value.plan.physical_schedule_sha256,
                    pallas_source_sha256=value.plan.source_sha256(),
                    stablehlo_sha256=placement_runner._text_sha256(stablehlo),
                    compiler_hlo_sha256=placement_runner._text_sha256(compiler_hlo),
                    source_commit=commit,
                    source_manifest=placement_runner._source_manifest(),
                    timing_input_sha256=arrays_sha256(host_inputs),
                    output_sha256=array_sha256(output),
                    parameter_bytes_per_device=value.candidate.expected_parameter_bytes_per_device,
                    device_bytes_limit=(100_000_000,) * 8,
                    peak_bytes_in_use=(1_000_000,) * 8,
                    largest_allocation_bytes=(100_000,) * 8,
                    isolated_process=True,
                    fits_observed_device_memory=True,
                )
            )
        return tuple(observations)

    def correctness(root, _contract, _compiled):
        _saved_contract, _saved_prepared, records = _correctness_fixture(root)
        return records

    def observations(_contract, _compiled, _inputs, orders):
        result = []
        for round_index, order in enumerate(orders):
            for position, candidate in enumerate(order):
                duration = 1_000 if candidate == "sharded" else 900
                result.append(
                    placement_runner.SeqaxPallasRoundObservation(
                        round_index=round_index,
                        position=position,
                        candidate=candidate,
                        samples_ns=(duration,) * 5,
                        median_ns=float(duration),
                    )
                )
        return tuple(result)

    monkeypatch.setattr(placement_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(placement_runner, "_runtime_identity", lambda: contract.runtime)
    monkeypatch.setattr(placement_runner, "_isolated_memory_observations", memory_observations)
    monkeypatch.setattr(
        placement_runner.jax,
        "devices",
        lambda backend=None: original_jax_devices("cpu") if backend == "cpu" else devices,
    )
    monkeypatch.setattr(placement_runner, "_validate_devices", lambda _devices, _contract: None)
    monkeypatch.setattr(placement_runner, "_source_state", source_state)
    monkeypatch.setattr(placement_runner, "_compile", compile_candidate)
    monkeypatch.setattr(
        placement_runner,
        "_resident_inputs",
        lambda host_inputs, _prepared, _mesh: host_inputs,
    )
    monkeypatch.setattr(placement_runner, "_candidate_correctness", correctness)
    monkeypatch.setattr(placement_runner, "_timing_observations", observations)
    monkeypatch.setattr(placement_runner.jax, "block_until_ready", lambda value: value)

    root = tmp_path / "run"
    result = run_seqax_weight_placement(root, contract)
    assert result.winner == "embedding-mlp"
    assert (root / "receipt.json").is_file()
    assert validate_seqax_weight_placement(root, contract) == result

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert validate_seqax_weight_placement(relocated, contract) == result

    changed_result = tmp_path / "changed-result"
    shutil.copytree(root, changed_result)
    payload = json.loads((changed_result / "result.json").read_text())
    payload["winner"] = None
    (changed_result / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_result, "result.json")
    with pytest.raises(ValueError, match="SELECTION_REPLAY_MISMATCH"):
        validate_seqax_weight_placement(changed_result, contract)

    changed_ledger = tmp_path / "changed-ledger"
    shutil.copytree(root, changed_ledger)
    with sqlite3.connect(changed_ledger / "ledger.sqlite") as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = 'timed'",
            ("0" * 64,),
        )
    _repair_receipt(changed_ledger, "ledger.sqlite")
    with pytest.raises(ValueError, match="LEDGER_PAYLOAD_MISMATCH"):
        validate_seqax_weight_placement(changed_ledger, contract)

    changed_memory = tmp_path / "changed-memory"
    shutil.copytree(root, changed_memory)
    memory = json.loads((changed_memory / "memory.json").read_text())
    memory[1]["stablehlo_sha256"] = "0" * 64
    (changed_memory / "memory.json").write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n")
    result_payload = json.loads((changed_memory / "result.json").read_text())
    result_payload["memory"] = memory
    (changed_memory / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_memory, "memory.json", "result.json")
    with pytest.raises(ValueError, match="MEMORY_REPLAY_MISMATCH"):
        validate_seqax_weight_placement(changed_memory, contract)

    changed_contract = tmp_path / "changed-contract"
    shutil.copytree(root, changed_contract)
    contract_payload = json.loads((changed_contract / "contract.json").read_text())
    contract_payload["runtime"]["python"] = "forged"
    (changed_contract / "contract.json").write_text(
        json.dumps(contract_payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_contract, "contract.json")
    with pytest.raises(ValueError, match="CONTRACT_MISMATCH"):
        validate_seqax_weight_placement(changed_contract, contract)
