from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tpu_cake.seqax_weight_confirmation_runner as confirmation_runner
import tpu_cake.seqax_weight_placement_runner as placement_runner
from tpu_cake.contracts import RuntimeIdentity, SourceFileContract
from tpu_cake.dtensor_interpreter import interpret_distributed_program
from tpu_cake.identity import arrays_sha256
from tpu_cake.seqax_pallas_search import SeqaxPallasRoundObservation
from tpu_cake.seqax_weight_confirmation import (
    SeqaxWeightConfirmationContract,
    confirmation_orders,
    confirmation_statistics,
    default_seqax_weight_confirmation_contract,
)
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementName,
    SeqaxWeightPlacementPlan,
)
from tpu_cake.seqax_weight_placement_runner import (
    CompiledPlacement,
    _physical_collective_counts,
)
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python="3.12.3",
        jax="0.11.0",
        jaxlib="0.11.0",
        libtpu="0.0.44.1",
        xla=" --xla_tpu_use_enhanced_launch_barrier=true",
    )


def _rounds(
    contract: SeqaxWeightConfirmationContract,
    *,
    baseline_ns: int,
    candidate_ns: int,
) -> tuple[SeqaxPallasRoundObservation, ...]:
    result = []
    durations = {
        SeqaxWeightPlacementName.SHARDED: baseline_ns,
        SeqaxWeightPlacementName.EMBEDDING_MLP: candidate_ns,
    }
    for round_index, order in enumerate(confirmation_orders(contract)):
        for position, candidate in enumerate(order):
            duration = durations[candidate]
            result.append(
                SeqaxPallasRoundObservation(
                    round_index=round_index,
                    position=position,
                    candidate=candidate,
                    samples_ns=(duration,) * contract.measured_iterations,
                    median_ns=float(duration),
                )
            )
    return tuple(result)


def test_confirmation_contract_is_canonical_and_tracked() -> None:
    contract = default_seqax_weight_confirmation_contract(_runtime())
    parsed = SeqaxWeightConfirmationContract.model_validate_json(
        contract.model_dump_json(exclude_computed_fields=True)
    )
    tracked = SeqaxWeightConfirmationContract.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "contracts"
            / "seqax-weight-placement-confirmation-tpu7x.json"
        ).read_text()
    )

    assert contract == parsed == tracked
    assert contract.confirmation_id == parsed.confirmation_id
    assert len(confirmation_orders(contract)) == 32
    assert confirmation_orders(contract)[:4] == (
        (SeqaxWeightPlacementName.SHARDED, SeqaxWeightPlacementName.EMBEDDING_MLP),
        (SeqaxWeightPlacementName.EMBEDDING_MLP, SeqaxWeightPlacementName.SHARDED),
        (SeqaxWeightPlacementName.SHARDED, SeqaxWeightPlacementName.EMBEDDING_MLP),
        (SeqaxWeightPlacementName.EMBEDDING_MLP, SeqaxWeightPlacementName.SHARDED),
    )


def test_confirmation_contract_rejects_second_look_and_provenance_drift() -> None:
    contract = default_seqax_weight_confirmation_contract(_runtime())
    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["confidence_level"] = 0.95
    with pytest.raises(ValueError, match="measurement protocol"):
        SeqaxWeightConfirmationContract.model_validate(payload)

    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["allow_further_retry"] = True
    with pytest.raises(ValueError, match="measurement protocol"):
        SeqaxWeightConfirmationContract.model_validate(payload)

    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["source_diagnostic_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        SeqaxWeightConfirmationContract.model_validate(payload)

    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["accepted_search_plans"][0]["compiler_hlo_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="search plan identities"):
        SeqaxWeightConfirmationContract.model_validate(payload)


def test_confirmation_binds_the_accepted_search_plan_identities() -> None:
    contract = default_seqax_weight_confirmation_contract(_runtime())
    search_result = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "imported"
            / "seqax-weight-placement-6a83e59"
            / "result.json"
        ).read_text()
    )
    plans = tuple(
        SeqaxWeightPlacementPlan.model_validate(value) for value in search_result["plans"]
    )

    confirmation_runner._validate_accepted_search_plans(contract, plans)
    changed = plans[0].model_copy(update={"stablehlo_sha256": "0" * 64})
    with pytest.raises(ValueError, match="ACCEPTED_SEARCH_PLAN_MISMATCH"):
        confirmation_runner._validate_accepted_search_plans(contract, (changed, plans[1]))


def test_confirmation_source_manifest_is_bound_to_the_recorded_git_blob(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    source = repository / "src" / "tpu_cake" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=TPU Cake Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = (
        SourceFileContract(
            path="tpu_cake/example.py",
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
    )

    confirmation_runner._validate_source_blobs(repository, commit, manifest)
    changed = (manifest[0].model_copy(update={"sha256": "0" * 64}),)
    with pytest.raises(ValueError, match="SOURCE_BLOB_MISMATCH"):
        confirmation_runner._validate_source_blobs(repository, commit, changed)


def test_confirmation_requires_99_percent_lower_bound_above_three_percent() -> None:
    contract = default_seqax_weight_confirmation_contract(_runtime())
    confirmed = confirmation_statistics(
        contract,
        _rounds(contract, baseline_ns=1_000, candidate_ns=950),
    )
    rejected = confirmation_statistics(
        contract,
        _rounds(contract, baseline_ns=1_000, candidate_ns=975),
    )

    assert confirmed.median_improvement == pytest.approx(0.05)
    assert confirmed.improvement_confidence_interval == pytest.approx((0.05, 0.05))
    assert confirmed.confidence_level == 0.99
    assert confirmed.confirmed is True
    assert rejected.median_improvement == pytest.approx(0.025)
    assert rejected.confirmed is False


def test_confirmation_rejects_missing_reordered_and_malformed_rounds() -> None:
    contract = default_seqax_weight_confirmation_contract(_runtime())
    rounds = _rounds(contract, baseline_ns=1_000, candidate_ns=950)
    with pytest.raises(ValueError, match="observation count"):
        confirmation_statistics(contract, rounds[:-1])

    reordered = (rounds[1], rounds[0], *rounds[2:])
    with pytest.raises(ValueError, match="execution order"):
        confirmation_statistics(contract, reordered)

    malformed = rounds[0].model_copy(update={"samples_ns": (1_000,) * 3})
    with pytest.raises(ValueError, match="sample count"):
        confirmation_statistics(contract, (malformed, *rounds[1:]))


def _synthetic_hlo(prepared) -> tuple[str, str]:
    tiles = tuple(
        (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
        for operation in prepared.physical.walk()
        if operation.name == "tpu_schedule.mxu_einsum"
    )
    stablehlo = "module @fixture {\n  func.func public @main() {\n"
    stablehlo += "".join(
        f"    %p{index} = stablehlo.custom_call @tpu_custom_call() "
        '{kernel_name = "seqax_named_einsum"}\n'
        for index in range(9)
    )
    stablehlo += "".join(
        f"    %g{index} = stablehlo.all_gather %p0\n"
        for index in range(prepared.candidate.expected_stablehlo_all_gathers)
    )
    stablehlo += "  }\n}\n"
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
    all_gathers, reduce_scatters = _physical_collective_counts(prepared.physical)
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


def test_confirmation_runner_builds_relocates_and_rejects_repaired_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = default_seqax_weight_confirmation_contract(_runtime())
    devices = _fake_devices()
    original_jax_devices = confirmation_runner.jax.devices
    repository_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    seed_inputs = {
        arrays_sha256(
            tuple(
                np.asarray(value)
                for value in seqax_forward_inputs(seed=seed, **contract.parameters)
            )
        ): seed
        for seed in contract.correctness_seeds
    }

    def source_state(_repository_root: Path, output_root: Path):
        (output_root / "source_diff.patch").write_bytes(b"")
        confirmation_runner._write_json(
            output_root / "source_state.json",
            {
                "git_commit": commit,
                "git_dirty": False,
                "git_status": [],
                "source_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "uv_lock_sha256": confirmation_runner._sha256(repository_root / "uv.lock"),
                "python_executable": "/test/python",
            },
        )
        return ()

    def compile_candidate(value, _host_inputs, _devices):
        stablehlo, compiler_hlo = _synthetic_hlo(value)

        def execute(*inputs):
            arrays = tuple(np.asarray(item) for item in inputs)
            output = interpret_distributed_program(value.distributed, arrays)[0]
            seed = seed_inputs[arrays_sha256(arrays)]
            if seed in contract.correctness_seeds[3:]:
                output = output + np.float32(0.1)
            return (output,)

        return CompiledPlacement(
            prepared=value,
            executable=execute,
            mesh=None,
            stablehlo=stablehlo,
            compiler_hlo=compiler_hlo,
        )

    def observations(_contract, _compiled, _resident):
        return _rounds(contract, baseline_ns=1_000, candidate_ns=950)

    monkeypatch.setattr(confirmation_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(confirmation_runner, "_runtime_identity", lambda: contract.runtime)
    monkeypatch.setattr(
        confirmation_runner.jax,
        "devices",
        lambda backend=None: original_jax_devices("cpu") if backend == "cpu" else devices,
    )
    monkeypatch.setattr(confirmation_runner, "_validate_devices", lambda _devices, _contract: None)
    monkeypatch.setattr(confirmation_runner, "_source_state", source_state)
    monkeypatch.setattr(confirmation_runner, "_validate_source_blobs", lambda *_args: None)
    monkeypatch.setattr(
        confirmation_runner,
        "_validate_accepted_search_plans",
        lambda *_args: None,
    )
    monkeypatch.setattr(confirmation_runner, "_compile", compile_candidate)
    monkeypatch.setattr(
        confirmation_runner,
        "_resident_inputs",
        lambda host_inputs, _prepared, _mesh: host_inputs,
    )
    monkeypatch.setattr(
        placement_runner,
        "_resident_inputs",
        lambda host_inputs, _prepared, _mesh: host_inputs,
    )
    monkeypatch.setattr(confirmation_runner, "_timing_observations", observations)
    monkeypatch.setattr(confirmation_runner.jax, "block_until_ready", lambda value: value)

    root = tmp_path / "run"
    result = confirmation_runner.run_seqax_weight_confirmation(root, contract)
    assert result.winner is SeqaxWeightPlacementName.EMBEDDING_MLP
    assert confirmation_runner.validate_seqax_weight_confirmation(root, contract) == result

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert confirmation_runner.validate_seqax_weight_confirmation(relocated, contract) == result

    changed_result = tmp_path / "changed-result"
    shutil.copytree(root, changed_result)
    payload = json.loads((changed_result / "result.json").read_text())
    payload["winner"] = None
    (changed_result / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_result, "result.json")
    with pytest.raises(ValueError):
        confirmation_runner.validate_seqax_weight_confirmation(changed_result, contract)

    changed_ledger = tmp_path / "changed-ledger"
    shutil.copytree(root, changed_ledger)
    with sqlite3.connect(changed_ledger / "ledger.sqlite") as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = 'timed'",
            ("0" * 64,),
        )
    _repair_receipt(changed_ledger, "ledger.sqlite")
    with pytest.raises(ValueError, match="LEDGER_PAYLOAD_MISMATCH"):
        confirmation_runner.validate_seqax_weight_confirmation(changed_ledger, contract)

    changed_output = tmp_path / "changed-output"
    shutil.copytree(root, changed_output)
    output_path = changed_output / "post_timing_outputs" / "embedding-mlp.npy"
    output = np.load(output_path, allow_pickle=False)
    output.flat[0] += np.float32(1e-4)
    np.save(output_path, output, allow_pickle=False)
    _repair_receipt(changed_output, "post_timing_outputs/embedding-mlp.npy")
    with pytest.raises(ValueError, match="POST_TIMING_REPLAY_MISMATCH"):
        confirmation_runner.validate_seqax_weight_confirmation(changed_output, contract)

    changed_contract = tmp_path / "changed-contract"
    shutil.copytree(root, changed_contract)
    payload = json.loads((changed_contract / "contract.json").read_text())
    payload["analysis_index"] = 3
    (changed_contract / "contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _repair_receipt(changed_contract, "contract.json")
    with pytest.raises(ValueError, match="measurement protocol"):
        confirmation_runner.validate_seqax_weight_confirmation(changed_contract, contract)
