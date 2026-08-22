from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

import tpu_cake.matmul_collective_surface_calibration_verifier as verifier
from tpu_cake.matmul_collective_surface_calibration_verifier import (
    _EXPECTED_DESIGN_FILE_SHA256,
    _EXPECTED_DESIGN_ID,
    _EXPECTED_PROTOCOL_FILE_SHA256,
    _EXPECTED_PROTOCOL_ID,
    _SCENARIOS,
    _STRATEGIES,
    _array_sha256,
    _canonical_relative_path,
    _derive_observations,
    _derive_seal,
    _expected_archive_files,
    _feature_row,
    _file_sha256,
    _identity_sha256,
    _load_exact_npy,
    _nonnegative_affine_fit,
    _pairs_to_dict,
    _pretty_json_bytes,
    _read_json,
    _resident_pair_sha256,
    _scenario_order,
    _semantic_sha256,
    _strategy_order,
    _tar_inventory,
    _validate_global_timeline,
    _validate_identity_and_claim,
    _validate_samples,
    _validate_warmups,
    _warmup_strategy_order,
)


def _residents(nonce: str = "a" * 64, worker_pid: int = 123) -> dict[str, dict[str, object]]:
    return {
        scenario: {
            "scenario_name": scenario,
            "xla_compile_record_sha256": f"{index:064x}",
            "pallas_compile_record_sha256": f"{index + 16:064x}",
            "invocation_nonce": nonce,
            "worker_pid": worker_pid,
        }
        for index, scenario in enumerate(_SCENARIOS)
    }


def _warmups(
    residents: dict[str, dict[str, object]],
    nonce: str = "a" * 64,
    worker_pid: int = 123,
    start_ns: int = 1,
) -> list[dict[str, object]]:
    values = []
    repetitions: dict[tuple[str, str], int] = {}
    clock = start_ns
    for scenario_index, scenario in enumerate(_SCENARIOS):
        for strategy in _warmup_strategy_order(scenario_index):
            key = (scenario, strategy)
            repetitions[key] = repetitions.get(key, 0) + 1
            values.append(
                {
                    "sequence": len(values) + 1,
                    "scenario_name": scenario,
                    "scenario_position": scenario_index + 1,
                    "strategy": strategy,
                    "strategy_repetition": repetitions[key],
                    "resident_pair_sha256": _resident_pair_sha256(residents[scenario]),
                    "invocation_nonce": nonce,
                    "worker_pid": worker_pid,
                    "start_ns": clock,
                    "stop_ns": clock + 1,
                }
            )
            clock += 2
    return values


def _samples(
    residents: dict[str, dict[str, object]],
    nonce: str = "a" * 64,
    worker_pid: int = 123,
    start_ns: int = 1,
) -> list[dict[str, object]]:
    values = []
    clock = start_ns
    for round_index in range(16):
        for scenario_position, scenario in enumerate(_scenario_order(round_index), start=1):
            for arm_position, strategy in enumerate(_strategy_order(round_index), start=1):
                for call_index in range(5):
                    duration = 10_000 + 101 * int(scenario.removeprefix("calibration-"))
                    duration += 37 * _STRATEGIES.index(strategy) + 3 * round_index + call_index
                    values.append(
                        {
                            "sequence": len(values) + 1,
                            "round_index": round_index,
                            "scenario_name": scenario,
                            "scenario_position": scenario_position,
                            "strategy": strategy,
                            "arm_position": arm_position,
                            "call_index": call_index,
                            "resident_pair_sha256": _resident_pair_sha256(residents[scenario]),
                            "invocation_nonce": nonce,
                            "worker_pid": worker_pid,
                            "start_ns": clock,
                            "stop_ns": clock + duration,
                            "duration_ns": duration,
                        }
                    )
                    clock += duration + 1
    return values


def test_standalone_verifier_has_only_declared_third_party_imports() -> None:
    path = Path("src/tpu_cake/matmul_collective_surface_calibration_verifier.py")
    tree = ast.parse(path.read_text())
    roots = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "tpu_cake" not in roots
    assert roots - {"__future__", "ml_dtypes", "numpy"} <= set(
        __import__("sys").stdlib_module_names
    )


def test_canonical_contract_identities_are_pinned() -> None:
    protocol = Path("contracts/matmul-collective-surface-calibration-v1.json")
    design = Path("contracts/matmul-collective-surface-design-v1.json")
    assert _file_sha256(protocol) == _EXPECTED_PROTOCOL_FILE_SHA256
    assert _identity_sha256(json.loads(protocol.read_text())) == _EXPECTED_PROTOCOL_ID
    assert _file_sha256(design) == _EXPECTED_DESIGN_FILE_SHA256
    assert _identity_sha256(json.loads(design.read_text())) == _EXPECTED_DESIGN_ID


@pytest.mark.parametrize(
    "payload,error",
    [
        ('{"a":1,"a":2}', "DUPLICATE_KEY"),
        ('{"a":NaN}', "JSON_CONSTANT"),
        ('{"a":1e999}', "JSON_NONFINITE"),
        ("[]", "JSON_OBJECT_REQUIRED"),
    ],
)
def test_strict_json_rejects_ambiguous_or_nonfinite_payloads(
    tmp_path: Path, payload: str, error: str
) -> None:
    path = tmp_path / "value.json"
    path.write_text(payload)
    with pytest.raises((TypeError, ValueError), match=error):
        _read_json(path)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/./b", "a//b", ""])
def test_canonical_relative_path_rejects_archive_escapes(path: str) -> None:
    with pytest.raises((TypeError, ValueError), match="PATH_INVALID"):
        _canonical_relative_path(path, "PATH")


def test_exact_npy_rejects_trailing_bytes_and_wrong_abi(tmp_path: Path) -> None:
    path = tmp_path / "array.npy"
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(path, array, allow_pickle=False)
    assert np.array_equal(_load_exact_npy(path, (3, 4)), array)
    path.write_bytes(path.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match="NPY_STRUCTURE_INVALID"):
        _load_exact_npy(path, (3, 4))


def test_array_identity_binds_dtype_shape_and_bytes() -> None:
    array = np.arange(8, dtype=np.float32).reshape(2, 4)
    assert _array_sha256(array) != _array_sha256(array.reshape(4, 2))
    assert _array_sha256(array) != _array_sha256(array.astype(np.float64))


def test_timing_sequences_accept_exact_fixture_and_reject_mutations() -> None:
    residents = _residents()
    warmups = _warmups(residents)
    samples = _samples(residents, start_ns=10_000)
    assert len(_validate_warmups(warmups, residents, "a" * 64, 123)) == 320
    assert len(_validate_samples(samples, residents, "a" * 64, 123)) == 2560

    reordered = [dict(value) for value in samples]
    reordered[1]["call_index"] = 4
    with pytest.raises(ValueError, match="SAMPLE_SEQUENCE_MISMATCH"):
        _validate_samples(reordered, residents, "a" * 64, 123)

    wrong_pid = [dict(value) for value in warmups]
    wrong_pid[-1]["worker_pid"] = 124
    with pytest.raises(ValueError, match="WARMUP_SEQUENCE_MISMATCH"):
        _validate_warmups(wrong_pid, residents, "a" * 64, 123)


def test_global_timeline_rejects_overlap() -> None:
    before = ({"phase": "before_timing", "start_ns": 0, "stop_ns": 10},)
    warmups = ({"start_ns": 9, "stop_ns": 11},)
    after = ({"phase": "after_timing", "start_ns": 12, "stop_ns": 13},)
    with pytest.raises(ValueError, match="CLOCK_ORDER_MISMATCH"):
        _validate_global_timeline((*before, *after), warmups, ())


def test_observations_use_median_of_each_five_call_round() -> None:
    observations = _derive_observations(tuple(_samples(_residents())))
    first = observations[0]
    assert first["round_medians_ns"] == [10_002 + 3 * index for index in range(16)]
    assert first["median_ns"] == 10_024.5


def test_nonnegative_fit_checks_all_active_sets() -> None:
    matrix = np.asarray([_feature_row(s, t) for s in _SCENARIOS for t in _STRATEGIES])
    expected = np.asarray([100.0, 120.0, 3.0, 2.0, 5.0, 7.0])
    observed = matrix @ expected
    fitted = _nonnegative_affine_fit(matrix, observed)
    assert np.allclose(fitted, expected, rtol=1e-10, atol=1e-10)


def test_full_seal_replay_runs_10000_global_round_bootstraps() -> None:
    samples = tuple(_samples(_residents()))
    seal = _derive_seal(samples, _EXPECTED_PROTOCOL_ID, "1" * 64, "2" * 64)
    assert len(seal["observations"]) == 32
    assert len(seal["holdout_predictions"]) == 8
    assert len(seal["strategy_predictions"]) == 4
    assert seal["bootstrap_sample_count"] == 10_000
    assert all(
        len(seal[key]) == 64
        for key in (
            "bootstrap_index_sha256",
            "bootstrap_coefficient_sha256",
            "bootstrap_prediction_sha256",
            "bootstrap_improvement_sha256",
        )
    )
    assert seal["holdout_authorization"] in {
        "pending_independent_replay",
        "denied_prediction_interval_width",
    }


def test_semantic_claim_identity_is_length_prefixed() -> None:
    assert _semantic_sha256("ab", "c") != _semantic_sha256("a", "bc")


def test_duplicate_pair_parser_rejects_before_dict_overwrite() -> None:
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        _pairs_to_dict([("field", 1), ("field", 2)])


def test_attempt_claim_requires_live_permanent_registry_copy(tmp_path: Path) -> None:
    receipt_sha256 = "1" * 64
    claim_key = _semantic_sha256(receipt_sha256, "calibration-timing-v1")
    registry = tmp_path / "registry"
    registry.mkdir()
    root = tmp_path / "archive"
    root.mkdir()
    output_root = str(root)
    protocol = {
        "correctness_parent": {"receipt_sha256": receipt_sha256, "archive_root_name": "parent"},
        "attempt_registry_root": str(registry),
    }
    protocol_id = "2" * 64
    authority = {"source": {"source_commit": "3" * 40}}
    source_authority_sha256 = "4" * 64
    execution_authority_sha256 = "5" * 64
    attempt_id = "6" * 64
    claim_path = registry / f"{claim_key}.json"
    claim = {
        "schema_version": "matmul-collective-surface-calibration-attempt-claim-v1",
        "attempt_id": attempt_id,
        "protocol_id": protocol_id,
        "permanent_claim_key": claim_key,
        "correctness_parent_receipt_sha256": receipt_sha256,
        "source_commit": "3" * 40,
        "output_root": output_root,
        "state": "claimed",
    }
    claim_bytes = _pretty_json_bytes(claim)
    claim_path.write_bytes(claim_bytes)
    (root / "attempt_claim.json").write_bytes(claim_bytes)
    identity = {
        "attempt_id": attempt_id,
        "protocol_id": protocol_id,
        "execution_authority_sha256": execution_authority_sha256,
        "source_authority_sha256": source_authority_sha256,
        "attempt_claim_path": str(claim_path),
        "attempt_claim_sha256": __import__("hashlib").sha256(claim_bytes).hexdigest(),
        "output_root": output_root,
        "parent_correctness_root": str(root / "parent" / "parent"),
        "compilation_cache_path": str(tmp_path / "cache"),
    }
    (root / "run_identity.json").write_text(json.dumps(identity))
    assert (
        _validate_identity_and_claim(
            root,
            protocol,
            protocol_id,
            authority,
            source_authority_sha256,
            execution_authority_sha256,
        )
        == identity
    )
    claim_path.unlink()
    with pytest.raises(ValueError, match="EXTERNAL_CLAIM_MISSING"):
        _validate_identity_and_claim(
            root,
            protocol,
            protocol_id,
            authority,
            source_authority_sha256,
            execution_authority_sha256,
        )


def test_closed_world_inventory_does_not_admit_unlisted_files(tmp_path: Path) -> None:
    parent_root = tmp_path / "parent" / "correctness-root"
    parent_root.mkdir(parents=True)
    (parent_root / "manifest.json").write_text("{}")
    protocol = {
        "correctness_parent": {
            "archive_filename": "correctness.tar.zst",
            "archive_root_name": "correctness-root",
        }
    }
    expected = _expected_archive_files(tmp_path, protocol)
    assert "parent/correctness-root/manifest.json" in expected
    assert "parent/correctness.tar.zst" in expected
    assert "unexpected.json" not in expected
    assert sum(path.startswith("continuity/") for path in expected) == 64
    assert sum(path.startswith("outputs/") for path in expected) == 64
    assert sum(path.startswith("oracles/") for path in expected) == 16


def _compressed_tar(
    tmp_path: Path,
    members: tuple[tuple[str, bytes | None, str], ...],
) -> Path:
    tar_path = tmp_path / "parent.tar"
    with tarfile.open(tar_path, "w") as stream:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "directory":
                info.type = tarfile.DIRTYPE
                stream.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                stream.addfile(info)
            else:
                assert payload is not None
                info.size = len(payload)
                stream.addfile(info, __import__("io").BytesIO(payload))
    archive = tmp_path / "parent.tar.zst"
    zstd = shutil.which("zstd")
    assert zstd is not None
    subprocess.run([zstd, "-q", "-f", str(tar_path), "-o", str(archive)], check=True)
    return archive


def _archive_limits(maximum_members: int = 10) -> dict[str, object]:
    return {
        "archive_root_name": "root",
        "archive_maximum_members": maximum_members,
        "archive_maximum_member_size_bytes": 1024,
        "archive_maximum_total_size_bytes": 4096,
    }


def test_parent_archive_inventory_counts_directories_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zstd = shutil.which("zstd")
    assert zstd is not None
    monkeypatch.setattr(verifier, "_ZSTD_PATH", Path(zstd))
    archive = _compressed_tar(
        tmp_path,
        (("root", None, "directory"), ("root/value", b"payload", "file")),
    )
    assert _tar_inventory(archive, _archive_limits()) == (
        ("root", 0, "directory"),
        ("root/value", 7, __import__("hashlib").sha256(b"payload").hexdigest()),
    )
    with pytest.raises(ValueError, match="ARCHIVE_LIMIT"):
        _tar_inventory(archive, _archive_limits(maximum_members=1))


@pytest.mark.parametrize(
    "member",
    [
        ("root/../escape", b"payload", "file"),
        ("root/link", None, "symlink"),
    ],
)
def test_parent_archive_inventory_rejects_unsafe_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member: tuple[str, bytes | None, str],
) -> None:
    zstd = shutil.which("zstd")
    assert zstd is not None
    monkeypatch.setattr(verifier, "_ZSTD_PATH", Path(zstd))
    archive = _compressed_tar(
        tmp_path,
        (("root", None, "directory"), member),
    )
    with pytest.raises(ValueError, match="ARCHIVE_(PATH|TYPE)_INVALID"):
        _tar_inventory(archive, _archive_limits())
