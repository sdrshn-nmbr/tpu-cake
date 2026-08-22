from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tpu_cake.matmul_collective_surface_correctness import (
    correctness_sentinel_coordinates,
    make_correctness_operand_shard,
)
from tpu_cake.matmul_collective_surface_correctness_executor import (
    CORRECTNESS_EXECUTABLE_DEPENDENCIES,
)
from tpu_cake.matmul_collective_surface_correctness_oracle import make_correctness_oracle
from tpu_cake.matmul_collective_surface_correctness_verifier import (
    _SOURCE_DEPENDENCIES,
    _array_sha256,
    _load_exact_npy,
    _operand_identity,
    _oracle,
    _pretty_json_bytes,
    _read_json,
    _semantic_compiler_hlo,
    _sentinel_coordinates,
    _validate_archive_tree,
    _validate_design,
    _validate_identity_and_claim,
    _validate_protocol,
    _validate_saved_array,
)


@pytest.mark.parametrize("invalid", ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'))
def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path, invalid: str
) -> None:
    path = tmp_path / "value.json"
    path.write_text(invalid)

    with pytest.raises(ValueError, match="JSON_(DUPLICATE_KEY|CONSTANT|NONFINITE)"):
        _read_json(path)


def test_protocol_and_design_are_replayed_without_producer_models(tmp_path: Path) -> None:
    protocol_path = Path("contracts/matmul-collective-surface-correctness-v1.json")
    design_path = Path("contracts/matmul-collective-surface-design-v1.json")
    recorded_protocol = tmp_path / "protocol.json"
    recorded_design = tmp_path / "design.json"
    recorded_protocol.write_text(
        json.dumps(json.loads(protocol_path.read_text()), indent=2, sort_keys=True) + "\n"
    )
    recorded_design.write_text(
        json.dumps(json.loads(design_path.read_text()), indent=2, sort_keys=True) + "\n"
    )

    protocol, protocol_id, protocol_file_hash = _validate_protocol(protocol_path, recorded_protocol)
    design = _validate_design(design_path, recorded_design, protocol)

    assert len(protocol_id) == len(protocol_file_hash) == 64
    assert protocol_file_hash != hashlib.sha256(recorded_protocol.read_bytes()).hexdigest()
    assert [scenario["name"] for scenario in design["scenarios"]] == protocol["scenarios"]

    mutated = json.loads(recorded_protocol.read_text())
    mutated["patterns"]["constant_formula"] = "C=0"
    recorded_protocol.write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match="PROTOCOL_CONTRACT_MISMATCH"):
        _validate_protocol(recorded_protocol, recorded_protocol)


def test_standalone_verifier_pins_the_complete_executable_source_closure() -> None:
    assert _SOURCE_DEPENDENCIES == CORRECTNESS_EXECUTABLE_DEPENDENCIES


def test_attempt_claim_copy_is_exact_and_bound_to_permanent_registry_key(
    tmp_path: Path,
) -> None:
    protocol_id = "1" * 64
    execution_authority = "2" * 64
    source_commit = "3" * 40
    attempt_id = "4" * 64
    split = "calibration"
    registry = tmp_path / "registry"
    protocol = {"attempt_registry_root": str(registry)}
    authority = {"source": {"source_commit": source_commit}}
    claim_key = hashlib.sha256(f"{protocol_id}:{split}".encode()).hexdigest()
    claim_path = registry / f"{claim_key}.json"
    claim = {
        "schema_version": "matmul-collective-surface-correctness-executor-v1",
        "attempt_id": attempt_id,
        "protocol_id": protocol_id,
        "split": split,
        "source_commit": source_commit,
        "output_root": str(tmp_path),
        "state": "claimed",
    }
    copied = tmp_path / "attempt_claim.json"
    copied.write_bytes(_pretty_json_bytes(claim))
    identity = {
        "attempt_id": attempt_id,
        "protocol_id": protocol_id,
        "split": split,
        "execution_authority_sha256": execution_authority,
        "attempt_claim_path": str(claim_path),
        "attempt_claim_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        "output_root": str(tmp_path),
    }
    (tmp_path / "run_identity.json").write_bytes(_pretty_json_bytes(identity))

    assert (
        _validate_identity_and_claim(
            tmp_path, protocol, protocol_id, authority, execution_authority
        )
        == identity
    )

    copied.write_bytes(_pretty_json_bytes({**claim, "state": "rebound"}))
    with pytest.raises(ValueError, match="ATTEMPT_CLAIM_MISMATCH"):
        _validate_identity_and_claim(
            tmp_path, protocol, protocol_id, authority, execution_authority
        )


def test_holdout_replay_fails_before_artifact_inspection(tmp_path: Path) -> None:
    identity = {
        "attempt_id": "1" * 64,
        "protocol_id": "2" * 64,
        "split": "holdout",
        "execution_authority_sha256": "3" * 64,
        "attempt_claim_path": "/unused",
        "attempt_claim_sha256": "4" * 64,
        "output_root": str(tmp_path),
    }
    (tmp_path / "run_identity.json").write_bytes(_pretty_json_bytes(identity))

    with pytest.raises(ValueError, match="HOLDOUT_NOT_AUTHORIZED"):
        _validate_identity_and_claim(
            tmp_path,
            {"attempt_registry_root": "/unused"},
            "2" * 64,
            {"source": {"source_commit": "5" * 40}},
            "3" * 64,
        )


def test_candidate_array_cannot_broadcast_across_scenario_rows(tmp_path: Path) -> None:
    relative = "outputs/candidate.npy"
    path = tmp_path / relative
    path.parent.mkdir()
    candidate = np.zeros((1, 4), dtype=np.float32)
    np.save(path, candidate, allow_pickle=False)
    record = {
        "path": relative,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "array_sha256": _array_sha256(candidate),
        "shape": [1, 4],
        "dtype": "float32",
        "numpy_dtype_str": "<f4",
        "nan_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
    }

    with pytest.raises(ValueError, match="OUTPUT_ABI_MISMATCH"):
        _validate_saved_array(tmp_path, record, relative, None, (3, 4))


def test_exact_npy_rejects_fortran_object_and_trailing_bytes(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npy"
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(valid, value, allow_pickle=False)
    np.testing.assert_array_equal(_load_exact_npy(valid, (3, 4)), value)

    trailing = tmp_path / "trailing.npy"
    trailing.write_bytes(valid.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match="NPY_STRUCTURE_INVALID"):
        _load_exact_npy(trailing, (3, 4))

    fortran = tmp_path / "fortran.npy"
    np.save(fortran, np.asfortranarray(value), allow_pickle=False)
    with pytest.raises(ValueError, match="NPY_STRUCTURE_INVALID"):
        _load_exact_npy(fortran, (3, 4))

    object_array = tmp_path / "object.npy"
    np.save(object_array, np.asarray([[object()]], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="NPY_STRUCTURE_INVALID"):
        _load_exact_npy(object_array, (1, 1))


@pytest.mark.parametrize(
    "pattern",
    ("constant", "one-hot-stripes", "signed-periodic", "block-diagonal", "low-rank"),
)
@pytest.mark.parametrize("role", ("lhs", "rhs"))
def test_independent_operand_hash_and_sentinels_match_producer(pattern: str, role: str) -> None:
    dimensions = {"m": 16, "k": 128, "n": 32}
    device = 3
    local_k = dimensions["k"] // 8
    shard = make_correctness_operand_shard(
        pattern,
        role,
        **dimensions,
        k_start=device * local_k,
        k_stop=(device + 1) * local_k,
    )
    digest, sentinels = _operand_identity(
        pattern,
        role,
        **dimensions,
        device=device,
        protocol_id="a" * 64,
        scenario_name="calibration-0",
    )
    coordinates = correctness_sentinel_coordinates(
        pattern,
        role,
        **dimensions,
        protocol_id="a" * 64,
        scenario_name="calibration-0",
        device_id=device,
    )

    assert digest == hashlib.sha256(shard.tobytes(order="C")).hexdigest()
    assert tuple(sentinels) == coordinates
    assert coordinates == _sentinel_coordinates(
        pattern,
        role,
        **dimensions,
        protocol_id="a" * 64,
        scenario_name="calibration-0",
        device_id=device,
    )
    for coordinate in coordinates:
        local = (
            (coordinate[0], coordinate[1] - device * local_k)
            if role == "lhs"
            else (coordinate[0] - device * local_k, coordinate[1])
        )
        expected = np.asarray(shard[local], dtype=np.dtype(ml_dtypes.bfloat16)).reshape(1)
        assert sentinels[coordinate] == expected.tobytes().hex()


@pytest.mark.parametrize(
    "pattern",
    ("constant", "one-hot-stripes", "signed-periodic", "block-diagonal", "low-rank"),
)
def test_independent_oracle_matches_producer_bitwise(pattern: str) -> None:
    expected = make_correctness_oracle(pattern, m=16, k=128, n=32)
    observed = _oracle(pattern, m=16, k=128, n=32)

    np.testing.assert_array_equal(observed, expected)
    assert _array_sha256(observed) == _array_sha256(expected)


def test_compiler_hlo_normalization_removes_only_unstable_metadata() -> None:
    value = (
        "HloModule test\n"
        "FileNames\n"
        "1 foo.py\n"
        "%entry (x: f32[]) -> f32[] {\n"
        "  ROOT x = f32[] parameter(0), stack_frame_id=17\n"
        "}\n"
    )

    assert _semantic_compiler_hlo(value) == (
        "HloModule test\n%entry (x: f32[]) -> f32[] {\n  ROOT x = f32[] parameter(0),\n}\n"
    )


def test_archive_tree_rejects_links_and_hardlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text(json.dumps({"ok": True}))
    symlink = tmp_path / "symlink"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="ARCHIVE_LINK"):
        _validate_archive_tree(tmp_path)
    symlink.unlink()

    hardlink = tmp_path / "hardlink"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="ARCHIVE_HARDLINK"):
        _validate_archive_tree(tmp_path)
