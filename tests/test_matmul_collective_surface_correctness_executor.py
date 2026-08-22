from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.matmul_collective_surface_correctness_executor import (
    SurfaceCorrectnessManifestEntry,
    SurfaceCorrectnessPhaseReceipt,
    SurfaceCorrectnessSourceAuthority,
    _attempt_claim_path,
    _copy_parent_compile_snapshot,
    _manifest_entries,
    _write_bytes_exclusive,
    execute_surface_correctness,
)
from tpu_cake.matmul_collective_surface_correctness_protocol import (
    default_matmul_collective_surface_correctness_protocol,
)
from tpu_cake.matmul_collective_surface_prediction import MatmulCollectiveSurfaceSplit


def test_source_authority_requires_exact_main_and_sorted_dependencies() -> None:
    dependency = {
        "path": "tpu_cake/a.py",
        "sha256": "a" * 64,
    }
    authority = SurfaceCorrectnessSourceAuthority(
        source_commit="1" * 40,
        origin_main_commit="1" * 40,
        remote_main_commit="1" * 40,
        runtime={"python": "3.12.3"},
        uv_lock_sha256="2" * 64,
        dependencies=(dependency,),
    )

    assert authority.authority_sha256
    with pytest.raises(ValueError, match="SOURCE_MAIN_MISMATCH"):
        SurfaceCorrectnessSourceAuthority.model_validate(
            authority.model_copy(update={"remote_main_commit": "3" * 40}).model_dump(
                mode="python", exclude_computed_fields=True
            )
        )


def test_receipt_phase_and_chain_are_fail_closed() -> None:
    payload = {
        "attempt_id": "1" * 64,
        "protocol_id": "2" * 64,
        "phase_sequence": 1,
        "phase": "correctness",
        "split": MatmulCollectiveSurfaceSplit.CALIBRATION,
        "parent_compile_manifest_file_sha256": "3" * 64,
        "evidence_file_sha256": "4" * 64,
        "evidence_sha256": "5" * 64,
        "artifact_set_sha256": "6" * 64,
        "ledger_snapshot_sha256": "7" * 64,
        "attempt_claim_path": "/evidence/claim.json",
        "attempt_claim_sha256": "8" * 64,
    }
    receipt = SurfaceCorrectnessPhaseReceipt(**payload)

    assert receipt.receipt_sha256
    with pytest.raises((ValidationError, ValueError)):
        SurfaceCorrectnessPhaseReceipt(
            **{
                **payload,
                "split": MatmulCollectiveSurfaceSplit.HOLDOUT,
                "phase_sequence": 6,
                "phase": "holdout_correctness",
            }
        )


def test_manifest_path_and_exclusive_write_reject_rebinding(tmp_path) -> None:
    with pytest.raises(ValueError, match="MANIFEST_PATH_INVALID"):
        SurfaceCorrectnessManifestEntry(path="../escape", size_bytes=1, sha256="a" * 64)

    path = tmp_path / "artifact.json"
    _write_bytes_exclusive(path, b"first\n")
    with pytest.raises(FileExistsError):
        _write_bytes_exclusive(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_manifest_entries_use_canonical_relative_path_order(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    (tmp_path / "parent").mkdir()
    (tmp_path / "parent-extraction").mkdir()
    (tmp_path / "parent" / "artifact").write_bytes(b"parent")
    (tmp_path / "parent-extraction" / "artifact").write_bytes(b"extraction")

    entries = _manifest_entries(tmp_path)

    assert tuple(entry.path for entry in entries) == (
        "parent-extraction/artifact",
        "parent/artifact",
    )


def test_parent_copy_requires_verified_source_before_writing(tmp_path, monkeypatch) -> None:
    def fake_copytree(_source, destination, *, symlinks):
        assert not symlinks
        destination.mkdir()
        (destination / "manifest.json").write_text("{}")

    protocol = default_matmul_collective_surface_correctness_protocol()
    calls = []
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_correctness_executor._verify_parent_compile_snapshot",
        lambda root, _protocol: calls.append(root) or {},
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_correctness_executor.shutil.copytree",
        fake_copytree,
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_correctness_executor.json.loads",
        lambda _value: {
            "identity": {
                "attempt_claim_path": str(tmp_path / "missing-claim.json"),
                "attempt_claim_sha256": "a" * 64,
            }
        },
    )
    destination = tmp_path / "archive"
    destination.mkdir()

    with pytest.raises(ValueError, match="PARENT_CLAIM_INVALID"):
        _copy_parent_compile_snapshot(destination, protocol)
    assert calls == [Path(protocol.parent_compile.archive_path), destination / "parent_compile"]


def test_attempt_claim_key_permanently_binds_protocol_and_split() -> None:
    protocol = default_matmul_collective_surface_correctness_protocol()
    split = MatmulCollectiveSurfaceSplit.CALIBRATION

    claim = _attempt_claim_path(protocol, split)

    expected = hashlib.sha256(f"{protocol.protocol_id}:{split.value}".encode()).hexdigest()
    assert claim.name == f"{expected}.json"
    assert "source_commit" not in claim.name


def test_holdout_is_rejected_before_any_attempt_is_claimed(tmp_path) -> None:
    with pytest.raises(ValueError, match="HOLDOUT_NOT_AUTHORIZED"):
        execute_surface_correctness(
            tmp_path / "holdout",
            Path("contracts/matmul-collective-surface-correctness-v1.json"),
            Path("contracts/matmul-collective-surface-design-v1.json"),
            "a" * 64,
            MatmulCollectiveSurfaceSplit.HOLDOUT,
        )
