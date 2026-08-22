from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tpu_cake.matmul_collective_surface_calibration_executor import (
    SurfaceCalibrationPhaseReceipt,
    _artifact_set_sha256,
    _claim_attempt,
    _manifest_entries,
    _stage_and_verify_parent,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    default_matmul_collective_surface_calibration_protocol,
)


def _zstd() -> Path:
    path = shutil.which("zstd")
    if path is None:
        pytest.skip("zstd unavailable")
    return Path(path)


def test_permanent_claim_is_parent_bound_and_exclusive(tmp_path: Path) -> None:
    protocol = default_matmul_collective_surface_calibration_protocol().model_copy(
        update={"attempt_registry_root": str(tmp_path / "registry")}
    )
    root = tmp_path / "attempt"
    root.mkdir()

    path, claim = _claim_attempt(root, "1" * 64, protocol, "2" * 40)

    assert path == Path(protocol.attempt_registry_root) / f"{protocol.permanent_claim_key}.json"
    assert claim.correctness_parent_receipt_sha256 == (protocol.correctness_parent.receipt_sha256)
    assert json.loads(path.read_text()) == claim.model_dump(
        mode="json", exclude_computed_fields=True
    )
    with pytest.raises(ValueError, match="PERMANENTLY_CLAIMED"):
        _claim_attempt(root, "3" * 64, protocol, "4" * 40)


def test_artifact_set_excludes_circular_files_and_rejects_links(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    (tmp_path / "evidence.json").write_bytes(b"evidence")
    (tmp_path / "ledger.sqlite").write_bytes(b"ledger")
    (tmp_path / "receipt.json").write_bytes(b"receipt")
    (tmp_path / "manifest.json").write_bytes(b"manifest")

    expected = hashlib.sha256(
        json.dumps(
            {
                "evidence.json": [8, hashlib.sha256(b"evidence").hexdigest()],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert _artifact_set_sha256(tmp_path) == expected

    (tmp_path / "link").symlink_to(tmp_path / "evidence.json")
    with pytest.raises(ValueError, match="ARTIFACT_SYMLINK"):
        _manifest_entries(tmp_path)


def test_calibration_receipt_identity_binds_parent_and_seal() -> None:
    receipt = SurfaceCalibrationPhaseReceipt(
        attempt_id="1" * 64,
        protocol_id="2" * 64,
        attempt_claim_path="/claim",
        attempt_claim_sha256="3" * 64,
        correctness_parent_receipt_file_sha256="4" * 64,
        correctness_parent_receipt_sha256="5" * 64,
        evidence_file_sha256="6" * 64,
        evidence_sha256="7" * 64,
        calibration_seal_file_sha256="8" * 64,
        calibration_seal_sha256="9" * 64,
        ledger_snapshot_sha256="a" * 64,
        phase_ledger_file_sha256="b" * 64,
        phase_ledger_sha256="c" * 64,
        previous_phase_receipt_sha256="5" * 64,
        artifact_set_sha256="d" * 64,
    )
    changed = receipt.model_copy(update={"calibration_seal_sha256": "e" * 64})

    assert receipt.phase_sequence == 4
    assert receipt.receipt_sha256 != changed.receipt_sha256


def test_parent_is_staged_verified_then_moved_into_final_layout(tmp_path: Path) -> None:
    parent_protocol = default_matmul_collective_surface_calibration_protocol()
    parent = parent_protocol.correctness_parent
    root_name = "parent-correctness"
    replay = {
        "attempt_id": parent.attempt_id,
        "protocol_id": parent.protocol_id,
        "source_authority_sha256": parent.source_authority_sha256,
        "execution_authority_sha256": parent.execution_authority_sha256,
        "evidence_sha256": parent.evidence_sha256,
        "ledger_sha256": parent.ledger_file_sha256,
        "phase_ledger_sha256": parent.phase_ledger_sha256,
        "receipt_sha256": parent.receipt_sha256,
        "case_count": parent.case_count,
        "execution_count": parent.execution_count,
        "split": parent.split,
    }
    script = f"import json\nprint(json.dumps({replay!r}, sort_keys=True))\n".encode()
    raw = tmp_path / "parent.tar"
    with tarfile.open(raw, "w") as stream:
        directory = tarfile.TarInfo(root_name)
        directory.type = tarfile.DIRTYPE
        stream.addfile(directory)
        source = tarfile.TarInfo(f"{root_name}/source")
        source.type = tarfile.DIRTYPE
        stream.addfile(source)
        for name, payload in (
            (f"{root_name}/source/verifier.py", script),
            (f"{root_name}/protocol.json", b"{}\n"),
            (f"{root_name}/design.json", b"{}\n"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    archive = tmp_path / "parent.tar.zst"
    subprocess.run(
        [str(_zstd()), "-q", "-f", str(raw), "-o", str(archive)],
        check=True,
    )
    rebound_parent = parent.model_copy(
        update={
            "archive_path": str(archive),
            "archive_filename": archive.name,
            "archive_root_name": root_name,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "archive_size_bytes": archive.stat().st_size,
        }
    )
    protocol = parent_protocol.model_copy(update={"correctness_parent": rebound_parent})
    output = tmp_path / "output"
    output.mkdir(mode=0o700)

    extracted = _stage_and_verify_parent(output, protocol, zstd_path=_zstd())

    assert extracted == output / "parent" / root_name
    assert (output / "parent" / archive.name).read_bytes() == archive.read_bytes()
    assert (extracted / "source/verifier.py").read_bytes() == script
    assert not (output / "parent-extraction").exists()
