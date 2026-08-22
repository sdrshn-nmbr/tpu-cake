from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import tpu_cake.matmul_collective_surface_calibration_failure_verifier as failure_verifier
from tpu_cake.matmul_collective_surface_calibration_failure_verifier import (
    SurfaceCalibrationFailureVerification,
    _failed_archive_files,
    _pre_failure_artifact_set_sha256,
    _validate_manifest_failure,
)
from tpu_cake.matmul_collective_surface_calibration_verifier import _identity_sha256

_MARKER = "MATMUL_COLLECTIVE_SURFACE_CALIBRATION_MANIFEST_ORDER_INVALID"


def _failure_text(marker: str = _MARKER) -> str:
    return (
        "1 validation error for SurfaceCalibrationManifest\n"
        f"  Value error, {marker} "
        "[type=value_error, input_value={'identity': SurfaceCal...artifacts=(...)} , "
        "input_type=dict]\n"
        "    For further information visit https://errors.pydantic.dev/2.13/v/value_error"
    )


def _write_failure(
    root: Path, *, marker: str = _MARKER, error_type: str = "ValidationError"
) -> None:
    (root / "failure.json").write_text(
        json.dumps(
            {"error": _failure_text(marker), "error_type": error_type},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_manifest_failure_requires_the_exact_pydantic_failure(tmp_path: Path) -> None:
    _write_failure(tmp_path)

    _validate_manifest_failure(tmp_path)

    for marker, error_type in (
        ("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_MANIFEST_PATH_INVALID", "ValidationError"),
        (_MARKER, "ValueError"),
    ):
        _write_failure(tmp_path, marker=marker, error_type=error_type)
        with pytest.raises(ValueError, match="FAILURE_CLASS_MISMATCH"):
            _validate_manifest_failure(tmp_path)


def test_manifest_failure_rejects_a_present_manifest(tmp_path: Path) -> None:
    _write_failure(tmp_path)
    (tmp_path / "manifest.json").write_text("{}\n")

    with pytest.raises(ValueError, match="MANIFEST_PRESENT"):
        _validate_manifest_failure(tmp_path)


def test_failed_inventory_is_success_inventory_minus_manifest_plus_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol: dict[str, object] = {}
    monkeypatch.setattr(
        failure_verifier.independent,
        "_expected_archive_files",
        lambda root, protocol: {"evidence.json", "receipt.json", "manifest.json"},
    )

    assert _failed_archive_files(tmp_path, protocol) == {
        "evidence.json",
        "receipt.json",
        "failure.json",
    }


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_closed_world_failed_inventory_rejects_missing_or_extra_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    expected = {"evidence.json", "receipt.json", "failure.json"}
    monkeypatch.setattr(failure_verifier, "_failed_archive_files", lambda root, protocol: expected)
    for path in expected:
        (tmp_path / path).write_text("{}\n")
    if mutation == "missing":
        (tmp_path / "evidence.json").unlink()
    else:
        (tmp_path / "unexpected.json").write_text("{}\n")

    with pytest.raises(ValueError, match="ARCHIVE_INVENTORY_MISMATCH"):
        failure_verifier._validate_failed_inventory(tmp_path, {})


def test_pre_failure_artifact_set_excludes_receipt_ledger_manifest_and_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / "evidence.json").write_bytes(b"evidence")
    (tmp_path / "ledger.sqlite").write_bytes(b"ledger")
    (tmp_path / "receipt.json").write_bytes(b"receipt")
    (tmp_path / "manifest.json").write_bytes(b"manifest")
    (tmp_path / "failure.json").write_bytes(b"failure")

    expected = _identity_sha256(
        {
            "evidence.json": [
                len(b"evidence"),
                __import__("hashlib").sha256(b"evidence").hexdigest(),
            ]
        }
    )
    assert _pre_failure_artifact_set_sha256(tmp_path) == expected

    wrong_receipt_inclusion = _identity_sha256(
        {
            "evidence.json": [
                len(b"evidence"),
                __import__("hashlib").sha256(b"evidence").hexdigest(),
            ],
            "receipt.json": [
                len(b"receipt"),
                __import__("hashlib").sha256(b"receipt").hexdigest(),
            ],
        }
    )
    wrong_failure_inclusion = _identity_sha256(
        {
            "evidence.json": [
                len(b"evidence"),
                __import__("hashlib").sha256(b"evidence").hexdigest(),
            ],
            "failure.json": [
                len(b"failure"),
                __import__("hashlib").sha256(b"failure").hexdigest(),
            ],
        }
    )
    assert expected not in {wrong_receipt_inclusion, wrong_failure_inclusion}


def test_result_is_frozen_and_distinguishes_measurements_from_attempt_completion() -> None:
    result = SurfaceCalibrationFailureVerification(
        attempt_id="1" * 64,
        protocol_id="2" * 64,
        source_authority_sha256="3" * 64,
        execution_authority_sha256="4" * 64,
        correctness_parent_receipt_sha256="5" * 64,
        evidence_sha256="6" * 64,
        seal_sha256="7" * 64,
        ledger_sha256="8" * 64,
        phase_ledger_sha256="9" * 64,
        receipt_sha256="a" * 64,
        failure_file_sha256="b" * 64,
        sample_count=2560,
        observed_seal_authorization="pending_independent_replay",
    )

    assert json.loads(result.as_json()) == {
        "attempt_complete": False,
        "attempt_id": "1" * 64,
        "correctness_parent_receipt_sha256": "5" * 64,
        "evidence_sha256": "6" * 64,
        "execution_authority_sha256": "4" * 64,
        "failure_file_sha256": "b" * 64,
        "holdout_authorized": False,
        "ledger_sha256": "8" * 64,
        "measurements_valid": True,
        "observed_seal_authorization": "pending_independent_replay",
        "phase_ledger_sha256": "9" * 64,
        "protocol_id": "2" * 64,
        "receipt_sha256": "a" * 64,
        "sample_count": 2560,
        "seal_sha256": "7" * 64,
        "source_authority_sha256": "3" * 64,
    }
    with pytest.raises(FrozenInstanceError):
        result.attempt_complete = True  # type: ignore[misc]
