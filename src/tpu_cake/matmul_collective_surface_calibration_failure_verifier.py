from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import tpu_cake.matmul_collective_surface_calibration_verifier as independent

_MANIFEST_ORDER_MARKER = "MATMUL_COLLECTIVE_SURFACE_CALIBRATION_MANIFEST_ORDER_INVALID"
_MANIFEST_FAILURE = re.compile(
    rf"\A1 validation error for SurfaceCalibrationManifest\n"
    rf"  Value error, {_MANIFEST_ORDER_MARKER} "
    rf"\[type=value_error, input_value=.+, input_type=dict\]\n"
    rf"    For further information visit "
    rf"https://errors\.pydantic\.dev/2\.13/v/value_error\Z",
    re.DOTALL,
)
_ARTIFACT_SET_EXCLUSIONS = {
    "failure.json",
    "ledger.sqlite",
    "manifest.json",
    "receipt.json",
}


@dataclass(frozen=True)
class SurfaceCalibrationFailureVerification:
    attempt_id: str
    protocol_id: str
    source_authority_sha256: str
    execution_authority_sha256: str
    correctness_parent_receipt_sha256: str
    evidence_sha256: str
    seal_sha256: str
    ledger_sha256: str
    phase_ledger_sha256: str
    receipt_sha256: str
    failure_file_sha256: str
    sample_count: int
    observed_seal_authorization: str
    measurements_valid: bool = field(init=False, default=True)
    attempt_complete: bool = field(init=False, default=False)
    holdout_authorized: bool = field(init=False, default=False)

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _validate_manifest_failure(root: Path) -> str:
    manifest = root / "manifest.json"
    if manifest.exists() or manifest.is_symlink():
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_MANIFEST_PRESENT")
    failure_path = root / "failure.json"
    failure = independent._read_json(failure_path)
    independent._expect_keys(failure, {"error", "error_type"}, "FAILURE_FORENSIC_FAILURE")
    if (
        failure["error_type"] != "ValidationError"
        or not isinstance(failure["error"], str)
        or failure["error"].count(_MANIFEST_ORDER_MARKER) != 1
        or _MANIFEST_FAILURE.fullmatch(failure["error"]) is None
        or failure_path.read_bytes() != independent._pretty_json_bytes(failure)
    ):
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_FAILURE_CLASS_MISMATCH")
    return independent._file_sha256(failure_path)


def _pre_failure_artifact_entries(root: Path) -> list[dict[str, Any]]:
    return independent._artifact_entries(root, _ARTIFACT_SET_EXCLUSIONS)


def _pre_failure_artifact_set_sha256(root: Path) -> str:
    entries = _pre_failure_artifact_entries(root)
    return independent._identity_sha256(
        {entry["path"]: [entry["size_bytes"], entry["sha256"]] for entry in entries}
    )


def _validate_phase_receipt(
    root: Path,
    identity: dict[str, Any],
    protocol: dict[str, Any],
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
    evidence: dict[str, Any],
    evidence_sha256: str,
    seal: dict[str, Any],
    seal_sha256: str,
    phase_ledger_sha256: str,
) -> tuple[str, str]:
    artifact_set_sha256 = _pre_failure_artifact_set_sha256(root)
    ledger_sha256 = independent._validate_ledger(
        root / "ledger.sqlite",
        identity,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
        protocol,
        evidence,
        evidence_sha256,
        seal,
        seal_sha256,
        phase_ledger_sha256,
        artifact_set_sha256,
    )
    receipt = independent._read_json(root / "receipt.json")
    independent._expect_keys(
        receipt,
        {
            "schema_version",
            "attempt_id",
            "protocol_id",
            "attempt_claim_path",
            "attempt_claim_sha256",
            "correctness_parent_receipt_file_sha256",
            "correctness_parent_receipt_sha256",
            "evidence_file_sha256",
            "evidence_sha256",
            "calibration_seal_file_sha256",
            "calibration_seal_sha256",
            "ledger_snapshot_sha256",
            "phase_ledger_file_sha256",
            "phase_ledger_sha256",
            "phase_sequence",
            "previous_phase_receipt_sha256",
            "artifact_set_sha256",
        },
        "FAILURE_FORENSIC_RECEIPT",
    )
    parent = protocol["correctness_parent"]
    expected = {
        "schema_version": independent._RECEIPT_SCHEMA,
        "attempt_id": identity["attempt_id"],
        "protocol_id": identity["protocol_id"],
        "attempt_claim_path": identity["attempt_claim_path"],
        "attempt_claim_sha256": identity["attempt_claim_sha256"],
        "correctness_parent_receipt_file_sha256": parent["receipt_file_sha256"],
        "correctness_parent_receipt_sha256": parent["receipt_sha256"],
        "evidence_file_sha256": independent._file_sha256(root / "evidence.json"),
        "evidence_sha256": evidence_sha256,
        "calibration_seal_file_sha256": independent._file_sha256(root / "calibration-seal.json"),
        "calibration_seal_sha256": seal_sha256,
        "ledger_snapshot_sha256": ledger_sha256,
        "phase_ledger_file_sha256": independent._file_sha256(root / "phase_ledger.json"),
        "phase_ledger_sha256": phase_ledger_sha256,
        "phase_sequence": 4,
        "previous_phase_receipt_sha256": parent["receipt_sha256"],
        "artifact_set_sha256": artifact_set_sha256,
    }
    if receipt != expected:
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_RECEIPT_MISMATCH")
    return ledger_sha256, independent._identity_sha256(receipt)


def _failed_archive_files(root: Path, protocol: dict[str, Any]) -> set[str]:
    return (independent._expected_archive_files(root, protocol) - {"manifest.json"}) | {
        "failure.json"
    }


def _validate_failed_inventory(root: Path, protocol: dict[str, Any]) -> None:
    expected_files = _failed_archive_files(root, protocol)
    observed_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_ARCHIVE_INVENTORY_MISMATCH")
    expected_directories = {
        parent.as_posix()
        for path in expected_files
        for parent in PurePosixPath(path).parents
        if parent != PurePosixPath(".")
    }
    observed_directories = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_DIRECTORY_INVENTORY_MISMATCH")


def verify_surface_calibration_failure(
    root: Path,
    protocol_path: Path,
    design_path: Path,
) -> SurfaceCalibrationFailureVerification:
    if root.is_symlink():
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_ROOT_INVALID")
    root = root.resolve(strict=True)
    independent._validate_archive_tree(root)
    failure_file_sha256 = _validate_manifest_failure(root)
    protocol, protocol_id, protocol_file_sha256 = independent._validate_protocol(
        protocol_path, root / "protocol.json"
    )
    design = independent._validate_design(design_path, root / "design.json", protocol)
    parent = independent._verify_parent(root, protocol)
    authority, source_authority_sha256, execution_authority_sha256 = (
        independent._validate_authority(root, protocol, protocol_id, protocol_file_sha256, design)
    )
    identity = independent._validate_identity_and_claim(
        root,
        protocol,
        protocol_id,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
    )
    request, worker_result, evidence = independent._validate_worker(
        root,
        identity,
        protocol,
        design,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
    )
    evidence_sha256, samples = independent._validate_evidence(
        root,
        evidence,
        protocol,
        protocol_id,
        design,
        parent["evidence"],
        execution_authority_sha256,
        request,
        worker_result,
    )
    seal, seal_sha256 = independent._validate_seal(
        root, samples, protocol, protocol_id, evidence_sha256
    )
    _, phase_ledger_sha256 = independent._validate_phase_ledger(
        root, protocol, evidence_sha256, seal_sha256
    )
    ledger_sha256, receipt_sha256 = _validate_phase_receipt(
        root,
        identity,
        protocol,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
        evidence,
        evidence_sha256,
        seal,
        seal_sha256,
        phase_ledger_sha256,
    )
    _validate_failed_inventory(root, protocol)
    if independent._file_sha256(root / "failure.json") != failure_file_sha256:
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_FAILURE_CHANGED")
    return SurfaceCalibrationFailureVerification(
        attempt_id=identity["attempt_id"],
        protocol_id=protocol_id,
        source_authority_sha256=source_authority_sha256,
        execution_authority_sha256=execution_authority_sha256,
        correctness_parent_receipt_sha256=protocol["correctness_parent"]["receipt_sha256"],
        evidence_sha256=evidence_sha256,
        seal_sha256=seal_sha256,
        ledger_sha256=ledger_sha256,
        phase_ledger_sha256=phase_ledger_sha256,
        receipt_sha256=receipt_sha256,
        failure_file_sha256=failure_file_sha256,
        sample_count=len(samples),
        observed_seal_authorization=seal["holdout_authorization"],
    )


def main() -> None:
    if not sys.flags.safe_path:
        raise ValueError("SURFACE_CALIBRATION_FAILURE_FORENSIC_SAFE_PATH_REQUIRED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    args = parser.parse_args()
    print(verify_surface_calibration_failure(args.root, args.protocol, args.design).as_json())


if __name__ == "__main__":
    main()
