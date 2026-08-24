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
from test_matmul_collective_surface_calibration_evidence import _evidence

import tpu_cake.matmul_collective_surface_calibration_executor as executor
from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256
from tpu_cake.matmul_collective_surface_calibration_archive import (
    validate_and_extract_parent_archive,
)
from tpu_cake.matmul_collective_surface_calibration_evidence import (
    MatmulCollectiveSurfaceCalibrationEvidence,
    SurfaceCalibrationCallSample,
    SurfaceCalibrationOutputGate,
    SurfaceCalibrationWarmupExecution,
)
from tpu_cake.matmul_collective_surface_calibration_executor import (
    SurfaceCalibrationManifest,
    SurfaceCalibrationPhaseReceipt,
    _artifact_set_sha256,
    _claim_attempt,
    _file_sha256,
    _manifest_entries,
    _stage_and_verify_parent,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    MatmulCollectiveSurfaceCalibrationProtocol,
    default_matmul_collective_surface_calibration_protocol,
)
from tpu_cake.matmul_collective_surface_calibration_seal import (
    MatmulCollectiveSurfaceCalibrationSealedEvidence,
)
from tpu_cake.matmul_collective_surface_calibration_worker import (
    CALIBRATION_EXECUTABLE_DEPENDENCIES,
    CALIBRATION_EXECUTOR_SOURCE_PATH,
    CALIBRATION_VERIFIER_SOURCE_PATH,
    CALIBRATION_WORKER_SOURCE_PATH,
    SurfaceCalibrationAttemptClaim,
    SurfaceCalibrationDevice,
    SurfaceCalibrationExecutionAuthority,
    SurfaceCalibrationSourceAuthority,
    SurfaceCalibrationWorkerRequest,
    SurfaceCalibrationWorkerResult,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    default_matmul_collective_surface_design_contract,
)
from tpu_cake.matmul_collective_surface_runner import (
    SurfacePhase,
    SurfacePhaseLedger,
    record_surface_phase,
)


def _zstd() -> Path:
    path = shutil.which("zstd")
    if path is None:
        pytest.skip("zstd unavailable")
    return Path(path)


def _synthetic_authority(
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCalibrationExecutionAuthority, dict[str, bytes]]:
    source_blobs = {
        path: f"synthetic lifecycle source: {path}\n".encode()
        for path in CALIBRATION_EXECUTABLE_DEPENDENCIES
    }
    source_blobs["uv.lock"] = b"synthetic lifecycle lock\n"
    source = SurfaceCalibrationSourceAuthority(
        source_commit="1" * 40,
        origin_main_commit="1" * 40,
        remote_main_commit="1" * 40,
        runtime={},
        uv_lock_sha256=hashlib.sha256(source_blobs["uv.lock"]).hexdigest(),
        dependencies=tuple(
            SourceFileContract(path=path, sha256=hashlib.sha256(source_blobs[path]).hexdigest())
            for path in CALIBRATION_EXECUTABLE_DEPENDENCIES
        ),
    )
    authority = SurfaceCalibrationExecutionAuthority(
        protocol_id=protocol.protocol_id,
        protocol_file_sha256="2" * 64,
        design_id=design.design_id,
        design_file_sha256="3" * 64,
        source=source,
        executor_source_sha256=hashlib.sha256(
            source_blobs[CALIBRATION_EXECUTOR_SOURCE_PATH.removeprefix("src/")]
        ).hexdigest(),
        worker_source_sha256=hashlib.sha256(
            source_blobs[CALIBRATION_WORKER_SOURCE_PATH.removeprefix("src/")]
        ).hexdigest(),
        verifier_source_sha256=hashlib.sha256(
            source_blobs[CALIBRATION_VERIFIER_SOURCE_PATH.removeprefix("src/")]
        ).hexdigest(),
        compiler_environment=design.compiler_environment,
        devices=tuple(SurfaceCalibrationDevice(id=index) for index in range(8)),
    )
    return authority, source_blobs


def _bind_synthetic_worker_result(
    template: MatmulCollectiveSurfaceCalibrationEvidence,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
    authority: SurfaceCalibrationExecutionAuthority,
    request: SurfaceCalibrationWorkerRequest,
) -> SurfaceCalibrationWorkerResult:
    nonce = request.invocation_nonce
    worker_pid = 4242
    pairs = tuple(
        value.model_copy(update={"invocation_nonce": nonce, "worker_pid": worker_pid})
        for value in template.resident_pairs
    )
    pair_hashes = {value.scenario_name: value.resident_pair_sha256 for value in pairs}

    def bind_observation(
        value: (
            SurfaceCalibrationOutputGate
            | SurfaceCalibrationWarmupExecution
            | SurfaceCalibrationCallSample
        ),
    ) -> (
        SurfaceCalibrationOutputGate
        | SurfaceCalibrationWarmupExecution
        | SurfaceCalibrationCallSample
    ):
        return value.model_copy(
            update={
                "resident_pair_sha256": pair_hashes[value.scenario_name],
                "invocation_nonce": nonce,
                "worker_pid": worker_pid,
            }
        )

    evidence = template.model_copy(
        update={
            "protocol_id": protocol.protocol_id,
            "protocol_file_sha256": authority.protocol_file_sha256,
            "design_id": design.design_id,
            "design_file_sha256": authority.design_file_sha256,
            "calibration_execution_authority_sha256": authority.authority_sha256,
            "invocation_nonce": nonce,
            "worker_pid": worker_pid,
            "resident_pairs": pairs,
            "output_gates": tuple(bind_observation(value) for value in template.output_gates),
            "warmups": tuple(bind_observation(value) for value in template.warmups),
            "samples": tuple(bind_observation(value) for value in template.samples),
        }
    )
    return SurfaceCalibrationWorkerResult(
        attempt_id=request.attempt_id,
        invocation_nonce=nonce,
        worker_pid=worker_pid,
        execution_authority_sha256=authority.authority_sha256,
        evidence=evidence,
    )


def _replay_relocated_lifecycle(
    root: Path,
    temporary_root: Path,
    protocol: MatmulCollectiveSurfaceCalibrationProtocol,
) -> dict[str, object]:
    raw_archive = temporary_root / "terminal.tar"
    with tarfile.open(raw_archive, "w") as bundle:
        bundle.add(root, arcname=root.name)
    compressed_archive = temporary_root / "terminal.tar.zst"
    subprocess.run(
        [str(_zstd()), "-q", "-f", str(raw_archive), "-o", str(compressed_archive)],
        check=True,
    )
    relocated = temporary_root / "relocated"
    validate_and_extract_parent_archive(
        compressed_archive,
        relocated,
        expected_root_name=root.name,
        maximum_members=10_000,
        maximum_member_size_bytes=1 << 30,
        maximum_total_size_bytes=4 << 30,
        zstd_path=_zstd(),
    )
    replay_root = relocated / root.name
    manifest = SurfaceCalibrationManifest.model_validate_json(
        (replay_root / "manifest.json").read_text()
    )
    assert tuple(value.path for value in manifest.artifacts) == tuple(
        sorted(value.path for value in manifest.artifacts)
    )
    assert all(
        _file_sha256(replay_root / value.path) == value.sha256
        and (replay_root / value.path).stat().st_size == value.size_bytes
        for value in manifest.artifacts
    )
    evidence = MatmulCollectiveSurfaceCalibrationEvidence.model_validate_json(
        (replay_root / "evidence.json").read_text()
    )
    seal = MatmulCollectiveSurfaceCalibrationSealedEvidence.model_validate_json(
        (replay_root / "calibration-seal.json").read_text()
    )
    receipt = SurfaceCalibrationPhaseReceipt.model_validate_json(
        (replay_root / "receipt.json").read_text()
    )
    phase_ledger = SurfacePhaseLedger.model_validate_json(
        (replay_root / "phase_ledger.json").read_text()
    )
    return {
        "attempt_id": manifest.identity.attempt_id,
        "protocol_id": manifest.identity.protocol_id,
        "source_authority_sha256": manifest.identity.source_authority_sha256,
        "execution_authority_sha256": manifest.identity.execution_authority_sha256,
        "correctness_parent_receipt_sha256": protocol.correctness_parent.receipt_sha256,
        "evidence_sha256": evidence.evidence_sha256,
        "seal_sha256": seal.seal_sha256,
        "ledger_sha256": _file_sha256(replay_root / "ledger.sqlite"),
        "phase_ledger_sha256": model_identity_sha256(phase_ledger),
        "receipt_sha256": receipt.receipt_sha256,
        "sample_count": len(evidence.samples),
        "holdout_authorization": seal.holdout_authorization,
    }


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


def test_manifest_entries_use_canonical_relative_path_order(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    (tmp_path / "parent").mkdir()
    (tmp_path / "parent-extraction").mkdir()
    (tmp_path / "parent" / "artifact").write_bytes(b"parent")
    (tmp_path / "parent-extraction" / "artifact").write_bytes(b"extraction")

    entries = _manifest_entries(tmp_path)

    assert tuple(entry.path for entry in entries) == (
        "parent-extraction/artifact",
        "parent/artifact",
    )


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
        committed = tarfile.TarInfo(f"{root_name}/source/committed")
        committed.type = tarfile.DIRTYPE
        stream.addfile(committed)
        contracts = tarfile.TarInfo(f"{root_name}/source/committed/contracts")
        contracts.type = tarfile.DIRTYPE
        stream.addfile(contracts)
        contract_root = f"{root_name}/source/committed/contracts"
        for name, payload in (
            (f"{root_name}/source/verifier.py", script),
            (f"{root_name}/protocol.json", b"{}\n"),
            (f"{root_name}/design.json", b"{}\n"),
            (
                f"{contract_root}/matmul-collective-surface-correctness-v1.json",
                b"{}\n",
            ),
            (
                f"{contract_root}/matmul-collective-surface-design-v1.json",
                b"{}\n",
            ),
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


def test_mock_lifecycle_replays_after_safe_archive_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.chmod(tmp_path, 0o700)
    design = default_matmul_collective_surface_design_contract()
    protocol = default_matmul_collective_surface_calibration_protocol()
    protocol_path = tmp_path / "protocol.json"
    design_path = tmp_path / "design.json"
    protocol_path.write_text("{}\n")
    design_path.write_text("{}\n")
    authority, source_blobs = _synthetic_authority(protocol, design)
    evidence_template = _evidence()

    def stage_parent(
        root: Path,
        _protocol: MatmulCollectiveSurfaceCalibrationProtocol,
        *,
        zstd_path: Path,
    ) -> Path:
        assert zstd_path.is_file()
        parent = root / "parent" / protocol.correctness_parent.archive_root_name
        parent.mkdir(mode=0o700, parents=True)
        ledger = SurfacePhaseLedger(attempt_id=protocol.correctness_parent.attempt_id)
        ledger = record_surface_phase(ledger, SurfacePhase.COMPILE, "4" * 64)
        ledger = record_surface_phase(ledger, SurfacePhase.CORRECTNESS, "5" * 64)
        executor._write_model_exclusive(parent / "phase_ledger.json", ledger)
        (parent / "parent-evidence-marker").write_bytes(b"parent\n")
        return parent

    def claim_attempt(
        root: Path,
        attempt_id: str,
        _protocol: MatmulCollectiveSurfaceCalibrationProtocol,
        source_commit: str,
    ) -> tuple[Path, SurfaceCalibrationAttemptClaim]:
        claim = SurfaceCalibrationAttemptClaim(
            attempt_id=attempt_id,
            protocol_id=protocol.protocol_id,
            permanent_claim_key=protocol.permanent_claim_key,
            correctness_parent_receipt_sha256=protocol.correctness_parent.receipt_sha256,
            source_commit=source_commit,
            output_root=str(root),
        )
        path = tmp_path / "registry" / f"{protocol.permanent_claim_key}.json"
        executor._write_model_exclusive(path, claim)
        return path, claim

    def worker_result(
        _root: Path,
        request: SurfaceCalibrationWorkerRequest,
        _authority: SurfaceCalibrationExecutionAuthority,
    ) -> SurfaceCalibrationWorkerResult:
        return _bind_synthetic_worker_result(
            evidence_template,
            protocol,
            design,
            authority,
            request,
        )

    def relocated_replay(
        root: Path,
        _protocol_path: Path,
        _design_path: Path,
    ) -> dict[str, object]:
        return _replay_relocated_lifecycle(root, tmp_path, protocol)

    parent_schedules = executor._schedule_payload(evidence_template.continuity)
    monkeypatch.setattr(executor, "_canonical_design", lambda _path: design)
    monkeypatch.setattr(executor, "_canonical_protocol", lambda _path, _design: protocol)
    monkeypatch.setattr(
        executor,
        "_probe_execution_authority",
        lambda *_args: (authority, source_blobs),
    )
    monkeypatch.setattr(executor, "_stage_and_verify_parent", stage_parent)
    monkeypatch.setattr(executor, "_claim_attempt", claim_attempt)
    monkeypatch.setattr(executor, "_parent_schedule_payload", lambda *_args: parent_schedules)
    monkeypatch.setattr(executor, "_launch_worker", lambda *_args: None)
    monkeypatch.setattr(executor, "_validate_worker_result", worker_result)
    monkeypatch.setattr(executor, "_run_archived_independent_verifier", relocated_replay)

    root = tmp_path / "attempt"
    manifest = executor.execute_surface_calibration(
        root,
        protocol_path,
        design_path,
        "6" * 64,
        zstd_path=_zstd(),
    )

    assert manifest == SurfaceCalibrationManifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    assert not (root / "failure.json").exists()
    assert (tmp_path / "terminal.tar.zst").is_file()
