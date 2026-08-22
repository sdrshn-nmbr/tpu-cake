from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tpu_cake.artifacts import (
    build_artifact_manifest,
    file_sha256,
    text_file_sha256,
    validate_artifact_manifest,
    write_json,
    write_text,
)
from tpu_cake.contracts import ArtifactReference, ArtifactRole


def test_artifact_writers_preserve_existing_bytes(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "record.json"
    text_path = tmp_path / "nested" / "stablehlo.txt"

    write_json(json_path, {"z": 1, "a": [2, 3]})
    write_text(text_path, "module @fixture {}\n")

    expected_json = b'{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n'
    expected_text = b"module @fixture {}\n"
    assert json_path.read_bytes() == expected_json
    assert text_path.read_bytes() == expected_text
    assert file_sha256(json_path) == hashlib.sha256(expected_json).hexdigest()
    assert file_sha256(text_path) == hashlib.sha256(expected_text).hexdigest()


def test_text_file_sha256_includes_the_persisted_newline() -> None:
    expected = hashlib.sha256(b"stablehlo\n").hexdigest()

    assert text_file_sha256("stablehlo") == expected


def test_artifact_manifest_is_sorted_and_excludes_receipt(tmp_path: Path) -> None:
    write_text(tmp_path / "z.txt", "z")
    write_text(tmp_path / "nested" / "a.txt", "a")
    write_text(tmp_path / "receipt.json", "receipt")

    manifest = build_artifact_manifest(
        tmp_path,
        role_for_path=lambda _path: ArtifactRole.SEARCH_EVIDENCE,
    )

    assert tuple(value.path for value in manifest) == ("nested/a.txt", "z.txt")
    assert tuple(value.sha256 for value in manifest) == (
        hashlib.sha256(b"a").hexdigest(),
        hashlib.sha256(b"z").hexdigest(),
    )


def _validate_test_manifest(
    tmp_path: Path,
    artifacts: tuple[ArtifactReference, ...],
) -> None:
    validate_artifact_manifest(
        tmp_path,
        artifacts,
        role_for_path=lambda _path: ArtifactRole.SEARCH_EVIDENCE,
        duplicate_error="DUPLICATE",
        closed_world_error="CLOSED_WORLD",
        symlink_error="SYMLINK",
        mismatch_error=lambda path: f"MISMATCH path={path}",
    )


def test_artifact_manifest_validation_accepts_the_closed_world(tmp_path: Path) -> None:
    write_text(tmp_path / "evidence.txt", "evidence")
    manifest = build_artifact_manifest(
        tmp_path,
        role_for_path=lambda _path: ArtifactRole.SEARCH_EVIDENCE,
    )

    _validate_test_manifest(tmp_path, manifest)


def test_artifact_manifest_validation_rejects_duplicate_and_extra_artifacts(
    tmp_path: Path,
) -> None:
    write_text(tmp_path / "evidence.txt", "evidence")
    manifest = build_artifact_manifest(
        tmp_path,
        role_for_path=lambda _path: ArtifactRole.SEARCH_EVIDENCE,
    )

    with pytest.raises(ValueError, match="DUPLICATE"):
        _validate_test_manifest(tmp_path, (*manifest, manifest[0]))

    write_text(tmp_path / "undeclared.txt", "extra")
    with pytest.raises(ValueError, match="CLOSED_WORLD"):
        _validate_test_manifest(tmp_path, manifest)


def test_artifact_manifest_validation_rejects_mutation_and_symlinks(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    write_text(evidence, "evidence")
    manifest = build_artifact_manifest(
        tmp_path,
        role_for_path=lambda _path: ArtifactRole.SEARCH_EVIDENCE,
    )

    write_text(evidence, "mutated")
    with pytest.raises(ValueError, match="MISMATCH path=evidence.txt"):
        _validate_test_manifest(tmp_path, manifest)

    evidence.unlink()
    evidence.symlink_to(tmp_path / "target.txt")
    write_text(tmp_path / "target.txt", "evidence")
    symlink_manifest = build_artifact_manifest(
        tmp_path,
        role_for_path=lambda _path: ArtifactRole.SEARCH_EVIDENCE,
    )
    with pytest.raises(ValueError, match="SYMLINK"):
        _validate_test_manifest(tmp_path, symlink_manifest)
