from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

from tpu_cake.contracts import ArtifactReference, ArtifactRole


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def text_file_sha256(value: str) -> str:
    return hashlib.sha256((value + "\n").encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def save_array_reference(
    root: Path,
    path: Path,
    value: np.ndarray,
    role: ArtifactRole,
) -> ArtifactReference:
    save_array(path, value)
    return artifact_reference(root, path, role)


def artifact_reference(root: Path, path: Path, role: ArtifactRole) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=file_sha256(path),
        role=role,
    )


def write_relative_text_artifact(
    root: Path,
    relative: Path,
    value: str,
    role: ArtifactRole,
) -> ArtifactReference:
    path = root / relative
    write_text(path, value)
    return artifact_reference(root, path, role)


def save_relative_array_artifact(
    root: Path,
    relative: Path,
    value: np.ndarray,
    role: ArtifactRole,
) -> ArtifactReference:
    path = root / relative
    save_array(path, value)
    return artifact_reference(root, path, role)


def resolved_artifact_reference(
    root: Path,
    path: Path,
    role: ArtifactRole,
) -> ArtifactReference:
    return artifact_reference(root.resolve(), path.resolve(), role)


def build_artifact_manifest(
    root: Path,
    *,
    role_for_path: Callable[[Path], ArtifactRole],
    excluded_paths: tuple[str, ...] = ("receipt.json",),
    exclude_path: Callable[[Path], bool] | None = None,
) -> tuple[ArtifactReference, ...]:
    return tuple(
        ArtifactReference(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
            role=role_for_path(path.relative_to(root)),
        )
        for path in sorted(
            (value for value in root.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(root).as_posix(),
        )
        if path.relative_to(root).as_posix() not in excluded_paths
        and (exclude_path is None or not exclude_path(path.relative_to(root)))
    )


def validate_artifact_manifest(
    root: Path,
    artifacts: Sequence[ArtifactReference],
    *,
    role_for_path: Callable[[Path], ArtifactRole],
    duplicate_error: str,
    closed_world_error: str,
    mismatch_error: Callable[[str], str],
    symlink_error: str | None = None,
    excluded_paths: tuple[str, ...] = ("receipt.json",),
) -> None:
    declared = tuple(artifact.path for artifact in artifacts)
    if len(declared) != len(set(declared)):
        raise ValueError(duplicate_error)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded_paths
    }
    if set(declared) != observed:
        raise ValueError(closed_world_error)
    if symlink_error is not None and any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(symlink_error)
    for artifact in artifacts:
        path = root / artifact.path
        if (
            path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != artifact.size_bytes
            or file_sha256(path) != artifact.sha256
            or role_for_path(Path(artifact.path)) is not artifact.role
        ):
            raise ValueError(mismatch_error(artifact.path))


def resolve_bundle_artifact(root: Path, declared_path: str) -> Path:
    root = root.resolve()
    relative = PurePosixPath(declared_path)
    candidate = root.joinpath(*relative.parts)

    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"ARTIFACT_SYMLINK_FORBIDDEN path={declared_path}")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError(f"ARTIFACT_ESCAPES_BUNDLE root={root} path={declared_path}") from error

    return candidate


def resolve_recorded_artifact(
    root: Path,
    declared_path: str,
    *,
    size_bytes: int,
    sha256: str,
) -> Path:
    declared = PurePosixPath(declared_path)
    direct = resolve_bundle_artifact(root, declared.as_posix())
    if direct.is_file():
        return direct
    if len(declared.parts) == 1:
        return direct

    relocated = resolve_bundle_artifact(root, declared.name)
    if not relocated.is_file():
        return direct
    if relocated.stat().st_size != size_bytes:
        raise ValueError(f"LEGACY_ARTIFACT_SIZE_MISMATCH path={declared_path}")
    digest = file_sha256(relocated)
    if digest != sha256:
        raise ValueError(f"LEGACY_ARTIFACT_HASH_MISMATCH path={declared_path}")
    return relocated
