from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from tpu_cake.contracts import ArtifactReference, ArtifactRole


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_file_sha256(value: str) -> str:
    return hashlib.sha256((value + "\n").encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def build_artifact_manifest(
    root: Path,
    *,
    role_for_path: Callable[[Path], ArtifactRole],
    excluded_names: tuple[str, ...] = ("receipt.json",),
) -> tuple[ArtifactReference, ...]:
    return tuple(
        ArtifactReference(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
            role=role_for_path(path.relative_to(root)),
        )
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.name not in excluded_names
    )


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
        raise ValueError(
            f"ARTIFACT_ESCAPES_BUNDLE root={root} path={declared_path}"
        ) from error

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
    digest = hashlib.sha256(relocated.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"LEGACY_ARTIFACT_HASH_MISMATCH path={declared_path}")
    return relocated
