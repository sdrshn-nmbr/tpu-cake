from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath


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
