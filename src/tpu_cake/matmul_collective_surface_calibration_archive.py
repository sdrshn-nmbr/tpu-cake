from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field


class SurfaceCalibrationArchiveMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    kind: str
    size_bytes: int = Field(ge=0)


def copy_parent_archive(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> str:
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        source_status = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_status.st_mode):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_SOURCE_INVALID")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        digest = hashlib.sha256()
        size = 0
        try:
            while payload := os.read(source_descriptor, 1024 * 1024):
                digest.update(payload)
                size += len(payload)
                view = memoryview(payload)
                while view:
                    written = os.write(destination_descriptor, view)
                    view = view[written:]
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)
    observed_sha256 = digest.hexdigest()
    if size != expected_size_bytes or observed_sha256 != expected_sha256:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_COPY_MISMATCH")
    _fsync_directory(destination.parent)
    return observed_sha256


def validate_and_extract_parent_archive(
    archive: Path,
    destination: Path,
    *,
    expected_root_name: str,
    maximum_members: int,
    maximum_member_size_bytes: int,
    maximum_total_size_bytes: int,
    zstd_path: Path,
) -> tuple[SurfaceCalibrationArchiveMember, ...]:
    inventory = _read_inventory(
        archive,
        expected_root_name=expected_root_name,
        maximum_members=maximum_members,
        maximum_member_size_bytes=maximum_member_size_bytes,
        maximum_total_size_bytes=maximum_total_size_bytes,
        zstd_path=zstd_path,
    )
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    observed = _extract_inventory(
        archive,
        destination,
        expected_root_name=expected_root_name,
        maximum_members=maximum_members,
        maximum_member_size_bytes=maximum_member_size_bytes,
        maximum_total_size_bytes=maximum_total_size_bytes,
        zstd_path=zstd_path,
    )
    if observed != inventory:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_CHANGED")
    _fsync_directory(destination)
    return inventory


def _read_inventory(
    archive: Path,
    *,
    expected_root_name: str,
    maximum_members: int,
    maximum_member_size_bytes: int,
    maximum_total_size_bytes: int,
    zstd_path: Path,
) -> tuple[SurfaceCalibrationArchiveMember, ...]:
    inventory = []
    names: set[str] = set()
    total_size = 0
    with _tar_stream(archive, zstd_path) as stream:
        for member in stream:
            record = _validate_member(member, expected_root_name)
            if record.path in names:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_DUPLICATE")
            names.add(record.path)
            inventory.append(record)
            total_size += record.size_bytes
            if (
                len(inventory) > maximum_members
                or record.size_bytes > maximum_member_size_bytes
                or total_size > maximum_total_size_bytes
            ):
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_LIMIT")
    if not inventory or inventory[0].path != expected_root_name or inventory[0].kind != "directory":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_ROOT_INVALID")
    return tuple(inventory)


def _extract_inventory(
    archive: Path,
    destination: Path,
    *,
    expected_root_name: str,
    maximum_members: int,
    maximum_member_size_bytes: int,
    maximum_total_size_bytes: int,
    zstd_path: Path,
) -> tuple[SurfaceCalibrationArchiveMember, ...]:
    inventory = []
    names: set[str] = set()
    total_size = 0
    with _tar_stream(archive, zstd_path) as stream:
        for member in stream:
            record = _validate_member(member, expected_root_name)
            if record.path in names:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_DUPLICATE")
            names.add(record.path)
            inventory.append(record)
            total_size += record.size_bytes
            if (
                len(inventory) > maximum_members
                or record.size_bytes > maximum_member_size_bytes
                or total_size > maximum_total_size_bytes
            ):
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_LIMIT")
            target = destination.joinpath(*PurePosixPath(record.path).parts)
            if record.kind == "directory":
                target.mkdir(mode=0o700, parents=True, exist_ok=False)
                _fsync_directory(target.parent)
                continue
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_FILE_MISSING")
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            remaining = record.size_bytes
            try:
                while remaining:
                    payload = source.read(min(1024 * 1024, remaining))
                    if not payload:
                        raise ValueError(
                            "MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_FILE_TRUNCATED"
                        )
                    view = memoryview(payload)
                    while view:
                        written = os.write(descriptor, view)
                        view = view[written:]
                    remaining -= len(payload)
                if source.read(1):
                    raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_FILE_OVERRUN")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                source.close()
            _fsync_directory(target.parent)
    return tuple(inventory)


def _validate_member(
    member: tarfile.TarInfo, expected_root_name: str
) -> SurfaceCalibrationArchiveMember:
    path = member.name.removesuffix("/")
    pure = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.parts[0] != expected_root_name
        or getattr(member, "sparse", None)
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_PATH_INVALID")
    if member.isdir():
        kind = "directory"
        size = 0
    elif member.isreg():
        kind = "file"
        size = member.size
    else:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_TYPE_INVALID")
    return SurfaceCalibrationArchiveMember(path=path, kind=kind, size_bytes=size)


@contextmanager
def _tar_stream(archive: Path, zstd_path: Path) -> Iterator[tarfile.TarFile]:
    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
    process: subprocess.Popen[bytes] | None = None
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ARCHIVE_INVALID")
        process = subprocess.Popen(
            [str(zstd_path), "-dc"],
            stdin=descriptor,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ZSTD_PIPE_INVALID")
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
                yield stream
        except Exception:
            process.kill()
            raise
        finally:
            process.stdout.close()
        stderr = process.stderr.read(4097)
        process.stderr.close()
        return_code = process.wait()
        if return_code != 0 or len(stderr) > 4096:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CALIBRATION_ZSTD_FAILED")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
