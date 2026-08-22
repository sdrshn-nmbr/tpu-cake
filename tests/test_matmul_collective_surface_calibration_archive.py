from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

import tpu_cake.matmul_collective_surface_calibration_archive as archive_module
from tpu_cake.matmul_collective_surface_calibration_archive import (
    copy_parent_archive,
    validate_and_extract_parent_archive,
)


def _zstd() -> Path:
    path = shutil.which("zstd")
    if path is None:
        pytest.skip("zstd unavailable")
    return Path(path)


def _archive(tmp_path: Path, members: tuple[tarfile.TarInfo, ...]) -> Path:
    raw = tmp_path / "parent.tar"
    with tarfile.open(raw, "w") as stream:
        for member in members:
            payload = None if member.isdir() else io.BytesIO(b"x" * member.size)
            stream.addfile(member, payload)
    archive = tmp_path / "parent.tar.zst"
    subprocess.run(
        [str(_zstd()), "-q", "-f", str(raw), "-o", str(archive)],
        check=True,
    )
    return archive


def _directory(name: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    return member


def _file(name: str, size: int = 1) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = size
    return member


def _extract(archive: Path, destination: Path, *, maximum_member_size: int = 1024) -> None:
    validate_and_extract_parent_archive(
        archive,
        destination,
        expected_root_name="parent",
        maximum_members=10,
        maximum_member_size_bytes=maximum_member_size,
        maximum_total_size_bytes=2048,
        zstd_path=_zstd(),
    )


def test_parent_archive_copy_and_safe_extraction_execute_real_zstd_boundary(
    tmp_path: Path,
) -> None:
    archive = _archive(
        tmp_path,
        (_directory("parent"), _directory("parent/data"), _file("parent/data/result.json", 3)),
    )
    staged = tmp_path / "private" / "parent.tar.zst"
    expected_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

    assert (
        copy_parent_archive(
            archive,
            staged,
            expected_sha256=expected_sha256,
            expected_size_bytes=archive.stat().st_size,
        )
        == expected_sha256
    )
    _extract(staged, tmp_path / "extracted")

    assert (tmp_path / "extracted/parent/data/result.json").read_bytes() == b"xxx"


def test_zstd_reads_the_open_archive_descriptor_from_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, (_directory("parent"), _file("parent/value")))
    real_popen = subprocess.Popen
    observed: list[tuple[list[str], object, object]] = []

    def recording_popen(arguments: list[str], **keywords: object) -> subprocess.Popen[bytes]:
        observed.append((arguments, keywords.get("stdin"), keywords.get("pass_fds")))
        return real_popen(arguments, **keywords)

    monkeypatch.setattr(archive_module.subprocess, "Popen", recording_popen)

    _extract(archive, tmp_path / "extracted")

    assert len(observed) == 2
    assert all(arguments == [str(_zstd()), "-dc"] for arguments, _, _ in observed)
    assert all(isinstance(stdin, int) and pass_fds is None for _, stdin, pass_fds in observed)


def test_parent_archive_rejects_copy_hash_mismatch_and_symlink(tmp_path: Path) -> None:
    archive = _archive(tmp_path, (_directory("parent"),))
    with pytest.raises(ValueError, match="COPY_MISMATCH"):
        copy_parent_archive(
            archive,
            tmp_path / "bad.tar.zst",
            expected_sha256="0" * 64,
            expected_size_bytes=archive.stat().st_size,
        )

    link = tmp_path / "link.tar.zst"
    link.symlink_to(archive)
    with pytest.raises(OSError):
        copy_parent_archive(
            link,
            tmp_path / "linked.tar.zst",
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_size_bytes=archive.stat().st_size,
        )


@pytest.mark.parametrize(
    "members,error",
    [
        ((_directory("parent"), _file("parent/../escape")), "PATH_INVALID"),
        ((_directory("parent"), _file("/parent/absolute")), "PATH_INVALID"),
        ((_directory("other"), _file("other/file")), "PATH_INVALID"),
        ((_directory("parent"), _file("parent/file"), _file("parent/file")), "DUPLICATE"),
    ],
)
def test_parent_archive_rejects_path_and_inventory_attacks(
    tmp_path: Path,
    members: tuple[tarfile.TarInfo, ...],
    error: str,
) -> None:
    archive = _archive(tmp_path, members)

    with pytest.raises(ValueError, match=error):
        _extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_parent_archive_rejects_links_and_devices(tmp_path: Path, kind: bytes) -> None:
    hostile = tarfile.TarInfo("parent/hostile")
    hostile.type = kind
    hostile.linkname = "parent/target"
    archive = _archive(tmp_path, (_directory("parent"), hostile))

    with pytest.raises(ValueError, match="TYPE_INVALID"):
        _extract(archive, tmp_path / "extracted")


def test_parent_archive_rejects_member_size_limit_before_extraction(tmp_path: Path) -> None:
    archive = _archive(tmp_path, (_directory("parent"), _file("parent/large", 2)))

    with pytest.raises(ValueError, match="ARCHIVE_LIMIT"):
        _extract(archive, tmp_path / "extracted", maximum_member_size=1)
    assert not (tmp_path / "extracted").exists()


def test_parent_archive_rejects_existing_destination(tmp_path: Path) -> None:
    archive = _archive(tmp_path, (_directory("parent"),))
    destination = tmp_path / "extracted"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        _extract(archive, destination)


def test_parent_archive_copy_is_exclusive(tmp_path: Path) -> None:
    archive = _archive(tmp_path, (_directory("parent"),))
    destination = tmp_path / "destination.tar.zst"
    destination.write_bytes(b"occupied")

    with pytest.raises(FileExistsError):
        copy_parent_archive(
            archive,
            destination,
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_size_bytes=os.stat(archive).st_size,
        )
