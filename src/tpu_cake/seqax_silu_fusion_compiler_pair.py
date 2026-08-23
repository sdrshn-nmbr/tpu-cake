from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from tpu_cake.seqax_silu_fusion import (
    SeqaxSiluFusionDesignContract,
    default_seqax_silu_fusion_design_contract,
)
from tpu_cake.seqax_silu_fusion_compiler import (
    SeqaxSiluFusionCompilerPair,
    SeqaxSiluFusionCompilerReceipt,
    SeqaxSiluFusionCompilerReplaySeal,
    seqax_silu_fusion_compiler_pair_member,
)

_EVIDENCE_ROOT = Path("/home/sudarshan/tpu-cake-evidence")
_DESIGN_RELATIVE_PATH = Path("contracts/seqax-silu-fusion-design-v1.json")
_VERIFIER_MODULE = "tpu_cake.seqax_silu_fusion_compiler_verifier"


def _write_json_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _safe_pair_path(path: Path, design: SeqaxSiluFusionDesignContract) -> Path:
    if not path.is_absolute():
        raise ValueError("SEQAX_SILU_FUSION_PAIR_PATH_NOT_ABSOLUTE")
    resolved = path.resolve(strict=False)
    if path != resolved:
        raise ValueError("SEQAX_SILU_FUSION_PAIR_PATH_NOT_CANONICAL")
    if resolved.parent != _EVIDENCE_ROOT.resolve(strict=True):
        raise ValueError("SEQAX_SILU_FUSION_PAIR_PATH_OUTSIDE_EVIDENCE_ROOT")
    pattern = re.compile(
        rf"seqax-silu-fusion-compiler-pair-{design.design_id[:7]}-[0-9a-f]{{8}}\.json$"
    )
    if pattern.fullmatch(resolved.name) is None:
        raise ValueError("SEQAX_SILU_FUSION_PAIR_PATH_NAME_INVALID")
    if resolved.exists() or resolved.is_symlink():
        raise ValueError("SEQAX_SILU_FUSION_PAIR_PATH_EXISTS")
    return resolved


def _replay_seal_path(
    design: SeqaxSiluFusionDesignContract,
    ordinal: int,
) -> Path:
    registry = Path(design.compiler_claim_registry_root)
    if registry.is_symlink() or not registry.is_dir():
        raise ValueError("SEQAX_SILU_FUSION_CLAIM_REGISTRY_INVALID")
    registry_info = registry.lstat()
    if (
        not stat.S_ISDIR(registry_info.st_mode)
        or registry_info.st_uid != os.getuid()
        or registry_info.st_mode & 0o077
    ):
        raise ValueError("SEQAX_SILU_FUSION_CLAIM_REGISTRY_INVALID")
    path = registry / (f"{design.compiler_claim_key}-{design.design_id}-{ordinal}.replay.json")
    if path.is_symlink() or not path.is_file():
        raise ValueError("SEQAX_SILU_FUSION_REPLAY_SEAL_FILE_INVALID")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_mode & 0o077
    ):
        raise ValueError("SEQAX_SILU_FUSION_REPLAY_SEAL_FILE_INVALID")
    return path


def _verifier_environment(bundle: Path) -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(bundle / "src"),
        "PYTHONSAFEPATH": "1",
    }


def _independent_verify_capture(root: Path) -> dict[str, object]:
    bundle = root / "source" / "committed"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-m",
            _VERIFIER_MODULE,
            "--root",
            str(root),
            "--design",
            str(bundle / _DESIGN_RELATIVE_PATH),
        ],
        cwd="/",
        env=_verifier_environment(bundle),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _independent_verify_pair(pair_path: Path, bundle: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-m",
            _VERIFIER_MODULE,
            "--pair",
            str(pair_path),
            "--design",
            str(bundle / _DESIGN_RELATIVE_PATH),
        ],
        cwd="/",
        env=_verifier_environment(bundle),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def create_pair(
    pair_path: Path,
    design_path: Path,
    capture_roots: tuple[Path, Path],
) -> SeqaxSiluFusionCompilerPair:
    design = SeqaxSiluFusionDesignContract.model_validate_json(design_path.read_text())
    if design != default_seqax_silu_fusion_design_contract(design.runtime):
        raise ValueError("SEQAX_SILU_FUSION_DESIGN_NONCANONICAL")
    pair_path = _safe_pair_path(pair_path, design)
    roots = tuple(root.resolve(strict=True) for root in capture_roots)
    receipts = []
    seals = []
    for root in roots:
        receipt = SeqaxSiluFusionCompilerReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        replay = _independent_verify_capture(root)
        expected = {
            "capture_id": receipt.capture.capture_id,
            "receipt_id": receipt.receipt_id,
            "semantic_pair_id": receipt.capture.semantic_pair_id,
        }
        if replay != expected:
            raise ValueError("SEQAX_SILU_FUSION_PAIR_CAPTURE_REPLAY_MISMATCH")
        receipts.append(receipt)
        seals.append(
            SeqaxSiluFusionCompilerReplaySeal.model_validate_json(
                _replay_seal_path(design, receipt.capture.capture_ordinal).read_text()
            )
        )
    pair = SeqaxSiluFusionCompilerPair(
        design_id=design.design_id,
        captures=tuple(
            seqax_silu_fusion_compiler_pair_member(root, receipt, seal)
            for root, receipt, seal in zip(roots, receipts, seals, strict=True)
        ),
        independent_replay_performed=True,
        model_outputs_executed=False,
        correctness_outputs_collected=False,
        timing_collected=False,
        profile_collected=False,
    )
    _write_json_exclusive(
        pair_path,
        pair.model_dump(mode="json", exclude_computed_fields=True),
    )
    replay = _independent_verify_pair(pair_path, roots[0] / "source" / "committed")
    expected_replay = {
        "capture_ids": [value.capture_id for value in pair.captures],
        "pair_id": pair.pair_id,
    }
    if replay != expected_replay:
        raise ValueError("SEQAX_SILU_FUSION_PAIR_REPLAY_MISMATCH")
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    if len(arguments.capture_root) != 2:
        raise ValueError("SEQAX_SILU_FUSION_PAIR_CAPTURE_COUNT_MISMATCH")
    pair = create_pair(
        arguments.output,
        arguments.design,
        tuple(arguments.capture_root),
    )
    print(
        json.dumps(
            {
                "capture_ids": [value.capture_id for value in pair.captures],
                "pair_id": pair.pair_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
