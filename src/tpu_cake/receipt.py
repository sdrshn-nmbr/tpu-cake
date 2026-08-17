from __future__ import annotations

import hashlib
from pathlib import Path

from tpu_cake.contracts import KernelExperiment, RunReceipt, RunStatus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(
    receipt: RunReceipt,
    experiment: KernelExperiment,
    *,
    root: Path | None = None,
) -> None:
    if receipt.experiment_id != experiment.experiment_id:
        raise ValueError("receipt experiment identity does not match the experiment")
    if receipt.schedule_sha256 != experiment.schedule_sha256:
        raise ValueError("receipt schedule identity does not match the experiment")
    required_properties = tuple(experiment.workload.numerical.semantic_properties)
    if receipt.required_semantic_properties != required_properties:
        raise ValueError("receipt semantic requirements do not match the experiment")
    if receipt.status is not RunStatus.PASSED:
        return
    for artifact in receipt.artifacts:
        path = Path(artifact.path)
        if root is not None and not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ValueError(f"receipt artifact is missing: {path}")
        if path.stat().st_size != artifact.size_bytes:
            raise ValueError(f"receipt artifact size changed: {path}")
        if _sha256(path) != artifact.sha256:
            raise ValueError(f"receipt artifact hash changed: {path}")
