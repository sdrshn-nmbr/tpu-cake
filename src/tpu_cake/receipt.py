from __future__ import annotations

import hashlib
from pathlib import Path

from tpu_cake.contracts import ArtifactRole, KernelExperiment, RunReceipt, RunStatus
from tpu_cake.runner import MatmulRunResult
from tpu_cake.search import (
    MatmulSearchContract,
    MatmulSearchResult,
    validate_matmul_search_result,
)


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
    provenance = receipt.search_provenance
    if provenance is None:
        return
    by_role = {
        role: [artifact for artifact in receipt.artifacts if artifact.role is role]
        for role in (ArtifactRole.SEARCH_CONTRACT, ArtifactRole.SEARCH_RESULT)
    }
    if any(len(artifacts) != 1 for artifacts in by_role.values()):
        raise ValueError("search provenance requires one contract and one result artifact")
    contract_artifact = by_role[ArtifactRole.SEARCH_CONTRACT][0]
    result_artifact = by_role[ArtifactRole.SEARCH_RESULT][0]

    def resolve(path: str) -> Path:
        value = Path(path)
        return root / value if root is not None and not value.is_absolute() else value

    contract_path = resolve(contract_artifact.path)
    result_path = resolve(result_artifact.path)
    if provenance.contract_sha256 != contract_artifact.sha256:
        raise ValueError("search contract identity does not match provenance")
    if provenance.result_sha256 != result_artifact.sha256:
        raise ValueError("search result identity does not match provenance")
    contract = MatmulSearchContract.model_validate_json(contract_path.read_text())
    result = MatmulSearchResult.model_validate_json(result_path.read_text())
    if provenance.search_id != contract.search_id or result.search_id != contract.search_id:
        raise ValueError("search identity does not match provenance")
    validate_matmul_search_result(contract_path.parent, contract, result)
    if result.winner != provenance.winner or len(result.run_results) != provenance.run_count:
        raise ValueError("search winner or run count does not match provenance")
    candidate = next(
        (candidate for candidate in contract.candidates if candidate.name == result.winner),
        None,
    )
    if candidate is None or (candidate.tile_m, candidate.tile_n) != (
        provenance.tile_m,
        provenance.tile_n,
    ):
        raise ValueError("search winner tile does not match provenance")
    winner_schedules = {
        MatmulRunResult.model_validate_json(
            (contract_path.parent / run_path / "result.json").read_text()
        ).schedule_sha256
        for run_path in result.run_results
        if Path(run_path).name == result.winner
    }
    if winner_schedules != {provenance.winner_schedule_sha256}:
        raise ValueError("search winner schedule does not match provenance")
    if provenance.winner_schedule_sha256 != receipt.schedule_sha256:
        raise ValueError("search winner schedule does not match finalist receipt")
