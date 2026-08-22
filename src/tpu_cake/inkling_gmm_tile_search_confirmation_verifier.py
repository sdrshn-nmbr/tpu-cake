from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tpu_cake.identity import json_sha256, model_identity_sha256
from tpu_cake.inkling_gmm_route_corpus import InklingGmmRouteCorpusReport
from tpu_cake.inkling_gmm_tile_search import (
    GmmArmName,
    GmmConfirmationObservation,
    GmmConfirmationStatistics,
    GmmPolicyPair,
    InklingGmmTileSearchContract,
    confirmation_statistics,
)
from tpu_cake.inkling_gmm_tile_search_correctness import (
    GMM_CONFIRMATION_CORRECTNESS_SCHEMA,
    GmmConfirmationCorrectnessReport,
)
from tpu_cake.inkling_gmm_tile_search_verifier import (
    _COMPILED_POLICY_KEYS,
    _POLICY_KEYS,
    _artifact_path,
    _exact_keys,
    _expected_custom_call_counts,
    _expected_scopes,
    _hex,
    _object,
    _verify_compiler_hlo,
    _verify_stablehlo,
    verify_screening,
)

_CONFIRMATION_SCHEMA = "inkling-gmm-tile-search-confirmation-observations-v1"
_VERIFIED_SCHEMA = "inkling-gmm-tile-search-confirmation-verification-v1"
_CONFIRMATION_KEYS = {
    "schema_version",
    "search_id",
    "contract_sha256",
    "route_report_id",
    "route_report_sha256",
    "raw_screening_path",
    "raw_screening_sha256",
    "verified_screening_path",
    "verified_screening_sha256",
    "screening_verification_id",
    "confirmation_verifier_source_sha256",
    "candidate_correctness",
    "candidate_compiled_policy",
    "warmup",
    "confirmation_observations",
    "confirmation_statistics",
    "claims",
    "limitations",
}
_CORRECTNESS_KEYS = {"schema_version", "report_id", "path", "sha256"}
_WARMUP_KEYS = {"order", "durations_ns"}
_CLAIMS_KEYS = {
    "candidate_correctness_gate_passed",
    "paired_confirmation_run",
    "promotion_authorized",
    "immutable_receipt_created",
}
_LIMITATIONS = [
    "This observation file is mutable until an immutable receipt is created.",
    "The confirmation applies only to the declared route corpus, ABI, runtime, and source revisions.",
]


def _fail(code: str, **context: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
    raise ValueError(f"INKLING_GMM_CONFIRM_VERIFY_{code} {suffix}".rstrip())


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_contract(path: Path) -> tuple[InklingGmmTileSearchContract, bytes]:
    try:
        raw = path.read_bytes()
        return InklingGmmTileSearchContract.model_validate_json(raw), raw
    except (OSError, ValueError) as error:
        _fail("CONTRACT_READ", path=path, error=error)


def _read_route_report(path: Path) -> tuple[InklingGmmRouteCorpusReport, bytes]:
    try:
        raw = path.read_bytes()
        return InklingGmmRouteCorpusReport.model_validate_json(raw), raw
    except (OSError, ValueError) as error:
        _fail("ROUTE_REPORT_READ", path=path, error=error)


def _verify_candidate_compilation(
    contract: InklingGmmTileSearchContract,
    value: object,
    *,
    artifact_root: Path,
    candidate: GmmPolicyPair,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("COMPILED_POLICY_OBJECT")
    _exact_keys(value, _COMPILED_POLICY_KEYS, label="CONFIRMATION_COMPILED_POLICY")
    policy_payload = value["policy"]
    if not isinstance(policy_payload, dict):
        _fail("COMPILED_POLICY_VALUE")
    _exact_keys(policy_payload, _POLICY_KEYS, label="CONFIRMATION_POLICY")
    try:
        policy = GmmPolicyPair.model_validate(
            {key: policy_payload[key] for key in ("gate_up", "down")}
        )
    except ValueError as error:
        _fail("COMPILED_POLICY_VALUE", error=error)
    if policy != candidate or policy_payload["name"] != candidate.name:
        _fail("COMPILED_POLICY_BINDING")
    hashes = (value["stablehlo_sha256"], value["compiler_hlo_sha256"])
    sizes = (value["stablehlo_bytes"], value["compiler_hlo_bytes"])
    if any(not _hex(item, 64) for item in hashes) or any(
        type(size) is not int or size <= 0 for size in sizes
    ):
        _fail("COMPILED_POLICY_METADATA")
    expected_scopes = _expected_scopes(contract, policy)
    expected_counts = _expected_custom_call_counts(contract, policy)
    if value["gmm_scope_labels"] != list(expected_scopes):
        _fail("COMPILED_POLICY_SCOPES")
    if value["gmm_custom_call_counts"] != expected_counts:
        _fail("COMPILED_POLICY_CUSTOM_CALL_COUNTS")
    stablehlo = _artifact_path(artifact_root, value["stablehlo_path"], label="STABLEHLO")
    compiler_hlo = _artifact_path(
        artifact_root,
        value["compiler_hlo_path"],
        label="COMPILER_HLO",
    )
    _verify_stablehlo(
        stablehlo,
        expected_sha256=value["stablehlo_sha256"],
        expected_bytes=value["stablehlo_bytes"],
    )
    _verify_compiler_hlo(
        compiler_hlo,
        expected_sha256=value["compiler_hlo_sha256"],
        expected_bytes=value["compiler_hlo_bytes"],
        expected_scopes=expected_scopes,
        expected_custom_call_counts=expected_counts,
    )
    return {
        "policy": policy.model_dump(mode="json"),
        "stablehlo_sha256": value["stablehlo_sha256"],
        "compiler_hlo_sha256": value["compiler_hlo_sha256"],
        "gmm_scope_labels": list(expected_scopes),
        "gmm_custom_call_counts": expected_counts,
    }


def _verify_candidate_correctness(
    contract: InklingGmmTileSearchContract,
    route_report: InklingGmmRouteCorpusReport,
    value: object,
    *,
    artifact_root: Path,
    candidate: GmmPolicyPair,
) -> dict[str, str]:
    if not isinstance(value, dict):
        _fail("CORRECTNESS_OBJECT")
    _exact_keys(value, _CORRECTNESS_KEYS, label="CONFIRMATION_CORRECTNESS")
    if (
        value["schema_version"] != GMM_CONFIRMATION_CORRECTNESS_SCHEMA
        or not _hex(value["report_id"], 64)
        or not _hex(value["sha256"], 64)
    ):
        _fail("CORRECTNESS_METADATA")
    path = _artifact_path(artifact_root, value["path"], label="CONFIRMATION_CORRECTNESS")
    raw = path.read_bytes()
    if _sha256(raw) != value["sha256"]:
        _fail("CORRECTNESS_SHA256")
    try:
        report = GmmConfirmationCorrectnessReport.model_validate_json(raw)
    except ValueError as error:
        _fail("CORRECTNESS_REPORT", error=error)
    if (
        report.report_id
        != model_identity_sha256(report, exclude={"report_id"})
        or report.report_id != value["report_id"]
        or report.search_id != contract.search_id
        or report.route_report_id != route_report.report_id
        or report.numerical_contract_id != contract.correctness.numerical_contract_id
        or report.candidate != candidate
    ):
        _fail("CORRECTNESS_BINDING")
    return {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "sha256": value["sha256"],
    }


def verify_confirmation(
    *,
    contract_path: Path,
    route_report_path: Path,
    confirmation_path: Path,
) -> dict[str, Any]:
    contract, contract_raw = _read_contract(contract_path)
    if _sha256(Path(__file__).read_bytes()) != contract.confirmation_verifier_source_sha256:
        _fail("VERIFIER_SOURCE_SHA256")
    route_report, route_report_raw = _read_route_report(route_report_path)
    raw_bytes = confirmation_path.read_bytes()
    raw = _object(raw_bytes, label="CONFIRMATION")
    _exact_keys(raw, _CONFIRMATION_KEYS, label="CONFIRMATION")
    if (
        raw["schema_version"] != _CONFIRMATION_SCHEMA
        or raw["search_id"] != contract.search_id
        or raw["contract_sha256"] != _sha256(contract_raw)
        or raw["route_report_id"] != route_report.report_id
        or raw["route_report_sha256"] != _sha256(route_report_raw)
        or raw["confirmation_verifier_source_sha256"]
        != contract.confirmation_verifier_source_sha256
    ):
        _fail("ROOT_BINDING")
    artifact_root = confirmation_path.parent
    raw_screening_path = _artifact_path(
        artifact_root,
        raw["raw_screening_path"],
        label="RAW_SCREENING",
    )
    verified_screening_path = _artifact_path(
        artifact_root,
        raw["verified_screening_path"],
        label="VERIFIED_SCREENING",
    )
    if (
        _sha256(raw_screening_path.read_bytes()) != raw["raw_screening_sha256"]
        or _sha256(verified_screening_path.read_bytes())
        != raw["verified_screening_sha256"]
    ):
        _fail("SCREENING_SHA256")
    recomputed_screening = verify_screening(
        contract_path=contract_path,
        route_report_path=route_report_path,
        raw_observations_path=raw_screening_path,
    )
    stored_screening = _object(verified_screening_path.read_bytes(), label="VERIFIED_SCREENING")
    if stored_screening != recomputed_screening or (
        raw["screening_verification_id"] != recomputed_screening["verification_id"]
    ):
        _fail("SCREENING_VERIFICATION_BINDING")
    candidate = GmmPolicyPair(
        gate_up=GmmArmName(recomputed_screening["finalists"]["gate-up"]),
        down=GmmArmName(recomputed_screening["finalists"]["down"]),
    )
    correctness = _verify_candidate_correctness(
        contract,
        route_report,
        raw["candidate_correctness"],
        artifact_root=artifact_root,
        candidate=candidate,
    )
    compiled = _verify_candidate_compilation(
        contract,
        raw["candidate_compiled_policy"],
        artifact_root=artifact_root,
        candidate=candidate,
    )
    warmup = raw["warmup"]
    if not isinstance(warmup, dict):
        _fail("WARMUP_OBJECT")
    _exact_keys(warmup, _WARMUP_KEYS, label="WARMUP")
    baseline = GmmPolicyPair(
        gate_up=GmmArmName.INCUMBENT,
        down=GmmArmName.INCUMBENT,
    )
    expected_warmup_count = 2 * contract.confirmation.warmup_full_corpus_blocks_per_arm
    if (
        warmup["order"] != [baseline.name, candidate.name]
        or not isinstance(warmup["durations_ns"], list)
        or len(warmup["durations_ns"]) != expected_warmup_count
        or any(type(item) is not int or item <= 0 for item in warmup["durations_ns"])
    ):
        _fail("WARMUP_INVENTORY")
    try:
        observations = tuple(
            GmmConfirmationObservation.model_validate(item)
            for item in raw["confirmation_observations"]
        )
        claimed_statistics = GmmConfirmationStatistics.model_validate(
            raw["confirmation_statistics"]
        )
    except (TypeError, ValueError) as error:
        _fail("CONFIRMATION_VALUES", error=error)
    recomputed_statistics = confirmation_statistics(contract, candidate, observations)
    if claimed_statistics != recomputed_statistics:
        _fail("STATISTICS_MISMATCH")
    claims = raw["claims"]
    if not isinstance(claims, dict):
        _fail("CLAIMS_OBJECT")
    _exact_keys(claims, _CLAIMS_KEYS, label="CLAIMS")
    expected_claims = {
        "candidate_correctness_gate_passed": True,
        "paired_confirmation_run": True,
        "promotion_authorized": recomputed_statistics.confirmed,
        "immutable_receipt_created": False,
    }
    if claims != expected_claims or raw["limitations"] != _LIMITATIONS:
        _fail("CLAIMS_BINDING")
    report_payload = {
        "schema_version": _VERIFIED_SCHEMA,
        "search_id": contract.search_id,
        "contract_sha256": _sha256(contract_raw),
        "route_report_id": route_report.report_id,
        "route_report_sha256": _sha256(route_report_raw),
        "confirmation_observations_sha256": _sha256(raw_bytes),
        "screening_verification_id": recomputed_screening["verification_id"],
        "candidate_correctness": correctness,
        "candidate_compiled_policy": compiled,
        "warmup": warmup,
        "confirmation_statistics": recomputed_statistics.model_dump(mode="json"),
        "claims": {
            "screening_independently_replayed": True,
            "candidate_correctness_gate_bound": True,
            "confirmation_independently_replayed": True,
            "promotion_authorized": recomputed_statistics.confirmed,
            "immutable_receipt_created": False,
        },
        "limitations": _LIMITATIONS,
    }
    return {
        "verification_id": json_sha256(report_payload),
        **report_payload,
    }


def write_verified_confirmation(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify an Inkling GMM paired confirmation."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = verify_confirmation(
            contract_path=arguments.contract,
            route_report_path=arguments.route_report,
            confirmation_path=arguments.confirmation,
        )
        write_verified_confirmation(arguments.output, report)
    except (OSError, ValueError) as error:
        print(error)
        return 1
    print(
        "INKLING_GMM_CONFIRMATION_INDEPENDENTLY_VERIFIED "
        f"verification_id={report['verification_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
