import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu_cake import inkling_decode_operation_selection_verifier as independent_verifier
from tpu_cake.evidence import (
    ArtifactEvidence,
    CaptureAssessment,
    CaptureEvidence,
    CounterEvidence,
    ProgramEvidence,
)
from tpu_cake.inkling_decode_operation_selection import (
    InklingDecodeOperationSelectionContract,
    InklingDecodeOperationSelectionReport,
    _parser,
    derive_inkling_decode_operation_selection,
    validate_inkling_decode_operation_selection,
    write_inkling_decode_operation_selection,
)
from tpu_cake.inkling_decode_profile import InklingDecodeProfileContract

_MAIN_HLO = "59b0140305f4e5c9780bbc3c8bdbbd8fbcba84669b9ae7fd3e3baee35602038a"
_FIRST = "gmm_v2-g_32-m_288-k_4096-n_2048-tm_128-tk_4096-tn_2048"
_SECOND = "gmm_v2-g_32-m_288-k_2048-n_4096-tm_128-tk_2048-tn_4096"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, separators=(",", ":")))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_contract() -> InklingDecodeProfileContract:
    return InklingDecodeProfileContract.model_validate_json(
        Path("contracts/inkling-whole-decode-profile-v1.json").read_text()
    )


def _assessment(*, accepted: bool = True) -> CaptureAssessment:
    capture = CaptureEvidence(
        xplane=ArtifactEvidence(path=Path("capture.xplane.pb"), size_bytes=1, sha256="a" * 64),
        hlo_stats=ArtifactEvidence(path=Path("hlo_stats.json"), size_bytes=1, sha256="b" * 64),
        planes=(),
        counters=CounterEvidence(
            hbm_read_names=0,
            hbm_write_names=0,
            cycle_names=0,
            periodic_counter_names=(),
            periodic_samples_per_tpu_core={},
        ),
        programs=(
            ProgramEvidence(
                program_id="7",
                name="jit_jitted_run_model(7)",
                timed_self_us=12.0,
                marker_counts={"gmm_v2": 2},
                forbidden_fragment_hits={},
            ),
        ),
        timed_program_ids=frozenset({"7"}),
    )
    return CaptureAssessment(
        expectation={
            "name": "decode",
            "stage": "steady_decode",
            "minimum_tpu_device_planes": 1,
        },
        capture=capture,
        findings=()
        if accepted
        else (
            {
                "code": "REJECTED",
                "severity": "error",
                "message": "rejected",
                "evidence": [],
            },
        ),
    )


def _profile_payload() -> dict[str, object]:
    return {
        "deviceType": "TPU",
        "byProgram": {
            "name": "by_program",
            "metrics": {"rawTime": 12_000_000},
            "children": [
                {
                    "name": "jit_jitted_run_model(7)",
                    "metrics": {"rawTime": 10_000_000},
                    "children": [
                        {
                            "name": "custom-call",
                            "metrics": {"rawTime": 5_000_000, "occurrences": 9},
                            "children": [
                                {
                                    "name": f"{_FIRST}.83 and its duplicate(s)",
                                    "metrics": {"rawTime": 3_000_000, "occurrences": 3},
                                    "children": [],
                                },
                                {
                                    "name": f"{_SECOND}.41 and its duplicate(s)",
                                    "metrics": {"rawTime": 1_000_000, "occurrences": 1},
                                    "children": [],
                                },
                                {
                                    "name": "other_kernel.9 and its duplicate(s)",
                                    "metrics": {"rawTime": 1_000_000, "occurrences": 5},
                                    "children": [],
                                },
                            ],
                        },
                        {
                            "name": "async-done",
                            "metrics": {"rawTime": 2_000_000, "occurrences": 6},
                            "children": [{"name": "capped", "metrics": {"rawTime": 1}}],
                        },
                        {
                            "name": "loop fusion",
                            "metrics": {"rawTime": 2_500_000, "occurrences": 7},
                            "children": [],
                        },
                    ],
                }
            ],
        },
        "byProgramExcludeIdle": {},
    }


def _hlo_stats_payload() -> dict[str, object]:
    columns = [
        "program_id",
        "category",
        "hlo_op_name",
        "occurrences",
        "total_self_time",
        "dma_stall_percent",
        "operational_intensity",
        "bound_by",
    ]
    values = [
        ["7", "custom-call", f"{_FIRST}.83", 3, 3.0, 0.0, 8.9, "HBM"],
        ["7", "custom-call", f"{_SECOND}.41", 1, 1.0, 0.0, 8.8, "HBM"],
    ]
    return {
        "cols": [{"id": name} for name in columns],
        "rows": [{"c": [{"v": value} for value in row]} for row in values],
    }


def _contract(
    op_profile: Path,
    hlo_stats: Path,
    *,
    profile_contract: InklingDecodeProfileContract | None = None,
    maximum_unattributed_share: str = "0.1",
) -> InklingDecodeOperationSelectionContract:
    profile = profile_contract or _profile_contract()
    return InklingDecodeOperationSelectionContract(
        profile_contract_id=profile.contract_id,
        profile_request_sha256="e" * 64,
        prompt_corpus_sha256="f" * 64,
        xplane_sha256="a" * 64,
        op_profile_sha256=_sha256(op_profile),
        hlo_stats_sha256=_sha256(hlo_stats),
        producer_source_sha256="1" * 64,
        verifier_source_sha256="2" * 64,
        uv_lock_sha256="3" * 64,
        required_winner_partition_key=f"custom-call-family/{_FIRST}",
        maximum_unattributed_device_op_share_of_main_program=maximum_unattributed_share,
        candidate_hlo_prefix="gmm_v2-",
        expected_candidate_kernel_families=tuple(sorted((_FIRST, _SECOND))),
        minimum_candidate_device_op_share_of_main_program="0.3",
        expected_bound_by=("HBM",),
        expected_claim={
            "main_program_raw_time_ps": 10_000_000,
            "winner_raw_time_ps": 3_000_000,
            "candidate_raw_time_ps": 4_000_000,
            "candidate_occurrences": 4,
            "candidate_hlo_rows": 2,
            "candidate_operational_intensity_min": "8.8",
            "candidate_operational_intensity_max": "8.9",
        },
    )


def _derive(
    tmp_path: Path,
    *,
    profile_payload: dict[str, object] | None = None,
    hlo_payload: dict[str, object] | None = None,
    assessment: CaptureAssessment | None = None,
    profile_contract: InklingDecodeProfileContract | None = None,
    contract_update: dict[str, object] | None = None,
):
    op_profile = _write_json(tmp_path / "op_profile.json", profile_payload or _profile_payload())
    hlo_stats = _write_json(tmp_path / "hlo_stats.json", hlo_payload or _hlo_stats_payload())
    profile = profile_contract or _profile_contract()
    contract = _contract(op_profile, hlo_stats, profile_contract=profile)
    if contract_update:
        contract = contract.model_copy(update=contract_update)
    return derive_inkling_decode_operation_selection(
        assessment=assessment or _assessment(),
        semantic_hlo_sha256_by_program={"jit_jitted_run_model(7)": _MAIN_HLO},
        profile_contract=profile,
        profile_request_sha256="e" * 64,
        prompt_corpus_sha256="f" * 64,
        producer_source_sha256="1" * 64,
        verifier_source_sha256="2" * 64,
        uv_lock_sha256="3" * 64,
        op_profile_path=op_profile,
        hlo_stats_path=hlo_stats,
        contract=contract,
    )


def test_operation_selection_ranks_an_exhaustive_disjoint_partition(tmp_path: Path) -> None:
    report = _derive(tmp_path)

    assert report.main_program_raw_time_ps == 10_000_000
    assert report.attributed_raw_time_ps == 9_500_000
    assert report.unattributed_raw_time_ps == 500_000
    assert report.winner_partition_key == f"custom-call-family/{_FIRST}"
    assert report.winner_raw_time_ps == 3_000_000
    assert [item.key for item in report.operation_ranking] == [
        f"custom-call-family/{_FIRST}",
        "xprof-category/loop fusion",
        "xprof-category/async-done",
        f"custom-call-family/{_SECOND}",
        "custom-call-family/other_kernel",
    ]
    assert report.candidate_raw_time_ps == 4_000_000
    assert report.candidate_occurrences == 4
    assert report.candidate_hlo_rows == 2
    assert str(report.candidate_device_op_share_of_main_program) == "0.4"
    assert report.attribution_scope == "xprof-summed-device-op-self-time"
    assert report.is_end_to_end_latency is False


@pytest.mark.parametrize("competitor", ("custom", "category"))
def test_operation_selection_rejects_a_hotter_non_candidate(
    tmp_path: Path, competitor: str
) -> None:
    profile = _profile_payload()
    main = profile["byProgram"]["children"][0]
    if competitor == "custom":
        custom = main["children"][0]
        custom["children"][2]["metrics"]["rawTime"] = 4_000_000
        custom["metrics"]["rawTime"] = 8_000_000
        main["metrics"]["rawTime"] = 13_000_000
    else:
        main["children"][2]["metrics"]["rawTime"] = 4_000_000
        main["metrics"]["rawTime"] = 11_500_000

    with pytest.raises(ValueError, match="REQUIRED_WINNER_MISMATCH"):
        _derive(tmp_path, profile_payload=profile)


@pytest.mark.parametrize("metric", ("rawTime", "occurrences"))
def test_operation_selection_rejects_custom_container_drift(tmp_path: Path, metric: str) -> None:
    profile = _profile_payload()
    profile["byProgram"]["children"][0]["children"][0]["metrics"][metric] += 1

    with pytest.raises(ValueError, match="CUSTOM_CALL_CONTAINER_MISMATCH"):
        _derive(tmp_path, profile_payload=profile)


def test_operation_selection_rejects_a_malformed_custom_call_name(tmp_path: Path) -> None:
    profile = _profile_payload()
    profile["byProgram"]["children"][0]["children"][0]["children"][2]["name"] = "bad"

    with pytest.raises(ValueError, match="CUSTOM_CALL_FAMILY_INVALID"):
        _derive(tmp_path, profile_payload=profile)


def test_operation_selection_rejects_main_time_underflow(tmp_path: Path) -> None:
    profile = _profile_payload()
    profile["byProgram"]["children"][0]["metrics"]["rawTime"] = 9_000_000

    with pytest.raises(ValueError, match="MAIN_TIME_UNDERFLOW"):
        _derive(tmp_path, profile_payload=profile)


def test_operation_selection_enforces_the_unattributed_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="UNATTRIBUTED_SHARE_ABOVE_GATE"):
        _derive(
            tmp_path,
            contract_update={
                "maximum_unattributed_device_op_share_of_main_program": Decimal("0.01")
            },
        )


def test_operation_selection_rejects_an_unaccepted_capture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PROFILE_NOT_ACCEPTED"):
        _derive(tmp_path, assessment=_assessment(accepted=False))


def test_operation_selection_rejects_artifact_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="OP_PROFILE_HASH_MISMATCH"):
        _derive(tmp_path, contract_update={"op_profile_sha256": "0" * 64})


@pytest.mark.parametrize(
    ("cell_index", "value", "error"),
    (
        (3, 2, "OCCURRENCES_MISMATCH"),
        (4, 2.0, "TIME_MISMATCH"),
        (7, "MXU", "BOUND_CLASS_MISMATCH"),
    ),
)
def test_operation_selection_rejects_cross_view_semantic_drift(
    tmp_path: Path, cell_index: int, value: object, error: str
) -> None:
    hlo = _hlo_stats_payload()
    hlo["rows"][0]["c"][cell_index]["v"] = value

    with pytest.raises(ValueError, match=error):
        _derive(tmp_path, hlo_payload=hlo)


def test_operation_selection_uses_the_profile_main_program_authority(tmp_path: Path) -> None:
    profile = _profile_contract()
    programs = tuple(
        program.model_copy(update={"semantic_hlo_sha256": "0" * 64})
        if program.name_prefix == profile.main_program_prefix
        else program
        for program in profile.programs
    )
    changed = profile.model_copy(update={"programs": programs})

    with pytest.raises(ValueError, match="MAIN_HLO_MISMATCH"):
        _derive(tmp_path, profile_contract=changed)


def test_report_rejects_impossible_internal_arithmetic(tmp_path: Path) -> None:
    report = _derive(tmp_path)
    payload = report.model_dump(mode="json")
    payload["candidate_raw_time_ps"] = 1

    with pytest.raises(ValueError, match="candidate aggregate mismatch"):
        InklingDecodeOperationSelectionReport.model_validate(payload)


def test_report_write_read_replay_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _derive(tmp_path)
    path = tmp_path / "report.json"
    write_inkling_decode_operation_selection(path, report)
    assert InklingDecodeOperationSelectionReport.model_validate_json(path.read_text()) == report

    monkeypatch.setattr(
        "tpu_cake.inkling_decode_operation_selection.select_inkling_decode_operation",
        lambda **_kwargs: report,
    )
    monkeypatch.setattr(
        "tpu_cake.inkling_decode_operation_selection.verify_report_independently",
        lambda **_kwargs: None,
    )
    assert (
        validate_inkling_decode_operation_selection(
            path,
            capture_root=tmp_path,
            request_path=tmp_path,
            prompt_cases_path=tmp_path,
            profile_contract=_profile_contract(),
            selection_contract=_contract(tmp_path / "op_profile.json", tmp_path / "hlo_stats.json"),
        )
        == report
    )


def test_independent_verifier_recomputes_the_public_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    op_profile = _write_json(tmp_path / "op_profile.json", _profile_payload())
    hlo_stats = _write_json(tmp_path / "hlo_stats.json", _hlo_stats_payload())
    profile = _profile_contract()
    producer_hash = _sha256(Path("src/tpu_cake/inkling_decode_operation_selection.py"))
    verifier_hash = _sha256(Path("src/tpu_cake/inkling_decode_operation_selection_verifier.py"))
    lock_hash = _sha256(Path("uv.lock"))
    contract = _contract(op_profile, hlo_stats).model_copy(
        update={
            "producer_source_sha256": producer_hash,
            "verifier_source_sha256": verifier_hash,
            "uv_lock_sha256": lock_hash,
        }
    )
    report = derive_inkling_decode_operation_selection(
        assessment=_assessment(),
        semantic_hlo_sha256_by_program={"jit_jitted_run_model(7)": _MAIN_HLO},
        profile_contract=profile,
        profile_request_sha256="e" * 64,
        prompt_corpus_sha256="f" * 64,
        producer_source_sha256=producer_hash,
        verifier_source_sha256=verifier_hash,
        uv_lock_sha256=lock_hash,
        op_profile_path=op_profile,
        hlo_stats_path=hlo_stats,
        contract=contract,
    )
    report_path = tmp_path / "report.json"
    write_inkling_decode_operation_selection(report_path, report)
    monkeypatch.setattr(
        independent_verifier,
        "export_xprof_capture",
        lambda *_args, **_kwargs: SimpleNamespace(
            exports=(
                SimpleNamespace(tool="op_profile", output=op_profile),
                SimpleNamespace(tool="hlo_stats", output=hlo_stats),
            )
        ),
    )

    independent_verifier.verify_report_independently(
        report_path=report_path,
        capture_root=tmp_path,
        profile_contract=profile,
        selection_contract=contract,
    )

    forged = json.loads(report_path.read_text())
    forged["candidate_kernel_families"][0]["name"] = "fabricated-family"
    forged["candidate_kernel_families"][0]["bound_by"] = ["MXU"]
    identity = dict(forged)
    identity.pop("report_id")
    forged["report_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(report_path, forged)

    with pytest.raises(ValueError, match="INDEPENDENT_CANDIDATE_KERNEL_FAMILIES_MISMATCH"):
        independent_verifier.verify_report_independently(
            report_path=report_path,
            capture_root=tmp_path,
            profile_contract=profile,
            selection_contract=contract,
        )


def test_operation_selection_contract_is_committed_and_module_cli_is_public() -> None:
    contract = InklingDecodeOperationSelectionContract.model_validate_json(
        Path("contracts/inkling-whole-decode-operation-selection-v1.json").read_text()
    )

    assert contract.selection_rule == "largest-xprof-device-op-self-time-partition"
    assert "main_program_prefix" not in type(contract).model_fields
    assert (
        _parser()
        .parse_args(
            [
                "select",
                "capture",
                "--request",
                "request.json",
                "--prompt-cases",
                "prompts.json",
                "--profile-contract",
                "profile.json",
                "--selection-contract",
                "selection.json",
                "--output",
                "report.json",
            ]
        )
        .command
        == "select"
    )
    assert (
        _parser()
        .parse_args(
            [
                "verify",
                "report.json",
                "--capture",
                "capture",
                "--request",
                "request.json",
                "--prompt-cases",
                "prompts.json",
                "--profile-contract",
                "profile.json",
                "--selection-contract",
                "selection.json",
            ]
        )
        .command
        == "verify"
    )


def test_committed_report_binds_the_documented_claim() -> None:
    report_path = Path("evidence/inkling/whole-decode-operation-selection-v1.json")
    report = InklingDecodeOperationSelectionReport.model_validate_json(report_path.read_text())
    contract = InklingDecodeOperationSelectionContract.model_validate_json(
        Path("contracts/inkling-whole-decode-operation-selection-v1.json").read_text()
    )
    claim = contract.expected_claim

    assert report.contract_id == contract.contract_id
    assert report.main_program_raw_time_ps == claim.main_program_raw_time_ps
    assert report.winner_raw_time_ps == claim.winner_raw_time_ps
    assert report.candidate_raw_time_ps == claim.candidate_raw_time_ps
    assert report.candidate_occurrences == claim.candidate_occurrences
    assert report.candidate_hlo_rows == claim.candidate_hlo_rows
    assert (
        min(family.operational_intensity_min for family in report.candidate_kernel_families)
        == claim.candidate_operational_intensity_min
    )
    assert (
        max(family.operational_intensity_max for family in report.candidate_kernel_families)
        == claim.candidate_operational_intensity_max
    )
    assert report.producer_source_sha256 == _sha256(
        Path("src/tpu_cake/inkling_decode_operation_selection.py")
    )
    assert report.verifier_source_sha256 == _sha256(
        Path("src/tpu_cake/inkling_decode_operation_selection_verifier.py")
    )
    assert report.uv_lock_sha256 == _sha256(Path("uv.lock"))
    assert (
        _sha256(report_path) == "1d39166dcddb646a0071348d91576367b52a35cd783901633f698733ff4de34f"
    )
    assert (
        "The committed replay is report "
        "`22d07f2f327f24c8d22e628268d7de96481acda0d1d2df6daf82c068b84766a1` "
        "with file SHA-256 "
        "`1d39166dcddb646a0071348d91576367b52a35cd783901633f698733ff4de34f`."
        in Path("README.md").read_text()
    )
