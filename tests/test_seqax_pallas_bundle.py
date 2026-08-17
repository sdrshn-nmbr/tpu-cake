import hashlib
import json
from pathlib import Path

import pytest

from tpu_cake.contracts import (
    PHASE_REQUIRED_ROLES_BY_PROFILE,
    ArtifactReference,
    ArtifactRole,
    EvidenceProfile,
)
from tpu_cake.evidence import (
    ArtifactEvidence,
    CaptureAssessment,
    CaptureEvidence,
    CounterEvidence,
    PlaneEvidence,
    ProgramEvidence,
)
from tpu_cake.seqax_pallas_bundle import (
    _canonical_profile_assessment,
    _counter_experiment,
    _expected_result_role,
    _preflight_phase_files,
    _require_owned_safe_root,
    _resolve_result_artifact,
    _trusted_experiment,
    _validate_capture,
    _validate_xprof_exports,
)


def _assessment(*, counter_names: tuple[str, ...]) -> CaptureAssessment:
    artifact = ArtifactEvidence(path=Path("trace"), size_bytes=1, sha256="0" * 64)
    counters = CounterEvidence(
        hbm_read_names=128,
        hbm_write_names=128,
        cycle_names=185,
        periodic_counter_names=counter_names,
        periodic_samples_per_tpu_core=({"0": 2, "2": 2, "4": 2, "6": 2} if counter_names else {}),
    )
    expectation = _trusted_experiment().profile
    if counter_names:
        expectation = _counter_experiment(_trusted_experiment()).profile
    return CaptureAssessment(
        expectation=expectation,
        capture=CaptureEvidence(
            xplane=artifact,
            hlo_stats=artifact,
            planes=tuple(
                PlaneEvidence(
                    name=f"/device:TPU:{index}",
                    line_count=1,
                    event_count=1,
                    tensor_core_event_count=0,
                )
                for index in range(8)
            ),
            counters=counters,
            programs=(
                ProgramEvidence(
                    program_id="1",
                    name="main",
                    timed_self_us=1,
                    marker_counts={
                        "pallas_call": 1,
                        "all-gather": 1,
                        "reduce_scatter": 1,
                    },
                    forbidden_fragment_hits={},
                ),
            ),
            timed_program_ids=frozenset({"1"}),
        ),
        findings=(),
    )


def test_seqax_pallas_receipt_profile_has_complete_phase_contracts() -> None:
    roles = PHASE_REQUIRED_ROLES_BY_PROFILE[EvidenceProfile.SEQAX_PHYSICAL_PALLAS_FORWARD]

    assert len(roles) == 5
    assert all(phase_roles for phase_roles in roles.values())


def test_seqax_pallas_profile_binds_the_physical_program() -> None:
    experiment = _trusted_experiment()

    assert experiment.schedule_sha256
    assert experiment.profile.required_timed_hlo_markers == (
        "pallas_call",
        "all-gather",
        "reduce_scatter",
    )
    assert experiment.profile.minimum_tpu_device_planes == 8


def test_seqax_pallas_counter_contract_requires_hardware_families() -> None:
    experiment = _counter_experiment(_trusted_experiment())

    assert experiment.profile.require_hbm_read_counters
    assert experiment.profile.require_hbm_write_counters
    assert experiment.profile.require_cycle_counters
    assert experiment.profile.minimum_counter_device_planes == 4


def test_seqax_pallas_capture_requires_periodic_mxu_series() -> None:
    _validate_capture(_assessment(counter_names=("COUNT_MXU_BUSY_0",)), counters=True)

    with pytest.raises(ValueError, match="MXU_PERIODIC_COUNTER_MISSING"):
        _validate_capture(_assessment(counter_names=("COUNT_OTHER",)), counters=True)


def test_seqax_pallas_result_artifact_roles_come_from_trusted_paths() -> None:
    assert _expected_result_role("physical.xdsl") is ArtifactRole.PHYSICAL_IR
    assert _expected_result_role("lowered_pallas.py") is ArtifactRole.PALLAS_SOURCE
    assert _expected_result_role("inputs/12.npy") is ArtifactRole.CORRECTNESS_INPUT
    assert (
        _expected_result_role("profile/plugins/profile/run/jit_main(1).hlo_proto.pb")
        is ArtifactRole.PROFILE_AUXILIARY
    )
    with pytest.raises(ValueError, match="PATH_UNRECOGNIZED"):
        _expected_result_role("renamed-physical.xdsl")
    with pytest.raises(ValueError, match="PATH_UNRECOGNIZED"):
        _expected_result_role("inputs/0.npy")


def test_seqax_pallas_result_artifact_rechecks_direct_identity(tmp_path: Path) -> None:
    phase = tmp_path / "timing"
    phase.mkdir()
    path = phase / "artifact.bin"
    path.write_bytes(b"real")
    reference = ArtifactReference(
        path="artifact.bin",
        size_bytes=4,
        sha256="0" * 64,
        role=ArtifactRole.STABLEHLO,
    )

    with pytest.raises(ValueError, match="ARTIFACT_HASH_MISMATCH"):
        _resolve_result_artifact(tmp_path, "timing", reference)

    bad_size = reference.model_copy(
        update={"size_bytes": 999, "sha256": hashlib.sha256(b"real").hexdigest()}
    )
    with pytest.raises(ValueError, match="ARTIFACT_SIZE_MISMATCH"):
        _resolve_result_artifact(tmp_path, "timing", bad_size)


def test_seqax_pallas_safe_root_rejects_unowned_top_level_file(tmp_path: Path) -> None:
    sentinel = tmp_path / "valuable.txt"
    sentinel.write_text("keep")

    with pytest.raises(ValueError, match="UNOWNED_RUN_ROOT"):
        _require_owned_safe_root(tmp_path)

    assert sentinel.read_text() == "keep"


def test_seqax_pallas_safe_root_rejects_finalizer_symlink(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    for phase in ("timing", "trace", "counters"):
        phase_root = run_root / phase
        phase_root.mkdir()
        (phase_root / "invocation.json").write_text("{}")
        (phase_root / "result.json").write_text("{}")
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "source_state.json"
    sentinel.write_text("keep")
    (run_root / "finalizer").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="WRITE_TARGET_SYMLINK"):
        _require_owned_safe_root(run_root)

    assert sentinel.read_text() == "keep"


def test_seqax_pallas_safe_root_rejects_write_target_hardlink(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    for phase in ("timing", "trace", "counters"):
        phase_root = run_root / phase
        phase_root.mkdir()
        (phase_root / "invocation.json").write_text("{}")
        (phase_root / "result.json").write_text("{}")
    external = tmp_path / "external.json"
    external.write_text("keep")
    (run_root / "profile_assessment.json").hardlink_to(external)

    with pytest.raises(ValueError, match="WRITE_TARGET_HARDLINK"):
        _require_owned_safe_root(run_root)

    assert external.read_text() == "keep"


def test_seqax_pallas_safe_root_rejects_phase_root_symlink(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    external = tmp_path / "external-timing"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("keep")
    for marker in ("invocation.json", "result.json"):
        (external / marker).write_text("{}")
    (run_root / "timing").symlink_to(external, target_is_directory=True)
    for phase in ("trace", "counters"):
        phase_root = run_root / phase
        phase_root.mkdir()
        (phase_root / "invocation.json").write_text("{}")
        (phase_root / "result.json").write_text("{}")

    with pytest.raises(ValueError, match="PHASE_ROOT_NOT_OWNED"):
        _require_owned_safe_root(run_root)

    assert sentinel.read_text() == "keep"


def test_seqax_pallas_phase_preflight_rejects_nested_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase = tmp_path / "timing"
    phase.mkdir()
    invocation = phase / "invocation.json"
    result = phase / "result.json"
    sentinel = phase / "sentinel.txt"
    invocation.write_text("{}")
    result.write_text("{}")
    sentinel.write_text("keep")
    monkeypatch.setattr(
        "tpu_cake.seqax_pallas_bundle._result_artifacts",
        lambda *_args: {"invocation.json": invocation},
    )

    with pytest.raises(ValueError, match="PHASE_FILE_SET_MISMATCH"):
        _preflight_phase_files(tmp_path, "timing", object())  # type: ignore[arg-type]

    assert sentinel.read_text() == "keep"


def test_seqax_pallas_xprof_manifest_binds_the_raw_xplane(tmp_path: Path) -> None:
    phase = tmp_path / "trace"
    xprof = phase / "xprof"
    profile = phase / "profile"
    xprof.mkdir(parents=True)
    profile.mkdir()
    xplane = profile / "run.xplane.pb"
    xplane.write_bytes(b"xplane")
    hlo_stats = xprof / "hlo_stats.json"
    hlo_stats.write_text("{}")
    manifest = {
        "xplane": "profile/run.xplane.pb",
        "available_tools": ["hlo_stats"],
        "exports": [
            {
                "tool": "hlo_stats",
                "mime_type": "application/json",
                "output": "xprof/hlo_stats.json",
                "size_bytes": 2,
            }
        ],
    }
    (xprof / "manifest.json").write_text(json.dumps(manifest))

    _validate_xprof_exports(xprof, phase_root=phase, expected_xplane=xplane)

    manifest["xplane"] = "profile/other.xplane.pb"
    (xprof / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="XPLANE_IDENTITY_MISMATCH"):
        _validate_xprof_exports(xprof, phase_root=phase, expected_xplane=xplane)


def test_seqax_pallas_profile_assessment_canonicalizes_program_ids() -> None:
    first = {
        "timing_trace": {"capture": {"timed_program_ids": ["2", "1"]}},
        "counter_trace": {"capture": {"timed_program_ids": ["1", "2"]}},
    }
    second = {
        "timing_trace": {"capture": {"timed_program_ids": ["1", "2"]}},
        "counter_trace": {"capture": {"timed_program_ids": ["2", "1"]}},
    }

    assert _canonical_profile_assessment(first) == _canonical_profile_assessment(second)
