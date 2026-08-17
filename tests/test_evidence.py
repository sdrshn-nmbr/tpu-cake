from pathlib import Path

from tpu_cake.contracts import ProfileExpectation
from tpu_cake.evidence import (
    ArtifactEvidence,
    CaptureEvidence,
    CounterEvidence,
    PlaneEvidence,
    ProgramEvidence,
)
from tpu_cake.xprof_evidence import _hlo_proto_paths, assess_evidence, capture_metrics


def _artifact(path: str) -> ArtifactEvidence:
    return ArtifactEvidence(path=Path(path), size_bytes=1, sha256="0" * 64)


def _capture(*, rpa_markers: int, forbidden_hits: int) -> CaptureEvidence:
    return CaptureEvidence(
        xplane=_artifact("capture.xplane.pb"),
        hlo_stats=_artifact("hlo_stats.json"),
        planes=(
            PlaneEvidence(
                name="/device:TPU:0",
                device_type="TPU v7x",
                line_count=1,
                event_count=10,
                tensor_core_event_count=10,
            ),
        ),
        counters=CounterEvidence(
            hbm_read_names=1,
            hbm_write_names=1,
            cycle_names=1,
            snapshots_per_tpu_core={"0": 2},
        ),
        programs=(
            ProgramEvidence(
                program_id="7",
                name="jit_model(7)",
                timed_self_us=10,
                hlo=_artifact("jit_model(7).hlo_proto.pb"),
                marker_counts={"ragged_paged_attention": rpa_markers},
                forbidden_fragment_hits={"prompt_gather": forbidden_hits},
            ),
        ),
        timed_program_ids=frozenset({"7"}),
    )


def test_hlo_discovery_accepts_any_program_name_with_an_xprof_program_id(
    tmp_path,
) -> None:
    expected = tmp_path / "jit_distributed(7).hlo_proto.pb"
    expected.write_bytes(b"hlo")
    (tmp_path / "unrelated.pb").write_bytes(b"not hlo")

    assert _hlo_proto_paths(tmp_path) == (expected,)


def _expectation() -> ProfileExpectation:
    return ProfileExpectation(
        name="decode",
        stage="steady_decode",
        required_timed_hlo_markers=("ragged_paged_attention",),
        forbidden_timed_hlo_fragments=("prompt_gather",),
    )


def test_matching_timed_program_is_accepted() -> None:
    assessment = assess_evidence(_capture(rpa_markers=1, forbidden_hits=0), _expectation())
    assert assessment.accepted


def test_wrong_timed_program_is_rejected() -> None:
    assessment = assess_evidence(_capture(rpa_markers=0, forbidden_hits=1), _expectation())
    assert not assessment.accepted
    assert {finding.code for finding in assessment.findings} == {
        "REQUIRED_TIMED_HLO_MARKER_MISSING",
        "FORBIDDEN_TIMED_HLO_FRAGMENT",
    }


def test_normalized_metrics_do_not_invent_counter_rates() -> None:
    metrics = capture_metrics(_capture(rpa_markers=1, forbidden_hits=0))
    assert {metric.name for metric in metrics} == {
        "tpu_device_plane_count",
        "tensor_core_event_count",
        "hbm_read_counter_name_count",
        "hbm_write_counter_name_count",
        "cycle_counter_name_count",
        "summed_timed_hlo_self_time",
    }
    assert not any("bytes_per_second" in metric.name or "mfu" in metric.name for metric in metrics)
    assert all(metric.formula is not None for metric in metrics)


def test_required_markers_must_share_one_timed_program() -> None:
    capture = _capture(rpa_markers=1, forbidden_hits=0)
    second = ProgramEvidence(
        program_id="8",
        name="jit_other(8)",
        timed_self_us=10,
        marker_counts={"other_required": 1},
        forbidden_fragment_hits={"prompt_gather": 0},
    )
    split = capture.model_copy(
        update={
            "programs": (*capture.programs, second),
            "timed_program_ids": frozenset({"7", "8"}),
        }
    )
    expectation = _expectation().model_copy(
        update={"required_timed_hlo_markers": ("ragged_paged_attention", "other_required")}
    )
    assessment = assess_evidence(split, expectation)
    assert not assessment.accepted
    assert "REQUIRED_MARKERS_SPLIT_ACROSS_PROGRAMS" in {
        finding.code for finding in assessment.findings
    }


def test_counter_contract_rejects_missing_device_planes() -> None:
    expectation = _expectation().model_copy(update={"minimum_counter_device_planes": 2})
    assessment = assess_evidence(_capture(rpa_markers=1, forbidden_hits=0), expectation)
    assert not assessment.accepted
    assert "INSUFFICIENT_COUNTER_DEVICE_PLANES" in {
        finding.code for finding in assessment.findings
    }
