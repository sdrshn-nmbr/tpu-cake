from pathlib import Path

from tpu_cake.contracts import ProfileExpectation
from tpu_cake.evidence import (
    ArtifactEvidence,
    CaptureEvidence,
    CounterEvidence,
    PlaneEvidence,
    ProgramEvidence,
)
from tpu_cake.xprof_evidence import assess_evidence


def _artifact(path: str) -> ArtifactEvidence:
    return ArtifactEvidence(path=Path(path), size_bytes=1, sha256="0" * 64)


def _capture(*, rpa_markers: int, forbidden_hits: int) -> CaptureEvidence:
    return CaptureEvidence(
        xplane=_artifact("capture.xplane.pb"),
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
