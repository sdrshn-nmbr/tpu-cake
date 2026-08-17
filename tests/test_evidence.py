import hashlib
from pathlib import Path

from tpu_cake.contracts import ProfileExpectation
from tpu_cake.evidence import (
    ArtifactEvidence,
    CaptureEvidence,
    CounterEvidence,
    PlaneEvidence,
    ProgramEvidence,
)
from tpu_cake.metrics import MetricSource
from tpu_cake.receipt_metrics import _relocate_metric_source
from tpu_cake.xprof_evidence import _hlo_proto_paths, assess_evidence, capture_metrics


def _artifact(path: str) -> ArtifactEvidence:
    return ArtifactEvidence(path=Path(path), size_bytes=1, sha256="0" * 64)


def test_metric_source_relocation_uses_content_identity(tmp_path) -> None:
    artifact = tmp_path / "timing" / "cost_model_input.json"
    artifact.parent.mkdir()
    artifact.write_text("evidence\n")
    duplicate = tmp_path / "counters" / artifact.name
    duplicate.parent.mkdir()
    duplicate.write_text("evidence\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    source = MetricSource(
        artifact_sha256=digest,
        artifact_path=f"runs/{tmp_path.name}/timing/cost_model_input.json",
        tool="tpu-cake",
        field="cost",
    )

    relocated = _relocate_metric_source(tmp_path, source)
    assert relocated.artifact_path == "timing/cost_model_input.json"


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
            periodic_counter_names=("COUNT_MXU_BUSY_0",),
            periodic_samples_per_tpu_core={"0": 2},
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
        "periodic_counter_name_count",
        "minimum_periodic_samples_per_tpu_core",
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


def test_profile_contract_rejects_empty_physical_device_planes() -> None:
    capture = _capture(rpa_markers=1, forbidden_hits=0)
    empty = capture.model_copy(
        update={
            "planes": (
                capture.planes[0].model_copy(
                    update={"event_count": 0, "tensor_core_event_count": 0}
                ),
            )
        }
    )
    assessment = assess_evidence(empty, _expectation())
    assert not assessment.accepted
    assert "INSUFFICIENT_TPU_DEVICE_EVENTS" in {
        finding.code for finding in assessment.findings
    }


def test_periodic_counter_series_are_fail_closed_when_requested() -> None:
    capture = _capture(rpa_markers=1, forbidden_hits=0)
    one_snapshot = capture.model_copy(
        update={
            "counters": capture.counters.model_copy(
                update={"periodic_samples_per_tpu_core": {"0": 1}}
            )
        }
    )
    expectation = _expectation().model_copy(
        update={
            "require_hbm_read_counters": True,
            "require_hbm_write_counters": True,
            "require_cycle_counters": True,
        }
    )
    assessment = assess_evidence(one_snapshot, expectation)
    assert not assessment.accepted
    assert "PERIODIC_COUNTER_SERIES_NOT_DERIVABLE" in {
        finding.code for finding in assessment.findings
    }
