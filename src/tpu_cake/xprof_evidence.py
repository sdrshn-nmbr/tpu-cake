from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from xprof import profile_data

from tpu_cake.contracts import ProfileExpectation
from tpu_cake.evidence import (
    ArtifactEvidence,
    CaptureAssessment,
    CaptureEvidence,
    CounterEvidence,
    Finding,
    FindingSeverity,
    PlaneEvidence,
    ProgramEvidence,
)
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)

_PROGRAM_ID = re.compile(r"\((\d+)\)\.hlo_proto\.pb$")
_TPU_CORE = re.compile(r"^/device:TPU:(\d+)$")
_MARKERS = (
    "distributed_matmul_physical",
    "reduce-scatter",
    "ragged_paged_attention",
    "pallas_call",
    "EPMoE",
    "gmm_v2",
    "ragged_causal_conv1d",
    "NativeAttention",
)


def _artifact(path: Path) -> ArtifactEvidence:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return ArtifactEvidence(path=path, size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"ARTIFACT_LOOKUP_FAILED pattern={pattern!r} matches={matches}")
    return matches[0]


def _hlo_proto_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.hlo_proto.pb")))


def _gviz_rows(path: Path) -> list[dict[str, Any]]:
    table = json.loads(path.read_bytes())
    columns = [column["id"] for column in table["cols"]]
    rows = []
    for row in table["rows"]:
        values = [cell.get("v") for cell in row["c"]]
        if len(values) != len(columns):
            raise ValueError(f"XPROF_TABLE_WIDTH_MISMATCH path={path}")
        rows.append(dict(zip(columns, values, strict=True)))
    return rows


def _profile_planes(xplane: Path) -> tuple[tuple[PlaneEvidence, ...], CounterEvidence]:
    profile = profile_data.ProfileData.from_file(xplane)
    planes: list[PlaneEvidence] = []
    hbm_read_names: set[str] = set()
    hbm_write_names: set[str] = set()
    cycle_names: set[str] = set()
    periodic_counter_names: set[str] = set()
    periodic_samples: dict[str, set[float]] = defaultdict(set)
    try:
        for plane in profile.planes:
            plane_stats = dict(plane.stats)
            tensor_core_events = 0
            event_count = 0
            core_match = _TPU_CORE.fullmatch(plane.name)
            for line in plane.lines:
                event_count += len(line.events)
                if line.name == "Tensor Core":
                    tensor_core_events += len(line.events)
                if core_match and line.name == "_counters_":
                    sample_points = {event.start_ns for event in line.events}
                    if len(sample_points) >= 2:
                        periodic_samples[core_match.group(1)].update(sample_points)
                        periodic_counter_names.update(event.name for event in line.events)
                if core_match and line.name.startswith("counters_"):
                    for event in line.events:
                        upper_name = event.name.upper()
                        if "RD_RSP_BEAT_FROM_HBM" in upper_name:
                            hbm_read_names.add(event.name)
                        if "WR_REQ_BEAT_TO_HBM" in upper_name:
                            hbm_write_names.add(event.name)
                        if "CYCLE_COUNT_WINDOW" in upper_name:
                            cycle_names.add(event.name)
            planes.append(
                PlaneEvidence(
                    name=plane.name,
                    device_type=plane_stats.get("device_type_string"),
                    line_count=len(plane.lines),
                    event_count=event_count,
                    tensor_core_event_count=tensor_core_events,
                )
            )
    finally:
        profile.close()
    return (
        tuple(planes),
        CounterEvidence(
            hbm_read_names=len(hbm_read_names),
            hbm_write_names=len(hbm_write_names),
            cycle_names=len(cycle_names),
            periodic_counter_names=tuple(sorted(periodic_counter_names)),
            periodic_samples_per_tpu_core={
                core: len(points) for core, points in sorted(periodic_samples.items())
            },
        ),
    )


def collect_capture(root: Path, expectation: ProfileExpectation) -> CaptureEvidence:
    root = root.resolve()
    xplane = _single(root, "*.xplane.pb")
    hlo_stats = _single(root, "hlo_stats.json")
    rows = _gviz_rows(hlo_stats)
    markers = tuple(dict.fromkeys((*_MARKERS, *expectation.required_timed_hlo_markers)))
    timed_us: dict[str, float] = defaultdict(float)
    timed_text: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        program_id = str(row["program_id"])
        timed_us[program_id] += float(row.get("total_self_time") or 0)
        timed_text[program_id].extend(
            str(row.get(field) or "")
            for field in ("hlo_op_name", "hlo_op_expression", "source_info", "tf_op_name")
        )

    programs = []
    captured_program_ids: set[str] = set()
    for hlo_path in _hlo_proto_paths(root):
        match = _PROGRAM_ID.search(hlo_path.name)
        if match is None:
            continue
        program_id = match.group(1)
        payload = hlo_path.read_bytes()
        timed_payload = "\n".join(timed_text.get(program_id, ()))
        programs.append(
            ProgramEvidence(
                program_id=program_id,
                name=hlo_path.name.removesuffix(".hlo_proto.pb"),
                timed_self_us=timed_us.get(program_id, 0),
                hlo=_artifact(hlo_path),
                marker_counts={marker: payload.count(marker.encode()) for marker in markers},
                forbidden_fragment_hits={
                    fragment: timed_payload.count(fragment)
                    for fragment in expectation.forbidden_timed_hlo_fragments
                },
            )
        )
        captured_program_ids.add(program_id)

    for program_id in sorted(set(timed_us) - captured_program_ids):
        payload = "\n".join(timed_text.get(program_id, ()))
        programs.append(
            ProgramEvidence(
                program_id=program_id,
                name=f"timed_program_{program_id}",
                timed_self_us=timed_us[program_id],
                marker_counts={marker: payload.count(marker) for marker in markers},
                forbidden_fragment_hits={
                    fragment: payload.count(fragment)
                    for fragment in expectation.forbidden_timed_hlo_fragments
                },
            )
        )

    planes, counters = _profile_planes(xplane)
    return CaptureEvidence(
        xplane=_artifact(xplane),
        hlo_stats=_artifact(hlo_stats),
        planes=planes,
        counters=counters,
        programs=tuple(programs),
        timed_program_ids=frozenset(timed_us),
    )


def assess_evidence(capture: CaptureEvidence, expectation: ProfileExpectation) -> CaptureAssessment:
    findings: list[Finding] = []
    tpu_planes = [plane for plane in capture.planes if _TPU_CORE.fullmatch(plane.name)]
    tensor_core_events = sum(plane.tensor_core_event_count for plane in tpu_planes)
    tpu_device_events = sum(plane.event_count for plane in tpu_planes)

    if len(tpu_planes) < expectation.minimum_tpu_device_planes:
        findings.append(
            Finding(
                code="INSUFFICIENT_TPU_DEVICE_PLANES",
                severity=FindingSeverity.ERROR,
                message="capture does not contain the required physical TPU core planes",
                evidence=(
                    f"observed={len(tpu_planes)}",
                    f"required={expectation.minimum_tpu_device_planes}",
                ),
            )
        )
    if expectation.require_tensor_core_activity and tensor_core_events == 0:
        findings.append(
            Finding(
                code="NO_TENSOR_CORE_ACTIVITY",
                severity=FindingSeverity.ERROR,
                message="capture has no Tensor Core events",
            )
        )
    if tpu_device_events < 1:
        findings.append(
            Finding(
                code="INSUFFICIENT_TPU_DEVICE_EVENTS",
                severity=FindingSeverity.ERROR,
                message="capture does not contain enough physical TPU device activity",
                evidence=(
                    f"observed={tpu_device_events}",
                    "required=1",
                ),
            )
        )

    counter_requirements = (
        (expectation.require_hbm_read_counters, capture.counters.hbm_read_names, "HBM_READ"),
        (expectation.require_hbm_write_counters, capture.counters.hbm_write_names, "HBM_WRITE"),
        (expectation.require_cycle_counters, capture.counters.cycle_names, "CYCLE"),
    )
    for required, observed, label in counter_requirements:
        if required and observed == 0:
            findings.append(
                Finding(
                    code=f"MISSING_{label}_COUNTERS",
                    severity=FindingSeverity.ERROR,
                    message=f"capture has no {label.lower()} hardware counters",
                )
            )

    counter_planes = len(capture.counters.periodic_samples_per_tpu_core)
    if counter_planes < expectation.minimum_counter_device_planes:
        findings.append(
            Finding(
                code="INSUFFICIENT_COUNTER_DEVICE_PLANES",
                severity=FindingSeverity.ERROR,
                message="capture does not contain counters for the required number of device planes",
                evidence=(
                    f"observed={counter_planes}",
                    f"required={expectation.minimum_counter_device_planes}",
                ),
            )
        )

    timed_programs = [
        program for program in capture.programs if program.program_id in capture.timed_program_ids
    ]
    if not timed_programs:
        findings.append(
            Finding(
                code="NO_TIMED_MODEL_PROGRAM",
                severity=FindingSeverity.ERROR,
                message="XProf timed rows do not map to a captured model HLO program",
            )
        )

    for marker in expectation.required_timed_hlo_markers:
        matching = [
            program.program_id
            for program in timed_programs
            if program.marker_counts.get(marker, 0) > 0
        ]
        if not matching:
            findings.append(
                Finding(
                    code="REQUIRED_TIMED_HLO_MARKER_MISSING",
                    severity=FindingSeverity.ERROR,
                    message=f"required marker {marker!r} is absent from every timed model program",
                )
            )

    all_required_markers_present = all(
        any(program.marker_counts.get(marker, 0) > 0 for program in timed_programs)
        for marker in expectation.required_timed_hlo_markers
    )
    if expectation.required_timed_hlo_markers and all_required_markers_present:
        bound_programs = [
            program.program_id
            for program in timed_programs
            if all(
                program.marker_counts.get(marker, 0) > 0
                for marker in expectation.required_timed_hlo_markers
            )
        ]
        if not bound_programs:
            findings.append(
                Finding(
                    code="REQUIRED_MARKERS_SPLIT_ACROSS_PROGRAMS",
                    severity=FindingSeverity.ERROR,
                    message="no single timed program contains every required marker",
                )
            )

    for program in timed_programs:
        for fragment, count in program.forbidden_fragment_hits.items():
            if count:
                findings.append(
                    Finding(
                        code="FORBIDDEN_TIMED_HLO_FRAGMENT",
                        severity=FindingSeverity.ERROR,
                        message=f"timed program contains forbidden HLO fragment {fragment!r}",
                        evidence=(f"program_id={program.program_id}", f"hits={count}"),
                    )
                )

    counters_requested = any(
        requirement
        for requirement in (
            expectation.require_hbm_read_counters,
            expectation.require_hbm_write_counters,
            expectation.require_cycle_counters,
        )
    )
    if counters_requested and not capture.counters.rates_derivable:
        findings.append(
            Finding(
                code="COUNTER_RATES_NOT_DERIVABLE",
                severity=FindingSeverity.ERROR,
                message="hardware counters exist but have fewer than two snapshots per TPU core",
            )
        )

    return CaptureAssessment(expectation=expectation, capture=capture, findings=tuple(findings))


def assess_capture(root: Path, expectation: ProfileExpectation) -> CaptureAssessment:
    return assess_evidence(collect_capture(root, expectation), expectation)


def capture_metrics(capture: CaptureEvidence) -> tuple[Metric, ...]:
    xplane_source = MetricSource(
        artifact_sha256=capture.xplane.sha256,
        artifact_path=str(capture.xplane.path),
        tool="XPlane",
        field="planes",
    )
    hlo_source = MetricSource(
        artifact_sha256=capture.hlo_stats.sha256,
        artifact_path=str(capture.hlo_stats.path),
        tool="XProf",
        field="total_self_time",
    )
    interval = MeasurementInterval(scope="complete captured interval")
    tpu_planes = sum(bool(_TPU_CORE.fullmatch(plane.name)) for plane in capture.planes)
    tensor_core_events = sum(
        plane.tensor_core_event_count for plane in capture.planes if _TPU_CORE.fullmatch(plane.name)
    )
    timed_self_us = sum(
        program.timed_self_us
        for program in capture.programs
        if program.program_id in capture.timed_program_ids
    )

    def count_metric(name: str, value: int, field: str, expression: str) -> Metric:
        return Metric(
            name=name,
            quantity=Quantity(value=Decimal(value), unit=Unit.COUNT),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(xplane_source.model_copy(update={"field": field}),),
            formula=FormulaIdentity(
                name=name,
                version="1",
                expression=expression,
            ),
        )

    return (
        count_metric(
            "tpu_device_plane_count",
            tpu_planes,
            "planes",
            "count(plane where name matches /device:TPU:<core>)",
        ),
        count_metric(
            "tensor_core_event_count",
            tensor_core_events,
            "Tensor Core line events",
            "sum(event count for Tensor Core lines on TPU device planes)",
        ),
        count_metric(
            "hbm_read_counter_name_count",
            capture.counters.hbm_read_names,
            "HBM read counter names",
            "count(distinct counter names containing RD_RSP_BEAT_FROM_HBM)",
        ),
        count_metric(
            "hbm_write_counter_name_count",
            capture.counters.hbm_write_names,
            "HBM write counter names",
            "count(distinct counter names containing WR_REQ_BEAT_TO_HBM)",
        ),
        count_metric(
            "cycle_counter_name_count",
            capture.counters.cycle_names,
            "cycle counter names",
            "count(distinct counter names containing CYCLE_COUNT_WINDOW)",
        ),
        count_metric(
            "periodic_counter_name_count",
            len(capture.counters.periodic_counter_names),
            "periodic counter names",
            "count(distinct names sampled on periodic TPU counter lines)",
        ),
        count_metric(
            "minimum_periodic_samples_per_tpu_core",
            min(capture.counters.periodic_samples_per_tpu_core.values(), default=0),
            "periodic sample timestamps",
            "min(count(distinct periodic timestamps) per TPU core)",
        ),
        Metric(
            name="summed_timed_hlo_self_time",
            quantity=Quantity(value=Decimal(str(timed_self_us)), unit=Unit.MICROSECOND),
            kind=MeasurementKind.DERIVED,
            interval=MeasurementInterval(scope="sum of timed HLO rows across device rows"),
            sources=(hlo_source,),
            formula=FormulaIdentity(
                name="sum_timed_hlo_self_time",
                version="1",
                expression="sum(program.timed_self_us for timed programs)",
            ),
        ),
    )
