from __future__ import annotations

import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

from tpu_cake.cost_model import CostModelReport
from tpu_cake.evidence import CaptureAssessment
from tpu_cake.metrics import (
    FormulaIdentity,
    MeasurementInterval,
    MeasurementKind,
    Metric,
    MetricSource,
    Quantity,
    Unit,
)
from tpu_cake.runner import MatmulRunResult
from tpu_cake.xprof_evidence import capture_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relocate_metric_source(root: Path, source: MetricSource) -> MetricSource:
    declared = Path(source.artifact_path)
    direct = declared if declared.is_absolute() else root / declared
    if direct.is_file() and _sha256(direct) == source.artifact_sha256:
        path = direct.resolve().relative_to(root.resolve())
        return source.model_copy(update={"artifact_path": str(path)})
    matches = [
        candidate
        for candidate in root.rglob(declared.name)
        if candidate.is_file() and _sha256(candidate) == source.artifact_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            "METRIC_SOURCE_CANNOT_BE_RELOCATED "
            f"path={source.artifact_path} sha256={source.artifact_sha256}"
        )
    return source.model_copy(
        update={"artifact_path": str(matches[0].resolve().relative_to(root.resolve()))}
    )


def timing_metrics(root: Path, result: MatmulRunResult) -> tuple[Metric, ...]:
    if not result.samples_ns:
        raise ValueError("TIMING_RUN_HAS_NO_SAMPLES")
    timing_path = root / "timing" / "result.json"
    source = MetricSource(
        artifact_sha256=_sha256(timing_path),
        artifact_path=str(timing_path.relative_to(root)),
        tool="tpu-cake",
        field="samples_ns",
    )
    interval = MeasurementInterval(
        scope=f"{len(result.samples_ns)} warm synchronized distributed matmul executions"
    )
    ordered = sorted(result.samples_ns)
    p90_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))
    definitions = (
        (
            "median_device_duration",
            Decimal(statistics.median(result.samples_ns)),
            "median",
            "median(samples_ns)",
        ),
        (
            "p90_device_duration",
            Decimal(ordered[p90_index]),
            "nearest_rank_p90",
            "sort(samples_ns)[round((N-1)*0.9)]",
        ),
    )
    return tuple(
        Metric(
            name=name,
            quantity=Quantity(value=value, unit=Unit.NANOSECOND),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(name=formula, version="1", expression=expression),
        )
        for name, value, formula, expression in definitions
    )


def roofline_metrics(root: Path) -> tuple[Metric, ...]:
    path = root / "roofline" / "metrics.json"
    payload = json.loads(path.read_text())
    source = MetricSource(
        artifact_sha256=_sha256(path),
        artifact_path=str(path.relative_to(root)),
        tool="roofline skill",
        field="kernel roofline metrics",
    )
    interval = MeasurementInterval(scope="one synchronized distributed matmul execution")
    lower_bound = Decimal(str(payload["lower_bound_time_s"]))
    measured = Decimal(str(payload["measured_time_s"]))
    return (
        Metric(
            name="roofline_two_resource_time_floor",
            quantity=Quantity(value=lower_bound, unit=Unit.SECOND),
            kind=MeasurementKind.ESTIMATED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="roofline_two_resource_floor",
                version="1",
                expression="max(operations/compute_peak,total_bytes/hbm_bandwidth)",
            ),
        ),
        Metric(
            name="measured_to_two_resource_floor",
            quantity=Quantity(value=measured / lower_bound, unit=Unit.RATIO),
            kind=MeasurementKind.DERIVED,
            interval=interval,
            sources=(source,),
            formula=FormulaIdentity(
                name="measured_to_two_resource_floor",
                version="1",
                expression="measured_time/two_resource_time_floor",
            ),
            numerator=Quantity(value=measured, unit=Unit.SECOND),
            denominator=Quantity(value=lower_bound, unit=Unit.SECOND),
        ),
    )


def three_resource_gap_metric(
    root: Path,
    timing: MatmulRunResult,
    cost_report: CostModelReport,
) -> Metric:
    if timing.median_ns is None:
        raise ValueError("TIMING_RUN_HAS_NO_MEDIAN")
    floor_metric = next(
        metric for metric in cost_report.metrics if metric.name == "idealized_time_floor"
    )
    measured = Decimal(timing.median_ns)
    floor = floor_metric.quantity.value
    timing_path = root / "timing" / "result.json"
    return Metric(
        name="measured_to_compute_hbm_ici_floor",
        quantity=Quantity(value=measured / floor, unit=Unit.RATIO),
        kind=MeasurementKind.DERIVED,
        interval=MeasurementInterval(scope="one synchronized distributed matmul execution"),
        sources=(
            MetricSource(
                artifact_sha256=_sha256(timing_path),
                artifact_path=str(timing_path.relative_to(root)),
                tool="tpu-cake",
                field="median_ns",
            ),
            *floor_metric.sources,
        ),
        formula=FormulaIdentity(
            name="measured_to_compute_hbm_ici_floor",
            version="1",
            expression="measured_median_ns/max(compute_floor,hbm_floor,ici_floor)",
        ),
        numerator=Quantity(value=measured, unit=Unit.NANOSECOND),
        denominator=Quantity(value=floor, unit=Unit.NANOSECOND),
    )


def build_receipt_metrics(
    root: Path,
    timing: MatmulRunResult,
    cost_report: CostModelReport,
    trace_assessment: CaptureAssessment,
    counter_assessment: CaptureAssessment,
) -> tuple[Metric, ...]:
    metrics = list(timing_metrics(root, timing))
    metrics.extend(roofline_metrics(root))
    metrics.extend(cost_report.metrics)
    metrics.append(three_resource_gap_metric(root, timing, cost_report))
    metrics.extend(
        metric.model_copy(update={"name": f"timing_trace.{metric.name}"})
        for metric in capture_metrics(trace_assessment.capture)
    )
    metrics.extend(
        metric.model_copy(update={"name": f"counter_trace.{metric.name}"})
        for metric in capture_metrics(counter_assessment.capture)
    )
    normalized: list[Metric] = []
    for metric in metrics:
        sources = [_relocate_metric_source(root, source) for source in metric.sources]
        normalized.append(metric.model_copy(update={"sources": tuple(sources)}))
    return tuple(normalized)
