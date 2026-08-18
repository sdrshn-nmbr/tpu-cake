from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tpu_cake.cli import _parser
from tpu_cake.contracts import ArtifactRole
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import MxuEinsumOp
from tpu_cake.frontend import schedule_sha256
from tpu_cake.metrics import MetricSource
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_cost_model import estimate_seqax_forward
from tpu_cake.seqax_pallas_diagnostic import (
    SEQAX_PALLAS_CANONICAL_SEARCH_RECEIPT_SHA256,
    SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS,
    SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
    SEQAX_PALLAS_DIAGNOSTIC_WARMUPS,
    SeqaxPallasDiagnosticContract,
    SeqaxPallasHloCategoryAttribution,
    _artifact_manifest,
    _artifact_role,
    _attribution,
    _canonical_assessment,
    _export_xprof,
    _validate_counter_evidence,
    _validate_manifest,
    _validate_xprof,
    _validate_xprof_replay,
)
from tpu_cake.seqax_pallas_runner import _physical_collective_counts
from tpu_cake.seqax_pallas_search import default_seqax_pallas_search_contract
from tpu_cake.seqax_pallas_search_runner import prepare_seqax_pallas_candidates
from tpu_cake.xprof_export import XProfExport, XProfExportManifest


def _contract() -> SeqaxPallasDiagnosticContract:
    search = default_seqax_pallas_search_contract(_runtime_identity())
    return SeqaxPallasDiagnosticContract(
        diagnostic_schema=SEQAX_PALLAS_DIAGNOSTIC_SCHEMA,
        search_id=search.search_id,
        search_receipt_sha256=SEQAX_PALLAS_CANONICAL_SEARCH_RECEIPT_SHA256,
        candidate="incumbent",
        timing_seed=search.timing_seed,
        warmup_iterations=SEQAX_PALLAS_DIAGNOSTIC_WARMUPS,
        measured_iterations=SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS,
        runtime=search.runtime,
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
    )


def _hlo_stats(path: Path, occurrences: int) -> tuple[object, object]:
    search = default_seqax_pallas_search_contract(_runtime_identity())
    distributed, candidates = prepare_seqax_pallas_candidates(search)
    physical = candidates[0].physical
    operations = tuple(
        operation for operation in physical.walk() if isinstance(operation, MxuEinsumOp)
    )
    columns = (
        "program_id",
        "category",
        "hlo_op_name",
        "hlo_op_expression",
        "tf_op_name",
        "occurrences",
        "avg_self_time",
    )
    rows = []
    for index in range(len(operations)):
        values = (
            "42",
            "custom-call",
            f"seqax_named_einsum.{index}",
            'frontend_attributes={kernel_metadata={"region_index":' + str(index) + "}}",
            "jit(physical_call)/seqax_named_einsum/pallas_call:",
            occurrences,
            0.25 + index / 100,
        )
        values = list(values)
        values[3] = (
            'frontend_attributes={kernel_metadata={"region_index":'
            + str(index)
            + ',"schedule_sha256":"'
            + schedule_sha256(physical)
            + '","tile_m":'
            + str(operations[index].tile_m.data)
            + ',"tile_k":'
            + str(operations[index].tile_k.data)
            + ',"tile_n":'
            + str(operations[index].tile_n.data)
            + "}}"
        )
        rows.append({"c": [{"v": value} for value in values]})
    all_gathers, reduce_scatters = _physical_collective_counts(physical)
    for category, count in (
        ("all-gather", all_gathers),
        ("reduce-scatter", reduce_scatters),
    ):
        for index in range(count):
            rows.append(
                {
                    "c": [
                        {"v": "42"},
                        {"v": category},
                        {"v": f"{category}.{index}"},
                        {"v": category},
                        {"v": "jit(physical_call)"},
                        {"v": 50},
                        {"v": 2.0},
                    ]
                }
            )
            rows.append(
                {
                    "c": [
                        {"v": "42"},
                        {"v": "async-done"},
                        {"v": f"{category}.{index}.call-done"},
                        {"v": category},
                        {"v": "jit(physical_call)"},
                        {"v": 400},
                        {"v": 2.0},
                    ]
                }
            )
    path.write_text(
        json.dumps(
            {
                "cols": [{"id": value} for value in columns],
                "rows": rows,
            }
        )
    )
    report = estimate_seqax_forward(
        distributed,
        hardware=tpu7x_tensorcore_rates(),
        source=MetricSource(
            artifact_sha256=candidates[0].plan.distributed_schedule_sha256,
            artifact_path="search/distributed.xdsl",
            tool="test",
            field="program",
        ),
        expected_schedule_sha256=candidates[0].plan.distributed_schedule_sha256,
    )
    return physical, report


def test_diagnostic_contract_rejects_protocol_drift() -> None:
    with pytest.raises(ValueError, match="PROTOCOL_MISMATCH"):
        _contract().model_copy(update={"measured_iterations": 49}).model_validate(
            _contract().model_dump() | {"measured_iterations": 49}
        )


def test_region_attribution_binds_every_physical_pallas_region(tmp_path: Path) -> None:
    hlo_stats = tmp_path / "hlo_stats.json"
    physical, report = _hlo_stats(hlo_stats, SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS * 8)
    attribution = _attribution(
        physical=physical,
        program_id="42",
        durations=tuple(float(10_000 + index) for index in range(50)),
        hlo_stats=hlo_stats,
        cost_report=report,
    )
    assert tuple(value.region_index for value in attribution.regions) == tuple(range(9))
    assert (
        attribution.collective_completion_average_self_time_sum_ns_per_device
        == sum(_physical_collective_counts(physical)) * 2_000
    )
    assert attribution.pallas_average_self_time_sum_ns_per_device > 0
    assert attribution.module_to_idealized_floor_ratio > 0


def test_region_attribution_rejects_wrong_occurrence_count(tmp_path: Path) -> None:
    hlo_stats = tmp_path / "hlo_stats.json"
    physical, report = _hlo_stats(hlo_stats, 399)
    with pytest.raises(ValueError, match="REGION_OCCURRENCE_MISMATCH"):
        _attribution(
            physical=physical,
            program_id="42",
            durations=tuple(float(10_000 + index) for index in range(50)),
            hlo_stats=hlo_stats,
            cost_report=report,
        )


def test_region_attribution_rejects_unbound_region_metadata(tmp_path: Path) -> None:
    hlo_stats = tmp_path / "hlo_stats.json"
    physical, report = _hlo_stats(hlo_stats, SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS * 8)
    payload = json.loads(hlo_stats.read_text())
    payload["rows"][0]["c"][1]["v"] = "fake-category"
    hlo_stats.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="REGION_IDENTITY_MISMATCH"):
        _attribution(
            physical=physical,
            program_id="42",
            durations=tuple(float(10_000 + index) for index in range(50)),
            hlo_stats=hlo_stats,
            cost_report=report,
        )


def test_region_attribution_requires_collective_completion_rows(tmp_path: Path) -> None:
    hlo_stats = tmp_path / "hlo_stats.json"
    physical, report = _hlo_stats(hlo_stats, SEQAX_PALLAS_DIAGNOSTIC_ITERATIONS * 8)
    payload = json.loads(hlo_stats.read_text())
    removed = False
    retained = []
    for row in payload["rows"]:
        if not removed and row["c"][1]["v"] == "async-done":
            removed = True
            continue
        retained.append(row)
    payload["rows"] = retained
    hlo_stats.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="COLLECTIVE_COMPLETION_ROWS_MISMATCH"):
        _attribution(
            physical=physical,
            program_id="42",
            durations=tuple(float(10_000 + index) for index in range(50)),
            hlo_stats=hlo_stats,
            cost_report=report,
        )


def test_attribution_models_reject_nonfinite_values() -> None:
    with pytest.raises(ValidationError, match="CATEGORY_NONFINITE"):
        SeqaxPallasHloCategoryAttribution(
            category="custom-call",
            row_count=1,
            average_self_time_sum_ns=float("inf"),
        )


def test_counter_evidence_requires_mxu_and_all_counter_families() -> None:
    valid = SimpleNamespace(
        capture=SimpleNamespace(
            counters=SimpleNamespace(
                periodic_samples_per_tpu_core={"0": 2, "2": 2, "4": 2, "6": 2},
                periodic_counter_names=("COUNT_MXU_BUSY_0",),
                hbm_read_names=1,
                hbm_write_names=1,
                cycle_names=1,
            )
        )
    )
    _validate_counter_evidence(valid)
    invalid = SimpleNamespace(
        capture=SimpleNamespace(
            counters=SimpleNamespace(
                periodic_samples_per_tpu_core={"0": 2, "2": 2, "4": 2, "6": 2},
                periodic_counter_names=("COUNT_OTHER",),
                hbm_read_names=1,
                hbm_write_names=1,
                cycle_names=1,
            )
        )
    )
    with pytest.raises(ValueError, match="COUNTER_EVIDENCE_MISMATCH"):
        _validate_counter_evidence(invalid)
    invalid.capture.counters.periodic_counter_names = ("COUNT_MXU_BUSY_0",)
    invalid.capture.counters.periodic_samples_per_tpu_core = {
        "0": 1,
        "2": 2,
        "4": 2,
        "6": 2,
    }
    with pytest.raises(ValueError, match="COUNTER_EVIDENCE_MISMATCH"):
        _validate_counter_evidence(invalid)


def test_assessment_projection_ignores_only_nondeterministic_hlo_bytes() -> None:
    first = {
        "capture": {
            "timed_program_ids": ["2", "1"],
            "counters": {"periodic_counter_names": ["COUNT_MXU_BUSY_0"]},
            "programs": [
                {
                    "program_id": "1",
                    "timed_self_us": 2.5,
                    "marker_counts": {"pallas_call": 9},
                    "hlo": {
                        "path": "/saved/jit_main(1).hlo_proto.pb",
                        "size_bytes": 123,
                        "sha256": "a" * 64,
                    },
                }
            ],
        }
    }
    replayed = json.loads(json.dumps(first))
    replayed["capture"]["timed_program_ids"] = ["1", "2"]
    replayed["capture"]["programs"][0]["hlo"]["path"] = "/replayed/jit_main(1).hlo_proto.pb"
    replayed["capture"]["programs"][0]["hlo"]["sha256"] = "b" * 64
    assert _canonical_assessment(first) == _canonical_assessment(replayed)

    replayed["capture"]["programs"][0]["marker_counts"]["pallas_call"] = 8
    assert _canonical_assessment(first) != _canonical_assessment(replayed)


def test_manifest_includes_nested_search_receipt_but_not_its_own(tmp_path: Path) -> None:
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "receipt.json").write_text("{}\n")
    (tmp_path / "result.json").write_text("{}\n")
    artifacts = _artifact_manifest(tmp_path)
    assert {value.path for value in artifacts} == {"result.json", "search/receipt.json"}
    (tmp_path / "receipt.json").write_text("{}\n")
    _validate_manifest(tmp_path, artifacts)


def test_manifest_rejects_unclaimed_artifact(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text("{}\n")
    artifacts = _artifact_manifest(tmp_path)
    (tmp_path / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="CLOSED_WORLD_MISMATCH"):
        _validate_manifest(tmp_path, artifacts)


def test_artifact_roles_distinguish_trace_and_counter_captures() -> None:
    assert _artifact_role(Path("trace/profile/run.xplane.pb")) is ArtifactRole.TIMING_TRACE
    assert _artifact_role(Path("counters/profile/run.xplane.pb")) is ArtifactRole.COUNTER_TRACE
    assert _artifact_role(Path("trace/profile/run.trace.json.gz")) is ArtifactRole.PROFILE_AUXILIARY
    with pytest.raises(ValueError, match="ARTIFACT_UNRECOGNIZED"):
        _artifact_role(Path("trace/profile/evil.bin"))


def test_xprof_export_isolated_and_closed_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = tmp_path / "trace"
    profile = phase / "profile"
    profile.mkdir(parents=True)
    xplane = profile / "run.xplane.pb"
    xplane.write_bytes(b"xplane")
    trace_json = profile / "run.trace.json.gz"
    trace_json.write_bytes(b"trace")

    def fake_export(capture_root: Path, output_root: Path) -> XProfExportManifest:
        staged = capture_root / "run.xplane.pb"
        assert staged.read_bytes() == b"xplane"
        (capture_root / "jit_main(1).hlo_proto.pb").write_bytes(b"hlo")
        (capture_root / "ALL_HOSTS.op_stats_v2.pb").write_bytes(b"stats")
        hlo_stats = output_root / "hlo_stats.json"
        hlo_stats.write_text("{}")
        return XProfExportManifest(
            xplane=staged,
            available_tools=("hlo_stats",),
            exports=(
                XProfExport(
                    tool="hlo_stats",
                    mime_type="application/json",
                    output=hlo_stats,
                    size_bytes=hlo_stats.stat().st_size,
                ),
            ),
        )

    monkeypatch.setattr("tpu_cake.seqax_pallas_diagnostic.export_xprof_capture", fake_export)
    _export_xprof(profile, phase / "xprof")
    _validate_xprof(phase, xplane)

    assert {path.name for path in profile.iterdir()} == {
        "run.trace.json.gz",
        "run.xplane.pb",
    }
    assert not (phase / "xprof/.xprof-input").exists()
    assert (phase / "xprof/derived/jit_main(1).hlo_proto.pb").read_bytes() == b"hlo"

    replay = tmp_path / "replay"
    (replay / "xprof").mkdir(parents=True)
    for source in (phase / "xprof").rglob("*"):
        if source.is_file():
            destination = replay / "xprof" / source.relative_to(phase / "xprof")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    _validate_xprof_replay(phase, replay)
    (replay / "xprof/hlo_stats.json").write_text('{"forged": true}')
    with pytest.raises(ValueError, match="XPROF_REPLAY_MISMATCH"):
        _validate_xprof_replay(phase, replay)


def test_diagnostic_commands_are_public() -> None:
    parser = _parser()
    run = parser.parse_args(
        [
            "diagnose-seqax-physical-pallas",
            "--search-root",
            "search",
            "--contract",
            "contract.json",
            "--output-dir",
            "run",
        ]
    )
    assert run.search_root == Path("search")
    assert run.output_dir == Path("run")
    assert (
        parser.parse_args(["verify-seqax-physical-pallas-diagnostic", "run"]).command
        == "verify-seqax-physical-pallas-diagnostic"
    )
