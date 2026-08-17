from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.cli import _parser
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.runner import RunMode
from tpu_cake.seqax_surface import seqax_forward_workload_surface
from tpu_cake.seqax_surface_profile import (
    SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS,
    SEQAX_SURFACE_PROFILE_SCHEMA,
    SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS,
    SeqaxSurfaceProfileInvocation,
    SeqaxSurfaceProfileReceipt,
    _expectation,
    _prepare_phase_output,
    _validate_xprof_exports,
    build_seqax_surface_profile_receipt,
    run_seqax_surface_profile_phase,
)
from tpu_cake.xprof_export import XProfExport, XProfExportManifest


def _invocation(mode: RunMode) -> SeqaxSurfaceProfileInvocation:
    surface = seqax_forward_workload_surface()
    return SeqaxSurfaceProfileInvocation(
        schema_version=SEQAX_SURFACE_PROFILE_SCHEMA,
        surface_id=surface.surface_id,
        surface_receipt_sha256="a" * 64,
        scenario="tiny",
        mode=mode,
        seed=1,
        warmup_iterations=SEQAX_SURFACE_PROFILE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_SURFACE_PROFILE_MEASURED_ITERATIONS,
        schedule_sha256="b" * 64,
        jax_source_sha256="c" * 64,
        stablehlo_sha256="d" * 64,
        compiler_hlo_sha256="e" * 64,
        profiler_config_sha256="1" * 64,
        input_sha256=("2" * 64,),
        output_sha256="3" * 64,
        oracle_sha256="4" * 64,
        execution_identity_sha256="5" * 64,
        input_placement="resident-named-sharding-before-warmup",
        execution_scope="eight-device complete forward",
        runtime=RuntimeIdentity(python="3.13"),
        device_kind="TPU7x",
        device_count=8,
        run_id="f" * 64,
    )


def test_seqax_surface_profile_commands_are_public() -> None:
    parser = _parser()

    run = parser.parse_args(
        [
            "run-seqax-surface-profile",
            "--output-dir",
            "run/tiny/trace",
            "--surface-root",
            "surface",
            "--scenario",
            "tiny",
            "--mode",
            "trace",
        ]
    )
    assert run.output_dir == Path("run/tiny/trace")
    assert run.surface_root == Path("surface")
    assert run.scenario == "tiny"
    assert run.mode == "trace"
    assert (
        parser.parse_args(
            [
                "finalize-seqax-surface-profile",
                "run",
                "--surface-root",
                "surface",
            ]
        ).command
        == "finalize-seqax-surface-profile"
    )
    assert (
        parser.parse_args(["verify-seqax-surface-profile", "run"]).command
        == "verify-seqax-surface-profile"
    )


def test_seqax_surface_profile_protocol_rejects_timing_mode() -> None:
    with pytest.raises(ValidationError, match="SEQAX_SURFACE_PROFILE_PROTOCOL_MISMATCH"):
        _invocation(RunMode.TIMING)


def test_seqax_surface_profile_counter_contract_requires_physical_counters() -> None:
    trace = _expectation("tiny", RunMode.TRACE)
    counters = _expectation("tiny", RunMode.COUNTERS)

    assert trace.minimum_tpu_device_planes == 8
    assert trace.required_timed_hlo_markers == (
        "all-gather",
        "reduce_scatter",
        "dot_general",
    )
    assert not trace.require_tensor_core_activity
    assert not trace.require_hbm_read_counters
    assert counters.require_hbm_read_counters
    assert counters.require_hbm_write_counters
    assert counters.require_cycle_counters
    assert counters.minimum_counter_device_planes == 4


def test_seqax_surface_profile_does_not_write_inside_accepted_surface(
    tmp_path: Path, monkeypatch
) -> None:
    surface = tmp_path / "surface"
    surface.mkdir()
    output = surface / "profile"
    monkeypatch.setattr("tpu_cake.seqax_surface_profile.jax.default_backend", lambda: "tpu")

    with pytest.raises(ValueError, match="OUTPUT_OVERLAPS_SURFACE"):
        run_seqax_surface_profile_phase(
            output,
            surface_root=surface,
            scenario_name="tiny",
            mode=RunMode.TRACE,
        )
    assert not output.exists()


def test_seqax_surface_profile_finalizer_rejects_nested_roots(tmp_path: Path) -> None:
    surface = tmp_path / "surface"
    surface.mkdir()

    with pytest.raises(ValueError, match="ROOTS_MUST_BE_DISJOINT"):
        build_seqax_surface_profile_receipt(
            surface / "profile",
            surface_root=surface,
        )


def test_seqax_surface_profile_finalizer_does_not_write_to_unowned_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "valuable"
    root.mkdir()
    sentinel = root / "valuable.txt"
    sentinel.write_text("keep")
    surface = tmp_path / "surface"
    surface.mkdir()

    with pytest.raises(ValueError, match="ROOT_NOT_OWNED"):
        build_seqax_surface_profile_receipt(root, surface_root=surface)
    assert sentinel.read_text() == "keep"
    assert {path.name for path in root.iterdir()} == {"valuable.txt"}


def test_seqax_surface_profile_does_not_archive_an_unowned_directory(
    tmp_path: Path,
) -> None:
    surface = tmp_path / "surface"
    surface.mkdir()
    output = tmp_path / "valuable"
    output.mkdir()
    valuable = output / "valuable.txt"
    valuable.write_text("keep")

    with pytest.raises(ValueError, match="OUTPUT_NOT_OWNED"):
        _prepare_phase_output(
            output,
            surface_root=surface,
            surface_id="a" * 64,
            surface_receipt_sha256="b" * 64,
            scenario="tiny",
            mode=RunMode.TRACE,
        )
    assert valuable.read_text() == "keep"
    assert not tuple(tmp_path.glob("valuable.incomplete-*"))


def test_seqax_surface_profile_xprof_exports_are_closed_world(tmp_path: Path) -> None:
    output = tmp_path / "xprof"
    output.mkdir()
    hlo = output / "hlo_stats.json"
    hlo.write_text("{}")
    manifest = XProfExportManifest(
        xplane=tmp_path / "capture.xplane.pb",
        available_tools=("hlo_stats",),
        exports=(
            XProfExport(
                tool="hlo_stats",
                mime_type="application/json",
                output=hlo,
                size_bytes=hlo.stat().st_size,
            ),
        ),
    )
    (output / "manifest.json").write_text(manifest.model_dump_json())

    _validate_xprof_exports(output)
    (output / "unclaimed.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="EXPORT_SET_MISMATCH"):
        _validate_xprof_exports(output)


def test_seqax_surface_profile_finalizer_does_not_rewrite_an_existing_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    surface = tmp_path / "external-surface"
    receipt = SeqaxSurfaceProfileReceipt(
        schema_version=SEQAX_SURFACE_PROFILE_SCHEMA,
        surface_id="a" * 64,
        surface_receipt_sha256="b" * 64,
        results=(),
        metrics=(),
        artifacts=(),
        accepted=True,
    )
    receipt_path = root / "receipt.json"
    payload = receipt.model_dump_json(indent=2) + "\n"
    receipt_path.write_text(payload)
    observed = []
    monkeypatch.setattr(
        "tpu_cake.seqax_surface_profile.validate_seqax_surface_profile_receipt",
        lambda value, *, root: observed.append((value, root)),
    )

    assert build_seqax_surface_profile_receipt(root, surface_root=surface) == receipt
    assert receipt_path.read_text() == payload
    assert observed == [(receipt, root.resolve())]
    assert {path.name for path in root.iterdir()} == {"receipt.json"}
