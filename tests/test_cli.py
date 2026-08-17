from pathlib import Path

from tpu_cake.cli import _parser, _render_workload, _verify_rpa_bundle, _verify_schedule
from tpu_cake.contracts import CorrectnessResult, RunReceipt, RuntimeIdentity
from tpu_cake.workloads import inkling_fused_rpa_experiment


def test_frontend_schedule_round_trips_through_parser(tmp_path: Path) -> None:
    schedule = tmp_path / "matmul.mlir"
    assert _render_workload("matmul", schedule) == 0
    assert _verify_schedule(schedule) == 0


def test_rpa_bundle_commands_are_public() -> None:
    parser = _parser()

    assert parser.parse_args(["finalize-rpa-run", "bundle"]).command == "finalize-rpa-run"
    assert parser.parse_args(["verify-rpa-bundle", "bundle"]).command == "verify-rpa-bundle"


def test_public_rpa_verifier_uses_trusted_experiment_and_rejects_rejected_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = inkling_fused_rpa_experiment()
    receipt = RunReceipt(
        experiment_id=experiment.experiment_id,
        evidence_profile="opaque_rpa_adapter_v1",
        schedule_sha256=experiment.schedule_sha256,
        status="rejected",
        runtime=RuntimeIdentity(python="3.13"),
        correctness=CorrectnessResult(passed=False, oracle="trusted"),
        required_semantic_properties=(),
        metrics=(),
        artifacts=(),
        phases=(),
    )
    (tmp_path / "receipt.json").write_text(receipt.model_dump_json())
    observed = {}

    def capture_authority(receipt_arg, experiment_arg, *, root) -> None:
        observed["receipt"] = receipt_arg
        observed["experiment"] = experiment_arg
        observed["root"] = root

    monkeypatch.setattr("tpu_cake.cli.validate_fused_rpa_receipt", capture_authority)

    assert _verify_rpa_bundle(tmp_path) == 1
    assert observed["experiment"] == experiment
    assert observed["root"] == tmp_path.resolve()
