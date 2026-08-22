import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from tpu_cake.cli import _parser
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.ledger import EvidenceRun, RunState, seal_ledger
from tpu_cake.matmul_collective_confirmation import (
    MatmulCollectiveConfirmationContract,
    MatmulCollectiveConfirmationRunIdentity,
    MatmulCollectiveHost,
    MatmulCollectiveTimingRound,
    collective_confirmation_orders,
    collective_confirmation_statistics,
    default_matmul_collective_confirmation_contract,
)
from tpu_cake.matmul_collective_confirmation_runner import (
    _assessment,
    _host_matches_contract,
    _plan_manifest,
    _plan_sources,
    _prepare_output_root,
    _semantic_compiler_hlo,
    _source_manifest,
    _timing_attempt_payload,
    _validate_devices,
    _validate_ledger_database,
)
from tpu_cake.pallas_lowering import PallasMatmulPlan
from tpu_cake.runner import MatmulCollectiveStrategy


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python="3.12.3",
        jax="0.11.0",
        jaxlib="0.11.0",
        libtpu="0.0.44.1",
        xla=" --xla_tpu_use_enhanced_launch_barrier=true",
    )


def _rounds(
    contract: MatmulCollectiveConfirmationContract,
    *,
    baseline_ns: int,
    candidate_ns: int,
) -> tuple[MatmulCollectiveTimingRound, ...]:
    durations = {
        contract.baseline: baseline_ns,
        contract.candidate: candidate_ns,
    }
    return tuple(
        MatmulCollectiveTimingRound(
            round_index=round_index,
            position=position,
            strategy=strategy,
            samples_ns=(durations[strategy],) * contract.calls_per_position,
            median_ns=float(durations[strategy]),
        )
        for round_index, order in enumerate(collective_confirmation_orders(contract))
        for position, strategy in enumerate(order)
    )


def test_matmul_collective_confirmation_contract_is_canonical_json() -> None:
    payload = json.loads(
        Path("contracts/matmul-collective-confirmation-v1.json").read_text()
    )
    saved = MatmulCollectiveConfirmationContract.model_validate_json(json.dumps(payload))
    expected = default_matmul_collective_confirmation_contract(
        RuntimeIdentity.model_validate(payload["runtime"])
    )

    assert saved == expected
    assert saved.baseline is MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
    assert saved.candidate is MatmulCollectiveStrategy.PALLAS_BIDIRECTIONAL_RING
    assert saved.paired_rounds == 32
    assert saved.calls_per_position == 5
    assert not saved.allow_early_stopping
    assert not saved.allow_retry
    assert not saved.allow_outlier_removal
    assert not saved.reuse_diagnostic_timing_samples


def test_matmul_collective_confirmation_statistics_use_a_symmetric_gate() -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())

    promoted = collective_confirmation_statistics(
        contract,
        _rounds(contract, baseline_ns=100, candidate_ns=90),
    )
    retained = collective_confirmation_statistics(
        contract,
        _rounds(contract, baseline_ns=100, candidate_ns=110),
    )
    inconclusive = collective_confirmation_statistics(
        contract,
        _rounds(contract, baseline_ns=100, candidate_ns=99),
    )

    assert promoted.decision == "promote_candidate"
    assert promoted.candidate_promoted
    assert promoted.selected_strategy is contract.candidate
    assert retained.decision == "keep_baseline"
    assert not retained.candidate_promoted
    assert retained.selected_strategy is contract.baseline
    assert inconclusive.decision == "inconclusive"
    assert not inconclusive.candidate_promoted
    assert inconclusive.selected_strategy is contract.baseline


def test_matmul_collective_confirmation_statistics_reject_reordered_evidence() -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())
    rounds = list(_rounds(contract, baseline_ns=100, candidate_ns=90))
    rounds[0], rounds[1] = rounds[1], rounds[0]

    with pytest.raises(ValueError, match="execution order mismatch"):
        collective_confirmation_statistics(contract, tuple(rounds))


@pytest.mark.parametrize(
    ("baseline_ns", "candidate_ns"),
    ((100, 97), (100, 103)),
)
def test_matmul_collective_confirmation_threshold_is_strict(
    baseline_ns: int,
    candidate_ns: int,
) -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())

    statistics = collective_confirmation_statistics(
        contract,
        _rounds(
            contract,
            baseline_ns=baseline_ns,
            candidate_ns=candidate_ns,
        ),
    )

    assert statistics.decision == "inconclusive"
    assert not statistics.candidate_promoted
    assert statistics.selected_strategy is contract.baseline


def test_matmul_collective_confirmation_accepts_distinct_oracle_valid_outputs() -> None:
    canonical = default_matmul_collective_confirmation_contract(_runtime())
    contract = canonical.model_copy(
        update={"parameters": {**canonical.parameters, "m": 1, "n": 2}}
    )
    oracle = np.asarray([[1.0, 2.0]], dtype=np.float32)
    baseline = np.asarray([[1.0001, 2.0]], dtype=np.float32)
    candidate = np.asarray([[1.0, 1.9999]], dtype=np.float32)

    baseline_passed, _, _ = _assessment(baseline, oracle, contract)
    candidate_passed, _, _ = _assessment(candidate, oracle, contract)

    assert baseline_passed
    assert candidate_passed
    assert not np.array_equal(baseline, candidate)

    wrong_dtype = oracle.astype(np.float64)
    broadcastable = np.asarray([[1.0]], dtype=np.float32)
    assert not _assessment(wrong_dtype, oracle, contract)[0]
    assert not _assessment(broadcastable, oracle, contract)[0]


def test_matmul_collective_plan_replay_does_not_build_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())

    def reject_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("TPU executable construction is forbidden during replay")

    monkeypatch.setattr(PallasMatmulPlan, "build", reject_build)

    distributed, sources = _plan_sources(contract)

    assert distributed
    assert tuple(value.strategy for value in sources) == (
        contract.baseline,
        contract.candidate,
    )
    assert json.loads(json.dumps(_plan_manifest(sources[0].plan))) == _plan_manifest(
        sources[0].plan
    )


def test_matmul_collective_confirmation_rejects_retry_after_timing_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "confirmation"
    root.mkdir()
    canonical = default_matmul_collective_confirmation_contract(_runtime())
    contract = canonical.model_copy(
        update={"attempt_registry_root": str(tmp_path / "attempts")}
    )
    identity = MatmulCollectiveConfirmationRunIdentity(
        confirmation_id="1" * 64,
        run_id="2" * 64,
        source_commit="3" * 40,
    )
    (root / "run_identity.json").write_text(identity.model_dump_json())
    claim = _timing_attempt_payload(root, identity, contract, "started")
    registry = Path(contract.attempt_registry_root)
    registry.mkdir()
    (registry / f"{identity.run_id}.json").write_text(json.dumps(claim))
    (root / "timing_started.json").write_text(json.dumps(claim))

    with pytest.raises(ValueError, match="TIMING_ATTEMPT_NOT_RETRYABLE"):
        _prepare_output_root(root, identity, contract)


def test_matmul_collective_confirmation_rejects_second_output_root(
    tmp_path: Path,
) -> None:
    canonical = default_matmul_collective_confirmation_contract(_runtime())
    contract = canonical.model_copy(
        update={"attempt_registry_root": str(tmp_path / "attempts")}
    )
    identity = MatmulCollectiveConfirmationRunIdentity(
        confirmation_id="1" * 64,
        run_id="2" * 64,
        source_commit="3" * 40,
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    registry = Path(contract.attempt_registry_root)
    registry.mkdir()
    reserved = _timing_attempt_payload(first_root, identity, contract, "reserved")
    (registry / f"{identity.run_id}.json").write_text(json.dumps(reserved))

    with pytest.raises(ValueError, match="TIMING_ATTEMPT_NOT_RETRYABLE"):
        _prepare_output_root(second_root, identity, contract)


def test_matmul_collective_confirmation_recovers_reserved_attempt_before_timing(
    tmp_path: Path,
) -> None:
    canonical = default_matmul_collective_confirmation_contract(_runtime())
    contract = canonical.model_copy(
        update={"attempt_registry_root": str(tmp_path / "attempts")}
    )
    identity = MatmulCollectiveConfirmationRunIdentity(
        confirmation_id="1" * 64,
        run_id="2" * 64,
        source_commit="3" * 40,
    )
    root = tmp_path / "confirmation"
    root.mkdir()
    (root / "run_identity.json").write_text(identity.model_dump_json())
    (root / "partial.json").write_text("{}")
    registry = Path(contract.attempt_registry_root)
    registry.mkdir()
    reserved = _timing_attempt_payload(root, identity, contract, "reserved")
    (registry / f"{identity.run_id}.json").write_text(json.dumps(reserved))

    state = _prepare_output_root(root, identity, contract)

    assert state is None
    assert not any(root.iterdir())
    assert len(tuple(tmp_path.glob("confirmation.incomplete-*"))) == 1


def test_matmul_collective_confirmation_rejects_hidden_ledger_run(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.sqlite"
    run_id = "1" * 64
    EvidenceRun(ledger, run_id).record(RunState.CREATED, {"run": "expected"})
    EvidenceRun(ledger, "2" * 64).record(RunState.CREATED, {"run": "hidden"})
    seal_ledger(ledger, "sidecars")

    with pytest.raises(ValueError, match="LEDGER_SCOPE_MISMATCH"):
        _validate_ledger_database(ledger, run_id, 1)


def test_matmul_collective_confirmation_manifest_covers_authorities() -> None:
    paths = {value.path for value in _source_manifest()}

    assert "tpu_cake/cli.py" in paths
    assert "tpu_cake/contracts.py" in paths
    assert "tpu_cake/ledger.py" in paths
    assert "tpu_cake/matmul_collective_confirmation_runner.py" in paths


def test_matmul_collective_confirmation_compiler_identity_ignores_only_stack_metadata() -> None:
    first = """HloModule test

FileNames
1 "first.py"

FunctionNames
1 "first"

FileLocations
1 {file_name_id=1 function_name_id=1 line=1}

StackFrames
1 {file_location_id=1 parent_frame_id=1}

ENTRY %main () -> f32[] {
  ROOT %value = f32[] constant(1), metadata={op_name="value" stack_frame_id=1}
}
"""
    second = first.replace('1 "first.py"', '1 "second.py"').replace(
        "stack_frame_id=1",
        "stack_frame_id=99",
    )
    changed_program = second.replace("constant(1)", "constant(2)")

    assert _semantic_compiler_hlo(first) == _semantic_compiler_hlo(second)
    assert _semantic_compiler_hlo(first) != _semantic_compiler_hlo(changed_program)


def test_matmul_collective_confirmation_binds_raw_host_resources() -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())
    host = MatmulCollectiveHost(
        project=contract.project,
        numeric_project_id=contract.numeric_project_id,
        zone=contract.zone,
        hostname=contract.hostname,
        instance_hostname=contract.instance_hostname,
        machine_type=contract.machine_type,
        instance_id=contract.instance_id,
        cpu_platform=contract.cpu_platform,
        zone_resource=f"projects/{contract.numeric_project_id}/zones/{contract.zone}",
        machine_type_resource=(
            f"projects/{contract.numeric_project_id}/machineTypes/{contract.machine_type}"
        ),
    )

    assert _host_matches_contract(host, contract)
    assert not _host_matches_contract(
        host.model_copy(update={"zone_resource": "projects/other/zones/us-central1-c"}),
        contract,
    )


def test_matmul_collective_confirmation_rejects_alternate_device_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_confirmation_runner.jax.default_backend",
        lambda: "tpu",
    )
    devices = tuple(
        SimpleNamespace(platform="tpu", device_kind="TPU v7x")
        for _ in range(contract.device_count)
    )

    with pytest.raises(ValueError, match="DEVICE_MISMATCH"):
        _validate_devices(devices, contract)


def test_matmul_collective_confirmation_cli_is_explicit() -> None:
    run = _parser().parse_args(
        [
            "confirm-matmul-collective",
            "--output-dir",
            "/tmp/output",
            "--diagnostic-root",
            "/tmp/diagnostics",
            "--diagnostic-archive",
            "/tmp/diagnostics.tar.zst",
            "--contract",
            "contract.json",
        ]
    )
    verify = _parser().parse_args(
        [
            "verify-matmul-collective-confirmation",
            "/tmp/output",
            "--contract",
            "contract.json",
        ]
    )
    finalize = _parser().parse_args(
        [
            "finalize-matmul-collective-confirmation",
            "/tmp/output",
            "--contract",
            "contract.json",
        ]
    )

    assert run.command == "confirm-matmul-collective"
    assert verify.command == "verify-matmul-collective-confirmation"
    assert finalize.command == "finalize-matmul-collective-confirmation"


def test_matmul_collective_confirmation_rejects_changed_diagnostic_authority() -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())
    payload = contract.model_dump(mode="json", exclude_computed_fields=True)
    payload["diagnostics"][0]["receipt_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="diagnostic authority mismatch"):
        MatmulCollectiveConfirmationContract.model_validate_json(json.dumps(payload))


def test_matmul_collective_confirmation_rejects_changed_runtime() -> None:
    contract = default_matmul_collective_confirmation_contract(_runtime())
    payload = contract.model_dump(mode="json", exclude_computed_fields=True)
    payload["runtime"]["jax"] = "99.0.0"

    with pytest.raises(ValidationError, match="runtime mismatch"):
        MatmulCollectiveConfirmationContract.model_validate_json(json.dumps(payload))
