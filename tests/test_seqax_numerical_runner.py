import os
import subprocess
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import tpu_cake.seqax_numerical_runner as numerical_runner
from tpu_cake.cli import _parser
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.seqax_numerical import (
    SeqaxDiscriminatorClause,
    SeqaxNumericalDiscriminator,
    default_seqax_bf16_validation_contract,
    validate_strict_silu_stablehlo,
)
from tpu_cake.seqax_numerical_runner import (
    SeqaxBf16DiscriminatorObservation,
    SeqaxBf16RunIdentity,
    _drop_reduction_collective,
    _mutation_failure,
    _prepare_output_root,
    _record_failure,
    _remove_strict_barrier,
    _replace_silu_body,
    _require_safe_root,
    _transition_or_replay,
    _write_json_atomic,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)

_STRICT_HLO = """module {
  func.func public @main(%arg0: tensor<1x4xbf16>, %other: tensor<1x4xbf16>) -> tensor<1x4xbf16> {
    %0 = stablehlo.optimization_barrier %arg0 : tensor<1x4xbf16>
    %1 = func.call @silu(%0) : (tensor<1x4xbf16>) -> tensor<1x4xbf16>
    %2 = stablehlo.optimization_barrier %1 : tensor<1x4xbf16>
    %3 = stablehlo.multiply %other, %2 : tensor<1x4xbf16>
    return %3 : tensor<1x4xbf16>
  }
  func.func private @silu(%arg0: tensor<1x4xbf16>) -> tensor<1x4xbf16> {
    %0 = stablehlo.negate %arg0 : tensor<1x4xbf16>
    %1 = stablehlo.exponential %0 : tensor<1x4xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x4xbf16>
    %3 = stablehlo.add %2, %1 : tensor<1x4xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<1x4xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<1x4xbf16>
    %6 = stablehlo.multiply %arg0, %5 : tensor<1x4xbf16>
    return %6 : tensor<1x4xbf16>
  }
}"""


def test_runner_hlo_discriminators_mutate_the_real_strict_chain() -> None:
    validate_strict_silu_stablehlo(_STRICT_HLO, expected_count=1)
    mutants = (
        _remove_strict_barrier(_STRICT_HLO, input_barrier=True),
        _remove_strict_barrier(_STRICT_HLO, input_barrier=False),
        _replace_silu_body(_STRICT_HLO, relu=False),
        _replace_silu_body(_STRICT_HLO, relu=True),
    )

    for mutant in mutants:
        with pytest.raises(ValueError):
            validate_strict_silu_stablehlo(mutant, expected_count=1)
    assert "stablehlo.maximum" in mutants[-1]
    assert "func.call @silu" in mutants[-1]


def test_instrumented_control_returns_global_checkpoints_on_eight_devices() -> None:
    script = r"""
import jax
import numpy as np
import re

from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.seqax_numerical import (
    default_seqax_bf16_validation_contract,
    validate_instrumented_strict_silu_stablehlo,
)
from tpu_cake.seqax_numerical_runner import (
    _compile_instrumented_control,
    _execute_outputs,
    _resident_inputs,
)
from tpu_cake.workloads.seqax_forward import SeqaxNumericalSemantics, seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs

scenario = default_seqax_bf16_validation_contract().scenarios[1]
parameters = scenario.parameters.model_dump()
distributed = seqax_forward_schedule(
    **parameters,
    numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_V1,
)
plan = lower_distributed_program_to_jax_mesh(distributed)
devices = tuple(jax.devices("cpu"))
inputs = tuple(np.asarray(value) for value in seqax_forward_inputs(
    seed=scenario.seeds[0], **parameters
))
compiled = _compile_instrumented_control(
    plan,
    inputs,
    devices,
    expected_layers=parameters["layers"],
)
validate_instrumented_strict_silu_stablehlo(
    compiled.stablehlo,
    expected_count=parameters["layers"],
)
checkpoint_return = next(
    line for line in compiled.stablehlo.splitlines()
    if "sdy.return" in line and line.count("xbf16") == 4
)
match = re.search(
    r"sdy.return (?P<output>%[A-Za-z0-9_]+), (?P<gate>%[A-Za-z0-9_]+), "
    r"(?P<silu>%[A-Za-z0-9_]+),",
    checkpoint_return,
)
assert match is not None
mutant_return = checkpoint_return.replace(
    f", {match.group('gate')}, {match.group('silu')},",
    f", {match.group('silu')}, {match.group('silu')},",
)
mutant = compiled.stablehlo.replace(checkpoint_return, mutant_return)
try:
    validate_instrumented_strict_silu_stablehlo(
        mutant,
        expected_count=parameters["layers"],
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a forged gate output")
outer_return = next(
    line for line in compiled.stablehlo.splitlines()
    if "return %0#0, %0#1, %0#2, %0#3, %0#4" in line
)
mutant = compiled.stablehlo.replace(
    outer_return,
    outer_return.replace("%0#0, %0#1, %0#2", "%0#0, %0#2, %0#2"),
)
try:
    validate_instrumented_strict_silu_stablehlo(
        mutant,
        expected_count=parameters["layers"],
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a dropped gate result")
resident = _resident_inputs(inputs, plan, compiled.mesh)
outputs = _execute_outputs(compiled.executable, resident)
assert tuple(value.shape for value in outputs) == (
    (2, 3, 32),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
)
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_instrumented_pallas_returns_global_checkpoints_on_eight_devices() -> None:
    script = r"""
import jax
import numpy as np

from tpu_cake.seqax_numerical import (
    _validate_strict_silu_stablehlo,
    default_seqax_bf16_validation_contract,
)
from tpu_cake.seqax_numerical_runner import (
    _compile_instrumented_pallas,
    _execute_outputs,
    _resident_inputs,
)
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import SeqaxNumericalSemantics, seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import seqax_forward_inputs

scenario = default_seqax_bf16_validation_contract().scenarios[0]
parameters = scenario.parameters.model_dump()
distributed = seqax_forward_schedule(
    **parameters,
    numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_V1,
)
physical = lower_seqax_forward_to_physical(distributed).module
plan = lower_seqax_physical_to_pallas(distributed, physical)
devices = tuple(jax.devices("cpu"))
inputs = tuple(np.asarray(value) for value in seqax_forward_inputs(
    seed=scenario.seeds[0], **parameters
))
compiled = _compile_instrumented_pallas(
    plan,
    inputs,
    devices,
    expected_layers=parameters["layers"],
    interpret=True,
)
_validate_strict_silu_stablehlo(
    compiled.stablehlo,
    expected_count=parameters["layers"],
    instrumented=True,
    leading_result_count=1,
    allow_callbacks=True,
)
resident = _resident_inputs(inputs, plan, compiled.mesh)
outputs = _execute_outputs(compiled.executable, resident)
assert tuple((value.shape, str(value.dtype)) for value in outputs) == (
    ((2, 1, 16), "float32"),
    ((2, 1, 16), "bfloat16"),
    ((2, 1, 16), "bfloat16"),
)
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_runner_numerical_discriminators_target_the_named_clause() -> None:
    contract = default_seqax_bf16_validation_contract()
    scenario = contract.scenarios[0]
    seed = scenario.seeds[0]
    inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=seed, **scenario.parameters.model_dump())
    )
    reference = np.asarray(
        seqax_forward_canonical_reference(
            inputs,
            quantization_decimals=contract.policy.cpu_reference_quantization_decimals,
            **scenario.parameters.model_dump(),
        )
    )
    gate = np.zeros(scenario.gate_checkpoints[0].shape, dtype=ml_dtypes.bfloat16)
    silu = gate.copy()
    spike = reference.copy()
    spike.reshape(-1)[0] += 1

    failure = _mutation_failure(
        spike,
        clause=SeqaxDiscriminatorClause.ROW_SCALED_MAXIMUM,
        contract=contract,
        scenario=scenario,
        seed=seed,
        inputs=inputs,
        gates=(gate,),
        silus=(silu,),
    )

    assert failure.startswith("row_scaled_maximum: rejected")


def test_collective_discriminator_removes_exactly_one_reduce_scatter() -> None:
    physical = """builtin.module {
      %0 = "tpu_schedule.collective"() <{kind = "all_gather"}> : () -> i32
      %1 = "tpu_schedule.collective"() <{kind = "reduce_scatter"}> : () -> i32
      %2 = "tpu_schedule.collective"() <{kind = "reduce_scatter"}> : () -> i32
    }
"""

    mutant = _drop_reduction_collective(physical)

    assert mutant.count("reduce_scatter") == 1
    assert mutant.count("all_gather") == 1


def test_discriminator_observation_rejects_a_wrong_clause() -> None:
    with pytest.raises(ValueError, match="clause mismatch"):
        SeqaxBf16DiscriminatorObservation(
            discriminator=SeqaxNumericalDiscriminator.LOCALIZED_SPIKE,
            clause=SeqaxDiscriminatorClause.RELATIVE_L2,
            artifact_paths=("mutant.npy",),
            artifact_sha256=("0" * 64,),
            rejected=True,
            failure="rejected",
        )


def test_runner_rejects_protected_output_roots() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="UNSAFE_ROOT"):
        _require_safe_root(repository_root)
    with pytest.raises(ValueError, match="UNSAFE_ROOT"):
        _require_safe_root(repository_root / "runs" / "numerical")
    with pytest.raises(ValueError, match="UNSAFE_ROOT"):
        _require_safe_root(Path.home())


def test_runner_archives_only_an_owned_incomplete_root(tmp_path: Path) -> None:
    contract = default_seqax_bf16_validation_contract()
    identity = SeqaxBf16RunIdentity(
        schema_version="seqax-bf16-forward-validation-run-v1",
        contract_id=contract.contract_id,
        run_id="1" * 64,
        source_commit="2" * 40,
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "valuable.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="ROOT_NOT_OWNED"):
        _prepare_output_root(unrelated, identity, contract)
    assert sentinel.read_text() == "preserve"
    assert not tuple(tmp_path.glob("unrelated.incomplete-*"))

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    wrong_identity = identity.model_copy(update={"run_id": "4" * 64})
    (mismatched / "run_identity.json").write_text(wrong_identity.model_dump_json(indent=2) + "\n")
    with pytest.raises(ValueError, match="ROOT_NOT_OWNED"):
        _prepare_output_root(mismatched, identity, contract)
    assert mismatched.is_dir()

    accepted = tmp_path / "accepted"
    accepted.mkdir()
    (accepted / "run_identity.json").write_text(identity.model_dump_json(indent=2) + "\n")
    (accepted / "receipt.json").write_text("preserve")
    with pytest.raises(ValueError, match="ACCEPTED_ROOT_NOT_RETRYABLE"):
        _prepare_output_root(accepted, identity, contract)
    assert (accepted / "receipt.json").read_text() == "preserve"

    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "run_identity.json").write_text(identity.model_dump_json(indent=2) + "\n")
    (owned / "partial.txt").write_text("negative evidence")
    archived = _prepare_output_root(owned, identity, contract)
    assert archived is not None
    assert archived.parent == tmp_path
    assert (archived / "partial.txt").read_text() == "negative evidence"
    assert owned.is_dir() and not any(owned.iterdir())


def test_runner_records_a_terminal_failure_before_retry(tmp_path: Path) -> None:
    run_id = "3" * 64
    ledger_path = tmp_path / "ledger.sqlite"
    with ExperimentLedger(ledger_path) as ledger:
        ledger.create(run_id, {"contract": "test"})

    _record_failure(tmp_path, run_id, RuntimeError("compile failed"))

    history = read_ledger_history(ledger_path, run_id)
    assert tuple(event.state for event in history) == (
        RunState.CREATED,
        RunState.REJECTED,
    )
    failure = (tmp_path / "failure.json").read_text()
    assert '"error_type": "RuntimeError"' in failure
    assert '"message": "compile failed"' in failure
    assert '"previous_state": "created"' in failure


def test_runner_replays_exact_historical_ledger_states(tmp_path: Path) -> None:
    run_id = "7" * 64
    created = {"contract": "test"}
    verified = {"schedule": "8" * 64}
    with ExperimentLedger(tmp_path / "ledger.sqlite") as ledger:
        _transition_or_replay(ledger, run_id, RunState.CREATED, created)
        _transition_or_replay(ledger, run_id, RunState.VERIFIED, verified)
        _transition_or_replay(ledger, run_id, RunState.CREATED, created)
        _transition_or_replay(ledger, run_id, RunState.VERIFIED, verified)
        with pytest.raises(ValueError, match="LEDGER_REPLAY_MISMATCH state=created"):
            _transition_or_replay(
                ledger,
                run_id,
                RunState.CREATED,
                {"contract": "changed"},
            )

    history = read_ledger_history(tmp_path / "ledger.sqlite", run_id)
    assert tuple(event.state for event in history) == (
        RunState.CREATED,
        RunState.VERIFIED,
    )


def test_runner_resumes_active_owned_root_and_archives_rejected_root(
    tmp_path: Path,
) -> None:
    contract = default_seqax_bf16_validation_contract()
    identity = SeqaxBf16RunIdentity(
        schema_version="seqax-bf16-forward-validation-run-v1",
        contract_id=contract.contract_id,
        run_id="8" * 64,
        source_commit="9" * 40,
    )
    active = tmp_path / "active"
    active.mkdir()
    (active / "run_identity.json").write_text(identity.model_dump_json(indent=2) + "\n")
    with ExperimentLedger(active / "ledger.sqlite") as ledger:
        ledger.create(identity.run_id, {"contract": "test"})
        active_before = {
            path.name: path.read_bytes() for path in active.iterdir() if path.is_file()
        }
        assert numerical_runner._root_is_resumable(active, identity, contract)
        assert {
            path.name: path.read_bytes() for path in active.iterdir() if path.is_file()
        } == active_before
    active_closed = {path.name: path.read_bytes() for path in active.iterdir() if path.is_file()}
    assert _prepare_output_root(active, identity, contract) is None
    assert {
        path.name: path.read_bytes() for path in active.iterdir() if path.is_file()
    } == active_closed
    assert active.is_dir()
    assert not tuple(tmp_path.glob("active.incomplete-*"))

    rejected = tmp_path / "rejected"
    rejected.mkdir()
    (rejected / "run_identity.json").write_text(identity.model_dump_json(indent=2) + "\n")
    with ExperimentLedger(rejected / "ledger.sqlite") as ledger:
        ledger.create(identity.run_id, {"contract": "test"})
        ledger.transition(identity.run_id, RunState.REJECTED, {"error": "failed"})
        rejected_before = {
            path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()
        }
        assert not numerical_runner._root_is_resumable(rejected, identity, contract)
        assert {
            path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()
        } == rejected_before
    rejected_closed = {
        path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()
    }
    archived = _prepare_output_root(rejected, identity, contract)
    assert archived is not None
    assert {
        path.name: path.read_bytes() for path in archived.iterdir() if path.is_file()
    } == rejected_closed
    assert (archived / "ledger.sqlite").is_file()
    assert rejected.is_dir() and not any(rejected.iterdir())


def test_runner_reuses_same_root_after_uncaught_active_run_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_seqax_bf16_validation_contract()
    root = tmp_path / "run"
    calls = 0

    def crash_once(
        active_root: Path,
        active_contract: object,
        _runtime: object,
        _devices: object,
        run_id: str,
        source_commit: str,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            with ExperimentLedger(active_root / "ledger.sqlite") as ledger:
                ledger.create(
                    run_id,
                    {
                        "schema": "seqax-bf16-forward-validation-run-v1",
                        "contract_id": active_contract.contract_id,
                        "source_commit": source_commit,
                    },
                )
            raise KeyboardInterrupt("simulated process death")
        return sentinel

    sentinel = object()
    monkeypatch.setattr(numerical_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(numerical_runner, "_runtime", lambda _contract: object())
    monkeypatch.setattr(numerical_runner.jax, "devices", list)
    monkeypatch.setattr(numerical_runner, "_validate_devices", lambda _devices, _contract: None)
    monkeypatch.setattr(numerical_runner, "_execute_seqax_bf16_validation", crash_once)

    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        numerical_runner.run_seqax_bf16_validation(root, contract)
    identity_bytes = (root / "run_identity.json").read_bytes()

    assert numerical_runner.run_seqax_bf16_validation(root, contract) is sentinel
    assert (root / "run_identity.json").read_bytes() == identity_bytes
    assert not tuple(tmp_path.glob("run.incomplete-*"))


def test_runner_rejects_a_concurrent_live_owner_without_mutation(
    tmp_path: Path,
) -> None:
    contract = default_seqax_bf16_validation_contract()
    root = tmp_path / "run"
    root.mkdir()
    repository_root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_id = numerical_runner.semantic_sha256(
        "seqax-bf16-forward-validation-run-v1",
        contract.contract_id,
        source_commit,
    )
    identity = SeqaxBf16RunIdentity(
        schema_version="seqax-bf16-forward-validation-run-v1",
        contract_id=contract.contract_id,
        run_id=run_id,
        source_commit=source_commit,
    )
    (root / "run_identity.json").write_text(identity.model_dump_json(indent=2) + "\n")
    with ExperimentLedger(root / "ledger.sqlite") as ledger:
        ledger.create(
            run_id,
            {
                "schema": "seqax-bf16-forward-validation-run-v1",
                "contract_id": contract.contract_id,
                "source_commit": source_commit,
            },
        )
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = """
import sys
import time
from pathlib import Path
from tpu_cake.seqax_numerical_runner import _exclusive_run_lock

root, ready, release = map(Path, sys.argv[1:])
with _exclusive_run_lock(root):
    ready.write_text("locked")
    while not release.exists():
        time.sleep(0.01)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(root), str(ready), str(release)])
    try:
        for _ in range(500):
            if ready.exists():
                break
            if process.poll() is not None:
                raise AssertionError(f"lock holder exited with {process.returncode}")
            time.sleep(0.01)
        else:
            raise AssertionError("lock holder did not acquire the run lock")
        before = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        with pytest.raises(ValueError, match="SEQAX_BF16_RUN_LOCKED"):
            numerical_runner.run_seqax_bf16_validation(root, contract)
        after = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert not tuple(tmp_path.glob("run.incomplete-*"))
    finally:
        release.write_text("release")
        process.wait(timeout=10)
    assert process.returncode == 0


def test_atomic_run_markers_do_not_publish_truncated_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_seqax_bf16_validation_contract()
    identity = SeqaxBf16RunIdentity(
        schema_version="seqax-bf16-forward-validation-run-v1",
        contract_id=contract.contract_id,
        run_id="5" * 64,
        source_commit="6" * 40,
    )
    root = tmp_path / "run"
    root.mkdir()
    original_replace = os.replace

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated crash before atomic publish")

    monkeypatch.setattr("tpu_cake.seqax_numerical_runner.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        _write_json_atomic(root / "run_identity.json", identity.model_dump(mode="json"))
    assert not (root / "run_identity.json").exists()
    assert not any(root.iterdir())
    assert _prepare_output_root(root, identity, contract) is None

    monkeypatch.setattr("tpu_cake.seqax_numerical_runner.os.replace", original_replace)
    _write_json_atomic(root / "run_identity.json", identity.model_dump(mode="json"))
    (root / "partial.txt").write_text("preserve")
    monkeypatch.setattr("tpu_cake.seqax_numerical_runner.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        _write_json_atomic(root / "receipt.json", {"status": "passed"})
    assert not (root / "receipt.json").exists()
    monkeypatch.setattr("tpu_cake.seqax_numerical_runner.os.replace", original_replace)
    archived = _prepare_output_root(root, identity, contract)
    assert archived is not None
    assert (archived / "partial.txt").read_text() == "preserve"


def test_runner_retries_after_identity_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_seqax_bf16_validation_contract()
    root = tmp_path / "run"
    original_write = numerical_runner._write_json_atomic
    attempts = 0

    def fail_first_identity(path: Path, value: object) -> None:
        nonlocal attempts
        if path.name == "run_identity.json" and attempts == 0:
            attempts += 1
            raise OSError("simulated identity publication failure")
        original_write(path, value)

    monkeypatch.setattr(numerical_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(numerical_runner, "_runtime", lambda _contract: object())
    monkeypatch.setattr(numerical_runner.jax, "devices", list)
    monkeypatch.setattr(numerical_runner, "_validate_devices", lambda _devices, _contract: None)
    monkeypatch.setattr(numerical_runner, "_write_json_atomic", fail_first_identity)

    with pytest.raises(OSError, match="identity publication failure"):
        numerical_runner.run_seqax_bf16_validation(root, contract)
    assert root.is_dir() and not any(root.iterdir())

    sentinel = object()
    monkeypatch.setattr(
        numerical_runner,
        "_execute_seqax_bf16_validation",
        lambda *_args: sentinel,
    )
    assert numerical_runner.run_seqax_bf16_validation(root, contract) is sentinel
    assert (root / "run_identity.json").is_file()


def test_bf16_validation_cli_requires_external_contract() -> None:
    parser = _parser()
    run = parser.parse_args(
        [
            "validate-seqax-bf16-forward",
            "--contract",
            "contract.json",
            "--output-dir",
            "run",
        ]
    )
    verify = parser.parse_args(["verify-seqax-bf16-forward", "run", "--contract", "contract.json"])

    assert run.command == "validate-seqax-bf16-forward"
    assert verify.command == "verify-seqax-bf16-forward"
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-seqax-bf16-forward", "run"])


def test_bf16_runner_builds_and_replays_a_relocated_receipt() -> None:
    script = r"""
import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

import tpu_cake.seqax_numerical_runner as runner
from tpu_cake.contracts import RuntimeIdentity, SourceFileContract
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.seqax_numerical import rounded_mathematical_silu_bf16
from tpu_cake.seqax_pallas_lowering import _parse_distributed

original_compile_path = runner._compile_path
original_instrumented_pallas = runner._compile_instrumented_pallas
original_activation_mutant = runner._compile_activation_mutant
original_execute_outputs = runner._execute_outputs
original_source_state = runner._source_state


def control_plan(plan):
    distributed = _parse_distributed(plan.canonical_distributed_xdsl)
    return lower_distributed_program_to_jax_mesh(distributed)


def compile_path(plan, inputs, devices, *, pallas, interpret_pallas=False):
    compiled = original_compile_path(
        plan,
        inputs,
        devices,
        pallas=pallas,
        interpret_pallas=pallas,
    )
    if not pallas:
        return compiled
    structural = original_compile_path(control_plan(plan), inputs, devices, pallas=False)
    return runner._CompiledPath(
        plan=compiled.plan,
        executable=compiled.executable,
        mesh=compiled.mesh,
        stablehlo=structural.stablehlo,
        compiler_hlo=structural.compiler_hlo,
    )


def instrumented_pallas(plan, inputs, devices, *, expected_layers, interpret=False):
    compiled = original_instrumented_pallas(
        plan,
        inputs,
        devices,
        expected_layers=expected_layers,
        interpret=True,
    )
    structural = runner._compile_instrumented_control(
        control_plan(plan),
        inputs,
        devices,
        expected_layers=expected_layers,
    )
    return runner._InstrumentedPath(
        plan=compiled.plan,
        executable=compiled.executable,
        mesh=compiled.mesh,
        stablehlo=structural.stablehlo,
        compiler_hlo=structural.compiler_hlo,
    )


def activation_mutant(
    path,
    inputs,
    devices,
    *,
    pallas,
    relu,
    interpret_pallas=False,
):
    return original_activation_mutant(
        path,
        inputs,
        devices,
        pallas=pallas,
        relu=relu,
        interpret_pallas=pallas,
    )


def execute_outputs(executable, inputs):
    outputs = list(original_execute_outputs(executable, inputs))
    if len(outputs) > 1:
        for gate_index in range(1, len(outputs), 2):
            outputs[gate_index + 1] = rounded_mathematical_silu_bf16(outputs[gate_index])
    return tuple(outputs)


def source_state(repository, root):
    artifacts = original_source_state(repository, root)
    (root / "source_diff.patch").write_text("")
    state = json.loads((root / "source_state.json").read_text())
    state["git_dirty"] = False
    state["git_status"] = []
    state["source_diff_sha256"] = hashlib.sha256(b"").hexdigest()
    (root / "source_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return artifacts


def source_manifest():
    path = Path(runner.__file__).resolve().parent / "canonical.py"
    return (
        SourceFileContract(
            path="tpu_cake/canonical.py",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )


def runtime(contract):
    expected = contract.runtime
    return runner.SeqaxBf16Runtime(
        runtime=RuntimeIdentity(
            python=expected.python_major_minor + ".3",
            jax=expected.jax,
            jaxlib=expected.jaxlib,
            libtpu=expected.libtpu,
            xla=expected.libtpu_init_args,
        ),
        ml_dtypes=expected.ml_dtypes,
    )


def device_inventory(devices):
    return tuple(
        runner.SeqaxBf16Device(
            id=index,
            process_index=0,
            platform="tpu",
            device_kind="TPU7x",
        )
        for index in range(8)
    )


def repair_receipt(root, relative_path):
    path = root / relative_path
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for artifact in receipt["artifacts"]:
        if artifact["path"] == relative_path:
            artifact["size_bytes"] = path.stat().st_size
            artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            break
    else:
        raise AssertionError(relative_path)
    if relative_path == "result.json":
        receipt["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if relative_path == "ledger.sqlite":
        receipt["ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def require_rejected(root, contract):
    try:
        runner.validate_seqax_bf16_validation(root, contract)
    except ValueError:
        return
    raise AssertionError(f"mutated receipt accepted: {root}")


runner._compile_path = compile_path
runner._compile_instrumented_pallas = instrumented_pallas
runner._compile_activation_mutant = activation_mutant
runner._execute_outputs = execute_outputs
runner._source_state = source_state
runner._source_manifest = source_manifest
runner._runtime = runtime
runner._device_inventory = device_inventory
runner._require_clean_repository = lambda root: None
runner._validate_compiled_program = lambda *args, **kwargs: None

contract = runner.default_seqax_bf16_validation_contract()
temporary = Path(tempfile.mkdtemp(prefix="seqax-bf16-lifecycle-"))
try:
    root = temporary / "run"
    result = runner.run_seqax_bf16_validation(root, contract)
    assert len(result.observations) == 17
    assert len(result.discriminators) == 14
    assert runner.run_seqax_bf16_validation(root, contract) == result
    relocated = temporary / "relocated"
    shutil.copytree(root, relocated)
    runner.validate_seqax_bf16_validation(relocated, contract)

    contract_mutant = temporary / "contract-mutant"
    shutil.copytree(root, contract_mutant)
    payload = json.loads((contract_mutant / "contract.json").read_text())
    payload["policy"]["cpu_relative_l2_units"] = 3.0
    (contract_mutant / "contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    repair_receipt(contract_mutant, "contract.json")
    require_rejected(contract_mutant, contract)

    result_mutant = temporary / "result-mutant"
    shutil.copytree(root, result_mutant)
    payload = json.loads((result_mutant / "result.json").read_text())
    payload["passed"] = False
    (result_mutant / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    repair_receipt(result_mutant, "result.json")
    require_rejected(result_mutant, contract)

    identity_mutant = temporary / "identity-mutant"
    shutil.copytree(root, identity_mutant)
    payload = json.loads((identity_mutant / "run_identity.json").read_text())
    payload["run_id"] = "0" * 64
    (identity_mutant / "run_identity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    repair_receipt(identity_mutant, "run_identity.json")
    require_rejected(identity_mutant, contract)

    hlo_mutant = temporary / "hlo-mutant"
    shutil.copytree(root, hlo_mutant)
    relative_hlo = "plans/calibration-m256-b2-s1-l1/instrumented_control_stablehlo.txt"
    with (hlo_mutant / relative_hlo).open("a") as stream:
        stream.write("garbage\n")
    repair_receipt(hlo_mutant, relative_hlo)
    require_rejected(hlo_mutant, contract)

    runtime_hlo_mutant = temporary / "runtime-hlo-mutant"
    shutil.copytree(root, runtime_hlo_mutant)
    relative_runtime_hlo = (
        "discriminators/identity_silu/pallas_runtime_stablehlo.txt"
    )
    (runtime_hlo_mutant / relative_runtime_hlo).write_text("garbage\n")
    repair_receipt(runtime_hlo_mutant, relative_runtime_hlo)
    require_rejected(runtime_hlo_mutant, contract)

    checkpoint_mutant = temporary / "checkpoint-mutant"
    shutil.copytree(root, checkpoint_mutant)
    checkpoint = next(checkpoint_mutant.glob("scenarios/*/seed-*/checkpoints/pallas_gate_00.npy"))
    storage = np.load(checkpoint, allow_pickle=False)
    storage.reshape(-1)[0] ^= np.uint16(1)
    np.save(checkpoint, storage, allow_pickle=False)
    relative_checkpoint = checkpoint.relative_to(checkpoint_mutant).as_posix()
    repair_receipt(checkpoint_mutant, relative_checkpoint)
    require_rejected(checkpoint_mutant, contract)

    ledger_mutant = temporary / "ledger-mutant"
    shutil.copytree(root, ledger_mutant)
    with sqlite3.connect(ledger_mutant / "ledger.sqlite") as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = ?",
            ("0" * 64, "validated"),
        )
    repair_receipt(ledger_mutant, "ledger.sqlite")
    require_rejected(ledger_mutant, contract)

    receipt_mutant = temporary / "receipt-mutant"
    shutil.copytree(root, receipt_mutant)
    payload = json.loads((receipt_mutant / "receipt.json").read_text())
    payload["contract_id"] = "0" * 64
    (receipt_mutant / "receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    require_rejected(receipt_mutant, contract)
finally:
    shutil.rmtree(temporary)
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
