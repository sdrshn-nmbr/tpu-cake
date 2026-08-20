import io
import os
import subprocess
import sys
import tarfile
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
    _validate_strict_silu_stablehlo,
    default_seqax_bf16_validation_contract,
    seqax_stablehlo_sha256,
    validate_strict_silu_stablehlo,
)
from tpu_cake.seqax_numerical_runner import (
    SeqaxBf16DiscriminatorObservation,
    SeqaxBf16RunIdentity,
    _checkpoint_mutants,
    _drop_reduction_collective,
    _mutation_failure,
    _prepare_output_root,
    _record_failure,
    _remove_strict_barrier,
    _replace_silu_body,
    _require_compilation_source_root,
    _require_relocation_runtime,
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
    %promoted = stablehlo.convert %0 : (tensor<1x4xbf16>) -> tensor<1x4xf32>
    %1 = func.call @silu(%promoted) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %rounded = stablehlo.convert %1 : (tensor<1x4xf32>) -> tensor<1x4xbf16>
    %2 = stablehlo.optimization_barrier %rounded : tensor<1x4xbf16>
    %3 = stablehlo.multiply %other, %2 : tensor<1x4xbf16>
    return %3 : tensor<1x4xbf16>
  }
  func.func private @silu(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = stablehlo.negate %arg0 : tensor<1x4xf32>
    %1 = stablehlo.exponential %0 : tensor<1x4xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %3 = stablehlo.add %2, %1 : tensor<1x4xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %5 = stablehlo.divide %4, %3 : tensor<1x4xf32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<1x4xf32>
    return %6 : tensor<1x4xf32>
  }
}"""


def _pinned_test_contract(monkeypatch: pytest.MonkeyPatch):
    contract = default_seqax_bf16_validation_contract().model_copy(
        update={"hlo_identity_status": "pinned"}
    )
    monkeypatch.setattr(
        numerical_runner,
        "default_seqax_bf16_validation_contract",
        lambda: contract,
    )
    return contract


def test_runner_rejects_the_retired_v1_lifecycle_schema() -> None:
    contract = default_seqax_bf16_validation_contract()

    with pytest.raises(ValueError, match="schema_version"):
        SeqaxBf16RunIdentity(
            schema_version="seqax-bf16-forward-validation-run-v1",
            contract_id=contract.contract_id,
            run_id="1" * 64,
            source_commit="2" * 40,
        )


def test_runner_hlo_discriminators_mutate_the_real_strict_chain() -> None:
    expected_sha256 = seqax_stablehlo_sha256(_STRICT_HLO)
    _validate_strict_silu_stablehlo(
        _STRICT_HLO,
        expected_count=1,
        instrumented=False,
        require_hidden_down=False,
    )
    mutants = (
        _remove_strict_barrier(_STRICT_HLO, input_barrier=True),
        _remove_strict_barrier(_STRICT_HLO, input_barrier=False),
        _replace_silu_body(_STRICT_HLO, relu=False),
        _replace_silu_body(_STRICT_HLO, relu=True),
    )

    for mutant in mutants:
        with pytest.raises(ValueError):
            validate_strict_silu_stablehlo(
                mutant,
                expected_count=1,
                expected_sha256=expected_sha256,
            )
    assert "stablehlo.maximum" in mutants[-1]
    assert "func.call @silu" in mutants[-1]


def test_instrumented_control_returns_global_checkpoints_on_eight_devices() -> None:
    script = r"""
import jax
import numpy as np
import re
import tempfile
from pathlib import Path

from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.seqax_numerical import (
    _validate_strict_silu_stablehlo,
    canonical_seqax_stablehlo,
    default_seqax_bf16_validation_contract,
    seqax_stablehlo_sha256,
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
    numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
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
with tempfile.TemporaryDirectory() as temporary:
    stablehlo_path = Path(temporary) / "instrumented_stablehlo.txt"
    stablehlo_path.write_text(canonical_seqax_stablehlo(compiled.stablehlo))
    replayed_stablehlo = stablehlo_path.read_text()
    validate_instrumented_strict_silu_stablehlo(
        replayed_stablehlo,
        expected_count=parameters["layers"],
        expected_sha256=seqax_stablehlo_sha256(compiled.stablehlo),
    )
    _validate_strict_silu_stablehlo(
        replayed_stablehlo,
        expected_count=parameters["layers"],
        instrumented=True,
    )
checkpoint_return = next(
    line for line in compiled.stablehlo.splitlines()
    if "sdy.return" in line and line.count(",") >= 8
)
match = re.search(
    r"sdy.return (?P<output>%[A-Za-z0-9_]+), "
    r"(?P<rms_input>%[A-Za-z0-9_]+), (?P<rms_mean_square>%[A-Za-z0-9_]+), "
    r"(?P<rms_inverse>%[A-Za-z0-9_]+), (?P<normalized_float32>%[A-Za-z0-9_]+), "
    r"(?P<normalized>%[A-Za-z0-9_]+), (?P<gate_float32>%[A-Za-z0-9_]+), "
    r"(?P<gate>%[A-Za-z0-9_]+), "
    r"(?P<silu>%[A-Za-z0-9_]+), (?P<up_float32>%[A-Za-z0-9_]+), "
    r"(?P<up_bfloat16>%[A-Za-z0-9_]+), (?P<hidden>%[A-Za-z0-9_]+), "
    r"(?P<down_float32>%[A-Za-z0-9_]+), (?P<down_bfloat16>%[A-Za-z0-9_]+)",
    checkpoint_return,
)
assert match is not None
rms_input_mutant_return = checkpoint_return.replace(
    f", {match.group('rms_input')}, {match.group('rms_mean_square')},",
    f", {match.group('normalized')}, {match.group('rms_mean_square')},",
)
rms_input_mutant = compiled.stablehlo.replace(
    checkpoint_return, rms_input_mutant_return
)
try:
    _validate_strict_silu_stablehlo(
        rms_input_mutant,
        expected_count=parameters["layers"],
        instrumented=True,
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a forged RMS input output")
normalized_mutant_return = checkpoint_return.replace(
    f", {match.group('normalized')}, {match.group('gate_float32')},",
    f", {match.group('gate')}, {match.group('gate_float32')},",
)
normalized_mutant = compiled.stablehlo.replace(checkpoint_return, normalized_mutant_return)
try:
    _validate_strict_silu_stablehlo(
        normalized_mutant,
        expected_count=parameters["layers"],
        instrumented=True,
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a forged normalized input output")
gate_float32_mutant_return = checkpoint_return.replace(
    f", {match.group('gate_float32')}, {match.group('gate')},",
    f", {match.group('up_float32')}, {match.group('gate')},",
)
gate_float32_mutant = compiled.stablehlo.replace(
    checkpoint_return, gate_float32_mutant_return
)
try:
    _validate_strict_silu_stablehlo(
        gate_float32_mutant,
        expected_count=parameters["layers"],
        instrumented=True,
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a forged float32 gate output")
mutant_return = checkpoint_return.replace(
    f", {match.group('gate')}, {match.group('silu')},",
    f", {match.group('silu')}, {match.group('silu')},",
)
mutant = compiled.stablehlo.replace(checkpoint_return, mutant_return)
try:
    validate_instrumented_strict_silu_stablehlo(
        mutant,
        expected_count=parameters["layers"],
        expected_sha256=seqax_stablehlo_sha256(compiled.stablehlo),
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a forged gate output")
hidden_mutant_return = checkpoint_return.replace(
    f", {match.group('up_bfloat16')}, {match.group('hidden')},",
    f", {match.group('up_bfloat16')}, {match.group('silu')},",
)
hidden_mutant = compiled.stablehlo.replace(checkpoint_return, hidden_mutant_return)
try:
    _validate_strict_silu_stablehlo(
        hidden_mutant,
        expected_count=parameters["layers"],
        instrumented=True,
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a forged hidden output")
hidden_barrier = None
hidden_match = None
for line in compiled.stablehlo.splitlines():
    candidate = re.search(
        r"(?P<result>%[A-Za-z0-9_]+) = stablehlo\.optimization_barrier "
        r"(?P<input>%[A-Za-z0-9_]+) : tensor<1x3x6xbf16>",
        line,
    )
    if candidate is None:
        continue
    if any(
        "stablehlo.dot_general" in consumer
        and "contracting_dims = [2] x [1]" in consumer
        and candidate.group("result") in consumer
        for consumer in compiled.stablehlo.splitlines()
    ):
        hidden_barrier = line
        hidden_match = candidate
        break
assert hidden_barrier is not None and hidden_match is not None
down_projection = next(
    line for line in compiled.stablehlo.splitlines()
    if "stablehlo.dot_general" in line and hidden_match.group("result") in line
)
bypass = compiled.stablehlo.replace(
    down_projection,
    down_projection.replace(hidden_match.group("result"), hidden_match.group("input")),
)
try:
    _validate_strict_silu_stablehlo(
        bypass,
        expected_count=parameters["layers"],
        instrumented=True,
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a hidden barrier bypass")
outer_return = next(
    line for line in compiled.stablehlo.splitlines()
    if "return " + ", ".join(f"%0#{index}" for index in range(27)) in line
)
mutant = compiled.stablehlo.replace(
    outer_return,
    outer_return.replace("%0#0, %0#1, %0#2", "%0#0, %0#2, %0#2"),
)
try:
    validate_instrumented_strict_silu_stablehlo(
        mutant,
        expected_count=parameters["layers"],
        expected_sha256=seqax_stablehlo_sha256(compiled.stablehlo),
    )
except ValueError:
    pass
else:
    raise AssertionError("instrumented executable accepted a dropped gate result")
resident = _resident_inputs(inputs, plan, compiled.mesh)
outputs = _execute_outputs(compiled.executable, resident)
actual_shapes = tuple(value.shape for value in outputs)
assert actual_shapes == (
    (2, 3, 32),
    (2, 3, 128),
    (2, 3, 1),
    (2, 3, 1),
    (2, 3, 128),
    (2, 3, 128),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 128),
    (2, 3, 128),
    (2, 3, 128),
    (2, 3, 1),
    (2, 3, 1),
    (2, 3, 128),
    (2, 3, 128),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 24),
    (2, 3, 128),
    (2, 3, 128),
), actual_shapes
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
    numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_HIDDEN_V2,
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
actual_outputs = tuple((value.shape, str(value.dtype)) for value in outputs)
assert actual_outputs == (
    ((2, 1, 16), "float32"),
    ((2, 1, 256), "bfloat16"),
    ((2, 1, 1), "float32"),
    ((2, 1, 1), "float32"),
    ((2, 1, 256), "float32"),
    ((2, 1, 256), "bfloat16"),
    ((2, 1, 16), "float32"),
    ((2, 1, 16), "bfloat16"),
    ((2, 1, 16), "bfloat16"),
    ((2, 1, 16), "float32"),
    ((2, 1, 16), "bfloat16"),
    ((2, 1, 16), "bfloat16"),
    ((2, 1, 256), "float32"),
    ((2, 1, 256), "bfloat16"),
), actual_outputs
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
    gate_float32 = np.zeros(scenario.gate_float32_checkpoints[0].shape, dtype=np.float32)
    normalized_input = np.zeros(
        scenario.normalized_input_checkpoints[0].shape, dtype=ml_dtypes.bfloat16
    )
    rms_input = np.zeros(scenario.rms_input_checkpoints[0].shape, dtype=ml_dtypes.bfloat16)
    rms_mean_square = np.zeros(scenario.rms_mean_square_checkpoints[0].shape, dtype=np.float32)
    rms_inverse = np.full(
        scenario.rms_inverse_checkpoints[0].shape,
        np.float32(1.0) / np.sqrt(np.float32(1e-6)),
        dtype=np.float32,
    )
    normalized_float32 = np.zeros(
        scenario.normalized_float32_checkpoints[0].shape, dtype=np.float32
    )
    silu = gate.copy()
    up_float32 = np.zeros(scenario.up_float32_checkpoints[0].shape, dtype=np.float32)
    up = gate.copy()
    hidden = np.zeros(scenario.hidden_checkpoints[0].shape, dtype=ml_dtypes.bfloat16)
    down_float32 = np.zeros(scenario.down_float32_checkpoints[0].shape, dtype=np.float32)
    down_bfloat16 = np.zeros(scenario.down_bfloat16_checkpoints[0].shape, dtype=ml_dtypes.bfloat16)
    spike = reference.copy()
    spike.reshape(-1)[0] += 1

    failure = _mutation_failure(
        spike,
        clause=SeqaxDiscriminatorClause.ROW_SCALED_MAXIMUM,
        contract=contract,
        scenario=scenario,
        seed=seed,
        inputs=inputs,
        rms_inputs=(rms_input,),
        rms_mean_square=(rms_mean_square,),
        rms_inverse=(rms_inverse,),
        normalized_float32=(normalized_float32,),
        normalized_inputs=(normalized_input,),
        gate_float32=(gate_float32,),
        gates=(gate,),
        silus=(silu,),
        up_float32=(up_float32,),
        up=(up,),
        hidden=(hidden,),
        down_float32=(down_float32,),
        down_bfloat16=(down_bfloat16,),
    )

    assert failure.startswith("row_scaled_maximum: rejected")


def test_checkpoint_mutants_build_the_wrong_rms_scale_case() -> None:
    contract = default_seqax_bf16_validation_contract()
    scenario = contract.scenarios[0]
    inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(
            seed=scenario.seeds[0],
            **scenario.parameters.model_dump(),
        )
    )
    rms_input = np.ones(scenario.rms_input_checkpoints[0].shape, dtype=ml_dtypes.bfloat16)
    rms_mean_square = np.ones(scenario.rms_mean_square_checkpoints[0].shape, dtype=np.float32)
    rms_inverse = np.ones(scenario.rms_inverse_checkpoints[0].shape, dtype=np.float32)
    normalized_float32 = np.ones(
        scenario.normalized_float32_checkpoints[0].shape,
        dtype=np.float32,
    )
    normalized = normalized_float32.astype(ml_dtypes.bfloat16)
    feed_forward_shape = scenario.gate_float32_checkpoints[0].shape
    gate_float32 = np.zeros(feed_forward_shape, dtype=np.float32)
    gate = gate_float32.astype(ml_dtypes.bfloat16)

    mutants = _checkpoint_mutants(
        inputs,
        (rms_input,),
        (rms_mean_square,),
        (rms_inverse,),
        (normalized_float32,),
        (normalized,),
        (gate_float32,),
        (gate,),
        (gate_float32.copy(),),
        (gate.copy(),),
    )

    wrong_scale = mutants[SeqaxNumericalDiscriminator.WRONG_RMS_SCALE_CHECKPOINT]
    assert len(wrong_scale) == 1
    assert wrong_scale[0].shape == scenario.normalized_float32_checkpoints[0].shape
    assert not np.array_equal(wrong_scale[0], normalized_float32)


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
        schema_version="seqax-bf16-forward-validation-run-v6",
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
        schema_version="seqax-bf16-forward-validation-run-v6",
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
    contract = _pinned_test_contract(monkeypatch)
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
                        "schema": "seqax-bf16-forward-validation-run-v6",
                        "contract_id": active_contract.contract_id,
                        "source_commit": source_commit,
                    },
                )
            raise KeyboardInterrupt("simulated process death")
        return sentinel

    sentinel = object()
    monkeypatch.setattr(numerical_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(
        numerical_runner,
        "_require_compilation_source_root",
        lambda _root, _contract: None,
    )
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
        "seqax-bf16-forward-validation-run-v6",
        contract.contract_id,
        source_commit,
    )
    identity = SeqaxBf16RunIdentity(
        schema_version="seqax-bf16-forward-validation-run-v6",
        contract_id=contract.contract_id,
        run_id=run_id,
        source_commit=source_commit,
    )
    (root / "run_identity.json").write_text(identity.model_dump_json(indent=2) + "\n")
    with ExperimentLedger(root / "ledger.sqlite") as ledger:
        ledger.create(
            run_id,
            {
                "schema": "seqax-bf16-forward-validation-run-v6",
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
        schema_version="seqax-bf16-forward-validation-run-v6",
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
    contract = _pinned_test_contract(monkeypatch)
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
    monkeypatch.setattr(
        numerical_runner,
        "_require_compilation_source_root",
        lambda _root, _contract: None,
    )
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
    attest = parser.parse_args(
        [
            "attest-seqax-bf16-forward-relocation",
            "bundle.tar.zst",
            "--contract",
            "contract.json",
            "--output",
            "attestation.json",
        ]
    )
    replay_attestation = parser.parse_args(
        [
            "verify-seqax-bf16-forward-relocation",
            "attestation.json",
            "bundle.tar.zst",
            "--contract",
            "contract.json",
        ]
    )

    assert run.command == "validate-seqax-bf16-forward"
    assert verify.command == "verify-seqax-bf16-forward"
    assert attest.command == "attest-seqax-bf16-forward-relocation"
    assert replay_attestation.command == "verify-seqax-bf16-forward-relocation"
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-seqax-bf16-forward", "run"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["attest-seqax-bf16-forward-relocation", "bundle.tar.zst", "--output", "a.json"]
        )


def test_relocation_archive_rejects_canonical_path_aliases(tmp_path: Path) -> None:
    archive = tmp_path / "aliased.tar"
    payload = b"same target"
    with tarfile.open(archive, "w") as bundle:
        for name in ("bundle/a", "bundle/./a"):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="NONCANONICAL_PATH|CANONICAL_PATH_COLLISION"):
        numerical_runner._inspect_relocation_archive(
            archive,
            max_members=10,
            max_member_name_bytes=1024,
            max_member_bytes=1024,
            max_total_uncompressed_bytes=2048,
        )


def test_relocation_archive_rejects_compressed_size_before_copy(tmp_path: Path) -> None:
    archive = tmp_path / "oversized.tar"
    archive.write_bytes(b"12345")

    with (
        pytest.raises(ValueError, match="COMPRESSED_SIZE_EXCEEDED"),
        numerical_runner._staged_relocation_archive(
            archive,
            tmp_path / "staged.tar",
            max_compressed_bytes=4,
        ),
    ):
        pytest.fail("oversized archive must not be staged")


def test_relocation_archive_staging_stops_if_the_source_grows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "growing.tar"
    archive.write_bytes(b"1234")
    original_read = numerical_runner.os.read
    appended = False

    def growing_read(descriptor: int, count: int) -> bytes:
        nonlocal appended
        if not appended:
            appended = True
            with archive.open("ab") as stream:
                stream.write(b"5")
        return original_read(descriptor, count)

    monkeypatch.setattr(numerical_runner.os, "read", growing_read)
    with (
        pytest.raises(ValueError, match="COMPRESSED_SIZE_EXCEEDED"),
        numerical_runner._staged_relocation_archive(
            archive,
            tmp_path / "staged.tar",
            max_compressed_bytes=4,
        ),
    ):
        pytest.fail("growing archive must not be staged")


def test_relocation_archive_rejects_member_count_while_streaming(tmp_path: Path) -> None:
    archive = tmp_path / "too-many.tar"
    with tarfile.open(archive, "w") as bundle:
        for name in ("bundle/receipt.json", "bundle/result.json"):
            member = tarfile.TarInfo(name)
            bundle.addfile(member, io.BytesIO())

    with pytest.raises(ValueError, match="MEMBER_COUNT_EXCEEDED"):
        numerical_runner._inspect_relocation_archive(
            archive,
            max_members=1,
            max_member_name_bytes=1024,
            max_member_bytes=1024,
            max_total_uncompressed_bytes=2048,
        )


@pytest.mark.parametrize(
    ("max_member_bytes", "max_total_uncompressed_bytes", "message"),
    ((3, 16, "MEMBER_SIZE_EXCEEDED"), (16, 7, "TOTAL_SIZE_EXCEEDED")),
)
def test_relocation_archive_rejects_uncompressed_size_limits(
    tmp_path: Path,
    max_member_bytes: int,
    max_total_uncompressed_bytes: int,
    message: str,
) -> None:
    archive = tmp_path / "oversized-member.tar"
    payload = b"1234"
    with tarfile.open(archive, "w") as bundle:
        for name in ("bundle/receipt.json", "bundle/result.json"):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match=message):
        numerical_runner._inspect_relocation_archive(
            archive,
            max_members=10,
            max_member_name_bytes=1024,
            max_member_bytes=max_member_bytes,
            max_total_uncompressed_bytes=max_total_uncompressed_bytes,
        )


def test_relocation_archive_uses_header_size_not_spoofed_owner_fields(tmp_path: Path) -> None:
    archive = tmp_path / "spoofed-owner.tar"
    payload = b"x" * 1234
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("bundle/receipt.json")
        member.size = len(payload)
        member.uname = "123"
        member.gname = "2026-01-01"
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="MEMBER_SIZE_EXCEEDED"):
        numerical_runner._inspect_relocation_archive(
            archive,
            max_members=10,
            max_member_name_bytes=1024,
            max_member_bytes=200,
            max_total_uncompressed_bytes=200,
        )


def test_relocation_archive_rejects_an_oversized_member_name(tmp_path: Path) -> None:
    archive = tmp_path / "long-name.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("bundle/" + "a" * 200 + "/receipt.json")
        bundle.addfile(member, io.BytesIO())

    with pytest.raises(ValueError, match="MEMBER_NAME_TOO_LONG"):
        numerical_runner._inspect_relocation_archive(
            archive,
            max_members=10,
            max_member_name_bytes=64,
            max_member_bytes=1024,
            max_total_uncompressed_bytes=2048,
        )


def test_relocation_archive_caps_the_expanded_zstd_stream(tmp_path: Path) -> None:
    if numerical_runner.shutil.which("zstd") is None:
        pytest.skip("zstd is required for relocation archive verification")
    source = tmp_path / "payload.tar"
    source.write_bytes(b"0" * 4096)
    compressed = tmp_path / "payload.tar.zst"
    with compressed.open("wb") as output:
        subprocess.run(
            ["zstd", "--compress", "--stdout", str(source)],
            check=True,
            stdout=output,
        )

    with pytest.raises(ValueError, match="EXPANDED_SIZE_EXCEEDED"):
        numerical_runner._materialize_tar_archive(
            compressed,
            tmp_path / "expanded.tar",
            max_expanded_archive_bytes=1024,
        )


def test_relocation_archive_rejects_missing_zstd_with_a_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressed = tmp_path / "bundle.tar.zst"
    compressed.write_bytes(numerical_runner._ZSTD_MAGIC + b"payload")
    monkeypatch.setattr(numerical_runner.shutil, "which", lambda _name: None)

    with pytest.raises(ValueError, match="ZSTD_UNAVAILABLE"):
        numerical_runner._materialize_tar_archive(
            compressed,
            tmp_path / "expanded.tar",
            max_expanded_archive_bytes=1024,
        )


def test_relocation_archive_inspects_and_extracts_a_zstd_snapshot(tmp_path: Path) -> None:
    if numerical_runner.shutil.which("zstd") is None:
        pytest.skip("zstd is required for relocation archive verification")
    source = tmp_path / "bundle.tar"
    payload = b"receipt"
    with tarfile.open(source, "w") as bundle:
        member = tarfile.TarInfo("bundle/receipt.json")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    compressed = tmp_path / "bundle.tar.zst"
    with compressed.open("wb") as output:
        subprocess.run(
            ["zstd", "--compress", "--stdout", str(source)],
            check=True,
            stdout=output,
        )

    expanded = numerical_runner._materialize_tar_archive(
        compressed,
        tmp_path / "expanded.tar",
        max_expanded_archive_bytes=20_000,
    )
    members, top_level = numerical_runner._inspect_relocation_archive(
        expanded,
        max_members=10,
        max_member_name_bytes=1024,
        max_member_bytes=1024,
        max_total_uncompressed_bytes=2048,
    )
    destination = tmp_path / "extracted"
    destination.mkdir()
    numerical_runner._extract_relocation_archive(expanded, destination)

    assert members == ("bundle/receipt.json",)
    assert top_level == "bundle"
    assert (destination / "bundle" / "receipt.json").read_bytes() == payload


def test_relocation_attestation_rejects_pending_contract_before_archive_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_seqax_bf16_validation_contract().model_copy(
        update={"hlo_identity_status": "pending"}
    )
    monkeypatch.setattr(
        numerical_runner,
        "default_seqax_bf16_validation_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        numerical_runner.shutil,
        "copyfile",
        lambda *_args, **_kwargs: pytest.fail("archive must not be copied"),
    )

    with pytest.raises(ValueError, match="HLO_IDENTITIES_PENDING"):
        numerical_runner._relocation_attestation(
            tmp_path / "missing.tar",
            contract,
        )


def test_relocation_attestation_rejects_a_symlinked_output_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    (target / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        numerical_runner,
        "_relocation_attestation",
        lambda *_args: pytest.fail("attestation evaluation must not run"),
    )

    with pytest.raises(ValueError, match="ATTESTATION_PARENT_INVALID"):
        numerical_runner.write_seqax_bf16_relocation_attestation(
            alias / "nested" / "attestation.json",
            archive=tmp_path / "unused.tar",
            contract=default_seqax_bf16_validation_contract(),
        )


def test_relocation_runtime_requires_exact_software_and_distinct_architecture() -> None:
    contract = default_seqax_bf16_validation_contract()
    expected = contract.runtime
    producer = numerical_runner.SeqaxBf16Runtime(
        runtime=numerical_runner.RuntimeIdentity(
            python=expected.python_major_minor + ".3",
            jax=expected.jax,
            jaxlib=expected.jaxlib,
            libtpu=expected.libtpu,
            xla=expected.libtpu_init_args,
        ),
        ml_dtypes=expected.ml_dtypes,
        cpu_machine=expected.cpu_machine,
        cpu_system=expected.cpu_system,
    )
    portable = numerical_runner.SeqaxBf16RelocationRuntime(
        python=expected.python_major_minor + ".3",
        jax=expected.jax,
        jaxlib=expected.jaxlib,
        ml_dtypes=expected.ml_dtypes,
        machine="arm64",
        system="Darwin",
    )

    _require_relocation_runtime(portable, contract, producer)
    with pytest.raises(ValueError, match="RUNTIME_MISMATCH"):
        _require_relocation_runtime(
            portable.model_copy(update={"jax": "0.11.1"}),
            contract,
            producer,
        )
    with pytest.raises(ValueError, match="ARCHITECTURE_NOT_DISTINCT"):
        _require_relocation_runtime(
            portable.model_copy(
                update={"machine": expected.cpu_machine, "system": expected.cpu_system}
            ),
            contract,
            producer,
        )


def test_runner_requires_the_contract_compilation_source_root(tmp_path: Path) -> None:
    contract = default_seqax_bf16_validation_contract()

    with pytest.raises(ValueError, match="SEQAX_BF16_COMPILATION_SOURCE_ROOT_MISMATCH"):
        _require_compilation_source_root(tmp_path, contract)


def test_runner_rejects_a_wrong_compilation_root_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_seqax_bf16_validation_contract()
    root = tmp_path / "run"
    monkeypatch.setattr(numerical_runner, "_require_clean_repository", lambda _root: None)

    with pytest.raises(ValueError, match="SEQAX_BF16_COMPILATION_SOURCE_ROOT_MISMATCH"):
        numerical_runner.run_seqax_bf16_validation(root, contract)

    assert not root.exists()


def test_runner_refuses_pending_hlo_identities_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_seqax_bf16_validation_contract().model_copy(
        update={"hlo_identity_status": "pending"}
    )
    root = tmp_path / "run"
    monkeypatch.setattr(
        numerical_runner,
        "default_seqax_bf16_validation_contract",
        lambda: contract,
    )
    monkeypatch.setattr(numerical_runner, "_require_clean_repository", lambda _root: None)
    monkeypatch.setattr(
        numerical_runner,
        "_require_compilation_source_root",
        lambda _root, _contract: None,
    )

    with pytest.raises(ValueError, match="SEQAX_BF16_HLO_IDENTITIES_PENDING"):
        numerical_runner.run_seqax_bf16_validation(root, contract)

    assert not root.exists()


def test_runner_rejects_a_caller_demoted_pinned_contract_before_writes(
    tmp_path: Path,
) -> None:
    contract = default_seqax_bf16_validation_contract().model_copy(
        update={"hlo_identity_status": "pending"}
    )
    root = tmp_path / "run"

    with pytest.raises(ValueError, match="SEQAX_BF16_EXTERNAL_CONTRACT_MISMATCH"):
        numerical_runner.run_seqax_bf16_validation(root, contract)

    assert not root.exists()


def test_bf16_runner_builds_and_replays_a_relocated_receipt() -> None:
    if numerical_runner.shutil.which("zstd") is None:
        pytest.skip("zstd is required for relocation lifecycle verification")
    script = r"""
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

import tpu_cake.seqax_numerical as numerical
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
    stablehlo, output = original_activation_mutant(
        path,
        inputs,
        devices,
        pallas=pallas,
        relu=relu,
        interpret_pallas=pallas,
    )
    if not pallas:
        return stablehlo, output
    structural_path = runner._CompiledPath(
        plan=control_plan(path.plan),
        executable=path.executable,
        mesh=path.mesh,
        stablehlo=path.stablehlo,
        compiler_hlo=path.compiler_hlo,
    )
    structural_hlo, _ = original_activation_mutant(
        structural_path,
        inputs,
        devices,
        pallas=False,
        relu=relu,
    )
    return structural_hlo, output


def execute_outputs(executable, inputs):
    outputs = list(original_execute_outputs(executable, inputs))
    if len(outputs) > 1:
        outputs[0] = outputs[0] + np.float32(1e-5)
        for checkpoint_index in range(1, len(outputs), 13):
            outputs[checkpoint_index + 7] = rounded_mathematical_silu_bf16(
                outputs[checkpoint_index + 6]
            )
            outputs[checkpoint_index + 10] = np.asarray(
                outputs[checkpoint_index + 7].astype(np.float32)
                * outputs[checkpoint_index + 9].astype(np.float32),
                dtype=outputs[checkpoint_index + 10].dtype,
            )
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
        cpu_machine=expected.cpu_machine,
        cpu_system=expected.cpu_system,
    )


def relocation_runtime():
    return runner.SeqaxBf16RelocationRuntime(
        python="3.12.3",
        jax="0.11.0",
        jaxlib="0.11.0",
        ml_dtypes="0.6.0",
        machine="arm64",
        system="Darwin",
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
runner._relocation_runtime = relocation_runtime
runner._device_inventory = device_inventory
runner._require_clean_repository = lambda root: None
runner._require_compilation_source_root = lambda root, contract: None
runner._validate_compiled_program = lambda *args, **kwargs: None
numerical._require_stablehlo_identity = lambda *args, **kwargs: None

contract = runner.default_seqax_bf16_validation_contract()
contract = contract.model_copy(update={"hlo_identity_status": "pinned"})
runner.default_seqax_bf16_validation_contract = lambda: contract
temporary = Path(tempfile.mkdtemp(prefix="seqax-bf16-lifecycle-")).resolve()
try:
    root = temporary / "run"
    result = runner.run_seqax_bf16_validation(root, contract)
    assert len(result.observations) == 53
    assert result.producer_passed
    assert result.claim_scope == "producer-host-bf16-validation-only-v1"
    assert len(result.discriminators) == len(tuple(numerical.SeqaxNumericalDiscriminator))
    assert not any(value.instrumentation_difference.exact for value in result.observations)
    assert runner.run_seqax_bf16_validation(root, contract) == result
    relocated = temporary / "relocated"
    shutil.copytree(root, relocated)
    runner.validate_seqax_bf16_validation(relocated, contract)
    archive_tar = temporary / "bundle.tar"
    with tarfile.open(archive_tar, "w") as bundle:
        bundle.add(root, arcname="bundle")
    archive = temporary / "bundle.tar.zst"
    with archive.open("wb") as output:
        subprocess.run(
            ["zstd", "--compress", "--stdout", str(archive_tar)],
            check=True,
            stdout=output,
        )
    attestation_path = temporary / "relocation-attestation.json"
    attestation = runner.write_seqax_bf16_relocation_attestation(
        attestation_path,
        archive=archive,
        contract=contract,
    )
    assert attestation.status == "portable_accepted"
    assert attestation.claim_scope == "declared-surface-dual-jax-cpu-bf16-agreement-v2"
    assert len(attestation.observations) == 53
    assert (
        runner.validate_seqax_bf16_relocation_attestation(
            attestation_path,
            archive=archive,
            contract=contract,
        )
        == attestation
    )

    mutated_attestation_path = temporary / "mutated-attestation.json"
    mutated_attestation = json.loads(attestation_path.read_text())
    mutated_attestation["observations"][0]["fresh_cpu_sha256"] = "0" * 64
    mutated_attestation_path.write_text(
        json.dumps(mutated_attestation, indent=2, sort_keys=True) + "\n"
    )
    try:
        runner.validate_seqax_bf16_relocation_attestation(
            mutated_attestation_path,
            archive=archive,
            contract=contract,
        )
    except ValueError as error:
        assert "SEQAX_BF16_RELOCATION_ATTESTATION_MISMATCH" in str(error)
    else:
        raise AssertionError("mutated relocation attestation was accepted")

    changed_archive_tar = temporary / "changed-bundle.tar"
    with tarfile.open(changed_archive_tar, "w") as bundle:
        bundle.add(root, arcname="bundle2")
    changed_archive = temporary / "changed-bundle.tar.zst"
    with changed_archive.open("wb") as output:
        subprocess.run(
            ["zstd", "--compress", "--stdout", str(changed_archive_tar)],
            check=True,
            stdout=output,
        )
    try:
        runner.validate_seqax_bf16_relocation_attestation(
            attestation_path,
            archive=changed_archive,
            contract=contract,
        )
    except ValueError as error:
        assert "SEQAX_BF16_RELOCATION_ATTESTATION_MISMATCH" in str(error)
    else:
        raise AssertionError("relocation attestation accepted a changed archive")

    original_runner_cpu_reference = runner.seqax_forward_canonical_reference
    original_numerical_cpu_reference = numerical.seqax_forward_canonical_reference

    def replay_cpu_reference(*args, delta, **kwargs):
        value = original_runner_cpu_reference(*args, **kwargs).copy()
        value.flat[0] += np.float32(delta)
        return value

    runner.seqax_forward_canonical_reference = lambda *args, **kwargs: replay_cpu_reference(
        *args, delta=1e-4, **kwargs
    )
    numerical.seqax_forward_canonical_reference = runner.seqax_forward_canonical_reference
    runner.validate_seqax_bf16_validation(relocated, contract)
    runner.seqax_forward_canonical_reference = lambda *args, **kwargs: replay_cpu_reference(
        *args, delta=0.1, **kwargs
    )
    numerical.seqax_forward_canonical_reference = runner.seqax_forward_canonical_reference
    require_rejected(relocated, contract)
    runner.seqax_forward_canonical_reference = original_runner_cpu_reference
    numerical.seqax_forward_canonical_reference = original_numerical_cpu_reference

    one_reference_mutant = temporary / "one-reference-mutant"
    shutil.copytree(root, one_reference_mutant)
    scenario = contract.scenarios[0]
    seed = scenario.seeds[0]
    seed_root = one_reference_mutant / "scenarios" / scenario.name / f"seed-{seed}"
    saved_cpu = np.load(seed_root / "cpu_reference.npy")
    one_reference_output = saved_cpu.copy()
    one_reference_output.flat[0] += np.float32(0.03)
    np.save(seed_root / "pallas_output.npy", one_reference_output, allow_pickle=False)
    np.save(seed_root / "control_output.npy", one_reference_output, allow_pickle=False)
    observation_path = seed_root / "observation.json"
    observation = json.loads(observation_path.read_text())
    observation["pallas_output_sha256"] = runner.array_sha256(one_reference_output)
    observation["control_output_sha256"] = runner.array_sha256(one_reference_output)
    observation["normal_assessment"] = numerical._assess_output_arrays(
        one_reference_output,
        one_reference_output,
        saved_cpu,
        policy=contract.policy,
        scenario=scenario,
    ).model_dump(mode="json")
    observation_path.write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n")

    def one_reference_cpu(*args, **kwargs):
        value = original_runner_cpu_reference(*args, **kwargs).copy()
        value.flat[0] -= np.float32(0.007)
        return value

    runner.seqax_forward_canonical_reference = one_reference_cpu
    numerical.seqax_forward_canonical_reference = one_reference_cpu
    try:
        runner._validate_seed(one_reference_mutant, contract, scenario, seed)
    except ValueError as error:
        assert "SEQAX_BF16_OBSERVATION_REPLAY_MISMATCH" in str(error)
    else:
        raise AssertionError("TPU output passing only the saved CPU reference was accepted")
    runner.seqax_forward_canonical_reference = original_runner_cpu_reference
    numerical.seqax_forward_canonical_reference = original_numerical_cpu_reference

    contract_mutant = temporary / "contract-mutant"
    shutil.copytree(root, contract_mutant)
    payload = json.loads((contract_mutant / "contract.json").read_text())
    payload["policy"]["cpu_relative_l2_units"] = 3.1
    (contract_mutant / "contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    repair_receipt(contract_mutant, "contract.json")
    require_rejected(contract_mutant, contract)

    result_mutant = temporary / "result-mutant"
    shutil.copytree(root, result_mutant)
    payload = json.loads((result_mutant / "result.json").read_text())
    payload["producer_passed"] = False
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
