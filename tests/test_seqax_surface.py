from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dtensor_interpreter import interpret_distributed_program
from tpu_cake.identity import semantic_seed, semantic_sha256
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.ledger import RunState
from tpu_cake.metrics import MetricSource
from tpu_cake.runner import _record_event
from tpu_cake.seqax_cost_model import estimate_seqax_forward
from tpu_cake.seqax_surface import (
    SEQAX_SURFACE_BASELINE,
    SEQAX_SURFACE_CANDIDATE,
    SEQAX_SURFACE_MEASURED_ITERATIONS,
    SEQAX_SURFACE_ROUNDS,
    SEQAX_SURFACE_SCHEMA,
    SEQAX_SURFACE_WARMUP_ITERATIONS,
    SeqaxSurfaceInvocation,
    SeqaxSurfaceReceipt,
    _array_tuple_sha256,
    _artifact,
    _artifact_roles,
    _plan_payload,
    _runtime_sha256,
    _save_array,
    _strategy_source,
    _write_json,
    _write_nested_json,
    _write_text,
    run_seqax_surface,
    seqax_forward_workload_surface,
    validate_seqax_surface_receipt,
)
from tpu_cake.surfaces import (
    ScenarioObservation,
    SurfaceCandidateObservation,
    compare_surface_candidates,
)
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)


def _receipt(tmp_path: Path, monkeypatch) -> SeqaxSurfaceReceipt:
    source_commit = "a" * 40
    monkeypatch.setattr(
        "tpu_cake.seqax_surface._source_identity",
        lambda *_args, **_kwargs: (source_commit, "b" * 64),
    )
    surface = seqax_forward_workload_surface()
    runtime = RuntimeIdentity(
        python="3.12",
        jax="0.11.0",
        jaxlib="0.11.0",
        libtpu="0.0.44.1",
        xla="test",
    )
    runtime_sha256 = _runtime_sha256(runtime, "TPU7x", 8)
    run_id = semantic_sha256(
        SEQAX_SURFACE_SCHEMA,
        surface.surface_id,
        source_commit,
        runtime_sha256,
    )
    baseline_scenarios = []
    candidate_scenarios = []
    artifacts = [
        _write_json(
            tmp_path / "surface.json",
            surface.model_dump(mode="json", exclude={"surface_id"}),
            ArtifactRole.SEARCH_CONTRACT,
        ),
        _write_json(
            tmp_path / "plans.json",
            _plan_payload(surface),
            ArtifactRole.PLAN_MANIFEST,
        ),
        _write_json(tmp_path / "source_state.json", {}, ArtifactRole.SOURCE_STATE),
        _write_json(tmp_path / "source_diff.patch", {}, ArtifactRole.SOURCE_DIFF),
        _write_text(
            tmp_path,
            Path("strategies.py"),
            _strategy_source(),
            ArtifactRole.JAX_SOURCE,
        ),
    ]
    for scenario in surface.scenarios:
        module = seqax_forward_schedule(**scenario.parameters())
        plan = lower_distributed_program_to_jax_mesh(module)
        ir_artifact = _write_text(
            tmp_path,
            Path("ir") / f"{scenario.name}.xdsl",
            canonical_text(module),
            ArtifactRole.DISTRIBUTED_IR,
        )
        artifacts.extend(
            (
                ir_artifact,
                _write_text(
                    tmp_path,
                    Path("lowering") / f"{scenario.name}.py",
                    plan.render_executable_source(),
                    ArtifactRole.JAX_SOURCE,
                ),
                _write_nested_json(
                    tmp_path,
                    Path("cost") / f"{scenario.name}.json",
                    estimate_seqax_forward(
                        module,
                        hardware=tpu7x_tensorcore_rates(),
                        source=MetricSource(
                            artifact_sha256=ir_artifact.sha256,
                            artifact_path=ir_artifact.path,
                            tool="tpu-cake",
                            field="seqax-surface-distributed-ir",
                        ),
                        expected_schedule_sha256=plan.schedule_sha256,
                    ).model_dump(mode="json"),
                    ArtifactRole.COST_MODEL,
                ),
                _write_text(
                    tmp_path,
                    Path("hlo") / scenario.name / "candidate_stablehlo.txt",
                    "stablehlo.all_gather stablehlo.reduce_scatter\n",
                    ArtifactRole.STABLEHLO,
                ),
                _write_text(
                    tmp_path,
                    Path("hlo") / scenario.name / "candidate_compiler_hlo.txt",
                    "all-gather reduce-scatter dot\n",
                    ArtifactRole.COMPILER_HLO,
                ),
            )
        )
        seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
        inputs = tuple(
            np.asarray(value)
            for value in seqax_forward_inputs(seed=seed, **scenario.parameters())
        )
        output = seqax_forward_canonical_reference(inputs, **scenario.parameters())
        artifacts.extend(
            _save_array(
                tmp_path,
                Path("inputs") / scenario.name / f"{index:02d}.npy",
                value,
                ArtifactRole.CORRECTNESS_INPUT,
            )
            for index, value in enumerate(inputs)
        )
        artifacts.extend(
            (
                _save_array(
                    tmp_path,
                    Path("oracle") / f"{scenario.name}.npy",
                    output,
                    ArtifactRole.ORACLE_OUTPUT,
                ),
                _save_array(
                    tmp_path,
                    Path("outputs") / scenario.name / "baseline.npy",
                    output,
                    ArtifactRole.CORRECTNESS_OUTPUT,
                ),
                _save_array(
                    tmp_path,
                    Path("outputs") / scenario.name / "candidate.npy",
                    output,
                    ArtifactRole.CORRECTNESS_OUTPUT,
                ),
            )
        )
        common = {
            "scenario": scenario.name,
            "input_sha256": _array_tuple_sha256(inputs),
            "output_sha256": _array_tuple_sha256((output,)),
            "runtime_sha256": runtime_sha256,
            "profiled": False,
            "passed": True,
        }
        baseline_medians = tuple(
            100 + index for index in range(SEQAX_SURFACE_ROUNDS)
        )
        candidate_medians = tuple(
            80 + index for index in range(SEQAX_SURFACE_ROUNDS)
        )
        baseline_scenarios.append(
            ScenarioObservation(
                round_medians_ns=baseline_medians,
                round_samples_ns=tuple(
                    (value - 1, value, value + 1) for value in baseline_medians
                ),
                ran_first=tuple(index % 2 == 0 for index in range(SEQAX_SURFACE_ROUNDS)),
                **common,
            )
        )
        candidate_scenarios.append(
            ScenarioObservation(
                round_medians_ns=candidate_medians,
                round_samples_ns=tuple(
                    (value - 1, value, value + 1) for value in candidate_medians
                ),
                ran_first=tuple(index % 2 == 1 for index in range(SEQAX_SURFACE_ROUNDS)),
                **common,
            )
        )
    baseline = SurfaceCandidateObservation(
        candidate=SEQAX_SURFACE_BASELINE,
        scenarios=tuple(baseline_scenarios),
    )
    candidate = SurfaceCandidateObservation(
        candidate=SEQAX_SURFACE_CANDIDATE,
        scenarios=tuple(candidate_scenarios),
    )
    comparison = compare_surface_candidates(surface, baseline, candidate)
    invocation = SeqaxSurfaceInvocation(
        schema_version=SEQAX_SURFACE_SCHEMA,
        surface_id=surface.surface_id,
        baseline=SEQAX_SURFACE_BASELINE,
        candidate=SEQAX_SURFACE_CANDIDATE,
        rounds=SEQAX_SURFACE_ROUNDS,
        warmup_iterations=SEQAX_SURFACE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_SURFACE_MEASURED_ITERATIONS,
        runtime=runtime,
        runtime_sha256=runtime_sha256,
        device_kind="TPU7x",
        device_count=8,
        baseline_strategy="unwrapped-jax-shard-map",
        candidate_strategy="whole-program-jit-over-jax-shard-map",
        input_placement="resident-named-sharding-before-warmup",
        timing_scope="synchronized-resident-input-complete-forward",
        comparison_kind="wrapper-dispatch-control",
        run_id=run_id,
    )
    artifacts.extend(
        (
            _write_json(
                tmp_path / "invocation.json",
                invocation.model_dump(mode="json"),
                ArtifactRole.INVOCATION,
            ),
            _write_json(
                tmp_path / "baseline.json",
                baseline.model_dump(mode="json"),
                ArtifactRole.SEARCH_EVIDENCE,
            ),
            _write_json(
                tmp_path / "candidate.json",
                candidate.model_dump(mode="json"),
                ArtifactRole.SEARCH_EVIDENCE,
            ),
            _write_json(
                tmp_path / "comparison.json",
                comparison.model_dump(mode="json"),
                ArtifactRole.SEARCH_RESULT,
            ),
        )
    )
    by_path = {artifact.path: artifact for artifact in artifacts}
    ledger_path = tmp_path / "ledger.sqlite"
    ledger_events = (
        (RunState.CREATED, invocation.model_dump(mode="json")),
        (
            RunState.VERIFIED,
            {
                "surface_id": surface.surface_id,
                "schedule_sha256": {
                    scenario.name: lower_distributed_program_to_jax_mesh(
                        seqax_forward_schedule(**scenario.parameters())
                    ).schedule_sha256
                    for scenario in surface.scenarios
                },
            },
        ),
        (
            RunState.LOWERED,
            {
                "plans_sha256": by_path["plans.json"].sha256,
                "strategy_sha256": by_path["strategies.py"].sha256,
            },
        ),
        (
            RunState.COMPILED,
            {
                path: artifact.sha256
                for path, artifact in by_path.items()
                if artifact.role in {ArtifactRole.STABLEHLO, ArtifactRole.COMPILER_HLO}
            },
        ),
        (
            RunState.CORRECT,
            {
                "absolute_tolerance": 0.016,
                "relative_tolerance": 0.05,
                "scenarios": tuple(scenario.name for scenario in surface.scenarios),
            },
        ),
        (
            RunState.TIMED,
            {
                "rounds": SEQAX_SURFACE_ROUNDS,
                "samples_per_round": SEQAX_SURFACE_MEASURED_ITERATIONS,
                "timing_scope": invocation.timing_scope,
            },
        ),
        (
            RunState.ACCEPTED,
            {
                "comparison_sha256": by_path["comparison.json"].sha256,
                "candidate_promoted": comparison.promotable,
            },
        ),
    )
    for state, payload in ledger_events:
        _record_event(ledger_path, run_id, state, payload)
    artifacts.append(_artifact(tmp_path, ledger_path, ArtifactRole.EXECUTION_LEDGER))
    receipt = SeqaxSurfaceReceipt(
        schema_version=SEQAX_SURFACE_SCHEMA,
        surface_id=surface.surface_id,
        invocation=invocation,
        comparison=comparison,
        candidate_promoted=comparison.promotable,
        artifacts=tuple(sorted(artifacts, key=lambda artifact: artifact.path)),
    )
    (tmp_path / "receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    return receipt


def _replace_artifact(
    receipt: SeqaxSurfaceReceipt,
    root: Path,
    path: str,
) -> SeqaxSurfaceReceipt:
    expected_role = _artifact_roles(seqax_forward_workload_surface())[path]
    replacement = ArtifactReference(
        path=path,
        size_bytes=(root / path).stat().st_size,
        sha256=hashlib.sha256((root / path).read_bytes()).hexdigest(),
        role=expected_role,
    )
    return receipt.model_copy(
        update={
            "artifacts": tuple(
                replacement if artifact.path == path else artifact
                for artifact in receipt.artifacts
            )
        }
    )


def test_seqax_surface_receipt_replays_complete_evidence(tmp_path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch)

    validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_runner_resumes_a_complete_validated_receipt(
    tmp_path, monkeypatch
) -> None:
    receipt = _receipt(tmp_path, monkeypatch)

    assert run_seqax_surface(tmp_path) == receipt


def test_seqax_surface_runner_does_not_move_an_unrelated_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.Path.home",
        lambda: tmp_path.parent / "unrelated-home",
    )
    path = tmp_path / "unrelated"
    path.mkdir()
    valuable = path / "valuable.txt"
    valuable.write_text("keep me")

    with pytest.raises(ValueError, match="OUTPUT_NOT_OWNED"):
        run_seqax_surface(path)

    assert valuable.read_text() == "keep me"
    assert not tuple(tmp_path.glob("unrelated.incomplete-*"))


def test_seqax_surface_runner_constructs_a_new_evidence_tree(
    tmp_path, monkeypatch
) -> None:
    class FakeDevice:
        device_kind = "TPU7x"

    class FakeHlo:
        def as_hlo_text(self) -> str:
            return "all-gather reduce-scatter dot"

    class FakeLowered:
        def __init__(self, function):
            self.function = function

        def compiler_ir(self, dialect):
            if dialect == "stablehlo":
                return "stablehlo.all_gather stablehlo.reduce_scatter"
            return FakeHlo()

        def compile(self):
            return self.function

    class FakeJitted:
        def __init__(self, function):
            self.function = function

        def __call__(self, *inputs):
            return self.function(*inputs)

        def lower(self, *inputs):
            del inputs
            return FakeLowered(self.function)

    class FakeModule:
        def __init__(self, identity):
            self.identity = identity

        def verify(self):
            return None

    class FakePlan:
        input_contracts = (object(),)
        input_partition_specs = (None,)

        def __init__(self, identity):
            self.schedule_sha256 = semantic_sha256("fake-plan", identity)

        def build_mapped(self, *, devices):
            del devices

            def execute(*inputs):
                return (inputs[0],)

            return execute, object()

        def render_executable_source(self):
            return f"SCHEDULE = {self.schedule_sha256!r}\n"

        def source_sha256(self):
            return hashlib.sha256(self.render_executable_source().encode()).hexdigest()

        def manifest(self):
            return {"schedule_sha256": self.schedule_sha256}

    class FakeCost:
        def __init__(self, schedule_sha256):
            self.schedule_sha256 = schedule_sha256

        def model_dump(self, *, mode):
            assert mode == "json"
            return {"schedule_sha256": self.schedule_sha256}

        def __eq__(self, other):
            return (
                isinstance(other, FakeCost)
                and self.schedule_sha256 == other.schedule_sha256
            )

    class FakeCostModel:
        @classmethod
        def model_validate_json(cls, value):
            return FakeCost(json.loads(value)["schedule_sha256"])

    def fake_source_state(_root, output):
        return (
            _write_json(
                output / "source_state.json",
                {
                    "git_dirty": False,
                    "git_commit": "a" * 40,
                    "uv_lock_sha256": "b" * 64,
                },
                ArtifactRole.SOURCE_STATE,
            ),
            _write_json(
                output / "source_diff.patch",
                {},
                ArtifactRole.SOURCE_DIFF,
            ),
        )

    monkeypatch.setattr("tpu_cake.seqax_surface.jax.devices", lambda: (FakeDevice(),) * 8)
    monkeypatch.setattr("tpu_cake.seqax_surface.jax.jit", FakeJitted)
    monkeypatch.setattr("tpu_cake.seqax_surface.jax.device_put", lambda value, _sharding: value)
    monkeypatch.setattr("tpu_cake.seqax_surface.jax.device_get", lambda value: value)
    monkeypatch.setattr("tpu_cake.seqax_surface.jax.block_until_ready", lambda value: value)
    monkeypatch.setattr("tpu_cake.seqax_surface.NamedSharding", lambda _mesh, _spec: None)
    monkeypatch.setattr("tpu_cake.seqax_surface._source_state", fake_source_state)
    monkeypatch.setattr(
        "tpu_cake.seqax_surface._source_identity",
        lambda *_args, **_kwargs: ("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface._runtime_identity",
        lambda: RuntimeIdentity(
            python="3.12",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla="test",
        ),
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.seqax_forward_schedule",
        lambda **parameters: FakeModule(json.dumps(parameters, sort_keys=True)),
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.lower_distributed_program_to_jax_mesh",
        lambda module: FakePlan(module.identity),
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.canonical_text",
        lambda module: module.identity,
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.estimate_seqax_forward",
        lambda module, **_kwargs: FakeCost(semantic_sha256("fake-plan", module.identity)),
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.seqax_forward_inputs",
        lambda *, seed, **parameters: (
            np.full(
                (parameters["batch"], parameters["sequence"]),
                seed % 17,
                dtype=np.float32,
            ),
        ),
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.seqax_forward_canonical_reference",
        lambda inputs, **_parameters: np.asarray(inputs[0]),
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_surface.SeqaxCostModelReport",
        FakeCostModel,
    )

    receipt = run_seqax_surface(tmp_path / "new-run")

    assert receipt.invocation.comparison_kind == "wrapper-dispatch-control"
    assert (tmp_path / "new-run/ledger.sqlite").is_file()
    assert (tmp_path / "new-run/hlo/tiny/candidate_stablehlo.txt").is_file()
    assert (tmp_path / "new-run/cost/deeper.json").is_file()
    assert (tmp_path / "new-run/receipt.json").is_file()


def test_seqax_surface_rejects_coordinated_round_count_change(tmp_path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    path = tmp_path / "baseline.json"
    baseline = json.loads(path.read_text())
    baseline["scenarios"][0]["round_medians_ns"] = baseline["scenarios"][0][
        "round_medians_ns"
    ][:-1]
    baseline["scenarios"][0]["round_samples_ns"] = baseline["scenarios"][0][
        "round_samples_ns"
    ][:-1]
    baseline["scenarios"][0]["ran_first"] = baseline["scenarios"][0]["ran_first"][:-1]
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    receipt = _replace_artifact(receipt, tmp_path, "baseline.json")

    with pytest.raises(ValueError, match="OBSERVATION_PROTOCOL_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_rejects_dtype_only_input_change(tmp_path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    path = tmp_path / "inputs/tiny/00.npy"
    np.save(path, np.load(path).astype(np.int32), allow_pickle=False)
    receipt = _replace_artifact(receipt, tmp_path, "inputs/tiny/00.npy")

    with pytest.raises(ValueError, match="DETERMINISTIC_INPUT_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_rejects_hlo_without_physical_collectives(
    tmp_path, monkeypatch
) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    path = tmp_path / "hlo/tiny/candidate_stablehlo.txt"
    path.write_text("stablehlo.add\n")
    receipt = _replace_artifact(
        receipt,
        tmp_path,
        "hlo/tiny/candidate_stablehlo.txt",
    )

    with pytest.raises(ValueError, match="HLO_MARKER_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_rejects_a_coordinated_cost_edit(tmp_path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    path = tmp_path / "cost/tiny.json"
    cost = json.loads(path.read_text())
    cost["predicted_limiting_resource"] = "fabricated"
    path.write_text(json.dumps(cost, indent=2, sort_keys=True) + "\n")
    receipt = _replace_artifact(receipt, tmp_path, "cost/tiny.json")

    with pytest.raises(ValueError, match="COST_REPLAY_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_rejects_a_coordinated_ledger_edit(tmp_path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    path = tmp_path / "ledger.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE events SET payload_sha256 = ? WHERE state = ?",
            ("f" * 64, RunState.ACCEPTED.value),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for suffix in ("-shm", "-wal"):
        sidecar = path.with_name(f"{path.name}{suffix}")
        if sidecar.exists():
            sidecar.unlink()
    receipt = _replace_artifact(receipt, tmp_path, "ledger.sqlite")

    with pytest.raises(ValueError, match="LEDGER_REPLAY_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_tolerance_still_rejects_swapped_mlp_weights() -> None:
    surface = seqax_forward_workload_surface()
    scenario = next(value for value in surface.scenarios if value.name == "deeper")
    seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
    inputs = seqax_forward_inputs(seed=seed, **scenario.parameters())
    expected = seqax_forward_canonical_reference(inputs, **scenario.parameters())
    swapped = (*inputs[:3], inputs[4], inputs[3], *inputs[5:])
    (wrong,) = interpret_distributed_program(
        seqax_forward_schedule(**scenario.parameters()),
        swapped,
    )

    assert not np.allclose(wrong, expected, atol=0.016, rtol=0.05)


def test_seqax_surface_accepts_valid_cross_mode_rounding(tmp_path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    output_path = tmp_path / "outputs/tiny/candidate.npy"
    output = np.load(output_path, allow_pickle=False)
    rounded = (output + np.float32(0.001)).astype(output.dtype)
    np.save(output_path, rounded, allow_pickle=False)
    candidate_path = tmp_path / "candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["scenarios"][0]["output_sha256"] = _array_tuple_sha256((rounded,))
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    receipt = _replace_artifact(
        receipt,
        tmp_path,
        "outputs/tiny/candidate.npy",
    )
    receipt = _replace_artifact(receipt, tmp_path, "candidate.json")

    validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_rejects_cross_mode_error_outside_contract(
    tmp_path, monkeypatch
) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    output_path = tmp_path / "outputs/tiny/candidate.npy"
    output = np.load(output_path, allow_pickle=False)
    wrong = (output + np.float32(1)).astype(output.dtype)
    np.save(output_path, wrong, allow_pickle=False)
    candidate_path = tmp_path / "candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["scenarios"][0]["output_sha256"] = _array_tuple_sha256((wrong,))
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    receipt = _replace_artifact(
        receipt,
        tmp_path,
        "outputs/tiny/candidate.npy",
    )
    receipt = _replace_artifact(receipt, tmp_path, "candidate.json")

    with pytest.raises(ValueError, match="OUTPUT_ORACLE_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_rejects_candidate_output_hash_mismatch(
    tmp_path, monkeypatch
) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    candidate_path = tmp_path / "candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["scenarios"][0]["output_sha256"] = "f" * 64
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    receipt = _replace_artifact(receipt, tmp_path, "candidate.json")

    with pytest.raises(ValueError, match="ARRAY_IDENTITY_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)


def test_seqax_surface_rejects_cross_mode_divergence_within_oracle_bands(
    tmp_path, monkeypatch
) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    baseline_path = tmp_path / "outputs/tiny/baseline.npy"
    candidate_path = tmp_path / "outputs/tiny/candidate.npy"
    oracle = np.load(tmp_path / "oracle/tiny.npy", allow_pickle=False)
    baseline = (oracle - np.float32(0.015)).astype(oracle.dtype)
    candidate = (oracle + np.float32(0.015)).astype(oracle.dtype)
    np.save(baseline_path, baseline, allow_pickle=False)
    np.save(candidate_path, candidate, allow_pickle=False)
    baseline_json_path = tmp_path / "baseline.json"
    candidate_json_path = tmp_path / "candidate.json"
    baseline_json = json.loads(baseline_json_path.read_text())
    candidate_json = json.loads(candidate_json_path.read_text())
    baseline_json["scenarios"][0]["output_sha256"] = _array_tuple_sha256(
        (baseline,)
    )
    candidate_json["scenarios"][0]["output_sha256"] = _array_tuple_sha256(
        (candidate,)
    )
    baseline_json_path.write_text(
        json.dumps(baseline_json, indent=2, sort_keys=True) + "\n"
    )
    candidate_json_path.write_text(
        json.dumps(candidate_json, indent=2, sort_keys=True) + "\n"
    )
    for path in (
        "outputs/tiny/baseline.npy",
        "outputs/tiny/candidate.npy",
        "baseline.json",
        "candidate.json",
    ):
        receipt = _replace_artifact(receipt, tmp_path, path)

    with pytest.raises(ValueError, match="CROSS_MODE_MISMATCH"):
        validate_seqax_surface_receipt(receipt, root=tmp_path)
