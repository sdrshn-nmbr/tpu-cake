from __future__ import annotations

import hashlib
import inspect
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.sharding import NamedSharding
from pydantic import BaseModel, ConfigDict, Field

from tpu_cake.artifacts import resolve_recorded_artifact
from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import array_sha256, semantic_seed, semantic_sha256
from tpu_cake.jax_lowering import lower_distributed_program_to_jax_mesh
from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history
from tpu_cake.metrics import MetricSource
from tpu_cake.receipt import _source_identity
from tpu_cake.runner import _record_event, _runtime_identity, _source_state
from tpu_cake.seqax_cost_model import SeqaxCostModelReport, estimate_seqax_forward
from tpu_cake.surface_runner import run_surface_pair
from tpu_cake.surfaces import (
    OutputEquivalencePolicy,
    SeqaxForwardScenario,
    SeqaxForwardWorkloadSurface,
    SurfaceCandidateObservation,
    SurfaceComparison,
    compare_surface_candidates,
)
from tpu_cake.workloads.seqax_forward import seqax_forward_schedule
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)

SEQAX_SURFACE_SCHEMA = "tpu-cake-seqax-surface-v1"
SEQAX_SURFACE_ROUNDS = 10
SEQAX_SURFACE_WARMUP_ITERATIONS = 2
SEQAX_SURFACE_MEASURED_ITERATIONS = 3
SEQAX_SURFACE_BASELINE = "eager-shard-map"
SEQAX_SURFACE_CANDIDATE = "whole-program-jit"
SEQAX_SURFACE_ATOL = 0.016
SEQAX_SURFACE_RTOL = 0.05


class SeqaxSurfaceInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^tpu-cake-seqax-surface-v1$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: str
    candidate: str
    rounds: int = Field(ge=5)
    warmup_iterations: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    runtime: RuntimeIdentity
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device_kind: str
    device_count: int = Field(gt=0)
    baseline_strategy: str
    candidate_strategy: str
    input_placement: str
    timing_scope: str
    comparison_kind: str
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class SeqaxSurfaceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^tpu-cake-seqax-surface-v1$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation: SeqaxSurfaceInvocation
    comparison: SurfaceComparison
    candidate_promoted: bool
    artifacts: tuple[ArtifactReference, ...]


def seqax_forward_workload_surface() -> SeqaxForwardWorkloadSurface:
    common = {
        "data_mesh": 2,
        "tensor_mesh": 4,
        "rope_max_timescale": 256,
    }
    return SeqaxForwardWorkloadSurface(
        name="seqax-complete-forward-compilation",
        scenarios=(
            SeqaxForwardScenario(
                name="tiny",
                batch=2,
                sequence=4,
                model=8,
                vocabulary=16,
                feed_forward=16,
                query_groups=2,
                key_value_heads=4,
                head=4,
                layers=2,
                weight=Decimal("0.25"),
                **common,
            ),
            SeqaxForwardScenario(
                name="wider",
                batch=4,
                sequence=8,
                model=16,
                vocabulary=32,
                feed_forward=32,
                query_groups=2,
                key_value_heads=4,
                head=4,
                layers=2,
                weight=Decimal("0.35"),
                **common,
            ),
            SeqaxForwardScenario(
                name="deeper",
                batch=4,
                sequence=8,
                model=16,
                vocabulary=32,
                feed_forward=32,
                query_groups=2,
                key_value_heads=4,
                head=4,
                layers=4,
                weight=Decimal("0.40"),
                **common,
            ),
        ),
        minimum_practical_improvement=Decimal("0.03"),
        maximum_scenario_regression=Decimal("0.01"),
        bootstrap_samples=10_000,
        output_equivalence=(
            OutputEquivalencePolicy.INDEPENDENT_ORACLE_AND_CROSS_MODE_TOLERANCE
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object, role: ArtifactRole) -> ArtifactReference:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return ArtifactReference(
        path=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
    )


def _artifact(root: Path, path: Path, role: ArtifactRole) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        role=role,
    )


def _write_text(
    root: Path,
    relative: Path,
    value: str,
    role: ArtifactRole,
) -> ArtifactReference:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return _artifact(root, path, role)


def _write_nested_json(
    root: Path,
    relative: Path,
    value: object,
    role: ArtifactRole,
) -> ArtifactReference:
    return _write_text(
        root,
        relative,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        role,
    )


def _save_array(
    root: Path,
    relative: Path,
    value: np.ndarray,
    role: ArtifactRole,
) -> ArtifactReference:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return _artifact(root, path, role)


def _array_tuple_sha256(values) -> str:
    return semantic_sha256(
        "array-tuple-v1",
        *(array_sha256(value) for value in values),
    )


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right)
    )


def _runtime_sha256(runtime: RuntimeIdentity, device_kind: str, device_count: int) -> str:
    return semantic_sha256(
        SEQAX_SURFACE_SCHEMA,
        json.dumps(runtime.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
        device_kind,
        str(device_count),
    )


def _plan_payload(surface: SeqaxForwardWorkloadSurface) -> dict[str, object]:
    payload: dict[str, object] = {}
    for scenario in surface.scenarios:
        plan = lower_distributed_program_to_jax_mesh(
            seqax_forward_schedule(**scenario.parameters())
        )
        payload[scenario.name] = {
            "schedule_sha256": plan.schedule_sha256,
            "source_sha256": plan.source_sha256(),
            "manifest": plan.manifest(),
        }
    return payload


def _build_surface_strategies(plan, *, devices):
    baseline, mesh = plan.build_mapped(devices=devices)
    candidate = jax.jit(baseline)
    return baseline, candidate, mesh


def _strategy_source() -> str:
    return "from __future__ import annotations\n\nimport jax\n\n\n" + inspect.getsource(
        _build_surface_strategies
    ).replace("_build_surface_strategies", "build", 1)


def _compiler_hlo(lowered: Any) -> str:
    computation = lowered.compiler_ir(dialect="hlo")
    return computation.as_hlo_text() if hasattr(computation, "as_hlo_text") else str(computation)


def _artifact_roles(surface: SeqaxForwardWorkloadSurface) -> dict[str, ArtifactRole]:
    roles = {
        "surface.json": ArtifactRole.SEARCH_CONTRACT,
        "invocation.json": ArtifactRole.INVOCATION,
        "plans.json": ArtifactRole.PLAN_MANIFEST,
        "strategies.py": ArtifactRole.JAX_SOURCE,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "baseline.json": ArtifactRole.SEARCH_EVIDENCE,
        "candidate.json": ArtifactRole.SEARCH_EVIDENCE,
        "comparison.json": ArtifactRole.SEARCH_RESULT,
        "source_diff.patch": ArtifactRole.SOURCE_DIFF,
        "source_state.json": ArtifactRole.SOURCE_STATE,
    }
    for scenario in surface.scenarios:
        input_count = len(
            lower_distributed_program_to_jax_mesh(
                seqax_forward_schedule(**scenario.parameters())
            ).input_contracts
        )
        for index in range(input_count):
            roles[f"inputs/{scenario.name}/{index:02d}.npy"] = (
                ArtifactRole.CORRECTNESS_INPUT
            )
        roles[f"oracle/{scenario.name}.npy"] = ArtifactRole.ORACLE_OUTPUT
        roles[f"outputs/{scenario.name}/baseline.npy"] = (
            ArtifactRole.CORRECTNESS_OUTPUT
        )
        roles[f"outputs/{scenario.name}/candidate.npy"] = (
            ArtifactRole.CORRECTNESS_OUTPUT
        )
        roles[f"ir/{scenario.name}.xdsl"] = ArtifactRole.DISTRIBUTED_IR
        roles[f"lowering/{scenario.name}.py"] = ArtifactRole.JAX_SOURCE
        roles[f"cost/{scenario.name}.json"] = ArtifactRole.COST_MODEL
        roles[f"hlo/{scenario.name}/candidate_stablehlo.txt"] = ArtifactRole.STABLEHLO
        roles[f"hlo/{scenario.name}/candidate_compiler_hlo.txt"] = (
            ArtifactRole.COMPILER_HLO
        )
    return roles


def run_seqax_surface(output_dir: Path) -> SeqaxSurfaceReceipt:
    output_dir = output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    protected = (Path("/").resolve(), Path.home().resolve(), repository_root)
    if any(output_dir == path or output_dir in path.parents for path in protected):
        raise ValueError(f"SEQAX_SURFACE_UNSAFE_OUTPUT_PATH path={output_dir}")
    receipt_path = output_dir / "receipt.json"
    if receipt_path.exists():
        receipt = SeqaxSurfaceReceipt.model_validate_json(receipt_path.read_text())
        validate_seqax_surface_receipt(receipt, root=output_dir)
        return receipt
    if output_dir.exists() and any(output_dir.iterdir()):
        required_markers = (
            output_dir / "invocation.json",
            output_dir / "surface.json",
            output_dir / "source_state.json",
        )
        if not all(path.is_file() for path in required_markers):
            raise ValueError(f"SEQAX_SURFACE_OUTPUT_NOT_OWNED path={output_dir}")
        partial_invocation = SeqaxSurfaceInvocation.model_validate_json(
            (output_dir / "invocation.json").read_text()
        )
        partial_surface = SeqaxForwardWorkloadSurface.model_validate_json(
            (output_dir / "surface.json").read_text()
        )
        if (
            partial_invocation.schema_version != SEQAX_SURFACE_SCHEMA
            or partial_invocation.surface_id != seqax_forward_workload_surface().surface_id
            or partial_surface != seqax_forward_workload_surface()
        ):
            raise ValueError(f"SEQAX_SURFACE_OUTPUT_NOT_OWNED path={output_dir}")
        archived = output_dir.with_name(
            f"{output_dir.name}.incomplete-{time.time_ns()}"
        )
        output_dir.rename(archived)
        print(f"SEQAX_SURFACE_ARCHIVED_INCOMPLETE source={output_dir} archive={archived}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_artifacts = _source_state(repository_root, output_dir)
    source_state = json.loads((output_dir / "source_state.json").read_text())
    if source_state["git_dirty"]:
        raise ValueError("SEQAX_SURFACE_SOURCE_MUST_BE_CLEAN")

    devices = tuple(jax.devices())
    device_kinds = {device.device_kind for device in devices}
    if len(devices) != 8 or device_kinds != {"TPU7x"}:
        raise ValueError(
            f"SEQAX_SURFACE_REQUIRES_TPU7X device_count={len(devices)} kinds={sorted(device_kinds)}"
        )
    surface = seqax_forward_workload_surface()
    runtime = _runtime_identity()
    runtime_sha256 = _runtime_sha256(runtime, "TPU7x", len(devices))
    run_id = semantic_sha256(
        SEQAX_SURFACE_SCHEMA,
        surface.surface_id,
        source_state["git_commit"],
        runtime_sha256,
    )
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
        device_count=len(devices),
        baseline_strategy="unwrapped-jax-shard-map",
        candidate_strategy="whole-program-jit-over-jax-shard-map",
        input_placement="resident-named-sharding-before-warmup",
        timing_scope="synchronized-resident-input-complete-forward",
        comparison_kind="wrapper-dispatch-control",
        run_id=run_id,
    )
    artifacts = [
        *source_artifacts,
        _write_json(
            output_dir / "surface.json",
            surface.model_dump(mode="json", exclude={"surface_id"}),
            ArtifactRole.SEARCH_CONTRACT,
        ),
        _write_json(
            output_dir / "invocation.json",
            invocation.model_dump(mode="json"),
            ArtifactRole.INVOCATION,
        ),
        _write_text(
            output_dir,
            Path("strategies.py"),
            _strategy_source(),
            ArtifactRole.JAX_SOURCE,
        ),
    ]
    ledger_path = output_dir / "ledger.sqlite"
    _record_event(
        ledger_path,
        run_id,
        RunState.CREATED,
        invocation.model_dump(mode="json"),
    )

    modules_by_scenario = {}
    plans_by_scenario = {}
    mapped_by_scenario = {}
    jitted_by_scenario = {}
    input_shardings_by_scenario = {}
    host_inputs_by_scenario = {}
    resident_inputs_by_scenario = {}
    for scenario in surface.scenarios:
        module = seqax_forward_schedule(**scenario.parameters())
        module.verify()
        plan = lower_distributed_program_to_jax_mesh(module)
        modules_by_scenario[scenario.name] = module
        plans_by_scenario[scenario.name] = plan
        mapped, jitted, mesh = _build_surface_strategies(plan, devices=devices)
        mapped_by_scenario[scenario.name] = mapped
        jitted_by_scenario[scenario.name] = jitted
        input_shardings_by_scenario[scenario.name] = tuple(
            NamedSharding(mesh, spec) for spec in plan.input_partition_specs
        )
        seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
        host_inputs = tuple(
            np.asarray(value)
            for value in seqax_forward_inputs(seed=seed, **scenario.parameters())
        )
        host_inputs_by_scenario[scenario.name] = host_inputs
        resident_inputs_by_scenario[scenario.name] = tuple(
            jax.device_put(value, sharding)
            for value, sharding in zip(
                host_inputs,
                input_shardings_by_scenario[scenario.name],
                strict=True,
            )
        )
    _record_event(
        ledger_path,
        run_id,
        RunState.VERIFIED,
        {
            "surface_id": surface.surface_id,
            "schedule_sha256": {
                name: plan.schedule_sha256 for name, plan in plans_by_scenario.items()
            },
        },
    )

    for scenario in surface.scenarios:
        module = modules_by_scenario[scenario.name]
        plan = plans_by_scenario[scenario.name]
        ir_artifact = _write_text(
            output_dir,
            Path("ir") / f"{scenario.name}.xdsl",
            canonical_text(module),
            ArtifactRole.DISTRIBUTED_IR,
        )
        artifacts.extend(
            (
                ir_artifact,
                _write_text(
                    output_dir,
                    Path("lowering") / f"{scenario.name}.py",
                    plan.render_executable_source(),
                    ArtifactRole.JAX_SOURCE,
                ),
                _write_nested_json(
                    output_dir,
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
            )
        )
    artifacts.append(
        _write_json(
            output_dir / "plans.json",
            _plan_payload(surface),
            ArtifactRole.PLAN_MANIFEST,
        )
    )
    _record_event(
        ledger_path,
        run_id,
        RunState.LOWERED,
        {
            "plans_sha256": artifacts[-1].sha256,
            "strategy_sha256": next(
                artifact.sha256 for artifact in artifacts if artifact.path == "strategies.py"
            ),
        },
    )

    hlo_artifacts = []
    for scenario in surface.scenarios:
        lowered = jitted_by_scenario[scenario.name].lower(
            *resident_inputs_by_scenario[scenario.name]
        )
        hlo_artifacts.extend(
            (
                _write_text(
                    output_dir,
                    Path("hlo") / scenario.name / "candidate_stablehlo.txt",
                    str(lowered.compiler_ir(dialect="stablehlo")) + "\n",
                    ArtifactRole.STABLEHLO,
                ),
                _write_text(
                    output_dir,
                    Path("hlo") / scenario.name / "candidate_compiler_hlo.txt",
                    _compiler_hlo(lowered) + "\n",
                    ArtifactRole.COMPILER_HLO,
                ),
            )
        )
        jitted_by_scenario[scenario.name] = lowered.compile()
    artifacts.extend(hlo_artifacts)
    _record_event(
        ledger_path,
        run_id,
        RunState.COMPILED,
        {artifact.path: artifact.sha256 for artifact in hlo_artifacts},
    )

    def input_factory(scenario, seed):
        expected_seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
        if seed != expected_seed:
            raise ValueError(f"SEQAX_SURFACE_INPUT_SEED_MISMATCH scenario={scenario.name}")
        return resident_inputs_by_scenario[scenario.name]

    def oracle(scenario, inputs):
        host_inputs = tuple(np.asarray(jax.device_get(value)) for value in inputs)
        return (
            seqax_forward_canonical_reference(host_inputs, **scenario.parameters()),
        )

    def baseline(scenario, inputs):
        return mapped_by_scenario[scenario.name](*inputs)

    def candidate(scenario, inputs):
        return jitted_by_scenario[scenario.name](*inputs)

    def record_correctness() -> None:
        _record_event(
            ledger_path,
            run_id,
            RunState.CORRECT,
            {
                "absolute_tolerance": SEQAX_SURFACE_ATOL,
                "relative_tolerance": SEQAX_SURFACE_RTOL,
                "scenarios": tuple(scenario.name for scenario in surface.scenarios),
            },
        )

    def record_timing() -> None:
        _record_event(
            ledger_path,
            run_id,
            RunState.TIMED,
            {
                "rounds": SEQAX_SURFACE_ROUNDS,
                "samples_per_round": SEQAX_SURFACE_MEASURED_ITERATIONS,
                "timing_scope": invocation.timing_scope,
            },
        )

    comparison, baseline_observation, candidate_observation = run_surface_pair(
        surface,
        baseline_name=SEQAX_SURFACE_BASELINE,
        candidate_name=SEQAX_SURFACE_CANDIDATE,
        baseline=baseline,
        candidate=candidate,
        input_factory=input_factory,
        oracle=oracle,
        runtime_sha256=runtime_sha256,
        rounds=SEQAX_SURFACE_ROUNDS,
        warmup_iterations=SEQAX_SURFACE_WARMUP_ITERATIONS,
        measured_iterations=SEQAX_SURFACE_MEASURED_ITERATIONS,
        absolute_tolerance=SEQAX_SURFACE_ATOL,
        relative_tolerance=SEQAX_SURFACE_RTOL,
        on_correctness_complete=record_correctness,
        on_timing_complete=record_timing,
    )
    for scenario in surface.scenarios:
        inputs = host_inputs_by_scenario[scenario.name]
        resident_inputs = resident_inputs_by_scenario[scenario.name]
        oracle_value = seqax_forward_canonical_reference(
            inputs, **scenario.parameters()
        )
        baseline_value = np.asarray(
            jax.block_until_ready(
                mapped_by_scenario[scenario.name](*resident_inputs)
            )[0]
        )
        candidate_value = np.asarray(
            jax.block_until_ready(
                jitted_by_scenario[scenario.name](*resident_inputs)
            )[0]
        )
        artifacts.extend(
            _save_array(
                output_dir,
                Path("inputs") / scenario.name / f"{index:02d}.npy",
                value,
                ArtifactRole.CORRECTNESS_INPUT,
            )
            for index, value in enumerate(inputs)
        )
        artifacts.extend(
            (
                _save_array(
                    output_dir,
                    Path("oracle") / f"{scenario.name}.npy",
                    oracle_value,
                    ArtifactRole.ORACLE_OUTPUT,
                ),
                _save_array(
                    output_dir,
                    Path("outputs") / scenario.name / "baseline.npy",
                    baseline_value,
                    ArtifactRole.CORRECTNESS_OUTPUT,
                ),
                _save_array(
                    output_dir,
                    Path("outputs") / scenario.name / "candidate.npy",
                    candidate_value,
                    ArtifactRole.CORRECTNESS_OUTPUT,
                ),
            )
        )
    artifacts.extend(
        (
            _write_json(
                output_dir / "baseline.json",
                baseline_observation.model_dump(mode="json"),
                ArtifactRole.SEARCH_EVIDENCE,
            ),
            _write_json(
                output_dir / "candidate.json",
                candidate_observation.model_dump(mode="json"),
                ArtifactRole.SEARCH_EVIDENCE,
            ),
            _write_json(
                output_dir / "comparison.json",
                comparison.model_dump(mode="json"),
                ArtifactRole.SEARCH_RESULT,
            ),
        )
    )
    _record_event(
        ledger_path,
        run_id,
        RunState.ACCEPTED,
        {
            "comparison_sha256": next(
                artifact.sha256 for artifact in artifacts if artifact.path == "comparison.json"
            ),
            "candidate_promoted": comparison.promotable,
        },
    )
    artifacts.append(_artifact(output_dir, ledger_path, ArtifactRole.EXECUTION_LEDGER))
    receipt = SeqaxSurfaceReceipt(
        schema_version=SEQAX_SURFACE_SCHEMA,
        surface_id=surface.surface_id,
        invocation=invocation,
        comparison=comparison,
        candidate_promoted=comparison.promotable,
        artifacts=tuple(sorted(artifacts, key=lambda artifact: artifact.path)),
    )
    validate_seqax_surface_receipt(receipt, root=output_dir)
    temporary = output_dir / "receipt.json.tmp"
    temporary.write_text(receipt.model_dump_json(indent=2) + "\n")
    temporary.replace(output_dir / "receipt.json")
    return receipt


def validate_seqax_surface_receipt(receipt: SeqaxSurfaceReceipt, *, root: Path) -> None:
    root = root.resolve()
    surface = seqax_forward_workload_surface()
    if (
        receipt.schema_version != SEQAX_SURFACE_SCHEMA
        or receipt.surface_id != surface.surface_id
    ):
        raise ValueError("SEQAX_SURFACE_RECEIPT_IDENTITY_MISMATCH")
    expected_roles = _artifact_roles(surface)
    expected_paths = set(expected_roles)
    observed_paths = {artifact.path for artifact in receipt.artifacts}
    if observed_paths != expected_paths or len(observed_paths) != len(receipt.artifacts):
        raise ValueError("SEQAX_SURFACE_ARTIFACT_MANIFEST_MISMATCH")
    by_path = {artifact.path: artifact for artifact in receipt.artifacts}
    for path, role in expected_roles.items():
        artifact = by_path[path]
        if artifact.role is not role:
            raise ValueError(f"SEQAX_SURFACE_ARTIFACT_ROLE_MISMATCH path={path}")
        resolve_recorded_artifact(
            root,
            path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
    allowed = expected_paths | {"receipt.json"}
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if files not in (expected_paths, allowed):
        raise ValueError("SEQAX_SURFACE_CLOSED_WORLD_MISMATCH")
    source_commit, _ = _source_identity(
        root / "source_state.json",
        root / "source_diff.patch",
        require_clean=True,
    )

    saved_surface = SeqaxForwardWorkloadSurface.model_validate_json(
        (root / "surface.json").read_text()
    )
    invocation = SeqaxSurfaceInvocation.model_validate_json(
        (root / "invocation.json").read_text()
    )
    baseline = SurfaceCandidateObservation.model_validate_json(
        (root / "baseline.json").read_text()
    )
    candidate = SurfaceCandidateObservation.model_validate_json(
        (root / "candidate.json").read_text()
    )
    comparison = SurfaceComparison.model_validate_json(
        (root / "comparison.json").read_text()
    )
    if saved_surface != surface or invocation != receipt.invocation:
        raise ValueError("SEQAX_SURFACE_CONTRACT_REPLAY_MISMATCH")
    if (
        baseline.candidate != invocation.baseline
        or candidate.candidate != invocation.candidate
    ):
        raise ValueError("SEQAX_SURFACE_CANDIDATE_IDENTITY_MISMATCH")
    for observation in (*baseline.scenarios, *candidate.scenarios):
        if (
            len(observation.round_medians_ns) != invocation.rounds
            or len(observation.ran_first) != invocation.rounds
            or len(observation.round_samples_ns) != invocation.rounds
            or any(
                len(samples) != invocation.measured_iterations
                for samples in observation.round_samples_ns
            )
            or observation.runtime_sha256 != invocation.runtime_sha256
        ):
            raise ValueError("SEQAX_SURFACE_OBSERVATION_PROTOCOL_MISMATCH")
    baseline_by_name = {value.scenario: value for value in baseline.scenarios}
    candidate_by_name = {value.scenario: value for value in candidate.scenarios}
    for scenario in surface.scenarios:
        seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
        expected_inputs = tuple(
            np.asarray(value)
            for value in seqax_forward_inputs(seed=seed, **scenario.parameters())
        )
        saved_inputs = tuple(
            np.load(
                root / "inputs" / scenario.name / f"{index:02d}.npy",
                allow_pickle=False,
            )
            for index in range(len(expected_inputs))
        )
        if any(
            not _same_array(saved, expected)
            for saved, expected in zip(saved_inputs, expected_inputs, strict=True)
        ):
            raise ValueError(
                f"SEQAX_SURFACE_DETERMINISTIC_INPUT_MISMATCH scenario={scenario.name}"
            )
        expected_oracle = seqax_forward_canonical_reference(
            expected_inputs, **scenario.parameters()
        )
        saved_oracle = np.load(
            root / "oracle" / f"{scenario.name}.npy", allow_pickle=False
        )
        baseline_output = np.load(
            root / "outputs" / scenario.name / "baseline.npy", allow_pickle=False
        )
        candidate_output = np.load(
            root / "outputs" / scenario.name / "candidate.npy", allow_pickle=False
        )
        if not _same_array(saved_oracle, expected_oracle):
            raise ValueError(f"SEQAX_SURFACE_ORACLE_REPLAY_MISMATCH scenario={scenario.name}")
        outputs = {
            "baseline": baseline_output,
            "candidate": candidate_output,
        }
        for name, output in outputs.items():
            if output.shape != expected_oracle.shape or output.dtype != expected_oracle.dtype:
                raise ValueError(
                    "SEQAX_SURFACE_OUTPUT_CONTRACT_MISMATCH "
                    f"scenario={scenario.name} candidate={name}"
                )
            if not np.allclose(
                output,
                expected_oracle,
                atol=SEQAX_SURFACE_ATOL,
                rtol=SEQAX_SURFACE_RTOL,
            ):
                raise ValueError(
                    "SEQAX_SURFACE_OUTPUT_ORACLE_MISMATCH "
                    f"scenario={scenario.name} candidate={name}"
                )
        if not np.allclose(
            baseline_output,
            candidate_output,
            atol=SEQAX_SURFACE_ATOL,
            rtol=SEQAX_SURFACE_RTOL,
        ):
            raise ValueError(
                f"SEQAX_SURFACE_CROSS_MODE_MISMATCH scenario={scenario.name}"
            )
        input_identity = _array_tuple_sha256(saved_inputs)
        baseline_output_identity = _array_tuple_sha256((baseline_output,))
        candidate_output_identity = _array_tuple_sha256((candidate_output,))
        if (
            baseline_by_name[scenario.name].input_sha256 != input_identity
            or candidate_by_name[scenario.name].input_sha256 != input_identity
            or baseline_by_name[scenario.name].output_sha256
            != baseline_output_identity
            or candidate_by_name[scenario.name].output_sha256
            != candidate_output_identity
        ):
            raise ValueError(f"SEQAX_SURFACE_ARRAY_IDENTITY_MISMATCH scenario={scenario.name}")
    if (
        invocation.schema_version != SEQAX_SURFACE_SCHEMA
        or invocation.surface_id != surface.surface_id
        or invocation.baseline != SEQAX_SURFACE_BASELINE
        or invocation.candidate != SEQAX_SURFACE_CANDIDATE
        or invocation.rounds != SEQAX_SURFACE_ROUNDS
        or invocation.warmup_iterations != SEQAX_SURFACE_WARMUP_ITERATIONS
        or invocation.measured_iterations != SEQAX_SURFACE_MEASURED_ITERATIONS
        or invocation.device_kind != "TPU7x"
        or invocation.device_count != 8
        or invocation.runtime_sha256
        != _runtime_sha256(invocation.runtime, invocation.device_kind, invocation.device_count)
        or invocation.baseline_strategy != "unwrapped-jax-shard-map"
        or invocation.candidate_strategy != "whole-program-jit-over-jax-shard-map"
        or invocation.input_placement != "resident-named-sharding-before-warmup"
        or invocation.timing_scope != "synchronized-resident-input-complete-forward"
        or invocation.comparison_kind != "wrapper-dispatch-control"
        or invocation.run_id
        != semantic_sha256(
            SEQAX_SURFACE_SCHEMA,
            surface.surface_id,
            source_commit,
            invocation.runtime_sha256,
        )
    ):
        raise ValueError("SEQAX_SURFACE_INVOCATION_REPLAY_MISMATCH")
    if json.loads((root / "plans.json").read_text()) != _plan_payload(surface):
        raise ValueError("SEQAX_SURFACE_PLAN_REPLAY_MISMATCH")
    if (root / "strategies.py").read_text() != _strategy_source():
        raise ValueError("SEQAX_SURFACE_STRATEGY_REPLAY_MISMATCH")
    for scenario in surface.scenarios:
        module = seqax_forward_schedule(**scenario.parameters())
        plan = lower_distributed_program_to_jax_mesh(module)
        if (root / "ir" / f"{scenario.name}.xdsl").read_text() != canonical_text(module):
            raise ValueError(
                f"SEQAX_SURFACE_DISTRIBUTED_IR_REPLAY_MISMATCH scenario={scenario.name}"
            )
        if (
            root / "lowering" / f"{scenario.name}.py"
        ).read_text() != plan.render_executable_source():
            raise ValueError(
                f"SEQAX_SURFACE_LOWERING_REPLAY_MISMATCH scenario={scenario.name}"
            )
        ir_artifact = by_path[f"ir/{scenario.name}.xdsl"]
        expected_cost = estimate_seqax_forward(
            module,
            hardware=tpu7x_tensorcore_rates(),
            source=MetricSource(
                artifact_sha256=ir_artifact.sha256,
                artifact_path=ir_artifact.path,
                tool="tpu-cake",
                field="seqax-surface-distributed-ir",
            ),
            expected_schedule_sha256=plan.schedule_sha256,
        )
        saved_cost = SeqaxCostModelReport.model_validate_json(
            (root / "cost" / f"{scenario.name}.json").read_text()
        )
        if saved_cost != expected_cost:
            raise ValueError(
                f"SEQAX_SURFACE_COST_REPLAY_MISMATCH scenario={scenario.name}"
            )
        stablehlo = (
            root / "hlo" / scenario.name / "candidate_stablehlo.txt"
        ).read_text()
        compiler_hlo = (
            root / "hlo" / scenario.name / "candidate_compiler_hlo.txt"
        ).read_text()
        if not all(
            marker in stablehlo
            for marker in ("stablehlo.all_gather", "stablehlo.reduce_scatter")
        ) or not all(
            marker in compiler_hlo
            for marker in ("all-gather", "reduce-scatter", "dot")
        ):
            raise ValueError(
                f"SEQAX_SURFACE_HLO_MARKER_MISMATCH scenario={scenario.name}"
            )
    expected_comparison = compare_surface_candidates(surface, baseline, candidate)
    if comparison != expected_comparison or receipt.comparison != expected_comparison:
        raise ValueError("SEQAX_SURFACE_COMPARISON_REPLAY_MISMATCH")
    if receipt.candidate_promoted is not expected_comparison.promotable:
        raise ValueError("SEQAX_SURFACE_PROMOTION_REPLAY_MISMATCH")
    expected_ledger = (
        (
            RunState.CREATED,
            invocation.model_dump(mode="json"),
        ),
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
                "absolute_tolerance": SEQAX_SURFACE_ATOL,
                "relative_tolerance": SEQAX_SURFACE_RTOL,
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
                "candidate_promoted": expected_comparison.promotable,
            },
        ),
    )
    history = read_ledger_history(root / "ledger.sqlite", invocation.run_id)
    if tuple(event.state for event in history) != tuple(
        state for state, _ in expected_ledger
    ) or tuple(event.payload_sha256 for event in history) != tuple(
        ExperimentLedger.payload_sha256(payload) for _, payload in expected_ledger
    ):
        raise ValueError("SEQAX_SURFACE_LEDGER_REPLAY_MISMATCH")
