from __future__ import annotations

import ast
import hashlib
import subprocess
from functools import partial
from pathlib import Path

import pytest
from pydantic import ValidationError

import tpu_cake.matmul_collective_surface_runner as surface_runner
from tpu_cake.contracts import SourceFileContract
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceSplit,
    SurfaceCalibrationObservation,
    default_matmul_collective_surface_design_contract,
    derive_matmul_collective_surface_design_report,
)
from tpu_cake.matmul_collective_surface_runner import (
    SURFACE_EXECUTABLE_DEPENDENCIES,
    CalibrationSealEnvelope,
    MatmulCollectiveSurfaceCompileReport,
    MatmulCollectiveSurfaceSourceAuthority,
    SurfaceCalibrationBatch,
    SurfaceCompileStatus,
    SurfaceCorrectnessArtifact,
    SurfaceCorrectnessObservation,
    SurfaceEvidenceKind,
    SurfaceMeasurement,
    SurfacePhase,
    SurfacePhaseLedger,
    begin_surface_holdout,
    capture_surface_source_authority,
    create_surface_attempt_root,
    derive_surface_input_identities,
    make_compile_capture_record,
    record_surface_holdout_correctness,
    record_surface_phase,
    replay_calibration_seal,
    replay_compile_capture_report,
    seal_surface_calibration,
    validate_calibration_seal,
    validate_compile_capture_report,
    validate_surface_calibration_batch,
    validate_surface_correctness_artifact,
    validate_surface_source_authority,
    write_calibration_seal,
    write_compile_capture_report,
    write_surface_calibration_batch,
    write_surface_correctness_artifact,
)
from tpu_cake.runner import MatmulCollectiveStrategy

EXECUTION_AUTHORITY_SHA256 = "e" * 64

begin_surface_holdout = partial(
    begin_surface_holdout,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
record_surface_holdout_correctness = partial(
    record_surface_holdout_correctness,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
replay_calibration_seal = partial(
    replay_calibration_seal,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
replay_compile_capture_report = partial(
    replay_compile_capture_report,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
seal_surface_calibration = partial(
    seal_surface_calibration,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
validate_calibration_seal = partial(
    validate_calibration_seal,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
validate_compile_capture_report = partial(
    validate_compile_capture_report,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
validate_surface_calibration_batch = partial(
    validate_surface_calibration_batch,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
validate_surface_correctness_artifact = partial(
    validate_surface_correctness_artifact,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
write_calibration_seal = partial(
    write_calibration_seal,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
write_compile_capture_report = partial(
    write_compile_capture_report,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
write_surface_calibration_batch = partial(
    write_surface_calibration_batch,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)
write_surface_correctness_artifact = partial(
    write_surface_correctness_artifact,
    execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _recursive_internal_imports(entrypoint: Path) -> tuple[str, ...]:
    source_root = entrypoint.parents[1]
    pending = [entrypoint, entrypoint.parent / "__init__.py"]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        observed.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if not module.startswith("tpu_cake"):
                    continue
                parts = module.split(".")
                for length in range(1, len(parts)):
                    initializer = source_root.joinpath(*parts[:length], "__init__.py")
                    if initializer.is_file() and initializer not in observed:
                        pending.append(initializer)
                dependency = source_root / Path(*module.split(".")).with_suffix(".py")
                if dependency.is_file() and dependency not in observed:
                    pending.append(dependency)
    return tuple(sorted(path.relative_to(source_root).as_posix() for path in observed))


def _native_collective_stablehlo(scenario: str, m: int, k: int, n: int) -> str:
    return f"""
module @{scenario.replace("-", "_")} {{
  func.func public @main(%lhs: tensor<{m}x{k}xbf16>, %rhs: tensor<{k}x{n}xbf16>) -> tensor<{m}x{n}xf32> {{
    %matmul = stablehlo.custom_call @tpu_custom_call(%lhs, %rhs) {{
      backend_config = "", kernel_name = "distributed_matmul_physical"
    }} : (tensor<{m}x{k}xbf16>, tensor<{k}x{n}xbf16>) -> tensor<{m}x{n}xf32>
    %collective = stablehlo.custom_call @tpu_custom_call(%matmul) {{
      backend_config = "",
      kernel_name = "distributed_matmul_physical_pallas_reduce_scatter"
    }} : (tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %collective : tensor<{m}x{n}xf32>
  }}
}}
"""


def _native_collective_compiler_hlo(
    scenario: str,
    m: int,
    k: int,
    n: int,
) -> str:
    return f"""
HloModule {scenario.replace("-", "_")}, is_scheduled=true, entry_computation_layout={{(bf16[{m},{k}],bf16[{k},{n}])->f32[{m},{n}]}}

ENTRY %main {{
  %input = f32[1] parameter(0)
  %pallas_call.1 = f32[1] custom-call(%input), custom_call_target="tpu_custom_call"
  ROOT %pallas_call.2 = f32[1] custom-call(%pallas_call.1), custom_call_target="tpu_custom_call"
}}
"""


def _xla_collective_stablehlo(scenario: str, m: int, k: int, n: int) -> str:
    return f"""
module @{scenario.replace("-", "_")} {{
  func.func public @main(%lhs: tensor<{m}x{k}xbf16>, %rhs: tensor<{k}x{n}xbf16>) -> tensor<{m}x{n}xf32> {{
    %matmul = stablehlo.custom_call @tpu_custom_call(%lhs, %rhs) {{
      backend_config = "", kernel_name = "distributed_matmul_physical"
    }} : (tensor<{m}x{k}xbf16>, tensor<{k}x{n}xbf16>) -> tensor<{m}x{n}xf32>
    %scatter = "stablehlo.reduce_scatter"(%matmul) <{{
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0]]> : tensor<1x1xi64>,
      scatter_dimension = 0 : i64,
      use_global_device_ids
    }}> ({{
      ^bb0(%left: tensor<f32>, %right: tensor<f32>):
        %sum = stablehlo.add %left, %right : tensor<f32>
        stablehlo.return %sum : tensor<f32>
    }}) : (tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %scatter : tensor<{m}x{n}xf32>
  }}
}}
"""


def _xla_collective_compiler_hlo(
    scenario: str,
    m: int,
    k: int,
    n: int,
) -> str:
    return f"""
HloModule {scenario.replace("-", "_")}, is_scheduled=true, entry_computation_layout={{(bf16[{m},{k}],bf16[{k},{n}])->f32[{m},{n}]}}

ENTRY %main {{
  %input = f32[1] parameter(0)
  %pallas_call.1 = f32[1] custom-call(%input), custom_call_target="tpu_custom_call"
  ROOT %reduce-scatter.1 = f32[1] reduce-scatter(%pallas_call.1), backend_config={{"device_type":"DEVICE_TYPE_SPARSECORE","reduce_scatter_offload_config":{{}}}}
}}
"""


@pytest.fixture
def contract():
    return default_matmul_collective_surface_design_contract()


@pytest.fixture
def source_blobs() -> dict[str, bytes]:
    return {
        **{path: f"source:{path}\n".encode() for path in SURFACE_EXECUTABLE_DEPENDENCIES},
        "uv.lock": b"locked-runtime\n",
    }


@pytest.fixture
def source(contract, source_blobs) -> MatmulCollectiveSurfaceSourceAuthority:
    commit = "a" * 40
    return MatmulCollectiveSurfaceSourceAuthority(
        source_commit=commit,
        branch="main",
        origin_main_commit=commit,
        remote_main_commit=commit,
        remote_url=contract.source_remote_url,
        compilation_source_root=contract.compilation_source_root,
        runtime=contract.runtime,
        uv_lock_sha256=hashlib.sha256(source_blobs["uv.lock"]).hexdigest(),
        dependencies=tuple(
            SourceFileContract(path=path, sha256=hashlib.sha256(source_blobs[path]).hexdigest())
            for path in SURFACE_EXECUTABLE_DEPENDENCIES
        ),
    )


@pytest.fixture(autouse=True)
def committed_source_blobs(monkeypatch, source_blobs) -> None:
    monkeypatch.setattr(
        surface_runner,
        "_read_committed_source_blobs",
        lambda _root, _commit: source_blobs,
    )


def _compile_report(contract, source) -> MatmulCollectiveSurfaceCompileReport:
    identities = {value.scenario_name: value for value in derive_surface_input_identities(contract)}
    design = derive_matmul_collective_surface_design_report(contract)
    arms = {(value.scenario_name, value.strategy): value for value in design.arms}
    records = []
    for scenario in contract.scenarios:
        for strategy in contract.strategies:
            stablehlo = (
                _xla_collective_stablehlo(scenario.name, scenario.m, scenario.k, scenario.n)
                if strategy is MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
                else _native_collective_stablehlo(scenario.name, scenario.m, scenario.k, scenario.n)
            )
            compiler_hlo = (
                _xla_collective_compiler_hlo(
                    scenario.name,
                    scenario.m,
                    scenario.k,
                    scenario.n,
                )
                if strategy is MatmulCollectiveStrategy.XLA_REDUCE_SCATTER
                else _native_collective_compiler_hlo(
                    scenario.name,
                    scenario.m,
                    scenario.k,
                    scenario.n,
                )
            )
            for repetition in (1, 2):
                arm = arms[(scenario.name, strategy)]
                records.append(
                    make_compile_capture_record(
                        scenario_name=scenario.name,
                        strategy=strategy,
                        repetition=repetition,
                        input_contract_sha256=identities[scenario.name].input_contract_sha256,
                        distributed_schedule_sha256=arm.distributed_schedule_sha256,
                        physical_schedule_sha256=arm.physical_schedule_sha256,
                        pallas_source_sha256=arm.pallas_source_sha256,
                        stablehlo=stablehlo,
                        compiler_hlo=compiler_hlo,
                    )
                )
    return MatmulCollectiveSurfaceCompileReport(
        design_id=contract.design_id,
        source_authority_sha256=source.authority_sha256,
        execution_authority_sha256=EXECUTION_AUTHORITY_SHA256,
        captures=tuple(records),
    )


def _observations(contract) -> tuple[SurfaceMeasurement, ...]:
    report = derive_matmul_collective_surface_design_report(contract)
    values = []
    for index, arm in enumerate(report.calibration_arms):
        values.append(
            SurfaceMeasurement(
                scenario_name=arm.scenario_name,
                strategy=arm.strategy,
                evidence_kind=SurfaceEvidenceKind.UNPROFILED_TIMING,
                median_ns=float(100_000 + index * 3_000),
            )
        )
    return tuple(values)


def _correctness_artifact(
    contract,
    compile_report,
    source,
    split=MatmulCollectiveSurfaceSplit.CALIBRATION,
) -> SurfaceCorrectnessArtifact:
    identities = {
        value.scenario_name: value.input_contract_sha256
        for value in derive_surface_input_identities(contract)
    }
    return SurfaceCorrectnessArtifact(
        design_id=contract.design_id,
        split=split,
        compile_report_sha256=compile_report.report_sha256,
        source_authority_sha256=source.authority_sha256,
        observations=tuple(
            SurfaceCorrectnessObservation(
                scenario_name=scenario.name,
                strategy=strategy,
                pattern=pattern,
                input_contract_sha256=identities[scenario.name],
                oracle_output_sha256=_sha(f"oracle:{scenario.name}:{pattern}"),
                candidate_output_sha256=_sha(f"output:{scenario.name}:{strategy}:{pattern}"),
                absolute_tolerance=1e-3,
                relative_tolerance=1e-3,
                maximum_absolute_error=5e-4,
                maximum_relative_error=5e-4,
                mismatched_element_count=0,
            )
            for scenario in contract.scenarios
            if scenario.split is split
            for strategy in contract.strategies
            for pattern in contract.correctness_patterns
        ),
    )


def _calibration_batch(contract, compile_report, source) -> SurfaceCalibrationBatch:
    return SurfaceCalibrationBatch(
        design_id=contract.design_id,
        compile_report_sha256=compile_report.report_sha256,
        source_authority_sha256=source.authority_sha256,
        observations=_observations(contract),
    )


def _calibration_ledger(
    compile_report,
    correctness: SurfaceCorrectnessArtifact,
    batch: SurfaceCalibrationBatch,
) -> SurfacePhaseLedger:
    ledger = SurfacePhaseLedger(attempt_id="c" * 64)
    for phase, artifact_sha256 in (
        (SurfacePhase.COMPILE, compile_report.report_sha256),
        (SurfacePhase.CORRECTNESS, correctness.artifact_sha256),
        (SurfacePhase.CALIBRATION, batch.artifact_sha256),
    ):
        ledger = record_surface_phase(ledger, phase, artifact_sha256)
    return ledger


def test_source_authority_binds_main_runtime_lock_and_explicit_dependencies(
    contract, source, source_blobs, monkeypatch
) -> None:
    validate_surface_source_authority(source, contract, source_blobs)
    assert tuple(value.path for value in source.dependencies) == SURFACE_EXECUTABLE_DEPENDENCIES

    tampered = dict(source_blobs)
    tampered[SURFACE_EXECUTABLE_DEPENDENCIES[-1]] += b"tamper"
    with pytest.raises(ValueError, match="SOURCE_DEPENDENCY_MISMATCH"):
        validate_surface_source_authority(source, contract, tampered)

    tampered = dict(source_blobs)
    tampered["uv.lock"] += b"tamper"
    with pytest.raises(ValueError, match="UV_LOCK_MISMATCH"):
        validate_surface_source_authority(source, contract, tampered)

    dependencies = list(source.dependencies)
    dependencies[-1] = dependencies[-1].model_copy(update={"sha256": "0" * 64})
    forged_manifest = source.model_copy(update={"dependencies": tuple(dependencies)})
    with pytest.raises(ValueError, match="SOURCE_DEPENDENCY_MISMATCH"):
        validate_surface_source_authority(forged_manifest, contract, source_blobs)

    forged_blobs = {path: blob + b"forged" for path, blob in source_blobs.items()}
    forged_source = source.model_copy(
        update={
            "source_commit": "b" * 40,
            "origin_main_commit": "b" * 40,
            "remote_main_commit": "b" * 40,
            "uv_lock_sha256": hashlib.sha256(forged_blobs["uv.lock"]).hexdigest(),
            "dependencies": tuple(
                SourceFileContract(
                    path=path,
                    sha256=hashlib.sha256(forged_blobs[path]).hexdigest(),
                )
                for path in SURFACE_EXECUTABLE_DEPENDENCIES
            ),
        }
    )
    monkeypatch.setattr(
        surface_runner,
        "_read_committed_source_blobs",
        lambda _root, _commit: source_blobs,
    )
    with pytest.raises(ValueError, match="SOURCE_COMMIT_BLOB_MISMATCH"):
        validate_surface_source_authority(forged_source, contract, forged_blobs)

    for update in (
        {"remote_url": "https://example.invalid/tpu-cake.git"},
        {"compilation_source_root": "/tmp/not-the-declared-root"},
    ):
        with pytest.raises(ValueError, match="SOURCE_AUTHORITY_MISMATCH"):
            validate_surface_source_authority(
                source.model_copy(update=update),
                contract,
                source_blobs,
            )

    with pytest.raises(TypeError):
        capture_surface_source_authority(Path("/unused"), contract, {"python": "forged"})


def test_executable_dependency_manifest_matches_recursive_import_closure() -> None:
    entrypoint = Path(__file__).parents[1] / "src/tpu_cake/matmul_collective_surface_runner.py"
    assert _recursive_internal_imports(entrypoint) == SURFACE_EXECUTABLE_DEPENDENCIES


def test_source_subprocess_environment_ignores_hostile_git_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile_home = tmp_path / "home"
    hostile_home.mkdir()
    (hostile_home / ".gitconfig").write_text(
        '[url "https://attacker.invalid/"]\n\tinsteadOf = https://github.com/\n'
    )
    monkeypatch.setenv("HOME", str(hostile_home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile_home / ".gitconfig"))
    environment = surface_runner._source_subprocess_environment()

    assert environment == {
        "GIT_ASKPASS": "/bin/false",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        ["/usr/bin/git", "config", "--global", "--get-regexp", r"^url\."],
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == ""


def test_compile_capture_is_complete_balanced_stable_and_strategy_checked(
    contract, source, source_blobs
) -> None:
    report = _compile_report(contract, source)
    validate_compile_capture_report(report, contract, source, source_blobs)
    assert len(report.captures) == 80

    forged_authority = report.model_copy(update={"execution_authority_sha256": "f" * 64})
    with pytest.raises(ValueError, match="COMPILE_AUTHORITY_MISMATCH"):
        surface_runner.validate_compile_capture_report(
            forged_authority,
            contract,
            source,
            source_blobs,
            EXECUTION_AUTHORITY_SHA256,
        )

    forged_schema = report.model_copy(update={"schema_version": "attacker-v999"})
    with pytest.raises(ValidationError):
        validate_compile_capture_report(forged_schema, contract, source, source_blobs)

    for mutation in (
        report.captures[1:],
        (*report.captures, report.captures[0]),
        tuple(value for value in report.captures if value.repetition == 1),
    ):
        broken = report.model_copy(update={"captures": mutation})
        with pytest.raises(ValueError, match="COMPILE_INVENTORY_MISMATCH"):
            validate_compile_capture_report(broken, contract, source, source_blobs)


def test_compile_capture_rejects_strategy_input_and_hlo_substitution(
    contract, source, source_blobs
) -> None:
    report = _compile_report(contract, source)
    first = report.captures[0]
    divergent_input = first.model_copy(update={"input_contract_sha256": "d" * 64})
    broken = report.model_copy(update={"captures": (divergent_input, *report.captures[1:])})
    with pytest.raises(ValueError, match="INPUT_IDENTITY_MISMATCH"):
        validate_compile_capture_report(broken, contract, source, source_blobs)

    wrong_arm = first.model_copy(update={"physical_schedule_sha256": "f" * 64})
    broken = report.model_copy(update={"captures": (wrong_arm, *report.captures[1:])})
    with pytest.raises(ValueError, match="COMPILE_ARM_IDENTITY_MISMATCH"):
        validate_compile_capture_report(broken, contract, source, source_blobs)

    replacement = next(value for value in report.captures if value.strategy is not first.strategy)
    substituted = first.model_copy(
        update={
            "stablehlo": replacement.stablehlo,
            "stablehlo_sha256": replacement.stablehlo_sha256,
            "semantic_stablehlo_sha256": replacement.semantic_stablehlo_sha256,
            "compiler_hlo": replacement.compiler_hlo,
            "compiler_hlo_sha256": replacement.compiler_hlo_sha256,
            "semantic_compiler_hlo_sha256": replacement.semantic_compiler_hlo_sha256,
        }
    )
    broken = report.model_copy(update={"captures": (substituted, *report.captures[1:])})
    with pytest.raises(ValueError, match="RUN_COMPILER_COLLECTIVE_STRATEGY_MISMATCH"):
        validate_compile_capture_report(broken, contract, source, source_blobs)

    same_strategy_other_shape = next(
        value
        for value in report.captures
        if value.strategy is first.strategy and value.scenario_name != first.scenario_name
    )
    substituted = first.model_copy(
        update={
            "stablehlo": same_strategy_other_shape.stablehlo,
            "stablehlo_sha256": same_strategy_other_shape.stablehlo_sha256,
            "semantic_stablehlo_sha256": (same_strategy_other_shape.semantic_stablehlo_sha256),
            "compiler_hlo": same_strategy_other_shape.compiler_hlo,
            "compiler_hlo_sha256": same_strategy_other_shape.compiler_hlo_sha256,
            "semantic_compiler_hlo_sha256": (
                same_strategy_other_shape.semantic_compiler_hlo_sha256
            ),
        }
    )
    broken = report.model_copy(update={"captures": (substituted, *report.captures[1:])})
    with pytest.raises(ValueError, match="STABLEHLO_STATIC_ABI_MISMATCH"):
        validate_compile_capture_report(broken, contract, source, source_blobs)

    target_indexes = tuple(
        index
        for index, value in enumerate(report.captures)
        if value.scenario_name == first.scenario_name and value.strategy is first.strategy
    )
    replacement_records = tuple(
        value
        for value in report.captures
        if value.scenario_name == same_strategy_other_shape.scenario_name
        and value.strategy is first.strategy
    )
    captures = list(report.captures)
    for target_index, replacement_record in zip(target_indexes, replacement_records, strict=True):
        captures[target_index] = captures[target_index].model_copy(
            update={
                "compiler_hlo": replacement_record.compiler_hlo,
                "compiler_hlo_sha256": replacement_record.compiler_hlo_sha256,
                "semantic_compiler_hlo_sha256": (replacement_record.semantic_compiler_hlo_sha256),
            }
        )
    broken = report.model_copy(update={"captures": tuple(captures)})
    with pytest.raises(ValueError, match="COMPILER_HLO_STATIC_ABI_MISMATCH"):
        validate_compile_capture_report(broken, contract, source, source_blobs)

    commented_stablehlo = (
        same_strategy_other_shape.stablehlo
        + f"// tensor<{contract.scenarios[0].m}x{contract.scenarios[0].k}xbf16> "
        + f"tensor<{contract.scenarios[0].k}x{contract.scenarios[0].n}xbf16> "
        + f"tensor<{contract.scenarios[0].m}x{contract.scenarios[0].n}xf32>\n"
    )
    captures = list(report.captures)
    for target_index in target_indexes:
        captures[target_index] = captures[target_index].model_copy(
            update={
                "stablehlo": commented_stablehlo,
                "stablehlo_sha256": _sha(commented_stablehlo),
                "semantic_stablehlo_sha256": _sha(commented_stablehlo),
            }
        )
    broken = report.model_copy(update={"captures": tuple(captures)})
    with pytest.raises(ValueError, match="STABLEHLO_STATIC_ABI_MISMATCH"):
        validate_compile_capture_report(broken, contract, source, source_blobs)

    other_strategy_index = next(
        index for index, value in enumerate(report.captures) if value.strategy is not first.strategy
    )
    unbalanced = report.captures[other_strategy_index].model_copy(
        update={"strategy": first.strategy}
    )
    captures = list(report.captures)
    captures[other_strategy_index] = unbalanced
    broken = report.model_copy(update={"captures": tuple(captures)})
    with pytest.raises(ValueError, match="COMPILE_INVENTORY_MISMATCH"):
        validate_compile_capture_report(broken, contract, source, source_blobs)


def test_compile_failure_aborts_the_one_shot_attempt(contract, source, source_blobs) -> None:
    report = _compile_report(contract, source)
    failed = report.captures[0].model_copy(
        update={
            "status": SurfaceCompileStatus.FAILED,
            "error_sha256": "e" * 64,
        }
    )
    broken = report.model_copy(update={"captures": (failed, *report.captures[1:])})
    with pytest.raises(ValueError, match="COMPILE_FAILED_NO_RETRY"):
        validate_compile_capture_report(broken, contract, source, source_blobs)


def test_correctness_and_calibration_artifacts_are_exact_and_writer_validated(
    contract,
    source,
    source_blobs,
    tmp_path,
) -> None:
    compile_report = _compile_report(contract, source)
    correctness = _correctness_artifact(contract, compile_report, source)
    batch = _calibration_batch(contract, compile_report, source)
    validate_surface_correctness_artifact(
        correctness,
        contract,
        compile_report,
        source,
        source_blobs,
    )
    assert correctness.split is MatmulCollectiveSurfaceSplit.CALIBRATION
    assert len(correctness.observations) == 160
    validate_surface_calibration_batch(batch, contract, compile_report, source, source_blobs)

    failed = correctness.observations[0].model_copy(update={"mismatched_element_count": 1})
    broken = correctness.model_copy(
        update={"observations": (failed, *correctness.observations[1:])}
    )
    output = tmp_path / "invalid-correctness.json"
    with pytest.raises(ValueError, match="CORRECTNESS_FAILED"):
        write_surface_correctness_artifact(
            output,
            broken,
            contract,
            compile_report,
            source,
            source_blobs,
        )
    assert not output.exists()

    inconsistent_oracle = correctness.observations[5].model_copy(
        update={"oracle_output_sha256": "f" * 64}
    )
    broken = correctness.model_copy(
        update={
            "observations": (
                *correctness.observations[:5],
                inconsistent_oracle,
                *correctness.observations[6:],
            )
        }
    )
    with pytest.raises(ValueError, match="CORRECTNESS_ORACLE_MISMATCH"):
        validate_surface_correctness_artifact(
            broken,
            contract,
            compile_report,
            source,
            source_blobs,
        )

    invalid_error = correctness.observations[0].model_copy(update={"maximum_absolute_error": -1.0})
    broken = correctness.model_copy(
        update={"observations": (invalid_error, *correctness.observations[1:])}
    )
    output = tmp_path / "invalid-error-correctness.json"
    with pytest.raises(ValidationError):
        write_surface_correctness_artifact(
            output,
            broken,
            contract,
            compile_report,
            source,
            source_blobs,
        )
    assert not output.exists()

    dropped = batch.model_copy(update={"observations": batch.observations[1:]})
    with pytest.raises(ValidationError):
        validate_surface_calibration_batch(
            dropped,
            contract,
            compile_report,
            source,
            source_blobs,
        )

    nonfinite = batch.observations[0].model_copy(update={"median_ns": float("nan")})
    broken_batch = batch.model_copy(update={"observations": (nonfinite, *batch.observations[1:])})
    output = tmp_path / "nonfinite-calibration.json"
    with pytest.raises(ValidationError):
        write_surface_calibration_batch(
            output,
            broken_batch,
            contract,
            compile_report,
            source,
            source_blobs,
        )
    assert not output.exists()

    ledger = _calibration_ledger(compile_report, correctness, batch)
    forged_events = list(ledger.events)
    forged_events[1] = forged_events[1].model_copy(update={"artifact_sha256": "0" * 64})
    forged_ledger = ledger.model_copy(update={"events": tuple(forged_events)})
    with pytest.raises(ValueError, match="PHASE_ARTIFACT_MISMATCH"):
        seal_surface_calibration(
            forged_ledger,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )


def test_phase_ledger_requires_seal_before_holdout_and_forbids_retry() -> None:
    ledger = SurfacePhaseLedger(attempt_id="f" * 64)
    ledger = record_surface_phase(ledger, SurfacePhase.COMPILE, "1" * 64)
    ledger = record_surface_phase(ledger, SurfacePhase.CORRECTNESS, "2" * 64)
    ledger = record_surface_phase(ledger, SurfacePhase.CALIBRATION, "3" * 64)
    with pytest.raises(ValueError, match="PHASE_ORDER"):
        record_surface_phase(ledger, SurfacePhase.HOLDOUT, "4" * 64)
    with pytest.raises(ValueError, match="PHASE_ORDER"):
        record_surface_phase(ledger, SurfacePhase.COMPILE, "1" * 64)


def test_calibration_seal_binds_model_and_all_holdout_predictions(
    contract, source, source_blobs, tmp_path
) -> None:
    compile_report = _compile_report(contract, source)
    correctness = _correctness_artifact(contract, compile_report, source)
    batch = _calibration_batch(contract, compile_report, source)
    ledger, envelope = seal_surface_calibration(
        _calibration_ledger(compile_report, correctness, batch),
        contract,
        compile_report,
        source,
        source_blobs,
        correctness,
        batch,
    )
    validate_calibration_seal(
        envelope,
        ledger,
        contract,
        compile_report,
        source,
        source_blobs,
        correctness,
        batch,
    )
    assert ledger.current_phase is SurfacePhase.CALIBRATION_SEALED
    assert len(envelope.seal.holdout_predictions) == 8
    forged_schema_seal = envelope.seal.model_copy(update={"schema_version": "attacker-v999"})
    forged_schema_envelope = envelope.model_copy(
        update={
            "seal": forged_schema_seal,
            "seal_sha256": forged_schema_seal.semantic_sha256,
        }
    )
    with pytest.raises(ValidationError):
        validate_calibration_seal(
            forged_schema_envelope,
            ledger,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )
    compile_path = tmp_path / "compile.json"
    correctness_path = tmp_path / "correctness.json"
    batch_path = tmp_path / "calibration.json"
    seal_path = tmp_path / "calibration-seal.json"
    write_compile_capture_report(compile_path, compile_report, contract, source, source_blobs)
    write_surface_correctness_artifact(
        correctness_path,
        correctness,
        contract,
        compile_report,
        source,
        source_blobs,
    )
    write_surface_calibration_batch(
        batch_path,
        batch,
        contract,
        compile_report,
        source,
        source_blobs,
    )
    write_calibration_seal(
        seal_path,
        envelope,
        ledger,
        contract,
        compile_report,
        source,
        source_blobs,
        correctness,
        batch,
    )
    assert (
        replay_compile_capture_report(compile_path, contract, source, source_blobs)
        == compile_report
    )
    assert (
        replay_calibration_seal(
            seal_path,
            ledger,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )
        == envelope
    )
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_EXISTS"):
        write_calibration_seal(
            seal_path,
            envelope,
            ledger,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )
    ledger = begin_surface_holdout(
        ledger,
        seal_path,
        contract,
        compile_report,
        source,
        source_blobs,
        correctness,
        batch,
    )
    holdout_correctness = _correctness_artifact(
        contract,
        compile_report,
        source,
        MatmulCollectiveSurfaceSplit.HOLDOUT,
    )
    holdout_ledger = record_surface_holdout_correctness(
        ledger,
        holdout_correctness,
        contract,
        compile_report,
        source,
        source_blobs,
    )
    assert len(holdout_correctness.observations) == 40
    assert holdout_ledger.current_phase is SurfacePhase.HOLDOUT_CORRECTNESS

    with pytest.raises(ValueError, match="HOLDOUT_CORRECTNESS_PHASE_MISMATCH"):
        record_surface_holdout_correctness(
            ledger,
            correctness,
            contract,
            compile_report,
            source,
            source_blobs,
        )

    forged_ledger = _calibration_ledger(compile_report, correctness, batch)
    forged_ledger = record_surface_phase(
        forged_ledger,
        SurfacePhase.CALIBRATION_SEALED,
        "0" * 64,
    )
    with pytest.raises(ValueError, match="HOLDOUT_SEAL_LEDGER_MISMATCH"):
        begin_surface_holdout(
            forged_ledger,
            seal_path,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )

    coefficients = list(envelope.seal.model.coefficients)
    coefficients[0] += 1.0
    forged_model = envelope.seal.model.model_copy(update={"coefficients": tuple(coefficients)})
    forged_seal = envelope.seal.model_copy(update={"model": forged_model})
    repaired_outer_hash = CalibrationSealEnvelope(
        seal=forged_seal,
        seal_sha256=forged_seal.semantic_sha256,
    )
    forged_seal_events = list(ledger.events[:4])
    forged_seal_events[-1] = forged_seal_events[-1].model_copy(
        update={"artifact_sha256": repaired_outer_hash.seal_sha256}
    )
    forged_seal_ledger = ledger.model_copy(update={"events": tuple(forged_seal_events)})
    with pytest.raises(ValueError, match="CALIBRATION_MODEL_REPLAY_MISMATCH"):
        validate_calibration_seal(
            repaired_outer_hash,
            forged_seal_ledger,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )

    predictions = list(envelope.seal.holdout_predictions)
    predictions[0] = predictions[0].model_copy(
        update={"predicted_median_ns": predictions[0].predicted_median_ns + 1.0}
    )
    forged_seal = envelope.seal.model_copy(update={"holdout_predictions": tuple(predictions)})
    repaired_outer_hash = CalibrationSealEnvelope(
        seal=forged_seal,
        seal_sha256=forged_seal.semantic_sha256,
    )
    forged_seal_events = list(ledger.events[:4])
    forged_seal_events[-1] = forged_seal_events[-1].model_copy(
        update={"artifact_sha256": repaired_outer_hash.seal_sha256}
    )
    forged_seal_ledger = ledger.model_copy(update={"events": tuple(forged_seal_events)})
    with pytest.raises(ValueError, match="HOLDOUT_PREDICTION_REPLAY_MISMATCH"):
        validate_calibration_seal(
            repaired_outer_hash,
            forged_seal_ledger,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )

    with pytest.raises(ValueError, match="PHASE_ORDER"):
        seal_surface_calibration(
            ledger,
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )


@pytest.mark.parametrize("kind", [SurfaceEvidenceKind.TRACE, SurfaceEvidenceKind.COUNTERS])
def test_profile_data_cannot_enter_calibration_fit(contract, source, source_blobs, kind) -> None:
    compile_report = _compile_report(contract, source)
    correctness = _correctness_artifact(contract, compile_report, source)
    observations = list(_observations(contract))
    observations[0] = observations[0].model_copy(update={"evidence_kind": kind})
    batch = SurfaceCalibrationBatch(
        design_id=contract.design_id,
        compile_report_sha256=compile_report.report_sha256,
        source_authority_sha256=source.authority_sha256,
        observations=tuple(observations),
    )
    with pytest.raises(ValueError, match="PROFILE_DATA_FORBIDDEN_IN_FIT"):
        seal_surface_calibration(
            _calibration_ledger(compile_report, correctness, batch),
            contract,
            compile_report,
            source,
            source_blobs,
            correctness,
            batch,
        )

    with pytest.raises(ValidationError):
        SurfaceCalibrationObservation.model_validate(
            {
                "scenario_name": "calibration-0",
                "strategy": contract.strategies[0],
                "median_ns": 1.0,
                "trace_cycles": 12,
            }
        )


def test_attempt_root_is_write_once(tmp_path: Path) -> None:
    invalid_root = tmp_path / "invalid-surface-attempt"
    with pytest.raises(ValidationError):
        create_surface_attempt_root(invalid_root, "not-a-sha256")
    assert not invalid_root.exists()

    root = tmp_path / "surface-attempt"
    create_surface_attempt_root(root, "9" * 64)
    assert root.is_dir()
    with pytest.raises(ValueError, match="ATTEMPT_ROOT_EXISTS"):
        create_surface_attempt_root(root, "9" * 64)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_measurements_are_rejected(contract, value: float) -> None:
    with pytest.raises(ValidationError):
        SurfaceMeasurement(
            scenario_name="calibration-0",
            strategy=contract.strategies[0],
            evidence_kind=SurfaceEvidenceKind.UNPROFILED_TIMING,
            median_ns=value,
        )
