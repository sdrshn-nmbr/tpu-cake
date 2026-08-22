from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu_cake.compiler_analysis import (
    CompilerCollectiveAnalysis,
    CompilerCostMetric,
    CompilerExecutableAnalysis,
    CompilerMemoryAnalysis,
)
from tpu_cake.contracts import SourceFileContract
from tpu_cake.ledger import read_ledger_history
from tpu_cake.matmul_collective_surface_executor import (
    SurfaceCompileAbstractInputABI,
    SurfaceCompileCaptureEnvelope,
    SurfaceCompileDevice,
    SurfaceCompileExecutionAuthority,
    SurfaceCompileWorkerRequest,
    SurfaceCompileWorkerResult,
    _canonical_subprocess_environment,
    _compiler_environment,
    _launch_worker,
    _manifest_entries,
    _validate_ledger_closed_world,
    _validate_source_bundle_offline,
    _validate_worker_results,
    compile_surface,
    validate_execution_authority,
    verify_surface_compile,
)
from tpu_cake.matmul_collective_surface_prediction import (
    default_matmul_collective_surface_design_contract,
)
from tpu_cake.matmul_collective_surface_runner import (
    SURFACE_EXECUTABLE_DEPENDENCIES,
    CompileCaptureRecord,
    MatmulCollectiveSurfaceSourceAuthority,
    SurfaceCompileStatus,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_manifest_entries_use_canonical_relative_path_order(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    (tmp_path / "parent").mkdir()
    (tmp_path / "parent-extraction").mkdir()
    (tmp_path / "parent" / "artifact").write_bytes(b"parent")
    (tmp_path / "parent-extraction" / "artifact").write_bytes(b"extraction")

    entries = _manifest_entries(tmp_path)

    assert tuple(entry.path for entry in entries) == (
        "parent-extraction/artifact",
        "parent/artifact",
    )


def _analysis(stablehlo: str, compiler_hlo: str) -> CompilerExecutableAnalysis:
    return CompilerExecutableAnalysis(
        stablehlo_sha256=_hash(stablehlo),
        compiler_hlo_sha256=_hash(compiler_hlo),
        cost_metrics=(CompilerCostMetric(name="flops", raw_value=1.0, value=1.0, available=True),),
        memory=CompilerMemoryAnalysis(
            generated_code_size_in_bytes=0,
            argument_size_in_bytes=0,
            output_size_in_bytes=0,
            alias_size_in_bytes=0,
            temp_size_in_bytes=0,
            host_generated_code_size_in_bytes=0,
            host_argument_size_in_bytes=0,
            host_output_size_in_bytes=0,
            host_alias_size_in_bytes=0,
            host_temp_size_in_bytes=0,
            peak_memory_in_bytes=0,
            buffer_assignment_available=False,
            buffer_assignment_size_bytes=0,
        ),
        collectives=CompilerCollectiveAnalysis(
            stablehlo_reduce_scatter_count=0,
            stablehlo_all_gather_count=0,
            compiler_reduce_scatter_count=0,
            compiler_all_reduce_count=0,
            compiler_all_gather_count=0,
            sparse_core_reduce_scatter_count=0,
            sparse_core_all_gather_count=0,
        ),
    )


def _authority() -> SurfaceCompileExecutionAuthority:
    contract = default_matmul_collective_surface_design_contract()
    source = MatmulCollectiveSurfaceSourceAuthority(
        source_commit="1" * 40,
        branch="main",
        origin_main_commit="1" * 40,
        remote_main_commit="1" * 40,
        remote_url=contract.source_remote_url,
        compilation_source_root=contract.compilation_source_root,
        runtime=contract.runtime,
        uv_lock_sha256=_hash("uv"),
        dependencies=tuple(
            SourceFileContract(path=path, sha256=_hash(f"source:{path}"))
            for path in SURFACE_EXECUTABLE_DEPENDENCIES
        ),
    )
    return SurfaceCompileExecutionAuthority(
        source=source,
        executor_source_sha256=_hash("executor"),
        worker_source_sha256=_hash("worker"),
        verifier_source_sha256=_hash("verifier"),
        project=contract.project,
        zone=contract.zone,
        hostname=contract.hostname,
        numeric_project_id=contract.numeric_project_id,
        instance_id=contract.instance_id,
        instance_hostname=contract.instance_hostname,
        machine_type=contract.machine_type,
        cpu_platform=contract.cpu_platform,
        backend=contract.backend,
        runtime=contract.runtime,
        compiler_environment=contract.compiler_environment,
        devices=tuple(
            SurfaceCompileDevice(
                id=index,
                process_index=0,
                platform=contract.backend,
                device_kind=contract.device_kind,
            )
            for index in range(8)
        ),
    )


def _capture_envelope(root: Path, repetition: int, scenario: object, strategy: object):
    scenario_name = scenario.name
    stablehlo = f"stable-{scenario_name}-{strategy.value}-{repetition}\n"
    compiler_hlo = f"compiler-{scenario_name}-{strategy.value}-{repetition}\n"
    base = f"repetition-{repetition}/arms/{scenario_name}/{strategy.value}"
    stablehlo_path = f"{base}/stablehlo.txt"
    compiler_hlo_path = f"{base}/compiler_hlo.txt"
    analysis_path = f"{base}/compiler_analysis.json"
    (root / base).mkdir(parents=True, exist_ok=True)
    (root / stablehlo_path).write_text(stablehlo)
    (root / compiler_hlo_path).write_text(compiler_hlo)
    analysis = _analysis(stablehlo, compiler_hlo)
    (root / analysis_path).write_text(
        json.dumps(analysis.model_dump(mode="json"), sort_keys=True) + "\n"
    )
    capture = CompileCaptureRecord(
        scenario_name=scenario_name,
        strategy=strategy,
        repetition=repetition,
        input_contract_sha256="5" * 64,
        distributed_schedule_sha256="6" * 64,
        physical_schedule_sha256="7" * 64,
        pallas_source_sha256="8" * 64,
        status=SurfaceCompileStatus.SUCCEEDED,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
        stablehlo_sha256=_hash(stablehlo),
        semantic_stablehlo_sha256=_hash(stablehlo),
        compiler_hlo_sha256=_hash(compiler_hlo),
        semantic_compiler_hlo_sha256=_hash(compiler_hlo),
    )
    return SurfaceCompileCaptureEnvelope(
        capture=capture,
        abstract_input_abi=SurfaceCompileAbstractInputABI(
            lhs_shape=(scenario.m, scenario.k),
            lhs_dtype="bfloat16",
            lhs_sharding="PartitionSpec(None, 't')",
            rhs_shape=(scenario.k, scenario.n),
            rhs_dtype="bfloat16",
            rhs_sharding="PartitionSpec('t', None)",
            output_shape=(scenario.m, scenario.n),
            output_dtype="float32",
            output_sharding="PartitionSpec(None, 't')",
        ),
        stablehlo_path=stablehlo_path,
        compiler_hlo_path=compiler_hlo_path,
        compiler_analysis_path=analysis_path,
        compiler_analysis=analysis,
    )


def _fake_launcher(root: Path, request_path: Path, repetition: int) -> None:
    request = SurfaceCompileWorkerRequest.model_validate_json(request_path.read_text())
    captures = tuple(
        _capture_envelope(root, repetition, scenario, strategy)
        for scenario in request.contract.scenarios
        for strategy in request.contract.strategies
    )
    result = SurfaceCompileWorkerResult(
        attempt_id=request.attempt_id,
        repetition=repetition,
        invocation_nonce=request.invocation_nonce,
        worker_pid=1000 + repetition,
        authority_sha256=request.authority_sha256,
        captures=captures,
    )
    (root / f"repetition-{repetition}/result.json").write_text(
        json.dumps(
            result.model_dump(mode="json", exclude_computed_fields=True),
            sort_keys=True,
        )
        + "\n"
    )


@pytest.fixture
def canonical_contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(
        default_matmul_collective_surface_design_contract().model_dump_json(
            exclude_computed_fields=True
        )
    )
    return path


def _patch_authority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    authority = _authority()
    source_blobs = {path: f"source:{path}".encode() for path in SURFACE_EXECUTABLE_DEPENDENCIES}
    source_blobs["uv.lock"] = b"uv"
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._probe_execution_authority",
        lambda *_args: (authority, source_blobs),
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor.validate_execution_authority",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor.validate_compile_capture_report",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._validate_compile_report_offline",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._executor_source_blob",
        lambda *_args: b"executor",
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._worker_source_blob",
        lambda *_args: b"worker",
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._verifier_source_blob",
        lambda *_args: b"verifier",
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._attempt_registry_root",
        lambda _contract: tmp_path / "registry",
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._launch_worker",
        _fake_launcher,
    )


def test_compile_and_independent_replay_are_closed_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_contract_path: Path,
) -> None:
    _patch_authority(monkeypatch, tmp_path)
    root = tmp_path / "attempt"
    manifest = compile_surface(
        root,
        canonical_contract_path,
        "a" * 64,
    )

    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._probe_execution_authority",
        lambda *_args: (_ for _ in ()).throw(AssertionError("offline replay used live probe")),
    )
    assert manifest == verify_surface_compile(root, canonical_contract_path)
    relocated = tmp_path / "relocated-archive"
    shutil.copytree(root, relocated)
    assert manifest == verify_surface_compile(relocated, canonical_contract_path)
    assert len(tuple(root.glob("repetition-*/arms/*/*/stablehlo.txt"))) == 80
    assert [
        value.state.value for value in read_ledger_history(root / "ledger.sqlite", "a" * 64)
    ] == [
        "created",
        "verified",
        "lowered",
        "compiled",
    ]

    (root / "unexpected.txt").write_text("tamper")
    with pytest.raises(ValueError, match="MANIFEST_MISMATCH"):
        verify_surface_compile(root, canonical_contract_path)


def test_offline_source_bundle_rejects_omitted_canonical_dependency(tmp_path: Path) -> None:
    contract = default_matmul_collective_surface_design_contract()
    authority = _authority()
    source = authority.source.model_copy(
        update={"dependencies": authority.source.dependencies[:-1]}
    )
    forged = authority.model_copy(update={"source": source})

    with pytest.raises(ValueError, match="DEPENDENCY_CLOSURE_MISMATCH"):
        _validate_source_bundle_offline(tmp_path, forged, contract)


def test_hidden_ledger_run_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_contract_path: Path,
) -> None:
    _patch_authority(monkeypatch, tmp_path)
    root = tmp_path / "attempt"
    compile_surface(root, canonical_contract_path, "9" * 64)
    with sqlite3.connect(root / "ledger.sqlite") as connection:
        connection.execute(
            "INSERT INTO events(run_id, state, timestamp_ns, payload_sha256) VALUES (?, ?, ?, ?)",
            ("8" * 64, "created", 0, "7" * 64),
        )
    with pytest.raises(ValueError, match="LEDGER_RUN_INVENTORY_MISMATCH"):
        _validate_ledger_closed_world(root / "ledger.sqlite", "9" * 64)


def test_partial_worker_output_is_permanently_nonretryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_contract_path: Path,
) -> None:
    _patch_authority(monkeypatch, tmp_path)
    root = tmp_path / "attempt"

    def crash_after_first(root: Path, request: Path, repetition: int) -> None:
        if repetition == 2:
            raise RuntimeError("worker crashed")
        _fake_launcher(root, request, repetition)

    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._launch_worker",
        crash_after_first,
    )
    with pytest.raises(RuntimeError, match="worker crashed"):
        compile_surface(root, canonical_contract_path, "b" * 64)
    assert (root / "failure.json").is_file()
    with pytest.raises(ValueError, match="PERMANENTLY_CLAIMED"):
        compile_surface(tmp_path / "different-root", canonical_contract_path, "c" * 64)


def test_repetition_pid_reuse_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    authority = _authority()
    identity = SimpleNamespace(attempt_id="d" * 64)
    requests = tuple(
        SurfaceCompileWorkerRequest(
            attempt_id="d" * 64,
            repetition=repetition,
            invocation_nonce=str(repetition) * 64,
            authority_sha256=authority.authority_sha256,
            contract=contract,
        )
        for repetition in (1, 2)
    )
    for request in requests:
        root = tmp_path / f"repetition-{request.repetition}"
        root.mkdir()
        result = SurfaceCompileWorkerResult(
            attempt_id=request.attempt_id,
            repetition=request.repetition,
            invocation_nonce=request.invocation_nonce,
            worker_pid=42,
            authority_sha256=authority.authority_sha256,
            captures=tuple(
                _capture_envelope(tmp_path, request.repetition, scenario, strategy)
                for scenario in contract.scenarios
                for strategy in contract.strategies
            ),
        )
        (root / "result.json").write_text(result.model_dump_json())

    with pytest.raises(ValueError, match="WORKER_PID_REUSED"):
        _validate_worker_results(
            tmp_path,
            identity,
            authority,
            contract,
            requests,
        )


def test_repetition_identity_reuse_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_contract_path: Path,
) -> None:
    _patch_authority(monkeypatch, tmp_path)

    def reuse_first_repetition(root: Path, request_path: Path, repetition: int) -> None:
        _fake_launcher(root, request_path, repetition)
        if repetition == 2:
            result_path = root / "repetition-2/result.json"
            result = SurfaceCompileWorkerResult.model_validate_json(result_path.read_text())
            result_path.write_text(result.model_copy(update={"repetition": 1}).model_dump_json())

    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._launch_worker",
        reuse_first_repetition,
    )
    with pytest.raises(ValueError, match="WORKER_RESULT_MISMATCH"):
        compile_surface(
            tmp_path / "attempt",
            canonical_contract_path,
            "f" * 64,
        )


def test_cross_shape_abstract_abi_substitution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_contract_path: Path,
) -> None:
    _patch_authority(monkeypatch, tmp_path)

    def substitute(root: Path, request_path: Path, repetition: int) -> None:
        _fake_launcher(root, request_path, repetition)
        result_path = root / f"repetition-{repetition}/result.json"
        result = SurfaceCompileWorkerResult.model_validate_json(result_path.read_text())
        first = result.captures[0]
        next_shape = result.captures[2]
        forged = first.model_copy(update={"abstract_input_abi": next_shape.abstract_input_abi})
        result_path.write_text(
            result.model_copy(update={"captures": (forged, *result.captures[1:])}).model_dump_json()
        )

    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._launch_worker",
        substitute,
    )
    with pytest.raises(ValueError, match="ABI_ANALYSIS_MISMATCH"):
        compile_surface(
            tmp_path / "attempt",
            canonical_contract_path,
            "e" * 64,
        )


def test_source_runtime_and_device_authority_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    authority = _authority()
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor.validate_surface_source_authority",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._executor_source_sha256",
        lambda *_args: authority.executor_source_sha256,
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._worker_source_sha256",
        lambda *_args: authority.worker_source_sha256,
    )
    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor._verifier_source_sha256",
        lambda *_args: authority.verifier_source_sha256,
    )
    validate_execution_authority(authority, contract, {})

    bad_runtime = authority.model_copy(update={"runtime": {**authority.runtime, "jax": "bad"}})
    with pytest.raises(ValueError, match="EXECUTION_AUTHORITY_MISMATCH"):
        validate_execution_authority(bad_runtime, contract, {})
    bad_devices = authority.model_copy(update={"devices": authority.devices[:-1]})
    with pytest.raises(ValueError, match="EXECUTION_AUTHORITY_MISMATCH"):
        validate_execution_authority(bad_devices, contract, {})
    reordered_devices = authority.model_copy(
        update={"devices": (*authority.devices[1:], authority.devices[0])}
    )
    with pytest.raises(ValueError, match="EXECUTION_AUTHORITY_MISMATCH"):
        validate_execution_authority(reordered_devices, contract, {})
    wrong_process = authority.devices[0].model_copy(update={"process_index": 1})
    wrong_process_devices = authority.model_copy(
        update={"devices": (wrong_process, *authority.devices[1:])}
    )
    with pytest.raises(ValueError, match="EXECUTION_AUTHORITY_MISMATCH"):
        validate_execution_authority(wrong_process_devices, contract, {})


@pytest.mark.parametrize("root", [Path("relative"), Path("/"), Path.home()])
def test_unsafe_attempt_roots_are_rejected(root: Path) -> None:
    from tpu_cake.matmul_collective_surface_executor import _require_safe_new_root

    with pytest.raises(ValueError, match="ROOT_NOT_ABSOLUTE|UNSAFE_ROOT"):
        _require_safe_new_root(root)


def test_worker_subprocess_environment_is_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    injected = {
        "PYTHONPATH": "/tmp/shadow",
        "PYTHONHOME": "/tmp/python",
        "VIRTUAL_ENV": "/tmp/venv",
        "UV_PROJECT": "/tmp/project",
        "TPU_LIBRARY_PATH": "/tmp/libtpu.so",
        "XLA_FLAGS": "--attacker",
        "JAX_PLATFORMS": "cpu",
        "PJRT_DEVICE": "CPU",
        "GIT_DIR": "/tmp/forged-git",
        "HTTP_PROXY": "http://attacker.invalid",
        "LD_PRELOAD": "/tmp/attacker.so",
    }
    for key, value in injected.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PATH", "/tmp/attacker-bin")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    environment = _canonical_subprocess_environment(
        contract,
        compilation_cache_dir=cache_root,
    )

    assert all(
        key not in environment for key in injected if key not in contract.compiler_environment
    )
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["LIBTPU_INIT_ARGS"] == contract.compiler_environment["LIBTPU_INIT_ARGS"]
    assert environment["TPU_LIBRARY_PATH"] == contract.compiler_environment["TPU_LIBRARY_PATH"]
    assert environment["JAX_COMPILATION_CACHE_DIR"] == str(cache_root)


def test_worker_launcher_uses_the_required_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    request = SurfaceCompileWorkerRequest(
        attempt_id="a" * 64,
        repetition=1,
        invocation_nonce="b" * 64,
        authority_sha256="c" * 64,
        contract=contract,
    )
    request_root = tmp_path / "repetition-1"
    request_root.mkdir()
    request_path = request_root / "request.json"
    request_path.write_text(request.model_dump_json(exclude_computed_fields=True))
    observed: list[str] = []
    real_run = subprocess.run

    def capture(command, **_kwargs):
        observed.extend(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "tpu_cake.matmul_collective_surface_executor.subprocess.run",
        capture,
    )
    _launch_worker(tmp_path, request_path, 1)
    assert observed[1:4] == [
        "-m",
        "tpu_cake.matmul_collective_surface_compile_worker",
        "worker",
    ]

    environment = _canonical_subprocess_environment(contract)
    real_run(
        [
            sys.executable,
            "-m",
            "tpu_cake.matmul_collective_surface_compile_worker",
            "worker",
            "--help",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("key", ["XLA_FLAGS"])
def test_worker_authority_rejects_undeclared_compiler_environment(
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = default_matmul_collective_surface_design_contract()
    for environment_key in tuple(os.environ):
        if environment_key.startswith(("JAX_", "XLA_", "TPU_", "PJRT_", "LIBTPU_")):
            monkeypatch.delenv(environment_key, raising=False)
    for environment_key, value in contract.compiler_environment.items():
        monkeypatch.setenv(environment_key, value)
    assert _compiler_environment(contract) == contract.compiler_environment

    monkeypatch.setenv(key, "attacker")
    with pytest.raises(ValueError, match="UNDECLARED_ENVIRONMENT"):
        _compiler_environment(contract)
