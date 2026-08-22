from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from tpu_cake.matmul_collective_surface_compile_verifier import (
    _EXPECTED_ARM_IDENTITIES,
    _EXPECTED_SOURCE_DEPENDENCIES,
    _file_sha256,
    _identity_sha256,
    _pretty_json_bytes,
    _semantic_compiler_hlo,
    _semantic_sha256,
    verify_surface_compile_independently,
)
from tpu_cake.matmul_collective_surface_prediction import (
    default_matmul_collective_surface_design_contract,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _stablehlo(scenario: dict[str, object], strategy: str) -> str:
    collective = (
        "  %2 = stablehlo.reduce_scatter %1 {scatter_dimension = 1 : i64}\n"
        if strategy == "xla_reduce_scatter"
        else ""
    )
    return (
        "module {\n"
        f" func.func public @main(%arg0: tensor<{scenario['m']}x{scenario['k']}xbf16>, "
        f"%arg1: tensor<{scenario['k']}x{scenario['n']}xbf16>) -> "
        f"tensor<{scenario['m']}x{scenario['n']}xf32> {{\n"
        f"  %1 = stablehlo.custom_call @matmul(%arg0, %arg1) : "
        f"(tensor<{scenario['m']}x{scenario['k']}xbf16>, "
        f"tensor<{scenario['k']}x{scenario['n']}xbf16>) -> "
        f"tensor<{scenario['m']}x{scenario['n']}xf32>\n"
        f"{collective}"
        " }\n"
        "}\n"
    )


def _compiler_hlo(scenario: dict[str, object], strategy: str, mesh_size: int) -> str:
    collective = (
        f"rs = f32[{scenario['m']},{scenario['n']}] reduce-scatter(x), "
        'backend_config={"reduce_scatter_offload_config":{"device_type":"DEVICE_TYPE_SPARSECORE"}}\n'
        if strategy == "xla_reduce_scatter"
        else ""
    )
    return (
        "HloModule module, entry_computation_layout="
        f"{{(bf16[{scenario['m']},{scenario['k'] // mesh_size}],"
        f"bf16[{scenario['k'] // mesh_size},{scenario['n']}])"
        f"->f32[{scenario['m']},{scenario['n'] // mesh_size}]}}\n"
        f"{collective}"
    )


def _collectives(strategy: str) -> dict[str, int]:
    one = int(strategy == "xla_reduce_scatter")
    return {
        "stablehlo_reduce_scatter_count": one,
        "stablehlo_all_gather_count": 0,
        "compiler_reduce_scatter_count": one,
        "compiler_all_reduce_count": 0,
        "compiler_all_gather_count": 0,
        "sparse_core_reduce_scatter_count": one,
        "sparse_core_all_gather_count": 0,
    }


def _analysis(stable_path: Path, compiler_path: Path, strategy: str) -> dict[str, object]:
    return {
        "analysis_schema": "compiler-executable-analysis-v2",
        "stablehlo_sha256": _file_sha256(stable_path),
        "compiler_hlo_sha256": _file_sha256(compiler_path),
        "cost_metrics": [{"name": "flops", "raw_value": -2.0, "value": None, "available": False}],
        "memory": {
            "generated_code_size_in_bytes": 0,
            "argument_size_in_bytes": 0,
            "output_size_in_bytes": 0,
            "alias_size_in_bytes": 0,
            "temp_size_in_bytes": 0,
            "host_generated_code_size_in_bytes": 0,
            "host_argument_size_in_bytes": 0,
            "host_output_size_in_bytes": 0,
            "host_alias_size_in_bytes": 0,
            "host_temp_size_in_bytes": 0,
            "peak_memory_in_bytes": 0,
            "buffer_assignment_available": False,
            "buffer_assignment_size_bytes": 0,
            "buffer_assignment_sha256": None,
        },
        "collectives": _collectives(strategy),
    }


def _create_archive(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "archive"
    root.mkdir(mode=0o700)
    repository_root = Path(__file__).parents[1]
    contract = default_matmul_collective_surface_design_contract().model_dump(
        mode="json", exclude_computed_fields=True
    )
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    _write_json(root / "contract.json", contract)
    design_id = _identity_sha256(contract)
    source_dependencies = []
    for relative in _EXPECTED_SOURCE_DEPENDENCIES:
        blob = (repository_root / "src" / relative).read_bytes()
        path = root / "source" / "committed" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        source_dependencies.append({"path": relative, "sha256": hashlib.sha256(blob).hexdigest()})
    uv_blob = (repository_root / "uv.lock").read_bytes()
    (root / "source" / "committed" / "uv.lock").write_bytes(uv_blob)
    executor_blob = (
        repository_root / "src/tpu_cake/matmul_collective_surface_executor.py"
    ).read_bytes()
    worker_blob = (
        repository_root / "src/tpu_cake/matmul_collective_surface_compile_worker.py"
    ).read_bytes()
    verifier_blob = (
        repository_root / "src/tpu_cake/matmul_collective_surface_compile_verifier.py"
    ).read_bytes()
    (root / "source" / "executor.py").write_bytes(executor_blob)
    (root / "source" / "worker.py").write_bytes(worker_blob)
    (root / "source" / "verifier.py").write_bytes(verifier_blob)
    source = {
        "source_commit": "1" * 40,
        "branch": contract["source_branch"],
        "origin_main_commit": "1" * 40,
        "remote_main_commit": "1" * 40,
        "remote_url": contract["source_remote_url"],
        "compilation_source_root": contract["compilation_source_root"],
        "runtime": contract["runtime"],
        "uv_lock_sha256": hashlib.sha256(uv_blob).hexdigest(),
        "dependencies": source_dependencies,
    }
    authority = {
        "schema_version": "matmul-collective-surface-compile-executor-v1",
        "source": source,
        "executor_source_sha256": hashlib.sha256(executor_blob).hexdigest(),
        "worker_source_sha256": hashlib.sha256(worker_blob).hexdigest(),
        "verifier_source_sha256": hashlib.sha256(verifier_blob).hexdigest(),
        **{
            key: contract[key]
            for key in (
                "project",
                "zone",
                "hostname",
                "numeric_project_id",
                "instance_id",
                "instance_hostname",
                "machine_type",
                "cpu_platform",
                "backend",
                "runtime",
                "compiler_environment",
            )
        },
        "devices": [
            {
                "id": index,
                "process_index": 0,
                "platform": contract["backend"],
                "device_kind": contract["device_kind"],
            }
            for index in range(contract["device_count"])
        ],
    }
    _write_json(root / "execution_authority.json", authority)
    execution_authority_sha256 = _identity_sha256(authority)
    attempt_id = "a" * 64
    producer_root = "/producer/evidence/attempt"
    claim_key = _hash(f"{design_id}:{source['source_commit']}")
    claim_path = str(Path(contract["attempt_registry_root"]) / f"{claim_key}.json")
    claim_payload = {
        "schema_version": "matmul-collective-surface-compile-executor-v1",
        "attempt_id": attempt_id,
        "design_id": design_id,
        "source_commit": source["source_commit"],
        "output_root": producer_root,
        "state": "claimed",
    }
    identity = {
        "attempt_id": attempt_id,
        "design_id": design_id,
        "execution_authority_sha256": execution_authority_sha256,
        "producer_output_root": producer_root,
        "attempt_claim_path": claim_path,
        "attempt_claim_sha256": hashlib.sha256(_pretty_json_bytes(claim_payload)).hexdigest(),
    }
    _write_json(root / "attempt.json", {"attempt_id": attempt_id})
    _write_json(root / "run_identity.json", identity)
    worker_captures: dict[tuple[str, str, int], dict[str, object]] = {}
    for repetition in (1, 2):
        nonce = str(repetition) * 64
        request = {
            "attempt_id": attempt_id,
            "repetition": repetition,
            "invocation_nonce": nonce,
            "authority_sha256": execution_authority_sha256,
            "compilation_cache_schema": "isolated-empty-temporary-directory-v1",
            "contract": contract,
        }
        _write_json(root / f"repetition-{repetition}/request.json", request)
        _write_json(
            root / f"repetition-{repetition}/STARTED.json",
            {
                "attempt_id": attempt_id,
                "invocation_nonce": nonce,
                "repetition": repetition,
                "state": "started",
            },
        )
        envelopes = []
        for scenario in contract["scenarios"]:
            workload = _semantic_sha256(
                "matmul-collective-surface-input",
                design_id,
                scenario["name"],
                str(scenario["m"]),
                str(scenario["k"]),
                str(scenario["n"]),
            )
            lhs = _semantic_sha256(workload, "lhs", contract["input_dtype"])
            rhs = _semantic_sha256(workload, "rhs", contract["input_dtype"])
            for strategy in contract["strategies"]:
                base = f"repetition-{repetition}/arms/{scenario['name']}/{strategy}"
                stable_path = root / base / "stablehlo.txt"
                compiler_path = root / base / "compiler_hlo.txt"
                stable_path.parent.mkdir(parents=True, exist_ok=True)
                stable_path.write_text(_stablehlo(scenario, strategy))
                compiler_path.write_text(_compiler_hlo(scenario, strategy, contract["mesh_size"]))
                analysis = _analysis(stable_path, compiler_path, strategy)
                analysis_path = root / base / "compiler_analysis.json"
                _write_json(analysis_path, analysis)
                stablehlo = stable_path.read_text()
                compiler_hlo = compiler_path.read_text()
                capture = {
                    "scenario_name": scenario["name"],
                    "strategy": strategy,
                    "repetition": repetition,
                    "input_contract_sha256": _semantic_sha256(lhs, rhs),
                    "distributed_schedule_sha256": _EXPECTED_ARM_IDENTITIES[
                        (scenario["name"], strategy)
                    ][0],
                    "physical_schedule_sha256": _EXPECTED_ARM_IDENTITIES[
                        (scenario["name"], strategy)
                    ][1],
                    "pallas_source_sha256": _EXPECTED_ARM_IDENTITIES[(scenario["name"], strategy)][
                        2
                    ],
                    "status": "succeeded",
                    "stablehlo": stablehlo,
                    "compiler_hlo": compiler_hlo,
                    "stablehlo_sha256": _hash(stablehlo),
                    "semantic_stablehlo_sha256": _hash(stablehlo.rstrip("\n") + "\n"),
                    "compiler_hlo_sha256": _hash(compiler_hlo),
                    "semantic_compiler_hlo_sha256": _hash(_semantic_compiler_hlo(compiler_hlo)),
                    "error_sha256": None,
                }
                envelope = {
                    "capture": capture,
                    "abstract_input_abi": {
                        "lhs_shape": [scenario["m"], scenario["k"]],
                        "lhs_dtype": "bfloat16",
                        "lhs_sharding": "PartitionSpec(None, 't')",
                        "rhs_shape": [scenario["k"], scenario["n"]],
                        "rhs_dtype": "bfloat16",
                        "rhs_sharding": "PartitionSpec('t', None)",
                        "output_shape": [scenario["m"], scenario["n"]],
                        "output_dtype": "float32",
                        "output_sharding": "PartitionSpec(None, 't')",
                        "schema_version": contract["compile_input_abi_schema"],
                    },
                    "stablehlo_path": f"{base}/stablehlo.txt",
                    "compiler_hlo_path": f"{base}/compiler_hlo.txt",
                    "compiler_analysis_path": f"{base}/compiler_analysis.json",
                    "compiler_analysis": analysis,
                }
                envelopes.append(envelope)
                worker_captures[(scenario["name"], strategy, repetition)] = capture
        _write_json(
            root / f"repetition-{repetition}/result.json",
            {
                "attempt_id": attempt_id,
                "repetition": repetition,
                "invocation_nonce": nonce,
                "worker_pid": 1000 + repetition,
                "authority_sha256": execution_authority_sha256,
                "captures": envelopes,
            },
        )
    report = {
        "schema_version": "matmul-collective-surface-compile-v1",
        "design_id": design_id,
        "source_authority_sha256": _identity_sha256(source),
        "execution_authority_sha256": execution_authority_sha256,
        "captures": [
            worker_captures[(scenario["name"], strategy, repetition)]
            for scenario in contract["scenarios"]
            for strategy in contract["strategies"]
            for repetition in (1, 2)
        ],
    }
    _write_json(root / "compile_report.json", report)
    report_sha256 = _identity_sha256(report)
    arm_identities = {
        f"{capture['scenario_name']}:{capture['strategy']}": [
            capture["physical_schedule_sha256"],
            capture["pallas_source_sha256"],
        ]
        for capture in report["captures"]
        if capture["repetition"] == 1
    }
    payloads = (
        {
            "design_id": design_id,
            "execution_authority_sha256": execution_authority_sha256,
            "attempt_claim_path": claim_path,
            "attempt_claim_sha256": identity["attempt_claim_sha256"],
        },
        {
            "source_authority_sha256": _identity_sha256(source),
            "executor_source_sha256": authority["executor_source_sha256"],
            "worker_source_sha256": authority["worker_source_sha256"],
            "verifier_source_sha256": authority["verifier_source_sha256"],
            "devices": authority["devices"],
        },
        {"arm_identities_sha256": _identity_sha256(arm_identities)},
        {"compile_report_sha256": report_sha256},
    )
    ledger_path = root / "ledger.sqlite"
    with sqlite3.connect(ledger_path) as connection:
        connection.execute(
            "CREATE TABLE events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "run_id TEXT NOT NULL, state TEXT NOT NULL, timestamp_ns INTEGER NOT NULL, "
            "payload_sha256 TEXT NOT NULL, UNIQUE(run_id, state))"
        )
        for sequence, (state, payload) in enumerate(
            zip(("created", "verified", "lowered", "compiled"), payloads, strict=True),
            start=1,
        ):
            connection.execute(
                "INSERT INTO events(run_id, state, timestamp_ns, payload_sha256) "
                "VALUES (?, ?, ?, ?)",
                (attempt_id, state, sequence, _identity_sha256(payload)),
            )
    ledger_sha256 = _file_sha256(ledger_path)
    artifacts = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "matmul-collective-surface-compile-executor-v1",
            "identity": identity,
            "report_sha256": report_sha256,
            "ledger_sha256": ledger_sha256,
            "artifacts": artifacts,
        },
    )
    os.chmod(root, 0o700)
    return root, contract_path


def _rebind_manifest(root: Path, relative: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["artifacts"] if item["path"] == relative)
    path = root / relative
    entry["size"] = path.stat().st_size
    entry["sha256"] = _file_sha256(path)
    _write_json(manifest_path, manifest)


def _fully_rebind_source_mutation(root: Path, relative: str) -> None:
    source_path = root / "source" / "committed" / relative
    source_path.write_bytes(source_path.read_bytes() + b"\nrebound mutation\n")
    authority_path = root / "execution_authority.json"
    authority = json.loads(authority_path.read_text())
    dependency = next(
        item for item in authority["source"]["dependencies"] if item["path"] == relative
    )
    dependency["sha256"] = _file_sha256(source_path)
    _write_json(authority_path, authority)
    source_authority_sha256 = _identity_sha256(authority["source"])
    execution_authority_sha256 = _identity_sha256(authority)

    identity_path = root / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["execution_authority_sha256"] = execution_authority_sha256
    _write_json(identity_path, identity)
    for repetition in (1, 2):
        request_path = root / f"repetition-{repetition}/request.json"
        request = json.loads(request_path.read_text())
        request["authority_sha256"] = execution_authority_sha256
        _write_json(request_path, request)
        result_path = root / f"repetition-{repetition}/result.json"
        result = json.loads(result_path.read_text())
        result["authority_sha256"] = execution_authority_sha256
        _write_json(result_path, result)

    report_path = root / "compile_report.json"
    report = json.loads(report_path.read_text())
    report["source_authority_sha256"] = source_authority_sha256
    report["execution_authority_sha256"] = execution_authority_sha256
    _write_json(report_path, report)
    report_sha256 = _identity_sha256(report)
    arm_identities = {
        f"{capture['scenario_name']}:{capture['strategy']}": [
            capture["physical_schedule_sha256"],
            capture["pallas_source_sha256"],
        ]
        for capture in report["captures"]
        if capture["repetition"] == 1
    }
    payloads = {
        "created": {
            "design_id": identity["design_id"],
            "execution_authority_sha256": execution_authority_sha256,
            "attempt_claim_path": identity["attempt_claim_path"],
            "attempt_claim_sha256": identity["attempt_claim_sha256"],
        },
        "verified": {
            "source_authority_sha256": source_authority_sha256,
            "executor_source_sha256": authority["executor_source_sha256"],
            "worker_source_sha256": authority["worker_source_sha256"],
            "verifier_source_sha256": authority["verifier_source_sha256"],
            "devices": authority["devices"],
        },
        "lowered": {"arm_identities_sha256": _identity_sha256(arm_identities)},
        "compiled": {"compile_report_sha256": report_sha256},
    }
    ledger_path = root / "ledger.sqlite"
    with sqlite3.connect(ledger_path) as connection:
        for state, payload in payloads.items():
            connection.execute(
                "UPDATE events SET payload_sha256 = ? WHERE state = ?",
                (_identity_sha256(payload), state),
            )

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"] = identity
    manifest["report_sha256"] = report_sha256
    manifest["ledger_sha256"] = _file_sha256(ledger_path)
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        artifact["size"] = path.stat().st_size
        artifact["sha256"] = _file_sha256(path)
    _write_json(manifest_path, manifest)


def test_independent_verifier_replays_complete_archive(tmp_path: Path) -> None:
    root, contract_path = _create_archive(tmp_path)

    result = verify_surface_compile_independently(root, contract_path)

    assert result.attempt_id == "a" * 64
    assert result.execution_authority_sha256 == _identity_sha256(
        json.loads((root / "execution_authority.json").read_text())
    )
    assert result.ledger_states == ("created", "verified", "lowered", "compiled")
    assert result.verifier_canonically_bound is True
    assert len(result.captures) == 80
    assert {capture.repetition for capture in result.captures} == {1, 2}


def test_independent_verifier_rejects_manifest_rebound_abi_substitution(tmp_path: Path) -> None:
    root, contract_path = _create_archive(tmp_path)
    result_path = root / "repetition-1/result.json"
    result = json.loads(result_path.read_text())
    result["captures"][0]["abstract_input_abi"]["lhs_shape"][0] += 16
    _write_json(result_path, result)
    _rebind_manifest(root, "repetition-1/result.json")

    with pytest.raises(ValueError, match="ABSTRACT_INPUT_ABI_MISMATCH"):
        verify_surface_compile_independently(root, contract_path)


def test_independent_verifier_rejects_global_shapes_in_compiler_abi(tmp_path: Path) -> None:
    root, contract_path = _create_archive(tmp_path)
    contract = json.loads(contract_path.read_text())
    scenario = contract["scenarios"][0]
    result_path = root / "repetition-1/result.json"
    result = json.loads(result_path.read_text())
    envelope = result["captures"][0]
    capture = envelope["capture"]
    local_abi = (
        f"bf16[{scenario['m']},{scenario['k'] // contract['mesh_size']}],"
        f"bf16[{scenario['k'] // contract['mesh_size']},{scenario['n']}])"
        f"->f32[{scenario['m']},{scenario['n'] // contract['mesh_size']}]"
    )
    global_abi = (
        f"bf16[{scenario['m']},{scenario['k']}],"
        f"bf16[{scenario['k']},{scenario['n']}])->f32[{scenario['m']},{scenario['n']}]"
    )
    compiler_hlo = capture["compiler_hlo"].replace(local_abi, global_abi)
    assert compiler_hlo != capture["compiler_hlo"]
    compiler_path = root / envelope["compiler_hlo_path"]
    compiler_path.write_text(compiler_hlo)
    capture["compiler_hlo"] = compiler_hlo
    capture["compiler_hlo_sha256"] = _hash(compiler_hlo)
    capture["semantic_compiler_hlo_sha256"] = _hash(_semantic_compiler_hlo(compiler_hlo))
    analysis_path = root / envelope["compiler_analysis_path"]
    analysis = json.loads(analysis_path.read_text())
    analysis["compiler_hlo_sha256"] = _file_sha256(compiler_path)
    envelope["compiler_analysis"] = analysis
    _write_json(analysis_path, analysis)
    _write_json(result_path, result)
    _rebind_manifest(root, envelope["compiler_hlo_path"])
    _rebind_manifest(root, envelope["compiler_analysis_path"])
    _rebind_manifest(root, "repetition-1/result.json")

    with pytest.raises(ValueError, match="COMPILER_HLO_ABI_MISMATCH"):
        verify_surface_compile_independently(root, contract_path)


@pytest.mark.parametrize(
    ("raw_value", "value", "available"),
    ((-2.0, -2.0, True), (0.0, None, False)),
)
def test_independent_verifier_rejects_compiler_metric_availability_lies(
    tmp_path: Path,
    raw_value: float,
    value: float | None,
    available: bool,
) -> None:
    root, contract_path = _create_archive(tmp_path)
    result_path = root / "repetition-1/result.json"
    result = json.loads(result_path.read_text())
    envelope = result["captures"][0]
    analysis_path = root / envelope["compiler_analysis_path"]
    analysis = json.loads(analysis_path.read_text())
    mutation = {
        "name": "flops",
        "raw_value": raw_value,
        "value": value,
        "available": available,
    }
    analysis["cost_metrics"][0] = mutation
    envelope["compiler_analysis"]["cost_metrics"][0] = mutation
    _write_json(analysis_path, analysis)
    _write_json(result_path, result)
    _rebind_manifest(root, envelope["compiler_analysis_path"])
    _rebind_manifest(root, "repetition-1/result.json")

    with pytest.raises(ValueError, match="COMPILER_METRIC_INVALID"):
        verify_surface_compile_independently(root, contract_path)


def test_independent_verifier_rejects_fully_rebound_source_mutation(tmp_path: Path) -> None:
    root, contract_path = _create_archive(tmp_path)
    _fully_rebind_source_mutation(root, "tpu_cake/runner.py")

    with pytest.raises(ValueError, match="CANONICAL_SOURCE_HASH_MISMATCH"):
        verify_surface_compile_independently(root, contract_path)


def test_independent_verifier_rejects_noncanonical_contract(tmp_path: Path) -> None:
    root, contract_path = _create_archive(tmp_path)
    contract = json.loads(contract_path.read_text())
    contract["maximum_condition_number"] += 1.0
    _write_json(contract_path, contract)
    _write_json(root / "contract.json", contract)
    _rebind_manifest(root, "contract.json")

    with pytest.raises(ValueError, match="CANONICAL_DESIGN_MISMATCH"):
        verify_surface_compile_independently(root, contract_path)


def test_independent_verifier_rejects_rebound_arm_identity(tmp_path: Path) -> None:
    root, contract_path = _create_archive(tmp_path)
    for repetition in (1, 2):
        result_path = root / f"repetition-{repetition}/result.json"
        result = json.loads(result_path.read_text())
        result["captures"][0]["capture"]["physical_schedule_sha256"] = "f" * 64
        _write_json(result_path, result)
        _rebind_manifest(root, f"repetition-{repetition}/result.json")
    report_path = root / "compile_report.json"
    report = json.loads(report_path.read_text())
    report["captures"][0]["physical_schedule_sha256"] = "f" * 64
    report["captures"][1]["physical_schedule_sha256"] = "f" * 64
    _write_json(report_path, report)
    _rebind_manifest(root, "compile_report.json")

    with pytest.raises(ValueError, match="CANONICAL_ARM_IDENTITY_MISMATCH"):
        verify_surface_compile_independently(root, contract_path)


def test_independent_verifier_rejects_manifest_rebound_hidden_ledger_run(tmp_path: Path) -> None:
    root, contract_path = _create_archive(tmp_path)
    with sqlite3.connect(root / "ledger.sqlite") as connection:
        connection.execute(
            "INSERT INTO events(run_id, state, timestamp_ns, payload_sha256) VALUES (?, ?, ?, ?)",
            ("b" * 64, "created", 5, "c" * 64),
        )
    _rebind_manifest(root, "ledger.sqlite")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["ledger_sha256"] = _file_sha256(root / "ledger.sqlite")
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="LEDGER_SEQUENCE_MISMATCH|LEDGER_STATE_MISMATCH"):
        verify_surface_compile_independently(root, contract_path)


def test_verifier_module_has_no_tpu_or_project_imports() -> None:
    source_path = (
        Path(__file__).parents[1] / "src/tpu_cake/matmul_collective_surface_compile_verifier.py"
    )
    tree = ast.parse(source_path.read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "jax" not in imports
    assert not any(name.startswith("tpu_cake") for name in imports | imported_from)
