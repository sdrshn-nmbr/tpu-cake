from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import ml_dtypes
import numpy as np

_SCHEMA = "matmul-collective-surface-correctness-executor-v1"
_PROTOCOL_SCHEMA = "matmul-collective-surface-correctness-protocol-v1"
_DESIGN_SCHEMA = "matmul-collective-surface-design-v1"
_EVIDENCE_SCHEMA = "matmul-collective-surface-correctness-evidence-v1"
_MANIFEST_SCHEMA = "matmul-collective-surface-correctness-manifest-v1"
_RECEIPT_SCHEMA = "matmul-collective-surface-correctness-receipt-v1"
_PHASE_LEDGER_SCHEMA = "matmul-collective-surface-phase-ledger-v1"
_PATTERN_SCHEMA = "structured-bf16-analytical-v1"
_EXPECTED_PROTOCOL_ID = "f5f81b36f9542334163bb86d893dc5fd4c1856c7ccc7d8023e29509a4876ccea"
_EXPECTED_PROTOCOL_FILE_SHA256 = "29b7a4a8d1a2215c69dfd5976d714e0090d1c2dbfe642d4605fd2882ada69027"
_PATTERNS = (
    "constant",
    "one-hot-stripes",
    "signed-periodic",
    "block-diagonal",
    "low-rank",
)
_STRATEGIES = ("xla_reduce_scatter", "pallas_bidirectional_ring")
_STATES = ("created", "verified", "lowered", "compiled", "correct", "validated", "accepted")
_PHASES = (
    "compile",
    "correctness",
    "calibration",
    "calibration_sealed",
    "holdout",
    "holdout_correctness",
)
_SOURCE_DEPENDENCIES = (
    "contracts/matmul-collective-surface-correctness-v1.json",
    "contracts/matmul-collective-surface-design-v1.json",
    "tpu_cake/__init__.py",
    "tpu_cake/artifacts.py",
    "tpu_cake/canonical.py",
    "tpu_cake/compiler_analysis.py",
    "tpu_cake/contracts.py",
    "tpu_cake/cost_model.py",
    "tpu_cake/dialects/__init__.py",
    "tpu_cake/dialects/distributed_tensor.py",
    "tpu_cake/dialects/tpu_schedule.py",
    "tpu_cake/distributed_frontend.py",
    "tpu_cake/evidence.py",
    "tpu_cake/frontend.py",
    "tpu_cake/identity.py",
    "tpu_cake/ledger.py",
    "tpu_cake/lowering.py",
    "tpu_cake/matmul_collective_surface_correctness.py",
    "tpu_cake/matmul_collective_surface_correctness_evidence.py",
    "tpu_cake/matmul_collective_surface_correctness_executor.py",
    "tpu_cake/matmul_collective_surface_correctness_oracle.py",
    "tpu_cake/matmul_collective_surface_correctness_protocol.py",
    "tpu_cake/matmul_collective_surface_correctness_verifier.py",
    "tpu_cake/matmul_collective_surface_correctness_worker.py",
    "tpu_cake/matmul_collective_surface_prediction.py",
    "tpu_cake/matmul_collective_surface_runner.py",
    "tpu_cake/metrics.py",
    "tpu_cake/pallas_lowering.py",
    "tpu_cake/receipt.py",
    "tpu_cake/receipt_metrics.py",
    "tpu_cake/rpa_lowering.py",
    "tpu_cake/rpa_owned_kernel.py",
    "tpu_cake/runner.py",
    "tpu_cake/search.py",
    "tpu_cake/source.py",
    "tpu_cake/stablehlo.py",
    "tpu_cake/workloads/__init__.py",
    "tpu_cake/workloads/distributed_matmul.py",
    "tpu_cake/workloads/inkling_rpa.py",
    "tpu_cake/workloads/matmul.py",
    "tpu_cake/xprof_evidence.py",
)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SIGNED_LHS = np.asarray(
    (1, -2, 3, -4, 2, -1, 4, -3, -1, 3, -2, 4, -4, 2, -3, 1),
    dtype=np.int64,
)
_SIGNED_RHS = np.asarray(
    (2, 1, -3, 4, -1, -4, 3, -2, 4, -3, 1, -2, 3, 2, -4, -1),
    dtype=np.int64,
)
_BF16 = np.dtype(ml_dtypes.bfloat16)
_F32 = np.dtype("<f4")
_FORMULAS = {
    "constant_formula": "A=1;B=2^-17;C=K*2^-17",
    "one_hot_formula": (
        "L=K/8;p=(i%8)*L+((257*i+17)%L);A=1[k=p];"
        "c=(j//16+3*(k%32)+5*(k//L))%8;"
        "B=(-1 if c>=4 else +1)*(c%4+1)*2^-3;C=B[p,j]"
    ),
    "signed_formula": (
        "s=k//(K/8);A=a[(k+i)%16]*2^-4;B=b[(k+3*j)%16]*(s+1)*2^-15;"
        "C=(K/128)*36*2^-19*sum_r(a[(r+i)%16]*b[(r+3*j)%16])"
    ),
    "block_formula": (
        "rb=16*i//M;kb=16*k//K;cb=16*j//N;A=1[rb=kb];B=2^-14*1[kb=cb];C=(K/16)*2^-14*1[rb=cb]"
    ),
    "low_rank_formula": (
        "q=[1,(-1)^bit0(k),(-1)^bit1(k),(-1)^bit2(k)];"
        "u=[1,i%3-1,(+1 if i%4<2 else -1),(+1 if i%5<3 else -1)];"
        "v=[(+1 if j%2=0 else -1),j%3-1,(+1 if j%4 in {0,3} else -1),"
        "(+1 if j%5<2 else -1)];"
        "A=sum(u*q);B=2^-17*sum(q*v);C=K*2^-17*sum(u*v)"
    ),
}


@dataclass(frozen=True)
class SurfaceCorrectnessVerification:
    attempt_id: str
    protocol_id: str
    split: str
    source_authority_sha256: str
    execution_authority_sha256: str
    evidence_sha256: str
    ledger_sha256: str
    phase_ledger_sha256: str
    receipt_sha256: str
    case_count: int
    execution_count: int

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_JSON_CONSTANT value={value}")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_JSON_NONFINITE value={value}")
    return parsed


def _pairs_to_dict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_JSON_DUPLICATE_KEY key={key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_pairs_to_dict,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_JSON_INVALID path={path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"SURFACE_CORRECTNESS_INDEPENDENT_JSON_OBJECT_REQUIRED path={path}")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_{label}_SCHEMA_MISMATCH")


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TypeError(f"SURFACE_CORRECTNESS_INDEPENDENT_{label}_INVALID")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_{label}_INVALID")
    return value


def _require_finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_{label}_INVALID")
    return float(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity_sha256(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _semantic_sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"tpu-cake-semantic-identity\x00length-prefixed-v2\x00")
    for part in parts:
        if not part:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SEMANTIC_IDENTITY_EMPTY")
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _semantic_seed(*parts: str) -> int:
    return int.from_bytes(bytes.fromhex(_semantic_sha256(*parts))[:8], "big")


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"SURFACE_CORRECTNESS_INDEPENDENT_{label}_INVALID")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not value
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"SURFACE_CORRECTNESS_INDEPENDENT_{label}_INVALID")
    return value


def _validate_archive_tree(root: Path) -> None:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ROOT_INVALID")
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ARCHIVE_LINK")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ARCHIVE_HARDLINK")
        if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ARCHIVE_FILE_TYPE")


def _validate_protocol(path: Path, recorded_path: Path) -> tuple[dict[str, Any], str, str]:
    supplied = _read_json(path)
    recorded = _read_json(recorded_path)
    if supplied != recorded:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PROTOCOL_MISMATCH")
    _expect_keys(
        recorded,
        {
            "schema_version",
            "parent_compile",
            "scenarios",
            "calibration_scenarios",
            "holdout_scenarios",
            "initial_execution_split",
            "strategies",
            "patterns",
            "numpy_version",
            "ml_dtypes_version",
            "logical_input_dtype",
            "lhs_sharding",
            "rhs_sharding",
            "output_dtype",
            "output_sharding",
            "output_file_format",
            "save_global_inputs",
            "save_oracle_outputs",
            "save_candidate_outputs",
            "absolute_tolerance",
            "relative_tolerance",
            "mismatch_rule",
            "normalized_error_rule",
            "shard_identity_schema",
            "sentinel_rule",
            "sentinel_count_per_shard",
            "strategy_order_rule",
            "correctness_repetitions_per_strategy",
            "compile_continuity_rule",
            "attempt_registry_root",
            "allow_retry",
            "one_shot_attempt_ledger",
        },
        "PROTOCOL",
    )
    calibration = [f"calibration-{index}" for index in range(16)]
    holdout = [f"holdout-{index}" for index in range(4)]
    patterns = recorded["patterns"]
    if not isinstance(patterns, dict):
        raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_PATTERN_CONTRACT_INVALID")
    _expect_keys(
        patterns,
        {
            "schema_version",
            "ordered_patterns",
            "constant_formula",
            "one_hot_formula",
            "signed_lhs_sequence",
            "signed_rhs_sequence",
            "signed_formula",
            "block_formula",
            "low_rank_formula",
        },
        "PATTERN_CONTRACT",
    )
    if (
        recorded["schema_version"] != _PROTOCOL_SCHEMA
        or recorded["scenarios"] != calibration + holdout
        or recorded["calibration_scenarios"] != calibration
        or recorded["holdout_scenarios"] != holdout
        or recorded["initial_execution_split"] != "calibration"
        or recorded["strategies"] != list(_STRATEGIES)
        or patterns["schema_version"] != _PATTERN_SCHEMA
        or patterns["ordered_patterns"] != list(_PATTERNS)
        or patterns["signed_lhs_sequence"] != _SIGNED_LHS.tolist()
        or patterns["signed_rhs_sequence"] != _SIGNED_RHS.tolist()
        or any(patterns[key] != formula for key, formula in _FORMULAS.items())
        or recorded["numpy_version"] != np.__version__
        or recorded["ml_dtypes_version"] != ml_dtypes.__version__
        or recorded["logical_input_dtype"] != "bfloat16"
        or recorded["lhs_sharding"] != "PartitionSpec(None, 't')"
        or recorded["rhs_sharding"] != "PartitionSpec('t', None)"
        or recorded["output_dtype"] != "float32"
        or recorded["output_sharding"] != "PartitionSpec(None, 't')"
        or recorded["output_file_format"] != "npy-allow-pickle-false-v1"
        or recorded["save_global_inputs"] is not False
        or recorded["save_oracle_outputs"] is not True
        or recorded["save_candidate_outputs"] is not True
        or recorded["absolute_tolerance"] != 0.001
        or recorded["relative_tolerance"] != 0.001
        or recorded["sentinel_count_per_shard"] != 32
        or recorded["correctness_repetitions_per_strategy"] != 2
        or recorded["mismatch_rule"] != "abs(candidate-oracle)>atol+rtol*abs(oracle)-v1"
        or recorded["normalized_error_rule"] != "abs(candidate-oracle)/(atol+rtol*abs(oracle))-v1"
        or recorded["shard_identity_schema"]
        != "logical-dtype-global-shape-sharding-device-slice-host-callback-payload-device-sentinels-v1"
        or recorded["sentinel_rule"]
        != "pattern-support-plus-32-semantic-coordinates-per-device-shard-v1"
        or recorded["strategy_order_rule"] != "pattern-parity-abba-baab-v1"
        or recorded["compile_continuity_rule"]
        != "fresh-schedule-pallas-semantic-stablehlo-semantic-compilerhlo-v1"
        or recorded["allow_retry"] is not False
        or recorded["one_shot_attempt_ledger"] is not True
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PROTOCOL_CONTRACT_MISMATCH")
    parent = recorded["parent_compile"]
    if not isinstance(parent, dict):
        raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_AUTHORITY_INVALID")
    _expect_keys(
        parent,
        {
            "archive_path",
            "manifest_file_sha256",
            "compile_report_file_sha256",
            "design_id",
            "attempt_id",
            "source_commit",
            "source_authority_sha256",
            "execution_authority_sha256",
            "compile_report_sha256",
            "compile_ledger_sha256",
        },
        "PARENT_AUTHORITY",
    )
    for key in parent:
        if key.endswith("sha256") or key in {"design_id", "attempt_id"}:
            _require_hex(parent[key], _HEX_64, f"PARENT_{key.upper()}")
    _require_hex(parent["source_commit"], _HEX_40, "PARENT_SOURCE_COMMIT")
    protocol_id = _identity_sha256(recorded)
    protocol_file_sha256 = _file_sha256(path)
    if (
        protocol_id != _EXPECTED_PROTOCOL_ID
        or protocol_file_sha256 != _EXPECTED_PROTOCOL_FILE_SHA256
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_CANONICAL_PROTOCOL_MISMATCH")
    return recorded, protocol_id, protocol_file_sha256


def _validate_design(path: Path, recorded_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    supplied = _read_json(path)
    recorded = _read_json(recorded_path)
    if supplied != recorded or recorded.get("schema_version") != _DESIGN_SCHEMA:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_DESIGN_MISMATCH")
    scenarios = recorded.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 20:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_DESIGN_INVENTORY_MISMATCH")
    observed = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_SCENARIO_INVALID")
        _expect_keys(scenario, {"name", "split", "m", "k", "n", "tile_m", "tile_n"}, "SCENARIO")
        name = scenario["name"]
        split = scenario["split"]
        if name not in protocol[f"{split}_scenarios"]:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SCENARIO_SPLIT_MISMATCH")
        for dimension in ("m", "k", "n", "tile_m", "tile_n"):
            _require_int(scenario[dimension], f"SCENARIO_{dimension.upper()}", minimum=1)
        if scenario["m"] % 16 or scenario["k"] % 16 or scenario["n"] % 16:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SCENARIO_SHAPE_INVALID")
        observed.append(name)
    if (
        observed != protocol["scenarios"]
        or recorded.get("identity_schema") != "length-prefixed-v2"
        or recorded.get("source_branch") != "main"
        or recorded.get("require_origin_main") is not True
        or recorded.get("source_remote_url") != "https://github.com/sdrshn-nmbr/tpu-cake.git"
        or recorded.get("compilation_source_root") != "/home/sudarshan/tpu-cake-main"
        or recorded.get("project") != "astral-medley-465922-b2"
        or recorded.get("zone") != "us-central1-c"
        or recorded.get("hostname") != "tpu-cake-v7x-rsag-wx7r"
        or recorded.get("machine_type") != "tpu7x-standard-4t"
        or recorded.get("device_count") != 8
        or recorded.get("device_ids") != list(range(8))
        or recorded.get("device_process_index") != 0
        or recorded.get("mesh_size") != 8
        or recorded.get("strategies") != list(_STRATEGIES)
        or recorded.get("correctness_patterns") != list(_PATTERNS)
        or _identity_sha256(recorded) != protocol["parent_compile"]["design_id"]
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_DESIGN_CONTRACT_MISMATCH")
    return recorded


def _verify_parent(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    parent_root = root / "parent_compile"
    parent = protocol["parent_compile"]
    if (
        _file_sha256(parent_root / "manifest.json") != parent["manifest_file_sha256"]
        or _file_sha256(parent_root / "compile_report.json") != parent["compile_report_file_sha256"]
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_FILE_HASH_MISMATCH")
    completed = subprocess.run(
        [
            sys.executable,
            str(parent_root / "source/verifier.py"),
            "--root",
            str(parent_root),
            "--contract",
            str(parent_root / "contract.json"),
        ],
        cwd="/",
        env={"HOME": "/nonexistent", "LC_ALL": "C", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        verification = json.loads(
            completed.stdout,
            object_pairs_hook=_pairs_to_dict,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except json.JSONDecodeError as error:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_VERIFIER_INVALID") from error
    expected = {
        "attempt_id": parent["attempt_id"],
        "design_id": parent["design_id"],
        "source_authority_sha256": parent["source_authority_sha256"],
        "execution_authority_sha256": parent["execution_authority_sha256"],
        "compile_report_sha256": parent["compile_report_sha256"],
        "ledger_sha256": parent["compile_ledger_sha256"],
    }
    if not isinstance(verification, dict) or any(
        verification.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_AUTHORITY_MISMATCH")
    manifest = _read_json(parent_root / "manifest.json")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_IDENTITY_INVALID")
    copied_claim = root / "parent_compile_claim.json"
    if _file_sha256(copied_claim) != identity.get("attempt_claim_sha256"):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_CLAIM_MISMATCH")
    _read_json(copied_claim)
    report = _read_json(parent_root / "compile_report.json")
    if _identity_sha256(report) != parent["compile_report_sha256"]:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_REPORT_MISMATCH")
    return report


def _validate_authority(
    root: Path,
    protocol: dict[str, Any],
    protocol_id: str,
    protocol_file_sha256: str,
    design: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    authority = _read_json(root / "execution_authority.json")
    _expect_keys(
        authority,
        {
            "schema_version",
            "protocol_id",
            "protocol_file_sha256",
            "source",
            "executor_source_sha256",
            "worker_source_sha256",
            "verifier_source_sha256",
            "generator_source_sha256",
            "oracle_source_sha256",
            "protocol_source_sha256",
            "evidence_source_sha256",
            "project",
            "zone",
            "hostname",
            "numeric_project_id",
            "instance_id",
            "instance_hostname",
            "machine_type",
            "cpu_platform",
            "compiler_environment",
            "devices",
        },
        "EXECUTION_AUTHORITY",
    )
    source = authority["source"]
    if not isinstance(source, dict):
        raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_SOURCE_AUTHORITY_INVALID")
    _expect_keys(
        source,
        {
            "source_commit",
            "branch",
            "origin_main_commit",
            "remote_main_commit",
            "remote_url",
            "source_root",
            "runtime",
            "uv_lock_sha256",
            "dependencies",
        },
        "SOURCE_AUTHORITY",
    )
    commit = _require_hex(source["source_commit"], _HEX_40, "SOURCE_COMMIT")
    expected_runtime = {
        **design["runtime"],
        "numpy": protocol["numpy_version"],
        "ml_dtypes": protocol["ml_dtypes_version"],
    }
    if (
        source["origin_main_commit"] != commit
        or source["remote_main_commit"] != commit
        or source["branch"] != "main"
        or source["remote_url"] != design["source_remote_url"]
        or source["source_root"] != design["compilation_source_root"]
        or source["runtime"] != expected_runtime
        or authority["schema_version"] != _SCHEMA
        or authority["protocol_id"] != protocol_id
        or authority["protocol_file_sha256"] != protocol_file_sha256
        or any(
            authority[key] != design[key]
            for key in (
                "project",
                "zone",
                "hostname",
                "numeric_project_id",
                "instance_id",
                "instance_hostname",
                "machine_type",
                "cpu_platform",
                "compiler_environment",
            )
        )
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_EXECUTION_AUTHORITY_MISMATCH")
    devices = authority["devices"]
    if not isinstance(devices, list) or devices != [
        {"id": index, "process_index": 0, "platform": "tpu", "device_kind": "TPU7x"}
        for index in range(8)
    ]:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_DEVICE_AUTHORITY_MISMATCH")
    dependencies = source["dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_SOURCE_DEPENDENCIES_INVALID")
    hashes: dict[str, str] = {}
    paths = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_SOURCE_DEPENDENCY_INVALID")
        _expect_keys(dependency, {"path", "sha256"}, "SOURCE_DEPENDENCY")
        path = _canonical_relative_path(dependency["path"], "SOURCE_DEPENDENCY_PATH")
        paths.append(path)
        hashes[path] = _require_hex(dependency["sha256"], _HEX_64, "SOURCE_DEPENDENCY_HASH")
    if tuple(paths) != _SOURCE_DEPENDENCIES:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SOURCE_DEPENDENCY_ORDER_MISMATCH")
    bundle = root / "source/committed"
    observed = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if observed != {*paths, "uv.lock"}:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SOURCE_BUNDLE_INVENTORY_MISMATCH")
    if any(_file_sha256(bundle / path) != digest for path, digest in hashes.items()):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SOURCE_BUNDLE_HASH_MISMATCH")
    if (
        hashes.get("contracts/matmul-collective-surface-correctness-v1.json")
        != protocol_file_sha256
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PROTOCOL_SOURCE_HASH_MISMATCH")
    uv_hash = _require_hex(source["uv_lock_sha256"], _HEX_64, "UV_LOCK_HASH")
    if _file_sha256(bundle / "uv.lock") != uv_hash:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_UV_LOCK_HASH_MISMATCH")
    operational = {
        "executor": "tpu_cake/matmul_collective_surface_correctness_executor.py",
        "worker": "tpu_cake/matmul_collective_surface_correctness_worker.py",
        "verifier": "tpu_cake/matmul_collective_surface_correctness_verifier.py",
        "generator": "tpu_cake/matmul_collective_surface_correctness.py",
        "oracle": "tpu_cake/matmul_collective_surface_correctness_oracle.py",
        "protocol": "tpu_cake/matmul_collective_surface_correctness_protocol.py",
        "evidence": "tpu_cake/matmul_collective_surface_correctness_evidence.py",
    }
    for label, dependency_path in operational.items():
        field = f"{label}_source_sha256"
        digest = _require_hex(authority[field], _HEX_64, field.upper())
        if hashes.get(dependency_path) != digest:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_OPERATIONAL_DEPENDENCY_MISMATCH")
        copy = root / "source" / f"{label}.py"
        if _file_sha256(copy) != digest:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_OPERATIONAL_SOURCE_HASH_MISMATCH")
    if _file_sha256(Path(__file__)) != authority["verifier_source_sha256"]:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_VERIFIER_SOURCE_HASH_MISMATCH")
    return authority, _identity_sha256(source), _identity_sha256(authority)


def _validate_identity_and_claim(
    root: Path,
    protocol: dict[str, Any],
    protocol_id: str,
    authority: dict[str, Any],
    execution_authority_sha256: str,
) -> dict[str, Any]:
    identity = _read_json(root / "run_identity.json")
    _expect_keys(
        identity,
        {
            "attempt_id",
            "protocol_id",
            "split",
            "execution_authority_sha256",
            "attempt_claim_path",
            "attempt_claim_sha256",
            "output_root",
        },
        "RUN_IDENTITY",
    )
    attempt_id = _require_hex(identity["attempt_id"], _HEX_64, "ATTEMPT_ID")
    split = identity["split"]
    if split not in {"calibration", "holdout"}:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SPLIT_INVALID")
    if split == "holdout":
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_HOLDOUT_NOT_AUTHORIZED")
    claim_key = _sha256_bytes(f"{protocol_id}:{split}".encode())
    expected_claim_path = str(Path(protocol["attempt_registry_root"]) / f"{claim_key}.json")
    claim = {
        "schema_version": _SCHEMA,
        "attempt_id": attempt_id,
        "protocol_id": protocol_id,
        "split": split,
        "source_commit": authority["source"]["source_commit"],
        "output_root": identity["output_root"],
        "state": "claimed",
    }
    if (
        identity["protocol_id"] != protocol_id
        or identity["execution_authority_sha256"] != execution_authority_sha256
        or identity["attempt_claim_path"] != expected_claim_path
        or identity["attempt_claim_sha256"] != _sha256_bytes(_pretty_json_bytes(claim))
        or not isinstance(identity["output_root"], str)
        or not Path(identity["output_root"]).is_absolute()
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ATTEMPT_IDENTITY_MISMATCH")
    copied_claim = root / "attempt_claim.json"
    if (
        _file_sha256(copied_claim) != identity["attempt_claim_sha256"]
        or _read_json(copied_claim) != claim
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ATTEMPT_CLAIM_MISMATCH")
    return identity


def _semantic_compiler_hlo(value: str) -> str:
    canonical = value.rstrip("\n") + "\n"
    metadata_start = canonical.find("\nFileNames\n")
    if metadata_start >= 0:
        starts = tuple(
            offset
            for marker in ("\n%", "\nENTRY ")
            if (offset := canonical.find(marker, metadata_start)) >= 0
        )
        if not starts:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_COMPILER_HLO_INVALID")
        canonical = canonical[:metadata_start] + canonical[min(starts) :]
    return re.sub(r" stack_frame_id=\d+", "", canonical)


def _parent_consensus(parent_report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    captures = parent_report.get("captures")
    if not isinstance(captures, list):
        raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_CAPTURES_INVALID")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for capture in captures:
        if not isinstance(capture, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_CAPTURE_INVALID")
        grouped.setdefault((capture["scenario_name"], capture["strategy"]), []).append(capture)
    result = {}
    keys = (
        "distributed_schedule_sha256",
        "physical_schedule_sha256",
        "pallas_source_sha256",
        "semantic_stablehlo_sha256",
        "semantic_compiler_hlo_sha256",
    )
    for key, values in grouped.items():
        if (
            len(values) != 2
            or [value["repetition"] for value in values] != [1, 2]
            or any(len({value[field] for value in values}) != 1 for field in keys)
        ):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PARENT_CONSENSUS_INVALID")
        result[key] = values[0]
    return result


def _validate_continuity(
    root: Path,
    evidence: dict[str, Any],
    scenario_names: list[str],
    parent_report: dict[str, Any],
) -> tuple[dict[tuple[str, str], str], str, str]:
    continuity = evidence.get("continuity")
    expected = [(scenario, strategy) for scenario in scenario_names for strategy in _STRATEGIES]
    if not isinstance(continuity, list) or len(continuity) != len(expected):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_CONTINUITY_INVENTORY_MISMATCH")
    parents = _parent_consensus(parent_report)
    records: dict[tuple[str, str], str] = {}
    schedules: dict[str, list[str]] = {}
    compiled: dict[str, str] = {}
    keys = {
        "scenario_name",
        "strategy",
        "stablehlo_path",
        "stablehlo_file_sha256",
        "compiler_hlo_path",
        "compiler_hlo_file_sha256",
        "parent_distributed_schedule_sha256",
        "observed_distributed_schedule_sha256",
        "parent_physical_schedule_sha256",
        "observed_physical_schedule_sha256",
        "parent_pallas_source_sha256",
        "observed_pallas_source_sha256",
        "parent_semantic_stablehlo_sha256",
        "observed_semantic_stablehlo_sha256",
        "parent_semantic_compiler_hlo_sha256",
        "observed_semantic_compiler_hlo_sha256",
    }
    for record, pair in zip(continuity, expected, strict=True):
        if not isinstance(record, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_CONTINUITY_RECORD_INVALID")
        _expect_keys(record, keys, "CONTINUITY_RECORD")
        scenario, strategy = pair
        base = f"continuity/{scenario}/{strategy}"
        stable_path = f"{base}/stablehlo.txt"
        compiler_path = f"{base}/compiler_hlo.txt"
        parent = parents[pair]
        stable = (root / stable_path).read_text()
        compiler = (root / compiler_path).read_text()
        identity_pairs = (
            (
                "distributed_schedule_sha256",
                "parent_distributed_schedule_sha256",
                "observed_distributed_schedule_sha256",
            ),
            (
                "physical_schedule_sha256",
                "parent_physical_schedule_sha256",
                "observed_physical_schedule_sha256",
            ),
            (
                "pallas_source_sha256",
                "parent_pallas_source_sha256",
                "observed_pallas_source_sha256",
            ),
            (
                "semantic_stablehlo_sha256",
                "parent_semantic_stablehlo_sha256",
                "observed_semantic_stablehlo_sha256",
            ),
            (
                "semantic_compiler_hlo_sha256",
                "parent_semantic_compiler_hlo_sha256",
                "observed_semantic_compiler_hlo_sha256",
            ),
        )
        if (
            (record["scenario_name"], record["strategy"]) != pair
            or record["stablehlo_path"] != stable_path
            or record["compiler_hlo_path"] != compiler_path
            or record["stablehlo_file_sha256"] != _file_sha256(root / stable_path)
            or record["compiler_hlo_file_sha256"] != _file_sha256(root / compiler_path)
            or record["observed_semantic_stablehlo_sha256"]
            != _sha256_bytes((stable.rstrip("\n") + "\n").encode())
            or record["observed_semantic_compiler_hlo_sha256"]
            != _sha256_bytes(_semantic_compiler_hlo(compiler).encode())
            or any(
                parent[parent_key] != record[parent_field]
                or record[parent_field] != record[observed_field]
                for parent_key, parent_field, observed_field in identity_pairs
            )
        ):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_COMPILE_CONTINUITY_MISMATCH")
        digest = _identity_sha256(record)
        records[pair] = digest
        label = f"{scenario}:{strategy}"
        schedules[label] = [
            record["observed_distributed_schedule_sha256"],
            record["observed_physical_schedule_sha256"],
            record["observed_pallas_source_sha256"],
        ]
        compiled[label] = digest
    return records, _identity_sha256(schedules), _identity_sha256(compiled)


def _sentinel_coordinates(
    pattern: str,
    role: str,
    *,
    protocol_id: str,
    scenario_name: str,
    m: int,
    k: int,
    n: int,
    device_id: int,
) -> tuple[tuple[int, int], ...]:
    local_k = k // 8
    k_start = device_id * local_k
    k_stop = (device_id + 1) * local_k
    first_bounds = (0, m) if role == "lhs" else (k_start, k_stop)
    second_bounds = (k_start, k_stop) if role == "lhs" else (0, n)
    coordinates: set[tuple[int, int]] = set()

    def add(first: int, second: int) -> None:
        if (
            first_bounds[0] <= first < first_bounds[1]
            and second_bounds[0] <= second < second_bounds[1]
        ):
            coordinates.add((first, second))

    for first_fraction, second_fraction in (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
    ):
        add(
            first_bounds[0] + (first_bounds[1] - first_bounds[0] - 1) * first_fraction // 3,
            second_bounds[0] + (second_bounds[1] - second_bounds[0] - 1) * second_fraction // 3,
        )
    if pattern == "one-hot-stripes" and role == "lhs":
        matching = tuple(row for row in range(m) if row % 8 == device_id)
        for row in (*matching[:4], *matching[-4:]):
            add(row, device_id * local_k + ((257 * row + 17) % local_k))
    elif pattern == "one-hot-stripes":
        for column in (0, min(15, n - 1), min(16, n - 1), n - 1):
            add(k_start, column)
            add(k_stop - 1, column)
    elif pattern == "block-diagonal":
        for block in (device_id * 2, device_id * 2 + 1):
            reduction = block * k // 16
            if role == "lhs":
                add(block * m // 16, reduction)
                add(((block + 1) % 16) * m // 16, reduction)
            else:
                add(reduction, block * n // 16)
                add(reduction, ((block + 1) % 16) * n // 16)
    seed = _semantic_seed(
        protocol_id,
        scenario_name,
        pattern,
        role,
        str(device_id),
        "surface-correctness-sentinels-v1",
    )
    first_size = first_bounds[1] - first_bounds[0]
    second_size = second_bounds[1] - second_bounds[0]
    counter = 0
    while len(coordinates) < 32:
        flat = (seed + counter * 0x9E3779B97F4A7C15) % (first_size * second_size)
        add(first_bounds[0] + flat // second_size, second_bounds[0] + flat % second_size)
        counter += 1
    return tuple(sorted(coordinates)[:32])


def _operand_chunk(
    pattern: str,
    role: str,
    *,
    m: int,
    k: int,
    n: int,
    k_start: int,
    k_stop: int,
    first_start: int,
    first_stop: int,
) -> np.ndarray:
    local_k = k // 8
    if role == "lhs":
        rows = np.arange(first_start, first_stop, dtype=np.int64)[:, None]
        reduction = np.arange(k_start, k_stop, dtype=np.int64)
        if pattern == "constant":
            return np.full((rows.size, local_k), 1.0, dtype=_BF16)
        if pattern == "one-hot-stripes":
            positions = (rows % 8) * local_k + ((257 * rows + 17) % local_k)
            return np.ascontiguousarray((positions == reduction[None, :]).astype(_BF16))
        if pattern == "signed-periodic":
            return np.ascontiguousarray(
                (_SIGNED_LHS[(rows + reduction[None, :]) % 16] * 2.0**-4).astype(_BF16)
            )
        if pattern == "block-diagonal":
            return np.ascontiguousarray(
                (((16 * rows) // m) == ((16 * reduction) // k)[None, :]).astype(_BF16)
            )
        q = np.stack(
            (
                np.ones(local_k, dtype=np.int8),
                np.where(reduction & 1, -1, 1),
                np.where(reduction & 2, -1, 1),
                np.where(reduction & 4, -1, 1),
            )
        )
        r = rows[:, 0]
        u = np.stack(
            (
                np.ones(r.size, dtype=np.int8),
                (r % 3 - 1).astype(np.int8),
                np.where(r % 4 < 2, 1, -1),
                np.where(r % 5 < 3, 1, -1),
            ),
            axis=1,
        )
        return np.ascontiguousarray((u @ q).astype(_BF16))
    reduction = np.arange(first_start, first_stop, dtype=np.int64)
    columns = np.arange(n, dtype=np.int64)[None, :]
    if pattern == "constant":
        return np.full((reduction.size, n), 2.0**-17, dtype=_BF16)
    if pattern == "one-hot-stripes":
        code = (
            columns // 16 + 3 * (reduction[:, None] % 32) + 5 * (reduction[:, None] // local_k)
        ) % 8
        return np.ascontiguousarray(
            (np.where(code >= 4, -1.0, 1.0) * ((code % 4) + 1) * 2.0**-3).astype(_BF16)
        )
    if pattern == "signed-periodic":
        weight = k_start // local_k + 1
        return np.ascontiguousarray(
            (_SIGNED_RHS[(reduction[:, None] + 3 * columns) % 16] * weight * 2.0**-15).astype(_BF16)
        )
    if pattern == "block-diagonal":
        return np.ascontiguousarray(
            ((((16 * reduction) // k)[:, None] == (16 * columns) // n) * 2.0**-14).astype(_BF16)
        )
    q = np.stack(
        (
            np.ones(reduction.size, dtype=np.int8),
            np.where(reduction & 1, -1, 1),
            np.where(reduction & 2, -1, 1),
            np.where(reduction & 4, -1, 1),
        )
    )
    c = columns[0]
    v = np.stack(
        (
            np.where(c % 2 == 0, 1, -1),
            (c % 3 - 1).astype(np.int8),
            np.where(np.isin(c % 4, (0, 3)), 1, -1),
            np.where(c % 5 < 2, 1, -1),
        )
    )
    return np.ascontiguousarray((q.T @ v * 2.0**-17).astype(_BF16))


def _operand_identity(
    pattern: str,
    role: str,
    *,
    m: int,
    k: int,
    n: int,
    device: int,
    protocol_id: str,
    scenario_name: str,
) -> tuple[str, dict[tuple[int, int], str]]:
    local_k = k // 8
    k_start = device * local_k
    k_stop = (device + 1) * local_k
    first_size = m if role == "lhs" else local_k
    first_origin = 0 if role == "lhs" else k_start
    digest = hashlib.sha256()
    sentinels: dict[tuple[int, int], str] = {}
    coordinates = _sentinel_coordinates(
        pattern,
        role,
        protocol_id=protocol_id,
        scenario_name=scenario_name,
        m=m,
        k=k,
        n=n,
        device_id=device,
    )
    by_first: dict[int, list[tuple[int, int]]] = {}
    for coordinate in coordinates:
        by_first.setdefault(coordinate[0], []).append(coordinate)
    chunk_rows = max(1, min(first_size, 1024 if role == "rhs" else 128))
    for local_start in range(0, first_size, chunk_rows):
        local_stop = min(local_start + chunk_rows, first_size)
        global_start = first_origin + local_start
        global_stop = first_origin + local_stop
        chunk = _operand_chunk(
            pattern,
            role,
            m=m,
            k=k,
            n=n,
            k_start=k_start,
            k_stop=k_stop,
            first_start=global_start,
            first_stop=global_stop,
        )
        digest.update(chunk.tobytes(order="C"))
        for first in range(global_start, global_stop):
            for coordinate in by_first.get(first, ()):
                local = (
                    (coordinate[0], coordinate[1] - k_start)
                    if role == "lhs"
                    else (coordinate[0] - k_start, coordinate[1])
                )
                chunk_local = (local[0] - local_start, local[1])
                sentinels[coordinate] = (
                    np.asarray(chunk[chunk_local], dtype=_BF16).reshape(1).tobytes().hex()
                )
    return digest.hexdigest(), sentinels


def _oracle(pattern: str, *, m: int, k: int, n: int) -> np.ndarray:
    rows = np.arange(m, dtype=np.int64)[:, None]
    columns = np.arange(n, dtype=np.int64)[None, :]
    if pattern == "constant":
        return np.full((m, n), k * 2.0**-17, dtype=np.float32)
    if pattern == "one-hot-stripes":
        local_k = k // 8
        positions = (rows % 8) * local_k + ((257 * rows + 17) % local_k)
        code = (columns // 16 + 3 * (positions % 32) + 5 * (positions // local_k)) % 8
        return np.ascontiguousarray(
            (np.where(code >= 4, -1.0, 1.0) * ((code % 4) + 1) * 2.0**-3).astype(np.float32)
        )
    if pattern == "signed-periodic":
        left = _SIGNED_LHS[(np.arange(16)[None, :] + rows) % 16]
        right = _SIGNED_RHS[(np.arange(16)[None, :, None] + 3 * columns[:, None, :]) % 16]
        dot = np.sum(left[:, :, None] * right, axis=1, dtype=np.int64)
        return np.ascontiguousarray((dot * (k // 128) * 36 * 2.0**-19).astype(np.float32))
    if pattern == "block-diagonal":
        return np.ascontiguousarray(
            (((16 * rows) // m == (16 * columns) // n) * (k // 16) * 2.0**-14).astype(np.float32)
        )
    r = rows[:, 0]
    c = columns[0]
    lhs = np.stack(
        (
            np.ones(m, dtype=np.int64),
            r % 3 - 1,
            np.where(r % 4 < 2, 1, -1),
            np.where(r % 5 < 3, 1, -1),
        ),
        axis=1,
    )
    rhs = np.stack(
        (
            np.where(c % 2 == 0, 1, -1),
            c % 3 - 1,
            np.where(np.isin(c % 4, (0, 3)), 1, -1),
            np.where(c % 5 < 2, 1, -1),
        )
    )
    return np.ascontiguousarray((lhs @ rhs * k * 2.0**-17).astype(np.float32))


def _load_exact_npy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with path.open("rb") as stream:
        major, minor = np.lib.format.read_magic(stream)
        if (major, minor) == (1, 0):
            observed_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif (major, minor) == (2, 0):
            observed_shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_NPY_VERSION_INVALID")
        offset = stream.tell()
        expected_size = offset + math.prod(observed_shape) * dtype.itemsize
        if (
            observed_shape != shape
            or fortran
            or dtype.hasobject
            or dtype != _F32
            or path.stat().st_size != expected_size
        ):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_NPY_STRUCTURE_INVALID")
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != _F32
        or array.shape != shape
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_NPY_ARRAY_INVALID")
    return array


def _validate_saved_array(
    root: Path,
    value: dict[str, Any],
    expected_path: str,
    expected: np.ndarray | None,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    _expect_keys(
        value,
        {
            "path",
            "file_sha256",
            "array_sha256",
            "shape",
            "dtype",
            "numpy_dtype_str",
            "nan_count",
            "positive_infinity_count",
            "negative_infinity_count",
        },
        "SAVED_ARRAY",
    )
    if (
        value["path"] != expected_path
        or _canonical_relative_path(value["path"], "OUTPUT_PATH") != expected_path
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_OUTPUT_PATH_MISMATCH")
    shape_value = value["shape"]
    if not isinstance(shape_value, list) or len(shape_value) != 2:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_OUTPUT_SHAPE_INVALID")
    shape = tuple(_require_int(item, "OUTPUT_DIMENSION", minimum=1) for item in shape_value)
    if shape != expected_shape:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_OUTPUT_ABI_MISMATCH")
    path = root / expected_path
    array = _load_exact_npy(path, shape)  # type: ignore[arg-type]
    if (
        value["file_sha256"] != _file_sha256(path)
        or value["array_sha256"] != _array_sha256(array)
        or value["dtype"] != "float32"
        or value["numpy_dtype_str"] != "<f4"
        or any(
            value[key] != 0
            for key in ("nan_count", "positive_infinity_count", "negative_infinity_count")
        )
        or (expected is not None and not np.array_equal(array, expected))
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SAVED_ARRAY_MISMATCH")
    return array


def _validate_shard(
    shard: dict[str, Any],
    pattern: str,
    role: str,
    scenario: dict[str, Any],
    device: int,
    protocol_id: str,
) -> None:
    _expect_keys(
        shard,
        {
            "role",
            "shard_index",
            "device_id",
            "process_index",
            "global_shape",
            "sharding",
            "global_slice",
            "local_shape",
            "logical_dtype",
            "numpy_dtype_str",
            "payload_byte_order",
            "host_callback_payload_nbytes",
            "host_callback_payload_sha256",
            "sentinels",
        },
        "SHARD",
    )
    m, k, n = scenario["m"], scenario["k"], scenario["n"]
    local_k = k // 8
    starts = (0, device * local_k) if role == "lhs" else (device * local_k, 0)
    stops = (m, (device + 1) * local_k) if role == "lhs" else ((device + 1) * local_k, n)
    expected_slices = [
        {"start": start, "stop": stop, "step": 1} for start, stop in zip(starts, stops, strict=True)
    ]
    expected_shape = [m, k] if role == "lhs" else [k, n]
    local_shape = [stop - start for start, stop in zip(starts, stops, strict=True)]
    payload_hash, sentinel_hex = _operand_identity(
        pattern,
        role,
        m=m,
        k=k,
        n=n,
        device=device,
        protocol_id=protocol_id,
        scenario_name=scenario["name"],
    )
    coordinates = _sentinel_coordinates(
        pattern,
        role,
        protocol_id=protocol_id,
        scenario_name=scenario["name"],
        m=m,
        k=k,
        n=n,
        device_id=device,
    )
    if (
        shard["role"] != role
        or shard["shard_index"] != device
        or shard["device_id"] != device
        or shard["process_index"] != 0
        or shard["global_shape"] != expected_shape
        or shard["sharding"]
        != ("PartitionSpec(None, 't')" if role == "lhs" else "PartitionSpec('t', None)")
        or shard["global_slice"] != expected_slices
        or shard["local_shape"] != local_shape
        or shard["logical_dtype"] != "bfloat16"
        or shard["numpy_dtype_str"] != "<V2"
        or shard["payload_byte_order"] != "little"
        or shard["host_callback_payload_nbytes"] != math.prod(local_shape) * 2
        or shard["host_callback_payload_sha256"] != payload_hash
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SHARD_IDENTITY_MISMATCH")
    sentinels = shard["sentinels"]
    if not isinstance(sentinels, list) or len(sentinels) != 32:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SENTINEL_INVENTORY_MISMATCH")
    for ordinal, (sentinel, coordinate) in enumerate(zip(sentinels, coordinates, strict=True)):
        if not isinstance(sentinel, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_SENTINEL_INVALID")
        _expect_keys(
            sentinel,
            {
                "ordinal",
                "global_coordinate",
                "local_coordinate",
                "expected_bfloat16_hex",
                "observed_bfloat16_hex",
            },
            "SENTINEL",
        )
        local = [coordinate[0] - starts[0], coordinate[1] - starts[1]]
        expected_hex = sentinel_hex[coordinate]
        if sentinel != {
            "ordinal": ordinal,
            "global_coordinate": list(coordinate),
            "local_coordinate": local,
            "expected_bfloat16_hex": expected_hex,
            "observed_bfloat16_hex": expected_hex,
        }:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SENTINEL_MISMATCH")


def _validate_evidence(
    root: Path,
    evidence: dict[str, Any],
    identity: dict[str, Any],
    protocol: dict[str, Any],
    protocol_id: str,
    protocol_file_sha256: str,
    design: dict[str, Any],
    parent_report: dict[str, Any],
    execution_authority_sha256: str,
    request: dict[str, Any],
    result: dict[str, Any],
) -> tuple[str, str, str]:
    _expect_keys(
        evidence,
        {
            "schema_version",
            "protocol_id",
            "protocol_file_sha256",
            "split",
            "parent_compile_manifest_file_sha256",
            "correctness_execution_authority_sha256",
            "continuity",
            "cases",
        },
        "EVIDENCE",
    )
    split = identity["split"]
    scenario_names = protocol[f"{split}_scenarios"]
    if (
        evidence["schema_version"] != _EVIDENCE_SCHEMA
        or evidence["protocol_id"] != protocol_id
        or evidence["protocol_file_sha256"] != protocol_file_sha256
        or evidence["split"] != split
        or evidence["parent_compile_manifest_file_sha256"]
        != protocol["parent_compile"]["manifest_file_sha256"]
        or evidence["correctness_execution_authority_sha256"] != execution_authority_sha256
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_EVIDENCE_AUTHORITY_MISMATCH")
    records, schedule_set, compile_set = _validate_continuity(
        root, evidence, scenario_names, parent_report
    )
    cases = evidence["cases"]
    expected_cases = [(scenario, pattern) for scenario in scenario_names for pattern in _PATTERNS]
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_CASE_INVENTORY_MISMATCH")
    scenarios = {scenario["name"]: scenario for scenario in design["scenarios"]}
    sequence = 0
    output_paths: set[str] = set()
    for case, (scenario_name, pattern) in zip(cases, expected_cases, strict=True):
        if not isinstance(case, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_CASE_INVALID")
        _expect_keys(case, {"input", "oracle", "executions"}, "CASE")
        input_case = case["input"]
        if not isinstance(input_case, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_INPUT_CASE_INVALID")
        _expect_keys(
            input_case,
            {
                "scenario_name",
                "pattern",
                "protocol_id",
                "pattern_contract_sha256",
                "lhs_shards",
                "rhs_shards",
            },
            "INPUT_CASE",
        )
        if (
            input_case["scenario_name"] != scenario_name
            or input_case["pattern"] != pattern
            or input_case["protocol_id"] != protocol_id
            or input_case["pattern_contract_sha256"] != _identity_sha256(protocol["patterns"])
        ):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_INPUT_AUTHORITY_MISMATCH")
        scenario = scenarios[scenario_name]
        for role in ("lhs", "rhs"):
            shards = input_case[f"{role}_shards"]
            if not isinstance(shards, list) or len(shards) != 8:
                raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_SHARD_INVENTORY_MISMATCH")
            for device, shard in enumerate(shards):
                if not isinstance(shard, dict):
                    raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_SHARD_INVALID")
                _validate_shard(shard, pattern, role, scenario, device, protocol_id)
        lhs_set = _identity_sha256({"shards": input_case["lhs_shards"]})
        rhs_set = _identity_sha256({"shards": input_case["rhs_shards"]})
        expected_oracle = _oracle(pattern, m=scenario["m"], k=scenario["k"], n=scenario["n"])
        oracle_path = f"outputs/{scenario_name}/{pattern}/oracle.npy"
        output_shape = (scenario["m"], scenario["n"])
        oracle = _validate_saved_array(
            root,
            case["oracle"],
            oracle_path,
            expected_oracle,
            output_shape,
        )
        output_paths.add(oracle_path)
        executions = case["executions"]
        pattern_index = _PATTERNS.index(pattern)
        order = (_STRATEGIES[0], _STRATEGIES[1], _STRATEGIES[1], _STRATEGIES[0])
        if pattern_index % 2:
            order = (_STRATEGIES[1], _STRATEGIES[0], _STRATEGIES[0], _STRATEGIES[1])
        repetitions = {strategy: 0 for strategy in _STRATEGIES}
        strategy_hashes: dict[str, list[str]] = {strategy: [] for strategy in _STRATEGIES}
        if not isinstance(executions, list) or len(executions) != 4:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_EXECUTION_INVENTORY_MISMATCH")
        for position, (execution, strategy) in enumerate(
            zip(executions, order, strict=True), start=1
        ):
            if not isinstance(execution, dict):
                raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_EXECUTION_INVALID")
            _expect_keys(
                execution,
                {
                    "sequence",
                    "position",
                    "strategy",
                    "strategy_repetition",
                    "invocation_nonce",
                    "worker_pid",
                    "fresh_compile_record_sha256",
                    "lhs_identity_set_sha256",
                    "rhs_identity_set_sha256",
                    "oracle_array_sha256",
                    "runtime_output_sharding",
                    "output",
                    "mismatched_element_count",
                    "maximum_absolute_error",
                    "maximum_normalized_error",
                },
                "EXECUTION",
            )
            sequence += 1
            repetitions[strategy] += 1
            output_path = (
                f"outputs/{scenario_name}/{pattern}/{strategy}-{repetitions[strategy]}.npy"
            )
            candidate = _validate_saved_array(
                root,
                execution["output"],
                output_path,
                None,
                output_shape,
            )
            if output_path in output_paths:
                raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_OUTPUT_PATH_REUSED")
            output_paths.add(output_path)
            absolute = np.abs(candidate - oracle)
            threshold = protocol["absolute_tolerance"] + protocol["relative_tolerance"] * np.abs(
                oracle
            )
            normalized = absolute / threshold
            mismatches = int(np.count_nonzero(absolute > threshold))
            maximum_absolute = float(absolute.max())
            maximum_normalized = float(normalized.max())
            recorded_absolute = _require_finite(
                execution["maximum_absolute_error"], "MAXIMUM_ABSOLUTE_ERROR"
            )
            recorded_normalized = _require_finite(
                execution["maximum_normalized_error"], "MAXIMUM_NORMALIZED_ERROR"
            )
            if (
                execution["sequence"] != sequence
                or execution["position"] != position
                or execution["strategy"] != strategy
                or execution["strategy_repetition"] != repetitions[strategy]
                or execution["invocation_nonce"] != request["invocation_nonce"]
                or execution["worker_pid"] != result["worker_pid"]
                or execution["fresh_compile_record_sha256"] != records[(scenario_name, strategy)]
                or execution["lhs_identity_set_sha256"] != lhs_set
                or execution["rhs_identity_set_sha256"] != rhs_set
                or execution["oracle_array_sha256"] != case["oracle"]["array_sha256"]
                or execution["runtime_output_sharding"] != "PartitionSpec(None, 't')"
                or execution["mismatched_element_count"] != mismatches
                or recorded_absolute != maximum_absolute
                or recorded_normalized != maximum_normalized
                or mismatches != 0
                or maximum_normalized > 1.0
            ):
                raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_EXECUTION_REPLAY_MISMATCH")
            strategy_hashes[strategy].append(execution["output"]["array_sha256"])
        if any(len(values) != 2 or len(set(values)) != 1 for values in strategy_hashes.values()):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_REPEAT_MISMATCH")
    expected_count = 320 if split == "calibration" else 80
    if sequence != expected_count:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_EXECUTION_COUNT_MISMATCH")
    return _identity_sha256(evidence), schedule_set, compile_set


def _validate_worker(
    root: Path,
    identity: dict[str, Any],
    protocol: dict[str, Any],
    design: dict[str, Any],
    execution_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = _read_json(root / "worker-request.json")
    _expect_keys(
        request,
        {
            "attempt_id",
            "split",
            "invocation_nonce",
            "execution_authority_sha256",
            "compilation_cache_schema",
            "parent_snapshot_path",
            "protocol",
            "design",
        },
        "WORKER_REQUEST",
    )
    nonce = _require_hex(request["invocation_nonce"], _HEX_64, "WORKER_NONCE")
    expected_parent_path = str(Path(identity["output_root"]) / "parent_compile")
    if (
        request["attempt_id"] != identity["attempt_id"]
        or request["split"] != identity["split"]
        or request["execution_authority_sha256"] != execution_authority_sha256
        or request["compilation_cache_schema"] != "isolated-empty-temporary-directory-v1"
        or request["parent_snapshot_path"] != expected_parent_path
        or request["protocol"] != protocol
        or request["design"] != design
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_WORKER_REQUEST_MISMATCH")
    if _read_json(root / "STARTED.json") != {
        "attempt_id": identity["attempt_id"],
        "invocation_nonce": nonce,
        "split": identity["split"],
        "state": "started",
    }:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_WORKER_START_MISMATCH")
    result = _read_json(root / "worker-result.json")
    _expect_keys(
        result,
        {
            "attempt_id",
            "split",
            "invocation_nonce",
            "worker_pid",
            "execution_authority_sha256",
            "evidence",
        },
        "WORKER_RESULT",
    )
    _require_int(result["worker_pid"], "WORKER_PID", minimum=1)
    if (
        result["attempt_id"] != identity["attempt_id"]
        or result["split"] != identity["split"]
        or result["invocation_nonce"] != nonce
        or result["execution_authority_sha256"] != execution_authority_sha256
        or not isinstance(result["evidence"], dict)
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_WORKER_RESULT_MISMATCH")
    evidence = _read_json(root / "evidence.json")
    if evidence != result["evidence"]:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_EVIDENCE_ENVELOPE_MISMATCH")
    return request, result, evidence


def _validate_phase_ledger(
    root: Path, identity: dict[str, Any], protocol: dict[str, Any], evidence_sha256: str
) -> tuple[dict[str, Any], str]:
    ledger = _read_json(root / "phase_ledger.json")
    _expect_keys(ledger, {"schema_version", "attempt_id", "events"}, "PHASE_LEDGER")
    events = ledger["events"]
    split = identity["split"]
    expected_length = 2 if split == "calibration" else 6
    if (
        ledger["schema_version"] != _PHASE_LEDGER_SCHEMA
        or ledger["attempt_id"] != identity["attempt_id"]
        or not isinstance(events, list)
        or len(events) != expected_length
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PHASE_LEDGER_MISMATCH")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TypeError("SURFACE_CORRECTNESS_INDEPENDENT_PHASE_EVENT_INVALID")
        _expect_keys(event, {"sequence", "phase", "artifact_sha256"}, "PHASE_EVENT")
        _require_hex(event["artifact_sha256"], _HEX_64, "PHASE_ARTIFACT")
        if event["sequence"] != index + 1 or event["phase"] != _PHASES[index]:
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PHASE_ORDER_MISMATCH")
    expected_phase = "correctness" if split == "calibration" else "holdout_correctness"
    if (
        events[0]["artifact_sha256"] != protocol["parent_compile"]["compile_report_sha256"]
        or events[-1]["phase"] != expected_phase
        or events[-1]["artifact_sha256"] != evidence_sha256
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_PHASE_BINDING_MISMATCH")
    return ledger, _identity_sha256(ledger)


def _manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _artifact_entries(root: Path, excluded: set[str]) -> list[dict[str, Any]]:
    return [
        _manifest_entry(path, root)
        for path in sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.relative_to(root).as_posix() not in excluded
            ),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]


def _expected_archive_files(
    root: Path,
    identity: dict[str, Any],
    protocol: dict[str, Any],
) -> set[str]:
    expected = {
        "STARTED.json",
        "attempt_claim.json",
        "design.json",
        "evidence.json",
        "execution_authority.json",
        "ledger.sqlite",
        "manifest.json",
        "parent_compile_claim.json",
        "phase_ledger.json",
        "protocol.json",
        "receipt.json",
        "run_identity.json",
        "worker-request.json",
        "worker-result.json",
        "source/executor.py",
        "source/worker.py",
        "source/verifier.py",
        "source/generator.py",
        "source/oracle.py",
        "source/protocol.py",
        "source/evidence.py",
        "source/committed/uv.lock",
        *(f"source/committed/{path}" for path in _SOURCE_DEPENDENCIES),
    }
    expected.update(
        path.relative_to(root).as_posix()
        for path in (root / "parent_compile").rglob("*")
        if path.is_file()
    )
    for scenario in protocol[f"{identity['split']}_scenarios"]:
        for strategy in _STRATEGIES:
            expected.add(f"continuity/{scenario}/{strategy}/stablehlo.txt")
            expected.add(f"continuity/{scenario}/{strategy}/compiler_hlo.txt")
        for pattern in _PATTERNS:
            base = f"outputs/{scenario}/{pattern}"
            expected.add(f"{base}/oracle.npy")
            for strategy in _STRATEGIES:
                expected.add(f"{base}/{strategy}-1.npy")
                expected.add(f"{base}/{strategy}-2.npy")
    return expected


def _validate_ledger(
    path: Path,
    identity: dict[str, Any],
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
    protocol: dict[str, Any],
    evidence: dict[str, Any],
    evidence_sha256: str,
    schedule_set: str,
    compile_set: str,
    phase_ledger_sha256: str,
    artifact_set_sha256: str,
) -> str:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_LEDGER_INTEGRITY_MISMATCH")
        objects = tuple(
            connection.execute(
                "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_autoindex_%' ORDER BY type, name"
            )
        )
        columns = tuple(
            (row[1], row[2], row[3], row[5])
            for row in connection.execute("PRAGMA table_info(events)")
        )
        indexes = tuple(
            (row[1], row[2], row[3], row[4])
            for row in connection.execute("PRAGMA index_list(events)")
        )
        rows = tuple(
            connection.execute(
                "SELECT sequence, run_id, state, timestamp_ns, payload_sha256 FROM events ORDER BY sequence"
            )
        )
        sqlite_sequence = tuple(connection.execute("SELECT name, seq FROM sqlite_sequence"))
    if (
        objects != (("table", "events"), ("table", "sqlite_sequence"))
        or columns
        != (
            ("sequence", "INTEGER", 0, 1),
            ("run_id", "TEXT", 1, 0),
            ("state", "TEXT", 1, 0),
            ("timestamp_ns", "INTEGER", 1, 0),
            ("payload_sha256", "TEXT", 1, 0),
        )
        or indexes != (("sqlite_autoindex_events_1", 1, "u", 0),)
        or len(rows) != 7
        or tuple(row[0] for row in rows) != tuple(range(1, 8))
        or any(row[1] != identity["attempt_id"] for row in rows)
        or tuple(row[2] for row in rows) != _STATES
        or any(
            isinstance(row[3], bool) or not isinstance(row[3], int) or row[3] < 0 for row in rows
        )
        or tuple(row[3] for row in rows) != tuple(sorted(row[3] for row in rows))
        or sqlite_sequence != (("events", 7),)
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_LEDGER_SCHEMA_MISMATCH")
    payloads = (
        {
            "protocol_id": identity["protocol_id"],
            "split": identity["split"],
            "execution_authority_sha256": execution_authority_sha256,
            "attempt_claim_path": identity["attempt_claim_path"],
            "attempt_claim_sha256": identity["attempt_claim_sha256"],
        },
        {
            "source_authority_sha256": source_authority_sha256,
            "parent_compile_manifest_file_sha256": protocol["parent_compile"][
                "manifest_file_sha256"
            ],
            "parent_compile_report_sha256": protocol["parent_compile"]["compile_report_sha256"],
            "executor_source_sha256": authority["executor_source_sha256"],
            "worker_source_sha256": authority["worker_source_sha256"],
            "verifier_source_sha256": authority["verifier_source_sha256"],
            "devices": authority["devices"],
        },
        {"continuity_schedule_set_sha256": schedule_set},
        {"fresh_compile_set_sha256": compile_set},
        {"evidence_sha256": evidence_sha256},
        {"phase_ledger_sha256": phase_ledger_sha256, "artifact_set_sha256": artifact_set_sha256},
        {
            "evidence_sha256": evidence_sha256,
            "validation": "producer-schema-and-artifact-replay-v1",
        },
    )
    if tuple(row[4] for row in rows) != tuple(_identity_sha256(payload) for payload in payloads):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_LEDGER_PAYLOAD_MISMATCH")
    return _file_sha256(path)


def _validate_receipt_and_manifest(
    root: Path,
    identity: dict[str, Any],
    protocol: dict[str, Any],
    evidence_sha256: str,
    phase_ledger_sha256: str,
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
    evidence: dict[str, Any],
    schedule_set: str,
    compile_set: str,
) -> tuple[str, str]:
    pre_receipt = _artifact_entries(root, {"ledger.sqlite", "receipt.json", "manifest.json"})
    artifact_set_sha256 = _identity_sha256(
        {entry["path"]: [entry["size_bytes"], entry["sha256"]] for entry in pre_receipt}
    )
    ledger_sha256 = _validate_ledger(
        root / "ledger.sqlite",
        identity,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
        protocol,
        evidence,
        evidence_sha256,
        schedule_set,
        compile_set,
        phase_ledger_sha256,
        artifact_set_sha256,
    )
    receipt = _read_json(root / "receipt.json")
    _expect_keys(
        receipt,
        {
            "schema_version",
            "attempt_id",
            "protocol_id",
            "phase_sequence",
            "phase",
            "split",
            "parent_compile_manifest_file_sha256",
            "previous_phase_receipt_sha256",
            "evidence_file_sha256",
            "evidence_sha256",
            "artifact_set_sha256",
            "ledger_snapshot_sha256",
            "attempt_claim_path",
            "attempt_claim_sha256",
        },
        "RECEIPT",
    )
    split = identity["split"]
    expected_sequence = 1 if split == "calibration" else 6
    expected_phase = "correctness" if split == "calibration" else "holdout_correctness"
    previous = receipt["previous_phase_receipt_sha256"]
    if previous is not None:
        _require_hex(previous, _HEX_64, "PREVIOUS_RECEIPT")
    if (
        receipt["schema_version"] != _RECEIPT_SCHEMA
        or receipt["attempt_id"] != identity["attempt_id"]
        or receipt["protocol_id"] != identity["protocol_id"]
        or receipt["phase_sequence"] != expected_sequence
        or receipt["phase"] != expected_phase
        or receipt["split"] != split
        or receipt["parent_compile_manifest_file_sha256"]
        != protocol["parent_compile"]["manifest_file_sha256"]
        or (split == "calibration") != (previous is None)
        or receipt["evidence_file_sha256"] != _file_sha256(root / "evidence.json")
        or receipt["evidence_sha256"] != evidence_sha256
        or receipt["artifact_set_sha256"] != artifact_set_sha256
        or receipt["ledger_snapshot_sha256"] != ledger_sha256
        or receipt["attempt_claim_path"] != identity["attempt_claim_path"]
        or receipt["attempt_claim_sha256"] != identity["attempt_claim_sha256"]
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_RECEIPT_MISMATCH")
    receipt_sha256 = _identity_sha256(receipt)
    manifest = _read_json(root / "manifest.json")
    _expect_keys(
        manifest,
        {
            "schema_version",
            "identity",
            "evidence_sha256",
            "ledger_snapshot_sha256",
            "receipt_sha256",
            "artifacts",
        },
        "MANIFEST",
    )
    expected_artifacts = _artifact_entries(root, {"manifest.json"})
    expected_files = _expected_archive_files(root, identity, protocol)
    if {entry["path"] for entry in expected_artifacts} != expected_files - {"manifest.json"}:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ARCHIVE_INVENTORY_MISMATCH")
    if (
        manifest["schema_version"] != _MANIFEST_SCHEMA
        or manifest["identity"] != identity
        or manifest["evidence_sha256"] != evidence_sha256
        or manifest["ledger_snapshot_sha256"] != ledger_sha256
        or manifest["receipt_sha256"] != receipt_sha256
        or manifest["artifacts"] != expected_artifacts
    ):
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_MANIFEST_MISMATCH")
    expected_directories = {
        parent.as_posix()
        for entry in expected_artifacts
        for parent in PurePosixPath(entry["path"]).parents
        if parent != PurePosixPath(".")
    }
    observed_directories = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_DIRECTORY_INVENTORY_MISMATCH")
    return ledger_sha256, receipt_sha256


def verify_surface_correctness_independently(
    root: Path,
    protocol_path: Path,
    design_path: Path,
) -> SurfaceCorrectnessVerification:
    if root.is_symlink():
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ROOT_INVALID")
    root = root.resolve(strict=True)
    _validate_archive_tree(root)
    if (root / "failure.json").exists():
        raise ValueError("SURFACE_CORRECTNESS_INDEPENDENT_ATTEMPT_INCOMPLETE")
    protocol, protocol_id, protocol_file_sha256 = _validate_protocol(
        protocol_path, root / "protocol.json"
    )
    design = _validate_design(design_path, root / "design.json", protocol)
    parent_report = _verify_parent(root, protocol)
    authority, source_authority_sha256, execution_authority_sha256 = _validate_authority(
        root, protocol, protocol_id, protocol_file_sha256, design
    )
    identity = _validate_identity_and_claim(
        root, protocol, protocol_id, authority, execution_authority_sha256
    )
    request, result, evidence = _validate_worker(
        root, identity, protocol, design, execution_authority_sha256
    )
    evidence_sha256, schedule_set, compile_set = _validate_evidence(
        root,
        evidence,
        identity,
        protocol,
        protocol_id,
        protocol_file_sha256,
        design,
        parent_report,
        execution_authority_sha256,
        request,
        result,
    )
    _, phase_ledger_sha256 = _validate_phase_ledger(root, identity, protocol, evidence_sha256)
    ledger_sha256, receipt_sha256 = _validate_receipt_and_manifest(
        root,
        identity,
        protocol,
        evidence_sha256,
        phase_ledger_sha256,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
        evidence,
        schedule_set,
        compile_set,
    )
    return SurfaceCorrectnessVerification(
        attempt_id=identity["attempt_id"],
        protocol_id=protocol_id,
        split=identity["split"],
        source_authority_sha256=source_authority_sha256,
        execution_authority_sha256=execution_authority_sha256,
        evidence_sha256=evidence_sha256,
        ledger_sha256=ledger_sha256,
        phase_ledger_sha256=phase_ledger_sha256,
        receipt_sha256=receipt_sha256,
        case_count=len(evidence["cases"]),
        execution_count=sum(len(case["executions"]) for case in evidence["cases"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    args = parser.parse_args()
    print(verify_surface_correctness_independently(args.root, args.protocol, args.design).as_json())


if __name__ == "__main__":
    main()
