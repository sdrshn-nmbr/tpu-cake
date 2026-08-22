from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import ml_dtypes
import numpy as np

_PROTOCOL_SCHEMA = "matmul-collective-surface-calibration-protocol-v1"
_DESIGN_SCHEMA = "matmul-collective-surface-design-v1"
_EXECUTOR_SCHEMA = "matmul-collective-surface-calibration-executor-v1"
_EVIDENCE_SCHEMA = "matmul-collective-surface-calibration-evidence-v1"
_SEAL_SCHEMA = "matmul-collective-surface-calibration-sealed-evidence-v1"
_MANIFEST_SCHEMA = "matmul-collective-surface-calibration-manifest-v1"
_RECEIPT_SCHEMA = "matmul-collective-surface-calibration-receipt-v1"
_PHASE_LEDGER_SCHEMA = "matmul-collective-surface-phase-ledger-v1"
_ARRAY_HASH_SCHEMA = "matmul-collective-surface-bootstrap-array-v1"
_EXPECTED_PROTOCOL_ID = "136d8c93106124da136084ba273b5f5ff32a1c29ce564527895e7f11393085c9"
_EXPECTED_PROTOCOL_FILE_SHA256 = "d7b2dcaa050e924bb16d64426faa4b4cee119be1ae760e1671506d89c3f9c48b"
_EXPECTED_DESIGN_ID = "f2f8a0eeba4842167780cd3d79043443d0d02392ed037a5250df1a2218691d83"
_EXPECTED_DESIGN_FILE_SHA256 = "9f4332de116319b0a5ad33703314ff2a7c293a8e9bda70c2c2596c424156f17f"
_STRATEGIES = ("xla_reduce_scatter", "pallas_bidirectional_ring")
_SCENARIOS = tuple(f"calibration-{index}" for index in range(16))
_HOLDOUTS = tuple(f"holdout-{index}" for index in range(4))
_STATES = ("created", "verified", "lowered", "compiled", "correct", "validated", "accepted")
_PHASES = ("compile", "correctness", "calibration", "calibration_sealed")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_F32 = np.dtype("<f4")
_ZSTD_PATH = Path("/usr/bin/zstd")
_SOURCE_DEPENDENCIES = (
    "contracts/matmul-collective-surface-calibration-v1.json",
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
    "tpu_cake/matmul_collective_surface_calibration_archive.py",
    "tpu_cake/matmul_collective_surface_calibration_evidence.py",
    "tpu_cake/matmul_collective_surface_calibration_executor.py",
    "tpu_cake/matmul_collective_surface_calibration_protocol.py",
    "tpu_cake/matmul_collective_surface_calibration_seal.py",
    "tpu_cake/matmul_collective_surface_calibration_verifier.py",
    "tpu_cake/matmul_collective_surface_calibration_worker.py",
    "tpu_cake/matmul_collective_surface_correctness.py",
    "tpu_cake/matmul_collective_surface_correctness_evidence.py",
    "tpu_cake/matmul_collective_surface_correctness_executor.py",
    "tpu_cake/matmul_collective_surface_correctness_oracle.py",
    "tpu_cake/matmul_collective_surface_correctness_protocol.py",
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

# Frozen output of the canonical physical-cost derivation. Keeping the feature
# matrix here makes replay independent of the producer and its package imports.
_PHYSICAL_FEATURES = {
    (name, strategy): values
    for name, strategy, *values in (
        (
            "calibration-0",
            "xla_reduce_scatter",
            58.178469007368875,
            1192.169105691057,
            191.14666666666668,
        ),
        (
            "calibration-0",
            "pallas_bidirectional_ring",
            58.178469007368875,
            1192.169105691057,
            232.10666666666665,
        ),
        (
            "calibration-1",
            "xla_reduce_scatter",
            465.427752058951,
            9273.166395663957,
            191.14666666666668,
        ),
        (
            "calibration-1",
            "pallas_bidirectional_ring",
            465.427752058951,
            9273.166395663957,
            232.10666666666665,
        ),
        (
            "calibration-2",
            "xla_reduce_scatter",
            465.427752058951,
            1580.6785907859078,
            1529.1733333333334,
        ),
        (
            "calibration-2",
            "pallas_bidirectional_ring",
            465.427752058951,
            1580.6785907859078,
            1856.8533333333332,
        ),
        (
            "calibration-3",
            "xla_reduce_scatter",
            3723.422016471608,
            10531.937127371273,
            1529.1733333333334,
        ),
        (
            "calibration-3",
            "pallas_bidirectional_ring",
            3723.422016471608,
            10531.937127371273,
            1856.8533333333332,
        ),
        (
            "calibration-4",
            "xla_reduce_scatter",
            1861.711008235804,
            2912.711111111111,
            6116.693333333334,
        ),
        (
            "calibration-4",
            "pallas_bidirectional_ring",
            1861.711008235804,
            2912.711111111111,
            7427.413333333333,
        ),
        (
            "calibration-5",
            "xla_reduce_scatter",
            14893.688065886432,
            14847.722493224932,
            6116.693333333334,
        ),
        (
            "calibration-5",
            "pallas_bidirectional_ring",
            14893.688065886432,
            14847.722493224932,
            7427.413333333333,
        ),
        ("calibration-6", "xla_reduce_scatter", 2792.566512353706, 3800.732791327913, 9175.04),
        (
            "calibration-6",
            "pallas_bidirectional_ring",
            2792.566512353706,
            3800.732791327913,
            11141.12,
        ),
        ("calibration-7", "xla_reduce_scatter", 22340.532098829648, 17724.91273712737, 9175.04),
        (
            "calibration-7",
            "pallas_bidirectional_ring",
            22340.532098829648,
            17724.91273712737,
            11141.12,
        ),
        (
            "calibration-8",
            "xla_reduce_scatter",
            116.35693801473775,
            2366.5777777777776,
            382.29333333333335,
        ),
        (
            "calibration-8",
            "pallas_bidirectional_ring",
            116.35693801473775,
            2366.5777777777776,
            464.2133333333333,
        ),
        (
            "calibration-9",
            "xla_reduce_scatter",
            581.7846900736888,
            11530.961517615176,
            382.29333333333335,
        ),
        (
            "calibration-9",
            "pallas_bidirectional_ring",
            581.7846900736888,
            11530.961517615176,
            464.2133333333333,
        ),
        (
            "calibration-10",
            "xla_reduce_scatter",
            930.855504117902,
            3019.2737127371274,
            3058.346666666667,
        ),
        (
            "calibration-10",
            "pallas_bidirectional_ring",
            930.855504117902,
            3019.2737127371274,
            3713.7066666666665,
        ),
        (
            "calibration-11",
            "xla_reduce_scatter",
            4654.2775205895105,
            12680.949593495934,
            3058.346666666667,
        ),
        (
            "calibration-11",
            "pallas_bidirectional_ring",
            4654.2775205895105,
            12680.949593495934,
            3713.7066666666665,
        ),
        (
            "calibration-12",
            "xla_reduce_scatter",
            3723.422016471608,
            5257.088346883469,
            12233.386666666667,
        ),
        (
            "calibration-12",
            "pallas_bidirectional_ring",
            3723.422016471608,
            5257.088346883469,
            14854.826666666666,
        ),
        (
            "calibration-13",
            "xla_reduce_scatter",
            18617.110082358042,
            16623.765853658537,
            12233.386666666667,
        ),
        (
            "calibration-13",
            "pallas_bidirectional_ring",
            18617.110082358042,
            16623.765853658537,
            14854.826666666666,
        ),
        ("calibration-14", "xla_reduce_scatter", 5585.133024707412, 6748.964769647697, 18350.08),
        (
            "calibration-14",
            "pallas_bidirectional_ring",
            5585.133024707412,
            6748.964769647697,
            22282.24,
        ),
        ("calibration-15", "xla_reduce_scatter", 27925.665123537063, 19252.31002710027, 18350.08),
        (
            "calibration-15",
            "pallas_bidirectional_ring",
            27925.665123537063,
            19252.31002710027,
            22282.24,
        ),
        (
            "holdout-0",
            "xla_reduce_scatter",
            349.07081404421325,
            3592.0476964769646,
            382.29333333333335,
        ),
        (
            "holdout-0",
            "pallas_bidirectional_ring",
            349.07081404421325,
            3592.0476964769646,
            464.2133333333333,
        ),
        ("holdout-1", "xla_reduce_scatter", 1047.2124421326398, 4022.7382113821136, 2293.76),
        ("holdout-1", "pallas_bidirectional_ring", 1047.2124421326398, 4022.7382113821136, 2785.28),
        (
            "holdout-2",
            "xla_reduce_scatter",
            13962.832561768531,
            12592.147425474255,
            7645.866666666667,
        ),
        (
            "holdout-2",
            "pallas_bidirectional_ring",
            13962.832561768531,
            12592.147425474255,
            9284.266666666666,
        ),
        (
            "holdout-3",
            "xla_reduce_scatter",
            9308.555041179021,
            8986.779403794038,
            15291.733333333334,
        ),
        (
            "holdout-3",
            "pallas_bidirectional_ring",
            9308.555041179021,
            8986.779403794038,
            18568.533333333333,
        ),
    )
}


@dataclass(frozen=True)
class SurfaceCalibrationVerification:
    attempt_id: str
    protocol_id: str
    source_authority_sha256: str
    execution_authority_sha256: str
    correctness_parent_receipt_sha256: str
    evidence_sha256: str
    seal_sha256: str
    ledger_sha256: str
    phase_ledger_sha256: str
    receipt_sha256: str
    sample_count: int
    holdout_authorization: str

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_JSON_CONSTANT value={value}")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_JSON_NONFINITE value={value}")
    return parsed


def _pairs_to_dict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_JSON_DUPLICATE_KEY key={key}")
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
        raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_JSON_INVALID path={path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"SURFACE_CALIBRATION_INDEPENDENT_JSON_OBJECT_REQUIRED path={path}")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_{label}_SCHEMA_MISMATCH")


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TypeError(f"SURFACE_CALIBRATION_INDEPENDENT_{label}_INVALID")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_{label}_INVALID")
    return value


def _require_finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_{label}_INVALID")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_{label}_INVALID")
    return parsed


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity_sha256(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _semantic_sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"tpu-cake-semantic-identity\x00length-prefixed-v2\x00")
    for part in parts:
        if not part:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SEMANTIC_IDENTITY_EMPTY")
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _semantic_seed(*parts: str) -> int:
    return int.from_bytes(bytes.fromhex(_semantic_sha256(*parts))[:8], "big")


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"SURFACE_CALIBRATION_INDEPENDENT_{label}_INVALID")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not value
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"SURFACE_CALIBRATION_INDEPENDENT_{label}_INVALID")
    return value


def _validate_archive_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ROOT_INVALID")
    for path in root.rglob("*"):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode) or (stat.S_ISREG(status.st_mode) and status.st_nlink != 1):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ARCHIVE_LINK_INVALID")
        if not stat.S_ISREG(status.st_mode) and not stat.S_ISDIR(status.st_mode):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ARCHIVE_FILE_TYPE")


def _feature_row(name: str, strategy: str) -> tuple[float, ...]:
    compute, hbm, ici = _PHYSICAL_FEATURES[(name, strategy)]
    first = strategy == _STRATEGIES[0]
    return (
        float(first),
        float(not first),
        compute / 1000.0,
        hbm / 1000.0,
        ici / 1000.0 if first else 0.0,
        0.0 if first else ici / 1000.0,
    )


def _nonnegative_affine_fit(matrix: np.ndarray, observations: np.ndarray) -> np.ndarray:
    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    for count in range(1, matrix.shape[1] + 1):
        for active in itertools.combinations(range(matrix.shape[1]), count):
            candidate = np.zeros(matrix.shape[1], dtype=np.float64)
            solved, *_ = np.linalg.lstsq(matrix[:, active], observations, rcond=None)
            if np.any(solved < -1e-9):
                continue
            candidate[list(active)] = np.maximum(solved, 0.0)
            residual = observations - matrix @ candidate
            score = float(residual @ residual)
            record = (score, active, candidate)
            if best is None or (score, active) < (best[0], best[1]):
                best = record
    if best is None:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_NONNEGATIVE_FIT_FAILED")
    return best[2]


def _bootstrap_array_sha256(value: np.ndarray, dtype: np.dtype[Any]) -> str:
    array = np.asarray(value, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(_ARRAY_HASH_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(array.ndim.to_bytes(8, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_protocol(path: Path, recorded_path: Path) -> tuple[dict[str, Any], str, str]:
    supplied = _read_json(path)
    recorded = _read_json(recorded_path)
    if supplied != recorded:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PROTOCOL_MISMATCH")
    protocol_id = _identity_sha256(recorded)
    file_sha256 = _file_sha256(path)
    if (
        recorded.get("schema_version") != _PROTOCOL_SCHEMA
        or protocol_id != _EXPECTED_PROTOCOL_ID
        or file_sha256 != _EXPECTED_PROTOCOL_FILE_SHA256
        or recorded.get("design_id") != _EXPECTED_DESIGN_ID
        or recorded.get("scenarios") != list(_SCENARIOS)
        or recorded.get("strategies") != list(_STRATEGIES)
        or recorded.get("split") != "calibration"
        or recorded.get("timing_input_pattern") != "signed-periodic"
        or recorded.get("warmup_iterations_per_strategy") != 10
        or recorded.get("calls_per_position") != 5
        or recorded.get("paired_rounds") != 16
        or recorded.get("coefficient_bootstrap_samples") != 10_000
        or recorded.get("coefficient_bootstrap_seed") != 17_012_026
        or recorded.get("allow_profile_data") is not False
        or recorded.get("allow_holdout_materialization") is not False
        or recorded.get("allow_early_stopping") is not False
        or recorded.get("allow_retry") is not False
        or recorded.get("allow_outlier_removal") is not False
        or recorded.get("allow_calibration_refit_after_seal") is not False
        or recorded.get("one_shot_attempt_ledger") is not True
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_CANONICAL_PROTOCOL_MISMATCH")
    parent = recorded.get("correctness_parent")
    if not isinstance(parent, dict):
        raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CONTRACT_INVALID")
    required_parent = {
        "archive_path",
        "archive_filename",
        "archive_root_name",
        "archive_sha256",
        "archive_size_bytes",
        "archive_container_schema",
        "archive_maximum_members",
        "archive_maximum_member_size_bytes",
        "archive_maximum_total_size_bytes",
        "attempt_id",
        "protocol_id",
        "source_commit",
        "source_contract_file_sha256",
        "source_design_file_sha256",
        "archived_protocol_file_sha256",
        "archived_design_file_sha256",
        "manifest_file_sha256",
        "evidence_file_sha256",
        "evidence_sha256",
        "receipt_file_sha256",
        "receipt_sha256",
        "artifact_set_sha256",
        "phase_ledger_file_sha256",
        "phase_ledger_sha256",
        "ledger_file_sha256",
        "run_identity_file_sha256",
        "attempt_claim_file_sha256",
        "execution_authority_file_sha256",
        "execution_authority_sha256",
        "source_authority_sha256",
        "verifier_source_sha256",
        "split",
        "case_count",
        "execution_count",
        "independent_replay_required",
    }
    _expect_keys(parent, required_parent, "PARENT_CONTRACT")
    if (
        parent["split"] != "calibration"
        or parent["case_count"] != 80
        or parent["execution_count"] != 320
        or parent["independent_replay_required"] is not True
        or parent["archive_container_schema"]
        != "tar-zstd-single-root-no-links-no-devices-no-duplicates-v1"
        or parent["archive_maximum_members"] != 2_000
        or parent["archive_maximum_member_size_bytes"] != 1_073_741_824
        or parent["archive_maximum_total_size_bytes"] != 4_294_967_296
        or PurePosixPath(parent["archive_filename"]).name != parent["archive_filename"]
        or PurePosixPath(parent["archive_root_name"]).name != parent["archive_root_name"]
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CONTRACT_MISMATCH")
    for key, value in parent.items():
        if key.endswith("sha256") or key in {"attempt_id", "protocol_id"}:
            _require_hex(value, _HEX_64, f"PARENT_{key.upper()}")
    _require_hex(parent["source_commit"], _HEX_40, "PARENT_SOURCE_COMMIT")
    _require_int(parent["archive_size_bytes"], "PARENT_ARCHIVE_SIZE", minimum=1)
    expected_claim_key = _semantic_sha256(parent["receipt_sha256"], "calibration-timing-v1")
    if (
        recorded.get("attempt_registry_root")
        != ("/home/sudarshan/tpu-cake-evidence/matmul-collective-surface-calibration-attempts-v1")
        or expected_claim_key != "a6334e879bc6d1d2cb9389e28456c86d077722024f4869d3536ca8748da1dc84"
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_CLAIM_CONTRACT_MISMATCH")
    return recorded, protocol_id, file_sha256


def _validate_design(path: Path, recorded_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    supplied = _read_json(path)
    recorded = _read_json(recorded_path)
    if supplied != recorded:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_DESIGN_MISMATCH")
    scenarios = recorded.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 20:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_DESIGN_INVENTORY_MISMATCH")
    expected_names = (*_SCENARIOS, *_HOLDOUTS)
    for expected_name, scenario in zip(expected_names, scenarios, strict=True):
        if not isinstance(scenario, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_SCENARIO_INVALID")
        _expect_keys(scenario, {"name", "split", "m", "k", "n", "tile_m", "tile_n"}, "SCENARIO")
        if scenario["name"] != expected_name or scenario["split"] != expected_name.split("-", 1)[0]:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SCENARIO_ORDER_MISMATCH")
        for field in ("m", "k", "n", "tile_m", "tile_n"):
            _require_int(scenario[field], f"SCENARIO_{field.upper()}", minimum=1)
    if (
        recorded.get("schema_version") != _DESIGN_SCHEMA
        or _identity_sha256(recorded) != _EXPECTED_DESIGN_ID
        or _file_sha256(path) != _EXPECTED_DESIGN_FILE_SHA256
        or protocol["design_id"] != _EXPECTED_DESIGN_ID
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
        or recorded.get("strategies") != list(_STRATEGIES)
        or recorded.get("input_dtype") != "bfloat16"
        or recorded.get("output_dtype") != "float32"
        or recorded.get("predictions_sealed_before_holdout") is not True
        or recorded.get("allow_calibration_refit_after_holdout") is not False
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_DESIGN_CONTRACT_MISMATCH")
    return recorded


def _tar_inventory(archive: Path, parent: dict[str, Any]) -> tuple[tuple[str, int, str], ...]:
    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
    process: subprocess.Popen[bytes] | None = None
    records: list[tuple[str, int, str]] = []
    names: set[str] = set()
    total = 0
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_INVALID")
        process = subprocess.Popen(
            [str(_ZSTD_PATH), "-dc"],
            stdin=descriptor,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_PIPE_INVALID")
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                raw = member.name.removesuffix("/")
                path = PurePosixPath(raw)
                if (
                    not raw
                    or "\\" in raw
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or path.parts[0] != parent["archive_root_name"]
                    or raw in names
                    or getattr(member, "sparse", None)
                ):
                    raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_PATH_INVALID")
                names.add(raw)
                if len(records) + 1 > parent["archive_maximum_members"]:
                    raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_LIMIT")
                if member.isdir():
                    records.append((raw, 0, "directory"))
                    continue
                if not member.isreg():
                    raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_TYPE_INVALID")
                total += member.size
                if (
                    member.size > parent["archive_maximum_member_size_bytes"]
                    or total > parent["archive_maximum_total_size_bytes"]
                ):
                    raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_LIMIT")
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_FILE_MISSING")
                digest = hashlib.sha256()
                remaining = member.size
                while remaining:
                    payload = source.read(min(1_048_576, remaining))
                    if not payload:
                        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_TRUNCATED")
                    digest.update(payload)
                    remaining -= len(payload)
                if source.read(1):
                    raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_OVERRUN")
                source.close()
                records.append((raw, member.size, digest.hexdigest()))
        process.stdout.close()
        stderr = process.stderr.read(4097)
        process.stderr.close()
        return_code = process.wait()
        if return_code or len(stderr) > 4096:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_ZSTD_FAILED")
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        if process is not None:
            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()
            if process.stderr is not None and not process.stderr.closed:
                process.stderr.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        os.close(descriptor)
    if not records or records[0] != (parent["archive_root_name"], 0, "directory"):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_ROOT_INVALID")
    return tuple(records)


def _verify_parent(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    parent = protocol["correctness_parent"]
    archive = root / "parent" / parent["archive_filename"]
    extracted = root / "parent" / parent["archive_root_name"]
    if (
        archive.is_symlink()
        or archive.stat().st_nlink != 1
        or archive.stat().st_size != parent["archive_size_bytes"]
        or _file_sha256(archive) != parent["archive_sha256"]
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ARCHIVE_MISMATCH")
    inventory = _tar_inventory(archive, parent)
    expected_files: dict[str, tuple[int, str]] = {}
    expected_directories: set[str] = set()
    prefix = f"{parent['archive_root_name']}/"
    for path, size, digest in inventory:
        if digest == "directory":
            expected_directories.add(path.removeprefix(prefix))
        else:
            expected_files[path.removeprefix(prefix)] = (size, digest)
    observed_files = {
        path.relative_to(extracted).as_posix(): (path.stat().st_size, _file_sha256(path))
        for path in extracted.rglob("*")
        if path.is_file()
    }
    observed_directories = {
        path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_dir()
    }
    expected_directories.discard(parent["archive_root_name"])
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_EXTRACTION_MISMATCH")
    pinned = {
        "manifest.json": parent["manifest_file_sha256"],
        "evidence.json": parent["evidence_file_sha256"],
        "receipt.json": parent["receipt_file_sha256"],
        "phase_ledger.json": parent["phase_ledger_file_sha256"],
        "ledger.sqlite": parent["ledger_file_sha256"],
        "run_identity.json": parent["run_identity_file_sha256"],
        "attempt_claim.json": parent["attempt_claim_file_sha256"],
        "execution_authority.json": parent["execution_authority_file_sha256"],
        "protocol.json": parent["archived_protocol_file_sha256"],
        "design.json": parent["archived_design_file_sha256"],
        "source/verifier.py": parent["verifier_source_sha256"],
    }
    if any(_file_sha256(extracted / path) != digest for path, digest in pinned.items()):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_FILE_HASH_MISMATCH")
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            str(extracted / "source/verifier.py"),
            "--root",
            str(extracted),
            "--protocol",
            str(
                extracted
                / "source/committed/contracts/matmul-collective-surface-correctness-v1.json"
            ),
            "--design",
            str(extracted / "source/committed/contracts/matmul-collective-surface-design-v1.json"),
        ],
        cwd="/",
        env={"HOME": "/nonexistent", "LC_ALL": "C", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        replay = json.loads(
            completed.stdout,
            object_pairs_hook=_pairs_to_dict,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except json.JSONDecodeError as error:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_REPLAY_INVALID") from error
    expected = {
        "attempt_id": parent["attempt_id"],
        "protocol_id": parent["protocol_id"],
        "source_authority_sha256": parent["source_authority_sha256"],
        "execution_authority_sha256": parent["execution_authority_sha256"],
        "evidence_sha256": parent["evidence_sha256"],
        "ledger_sha256": parent["ledger_file_sha256"],
        "phase_ledger_sha256": parent["phase_ledger_sha256"],
        "receipt_sha256": parent["receipt_sha256"],
        "case_count": parent["case_count"],
        "execution_count": parent["execution_count"],
        "split": parent["split"],
    }
    if not isinstance(replay, dict) or replay != expected:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_REPLAY_MISMATCH")
    parent_evidence = _read_json(extracted / "evidence.json")
    parent_receipt = _read_json(extracted / "receipt.json")
    parent_manifest = _read_json(extracted / "manifest.json")
    parent_ledger = _read_json(extracted / "phase_ledger.json")
    if (
        _identity_sha256(parent_evidence) != parent["evidence_sha256"]
        or _identity_sha256(parent_receipt) != parent["receipt_sha256"]
        or _identity_sha256(parent_ledger) != parent["phase_ledger_sha256"]
        or parent_manifest.get("receipt_sha256") != parent["receipt_sha256"]
        or parent_manifest.get("evidence_sha256") != parent["evidence_sha256"]
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_SEMANTIC_MISMATCH")
    return {"root": extracted, "evidence": parent_evidence, "receipt": parent_receipt}


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
            "design_id",
            "design_file_sha256",
            "source",
            "executor_source_sha256",
            "worker_source_sha256",
            "verifier_source_sha256",
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
        raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_SOURCE_AUTHORITY_INVALID")
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
        "numpy": np.__version__,
        "ml_dtypes": ml_dtypes.__version__,
    }
    if (
        source["origin_main_commit"] != commit
        or source["remote_main_commit"] != commit
        or source["branch"] != "main"
        or source["remote_url"] != design["source_remote_url"]
        or source["source_root"] != design["compilation_source_root"]
        or source["runtime"] != expected_runtime
        or authority["schema_version"]
        != "matmul-collective-surface-calibration-execution-authority-v1"
        or authority["protocol_id"] != protocol_id
        or authority["protocol_file_sha256"] != protocol_file_sha256
        or authority["design_id"] != _EXPECTED_DESIGN_ID
        or authority["design_file_sha256"] != _EXPECTED_DESIGN_FILE_SHA256
        or any(
            authority[field] != design[field]
            for field in (
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
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_EXECUTION_AUTHORITY_MISMATCH")
    if authority["devices"] != [
        {"id": index, "process_index": 0, "platform": "tpu", "device_kind": "TPU7x"}
        for index in range(8)
    ]:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_DEVICE_AUTHORITY_MISMATCH")
    dependencies = source["dependencies"]
    if not isinstance(dependencies, list):
        raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_SOURCE_DEPENDENCIES_INVALID")
    hashes: dict[str, str] = {}
    paths = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_SOURCE_DEPENDENCY_INVALID")
        _expect_keys(dependency, {"path", "sha256"}, "SOURCE_DEPENDENCY")
        path = _canonical_relative_path(dependency["path"], "SOURCE_DEPENDENCY_PATH")
        paths.append(path)
        hashes[path] = _require_hex(dependency["sha256"], _HEX_64, "SOURCE_DEPENDENCY_HASH")
    if tuple(paths) != _SOURCE_DEPENDENCIES:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SOURCE_DEPENDENCY_ORDER_MISMATCH")
    bundle = root / "source/committed"
    observed = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if observed != {*_SOURCE_DEPENDENCIES, "uv.lock"}:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SOURCE_BUNDLE_INVENTORY_MISMATCH")
    if any(_file_sha256(bundle / path) != digest for path, digest in hashes.items()):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SOURCE_BUNDLE_HASH_MISMATCH")
    uv_hash = _require_hex(source["uv_lock_sha256"], _HEX_64, "UV_LOCK_HASH")
    if _file_sha256(bundle / "uv.lock") != uv_hash:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_UV_LOCK_HASH_MISMATCH")
    operational = {
        "executor": "tpu_cake/matmul_collective_surface_calibration_executor.py",
        "worker": "tpu_cake/matmul_collective_surface_calibration_worker.py",
        "verifier": "tpu_cake/matmul_collective_surface_calibration_verifier.py",
    }
    for label, path in operational.items():
        field = f"{label}_source_sha256"
        digest = _require_hex(authority[field], _HEX_64, field.upper())
        if hashes.get(path) != digest or _file_sha256(root / f"source/{label}.py") != digest:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_OPERATIONAL_SOURCE_MISMATCH")
    if _file_sha256(Path(__file__)) != authority["verifier_source_sha256"]:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_VERIFIER_SOURCE_HASH_MISMATCH")
    aliases = {
        "archive.py": "tpu_cake/matmul_collective_surface_calibration_archive.py",
        "evidence.py": "tpu_cake/matmul_collective_surface_calibration_evidence.py",
        "protocol.py": "tpu_cake/matmul_collective_surface_calibration_protocol.py",
        "seal.py": "tpu_cake/matmul_collective_surface_calibration_seal.py",
    }
    if any(
        _file_sha256(root / f"source/{alias}") != hashes[path] for alias, path in aliases.items()
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SOURCE_ALIAS_MISMATCH")
    return authority, _identity_sha256(source), _identity_sha256(authority)


def _validate_identity_and_claim(
    root: Path,
    protocol: dict[str, Any],
    protocol_id: str,
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
) -> dict[str, Any]:
    identity = _read_json(root / "run_identity.json")
    _expect_keys(
        identity,
        {
            "attempt_id",
            "protocol_id",
            "execution_authority_sha256",
            "source_authority_sha256",
            "attempt_claim_path",
            "attempt_claim_sha256",
            "output_root",
            "parent_correctness_root",
            "compilation_cache_path",
        },
        "RUN_IDENTITY",
    )
    attempt_id = _require_hex(identity["attempt_id"], _HEX_64, "ATTEMPT_ID")
    parent = protocol["correctness_parent"]
    claim_key = _semantic_sha256(parent["receipt_sha256"], "calibration-timing-v1")
    expected_claim_path = str(Path(protocol["attempt_registry_root"]) / f"{claim_key}.json")
    claim = {
        "schema_version": "matmul-collective-surface-calibration-attempt-claim-v1",
        "attempt_id": attempt_id,
        "protocol_id": protocol_id,
        "permanent_claim_key": claim_key,
        "correctness_parent_receipt_sha256": parent["receipt_sha256"],
        "source_commit": authority["source"]["source_commit"],
        "output_root": identity["output_root"],
        "state": "claimed",
    }
    parent_root = str(Path(identity["output_root"]) / "parent" / parent["archive_root_name"])
    if (
        identity["protocol_id"] != protocol_id
        or identity["source_authority_sha256"] != source_authority_sha256
        or identity["execution_authority_sha256"] != execution_authority_sha256
        or identity["attempt_claim_path"] != expected_claim_path
        or identity["attempt_claim_sha256"] != _sha256_bytes(_pretty_json_bytes(claim))
        or identity["parent_correctness_root"] != parent_root
        or not isinstance(identity["output_root"], str)
        or not Path(identity["output_root"]).is_absolute()
        or not isinstance(identity["compilation_cache_path"], str)
        or not Path(identity["compilation_cache_path"]).is_absolute()
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ATTEMPT_IDENTITY_MISMATCH")
    copied_claim = root / "attempt_claim.json"
    external_claim = Path(identity["attempt_claim_path"])
    try:
        external_status = external_claim.lstat()
    except FileNotFoundError as error:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_EXTERNAL_CLAIM_MISSING") from error
    if (
        not stat.S_ISREG(external_status.st_mode)
        or external_status.st_nlink != 1
        or external_claim.is_symlink()
        or _file_sha256(external_claim) != identity["attempt_claim_sha256"]
        or _read_json(external_claim) != claim
        or _file_sha256(copied_claim) != identity["attempt_claim_sha256"]
        or _read_json(copied_claim) != claim
        or external_claim.read_bytes() != copied_claim.read_bytes()
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ATTEMPT_CLAIM_MISMATCH")
    return identity


def _validate_worker(
    root: Path,
    identity: dict[str, Any],
    protocol: dict[str, Any],
    design: dict[str, Any],
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = _read_json(root / "worker-request.json")
    _expect_keys(
        request,
        {
            "schema_version",
            "attempt_id",
            "invocation_nonce",
            "output_root",
            "parent_correctness_root",
            "compilation_cache_path",
            "protocol_file_sha256",
            "design_file_sha256",
            "execution_authority_sha256",
            "source_commit",
            "source_authority_sha256",
            "protocol",
            "design",
        },
        "WORKER_REQUEST",
    )
    nonce = _require_hex(request["invocation_nonce"], _HEX_64, "INVOCATION_NONCE")
    expected = {
        "schema_version": "matmul-collective-surface-calibration-worker-v1",
        "attempt_id": identity["attempt_id"],
        "output_root": identity["output_root"],
        "parent_correctness_root": identity["parent_correctness_root"],
        "compilation_cache_path": identity["compilation_cache_path"],
        "protocol_file_sha256": _EXPECTED_PROTOCOL_FILE_SHA256,
        "design_file_sha256": _EXPECTED_DESIGN_FILE_SHA256,
        "execution_authority_sha256": execution_authority_sha256,
        "source_commit": authority["source"]["source_commit"],
        "source_authority_sha256": source_authority_sha256,
        "protocol": protocol,
        "design": design,
    }
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_WORKER_REQUEST_MISMATCH")
    if _read_json(root / "STARTED.json") != {
        "attempt_id": identity["attempt_id"],
        "invocation_nonce": nonce,
        "protocol_id": identity["protocol_id"],
        "state": "started",
    }:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_WORKER_START_MISMATCH")
    result = _read_json(root / "worker-result.json")
    _expect_keys(
        result,
        {"attempt_id", "invocation_nonce", "worker_pid", "execution_authority_sha256", "evidence"},
        "WORKER_RESULT",
    )
    _require_int(result["worker_pid"], "WORKER_PID", minimum=1)
    evidence = _read_json(root / "evidence.json")
    if (
        result["attempt_id"] != identity["attempt_id"]
        or result["invocation_nonce"] != nonce
        or result["execution_authority_sha256"] != execution_authority_sha256
        or result["evidence"] != evidence
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_WORKER_RESULT_MISMATCH")
    return request, result, evidence


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_exact_npy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            observed_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            observed_shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_NPY_VERSION_INVALID")
        expected_size = stream.tell() + math.prod(observed_shape) * dtype.itemsize
        if (
            observed_shape != shape
            or fortran
            or dtype.hasobject
            or dtype != _F32
            or path.stat().st_size != expected_size
        ):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_NPY_STRUCTURE_INVALID")
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != _F32
        or array.shape != shape
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_NPY_ARRAY_INVALID")
    return array


def _validate_saved_array(
    root: Path,
    value: Any,
    expected_path: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(value, dict):
        raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_SAVED_ARRAY_INVALID")
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
        or _canonical_relative_path(value["path"], "ARRAY_PATH") != expected_path
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ARRAY_PATH_MISMATCH")
    shape_value = value["shape"]
    if not isinstance(shape_value, list) or len(shape_value) != 2:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ARRAY_SHAPE_INVALID")
    shape = tuple(_require_int(item, "ARRAY_DIMENSION", minimum=1) for item in shape_value)
    if shape != expected_shape:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ARRAY_ABI_MISMATCH")
    path = root / expected_path
    array = _load_exact_npy(path, expected_shape)
    if (
        value["file_sha256"] != _file_sha256(path)
        or value["array_sha256"] != _array_sha256(array)
        or value["dtype"] != "float32"
        or value["numpy_dtype_str"] != "<f4"
        or any(
            value[key] != 0
            for key in ("nan_count", "positive_infinity_count", "negative_infinity_count")
        )
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SAVED_ARRAY_MISMATCH")
    return array


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
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_COMPILER_HLO_INVALID")
        canonical = canonical[:metadata_start] + canonical[min(starts) :]
    return re.sub(r" stack_frame_id=\d+", "", canonical)


def _parent_signed_periodic(parent_evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = parent_evidence.get("cases")
    if not isinstance(cases, list):
        raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CASES_INVALID")
    result = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("input"), dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CASE_INVALID")
        input_case = case["input"]
        if input_case.get("pattern") != "signed-periodic":
            continue
        scenario = input_case.get("scenario_name")
        executions = case.get("executions")
        if scenario not in _SCENARIOS or not isinstance(executions, list) or len(executions) != 4:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_SIGNED_CASE_INVALID")
        hashes: dict[str, list[str]] = {strategy: [] for strategy in _STRATEGIES}
        for execution in executions:
            if not isinstance(execution, dict) or execution.get("strategy") not in hashes:
                raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_EXECUTION_INVALID")
            output = execution.get("output")
            if not isinstance(output, dict):
                raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_OUTPUT_INVALID")
            hashes[execution["strategy"]].append(output.get("array_sha256"))
        oracle = case.get("oracle")
        if not isinstance(oracle, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_ORACLE_INVALID")
        oracle_hash = oracle.get("array_sha256")
        if any(
            len(values) != 2 or values[0] != values[1] or values[0] != oracle_hash
            for values in hashes.values()
        ):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_OUTPUT_CONSENSUS_MISMATCH")
        result[scenario] = {
            "case_sha256": _identity_sha256(case),
            "input": input_case,
            "oracle_array_sha256": oracle_hash,
            "strategy_hashes": {strategy: values[0] for strategy, values in hashes.items()},
        }
    if tuple(result) != _SCENARIOS:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_SIGNED_INVENTORY_MISMATCH")
    return result


def _parent_continuity(parent_evidence: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    continuity = parent_evidence.get("continuity")
    if not isinstance(continuity, list):
        raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CONTINUITY_INVALID")
    result = {}
    for record in continuity:
        if not isinstance(record, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CONTINUITY_RECORD_INVALID")
        key = (record.get("scenario_name"), record.get("strategy"))
        if key in result:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CONTINUITY_DUPLICATE")
        result[key] = record
    expected = {(scenario, strategy) for scenario in _SCENARIOS for strategy in _STRATEGIES}
    if set(result) != expected:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PARENT_CONTINUITY_INVENTORY_MISMATCH")
    return result


def _validate_continuity(
    root: Path,
    values: Any,
    parent_evidence: dict[str, Any],
) -> dict[tuple[str, str], str]:
    if not isinstance(values, list) or len(values) != 32:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_CONTINUITY_INVENTORY_MISMATCH")
    expected = [(scenario, strategy) for scenario in _SCENARIOS for strategy in _STRATEGIES]
    parents = _parent_continuity(parent_evidence)
    result = {}
    keys = (
        "distributed_schedule_sha256",
        "physical_schedule_sha256",
        "pallas_source_sha256",
        "semantic_stablehlo_sha256",
        "semantic_compiler_hlo_sha256",
    )
    for record, pair in zip(values, expected, strict=True):
        if not isinstance(record, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_CONTINUITY_RECORD_INVALID")
        _expect_keys(
            record,
            {
                "scenario_name",
                "strategy",
                "stablehlo_path",
                "stablehlo_file_sha256",
                "compiler_hlo_path",
                "compiler_hlo_file_sha256",
                *(f"parent_{key}" for key in keys),
                *(f"observed_{key}" for key in keys),
            },
            "CONTINUITY_RECORD",
        )
        scenario, strategy = pair
        base = f"continuity/{scenario}/{strategy}"
        stable_path = f"{base}/stablehlo.txt"
        compiler_path = f"{base}/compiler_hlo.txt"
        stable = (root / stable_path).read_text()
        compiler = (root / compiler_path).read_text()
        parent = parents[pair]
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
                record[f"parent_{key}"] != parent[f"observed_{key}"]
                or record[f"parent_{key}"] != record[f"observed_{key}"]
                for key in keys
            )
        ):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_CONTINUITY_MISMATCH")
        result[pair] = _identity_sha256(record)
    return result


def _signed_oracle(m: int, k: int, n: int) -> np.ndarray:
    lhs_sequence = np.asarray(
        (1, -2, 3, -4, 2, -1, 4, -3, -1, 3, -2, 4, -4, 2, -3, 1),
        dtype=np.int64,
    )
    rhs_sequence = np.asarray(
        (2, 1, -3, 4, -1, -4, 3, -2, 4, -3, 1, -2, 3, 2, -4, -1),
        dtype=np.int64,
    )
    rows = np.arange(m, dtype=np.int64)[:, None]
    columns = np.arange(n, dtype=np.int64)[None, :]
    left = lhs_sequence[(np.arange(16)[None, :] + rows) % 16]
    right = rhs_sequence[(np.arange(16)[None, :, None] + 3 * columns[:, None, :]) % 16]
    dot = np.sum(left[:, :, None] * right, axis=1, dtype=np.int64)
    return np.ascontiguousarray((dot * (k // 128) * 36 * 2.0**-19).astype(np.float32))


def _validate_inputs(
    root: Path,
    values: Any,
    protocol: dict[str, Any],
    design: dict[str, Any],
    parent_evidence: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    if not isinstance(values, list) or len(values) != 16:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_INPUT_INVENTORY_MISMATCH")
    parents = _parent_signed_periodic(parent_evidence)
    scenarios = {scenario["name"]: scenario for scenario in design["scenarios"]}
    records = {}
    arrays = {}
    for value, scenario_name in zip(values, _SCENARIOS, strict=True):
        if not isinstance(value, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_INPUT_INVALID")
        _expect_keys(
            value,
            {
                "scenario_name",
                "parent_case_sha256",
                "parent_xla_array_sha256",
                "parent_pallas_array_sha256",
                "input",
                "oracle",
            },
            "TIMING_INPUT",
        )
        scenario = scenarios[scenario_name]
        parent = parents[scenario_name]
        expected_oracle = _signed_oracle(scenario["m"], scenario["k"], scenario["n"])
        oracle_path = f"oracles/{scenario_name}.npy"
        oracle = _validate_saved_array(
            root,
            value["oracle"],
            oracle_path,
            (scenario["m"], scenario["n"]),
        )
        if (
            value["scenario_name"] != scenario_name
            or value["parent_case_sha256"] != parent["case_sha256"]
            or value["parent_xla_array_sha256"] != parent["strategy_hashes"][_STRATEGIES[0]]
            or value["parent_pallas_array_sha256"] != parent["strategy_hashes"][_STRATEGIES[1]]
            or value["input"] != parent["input"]
            or value["input"].get("protocol_id") != protocol["correctness_parent"]["protocol_id"]
            or value["oracle"].get("array_sha256") != parent["oracle_array_sha256"]
            or not np.array_equal(oracle, expected_oracle)
        ):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_INPUT_PARENT_MISMATCH")
        records[scenario_name] = value
        arrays[scenario_name] = oracle
    return records, arrays


def _validate_resident_pairs(
    values: Any,
    continuity: dict[tuple[str, str], str],
    nonce: str,
    worker_pid: int,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list) or len(values) != 16:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_RESIDENCY_INVENTORY_MISMATCH")
    result = {}
    for value, scenario in zip(values, _SCENARIOS, strict=True):
        if not isinstance(value, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_RESIDENCY_INVALID")
        _expect_keys(
            value,
            {
                "scenario_name",
                "xla_compile_record_sha256",
                "pallas_compile_record_sha256",
                "invocation_nonce",
                "worker_pid",
            },
            "RESIDENT_PAIR",
        )
        if value != {
            "scenario_name": scenario,
            "xla_compile_record_sha256": continuity[(scenario, _STRATEGIES[0])],
            "pallas_compile_record_sha256": continuity[(scenario, _STRATEGIES[1])],
            "invocation_nonce": nonce,
            "worker_pid": worker_pid,
        }:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_RESIDENCY_BINDING_MISMATCH")
        result[scenario] = value
    return result


def _resident_pair_sha256(value: dict[str, Any]) -> str:
    return _identity_sha256(value)


def _validate_output_gates(
    root: Path,
    values: Any,
    inputs: dict[str, dict[str, Any]],
    oracles: dict[str, np.ndarray],
    residents: dict[str, dict[str, Any]],
    nonce: str,
    worker_pid: int,
) -> tuple[dict[str, Any], ...]:
    expected = [
        (scenario, strategy, phase)
        for phase in ("before_timing", "after_timing")
        for scenario in _SCENARIOS
        for strategy in _STRATEGIES
    ]
    if not isinstance(values, list) or len(values) != 64:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_OUTPUT_GATE_INVENTORY_MISMATCH")
    result = []
    by_arm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for value, (scenario, strategy, phase) in zip(values, expected, strict=True):
        if not isinstance(value, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_OUTPUT_GATE_INVALID")
        _expect_keys(
            value,
            {
                "scenario_name",
                "strategy",
                "phase",
                "resident_pair_sha256",
                "invocation_nonce",
                "worker_pid",
                "start_ns",
                "stop_ns",
                "oracle_array_sha256",
                "output",
                "mismatched_element_count",
                "maximum_absolute_error",
                "maximum_normalized_error",
            },
            "OUTPUT_GATE",
        )
        start = _require_int(value["start_ns"], "OUTPUT_START")
        stop = _require_int(value["stop_ns"], "OUTPUT_STOP", minimum=1)
        expected_path = f"outputs/{scenario}/{strategy}-{phase}.npy"
        candidate = _validate_saved_array(
            root, value["output"], expected_path, oracles[scenario].shape
        )
        absolute = np.abs(candidate - oracles[scenario])
        threshold = 0.001 + 0.001 * np.abs(oracles[scenario])
        normalized = absolute / threshold
        mismatches = int(np.count_nonzero(absolute > threshold))
        maximum_absolute = float(absolute.max())
        maximum_normalized = float(normalized.max())
        parent_hash = inputs[scenario][
            "parent_xla_array_sha256"
            if strategy == _STRATEGIES[0]
            else "parent_pallas_array_sha256"
        ]
        if (
            (value["scenario_name"], value["strategy"], value["phase"])
            != (scenario, strategy, phase)
            or value["resident_pair_sha256"] != _resident_pair_sha256(residents[scenario])
            or value["invocation_nonce"] != nonce
            or value["worker_pid"] != worker_pid
            or stop <= start
            or value["oracle_array_sha256"] != inputs[scenario]["oracle"]["array_sha256"]
            or value["output"]["array_sha256"] != parent_hash
            or value["mismatched_element_count"] != mismatches
            or _require_finite(value["maximum_absolute_error"], "OUTPUT_ABSOLUTE_ERROR", minimum=0)
            != maximum_absolute
            or _require_finite(
                value["maximum_normalized_error"], "OUTPUT_NORMALIZED_ERROR", minimum=0
            )
            != maximum_normalized
            or mismatches != 0
            or maximum_normalized > 1.0
        ):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_OUTPUT_GATE_MISMATCH")
        result.append(value)
        by_arm.setdefault((scenario, strategy), []).append(value)
    if any(
        len(gates) != 2
        or gates[0]["output"]["array_sha256"] != gates[1]["output"]["array_sha256"]
        or gates[0]["output"]["array_sha256"] != gates[0]["oracle_array_sha256"]
        for gates in by_arm.values()
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_OUTPUT_REPEAT_MISMATCH")
    return tuple(result)


def _warmup_strategy_order(scenario_index: int) -> tuple[str, ...]:
    pair = _STRATEGIES if scenario_index % 2 == 0 else tuple(reversed(_STRATEGIES))
    return pair * 10


def _scenario_order(round_index: int) -> tuple[str, ...]:
    return _SCENARIOS if round_index < 8 else tuple(reversed(_SCENARIOS))


def _strategy_order(round_index: int) -> tuple[str, str]:
    return _STRATEGIES if round_index % 2 == 0 else tuple(reversed(_STRATEGIES))


def _validate_warmups(
    values: Any,
    residents: dict[str, dict[str, Any]],
    nonce: str,
    worker_pid: int,
) -> tuple[dict[str, Any], ...]:
    expected = []
    repetitions: dict[tuple[str, str], int] = {}
    for scenario_index, scenario in enumerate(_SCENARIOS):
        for strategy in _warmup_strategy_order(scenario_index):
            key = (scenario, strategy)
            repetitions[key] = repetitions.get(key, 0) + 1
            expected.append((scenario, scenario_index + 1, strategy, repetitions[key]))
    if not isinstance(values, list) or len(values) != 320:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_WARMUP_INVENTORY_MISMATCH")
    result = []
    for sequence, (value, (scenario, position, strategy, repetition)) in enumerate(
        zip(values, expected, strict=True), start=1
    ):
        if not isinstance(value, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_WARMUP_INVALID")
        _expect_keys(
            value,
            {
                "sequence",
                "scenario_name",
                "scenario_position",
                "strategy",
                "strategy_repetition",
                "resident_pair_sha256",
                "invocation_nonce",
                "worker_pid",
                "start_ns",
                "stop_ns",
            },
            "WARMUP",
        )
        start = _require_int(value["start_ns"], "WARMUP_START")
        stop = _require_int(value["stop_ns"], "WARMUP_STOP", minimum=1)
        if (
            value["sequence"] != sequence
            or (
                value["scenario_name"],
                value["scenario_position"],
                value["strategy"],
                value["strategy_repetition"],
            )
            != (scenario, position, strategy, repetition)
            or value["resident_pair_sha256"] != _resident_pair_sha256(residents[scenario])
            or value["invocation_nonce"] != nonce
            or value["worker_pid"] != worker_pid
            or stop <= start
        ):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_WARMUP_SEQUENCE_MISMATCH")
        result.append(value)
    return tuple(result)


def _validate_samples(
    values: Any,
    residents: dict[str, dict[str, Any]],
    nonce: str,
    worker_pid: int,
) -> tuple[dict[str, Any], ...]:
    expected = []
    for round_index in range(16):
        for scenario_position, scenario in enumerate(_scenario_order(round_index), start=1):
            for arm_position, strategy in enumerate(_strategy_order(round_index), start=1):
                for call_index in range(5):
                    expected.append(
                        (
                            round_index,
                            scenario,
                            scenario_position,
                            strategy,
                            arm_position,
                            call_index,
                        )
                    )
    if not isinstance(values, list) or len(values) != 2560:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SAMPLE_INVENTORY_MISMATCH")
    result = []
    for sequence, (value, expected_value) in enumerate(zip(values, expected, strict=True), start=1):
        if not isinstance(value, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_SAMPLE_INVALID")
        _expect_keys(
            value,
            {
                "sequence",
                "round_index",
                "scenario_name",
                "scenario_position",
                "strategy",
                "arm_position",
                "call_index",
                "resident_pair_sha256",
                "invocation_nonce",
                "worker_pid",
                "start_ns",
                "stop_ns",
                "duration_ns",
            },
            "SAMPLE",
        )
        start = _require_int(value["start_ns"], "SAMPLE_START")
        stop = _require_int(value["stop_ns"], "SAMPLE_STOP", minimum=1)
        duration = _require_int(value["duration_ns"], "SAMPLE_DURATION", minimum=1)
        observed = (
            value["round_index"],
            value["scenario_name"],
            value["scenario_position"],
            value["strategy"],
            value["arm_position"],
            value["call_index"],
        )
        if (
            value["sequence"] != sequence
            or observed != expected_value
            or value["resident_pair_sha256"]
            != _resident_pair_sha256(residents[value["scenario_name"]])
            or value["invocation_nonce"] != nonce
            or value["worker_pid"] != worker_pid
            or stop - start != duration
        ):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SAMPLE_SEQUENCE_MISMATCH")
        result.append(value)
    return tuple(result)


def _validate_global_timeline(
    gates: tuple[dict[str, Any], ...],
    warmups: tuple[dict[str, Any], ...],
    samples: tuple[dict[str, Any], ...],
) -> None:
    before = tuple(value for value in gates if value["phase"] == "before_timing")
    after = tuple(value for value in gates if value["phase"] == "after_timing")
    timeline = (*before, *warmups, *samples, *after)
    if any(left["stop_ns"] > right["start_ns"] for left, right in itertools.pairwise(timeline)):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_CLOCK_ORDER_MISMATCH")


def _validate_evidence(
    root: Path,
    evidence: dict[str, Any],
    protocol: dict[str, Any],
    protocol_id: str,
    design: dict[str, Any],
    parent_evidence: dict[str, Any],
    execution_authority_sha256: str,
    request: dict[str, Any],
    result: dict[str, Any],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    _expect_keys(
        evidence,
        {
            "schema_version",
            "protocol_id",
            "protocol_file_sha256",
            "design_id",
            "design_file_sha256",
            "correctness_parent_attempt_id",
            "correctness_parent_evidence_sha256",
            "correctness_parent_receipt_sha256",
            "calibration_execution_authority_sha256",
            "invocation_nonce",
            "worker_pid",
            "continuity",
            "inputs",
            "resident_pairs",
            "output_gates",
            "warmups",
            "samples",
        },
        "EVIDENCE",
    )
    parent = protocol["correctness_parent"]
    if (
        evidence["schema_version"] != _EVIDENCE_SCHEMA
        or evidence["protocol_id"] != protocol_id
        or evidence["protocol_file_sha256"] != _EXPECTED_PROTOCOL_FILE_SHA256
        or evidence["design_id"] != _EXPECTED_DESIGN_ID
        or evidence["design_file_sha256"] != _EXPECTED_DESIGN_FILE_SHA256
        or evidence["correctness_parent_attempt_id"] != parent["attempt_id"]
        or evidence["correctness_parent_evidence_sha256"] != parent["evidence_sha256"]
        or evidence["correctness_parent_receipt_sha256"] != parent["receipt_sha256"]
        or evidence["calibration_execution_authority_sha256"] != execution_authority_sha256
        or evidence["invocation_nonce"] != request["invocation_nonce"]
        or evidence["worker_pid"] != result["worker_pid"]
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_EVIDENCE_AUTHORITY_MISMATCH")
    nonce = evidence["invocation_nonce"]
    worker_pid = evidence["worker_pid"]
    continuity = _validate_continuity(root, evidence["continuity"], parent_evidence)
    inputs, oracles = _validate_inputs(root, evidence["inputs"], protocol, design, parent_evidence)
    residents = _validate_resident_pairs(evidence["resident_pairs"], continuity, nonce, worker_pid)
    gates = _validate_output_gates(
        root, evidence["output_gates"], inputs, oracles, residents, nonce, worker_pid
    )
    warmups = _validate_warmups(evidence["warmups"], residents, nonce, worker_pid)
    samples = _validate_samples(evidence["samples"], residents, nonce, worker_pid)
    _validate_global_timeline(gates, warmups, samples)
    return _identity_sha256(evidence), samples


def _derive_observations(samples: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str, int], list[int]] = {}
    for sample in samples:
        grouped.setdefault(
            (sample["scenario_name"], sample["strategy"], sample["round_index"]), []
        ).append(sample["duration_ns"])
    observations = []
    for scenario in _SCENARIOS:
        for strategy in _STRATEGIES:
            round_medians = []
            for round_index in range(16):
                values = grouped.get((scenario, strategy, round_index), [])
                if len(values) != 5:
                    raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SAMPLE_GROUP_INVALID")
                round_medians.append(sorted(values)[2])
            ordered = sorted(round_medians)
            observations.append(
                {
                    "scenario_name": scenario,
                    "strategy": strategy,
                    "round_medians_ns": round_medians,
                    "median_ns": (ordered[7] + ordered[8]) / 2,
                }
            )
    if len(grouped) != 32 * 16:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SAMPLE_GROUP_INVENTORY")
    return tuple(observations)


def _model_payload(matrix: np.ndarray, measured: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    coefficients = _nonnegative_affine_fit(matrix, measured)
    predictions = matrix @ coefficients
    relative = np.abs(predictions - measured) / measured
    return (
        {
            "coefficient_names": [
                "xla_intercept",
                "pallas_intercept",
                "shared_compute",
                "shared_hbm",
                "xla_ici",
                "pallas_ici",
            ],
            "coefficients": [float(value) for value in coefficients],
            "calibration_predictions_ns": [float(value) for value in predictions],
            "calibration_relative_errors": [float(value) for value in relative],
            "maximum_calibration_relative_error": float(np.max(relative)),
            "median_calibration_relative_error": float(np.median(relative)),
        },
        coefficients,
    )


def _derive_seal(
    samples: tuple[dict[str, Any], ...],
    protocol_id: str,
    correctness_parent_receipt_sha256: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    observations = _derive_observations(samples)
    calibration_matrix = np.asarray(
        [_feature_row(value["scenario_name"], value["strategy"]) for value in observations],
        dtype=np.float64,
    )
    holdout_arms = [(scenario, strategy) for scenario in _HOLDOUTS for strategy in _STRATEGIES]
    holdout_matrix = np.asarray(
        [_feature_row(scenario, strategy) for scenario, strategy in holdout_arms],
        dtype=np.float64,
    )
    measured = np.asarray([value["median_ns"] for value in observations], dtype=np.float64)
    model, coefficients = _model_payload(calibration_matrix, measured)
    round_medians = np.asarray(
        [value["round_medians_ns"] for value in observations], dtype=np.float64
    )
    bootstrap_indices = np.asarray(
        [
            [
                _semantic_seed(
                    protocol_id,
                    "calibration-bootstrap",
                    str(replicate),
                    str(draw),
                    "round-index-v1",
                )
                % 16
                for draw in range(16)
            ]
            for replicate in range(10_000)
        ],
        dtype=np.uint8,
    )
    bootstrap_measured = np.median(round_medians[:, bootstrap_indices], axis=2).T
    bootstrap_coefficients = np.empty((10_000, 6), dtype=np.float64)
    for index, bootstrap_values in enumerate(bootstrap_measured):
        bootstrap_coefficients[index] = _nonnegative_affine_fit(
            calibration_matrix, bootstrap_values
        )
    bootstrap_predictions = bootstrap_coefficients @ holdout_matrix.T
    if not np.all(np.isfinite(bootstrap_predictions)) or np.any(bootstrap_predictions <= 0):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_BOOTSTRAP_PREDICTION_INVALID")
    point_predictions = holdout_matrix @ coefficients
    lower, upper = np.quantile(bootstrap_predictions, (0.005, 0.995), axis=0, method="linear")
    intervals = [
        {
            "scenario_name": scenario,
            "strategy": strategy,
            "point_prediction_ns": float(point),
            "lower_99pct_ns": float(low),
            "upper_99pct_ns": float(high),
            "relative_width": float((high - low) / point),
        }
        for (scenario, strategy), point, low, high in zip(
            holdout_arms, point_predictions, lower, upper, strict=True
        )
    ]
    point_pairs = point_predictions.reshape(4, 2)
    bootstrap_pairs = bootstrap_predictions.reshape(10_000, 4, 2)
    point_improvements = (point_pairs[:, 0] - point_pairs[:, 1]) / point_pairs[:, 0]
    bootstrap_improvements = (
        bootstrap_pairs[:, :, 0] - bootstrap_pairs[:, :, 1]
    ) / bootstrap_pairs[:, :, 0]
    improvement_lower, improvement_upper = np.quantile(
        bootstrap_improvements, (0.005, 0.995), axis=0, method="linear"
    )
    improvements = [
        {
            "scenario_name": scenario,
            "point_improvement": float(point_improvements[index]),
            "lower_99pct_improvement": float(improvement_lower[index]),
            "upper_99pct_improvement": float(improvement_upper[index]),
        }
        for index, scenario in enumerate(_HOLDOUTS)
    ]
    width_gate_passed = all(value["relative_width"] <= 0.2 for value in intervals)
    return {
        "schema_version": _SEAL_SCHEMA,
        "seal_schema": "matmul-collective-surface-calibration-seal-v1",
        "protocol_id": protocol_id,
        "design_id": _EXPECTED_DESIGN_ID,
        "correctness_parent_receipt_sha256": correctness_parent_receipt_sha256,
        "calibration_evidence_sha256": evidence_sha256,
        "observations": list(observations),
        "model": model,
        "bootstrap_sample_count": 10_000,
        "bootstrap_array_hash_schema": _ARRAY_HASH_SCHEMA,
        "bootstrap_index_sha256": _bootstrap_array_sha256(bootstrap_indices, np.dtype("u1")),
        "bootstrap_coefficient_sha256": _bootstrap_array_sha256(
            bootstrap_coefficients, np.dtype("<f8")
        ),
        "bootstrap_prediction_sha256": _bootstrap_array_sha256(
            bootstrap_predictions, np.dtype("<f8")
        ),
        "bootstrap_improvement_sha256": _bootstrap_array_sha256(
            bootstrap_improvements, np.dtype("<f8")
        ),
        "holdout_predictions": intervals,
        "strategy_predictions": improvements,
        "width_gate_passed": width_gate_passed,
        "holdout_authorization": (
            "pending_independent_replay"
            if width_gate_passed
            else "denied_prediction_interval_width"
        ),
    }


def _validate_seal(
    root: Path,
    samples: tuple[dict[str, Any], ...],
    protocol: dict[str, Any],
    protocol_id: str,
    evidence_sha256: str,
) -> tuple[dict[str, Any], str]:
    recorded = _read_json(root / "calibration-seal.json")
    expected = _derive_seal(
        samples,
        protocol_id,
        protocol["correctness_parent"]["receipt_sha256"],
        evidence_sha256,
    )
    if recorded != expected:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SEAL_REPLAY_MISMATCH")
    return recorded, _identity_sha256(recorded)


def _validate_phase_ledger(
    root: Path,
    protocol: dict[str, Any],
    evidence_sha256: str,
    seal_sha256: str,
) -> tuple[dict[str, Any], str]:
    ledger = _read_json(root / "phase_ledger.json")
    _expect_keys(ledger, {"schema_version", "attempt_id", "events"}, "PHASE_LEDGER")
    events = ledger["events"]
    parent_root = root / "parent" / protocol["correctness_parent"]["archive_root_name"]
    parent_ledger = _read_json(parent_root / "phase_ledger.json")
    if not isinstance(events, list) or len(events) != 4:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PHASE_LEDGER_MISMATCH")
    for sequence, (event, phase) in enumerate(zip(events, _PHASES, strict=True), start=1):
        if not isinstance(event, dict):
            raise TypeError("SURFACE_CALIBRATION_INDEPENDENT_PHASE_EVENT_INVALID")
        _expect_keys(event, {"sequence", "phase", "artifact_sha256"}, "PHASE_EVENT")
        _require_hex(event["artifact_sha256"], _HEX_64, "PHASE_ARTIFACT")
        if event["sequence"] != sequence or event["phase"] != phase:
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PHASE_ORDER_MISMATCH")
    if (
        ledger["schema_version"] != _PHASE_LEDGER_SCHEMA
        or ledger["attempt_id"] != protocol["correctness_parent"]["attempt_id"]
        or parent_ledger.get("events") != events[:2]
        or events[2]["artifact_sha256"] != evidence_sha256
        or events[3]["artifact_sha256"] != seal_sha256
    ):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_PHASE_BINDING_MISMATCH")
    return ledger, _identity_sha256(ledger)


def _continuity_payloads(evidence: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, str]]:
    schedules = {
        f"{value['scenario_name']}:{value['strategy']}": [
            value["observed_distributed_schedule_sha256"],
            value["observed_physical_schedule_sha256"],
            value["observed_pallas_source_sha256"],
        ]
        for value in evidence["continuity"]
    }
    compiled = {
        f"{value['scenario_name']}:{value['strategy']}": _identity_sha256(value)
        for value in evidence["continuity"]
    }
    return schedules, compiled


def _validate_ledger(
    path: Path,
    identity: dict[str, Any],
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
    protocol: dict[str, Any],
    evidence: dict[str, Any],
    evidence_sha256: str,
    seal: dict[str, Any],
    seal_sha256: str,
    phase_ledger_sha256: str,
    artifact_set_sha256: str,
) -> str:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_LEDGER_INTEGRITY_MISMATCH")
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
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_LEDGER_SCHEMA_MISMATCH")
    schedules, compiled = _continuity_payloads(evidence)
    payloads = (
        {
            "protocol_id": identity["protocol_id"],
            "execution_authority_sha256": execution_authority_sha256,
            "attempt_claim_path": identity["attempt_claim_path"],
            "attempt_claim_sha256": identity["attempt_claim_sha256"],
        },
        {
            "source_authority_sha256": source_authority_sha256,
            "parent_receipt_sha256": protocol["correctness_parent"]["receipt_sha256"],
            "parent_archive_sha256": protocol["correctness_parent"]["archive_sha256"],
            "devices": authority["devices"],
        },
        {"continuity_schedule_set_sha256": _identity_sha256(schedules)},
        {"fresh_compile_set_sha256": _identity_sha256(compiled)},
        {"evidence_sha256": evidence_sha256},
        {
            "calibration_seal_sha256": seal_sha256,
            "phase_ledger_sha256": phase_ledger_sha256,
            "artifact_set_sha256": artifact_set_sha256,
        },
        {
            "producer_validation": "schema-fit-bootstrap-and-artifact-replay-v1",
            "holdout_authorization": seal["holdout_authorization"],
        },
    )
    if tuple(row[4] for row in rows) != tuple(_identity_sha256(value) for value in payloads):
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_LEDGER_PAYLOAD_MISMATCH")
    return _file_sha256(path)


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
            key=lambda value: value.relative_to(root).as_posix(),
        )
    ]


def _expected_archive_files(root: Path, protocol: dict[str, Any]) -> set[str]:
    parent = protocol["correctness_parent"]
    expected = {
        "STARTED.json",
        "attempt_claim.json",
        "calibration-seal.json",
        "design.json",
        "evidence.json",
        "execution_authority.json",
        "ledger.sqlite",
        "manifest.json",
        "phase_ledger.json",
        "protocol.json",
        "receipt.json",
        "run_identity.json",
        "worker-request.json",
        "worker-result.json",
        "source/archive.py",
        "source/evidence.py",
        "source/executor.py",
        "source/protocol.py",
        "source/seal.py",
        "source/verifier.py",
        "source/worker.py",
        "source/committed/uv.lock",
        *(f"source/committed/{path}" for path in _SOURCE_DEPENDENCIES),
        f"parent/{parent['archive_filename']}",
    }
    parent_root = root / "parent" / parent["archive_root_name"]
    expected.update(
        path.relative_to(root).as_posix() for path in parent_root.rglob("*") if path.is_file()
    )
    for scenario in _SCENARIOS:
        expected.add(f"oracles/{scenario}.npy")
        for strategy in _STRATEGIES:
            expected.add(f"continuity/{scenario}/{strategy}/stablehlo.txt")
            expected.add(f"continuity/{scenario}/{strategy}/compiler_hlo.txt")
            for phase in ("before_timing", "after_timing"):
                expected.add(f"outputs/{scenario}/{strategy}-{phase}.npy")
    return expected


def _validate_receipt_and_manifest(
    root: Path,
    identity: dict[str, Any],
    protocol: dict[str, Any],
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
    evidence: dict[str, Any],
    evidence_sha256: str,
    seal: dict[str, Any],
    seal_sha256: str,
    phase_ledger_sha256: str,
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
        seal,
        seal_sha256,
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
            "attempt_claim_path",
            "attempt_claim_sha256",
            "correctness_parent_receipt_file_sha256",
            "correctness_parent_receipt_sha256",
            "evidence_file_sha256",
            "evidence_sha256",
            "calibration_seal_file_sha256",
            "calibration_seal_sha256",
            "ledger_snapshot_sha256",
            "phase_ledger_file_sha256",
            "phase_ledger_sha256",
            "phase_sequence",
            "previous_phase_receipt_sha256",
            "artifact_set_sha256",
        },
        "RECEIPT",
    )
    parent = protocol["correctness_parent"]
    expected_receipt = {
        "schema_version": _RECEIPT_SCHEMA,
        "attempt_id": identity["attempt_id"],
        "protocol_id": identity["protocol_id"],
        "attempt_claim_path": identity["attempt_claim_path"],
        "attempt_claim_sha256": identity["attempt_claim_sha256"],
        "correctness_parent_receipt_file_sha256": parent["receipt_file_sha256"],
        "correctness_parent_receipt_sha256": parent["receipt_sha256"],
        "evidence_file_sha256": _file_sha256(root / "evidence.json"),
        "evidence_sha256": evidence_sha256,
        "calibration_seal_file_sha256": _file_sha256(root / "calibration-seal.json"),
        "calibration_seal_sha256": seal_sha256,
        "ledger_snapshot_sha256": ledger_sha256,
        "phase_ledger_file_sha256": _file_sha256(root / "phase_ledger.json"),
        "phase_ledger_sha256": phase_ledger_sha256,
        "phase_sequence": 4,
        "previous_phase_receipt_sha256": parent["receipt_sha256"],
        "artifact_set_sha256": artifact_set_sha256,
    }
    if receipt != expected_receipt:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_RECEIPT_MISMATCH")
    receipt_sha256 = _identity_sha256(receipt)
    manifest = _read_json(root / "manifest.json")
    _expect_keys(
        manifest,
        {
            "schema_version",
            "identity",
            "evidence_file_sha256",
            "evidence_sha256",
            "calibration_seal_file_sha256",
            "calibration_seal_sha256",
            "ledger_snapshot_sha256",
            "phase_ledger_file_sha256",
            "phase_ledger_sha256",
            "receipt_file_sha256",
            "receipt_sha256",
            "artifacts",
        },
        "MANIFEST",
    )
    expected_artifacts = _artifact_entries(root, {"manifest.json"})
    expected_files = _expected_archive_files(root, protocol)
    if {entry["path"] for entry in expected_artifacts} != expected_files - {"manifest.json"}:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ARCHIVE_INVENTORY_MISMATCH")
    expected_manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "identity": identity,
        "evidence_file_sha256": _file_sha256(root / "evidence.json"),
        "evidence_sha256": evidence_sha256,
        "calibration_seal_file_sha256": _file_sha256(root / "calibration-seal.json"),
        "calibration_seal_sha256": seal_sha256,
        "ledger_snapshot_sha256": ledger_sha256,
        "phase_ledger_file_sha256": _file_sha256(root / "phase_ledger.json"),
        "phase_ledger_sha256": phase_ledger_sha256,
        "receipt_file_sha256": _file_sha256(root / "receipt.json"),
        "receipt_sha256": receipt_sha256,
        "artifacts": expected_artifacts,
    }
    if manifest != expected_manifest:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_MANIFEST_MISMATCH")
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
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_DIRECTORY_INVENTORY_MISMATCH")
    return ledger_sha256, receipt_sha256


def verify_surface_calibration_independently(
    root: Path,
    protocol_path: Path,
    design_path: Path,
) -> SurfaceCalibrationVerification:
    if root.is_symlink():
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ROOT_INVALID")
    root = root.resolve(strict=True)
    _validate_archive_tree(root)
    if (root / "failure.json").exists():
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_ATTEMPT_INCOMPLETE")
    protocol, protocol_id, protocol_file_sha256 = _validate_protocol(
        protocol_path, root / "protocol.json"
    )
    design = _validate_design(design_path, root / "design.json", protocol)
    parent = _verify_parent(root, protocol)
    authority, source_authority_sha256, execution_authority_sha256 = _validate_authority(
        root, protocol, protocol_id, protocol_file_sha256, design
    )
    identity = _validate_identity_and_claim(
        root,
        protocol,
        protocol_id,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
    )
    request, result, evidence = _validate_worker(
        root,
        identity,
        protocol,
        design,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
    )
    evidence_sha256, samples = _validate_evidence(
        root,
        evidence,
        protocol,
        protocol_id,
        design,
        parent["evidence"],
        execution_authority_sha256,
        request,
        result,
    )
    seal, seal_sha256 = _validate_seal(root, samples, protocol, protocol_id, evidence_sha256)
    _, phase_ledger_sha256 = _validate_phase_ledger(root, protocol, evidence_sha256, seal_sha256)
    ledger_sha256, receipt_sha256 = _validate_receipt_and_manifest(
        root,
        identity,
        protocol,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
        evidence,
        evidence_sha256,
        seal,
        seal_sha256,
        phase_ledger_sha256,
    )
    return SurfaceCalibrationVerification(
        attempt_id=identity["attempt_id"],
        protocol_id=protocol_id,
        source_authority_sha256=source_authority_sha256,
        execution_authority_sha256=execution_authority_sha256,
        correctness_parent_receipt_sha256=protocol["correctness_parent"]["receipt_sha256"],
        evidence_sha256=evidence_sha256,
        seal_sha256=seal_sha256,
        ledger_sha256=ledger_sha256,
        phase_ledger_sha256=phase_ledger_sha256,
        receipt_sha256=receipt_sha256,
        sample_count=len(samples),
        holdout_authorization=seal["holdout_authorization"],
    )


def main() -> None:
    if not sys.flags.safe_path:
        raise ValueError("SURFACE_CALIBRATION_INDEPENDENT_SAFE_PATH_REQUIRED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    args = parser.parse_args()
    print(verify_surface_calibration_independently(args.root, args.protocol, args.design).as_json())


if __name__ == "__main__":
    main()
