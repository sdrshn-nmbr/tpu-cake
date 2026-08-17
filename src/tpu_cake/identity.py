from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

LEGACY_SEMANTIC_IDENTITY_SCHEMA = "separator-v1"
SEMANTIC_IDENTITY_SCHEMA = "length-prefixed-v2"


def semantic_seed(*parts: str, schema: str = SEMANTIC_IDENTITY_SCHEMA) -> int:
    return int.from_bytes(
        bytes.fromhex(semantic_sha256(*parts, schema=schema))[:8], "big", signed=False
    )


def semantic_sha256(*parts: str, schema: str = SEMANTIC_IDENTITY_SCHEMA) -> str:
    if not parts or any(not part for part in parts):
        raise ValueError("semantic identity parts must be non-empty")
    if schema == LEGACY_SEMANTIC_IDENTITY_SCHEMA:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    if schema != SEMANTIC_IDENTITY_SCHEMA:
        raise ValueError(f"unsupported semantic identity schema: {schema}")
    digest = hashlib.sha256()
    digest.update(b"tpu-cake-semantic-identity\x00length-prefixed-v2\x00")
    for part in parts:
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def workload_rng(
    experiment_id: str,
    scenario_id: str,
    attempt: str,
    tensor_role: str,
) -> np.random.Generator:
    return np.random.default_rng(semantic_seed(experiment_id, scenario_id, attempt, tensor_role))


def candidate_rng(
    experiment_id: str,
    scenario_id: str,
    candidate_id: str,
    attempt: str,
    decision_role: str,
) -> np.random.Generator:
    return np.random.default_rng(
        semantic_seed(experiment_id, scenario_id, candidate_id, attempt, decision_role)
    )


def array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def arrays_sha256(arrays: Iterable[np.ndarray]) -> tuple[str, ...]:
    return tuple(array_sha256(array) for array in arrays)
