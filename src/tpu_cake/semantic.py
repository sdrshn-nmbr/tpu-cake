from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropertyObservation:
    name: str
    passed: bool
    maximum_absolute_error: float
    details: str


def compare_arrays(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> PropertyObservation:
    if expected.shape != actual.shape:
        return PropertyObservation(name, False, float("inf"), "shape mismatch")
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    maximum = float(difference.max(initial=0))
    passed = bool(
        np.allclose(
            expected,
            actual,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            equal_nan=False,
        )
    )
    return PropertyObservation(name, passed, maximum, "array comparison")


def prefix_invariance(
    run: Callable[[np.ndarray], np.ndarray],
    tokens: np.ndarray,
    prefix_length: int,
    *,
    replacement: np.ndarray,
    absolute_tolerance: float = 0,
    relative_tolerance: float = 0,
) -> PropertyObservation:
    if not 0 < prefix_length <= tokens.shape[-1]:
        raise ValueError("prefix length must select a non-empty prefix")
    changed = tokens.copy()
    changed[..., prefix_length:] = replacement[..., prefix_length:]
    expected = run(tokens)[..., :prefix_length]
    actual = run(changed)[..., :prefix_length]
    return compare_arrays(
        "prefix_invariance",
        expected,
        actual,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def batch_permutation_invariance(
    run: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    permutation: np.ndarray,
    *,
    absolute_tolerance: float = 0,
    relative_tolerance: float = 0,
) -> PropertyObservation:
    if sorted(permutation.tolist()) != list(range(inputs.shape[0])):
        raise ValueError("permutation must contain every batch position exactly once")
    inverse = np.argsort(permutation)
    expected = run(inputs)
    actual = run(inputs[permutation])[inverse]
    return compare_arrays(
        "batch_permutation_invariance",
        expected,
        actual,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def execution_equivalence(
    name: str,
    expected_run: Callable[[np.ndarray], np.ndarray],
    actual_run: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    *,
    absolute_tolerance: float = 0,
    relative_tolerance: float = 0,
) -> PropertyObservation:
    return compare_arrays(
        name,
        expected_run(inputs.copy()),
        actual_run(inputs.copy()),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def prefill_decode_equivalence(
    one_shot: Callable[[np.ndarray], np.ndarray],
    prefill_then_decode: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    **tolerances: float,
) -> PropertyObservation:
    return execution_equivalence(
        "prefill_decode_equivalence",
        one_shot,
        prefill_then_decode,
        inputs,
        **tolerances,
    )


def stepwise_equivalence(
    parallel: Callable[[np.ndarray], np.ndarray],
    stepwise: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    **tolerances: float,
) -> PropertyObservation:
    return execution_equivalence(
        "stepwise_equivalence",
        parallel,
        stepwise,
        inputs,
        **tolerances,
    )


def cache_equivalence(
    uncached: Callable[[np.ndarray], np.ndarray],
    cached: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    **tolerances: float,
) -> PropertyObservation:
    return execution_equivalence(
        "cache_equivalence",
        uncached,
        cached,
        inputs,
        **tolerances,
    )


def state_isolation(
    run_together: Callable[[np.ndarray], np.ndarray],
    run_one: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    *,
    absolute_tolerance: float = 0,
    relative_tolerance: float = 0,
) -> PropertyObservation:
    together = run_together(inputs.copy())
    isolated = np.stack([run_one(value.copy()) for value in inputs])
    return compare_arrays(
        "state_isolation",
        isolated,
        together,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
