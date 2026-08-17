from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence

import jax
import numpy as np

from tpu_cake.identity import array_sha256, semantic_seed, semantic_sha256
from tpu_cake.surfaces import (
    AttentionScenario,
    AttentionWorkloadSurface,
    ScenarioObservation,
    SurfaceCandidateObservation,
    SurfaceComparison,
    compare_surface_candidates,
)

ArrayTuple = tuple[np.ndarray | jax.Array, ...]
InputFactory = Callable[[AttentionScenario, int], ArrayTuple]
Oracle = Callable[[AttentionScenario, ArrayTuple], ArrayTuple]
Candidate = Callable[[AttentionScenario, ArrayTuple], ArrayTuple]


def _as_numpy(values: Sequence[np.ndarray | jax.Array]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(value) for value in jax.block_until_ready(tuple(values)))


def _arrays_identity(values: Sequence[np.ndarray | jax.Array]) -> str:
    return semantic_sha256(
        "array-tuple-v1",
        *(array_sha256(value) for value in _as_numpy(values)),
    )


def _correct(
    actual: Sequence[np.ndarray | jax.Array],
    expected: Sequence[np.ndarray | jax.Array],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    actual_arrays = _as_numpy(actual)
    expected_arrays = _as_numpy(expected)
    return len(actual_arrays) == len(expected_arrays) and all(
        np.allclose(
            observed,
            reference,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        for observed, reference in zip(actual_arrays, expected_arrays, strict=True)
    )


def _measure(
    candidate: Candidate,
    scenario: AttentionScenario,
    inputs: ArrayTuple,
    *,
    iterations: int,
) -> int:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = candidate(scenario, inputs)
        jax.block_until_ready(output)
        samples.append(time.perf_counter_ns() - started)
    return int(statistics.median(samples))


def run_surface_pair(
    surface: AttentionWorkloadSurface,
    *,
    baseline_name: str,
    candidate_name: str,
    baseline: Candidate,
    candidate: Candidate,
    input_factory: InputFactory,
    oracle: Oracle,
    runtime_sha256: str,
    rounds: int = 6,
    warmup_iterations: int = 2,
    measured_iterations: int = 5,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> tuple[SurfaceComparison, SurfaceCandidateObservation, SurfaceCandidateObservation]:
    if not baseline_name or not candidate_name or baseline_name == candidate_name:
        raise ValueError("surface execution needs two distinct candidate names")
    if len(runtime_sha256) != 64 or any(value not in "0123456789abcdef" for value in runtime_sha256):
        raise ValueError("surface execution needs a SHA-256 runtime identity")
    if rounds < 5 or warmup_iterations < 1 or measured_iterations < 1:
        raise ValueError("surface execution needs at least five rounds and positive iteration counts")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("surface tolerances must be nonnegative")

    observations: dict[str, list[ScenarioObservation]] = {
        baseline_name: [],
        candidate_name: [],
    }
    candidates = {baseline_name: baseline, candidate_name: candidate}
    for scenario in surface.scenarios:
        seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
        inputs = input_factory(scenario, seed)
        expected = oracle(scenario, inputs)
        input_identity = _arrays_identity(inputs)
        oracle_identity = _arrays_identity(expected)
        outputs = {
            name: implementation(scenario, inputs)
            for name, implementation in candidates.items()
        }
        passed = {
            name: _correct(
                output,
                expected,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            for name, output in outputs.items()
        }
        if not all(passed.values()):
            failed = sorted(name for name, value in passed.items() if not value)
            raise ValueError(
                f"surface candidates failed the numerical oracle: {failed}"
            )
        for name, implementation in candidates.items():
            for _ in range(warmup_iterations):
                jax.block_until_ready(implementation(scenario, inputs))

        round_samples = {baseline_name: [], candidate_name: []}
        ran_first = {baseline_name: [], candidate_name: []}
        for round_index in range(rounds):
            order = (
                (baseline_name, candidate_name)
                if round_index % 2 == 0
                else (candidate_name, baseline_name)
            )
            for position, name in enumerate(order):
                ran_first[name].append(position == 0)
                round_samples[name].append(
                    _measure(
                        candidates[name],
                        scenario,
                        inputs,
                        iterations=measured_iterations,
                    )
                )
        for name in candidates:
            observations[name].append(
                ScenarioObservation(
                    scenario=scenario.name,
                    round_medians_ns=tuple(round_samples[name]),
                    ran_first=tuple(ran_first[name]),
                    input_sha256=input_identity,
                    output_sha256=oracle_identity,
                    runtime_sha256=runtime_sha256,
                    profiled=False,
                    passed=True,
                )
            )

    baseline_observation = SurfaceCandidateObservation(
        candidate=baseline_name,
        scenarios=tuple(observations[baseline_name]),
    )
    candidate_observation = SurfaceCandidateObservation(
        candidate=candidate_name,
        scenarios=tuple(observations[candidate_name]),
    )
    comparison = compare_surface_candidates(
        surface,
        baseline_observation,
        candidate_observation,
    )
    return comparison, baseline_observation, candidate_observation
