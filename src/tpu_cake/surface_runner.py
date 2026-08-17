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
    SeqaxForwardScenario,
    SeqaxForwardWorkloadSurface,
    SurfaceCandidateObservation,
    SurfaceComparison,
    compare_surface_candidates,
)

ArrayTuple = tuple[np.ndarray | jax.Array, ...]
Scenario = AttentionScenario | SeqaxForwardScenario
Surface = AttentionWorkloadSurface | SeqaxForwardWorkloadSurface
InputFactory = Callable[[Scenario, int], ArrayTuple]
Oracle = Callable[[Scenario, ArrayTuple], ArrayTuple]
Candidate = Callable[[Scenario, ArrayTuple], ArrayTuple]
ProgressCallback = Callable[[], None]


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
    scenario: Scenario,
    inputs: ArrayTuple,
    *,
    iterations: int,
) -> tuple[int, ...]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = candidate(scenario, inputs)
        jax.block_until_ready(output)
        samples.append(time.perf_counter_ns() - started)
    return tuple(samples)


def run_surface_pair(
    surface: Surface,
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
    on_correctness_complete: ProgressCallback | None = None,
    on_timing_complete: ProgressCallback | None = None,
) -> tuple[SurfaceComparison, SurfaceCandidateObservation, SurfaceCandidateObservation]:
    if not baseline_name or not candidate_name or baseline_name == candidate_name:
        raise ValueError("surface execution needs two distinct candidate names")
    if len(runtime_sha256) != 64 or any(value not in "0123456789abcdef" for value in runtime_sha256):
        raise ValueError("surface execution needs a SHA-256 runtime identity")
    if rounds < 5 or warmup_iterations < 1 or measured_iterations < 1:
        raise ValueError("surface execution needs at least five rounds and positive iteration counts")
    if measured_iterations % 2 == 0:
        raise ValueError("surface execution needs an odd measured iteration count")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("surface tolerances must be nonnegative")

    observations: dict[str, list[ScenarioObservation]] = {
        baseline_name: [],
        candidate_name: [],
    }
    candidates = {baseline_name: baseline, candidate_name: candidate}
    prepared: dict[
        str,
        tuple[
            ArrayTuple,
            str,
            dict[str, str],
        ],
    ] = {}
    for scenario in surface.scenarios:
        seed = semantic_seed(surface.surface_id, scenario.name, "inputs")
        inputs = input_factory(scenario, seed)
        expected = oracle(scenario, inputs)
        input_identity = _arrays_identity(inputs)
        outputs = {
            name: implementation(scenario, inputs)
            for name, implementation in candidates.items()
        }
        output_identities = {
            name: _arrays_identity(output) for name, output in outputs.items()
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
                "surface candidates failed the numerical oracle: "
                f"scenario={scenario.name} candidates={failed}"
            )
        prepared[scenario.name] = (
            inputs,
            input_identity,
            output_identities,
        )

    if on_correctness_complete is not None:
        on_correctness_complete()

    for scenario in surface.scenarios:
        inputs, input_identity, output_identities = prepared[scenario.name]
        for name, implementation in candidates.items():
            for _ in range(warmup_iterations):
                jax.block_until_ready(implementation(scenario, inputs))

        round_samples = {baseline_name: [], candidate_name: []}
        raw_round_samples = {baseline_name: [], candidate_name: []}
        ran_first = {baseline_name: [], candidate_name: []}
        for round_index in range(rounds):
            order = (
                (baseline_name, candidate_name)
                if round_index % 2 == 0
                else (candidate_name, baseline_name)
            )
            for position, name in enumerate(order):
                ran_first[name].append(position == 0)
                samples = _measure(
                    candidates[name],
                    scenario,
                    inputs,
                    iterations=measured_iterations,
                )
                raw_round_samples[name].append(samples)
                round_samples[name].append(int(statistics.median(samples)))
        for name in candidates:
            observations[name].append(
                ScenarioObservation(
                    scenario=scenario.name,
                    round_medians_ns=tuple(round_samples[name]),
                    round_samples_ns=tuple(raw_round_samples[name]),
                    ran_first=tuple(ran_first[name]),
                    input_sha256=input_identity,
                    output_sha256=output_identities[name],
                    runtime_sha256=runtime_sha256,
                    profiled=False,
                    passed=True,
                )
            )

    if on_timing_complete is not None:
        on_timing_complete()

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
