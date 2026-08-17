from decimal import Decimal

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_cake.contracts import WorkloadStage
from tpu_cake.surface_runner import run_surface_pair
from tpu_cake.surfaces import AttentionScenario, AttentionWorkloadSurface


def _surface() -> AttentionWorkloadSurface:
    return AttentionWorkloadSurface(
        name="real-execution-control",
        scenarios=(
            AttentionScenario(
                name="short",
                stage=WorkloadStage.STEADY_DECODE,
                batch_size=2,
                query_tokens_per_request=1,
                context_lengths=(8, 13),
                page_size=16,
                dtype="f32",
                sharding=("",),
                weight=Decimal(1),
            ),
            AttentionScenario(
                name="long",
                stage=WorkloadStage.STEADY_DECODE,
                batch_size=2,
                query_tokens_per_request=1,
                context_lengths=(31, 47),
                page_size=16,
                dtype="f32",
                sharding=("",),
                weight=Decimal(1),
            ),
        ),
        minimum_practical_improvement=Decimal("0.01"),
        bootstrap_samples=1_000,
    )


def _inputs(scenario: AttentionScenario, seed: int):
    generator = np.random.default_rng(seed)
    return (generator.normal(size=(scenario.batch_size, 8)).astype(np.float32),)


def _oracle(_scenario: AttentionScenario, inputs):
    return (np.tanh(np.asarray(inputs[0])),)


def _jax_candidate(_scenario: AttentionScenario, inputs):
    return (jnp.tanh(inputs[0]),)


def test_surface_runner_measures_real_matched_executions() -> None:
    comparison, baseline, candidate = run_surface_pair(
        _surface(),
        baseline_name="baseline",
        candidate_name="candidate",
        baseline=_jax_candidate,
        candidate=_jax_candidate,
        input_factory=_inputs,
        oracle=_oracle,
        runtime_sha256="a" * 64,
        rounds=6,
        warmup_iterations=1,
        measured_iterations=3,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )

    assert comparison.surface_id == _surface().surface_id
    assert len(baseline.scenarios) == len(candidate.scenarios) == 2
    assert all(len(value.round_medians_ns) == 6 for value in baseline.scenarios)
    assert all(
        len(samples) == 3
        for observation in baseline.scenarios
        for samples in observation.round_samples_ns
    )
    assert baseline.scenarios[0].ran_first == (True, False, True, False, True, False)
    assert candidate.scenarios[0].ran_first == (False, True, False, True, False, True)
    assert baseline.scenarios[0].input_sha256 == candidate.scenarios[0].input_sha256
    assert baseline.scenarios[0].output_sha256 == candidate.scenarios[0].output_sha256


def test_surface_runner_rejects_a_wrong_real_candidate_before_timing() -> None:
    def wrong(_scenario: AttentionScenario, inputs):
        return (jnp.asarray(inputs[0]) + 1,)

    with pytest.raises(ValueError, match="failed the numerical oracle"):
        run_surface_pair(
            _surface(),
            baseline_name="baseline",
            candidate_name="wrong",
            baseline=_jax_candidate,
            candidate=wrong,
            input_factory=_inputs,
            oracle=_oracle,
            runtime_sha256="b" * 64,
            rounds=5,
            warmup_iterations=1,
            measured_iterations=1,
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
        )
