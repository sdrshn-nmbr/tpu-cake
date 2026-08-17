from __future__ import annotations

from tpu_cake.ledger import RunState
from tpu_cake.simulation import (
    LifecycleSimulator,
    SimulatedFault,
    SurfaceFault,
    WorkloadSurfaceSimulator,
)
from tpu_cake.surfaces import AttentionScenario, AttentionWorkloadSurface


def test_every_simulated_fault_preserves_promotion_invariants(tmp_path) -> None:
    benign = {
        SimulatedFault.NONE,
        SimulatedFault.DUPLICATE_COMPLETION,
        SimulatedFault.RESTART,
    }
    for index, fault in enumerate(SimulatedFault):
        outcome = LifecycleSimulator(tmp_path / f"{index}.sqlite").run(seed=index + 1, fault=fault)
        if fault in benign:
            assert outcome.state is RunState.ACCEPTED
            assert outcome.mode_histories["timing"][-1] is RunState.TIMED
            assert outcome.mode_histories["trace"][-1] is RunState.TRACED
            assert outcome.mode_histories["counters"][-1] is RunState.COUNTERED
        else:
            assert outcome.state is RunState.REJECTED
            assert all(
                RunState.ACCEPTED not in history
                for history in outcome.mode_histories.values()
            )


def test_seeded_fault_selection_is_replayable(tmp_path) -> None:
    first = LifecycleSimulator(tmp_path / "first.sqlite").run(seed=7919)
    second = LifecycleSimulator(tmp_path / "second.sqlite").run(seed=7919)
    assert first.fault is second.fault
    assert first.state is second.state
    assert first.mode_histories == second.mode_histories


def _surface() -> AttentionWorkloadSurface:
    return AttentionWorkloadSurface(
        name="dst-rpa",
        scenarios=(
            AttentionScenario(
                name="decode",
                stage="steady_decode",
                batch_size=2,
                query_tokens_per_request=1,
                context_lengths=(127, 257),
                page_size=16,
                dtype="bf16",
                sharding=("d", "t"),
                weight="0.8",
            ),
            AttentionScenario(
                name="prefill",
                stage="prefill",
                batch_size=2,
                query_tokens_per_request=33,
                context_lengths=(33, 65),
                page_size=16,
                dtype="bf16",
                sharding=("d", "t"),
                weight="0.2",
            ),
        ),
        minimum_practical_improvement="0.03",
        maximum_scenario_regression="0.01",
        bootstrap_samples=1_000,
    )


def test_surface_simulation_faults_cannot_promote_invalid_evidence() -> None:
    simulator = WorkloadSurfaceSimulator(_surface())
    benign = {SurfaceFault.NONE, SurfaceFault.RESTART}

    for index, fault in enumerate(SurfaceFault):
        outcome = simulator.run(seed=10_000 + index, fault=fault)
        assert outcome.promotable is (fault in benign)
        if fault not in benign:
            assert outcome.rejection_reason is not None


def test_surface_simulation_is_replayable_across_seeded_fault_runs() -> None:
    simulator = WorkloadSurfaceSimulator(_surface())

    for seed in range(128):
        first = simulator.run(seed=seed)
        second = simulator.run(seed=seed)
        assert first == second
