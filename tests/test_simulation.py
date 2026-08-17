from __future__ import annotations

from tpu_cake.ledger import RunState
from tpu_cake.simulation import LifecycleSimulator, SimulatedFault


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
