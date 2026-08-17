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
            assert outcome.history[-2:] == (RunState.COUNTERED, RunState.ACCEPTED)
        else:
            assert outcome.state is RunState.REJECTED
            assert RunState.ACCEPTED not in outcome.history


def test_seeded_fault_selection_is_replayable(tmp_path) -> None:
    first = LifecycleSimulator(tmp_path / "first.sqlite").run(seed=7919)
    second = LifecycleSimulator(tmp_path / "second.sqlite").run(seed=7919)
    assert first.fault is second.fault
    assert first.state is second.state
    assert first.history == second.history
