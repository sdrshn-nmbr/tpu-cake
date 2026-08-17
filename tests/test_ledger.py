from __future__ import annotations

import pytest

from tpu_cake.ledger import ExperimentLedger, RunState

RUN_ID = "1" * 64


def test_ledger_resumes_after_restart_and_reaches_acceptance_in_order(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite"
    clock = iter(range(100, 200)).__next__
    with ExperimentLedger(path, clock_ns=clock) as ledger:
        ledger.create(RUN_ID, {"schedule": "a"})
        ledger.transition(RUN_ID, RunState.VERIFIED, {"verified": True})
        ledger.transition(RUN_ID, RunState.LOWERED, {"physical": "b"})
    with ExperimentLedger(path, clock_ns=clock) as ledger:
        assert ledger.current_state(RUN_ID) is RunState.LOWERED
        for state in (
            RunState.COMPILED,
            RunState.CORRECT,
            RunState.TIMED,
            RunState.TRACED,
            RunState.COUNTERED,
            RunState.ACCEPTED,
        ):
            ledger.transition(RUN_ID, state, {"state": state.value})
        assert ledger.current_state(RUN_ID) is RunState.ACCEPTED
        assert [event.state for event in ledger.history(RUN_ID)] == list(RunState)[:-1]


def test_ledger_rejects_skipped_profile_and_counter_gates(tmp_path) -> None:
    with ExperimentLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.create(RUN_ID, {})
        with pytest.raises(ValueError, match="created -> accepted"):
            ledger.transition(RUN_ID, RunState.ACCEPTED, {})


def test_duplicate_completion_is_idempotent_only_for_same_evidence(tmp_path) -> None:
    with ExperimentLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.create(RUN_ID, {"schedule": "a"})
        first = ledger.transition(RUN_ID, RunState.VERIFIED, {"hash": "a"})
        second = ledger.transition(RUN_ID, RunState.VERIFIED, {"hash": "a"})
        assert first == second
        with pytest.raises(ValueError, match="conflicting duplicate"):
            ledger.transition(RUN_ID, RunState.VERIFIED, {"hash": "b"})
