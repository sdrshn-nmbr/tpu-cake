from __future__ import annotations

import hashlib

import pytest

from tpu_cake.ledger import ExperimentLedger, RunState, read_ledger_history

RUN_ID = "1" * 64


def test_ledger_resumes_after_restart_and_reaches_timing_evidence(tmp_path) -> None:
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
        ):
            ledger.transition(RUN_ID, state, {"state": state.value})
        assert ledger.current_state(RUN_ID) is RunState.TIMED
        assert [event.state for event in ledger.history(RUN_ID)] == [
            RunState.CREATED,
            RunState.VERIFIED,
            RunState.LOWERED,
            RunState.COMPILED,
            RunState.CORRECT,
            RunState.TIMED,
        ]


@pytest.mark.parametrize("terminal", (RunState.TIMED, RunState.TRACED, RunState.COUNTERED))
def test_each_measurement_mode_has_an_independent_terminal_state(tmp_path, terminal) -> None:
    with ExperimentLedger(tmp_path / f"{terminal.value}.sqlite") as ledger:
        ledger.create(RUN_ID, {})
        for state in (
            RunState.VERIFIED,
            RunState.LOWERED,
            RunState.COMPILED,
            RunState.CORRECT,
            terminal,
        ):
            ledger.transition(RUN_ID, state, {"state": state.value})
        assert ledger.current_state(RUN_ID) is terminal


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


def test_evidence_validation_reads_ledger_without_mutating_it(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite"
    with ExperimentLedger(path) as ledger:
        ledger.create(RUN_ID, {"schedule": "a"})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_ledger_history(path, RUN_ID)[0].state is RunState.CREATED
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert after == before
