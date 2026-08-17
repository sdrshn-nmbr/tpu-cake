from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tpu_cake.ledger import ExperimentLedger, RunState


class SimulatedFault(StrEnum):
    NONE = "none"
    COMPILER_FAILURE = "compiler_failure"
    TPU_LOSS = "tpu_loss"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    PARTIAL_ARTIFACT_COPY = "partial_artifact_copy"
    CORRUPTED_RECEIPT = "corrupted_receipt"
    WRONG_PHASE_PROFILE = "wrong_phase_profile"
    DUPLICATE_COMPLETION = "duplicate_completion"
    RESTART = "restart"
    STALE_HARDWARE = "stale_hardware"


SUPPORTED_FAULT_SPACE = tuple(SimulatedFault)
UNMODELLED_FAILURES = (
    "Pallas compiler correctness",
    "TPU numerical behavior",
    "TPU firmware and libtpu failure modes",
    "XProf collector fidelity",
    "device performance and contention",
    "filesystem and SQLite implementation bugs below their public contracts",
)


class InjectedFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class SimulationOutcome:
    seed: int
    fault: SimulatedFault
    state: RunState
    mode_histories: dict[str, tuple[RunState, ...]]


class StatefulBoundaryFake:
    """Stateful fake for hardware, compiler, profiler, and artifact-store boundaries."""

    def __init__(self, fault: SimulatedFault) -> None:
        self.fault = fault
        self.completed: list[RunState] = []

    def complete(self, state: RunState) -> dict[str, object]:
        failure_by_state = {
            RunState.COMPILED: SimulatedFault.COMPILER_FAILURE,
            RunState.CORRECT: SimulatedFault.TPU_LOSS,
            RunState.TIMED: SimulatedFault.TIMEOUT,
            RunState.TRACED: SimulatedFault.WRONG_PHASE_PROFILE,
            RunState.COUNTERED: SimulatedFault.CANCELLATION,
        }
        if failure_by_state.get(state) is self.fault:
            raise InjectedFailure(self.fault.value)
        payload: dict[str, object] = {"state": state.value, "complete": True}
        if self.fault is SimulatedFault.PARTIAL_ARTIFACT_COPY and state is RunState.COUNTERED:
            payload["artifacts_complete"] = False
        if self.fault is SimulatedFault.CORRUPTED_RECEIPT and state is RunState.COUNTERED:
            payload["receipt_hash_valid"] = False
        if self.fault is SimulatedFault.WRONG_PHASE_PROFILE and state is RunState.TRACED:
            payload["profile_stage"] = "prefill"
        else:
            payload["profile_stage"] = "distributed_matmul"
        if self.fault is SimulatedFault.STALE_HARDWARE and state is RunState.COMPILED:
            payload["hardware_identity"] = "stale"
        else:
            payload["hardware_identity"] = "expected"
        self.completed.append(state)
        return payload


class LifecycleSimulator:
    """Deterministic lifecycle sim.

    It exercises the real SQLite ledger and promotion state machine. It can inject
    only the faults in SUPPORTED_FAULT_SPACE. It cannot prove any item listed in
    UNMODELLED_FAILURES; those require live compiler, TPU, and profiler tests.
    """

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = ledger_path

    def run(self, *, seed: int, fault: SimulatedFault | None = None) -> SimulationOutcome:
        selected_fault = fault or random.Random(seed).choice(SUPPORTED_FAULT_SPACE)
        boundary = StatefulBoundaryFake(selected_fault)
        mode_histories: dict[str, tuple[RunState, ...]] = {}
        terminal_by_mode = {
            "timing": RunState.TIMED,
            "trace": RunState.TRACED,
            "counters": RunState.COUNTERED,
        }
        outcome_state = RunState.ACCEPTED
        for mode_index, (mode, terminal) in enumerate(terminal_by_mode.items()):
            run_id = f"{seed * 10 + mode_index:064x}"[-64:]
            path = self.ledger_path.with_name(f"{self.ledger_path.stem}-{mode}.sqlite")
            ledger = ExperimentLedger(path)
            try:
                ledger.create(
                    run_id,
                    {"seed": seed, "fault": selected_fault.value, "mode": mode},
                )
                for state in (
                    RunState.VERIFIED,
                    RunState.LOWERED,
                    RunState.COMPILED,
                    RunState.CORRECT,
                    terminal,
                ):
                    payload = boundary.complete(state)
                    if payload.get("hardware_identity") != "expected":
                        raise InjectedFailure("stale hardware identity")
                    if (
                        state is RunState.TRACED
                        and payload.get("profile_stage") != "distributed_matmul"
                    ):
                        raise InjectedFailure("wrong phase profile")
                    ledger.transition(run_id, state, payload)
                    if selected_fault is SimulatedFault.DUPLICATE_COMPLETION and state is terminal:
                        ledger.transition(run_id, state, payload)
                    if selected_fault is SimulatedFault.RESTART and state is RunState.LOWERED:
                        ledger.close()
                        ledger = ExperimentLedger(path)
            except InjectedFailure as error:
                ledger.transition(run_id, RunState.REJECTED, {"reason": str(error)})
                outcome_state = RunState.REJECTED
            finally:
                mode_histories[mode] = tuple(
                    event.state for event in ledger.history(run_id)
                )
                ledger.close()
            if outcome_state is RunState.REJECTED:
                break
        if outcome_state is RunState.ACCEPTED:
            try:
                counter_payload = boundary.complete(RunState.COUNTERED)
                if counter_payload.get("artifacts_complete") is False:
                    raise InjectedFailure("partial artifact copy")
                if counter_payload.get("receipt_hash_valid") is False:
                    raise InjectedFailure("corrupted receipt")
            except InjectedFailure:
                outcome_state = RunState.REJECTED
        return SimulationOutcome(
            seed=seed,
            fault=selected_fault,
            state=outcome_state,
            mode_histories=mode_histories,
        )
