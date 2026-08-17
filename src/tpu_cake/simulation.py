from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tpu_cake.ledger import ExperimentLedger, RunState
from tpu_cake.surfaces import (
    AttentionWorkloadSurface,
    ScenarioObservation,
    SurfaceCandidateObservation,
    SurfaceComparison,
    compare_surface_candidates,
)


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


class SurfaceFault(StrEnum):
    NONE = "none"
    RESTART = "restart"
    OUTPUT_CORRUPTION = "output_corruption"
    INPUT_DRIFT = "input_drift"
    RUNTIME_DRIFT = "runtime_drift"
    PROFILED_TIMING = "profiled_timing"
    WRONG_ORDER = "wrong_order"
    SCENARIO_REGRESSION = "scenario_regression"
    MISSING_SCENARIO = "missing_scenario"
    FAILED_CORRECTNESS = "failed_correctness"


UNMODELLED_SURFACE_FAILURES = (
    "TPU timing and numerical behavior",
    "compiler and runtime identity collection",
    "host scheduling and device contention",
    "profiler instrumentation overhead",
    "artifact durability below the observation contract",
)


@dataclass(frozen=True)
class SurfaceSimulationOutcome:
    seed: int
    fault: SurfaceFault
    promotable: bool
    rejection_reason: str | None
    comparison: SurfaceComparison | None


class WorkloadSurfaceSimulator:
    """Seeded promotion simulation over the real workload-surface contracts.

    The timing boundary is fake. Validation, matched comparison, deterministic
    bootstrap, and promotion rules are production code. The harness can express
    only SurfaceFault. It cannot prove any item in UNMODELLED_SURFACE_FAILURES;
    those require clean live TPU runs and separate device evidence.
    """

    def __init__(self, surface: AttentionWorkloadSurface) -> None:
        self.surface = surface

    @staticmethod
    def _digest(*parts: str) -> str:
        return hashlib.sha256(":".join(parts).encode()).hexdigest()

    def _observations(
        self,
        *,
        seed: int,
    ) -> tuple[SurfaceCandidateObservation, SurfaceCandidateObservation]:
        generator = random.Random(seed)
        baseline_scenarios: list[ScenarioObservation] = []
        candidate_scenarios: list[ScenarioObservation] = []
        baseline_starts = tuple(index % 2 == seed % 2 for index in range(7))
        candidate_starts = tuple(not value for value in baseline_starts)
        runtime = self._digest(self.surface.surface_id, "runtime")
        for scenario in self.surface.scenarios:
            base_ns = (
                sum(scenario.context_lengths)
                + scenario.batch_size * scenario.query_tokens_per_request
            ) * 1_000
            baseline_rounds = tuple(
                max(1, round(base_ns * (1 + generator.uniform(-0.01, 0.01))))
                for _ in range(7)
            )
            candidate_rounds = tuple(
                max(1, round(value * (0.9 + generator.uniform(-0.002, 0.002))))
                for value in baseline_rounds
            )
            shared = {
                "scenario": scenario.name,
                "input_sha256": self._digest(self.surface.surface_id, scenario.name, "input"),
                "output_sha256": self._digest(self.surface.surface_id, scenario.name, "output"),
                "runtime_sha256": runtime,
                "profiled": False,
                "passed": True,
            }
            baseline_scenarios.append(
                ScenarioObservation(
                    **shared,
                    round_medians_ns=baseline_rounds,
                    ran_first=baseline_starts,
                )
            )
            candidate_scenarios.append(
                ScenarioObservation(
                    **shared,
                    round_medians_ns=candidate_rounds,
                    ran_first=candidate_starts,
                )
            )
        return (
            SurfaceCandidateObservation(
                candidate="baseline",
                scenarios=tuple(baseline_scenarios),
            ),
            SurfaceCandidateObservation(
                candidate="candidate",
                scenarios=tuple(candidate_scenarios),
            ),
        )

    def run(
        self,
        *,
        seed: int,
        fault: SurfaceFault | None = None,
    ) -> SurfaceSimulationOutcome:
        selected_fault = fault or random.Random(seed).choice(tuple(SurfaceFault))
        baseline, candidate = self._observations(seed=seed)
        first = candidate.scenarios[0]
        if selected_fault is SurfaceFault.RESTART:
            baseline = SurfaceCandidateObservation.model_validate_json(
                baseline.model_dump_json()
            )
            candidate = SurfaceCandidateObservation.model_validate_json(
                candidate.model_dump_json()
            )
        elif selected_fault is SurfaceFault.OUTPUT_CORRUPTION:
            first = first.model_copy(update={"output_sha256": "f" * 64})
        elif selected_fault is SurfaceFault.INPUT_DRIFT:
            first = first.model_copy(update={"input_sha256": "f" * 64})
        elif selected_fault is SurfaceFault.RUNTIME_DRIFT:
            first = first.model_copy(update={"runtime_sha256": "f" * 64})
        elif selected_fault is SurfaceFault.PROFILED_TIMING:
            first = first.model_copy(update={"profiled": True})
        elif selected_fault is SurfaceFault.WRONG_ORDER:
            first = first.model_copy(update={"ran_first": baseline.scenarios[0].ran_first})
        elif selected_fault is SurfaceFault.SCENARIO_REGRESSION:
            first = first.model_copy(
                update={
                    "round_medians_ns": tuple(
                        round(value * 1.05)
                        for value in baseline.scenarios[0].round_medians_ns
                    )
                }
            )
        elif selected_fault is SurfaceFault.FAILED_CORRECTNESS:
            first = first.model_copy(update={"passed": False})
        if first is not candidate.scenarios[0]:
            candidate = candidate.model_copy(
                update={"scenarios": (first, *candidate.scenarios[1:])}
            )
        if selected_fault is SurfaceFault.MISSING_SCENARIO:
            candidate = candidate.model_copy(update={"scenarios": candidate.scenarios[:-1]})

        try:
            comparison = compare_surface_candidates(self.surface, baseline, candidate)
        except ValueError as error:
            return SurfaceSimulationOutcome(
                seed=seed,
                fault=selected_fault,
                promotable=False,
                rejection_reason=str(error),
                comparison=None,
            )
        return SurfaceSimulationOutcome(
            seed=seed,
            fault=selected_fault,
            promotable=comparison.promotable,
            rejection_reason=None if comparison.promotable else "promotion criteria not met",
            comparison=comparison,
        )
