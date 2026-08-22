from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class RunState(StrEnum):
    CREATED = "created"
    VERIFIED = "verified"
    LOWERED = "lowered"
    COMPILED = "compiled"
    CORRECT = "correct"
    TIMED = "timed"
    TRACED = "traced"
    COUNTERED = "countered"
    VALIDATED = "validated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


_NEXT_STATES = {
    RunState.CREATED: frozenset({RunState.VERIFIED}),
    RunState.VERIFIED: frozenset({RunState.LOWERED}),
    RunState.LOWERED: frozenset({RunState.COMPILED}),
    RunState.COMPILED: frozenset({RunState.CORRECT}),
    RunState.CORRECT: frozenset(
        {RunState.TIMED, RunState.TRACED, RunState.COUNTERED, RunState.VALIDATED}
    ),
    RunState.TIMED: frozenset({RunState.ACCEPTED}),
    RunState.TRACED: frozenset({RunState.ACCEPTED}),
    RunState.COUNTERED: frozenset({RunState.ACCEPTED}),
    RunState.VALIDATED: frozenset({RunState.ACCEPTED}),
}


def payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class LedgerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(gt=0)
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: RunState
    timestamp_ns: int = Field(ge=0)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentLedger:
    def __init__(self, path: Path, *, clock_ns: Callable[[], int] = time.time_ns) -> None:
        self.path = path
        self._clock_ns = clock_ns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                state TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL,
                UNIQUE(run_id, state)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def create(self, run_id: str, payload: Mapping[str, object]) -> LedgerEvent:
        return self._append(run_id, RunState.CREATED, payload, previous=None)

    def transition(
        self,
        run_id: str,
        state: RunState,
        payload: Mapping[str, object],
    ) -> LedgerEvent:
        previous = self.current_state(run_id)
        if previous is None:
            raise ValueError(f"run {run_id} does not exist")
        if previous is state:
            return self._append(run_id, state, payload, previous=previous)
        if previous in {RunState.ACCEPTED, RunState.REJECTED}:
            raise ValueError(f"run {run_id} is already terminal at {previous.value}")
        if state is not RunState.REJECTED and state not in _NEXT_STATES.get(previous, ()):
            raise ValueError(f"invalid run transition {previous.value} -> {state.value}")
        return self._append(run_id, state, payload, previous=previous)

    def _append(
        self,
        run_id: str,
        state: RunState,
        payload: Mapping[str, object],
        *,
        previous: RunState | None,
    ) -> LedgerEvent:
        payload_hash = payload_sha256(payload)
        with self._connection:
            existing = self._connection.execute(
                "SELECT sequence, timestamp_ns, payload_sha256 FROM events "
                "WHERE run_id = ? AND state = ?",
                (run_id, state.value),
            ).fetchone()
            if existing is not None:
                if existing[2] != payload_hash:
                    raise ValueError(
                        f"conflicting duplicate completion for {run_id} at {state.value}"
                    )
                return LedgerEvent(
                    sequence=existing[0],
                    run_id=run_id,
                    state=state,
                    timestamp_ns=existing[1],
                    payload_sha256=existing[2],
                )
            current = self._current_state_in_transaction(run_id)
            if current != previous:
                current_label = current.value if current is not None else "missing"
                raise ValueError(
                    f"concurrent run transition changed state from "
                    f"{previous.value if previous else 'missing'} to {current_label}"
                )
            timestamp = self._clock_ns()
            cursor = self._connection.execute(
                "INSERT INTO events(run_id, state, timestamp_ns, payload_sha256) "
                "VALUES (?, ?, ?, ?)",
                (run_id, state.value, timestamp, payload_hash),
            )
            return LedgerEvent(
                sequence=cursor.lastrowid,
                run_id=run_id,
                state=state,
                timestamp_ns=timestamp,
                payload_sha256=payload_hash,
            )

    def _current_state_in_transaction(self, run_id: str) -> RunState | None:
        row = self._connection.execute(
            "SELECT state FROM events WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return RunState(row[0]) if row is not None else None

    def current_state(self, run_id: str) -> RunState | None:
        return self._current_state_in_transaction(run_id)

    def history(self, run_id: str) -> tuple[LedgerEvent, ...]:
        rows = self._connection.execute(
            "SELECT sequence, state, timestamp_ns, payload_sha256 FROM events "
            "WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        return tuple(
            LedgerEvent(
                sequence=row[0],
                run_id=run_id,
                state=RunState(row[1]),
                timestamp_ns=row[2],
                payload_sha256=row[3],
            )
            for row in rows
        )


class EvidenceRun:
    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self._clock_ns = clock_ns

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def create(self, payload: Mapping[str, object]) -> LedgerEvent:
        with ExperimentLedger(self.path, clock_ns=self._clock_ns) as ledger:
            return ledger.create(self.run_id, payload)

    def transition(
        self,
        state: RunState,
        payload: Mapping[str, object],
    ) -> LedgerEvent:
        with ExperimentLedger(self.path, clock_ns=self._clock_ns) as ledger:
            return ledger.transition(self.run_id, state, payload)

    def record(
        self,
        state: RunState,
        payload: Mapping[str, object],
        *,
        conflict_error: str = "conflicting duplicate completion for {run_id} at {state}",
    ) -> LedgerEvent:
        expected_hash = payload_sha256(payload)
        with ExperimentLedger(self.path, clock_ns=self._clock_ns) as ledger:
            existing = next(
                (event for event in ledger.history(self.run_id) if event.state is state),
                None,
            )
            if existing is not None:
                if existing.payload_sha256 != expected_hash:
                    raise ValueError(conflict_error.format(run_id=self.run_id, state=state.value))
                return existing
            if state is RunState.CREATED:
                return ledger.create(self.run_id, payload)
            return ledger.transition(self.run_id, state, payload)

    def current_state(self) -> RunState | None:
        with ExperimentLedger(self.path, clock_ns=self._clock_ns) as ledger:
            return ledger.current_state(self.run_id)

    def seal(self, sidecar_error: str) -> None:
        seal_ledger(self.path, sidecar_error)


def finalize_ledger(path: Path) -> tuple[Path, ...]:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    sidecars = (
        path.with_name(f"{path.name}-shm"),
        path.with_name(f"{path.name}-wal"),
    )
    return tuple(sidecar for sidecar in sidecars if sidecar.exists())


def seal_ledger(path: Path, sidecar_error: str) -> None:
    present = finalize_ledger(path)
    if present:
        raise ValueError(sidecar_error.format(paths=present))


def read_ledger_history(path: Path, run_id: str) -> tuple[LedgerEvent, ...]:
    sidecars = (
        path.with_name(f"{path.name}-shm"),
        path.with_name(f"{path.name}-wal"),
    )
    present = tuple(sidecar for sidecar in sidecars if sidecar.exists())
    if present:
        names = ", ".join(sidecar.name for sidecar in present)
        raise ValueError(f"LEDGER_SIDECAR_PRESENT files={names}")
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT sequence, state, timestamp_ns, payload_sha256 FROM events "
            "WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
    return tuple(
        LedgerEvent(
            sequence=row[0],
            run_id=run_id,
            state=RunState(row[1]),
            timestamp_ns=row[2],
            payload_sha256=row[3],
        )
        for row in rows
    )
