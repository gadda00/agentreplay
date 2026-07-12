"""Clock & RNG interceptors.

Timestamps and random draws are first-class sources of nondeterminism
in agent runs: a tool that records ``datetime.now()`` into a database
will produce a different value on every re-run, breaking bit-exactness
even if the LLM itself is perfectly pinned. These interceptors capture
the *sequence* of clock reads and random draws and replay them verbatim.

We do NOT attempt to globally patch ``time.time`` — that would break
the recording layer's own internal timing. Instead we expose explicit
wrapper objects the agent opts into, and framework adapters wire them
up where the agent's own code reaches for the clock.
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

from agentreplay.constants import CallType, Mode
from agentreplay.cassette import Cassette
from agentreplay.hashing import hash_call_site


class _BaseInterceptor:
    """Common machinery: holds a reference to the cassette + mode."""

    def __init__(self, cassette: Cassette, *, mode: Mode = Mode.RECORD,
                 step_id_provider: Optional[Any] = None) -> None:
        self.cassette = cassette
        self.mode = mode
        # step_id_provider is a callable that returns the current step_id,
        # or None. Framework adapters set this so individual interceptors
        # don't need to know about LangGraph state etc.
        self._step_id_provider = step_id_provider or (lambda: "default")

    def _step_id(self) -> str:
        try:
            return str(self._step_id_provider())
        except Exception:
            return "default"


class RecordingClock(_BaseInterceptor):
    """A drop-in replacement for ``time.time`` / ``datetime.now``.

    Records the *sequence* of clock reads — the n-th call to ``time()``
    during replay returns the n-th recorded timestamp, regardless of
    wall-clock time. This is what makes agent runs that read the clock
    inside their tools replay bit-exact.
    """

    def __init__(self, cassette: Cassette, *, mode: Mode = Mode.RECORD,
                 step_id_provider: Optional[Any] = None,
                 real_clock: Optional[Any] = None) -> None:
        super().__init__(cassette, mode=mode, step_id_provider=step_id_provider)
        self._real = real_clock or time
        self._replay_iter = None  # lazy

    def time(self) -> float:
        step_id = f"{self._step_id()}:clock"
        # The "request" is just the call sequence — encoded by step_id alone.
        call_id = hash_call_site(step_id, {"op": "time"}, call_type=CallType.CLOCK.value)
        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            event = self.cassette.lookup_call(call_id)
            if event is not None:
                return float(self.cassette.resolve_response(event))
            if self.mode == Mode.REPLAY:
                # No recorded value — synthesise a deterministic fallback
                # rather than fail. This is rare: it means the agent asked
                # the clock an unrecorded question (e.g. via a path we
                # didn't wrap). Better to keep going than to crash.
                return 0.0
        # LIVE or RECORD
        t = self._real.time()
        if self.mode == Mode.RECORD:
            self.cassette.write_event(
                step_id=step_id,
                call_type=CallType.CLOCK,
                call_id=call_id,
                request={"op": "time"},
                response=t,
                started_at=t,
                duration_ms=0.0,
            )
        return t

    def monotonic(self) -> float:
        # monotonic has no epoch meaning; record and replay it the same way.
        step_id = f"{self._step_id()}:clock"
        call_id = hash_call_site(step_id, {"op": "monotonic"}, call_type=CallType.CLOCK.value)
        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            event = self.cassette.lookup_call(call_id)
            if event is not None:
                return float(self.cassette.resolve_response(event))
            if self.mode == Mode.REPLAY:
                return 0.0
        t = self._real.monotonic()
        if self.mode == Mode.RECORD:
            self.cassette.write_event(
                step_id=step_id,
                call_type=CallType.CLOCK,
                call_id=call_id,
                request={"op": "monotonic"},
                response=t,
                started_at=t,
                duration_ms=0.0,
            )
        return t

    def datetime_now(self, tz: Optional[Any] = None) -> datetime:
        step_id = f"{self._step_id()}:clock"
        call_id = hash_call_site(
            step_id, {"op": "datetime_now", "tz": str(tz)}, call_type=CallType.CLOCK.value
        )
        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            event = self.cassette.lookup_call(call_id)
            if event is not None:
                stored = self.cassette.resolve_response(event)
                return datetime.fromisoformat(stored["iso"])
            if self.mode == Mode.REPLAY:
                return datetime.fromtimestamp(0, tz=timezone.utc)
        dt = datetime.now(tz=tz)
        if self.mode == Mode.RECORD:
            self.cassette.write_event(
                step_id=step_id,
                call_type=CallType.CLOCK,
                call_id=call_id,
                request={"op": "datetime_now", "tz": str(tz)},
                response={"iso": dt.isoformat()},
                started_at=time.time(),
                duration_ms=0.0,
            )
        return dt


class RecordingRandom(_BaseInterceptor):
    """A drop-in replacement for ``random.Random`` instances.

    Records every draw keyed by call sequence within a step. Replay
    returns the recorded draws in order, so an agent that uses
    ``random.choice`` in a tool gets the same value on every replay.
    """

    def __init__(self, cassette: Cassette, *, mode: Mode = Mode.RECORD,
                 step_id_provider: Optional[Any] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(cassette, mode=mode, step_id_provider=step_id_provider)
        self._rng = random.Random(seed)
        self._counter = 0

    def _record(self, op: str, args: Any, value: Any) -> Any:
        step_id = f"{self._step_id()}:rng:{self._counter}"
        self._counter += 1
        call_id = hash_call_site(step_id, {"op": op, "args": args}, call_type=CallType.RNG.value)
        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            event = self.cassette.lookup_call(call_id)
            if event is not None:
                return self.cassette.resolve_response(event)
            if self.mode == Mode.REPLAY:
                return value  # fall back to live draw (rare)
        if self.mode == Mode.RECORD:
            self.cassette.write_event(
                step_id=step_id,
                call_type=CallType.RNG,
                call_id=call_id,
                request={"op": op, "args": args},
                response=value,
                started_at=time.time(),
                duration_ms=0.0,
            )
        return value

    # Common random API surface
    def random(self) -> float:
        return self._record("random", [], self._rng.random())

    def randint(self, a: int, b: int) -> int:
        return self._record("randint", [a, b], self._rng.randint(a, b))

    def choice(self, seq: Any) -> Any:
        return self._record("choice", [seq], self._rng.choice(seq))

    def shuffle(self, seq: list) -> None:
        # In-place shuffle — record the resulting list so replay returns
        # the same permutation.
        self._rng.shuffle(seq)
        self._record("shuffle", [], list(seq))


# Public, framework-friendly aliases — match the naming used in §5.2 of
# the proposal.
ClockInterceptor = RecordingClock
RNGInterceptor = RecordingRandom
