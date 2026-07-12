"""Exception hierarchy for agentreplay.

All errors raised by this library derive from :class:`AgentReplayError`
so callers can catch them with a single ``except`` clause.
"""
from __future__ import annotations

from typing import Any, Optional


class AgentReplayError(Exception):
    """Base class for every error raised by agentreplay."""


class CassetteError(AgentReplayError):
    """Cassette-level failure: corrupt files, schema mismatch, IO error."""


class CassetteNotFoundError(CassetteError):
    """A cassette path or ID did not resolve to an existing cassette."""


class DivergenceError(AgentReplayError):
    """Raised during pure replay when the agent asks for a call whose
    canonicalized input no longer matches any recorded call-site ID.

    Carries structured context so the CLI can render a useful diff
    instead of a bare traceback.
    """

    def __init__(
        self,
        step_id: str,
        call_type: str,
        expected_call_id: Optional[str],
        actual_call_id: Optional[str],
        recorded_request: Optional[Any] = None,
        actual_request: Optional[Any] = None,
        message: Optional[str] = None,
    ) -> None:
        self.step_id = step_id
        self.call_type = call_type
        self.expected_call_id = expected_call_id
        self.actual_call_id = actual_call_id
        self.recorded_request = recorded_request
        self.actual_request = actual_request
        super().__init__(
            message
            or f"Divergence at step {step_id!r} ({call_type}): "
            f"recorded call_id={expected_call_id!r} vs actual call_id={actual_call_id!r}"
        )


class MutationError(AgentReplayError):
    """A counterfactual mutation could not be applied (e.g. step index out
    of range, payload shape mismatch)."""


class ReplayExhaustedError(AgentReplayError):
    """The agent kept making calls after the cassette ran out of recorded
    events, in pure-replay mode."""


class ConfigurationError(AgentReplayError):
    """The library was misconfigured before any recording happened."""
