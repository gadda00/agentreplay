"""Typed primitives shared across the agentreplay package.

These are intentionally small ``TypedDict`` / ``dataclass`` objects so the
on-disk cassette format is plain JSON (no pickle, no opaque blobs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - Python <3.8 fallback
    from typing_extensions import TypedDict  # type: ignore


# A call-site ID is the SHA-256 hex digest of (step_id, canonicalized input).
# It is the join key between "what the agent is asking for right now" and
# "what was recorded for that exact ask" — see §5.2 of the product proposal.
CallSiteID = str

# A step ID names a single node/step inside an agent run.
# Examples: "langgraph:0:router", "step:7", "agent:planner:turn:3".
StepID = str


class RequestPayload(TypedDict, total=False):
    """Canonical request payload for an intercepted call."""

    # LLM
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    model: str
    temperature: float
    max_tokens: int
    # Tool / HTTP
    method: str
    url: str
    headers: Dict[str, str]
    body: Any
    args: List[Any]
    kwargs: Dict[str, Any]
    # Generic
    params: Dict[str, Any]


class ResponsePayload(TypedDict, total=False):
    """Canonical response payload for an intercepted call."""

    # LLM
    text: str
    tool_calls: List[Dict[str, Any]]
    usage: Dict[str, int]
    finish_reason: str
    raw: Dict[str, Any]
    # Tool / HTTP
    status: int
    body: Any
    # Generic
    value: Any
    error: Optional[str]


@dataclass
class Event:
    """A single recorded interaction.

    Stored as one row in the event log; the heavy payloads (request/response
    bodies) live in the content-addressed blob store and are referenced here
    by SHA-256 hash.
    """

    seq: int                       # 0-indexed position within the cassette
    step_id: StepID
    call_type: str                 # CallType value
    call_id: CallSiteID            # hash of (step_id, canonicalized input)
    request_hash: str              # blob hash of canonicalized request
    response_hash: str             # blob hash of recorded response
    started_at: float              # epoch seconds (real, captured)
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "step_id": self.step_id,
            "call_type": self.call_type,
            "call_id": self.call_id,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            seq=int(d["seq"]),
            step_id=str(d["step_id"]),
            call_type=str(d["call_type"]),
            call_id=str(d["call_id"]),
            request_hash=str(d["request_hash"]),
            response_hash=str(d["response_hash"]),
            started_at=float(d["started_at"]),
            duration_ms=float(d.get("duration_ms", 0.0)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class EventRecord:
    """Convenience bundle: an :class:`Event` plus its resolved payloads."""

    event: Event
    request: RequestPayload
    response: ResponsePayload


# JSON-serialisable alias used by the metadata index and CLI.
JSONValue = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]
