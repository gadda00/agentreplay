"""agentreplay — deterministic replay & counterfactual debugging for AI agents.

Public API:
    from agentreplay import Cassette, Recorder, Replayer, RecordingClient
    from agentreplay.frameworks import wrap_openai, wrap_anthropic

The library captures every non-deterministic input to an agent run (LLM
completions, tool/HTTP responses, clock, RNG) and guarantees that replaying
a recorded run reproduces the exact original trajectory — bit-for-bit,
with zero additional model calls.
"""
from agentreplay.cassette import Cassette, CassetteMeta, CassetteError
from agentreplay.constants import Mode, CallType
from agentreplay.diff import Diff, diff_structural
from agentreplay.errors import (
    AgentReplayError,
    DivergenceError,
    CassetteNotFoundError,
    MutationError,
)
from agentreplay.hashing import canonicalize, hash_call_site
from agentreplay.interceptors import (
    ClockInterceptor,
    RNGInterceptor,
    RecordingClock,
    RecordingRandom,
)
from agentreplay.recorder import Recorder
from agentreplay.replayer import Replayer
from agentreplay.session import Session
from agentreplay.storage import BlobStore, EventLog, MetaIndex
from agentreplay.types import (
    CallSiteID,
    Event,
    EventRecord,
    RequestPayload,
    ResponsePayload,
    StepID,
)
__all__ = [
    # Cassette + storage
    "Cassette",
    "CassetteMeta",
    "CassetteError",
    "BlobStore",
    "EventLog",
    "MetaIndex",
    # Hashing
    "canonicalize",
    "hash_call_site",
    # Modes & types
    "Mode",
    "CallType",
    "CallSiteID",
    "Event",
    "EventRecord",
    "RequestPayload",
    "ResponsePayload",
    "StepID",
    # Recorder / Replayer / Session
    "Recorder",
    "Replayer",
    "Session",
    # Interceptors
    "ClockInterceptor",
    "RNGInterceptor",
    "RecordingClock",
    "RecordingRandom",
    # Errors
    "AgentReplayError",
    "DivergenceError",
    "CassetteNotFoundError",
    "MutationError",
    # Diff
    "Diff",
    "diff_structural",
    # Version
    "__version__",
]

__version__ = "0.1.0"
