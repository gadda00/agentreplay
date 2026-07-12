"""Interceptors — the heart of the recording layer.

The interceptors sit at the framework boundary and capture every
non-deterministic input to an agent run:

    * :class:`RecordingClock`   — wraps ``time.time`` / ``datetime.now``
    * :class:`RecordingRandom`  — wraps ``random`` / ``secrets``
    * :class:`RecordingClient`  — wraps an LLM client (OpenAI / Anthropic)
    * :class:`RecordingHTTP`    — wraps ``httpx.Client`` / ``requests``
    * :class:`RecordingTool`    — wraps a single tool callable

Every interceptor follows the same contract: in RECORD mode it calls the
real thing and writes the request/response to the cassette; in REPLAY
mode it looks up the call-site ID in the cassette and returns the
recorded response without ever calling the real thing.

Crucially, the agent's own code never knows which mode it is in — this
is the property from §5.1 of the product proposal ("the agent's own
code should never know whether a call it makes is live or replayed").
"""
from agentreplay.interceptors.clock import (
    ClockInterceptor,
    RNGInterceptor,
    RecordingClock,
    RecordingRandom,
)
from agentreplay.interceptors.http import RecordingHTTP, RecordingTool
from agentreplay.interceptors.llm import RecordingClient

__all__ = [
    "ClockInterceptor",
    "RNGInterceptor",
    "RecordingClock",
    "RecordingRandom",
    "RecordingClient",
    "RecordingHTTP",
    "RecordingTool",
]
