"""Example: streaming LLM responses with AgentReplay.

Demonstrates recording and replaying a streaming call (stream=True),
which is how most production agents consume LLM output — chunk by chunk
for real-time UX.

Run::

    python examples/streaming_agent.py record
    python examples/streaming_agent.py replay
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

from agentreplay import Recorder, Replayer
from agentreplay.constants import Mode

CASSETTE = Path(__file__).parent.parent / "cassettes" / "streaming-demo"


class StubStreamingLLM:
    """Stub LLM that returns chunk iterators when stream=True."""

    def __init__(self, chunk_lists: List[List[Dict[str, Any]]]) -> None:
        self.chunk_lists = list(chunk_lists)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Any:
        if params.get("stream"):
            chunks = self.chunk_lists.pop(0)
            return iter(list(chunks))
        return {"text": "non-streamed", "usage": {}}


def run_agent(client: Any) -> str:
    """Simulate a streaming agent loop."""
    stream = client.complete(
        messages=[{"role": "user", "content": "Tell me a joke"}],
        model="stub",
        stream=True,
    )
    full_text = ""
    for chunk in stream:
        if isinstance(chunk, dict):
            full_text += chunk.get("text", "")
        else:
            full_text += str(chunk)
    return full_text


def record() -> None:
    stub = StubStreamingLLM([
        [{"text": "Why did "}, {"text": "the chicken "}, {"text": "cross the road?"}],
    ])
    with Recorder.create(CASSETTE, framework="raw", agent_name="streaming-demo") as rec:
        client = rec.wrap_custom_client(stub)
        result = run_agent(client)
        print(f"Agent said: {result!r}")
    print(f"Recorded streaming cassette to {CASSETTE}")


def replay() -> None:
    with Replayer.open(CASSETTE, mode=Mode.REPLAY) as rep:
        # Empty stub — should never be called during pure replay
        client = rep.wrap_custom_client(StubStreamingLLM([]))
        result = run_agent(client)
        print(f"Agent said: {result!r}")
    print(f"Replayed streaming cassette from {CASSETTE} (zero model calls)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    if mode == "record":
        record()
    elif mode == "replay":
        replay()
    else:
        print(f"usage: {sys.argv[0]} [record|replay]")
        sys.exit(1)
