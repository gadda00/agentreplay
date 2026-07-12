"""Example: OpenAI SDK integration.

Drop-in replacement for ``openai.OpenAI()`` — every
``client.chat.completions.create(...)`` call is captured.

Run::

    pip install agentreplay[openai]
    export OPENAI_API_KEY=...
    python examples/openai_agent.py record
    python examples/openai_agent.py replay      # no API key needed
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from agentreplay import Replayer, Recorder
from agentreplay.constants import Mode

CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai-demo"


def run_agent(client: object) -> str:
    """A trivial agent that asks the model one question."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    return resp.choices[0].message.content


def record() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("Please install with: pip install agentreplay[openai]")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY")
        sys.exit(1)
    real = OpenAI()
    with Recorder.create(CASSETTE, framework="openai", model="gpt-4o-mini") as rec:
        client = rec.wrap_openai(real)
        answer = run_agent(client)
        print(f"Agent said: {answer!r}")
    print(f"Recorded to {CASSETTE}")


def replay() -> None:
    # No API key needed — pure replay makes zero model calls.
    with Replayer.open(CASSETTE, mode=Mode.REPLAY) as rep:
        # Pass a dummy real client; it will never be called.
        client = rep.wrap_openai(object())
        answer = run_agent(client)
        print(f"Agent said: {answer!r}")
    print(f"Replayed from {CASSETTE} (zero model calls)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    if mode == "record":
        record()
    elif mode == "replay":
        replay()
    else:
        print(f"usage: {sys.argv[0]} [record|replay]")
        sys.exit(1)
