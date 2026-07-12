#!/usr/bin/env python3
"""Record the bundled sample cassette.

This script captures the canonical 2-step sample run that
``agentreplay.regression:run_agent`` expects to replay against.
The resulting cassette is committed to the repo so CI has something
to replay on every PR.

Usage::

    python scripts/record_sample_cassette.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Make the package importable when running from a fresh checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentreplay import Recorder  # noqa: E402

CASSETTE = ROOT / "cassettes" / "sample-001"


class _StubLLM:
    """Records a deterministic 2-call sequence."""

    def __init__(self) -> None:
        self._responses = [
            {"text": "Let me search for that.", "usage": {"total_tokens": 12}},
            {"text": "Based on the search, it's sunny.", "usage": {"total_tokens": 18}},
        ]

    def complete(self, *, messages, tools=None, **params):
        if not self._responses:
            raise RuntimeError("exhausted")
        return self._responses.pop(0)


def _search(query: str) -> str:
    return f"<results for '{query}'>"


def main() -> None:
    if CASSETTE.exists():
        shutil.rmtree(CASSETTE)
    CASSETTE.parent.mkdir(parents=True, exist_ok=True)

    with Recorder.create(
        CASSETTE,
        framework="raw",
        agent_name="sample",
        model="stub",
        tags=["sample", "regression"],
    ) as rec:
        client = rec.wrap_custom_client(_StubLLM())
        tool = rec.wrap_tool(_search, name="search")

        r1 = client.complete(
            messages=[{"role": "user", "content": "What's the weather?"}],
            model="stub",
        )
        tool_result = tool(query="weather")
        r2 = client.complete(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": r1["text"]},
                {"role": "tool", "content": tool_result},
            ],
            model="stub",
        )
        print(f"recorded; final assistant reply: {r2['text']!r}")

    events_file = CASSETTE / "events.jsonl"
    print(f"cassette written to {CASSETTE}")
    print(f"  events: {len(events_file.read_text().splitlines())}")


if __name__ == "__main__":
    main()
