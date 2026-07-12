"""Built-in regression entry point.

This module ships with the ``agentreplay`` package so that
``agentreplay ci`` has a known-good agent entry point to invoke even
when the host project hasn't defined its own. It is what the bundled
GitHub Actions workflow (``.github/workflows/regression.yml``) calls
by default.

The agent here is deliberately tiny: it makes one LLM call and one
tool call. The point is **not** to exercise interesting agent behaviour
— it is to give CI something to replay against the sample cassettes
shipped in ``cassettes/`` so the workflow is green on a fresh clone.

To use your own agent instead, point ``--agent-entry`` at your
``module:callable`` and replace the sample cassettes.
"""
from __future__ import annotations

from typing import Any, Dict, List

from agentreplay.replayer import Replayer


class _StubLLM:
    """A no-op stand-in LLM client.

    In pure-replay mode the real client is never called, so this class
    only exists to give :meth:`Replayer.wrap_custom_client` something
    to wrap. If replay ever falls through to a live call (i.e. the
    cassette diverged), the ``RuntimeError`` makes the failure
    immediately visible instead of silently returning garbage.
    """

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        raise RuntimeError(
            "agentreplay.regression._StubLLM.complete was called — this means "
            "replay diverged from the cassette. Inspect the divergence report."
        )


def _search(query: str) -> str:
    """Sample tool — also never called in pure-replay mode."""
    raise RuntimeError(
        "agentreplay.regression._search was called — this means replay "
        "diverged from the cassette. Inspect the divergence report."
    )


def run_agent(replayer: Replayer) -> str:
    """Replay the canonical 2-step sample agent.

    The recorded sequence (see ``cassettes/sample-001/``) is:

        1. LLM call with ``[{"role": "user", "content": "What's the weather?"}]``
        2. Tool call with ``query="weather"``
        3. LLM call with the conversation so far

    Returns the final assistant text. In pure-replay mode this matches
    the recorded value bit-for-bit and never touches the stub client.
    """
    client = replayer.wrap_custom_client(_StubLLM())
    tool = replayer.wrap_tool(_search, name="search")

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
    return r2["text"]


def main() -> None:
    """CLI entry point: replay the bundled sample cassette.

    Useful for a quick ``python -m agentreplay.regression`` smoke test.
    """
    import json
    import sys
    from pathlib import Path

    from agentreplay.constants import Mode
    from agentreplay.replayer import Replayer

    cassette = Path(__file__).parent.parent.parent / "cassettes" / "sample-001"
    if not cassette.exists():
        # Try relative to CWD (when running from a checkout)
        cassette = Path("cassettes/sample-001")
    if not cassette.exists():
        print(f"sample cassette not found at {cassette}", file=sys.stderr)
        sys.exit(1)

    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        result = run_agent(rep)
    print(json.dumps({"status": "ok", "result": result}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
