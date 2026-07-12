"""Example: counterfactual mutation — "what if the tool had returned an error?"

This is the incident-review workflow from §5.4 of the product proposal.
We take a recorded cassette where the tool returned success, fork it,
patch the tool's response to a permission-denied error, and replay to
see whether the agent would still have taken the harmful action.

Run::

    # 1. Record the baseline run
    python examples/counterfactual.py record

    # 2. Replay the baseline (sanity check)
    python examples/counterfactual.py replay

    # 3. Apply a counterfactual: what if the tool had returned an error?
    python examples/counterfactual.py mutate
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.constants import Mode
from agentreplay.mutate import mutate_response

BASELINE = Path(__file__).parent.parent / "cassettes" / "cf-baseline"
MUTATED = Path(__file__).parent.parent / "cassettes" / "cf-mutated"


class StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


def delete_record(record_id: str) -> str:
    """A dangerous tool the agent can call."""
    return f"deleted {record_id}"


def run_agent(client: Any, tool: Any) -> str:
    """The agent asks the model, gets told to delete a record, deletes it."""
    r1 = client.complete(
        messages=[{"role": "user", "content": "delete record 42"}],
        model="stub",
    )
    # The agent decides (based on r1) to call the delete tool.
    tool_result = tool(record_id="42")
    r2 = client.complete(
        messages=[
            {"role": "user", "content": "delete record 42"},
            {"role": "assistant", "content": r1["text"]},
            {"role": "tool", "content": tool_result},
        ],
        model="stub",
    )
    return r2["text"]


def record() -> None:
    stub = StubLLM(
        responses=[
            {"text": "OK, calling delete_record(42).", "usage": {}},
            {"text": "Done — record deleted successfully.", "usage": {}},
        ]
    )
    with Recorder.create(BASELINE, framework="raw", agent_name="demo", model="stub") as rec:
        client = rec.wrap_custom_client(stub)
        tool = rec.wrap_tool(delete_record, name="delete_record")
        result = run_agent(client, tool)
        print(f"Agent said: {result!r}")
    print(f"Recorded baseline to {BASELINE}")


def replay() -> None:
    with Replayer.open(BASELINE, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(StubLLM([]))
        tool = rep.wrap_tool(delete_record, name="delete_record")
        result = run_agent(client, tool)
        print(f"Agent said: {result!r}")
    print(f"Replayed baseline (zero model calls)")


def mutate() -> None:
    """Counterfactual: what if the delete tool had returned an error?"""
    # Inspect the baseline to find the tool-call step.
    baseline = Cassette.open(BASELINE, readonly=True)
    records = baseline.records()
    print("Baseline events:")
    for r in records:
        print(f"  seq={r.event.seq} type={r.event.call_type} step={r.event.step_id!r}")

    # Patch the tool response (the one TOOL event) to return an error.
    tool_seq = next(
        r.event.seq for r in records if r.event.call_type == "tool"
    )
    print(f"\nMutating seq {tool_seq} (tool call) to return PERMISSION-DENIED...")

    forked = mutate_response(
        BASELINE,
        seq=tool_seq,
        new_response={"value": None, "error": "PermissionError: not allowed"},
        target_root=MUTATED,
    )
    print(f"Wrote mutated cassette to {MUTATED}")

    # Now replay the mutated cassette to see what the agent would have done.
    print("\nReplaying mutated cassette (pure-replay mode):")
    with Replayer.open(MUTATED, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(StubLLM([]))
        tool = rep.wrap_tool(delete_record, name="delete_record")
        try:
            result = run_agent(client, tool)
            print(f"  Agent said: {result!r}")
        except Exception as exc:
            print(f"  Divergence detected: {exc!r}")
            print()
            print("  → The mutation changed the tool's response from 'deleted 42'")
            print("    to a permission-denied error. Because the agent's next LLM")
            print("    call now includes that error in its messages, the canonicalized")
            print("    request no longer matches the recorded call-site ID — this is")
            print("    the *divergence point* from §5.3 of the product proposal.")
            print()
            print("  → In a real workflow, you would re-run in HYBRID mode with a live")
            print("    client to see where the new trajectory goes from here. The")
            print("    steps BEFORE the mutation replay bit-exact and free; only the")
            print("    steps AFTER the mutation fall through to a live call.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    if mode == "record":
        record()
    elif mode == "replay":
        replay()
    elif mode == "mutate":
        mutate()
    else:
        print(f"usage: {sys.argv[0]} [record|replay|mutate]")
        sys.exit(1)
