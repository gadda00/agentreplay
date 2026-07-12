"""Tests for the counterfactual mutation engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.constants import Mode
from agentreplay.errors import MutationError
from agentreplay.mutate import apply_patch_set, mutate_and_replay, mutate_response


class StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.live_calls = 0

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.live_calls += 1
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


def search_tool(query: str) -> str:
    return f"result-for-{query}"


def _record_baseline(tmp_path: Path) -> Path:
    """Record a simple 2-step agent: llm → tool → llm."""
    cassette_path = tmp_path / "baseline"
    stub = StubLLM([
        {"text": "step-1-llm", "usage": {}},
        {"text": "step-2-llm", "usage": {}},
    ])
    with Recorder.create(cassette_path, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        tool = rec.wrap_tool(search_tool, name="search")
        client.complete(messages=[{"role": "user", "content": "q1"}], model="stub")
        tool(query="weather")
        client.complete(messages=[{"role": "user", "content": "q2"}], model="stub")
    return cassette_path


def test_mutate_response_replaces_recorded_value(tmp_path: Path):
    baseline = _record_baseline(tmp_path)
    # The tool call is at seq 1.
    forked = mutate_response(
        baseline,
        seq=1,
        new_response={"value": "PERMISSION-DENIED", "error": None},
        target_root=tmp_path / "mutated",
    )
    records = forked.records()
    assert records[1].response == {"value": "PERMISSION-DENIED", "error": None}
    # The mutation tag should be applied.
    assert "mutated" in forked.meta.tags
    assert forked.meta.extra["mutated_from"] is not None


def test_mutate_response_by_step_id(tmp_path: Path):
    baseline = _record_baseline(tmp_path)
    forked = mutate_response(
        baseline,
        step_id="step:1:tool:search:0",
        new_response={"value": "patched", "error": None},
        target_root=tmp_path / "mut-by-step",
    )
    records = forked.records()
    assert any(r.response == {"value": "patched", "error": None} for r in records)


def test_mutate_response_ambiguous_step_id_raises(tmp_path: Path):
    baseline = _record_baseline(tmp_path)
    # step:0:llm:0 doesn't exist but step_id matching is on the full step
    # string. Use a step_id that matches nothing.
    with pytest.raises(MutationError):
        mutate_response(
            baseline,
            step_id="nonexistent-step",
            new_response={"value": "x", "error": None},
            target_root=tmp_path / "mut-err",
        )


def test_mutate_response_no_target_raises(tmp_path: Path):
    baseline = _record_baseline(tmp_path)
    with pytest.raises(MutationError):
        mutate_response(baseline, new_response={"value": "x", "error": None})


def test_mutate_response_preserves_request_hash(tmp_path: Path):
    """The mutated cassette must keep the same request hash so the
    call-site ID stays matchable for the upstream trajectory."""
    baseline = _record_baseline(tmp_path)
    base = Cassette.open(baseline, readonly=True)
    base_events = base.events.all()

    forked = mutate_response(
        baseline,
        seq=1,
        new_response={"value": "patched", "error": None},
        target_root=tmp_path / "mut",
    )
    forked_events = forked.events.all()

    assert base_events[1].request_hash == forked_events[1].request_hash
    assert base_events[1].response_hash != forked_events[1].response_hash
    assert base_events[1].call_id == forked_events[1].call_id


def test_mutate_and_replay_serves_mutated_value(tmp_path: Path):
    """Hybrid replay against a mutated cassette must serve the mutated
    response at the mutated step, then fall through to live calls for
    anything downstream that diverges."""
    baseline = _record_baseline(tmp_path)

    # The third step (seq 2, llm) will diverge because... well, in this
    # test we don't change the request, so it should still match. But
    # the second step (seq 1, tool) was mutated, so the agent code can
    # observe the patched value.
    live = StubLLM([])  # should not be needed if all calls match

    def agent_run(rep: Replayer) -> Any:
        client = rep.wrap_custom_client(live)
        tool = rep.wrap_tool(search_tool, name="search")
        r1 = client.complete(messages=[{"role": "user", "content": "q1"}], model="stub")
        tr = tool(query="weather")
        r2 = client.complete(messages=[{"role": "user", "content": "q2"}], model="stub")
        return {"r1": r1, "tr": tr, "r2": r2}

    result = mutate_and_replay(
        baseline,
        agent_run=agent_run,
        seq=1,
        new_response={"value": "PATCHED-TOOL-RESULT", "error": None},
        target_root=tmp_path / "mut",
        live_client=live,
    )
    assert result["result"]["tr"] == "PATCHED-TOOL-RESULT"
    # Upstream and downstream calls were served from the (mutated) cassette.
    assert live.live_calls == 0


def test_apply_patch_set_applies_multiple_mutations(tmp_path: Path):
    baseline = _record_baseline(tmp_path)
    forked = apply_patch_set(
        baseline,
        patches=[
            {"seq": 0, "new_response": {"text": "patched-1", "usage": {}}},
            {"seq": 2, "new_response": {"text": "patched-2", "usage": {}}},
        ],
        target_root=tmp_path / "patched",
    )
    records = forked.records()
    assert records[0].response == {"text": "patched-1", "usage": {}}
    assert records[2].response == {"text": "patched-2", "usage": {}}
    assert "patched" in forked.meta.tags
    assert forked.meta.extra["num_patches"] == 2
