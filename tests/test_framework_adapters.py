"""Tests for the CrewAI and AutoGen framework adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.constants import Mode
from agentreplay.frameworks.crewai import (
    _CrewAIShim,
    restore_crewai_llm,
    wrap_crewai_llm,
)
from agentreplay.frameworks.autogen import (
    _AutoGenV4Shim,
    restore_autogen_v4_agent,
    wrap_autogen_v4_agent,
)


# ---------------------------------------------------------------------- #
# CrewAI
# ---------------------------------------------------------------------- #
class _FakeCrewAILLM:
    """Mimics crewai.LLM enough to exercise the adapter."""

    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.model = "fake-crewai-model"
        self._original_call_ref = self.call

    def call(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


def test_crewai_wrap_records_calls(tmp_path: Path):
    """wrap_crewai_llm patches the LLM's call method and records each call."""
    cassette = tmp_path / "cass"
    llm = _FakeCrewAILLM(responses=["first reply", "second reply"])

    with Recorder.create(cassette, framework="crewai") as rec:
        wrap_crewai_llm(llm, rec)
        r1 = llm.call("hello")
        r2 = llm.call("again")

    assert r1 == "first reply"
    assert r2 == "second reply"

    c = Cassette.open(cassette, readonly=True)
    assert len(c.events) == 2
    # Each event's response should be the text the LLM returned.
    records = c.records()
    assert records[0].response["text"] == "first reply"
    assert records[1].response["text"] == "second reply"


def test_crewai_restore_removes_patch(tmp_path: Path):
    """restore_crewai_llm should restore the original call behavior."""
    llm = _FakeCrewAILLM(responses=["x", "y"])
    with Recorder.create(tmp_path / "c", framework="crewai") as rec:
        wrap_crewai_llm(llm, rec)
        # While wrapped, calling llm.call goes through the recording layer
        assert llm.call("test") == "x"
    restore_crewai_llm(llm)
    # After restore, calling llm.call should use the original (un-patched) method.
    # The original method consumes from the same responses list, so the next
    # call should return "y" — proving the patched function is no longer active.
    assert llm.call("test") == "y"
    # And no _agentreplay_original_call attribute should remain.
    assert not hasattr(llm, "_agentreplay_original_call")


def test_crewai_shim_routes_to_original_call():
    """The _CrewAIShim should call the LLM's original (un-patched) call."""
    llm = _FakeCrewAILLM(responses=["shim response"])
    shim = _CrewAIShim(llm)
    response = shim.complete(messages=[{"role": "user", "content": "hi"}])
    assert response["text"] == "shim response"


# ---------------------------------------------------------------------- #
# AutoGen v0.4+
# ---------------------------------------------------------------------- #
class _FakeAutoGenV4Client:
    """Mimics an AutoGen v0.4+ model client."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def create(self, *, messages: Any, **kwargs: Any) -> Dict[str, Any]:
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


class _FakeAutoGenV4Agent:
    """Mimics an AutoGen v0.4+ agent that holds a _model_client."""

    def __init__(self, client: Any) -> None:
        self._model_client = client


def test_autogen_v4_wrap_records_calls(tmp_path: Path):
    """wrap_autogen_v4_agent patches the client's create method."""
    cassette = tmp_path / "cass"
    inner = _FakeAutoGenV4Client(responses=[{"text": "v4 reply", "usage": {}}])
    agent = _FakeAutoGenV4Agent(inner)

    with Recorder.create(cassette, framework="autogen") as rec:
        wrap_autogen_v4_agent(agent, rec)
        result = agent._model_client.create(messages=[{"role": "user", "content": "hi"}])

    assert result["text"] == "v4 reply"
    c = Cassette.open(cassette, readonly=True)
    assert len(c.events) == 1


def test_autogen_v4_restore_removes_patch(tmp_path: Path):
    inner = _FakeAutoGenV4Client(responses=[{"text": "x"}, {"text": "y"}])
    agent = _FakeAutoGenV4Agent(inner)
    with Recorder.create(tmp_path / "c", framework="autogen") as rec:
        wrap_autogen_v4_agent(agent, rec)
        # While wrapped, the call goes through the recording layer.
        r1 = agent._model_client.create(messages=[{"role": "user", "content": "q"}])
        assert r1["text"] == "x"
    restore_autogen_v4_agent(agent)
    # After restore, the original method should be active again.
    r2 = agent._model_client.create(messages=[{"role": "user", "content": "q"}])
    assert r2["text"] == "y"
    assert not hasattr(inner, "_agentreplay_original_create")


def test_autogen_v4_shim_routes_to_original():
    inner = _FakeAutoGenV4Client(responses=[{"text": "shim"}])
    shim = _AutoGenV4Shim(inner)
    response = shim.complete(messages=[{"role": "user", "content": "hi"}])
    assert response["text"] == "shim"


# ---------------------------------------------------------------------- #
# Framework registry / lazy imports
# ---------------------------------------------------------------------- #
def test_frameworks_lazy_import_crewai():
    """The lazy loader in __init__.py should expose wrap_crewai_llm."""
    from agentreplay.frameworks import wrap_crewai_llm as fn
    assert callable(fn)


def test_frameworks_lazy_import_autogen():
    """The lazy loader should expose both autogen v0.2 and v0.4 wrappers."""
    from agentreplay.frameworks import wrap_autogen_client as fn1
    from agentreplay.frameworks import wrap_autogen_v4_agent as fn2
    assert callable(fn1)
    assert callable(fn2)
