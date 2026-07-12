# Quick Start

This page walks through the three core workflows: **record**, **replay**,
and **mutate**.

## 1. Record a run

Wrap your LLM client and tools with a `Recorder`:

```python
from agentreplay import Recorder

with Recorder.create("cassettes/run-001", framework="raw") as rec:
    # Wrap your LLM client — pick the right dialect for your SDK
    client = rec.wrap_openai(openai_client)
    # client = rec.wrap_anthropic(anthropic_client)
    # client = rec.wrap_custom_client(my_custom_client)

    # Wrap your tools
    search = rec.wrap_tool(my_search_fn, name="search")

    # Wrap the clock and RNG if your agent uses them
    clock = rec.clock
    rng = rec.random

    # Run your agent — exactly as you would without AgentReplay
    result = my_agent.run(client=client, tools=[search], clock=clock)
```

That's it. Every LLM call, tool call, clock read, and random draw is
captured to a cassette at `cassettes/run-001/`.

## 2. Replay it (zero model calls)

```python
from agentreplay import Replayer, Mode

with Replayer.open("cassettes/run-001", mode=Mode.REPLAY) as rep:
    client = rep.wrap_openai(openai_client)  # never called
    search = rep.wrap_tool(my_search_fn, name="search")
    clock = rep.clock
    rng = rep.random

    # Same agent code — every external call is served from the cassette
    result = my_agent.run(client=client, tools=[search], clock=clock)
```

The agent runs **bit-exact** — every response matches the recording —
and the model is **never called**. This is the core product guarantee
(§7.1).

## 3. Counterfactual: "what if the tool had returned an error?"

```python
from agentreplay.mutate import mutate_response

mutate_response(
    "cassettes/run-001",
    seq=3,                                    # the step to patch
    new_response={"value": None, "error": "PermissionError"},
    target_root="cassettes/run-001-counterfactual",
)

# Replay the mutated cassette
with Replayer.open("cassettes/run-001-counterfactual", mode=Mode.REPLAY) as rep:
    ...
```

Everything **before** the mutated step replays bit-exact and free.
Everything **after** the mutated step diverges — in HYBRID mode, the
divergent calls fall through to a live model call so you can see where
the new trajectory goes.

## 4. CI regression suite

Every cassette committed to your repo becomes a permanent regression test:

```yaml
# .github/workflows/regression.yml
name: agentreplay-regression
on: [pull_request, push]
jobs:
  replay:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e .[dev]
      - run: agentreplay ci cassettes/ --agent-entry my_agent:run_agent
```

Because pure replay makes zero model calls, an arbitrarily large corpus
costs nothing in inference spend to run on every PR.

## Next steps

- [Examples](examples.md) — runnable end-to-end demos
- [Recording guide](guides/recording.md) — deep dive on the recording layer
- [Framework adapters](guides/frameworks.md) — LangGraph, OpenAI SDK, Anthropic SDK
- [API reference](api.md) — full API docs
- [CLI reference](cli.md) — `agentreplay` command-line tool
