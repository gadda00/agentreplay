# AgentReplay

> Deterministic replay and counterfactual debugging for AI agent reliability.
> Bit-exact replay of agent runs with **zero model calls** — record once,
> replay forever, mutate freely.

AgentReplay is a recording and replay layer that sits between an agent
and every external, non-deterministic thing it talks to — the model API,
tools, the network, the clock. It guarantees one property that nothing
else in the current market guarantees end to end: **replaying a captured
agent run reproduces the exact original outcome, byte for byte, without
calling the model again.**

## Why?

Existing agent observability tools (LangSmith, Langfuse, AgentOps,
Laminar) give you visibility into *what happened* — but none of them
guarantee that *running the same scenario again produces the same
result*. Their "time travel" features are checkpoint-restart: they
resume live execution from a saved point, which immediately diverges
again because the model call itself is not pinned.

AgentReplay closes that gap.

## Two capabilities that follow naturally

| Capability | What it gives you |
|---|---|
| **Counterfactual mutation** | Edit one recorded step and replay forward — "would the agent still have done X if Y had been different?" becomes a 5-second, zero-cost experiment. |
| **Free regression tests** | Every captured production or benchmark failure becomes a permanent, pinned CI test that replays deterministically forever, at the one-time cost of the original capture. |

## Quick start

```bash
pip install agentreplay
```

```python
from agentreplay import Recorder, Replayer, Mode

# Record
with Recorder.create("cassettes/run-001", framework="raw") as rec:
    client = rec.wrap_openai(openai_client)
    # ... your agent loop, using `client` as you would the raw client ...

# Replay (zero model calls)
with Replayer.open("cassettes/run-001", mode=Mode.REPLAY) as rep:
    client = rep.wrap_openai(openai_client)  # never called
    # ... same agent code ...
```

## Key links

- [GitHub](https://github.com/gadda00/agentreplay)
- [PyPI](https://pypi.org/project/agentreplay/)
- [Quick Start](quick-start.md)
- [API Reference](api.md)
- [CLI Reference](cli.md)
- [Architecture](architecture.md)
