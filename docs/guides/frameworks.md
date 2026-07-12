# Framework Adapters

Framework adapters wrap a popular agent framework's LLM client / tool
entry points so a team using that framework can add AgentReplay with a
single two-line initialization.

## Raw agent loops

If you're not using a framework — just calling the OpenAI or Anthropic
client directly inside your own loop — use the raw adapter:

```python
from agentreplay import Recorder
from agentreplay.frameworks import wrap_raw_client
from openai import OpenAI

with Recorder.create("cassettes/run-001", framework="raw") as rec:
    client = wrap_raw_client(OpenAI(), rec, dialect="openai")
    # ... your agent loop, using `client` exactly as you would the raw client
```

## OpenAI SDK

Drop-in replacement for `openai.OpenAI()`:

```python
from agentreplay import Recorder

with Recorder.create("cassettes/run-001", framework="openai") as rec:
    client = rec.wrap_openai(OpenAI())
    # client.chat.completions.create(...) — captured automatically
```

## Anthropic SDK

Drop-in replacement for `anthropic.Anthropic()`:

```python
from agentreplay import Recorder

with Recorder.create("cassettes/run-001", framework="anthropic") as rec:
    client = rec.wrap_anthropic(Anthropic())
    # client.messages.create(...) — captured automatically
```

## LangGraph (first-class)

LangGraph is the first-class integration target from §5.5. The
`bind_graph` context manager patches each node's runnable so node names
become step IDs automatically.

```python
from langgraph.graph import StateGraph, START, END
from agentreplay import Recorder
from agentreplay.frameworks.langgraph import bind_graph, wrap_llm

with Recorder.create("cassettes/run-001", framework="langgraph") as rec:
    client = wrap_llm(openai_client, rec, dialect="openai")

    g = StateGraph(MyState)
    g.add_node("router", router_fn)
    g.add_node("synthesizer", synth_fn)
    g.add_edge(START, "router")
    g.add_edge("router", "synthesizer")
    g.add_edge("synthesizer", END)

    # bind_graph BEFORE compile()
    with bind_graph(rec, g):
        compiled = g.compile()
        result = compiled.invoke(initial_state)
```

The cassette's step IDs will include the node names
(`langgraph:router`, `langgraph:synthesizer`), making divergence
detection much more useful — you can see *which node* diverged.

See [`examples/langgraph_real.py`](https://github.com/gadda00/agentreplay/blob/main/examples/langgraph_real.py)
for a runnable end-to-end example.

## Custom clients

If your client doesn't match the OpenAI or Anthropic shape, wrap it as
a "custom" client — it just needs a `.complete()` method:

```python
class MyClient:
    def complete(self, *, messages, tools=None, **params):
        # ... your implementation ...
        return {"text": "...", "usage": {}}

with Recorder.create("cassettes/run-001") as rec:
    client = rec.wrap_custom_client(MyClient())
```

## Adding a new framework adapter

A framework adapter has two jobs:

1. **Wrap the framework's LLM client** so every model call is captured.
   Usually a one-liner: `session.wrap_openai(client)` or
   `session.wrap_anthropic(client)`.

2. **Provide step IDs** so calls from different nodes/tasks have
   different call-site IDs. The framework adapter wires its own notion
   of a step (LangGraph node name, CrewAI task ID, ...) into the
   recorder's `enter_step()` method.

See [`CONTRIBUTING.md`](https://github.com/gadda00/agentreplay/blob/main/CONTRIBUTING.md)
for details on adding a new adapter.
