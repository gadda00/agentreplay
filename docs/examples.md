# Examples

All examples are in the [`examples/`](https://github.com/gadda00/agentreplay/tree/main/examples)
directory of the repo and are runnable without an API key (they use
stub LLM clients).

## Raw agent loop

[`examples/raw_agent.py`](https://github.com/gadda00/agentreplay/blob/main/examples/raw_agent.py)
— the simplest possible use of AgentReplay. A hand-rolled loop that
calls the model and a tool, with no framework.

```bash
python examples/raw_agent.py record    # record a run
python examples/raw_agent.py replay    # replay (zero model calls)
```

## Counterfactual mutation

[`examples/counterfactual.py`](https://github.com/gadda00/agentreplay/blob/main/examples/counterfactual.py)
— the incident-review workflow from §5.4. Records a baseline run where
a dangerous tool returns success, then patches the tool's response to
a permission-denied error and replays to see whether the agent would
still have taken the harmful action.

```bash
python examples/counterfactual.py record    # record baseline
python examples/counterfactual.py replay    # replay baseline
python examples/counterfactual.py mutate    # apply counterfactual + replay
```

## LangGraph integration (real)

[`examples/langgraph_real.py`](https://github.com/gadda00/agentreplay/blob/main/examples/langgraph_real.py)
— builds an actual `langgraph.graph.StateGraph` with two nodes
(`router` → `synthesizer`), records a run, and replays it bit-exact.
The `bind_graph` context manager patches each node's runnable so node
names become step IDs.

```bash
pip install agentreplay[langgraph]
python examples/langgraph_real.py record
python examples/langgraph_real.py replay
```

## OpenAI SDK

[`examples/openai_agent.py`](https://github.com/gadda00/agentreplay/blob/main/examples/openai_agent.py)
— drop-in replacement for `openai.OpenAI()`. Every
`client.chat.completions.create(...)` call is captured.

```bash
pip install agentreplay[openai]
export OPENAI_API_KEY=...
python examples/openai_agent.py record
python examples/openai_agent.py replay      # no API key needed
```

## Running the sample cassette

The repo ships with a sample cassette at `cassettes/sample-001/`. The
package's built-in regression entry point replays it:

```bash
agentreplay ci cassettes/ --agent-entry agentreplay.regression:run_agent
# AgentReplay regression: 1/1 passed
#   ✓ cass-bf9c3190d4c2 (0.5 ms)
```
