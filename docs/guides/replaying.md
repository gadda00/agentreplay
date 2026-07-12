# Replaying

The `Replayer` opens a cassette in REPLAY or HYBRID mode and exposes
the same wrapper surface as the `Recorder`, so the agent's code runs
unchanged.

## Modes

| Mode | Behaviour |
|---|---|
| `Mode.REPLAY` | Serve from cassette; never call real client. Raise `DivergenceError` on first mismatch. |
| `Mode.HYBRID` | Like REPLAY until first mismatch, then fall through to a live call. |
| `Mode.LIVE` | Pass-through (no recording, no replay). Use `Session.live()`. |
| `Mode.RECORD` | Use `Recorder` instead. |

## Pure replay

```python
from agentreplay import Replayer, Mode

with Replayer.open("cassettes/run-001", mode=Mode.REPLAY) as rep:
    client = rep.wrap_openai(openai_client)  # never called
    tool = rep.wrap_tool(my_search_fn, name="search")

    # Same agent code as recording
    result = my_agent.run(client=client, tools=[tool])
```

The agent runs **bit-exact** — every response matches the recording —
and the model is **never called**. This is the §7.1 guarantee.

## Hybrid replay

Hybrid mode is for "did my fix work?" — you've changed the agent's code
and want to see where the new trajectory diverges from the recording.

```python
with Replayer.open(
    "cassettes/run-001",
    mode=Mode.HYBRID,
    live_client=openai_client,   # used after divergence
) as rep:
    client = rep.wrap_openai()   # picks up live_client from rep
    ...
```

Everything **before** the divergence point replays bit-exact and free.
The first mismatched call falls through to `live_client`. Subsequent
calls are served from the cassette if their input still matches, or
fall through to live calls otherwise.

## Divergence detection

In REPLAY mode, the first time the agent asks for a call whose
canonicalized input does not match any recorded call-site ID, a
`DivergenceError` is raised:

```python
from agentreplay.errors import DivergenceError

try:
    with Replayer.open("cassettes/run-001", mode=Mode.REPLAY) as rep:
        client = rep.wrap_openai(openai_client)
        my_agent.run(client=client)  # diverges here
except DivergenceError as exc:
    print(f"diverged at {exc.step_id} ({exc.call_type})")
    print(f"recorded call_id: {exc.expected_call_id}")
    print(f"actual   call_id: {exc.actual_call_id}")
```

The CLI renders this as a structural diff:

```bash
agentreplay diff cassettes/baseline cassettes/mutated
```

```
Diff: cass-original → cass-patched
  3 steps | matching=2 diverged=1 extra_actual=0 extra_recorded=0
  ✗ first divergence at step 2 (llm, step_id='step:2:1')
    recorded call_id: 365f9fc8...
    actual   call_id: 8f6a2bd4...
    · messages[2].content
        recorded: 'deleted 42'
        actual  : 'PermissionError: not allowed'
```

## Why replay is faster than the original run

In REPLAY mode:

- The model is **never called** — every LLM response comes from the
  cassette, which is a local file read.
- Tool execution is **never invoked** — the recorded return value is
  returned directly.
- HTTP requests **never hit the network** — the recorded response is
  returned.

This makes replay typically **100×–1000× faster** than the original
run, and **100% cheaper** (zero inference spend).
