# Counterfactual Mutation

Counterfactual mutation answers "what if Y had been different?" — edit
one recorded step and replay forward to see how the trajectory changes.

## The mental model

A cassette is a sequence of recorded calls. Each call has:

- A **request** (what the agent asked for)
- A **response** (what it got back)
- A **call-site ID** (hash of the canonicalized request)

When you mutate a cassette, you:

1. **Fork** the cassette (cheap — blobs are hardlinked).
2. **Replace** the recorded response at one step. The request hash is
   preserved, so the call-site ID stays matchable.
3. **Replay** the mutated cassette. Everything before the mutation
   replays bit-exact and free. Everything after the mutation diverges
   — in HYBRID mode, the divergent calls fall through to live calls so
   you can see where the new trajectory goes.

## Single mutation

```python
from agentreplay.mutate import mutate_response

mutated = mutate_response(
    "cassettes/run-001",
    seq=3,                                    # the step to patch
    new_response={"value": None, "error": "PermissionError"},
    target_root="cassettes/run-001-mutated",
)
```

You can target the step by `seq` (positional index), `step_id`, or
`call_id`.

## Mutation + hybrid replay

```python
from agentreplay.mutate import mutate_and_replay

result = mutate_and_replay(
    "cassettes/run-001",
    agent_run=lambda rep: my_agent.run(client=rep.wrap_openai()),
    seq=3,
    new_response={"value": None, "error": "PermissionError"},
    live_client=openai_client,
)
# result["cassette"]    — the mutated cassette
# result["result"]      — whatever agent_run returned
# result["divergences"] — list of divergence points
```

## Multiple mutations

```python
from agentreplay.mutate import apply_patch_set

forked = apply_patch_set(
    "cassettes/run-001",
    patches=[
        {"seq": 0, "new_response": {"text": "patched-1", "usage": {}}},
        {"seq": 2, "new_response": {"text": "patched-2", "usage": {}}},
    ],
    target_root="cassettes/run-001-patched",
)
```

## CLI

```bash
agentreplay mutate cassettes/run-001 \
    --seq 3 \
    --response '{"value": null, "error": "PermissionError"}' \
    --out cassettes/run-001-mutated
```

## Use cases

### Incident review

> "Would the agent still have deleted the production record if the
> permission check had returned DENIED instead of ALLOWED?"

Record the incident, mutate the permission-check tool's response to
`DENIED`, replay in HYBRID mode, observe whether the agent still calls
`delete_record`.

### Prompt sensitivity

> "Would the agent have taken the same action if the system prompt
> had been worded differently?"

This requires re-recording (the system prompt is part of the LLM
request, not the response). Use two `Recorder` runs with different
system prompts, then `agentreplay diff` the two cassettes.

### Tool replacement

> "Would a faster (but less accurate) search tool have changed the
> outcome?"

Mutate the search tool's recorded response to a shorter/different
result and replay in HYBRID mode.

## Limitations

- Mutation only changes **recorded responses**, not requests. If you
  need to change what the agent *asks*, re-record with different agent
  code.
- In pure REPLAY mode, a mutation that changes a response will cause
  downstream requests to diverge (because the agent's next message
  includes the mutated response). Use HYBRID mode with a live client
  to see the full new trajectory.
