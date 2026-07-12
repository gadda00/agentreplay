# Recording

The recording layer captures every non-deterministic input to an agent
run: LLM completions, tool/HTTP responses, clock reads, random draws.
This page covers the practical details.

## The Recorder

The `Recorder` owns a cassette in RECORD mode and exposes wrapped
interceptors:

```python
from agentreplay import Recorder

with Recorder.create(
    "cassettes/run-001",
    framework="langgraph",
    agent_name="my-agent",
    task_id="swe-bench:214",
    model="claude-opus-4.6",
    tags=["regression", "langgraph"],
) as rec:
    client = rec.wrap_anthropic(anthropic_client)
    ...
```

### Wrappers

| Method | Wraps |
|---|---|
| `rec.wrap_openai(client)` | OpenAI SDK client (`client.chat.completions.create`) |
| `rec.wrap_anthropic(client)` | Anthropic SDK client (`client.messages.create`) |
| `rec.wrap_custom_client(client)` | Any object with a `.complete()` method |
| `rec.wrap_http(client, dialect="httpx")` | `httpx.Client` or `requests.Session` |
| `rec.wrap_tool(func, name="...")` | A single tool callable |
| `rec.clock` | `time.time()` / `datetime.now()` |
| `rec.random` | `random.Random`-compatible RNG |

### Step IDs

Every intercepted call is assigned a **call-site ID**: a SHA-256 hex
digest of `(step_id, canonicalized_input)`. By default the step ID is
a monotonic counter (`step:0`, `step:1`, ...).

Framework adapters call `rec.enter_step(name)` to set a more meaningful
step ID — e.g. `langgraph:router`, `langgraph:synthesizer`. This makes
the cassette easier to inspect and lets the divergence detector pinpoint
*which node* diverged.

```python
with Recorder.create("cassettes/run-001", framework="raw") as rec:
    client = rec.wrap_custom_client(stub)

    rec.enter_step("plan")
    r1 = client.complete(messages=[...], model="stub")

    rec.enter_step("execute")
    r2 = client.complete(messages=[...], model="stub")
```

## The cassette format

A cassette is a directory on disk:

```
cassettes/run-001/
  cassette.json     # metadata header
  events.jsonl      # append-only event log, one row per call
  blobs/            # content-addressed blob store
    ab/<sha256>     # sharded by first 2 hex chars
    cd/<sha256>
    ...
```

- `events.jsonl` holds the small, indexed event rows (call-site ID,
  timestamps, blob references).
- `blobs/` holds the heavy payloads (request/response bodies), deduplicated
  by SHA-256. A system prompt recorded once is referenced, not re-stored,
  on every subsequent call.
- `cassette.json` is the metadata header (agent name, task ID, git
  commit, model, pass/fail outcome).

See [cassette format](../cassette-format.md) for the full schema.

## Canonicalization

Two semantically identical requests must hash to the same call-site ID.
AgentReplay canonicalizes inputs before hashing:

- Dict keys are sorted.
- Non-deterministic keys are stripped (`request_id`, `id`, `created`,
  `system_fingerprint`, `x_request_id`, `seed`, `user-agent`).
- UUID-shaped strings are redacted to `<uuid>`.
- ISO-8601 timestamps in strings are redacted to `<iso8601>`.

This is the same pattern used by HTTP-mocking libraries (VCR, betamax)
and deterministic-replay debuggers (Mozilla `rr`). Without it, every
replay would diverge on the first call because the SDK generates a fresh
`request_id` per request.

## What gets recorded

| Call type | What's captured |
|---|---|
| LLM | Full request (messages, tools, params) + full response (text, tool_calls, usage, finish_reason) |
| Tool | Function name + args + return value (or exception) |
| HTTP | Method, URL, headers, body + status, headers, body |
| Clock | `time.time()` / `monotonic()` / `datetime.now()` return value |
| RNG | `random()` / `randint()` / `choice()` / `shuffle()` return value |

## What's NOT recorded

- **In-process state**: the agent's internal variables, data structures,
  etc. Only boundary calls are captured.
- **Uninstrumented side channels**: if a library hits the network without
  going through a wrapped transport, that call is not captured. See
  [Risks](../architecture.md#risks-and-limitations).

## Lifecycle

The `Recorder` is a context manager. On exit it:

1. Sets the cassette's `outcome` (pass/fail) based on whether an
   exception was raised, unless you've already set one explicitly.
2. Computes the total `duration_ms`.
3. Writes the final `cassette.json` metadata header.

```python
with Recorder.create("cassettes/run-001") as rec:
    ...  # agent code
# cassette.json is written here, even if an exception was raised
```
