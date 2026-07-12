# Architecture

This document describes the internal architecture of AgentReplay for
contributors. For usage, see the [README](https://github.com/gadda00/agentreplay#readme).

## Design principle

The single rule that shapes every design decision (§5.1 of the product
proposal):

> The agent's own code should never know whether it is being recorded,
> and should never know whether a call it makes is live or replayed.
> All non-determinism is pushed to the boundary of the system and
> intercepted there, never inside the agent's own logic.

This is what lets the same agent code run unmodified in four modes:
LIVE, RECORD, REPLAY, and HYBRID.

## Layered architecture

```
        ┌──────────────────────────────────────────┐
        │   CLI (cli.py) + auto-init (auto.py)     │  user-facing
        ├──────────────────────────────────────────┤
        │   Session (session.py)                    │  uniform API
        │   ┌───────────────┐ ┌─────────────────┐  │
        │   │ Recorder      │ │ Replayer        │  │  orchestrators
        │   │ (recorder.py) │ │ (replayer.py)   │  │
        │   └───────┬───────┘ └────────┬────────┘  │
        ├───────────┼──────────────────┼───────────┤
        │   ┌───────▼──────────────────▼────────┐  │
        │   │     Framework adapters             │  │  framework glue
        │   │  (openai_sdk, anthropic_sdk,       │  │
        │   │   langgraph, raw)                  │  │
        │   └────────────────┬───────────────────┘  │
        ├─────────────────────┼─────────────────────┤
        │   ┌─────────────────▼──────────────────┐  │
        │   │     Interceptors                   │  │  boundary
        │   │  RecordingClient (LLM)             │  │
        │   │  RecordingTool / RecordingHTTP     │  │
        │   │  RecordingClock / RecordingRandom  │  │
        │   └─────────────────┬──────────────────┘  │
        ├──────────────────────┼────────────────────┤
        │   ┌──────────────────▼─────────────────┐  │
        │   │     Cassette (cassette.py)         │  │  central abstraction
        │   └─────┬──────────┬──────────┬────────┘  │
        ├────────┼──────────┼──────────┼───────────┤
        │   ┌────▼────┐ ┌───▼─────┐ ┌──▼─────────┐  │
        │   │EventLog │ │BlobStore│ │MetaIndex   │  │  storage primitives
        │   │(JSONL)  │ │(content │ │(SQLite,    │  │
        │   │         │ │-addressed│ │optional)  │  │
        │   └─────────┘ └─────────┘ └────────────┘  │
        ├──────────────────────────────────────────┤
        │   hashing.py + types.py + errors.py      │  foundation
        └──────────────────────────────────────────┘
```

## The call-site ID

The single most important concept. Every intercepted call is assigned a
SHA-256 hex digest of `(step_id, canonicalized_input)`. This ID is the
join key between "what the agent is asking for right now" and "what was
recorded for that exact ask".

Canonicalization is the trick that makes this work in practice:

| Input shape | Canonicalization rule |
|---|---|
| `dict` | sort by key; strip non-deterministic keys (`request_id`, `id`, `created`, `system_fingerprint`, `x_request_id`, `seed`, `user-agent`) |
| `str` | redact UUID-shaped values to `<uuid>`; redact ISO-8601 timestamps to `<iso8601>` |
| `list` / `tuple` | canonicalize element-wise |
| scalar | pass through |
| exotic (`Path`, `Decimal`, `datetime`) | `repr()` |

Two semantically identical requests must hash to the same call-site ID.
Without canonicalization, every replay would diverge on the first call
because the SDK generates a fresh `request_id` per request.

## The cassette format

A cassette is a directory on disk with this layout:

```
<cassette>/
  cassette.json          # metadata header
  events.jsonl           # append-only event log, one row per call
  blobs/                 # content-addressed blob store
    <sha256>
    <sha256>
    ...
  meta.db                # SQLite metadata index (optional, local-dev)
```

The split between `events.jsonl` (small, indexed, frequently read) and
`blobs/` (large, content-addressed, deduplicated) is deliberate:

- Reading the structure of a 50-step run is milliseconds — you only
  read the small event log rows.
- Heavy payloads (system prompts, tool schemas, repository state) are
  deduplicated for free — a system prompt recorded once is referenced
  by hash on every subsequent call, not re-stored.
- The cassette is plain JSON — no pickle, no opaque blobs. A cassette
  is self-describing and portable across machines.

## Interceptors

Each interceptor is a transparent wrapper that records or serves from
the cassette depending on the current `Mode`:

| Mode | Behaviour |
|---|---|
| `LIVE` | pass-through; no recording |
| `RECORD` | call real client; write request/response to cassette |
| `REPLAY` | look up call-site ID in cassette; return recorded response; raise `DivergenceError` on mismatch |
| `HYBRID` | like REPLAY until first mismatch, then fall through to live call |

The interceptors share a common pattern:

```python
class RecordingX:
    def call(self, *, request, step_id, **params):
        call_id = hash_call_site(step_id, request)
        if mode in (REPLAY, HYBRID):
            cached = cassette.lookup_call(call_id)
            if cached is not None:
                return cassette.resolve_response(cached)
            if mode == REPLAY:
                raise DivergenceError(step_id, call_id, request)
            # HYBRID fallthrough:
        response = real_client.call(request=request, **params)
        if mode == RECORD:
            cassette.write_event(step_id=step_id, call_id=call_id,
                                 request=request, response=response)
        return response
```

## Divergence detection

In REPLAY mode, the first time the agent asks for a call whose
canonicalized input does not match any recorded call-site ID, a
`DivergenceError` is raised. This is the *divergence detector* from
§5.3 — the developer's signal that "your code changed something that
matters".

The CLI catches the `DivergenceError` and renders a structural diff:

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

## Counterfactual mutation

The mutation engine (`mutate.py`) works in three steps:

1. **Fork** the source cassette. The blob store is hardlinked where
   possible so a fork costs almost nothing on disk.
2. **Replace** the recorded response at the target step. The request
   hash is preserved, so the call-site ID stays matchable — the
   upstream trajectory replays bit-exact.
3. **Replay** the mutated cassette in REPLAY mode (to verify the
   upstream still matches) or HYBRID mode (to see where the new
   trajectory goes from the mutation point).

The key insight: in HYBRID mode, everything *before* the mutated step
replays bit-exact and free; everything *after* the mutated step
diverges on the first call whose input changed because of the mutation,
and falls through to a live call. This is what makes "what if Y had
been different?" a five-second experiment instead of a full re-run.

## CI regression corpus

The `ci` module discovers every cassette under a directory and replays
each through a configurable agent entry point. Because pure replay
makes zero model calls, an arbitrarily large corpus costs nothing in
inference spend to run on every PR.

The GitHub Actions workflow in `.github/workflows/regression.yml` is
the reference implementation. Adding a new regression test is:

```bash
agentreplay record cassettes/<name> -- python my_agent.py
git add cassettes/<name>
git commit
```

## Adding a new framework adapter

A framework adapter has two jobs:

1. **Wrap the framework's LLM client** so every model call is captured.
   Usually a one-liner: `session.wrap_openai(client)` or
   `session.wrap_anthropic(client)`. For custom client shapes, use
   `session.wrap_custom_client(client)`.

2. **Provide step IDs** so calls from different nodes/tasks have
   different call-site IDs. The framework adapter wires its own notion
   of a step (LangGraph node name, CrewAI task ID, ...) into the
   recorder's `enter_step()` method.

For a framework that exposes hooks around node execution (LangGraph's
`StateGraph.nodes`, CrewAI's task lifecycle), the adapter can patch
those hooks. For frameworks without such hooks (raw OpenAI/Anthropic
loops), the user calls `rec.enter_step("my-node-name")` manually.

## Testing strategy

The test suite (`tests/`) is organized by layer:

- `test_hashing.py` — canonicalization & call-site ID properties
- `test_storage.py` — blob store, event log, meta index
- `test_cassette.py` — the central `Cassette` class
- `test_record_replay.py` — **end-to-end reproduction fidelity tests** (most important)
- `test_mutate.py` — counterfactual mutation
- `test_diff.py` — structural diff
- `test_ci.py` — regression corpus runner
- `test_cli.py` — CLI commands

The end-to-end tests are the load-bearing ones: they verify the *core
product guarantee* — a pure replay reproduces the original recording
bit-exact, with zero model calls. Every new interceptor behavior must
add a corresponding test of this shape.

## Risks and limitations

Honest scope limits, not flaws (§8 of the product proposal):

- **Uninstrumented side channels.** If the target agent reads entropy or
  state through a path the interceptors do not cover (a library that hits
  the network directly, a side-effecting tool that writes to a live
  external system), pure replay will not be bit-exact. Mitigation: pair
  AgentReplay with Docker-level sandboxing for tool execution, and treat
  "verified fully intercepted" as an explicit, testable property of a
  cassette.

- **Storage growth.** Verbatim capture of every call can grow quickly.
  Mitigation: content-addressed deduplication removes most of the
  redundancy; a retention policy (keep failing cassettes indefinitely,
  sample passing cassettes) bounds growth further.

- **AgentReplay does not, by itself, make an agent more reliable.** It
  makes failures reproducible and cheap to study, which is a
  prerequisite for fixing reliability rather than a fix in itself.
