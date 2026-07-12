# Contributing to AgentReplay

Thanks for your interest in improving AgentReplay! This document covers
the basics. For architecture details, see
[architecture.md](architecture.md).

## Development setup

```bash
git clone https://github.com/gadda00/agentreplay.git
cd agentreplay
python -m pip install -e .[dev]   # editable install + test/lint deps
pytest                             # run the test suite (should be 75/75 green)
```

## The one rule

> The agent's own code should never know whether it is being recorded,
> and should never know whether a call it makes is live or replayed.

Every change must preserve this. If you find yourself adding mode
checks to agent code (as opposed to interceptor code), you're probably
doing it wrong — the interceptors should absorb all the mode logic.

## Test strategy

The end-to-end tests in `tests/test_record_replay.py` are the
load-bearing ones — they verify the *core product guarantee* (§7.1 of
the proposal): a pure replay must reproduce the original recording
bit-exact, with zero model calls.

Every new interceptor behavior should add a test of this shape:

```python
def test_my_new_thing_replays_bit_exact(tmp_path):
    cassette_path = tmp_path / "cass"
    stub = StubLLM([{"text": "expected", "usage": {}}])
    with Recorder.create(cassette_path, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        # ... exercise the new behavior ...

    # Replay with an empty stub — any live call would raise.
    fresh = StubLLM([])
    with Replayer.open(cassette_path, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(fresh)
        # ... exercise the same code path ...
    assert fresh.live_calls == 0
```

## Adding a new framework adapter

1. Create `src/agentreplay/frameworks/<framework>.py`.
2. Expose at least one `wrap_*` function that takes the framework's
   client object and a `Session` and returns a wrapped client.
3. If the framework has a notion of a "step" (node name, task ID),
   wire it into the session's `enter_step()` so call-site IDs are
   per-step.
4. Add an example in `examples/`.
5. Add a test that exercises record → replay through the adapter.

## Adding a new interceptor

1. Subclass `_BaseCallInterceptor` (or follow its pattern).
2. Implement the call method: compute `call_id`, look it up in the
   cassette in REPLAY/HYBRID mode, raise `DivergenceError` on mismatch
   in REPLAY mode, fall through to a real call otherwise.
3. Add the interceptor to `agentreplay/interceptors/__init__.py`.
4. Add a test in `tests/test_record_replay.py` that exercises the
   new interceptor's record → replay cycle.
5. If the interceptor captures a new kind of non-determinism, add an
   entry to the "How call-site IDs work" section of the README.

## Commit & PR conventions

- Branch from `main`, rebase before merging.
- Commit message format: `<type>: <subject>` (e.g. `feat: add
  AsyncOpenAI support`, `fix: handle None tool_calls in canonicalize`).
- Keep PRs focused — one feature or fix per PR.
- Tests must pass: `pytest`.
- New public API must be added to `__all__` in `__init__.py` and
  documented in the README.

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
