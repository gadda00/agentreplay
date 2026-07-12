# CI Regression

Every captured cassette is a candidate regression test. Once a fix is
confirmed, the passing cassette is pinned into a regression corpus that
CI replays on every subsequent pull request — in pure-replay mode, so
the entire corpus costs no additional inference spend to run.

## The `agentreplay ci` command

```bash
agentreplay ci cassettes/ \
    --agent-entry my_project.tests:run_agent \
    --json > regression-report.json
```

This:

1. Discovers every cassette under `cassettes/` (any directory with a
   `cassette.json` file).
2. Replays each through `my_project.tests:run_agent` in pure REPLAY mode.
3. Reports pass/fail per cassette. Exits 0 if all passed, 1 otherwise.

## The agent entry point

Your `run_agent` callable receives a `Replayer` and is expected to run
the agent's code using the replayer's `wrap_*` interceptors:

```python
# my_project/tests.py
from agentreplay import Replayer

def run_agent(replayer: Replayer) -> None:
    client = replayer.wrap_openai(openai_client)
    tool = replayer.wrap_tool(my_search_fn, name="search")

    # Run the agent — every call is served from the cassette
    my_agent.run(client=client, tools=[tool])
```

The package ships with a built-in entry point at
`agentreplay.regression:run_agent` that replays the sample cassette at
`cassettes/sample-001/`.

## GitHub Actions

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
      - run: agentreplay ci cassettes/ --agent-entry my_project.tests:run_agent
```

## Adding a regression cassette

```bash
# 1. Record a cassette (your agent, your entry point)
python -c "
from agentreplay import Recorder
with Recorder.create('cassettes/bug-123-fixed', framework='raw', tags=['regression']) as rec:
    client = rec.wrap_openai(openai_client)
    my_agent.run(client=client)
"

# 2. Verify it replays cleanly
agentreplay replay cassettes/bug-123-fixed --agent-entry my_project.tests:run_agent

# 3. Commit it
git add cassettes/bug-123-fixed
git commit -m "test: add regression cassette for bug-123"
```

The next PR will automatically replay this cassette alongside every
other one in `cassettes/`.

## Filtering

```bash
# Only replay cassettes tagged "regression"
agentreplay ci cassettes/ --agent-entry my_project:run --tag regression

# Only replay failing cassettes (re-verify a fix)
agentreplay ci cassettes/ --agent-entry my_project:run --outcome fail

# Stop on first failure (faster feedback)
agentreplay ci cassettes/ --agent-entry my_project:run --stop-on-first-failure
```

## Why this is free

Pure replay makes **zero model calls** — every response comes from the
cassette, which is a local file read. An arbitrarily large corpus costs
the same as an empty one in inference spend. This is the direct
mechanism from §5.7 of the product proposal for turning captured
failures into a permanent, zero-cost regression suite.
