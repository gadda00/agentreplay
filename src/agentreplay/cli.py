"""``agentreplay`` CLI.

Subcommands (per §5.7 of the product proposal):

    agentreplay record <cassette> -- python my_agent.py
    agentreplay replay <cassette>
    agentreplay diff   <cassette_a> <cassette_b>
    agentreplay mutate <cassette> --seq N --response-file patch.json --out <new_cassette>
    agentreplay ci     <corpus_root>
    agentreplay show   <cassette>
    agentreplay list   <corpus_root>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import click

from agentreplay import __version__
from agentreplay.cassette import Cassette
from agentreplay.ci import RegressionReport, discover_cassettes, run_corpus
from agentreplay.constants import Mode
from agentreplay.diff import diff_structural, render_diff
from agentreplay.errors import AgentReplayError, DivergenceError
from agentreplay.logging import set_verbose
from agentreplay.mutate import mutate_response
from agentreplay.replayer import Replayer
from agentreplay.storage import MetaIndex


@click.group()
@click.version_option(__version__)
@click.option("--verbose/--quiet", default=False, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """AgentReplay — deterministic replay & counterfactual debugging for AI agents."""
    if verbose:
        set_verbose(True)


# ---------------------------------------------------------------------- #
# show
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("cassette", type=click.Path(exists=True, file_okay=False))
@click.option("--events/--no-events", default=False, help="Print every event row.")
def show(cassette: str, events: bool) -> None:
    """Print cassette metadata (and optionally every event)."""
    c = Cassette.open(cassette, readonly=True)
    click.echo(json.dumps(c.meta.to_dict(), indent=2, ensure_ascii=False))
    if events:
        click.echo("--- events ---")
        for ev in c.events:
            click.echo(json.dumps(ev.to_dict(), ensure_ascii=False))


# ---------------------------------------------------------------------- #
# list
# ---------------------------------------------------------------------- #
@cli.command(name="list")
@click.argument("corpus_root", type=click.Path(exists=True, file_okay=False))
@click.option("--task", default=None)
@click.option("--commit", default=None)
@click.option("--model", default=None)
@click.option("--outcome", default=None)
@click.option("--tag", default=None)
@click.option("--limit", default=100, type=int)
@click.option("--json/--text", "as_json", default=False)
def list_cassettes(
    corpus_root: str,
    task: Optional[str],
    commit: Optional[str],
    model: Optional[str],
    outcome: Optional[str],
    tag: Optional[str],
    limit: int,
    as_json: bool,
) -> None:
    """List cassettes in a corpus, optionally filtered by metadata."""
    # If a meta.db exists in the corpus root, use it for indexed queries;
    # otherwise scan filesystem.
    meta_db = Path(corpus_root) / MetaIndex.FILENAME
    if meta_db.exists():
        with MetaIndex(corpus_root) as idx:
            rows = idx.list(
                task_id=task,
                git_commit=commit,
                model=model,
                outcome=outcome,
                tag=tag,
                limit=limit,
            )
        if as_json:
            click.echo(json.dumps(rows, indent=2, default=str))
        else:
            for r in rows:
                click.echo(
                    f"{r['id']}\t{r['framework']}\t{r['outcome']}\t"
                    f"{r['task_id']}\t{r['num_events']}"
                )
        return
    # Filesystem scan fallback.
    rows = []
    for path in discover_cassettes(corpus_root):
        c = Cassette.open(path, readonly=True)
        if task and c.meta.task_id != task:
            continue
        if commit and c.meta.git_commit != commit:
            continue
        if model and c.meta.model != model:
            continue
        if outcome and c.meta.outcome != outcome:
            continue
        if tag and tag not in c.meta.tags:
            continue
        rows.append(c.meta.to_dict())
        if len(rows) >= limit:
            break
    if as_json:
        click.echo(json.dumps(rows, indent=2, default=str))
    else:
        for r in rows:
            click.echo(
                f"{r['id']}\t{r['framework']}\t{r['outcome']}\t"
                f"{r['task_id']}\t{r['num_events']}"
            )


# ---------------------------------------------------------------------- #
# replay
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("cassette", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--mode",
    type=click.Choice([Mode.REPLAY.value, Mode.HYBRID.value]),
    default=Mode.REPLAY.value,
    help="Pure replay (default) or hybrid (fall through to live on divergence).",
)
@click.option(
    "--agent-entry",
    default=None,
    help="Python dotted path to a callable taking a Replayer. "
         "If omitted, just verifies the cassette parses and prints stats.",
)
@click.option("--json/--text", "as_json", default=False)
def replay(cassette: str, mode: str, agent_entry: Optional[str], as_json: bool) -> None:
    """Replay a cassette through an agent entry point.

    If ``--agent-entry`` is omitted, this just opens the cassette and
    prints stats — useful as a sanity check that the recording is intact.
    """
    c = Cassette.open(cassette, readonly=True)
    if agent_entry is None:
        stats = c.stats()
        if as_json:
            click.echo(json.dumps(stats, indent=2, default=str))
        else:
            click.echo(f"cassette: {stats['id']}")
            click.echo(f"  framework: {stats['framework']}")
            click.echo(f"  task_id:   {stats['task_id']}")
            click.echo(f"  outcome:   {stats['outcome']}")
            click.echo(f"  events:    {stats['num_events']}")
            click.echo(f"  blobs:     {stats['blobs']['blobs']} ({stats['blobs']['bytes']} bytes)")
        return

    # Import the entry point and run.
    fn = _import_dotted(agent_entry)
    replayer = Replayer.open(cassette, mode=Mode(mode))
    try:
        result = fn(replayer)
        click.echo(json.dumps({"status": "ok", "result": _safe(result)}, indent=2, default=str))
    except DivergenceError as exc:
        click.echo(
            json.dumps(
                {
                    "status": "diverged",
                    "step_id": exc.step_id,
                    "call_type": exc.call_type,
                    "recorded_call_id": exc.expected_call_id,
                    "actual_call_id": exc.actual_call_id,
                },
                indent=2,
            ),
            err=True,
        )
        sys.exit(2)


# ---------------------------------------------------------------------- #
# diff
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("cassette_a", type=click.Path(exists=True, file_okay=False))
@click.argument("cassette_b", type=click.Path(exists=True, file_okay=False))
@click.option("--json/--text", "as_json", default=False)
def diff(cassette_a: str, cassette_b: str, as_json: bool) -> None:
    """Structural diff between two cassettes."""
    a = Cassette.open(cassette_a, readonly=True)
    b = Cassette.open(cassette_b, readonly=True)
    d = diff_structural(a, b)
    if as_json:
        click.echo(json.dumps(d.summary(), indent=2, default=str))
    else:
        click.echo(render_diff(d))


# ---------------------------------------------------------------------- #
# mutate
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("cassette", type=click.Path(exists=True, file_okay=False))
@click.option("--seq", type=int, default=None, help="Step index to mutate.")
@click.option("--step-id", default=None, help="Step ID to mutate (alternative to --seq).")
@click.option("--call-id", default=None, help="Call-site ID to mutate (alternative to --seq).")
@click.option(
    "--response",
    default=None,
    help="Inline JSON to substitute as the new response.",
)
@click.option(
    "--response-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="File containing JSON to substitute as the new response.",
)
@click.option("--out", type=click.Path(file_okay=False), required=True, help="Output cassette directory.")
@click.option("--new-id", default=None)
def mutate(
    cassette: str,
    seq: Optional[int],
    step_id: Optional[str],
    call_id: Optional[str],
    response: Optional[str],
    response_file: Optional[str],
    out: str,
    new_id: Optional[str],
) -> None:
    """Create a counterfactual cassette by replacing one recorded response."""
    if [seq is not None, step_id is not None, call_id is not None].count(True) != 1:
        raise click.UsageError("exactly one of --seq / --step-id / --call-id is required")
    if response is None and response_file is None:
        raise click.UsageError("one of --response or --response-file is required")
    if response is not None and response_file is not None:
        raise click.UsageError("--response and --response-file are mutually exclusive")

    if response is not None:
        new_response = json.loads(response)
    else:
        new_response = json.loads(Path(response_file).read_text(encoding="utf-8"))  # type: ignore[union-attr]

    forked = mutate_response(
        cassette,
        seq=seq,
        step_id=step_id,
        call_id=call_id,
        new_response=new_response,
        target_root=out,
        new_id=new_id,
    )
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "new_cassette": str(forked.root),
                "new_id": forked.meta.id,
                "mutated_seq": forked.meta.extra.get("mutated_seq"),
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------- #
# ci
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("corpus_root", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--agent-entry",
    required=True,
    help="Python dotted path to a callable taking a Replayer. "
         "Example: my_project.tests:run_agent",
)
@click.option("--stop-on-first-failure/--no-stop", default=False)
@click.option("--tag", default=None)
@click.option("--outcome", default=None)
@click.option("--json/--text", "as_json", default=False)
def ci(
    corpus_root: str,
    agent_entry: str,
    stop_on_first_failure: bool,
    tag: Optional[str],
    outcome: Optional[str],
    as_json: bool,
) -> None:
    """Replay every cassette in a corpus through ``agent_entry``.

    Exits with code 0 if every cassette replayed bit-exact, 1 otherwise.
    In pure-replay mode this consumes zero model calls — see §5.7.
    """
    fn = _import_dotted(agent_entry)
    report = run_corpus(
        corpus_root,
        fn,
        stop_on_first_failure=stop_on_first_failure,
        tag_filter=tag,
        outcome_filter=outcome,
    )
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        click.echo(report.render())
    sys.exit(0 if report.passed else 1)


# ---------------------------------------------------------------------- #
# record
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("cassette", type=click.Path(file_okay=False))
@click.argument("command", nargs=-1, required=True)
@click.option("--framework", default="raw")
@click.option("--task-id", default="")
@click.option("--agent-name", default="")
@click.option("--model", default="")
@click.option("--tag", "tags", multiple=True)
@click.option("--outcome", default="")
def record(
    cassette: str,
    command: tuple,
    framework: str,
    task_id: str,
    agent_name: str,
    model: str,
    tags: tuple,
    outcome: str,
) -> None:
    """Run a subprocess with the AgentReplay recorder auto-installed.

    This is a convenience wrapper: it sets ``AGENTREPLAY_MODE=record``
    and ``AGENTREPLAY_CASSETTE=<cassette>`` in the child's environment
    and execs the given command. The child process must call
    ``agentreplay.auto.init()`` at startup to pick up these env vars.
    """
    env = os.environ.copy()
    env["AGENTREPLAY_MODE"] = "record"
    env["AGENTREPLAY_CASSETTE"] = cassette
    env["AGENTREPLAY_FRAMEWORK"] = framework
    env["AGENTREPLAY_TASK_ID"] = task_id
    env["AGENTREPLAY_AGENT_NAME"] = agent_name
    env["AGENTREPLAY_MODEL"] = model
    env["AGENTREPLAY_TAGS"] = ",".join(tags)
    env["AGENTREPLAY_OUTCOME"] = outcome
    # Pre-import agentreplay.auto via PYTHONPATH if needed.
    try:
        completed = subprocess.run(list(command), env=env)
        sys.exit(completed.returncode)
    except FileNotFoundError as exc:
        click.echo(f"command not found: {exc}", err=True)
        sys.exit(127)


# ---------------------------------------------------------------------- #
# benchmark-overhead
# ---------------------------------------------------------------------- #
@cli.command(name="benchmark-overhead")
@click.option("--iterations", type=int, default=200,
              help="Number of LLM calls per measurement (default: 200).")
@click.option("--repeats", type=int, default=3,
              help="Number of repeats per measurement (median taken).")
@click.option("--report", type=click.Path(dir_okay=False), default=None,
              help="Write JSON report to this path.")
@click.option("--json/--text", "as_json", default=False)
def benchmark_overhead(iterations: int, repeats: int, report: Optional[str], as_json: bool) -> None:
    """Measure recording-layer latency overhead vs. baseline (§7.2).

    Reports percentage overhead vs. an uninstrumented baseline, plus
    synthetic baselines matching the published 2026 figures for
    LangSmith (~0%), Laminar (~5%), AgentOps (~12%), Langfuse (~15%).

    Exits 0 if AgentReplay's overhead is ≤ 5% (the §7.2 target), 1 otherwise.
    """
    from agentreplay.benchmark.overhead import run_benchmark

    rep = run_benchmark(iterations=iterations, repeats=repeats)
    if as_json:
        click.echo(json.dumps(rep.to_dict(), indent=2))
    else:
        click.echo(rep.render())
    if report:
        Path(report).write_text(json.dumps(rep.to_dict(), indent=2))
        click.echo(f"\nReport written to {report}", err=True)
    ar = next((r for r in rep.results if r.name == "AgentReplay (record)"), None)
    sys.exit(0 if ar is not None and ar.overhead_pct <= 5.0 else 1)


# ---------------------------------------------------------------------- #
# info
# ---------------------------------------------------------------------- #
@cli.command()
def info() -> None:
    """Show installed version, supported frameworks, and optional deps status."""
    import importlib

    click.echo(f"AgentReplay v{__version__}")
    click.echo(f"  Python: {sys.version.split()[0]}")
    click.echo(f"  Install: {Path(__file__).parent.parent.parent}")

    # Check optional dependencies
    deps = [
        ("openai", "OpenAI SDK adapter"),
        ("anthropic", "Anthropic SDK adapter"),
        ("langgraph", "LangGraph adapter"),
        ("crewai", "CrewAI adapter"),
        ("autogen", "AutoGen v0.2 adapter"),
        ("datasets", "Real SWE-bench/GAIA validation loaders"),
        ("pytest", "Test suite"),
        ("mkdocs", "Docs build"),
    ]
    click.echo("\nOptional dependencies:")
    for module, label in deps:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "installed")
            click.echo(f"  ✓ {module:<12s} {version:<10s} {label}")
        except ImportError:
            click.echo(f"  ✗ {module:<12s} {'missing':<10s} {label}")

    # Count available framework adapters
    click.echo("\nFramework adapters:")
    from agentreplay.frameworks import __getattr__ as _  # noqa: F401
    for name in ["wrap_openai", "wrap_anthropic", "wrap_langgraph", "wrap_crewai_llm", "wrap_autogen_client"]:
        try:
            fn = __getattr__(name)  # type: ignore
            click.echo(f"  ✓ {name}")
        except Exception:
            click.echo(f"  - {name} (not installed)")


# ---------------------------------------------------------------------- #
# export
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("cassette", type=click.Path(exists=True, file_okay=False))
@click.argument("zip_path", type=click.Path(dir_okay=False))
def export(cassette: str, zip_path: str) -> None:
    """Export a cassette as a ZIP archive for sharing.

    The ZIP contains cassette.json, events.jsonl, and all blob files.
    Use ``agentreplay import`` to reconstruct the cassette from the archive.
    """
    c = Cassette.open(cassette, readonly=True)
    c.export_zip(zip_path)
    click.echo(json.dumps({
        "status": "ok",
        "cassette": cassette,
        "zip": zip_path,
        "size_bytes": Path(zip_path).stat().st_size,
    }, indent=2))


# ---------------------------------------------------------------------- #
# import
# ---------------------------------------------------------------------- #
@cli.command(name="import")
@click.argument("zip_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("target_root", type=click.Path(file_okay=False))
def import_cassette(zip_path: str, target_root: str) -> None:
    """Import a cassette from a ZIP archive created by ``agentreplay export``."""
    c = Cassette.import_zip(zip_path, target_root)
    click.echo(json.dumps({
        "status": "ok",
        "zip": zip_path,
        "cassette_root": target_root,
        "cassette_id": c.meta.id,
        "num_events": len(c.events),
    }, indent=2))


# ---------------------------------------------------------------------- #
# clean
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("corpus_root", type=click.Path(exists=True, file_okay=False))
@click.option("--older-than", default=None,
              help="Remove cassettes older than this (e.g. '30d', '2w', '6m').")
@click.option("--keep-outcome", default=None,
              help="Keep cassettes with this outcome (e.g. 'fail').")
@click.option("--dry-run/--no-dry-run", default=True,
              help="Show what would be deleted without actually deleting (default: dry-run).")
def clean(corpus_root: str, older_than: Optional[str], keep_outcome: Optional[str], dry_run: bool) -> None:
    """Remove old or unwanted cassettes from a corpus.

    By default runs in dry-run mode — shows what would be deleted but
    doesn't actually delete. Pass --no-dry-run to actually delete.

    Examples::

        agentreplay clean cassettes/ --older-than 30d --keep-outcome fail
        agentreplay clean cassettes/ --older-than 2w --no-dry-run
    """
    import re
    from datetime import datetime, timedelta

    # Parse --older-than
    delta: Optional[timedelta] = None
    if older_than:
        m = re.match(r"^(\d+)([dwm])$", older_than)
        if not m:
            raise click.UsageError(f"invalid --older-than: {older_than!r} (use e.g. '30d', '2w', '6m')")
        n, unit = int(m.group(1)), m.group(2)
        if unit == "d":
            delta = timedelta(days=n)
        elif unit == "w":
            delta = timedelta(weeks=n)
        elif unit == "m":
            delta = timedelta(days=n * 30)

    cutoff = datetime.now() - delta if delta else None
    removed = 0
    kept = 0

    for path in discover_cassettes(corpus_root):
        try:
            c = Cassette.open(path, readonly=True)
        except Exception:
            continue

        # Check age
        if cutoff is not None:
            created = datetime.fromtimestamp(c.meta.created_at)
            if created > cutoff:
                kept += 1
                continue

        # Check outcome keep filter
        if keep_outcome and c.meta.outcome == keep_outcome:
            kept += 1
            continue

        # Would remove
        removed += 1
        # Safety: never delete the corpus root itself
        if Path(path) == Path(corpus_root).resolve():
            click.echo(f"  [SKIP] refusing to delete corpus root: {path}")
            removed -= 1
            continue
        if dry_run:
            click.echo(f"  [dry-run] would remove: {path} (id={c.meta.id}, outcome={c.meta.outcome})")
        else:
            import shutil
            shutil.rmtree(path)
            click.echo(f"  removed: {path}")

    click.echo(f"\n{'Would remove' if dry_run else 'Removed'}: {removed}")
    click.echo(f"Kept: {kept}")


# ---------------------------------------------------------------------- #
# doctor
# ---------------------------------------------------------------------- #
@cli.command()
@click.argument("cassette", type=click.Path(exists=True, file_okay=False))
def doctor(cassette: str) -> None:
    """Validate cassette health — check for missing blobs, corrupted events, etc.

    Exits 0 if the cassette is healthy, 1 if issues are found.
    """
    c = Cassette.open(cassette, readonly=True)
    issues: list[str] = []

    # Check 1: every event's request_hash and response_hash must exist in the blob store
    for ev in c.events:
        if not c.blobs.has(ev.request_hash):
            issues.append(f"event seq={ev.seq}: missing request blob {ev.request_hash}")
        if not c.blobs.has(ev.response_hash):
            issues.append(f"event seq={ev.seq}: missing response blob {ev.response_hash}")

    # Check 2: metadata num_events should match actual event count
    actual_events = len(c.events)
    if c.meta.num_events != actual_events:
        issues.append(f"metadata num_events={c.meta.num_events} but actual={actual_events}")

    # Check 3: seq numbers should be contiguous starting from 0
    seqs = [e.seq for e in c.events]
    expected = list(range(len(seqs)))
    if seqs != expected:
        issues.append(f"seq numbers are not contiguous: {seqs[:10]}...")

    # Check 4: cassette.json should be valid JSON (already verified by Cassette.open)
    # Check 5: events.jsonl should be valid JSONL (already verified by EventLog iteration)

    if issues:
        click.echo(f"✗ {len(issues)} issue(s) found in cassette {c.meta.id}:")
        for issue in issues:
            click.echo(f"  · {issue}")
        sys.exit(1)
    else:
        click.echo(f"✓ cassette {c.meta.id} is healthy")
        click.echo(f"  events: {actual_events}")
        click.echo(f"  blobs: {c.blobs.stats()['blobs']} ({c.blobs.stats()['bytes']} bytes)")
        click.echo(f"  framework: {c.meta.framework}")
        click.echo(f"  outcome: {c.meta.outcome}")
        sys.exit(0)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _import_dotted(path: str) -> Any:
    if ":" not in path:
        raise click.UsageError(
            "--agent-entry must be a dotted path 'module:callable', got " + repr(path)
        )
    module_name, attr = path.split(":", 1)
    import importlib

    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise click.UsageError(f"attribute {attr!r} not found on module {module_name!r}")
    return fn


def _safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


if __name__ == "__main__":  # pragma: no cover
    cli()


# Register validation subcommands (validate-swebench, validate-gaia).
# Imported at module bottom to avoid circular imports.
from agentreplay.validation import cli as _validation_cli  # noqa: E402, F401
