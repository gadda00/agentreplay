"""CLI entry points for the validation harness.

Provides ``agentreplay validate-swebench`` and ``agentreplay validate-gaia``
commands that run the reproduction-fidelity check (§7.1) on either the
synthetic task set (CI-friendly, no API key) or the real task sets
(requires API key + dataset download).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from agentreplay.cli import cli
from agentreplay.validation import (
    run_validation,
    load_synthetic_tasks,
)
from agentreplay.validation.fidelity import FidelityReport
from agentreplay.validation.tasks import (
    SyntheticTaskSet,
    SwebenchVerifiedTaskSet,
    GaiaTaskSet,
    get_task_set,
)


def _run_validation_cmd(
    task_set_name: str,
    *,
    limit: Optional[int],
    cassette_root: str,
    report_path: Optional[str],
    as_json: bool,
) -> None:
    """Shared logic for the swebench / gaia validation subcommands."""
    try:
        task_set = get_task_set(task_set_name)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(2)

    try:
        report = run_validation(
            task_set,
            cassette_root=cassette_root,
            limit=limit,
        )
    except NotImplementedError as exc:
        # Real task set not implemented — surface a friendly message.
        click.echo(f"Task set {task_set_name!r} requires external setup:\n  {exc}", err=True)
        sys.exit(2)

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        click.echo(report.render())
    if report_path:
        Path(report_path).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        click.echo(f"\nReport written to {report_path}", err=True)

    # §7.1 target: 100% fidelity.
    sys.exit(0 if report.passed else 1)


@cli.command(name="validate-swebench")
@click.option("--tasks", default="synthetic",
              help="Task set: 'synthetic' (default, CI-friendly), "
                   "'swebench-verified' (requires setup).")
@click.option("--limit", type=int, default=None,
              help="Max number of tasks to run (default: all).")
@click.option("--cassette-root", default="cassettes/validation/swebench",
              help="Directory to write cassettes under.")
@click.option("--report", type=click.Path(dir_okay=False), default=None,
              help="Write JSON report to this path.")
@click.option("--json/--text", "as_json", default=False)
def validate_swebench(
    tasks: str,
    limit: Optional[int],
    cassette_root: str,
    report: Optional[str],
    as_json: bool,
) -> None:
    """Run reproduction-fidelity validation (§7.1) on a SWE-bench task set.

    By default uses the built-in synthetic task set, which runs without
    any API key or dataset download. To use the real SWE-bench Verified
    corpus, pass ``--tasks swebench-verified`` (requires setup — see
    :class:`agentreplay.validation.tasks.SwebenchVerifiedTaskSet`).

    Exits 0 if fidelity = 100% (§7.1 target), 1 otherwise.
    """
    _run_validation_cmd(
        tasks,
        limit=limit,
        cassette_root=cassette_root,
        report_path=report,
        as_json=as_json,
    )


@cli.command(name="validate-gaia")
@click.option("--tasks", default="synthetic",
              help="Task set: 'synthetic' (default, CI-friendly), "
                   "'gaia-subset' (requires setup).")
@click.option("--limit", type=int, default=None,
              help="Max number of tasks to run (default: all).")
@click.option("--cassette-root", default="cassettes/validation/gaia",
              help="Directory to write cassettes under.")
@click.option("--report", type=click.Path(dir_okay=False), default=None,
              help="Write JSON report to this path.")
@click.option("--json/--text", "as_json", default=False)
def validate_gaia(
    tasks: str,
    limit: Optional[int],
    cassette_root: str,
    report: Optional[str],
    as_json: bool,
) -> None:
    """Run reproduction-fidelity validation (§7.1) on a GAIA task set.

    By default uses the built-in synthetic task set, which runs without
    any API key or dataset download. To use the real GAIA corpus, pass
    ``--tasks gaia-subset`` (requires setup — see
    :class:`agentreplay.validation.tasks.GaiaTaskSet`).

    Exits 0 if fidelity = 100% (§7.1 target), 1 otherwise.
    """
    _run_validation_cmd(
        tasks,
        limit=limit,
        cassette_root=cassette_root,
        report_path=report,
        as_json=as_json,
    )
