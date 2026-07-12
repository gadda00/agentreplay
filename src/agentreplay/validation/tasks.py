"""Task definitions for the validation harness.

A *task* is a single agent run that gets recorded into a cassette and
then replayed for the fidelity check. The :class:`TaskSet` ABC defines
the interface; :class:`SyntheticTaskSet` provides a built-in set that
runs without any external API or dataset download.

To plug in the real SWE-bench Verified or GAIA task sets, subclass
:class:`TaskSet` and implement :meth:`load`. See the docstring there
for details.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Task:
    """A single validation task.

    A task bundles:

      - ``id``         : unique identifier (e.g. ``"swe-bench:django-12345"``)
      - ``description``: human-readable summary
      - ``messages``   : the initial message list to send to the model
      - ``expected``   : the expected terminal response (for assertion)
      - ``tools``      : optional list of tool callables the agent can use
      - ``metadata``   : arbitrary extra metadata (git commit, repo, etc.)
    """

    id: str
    description: str
    messages: List[Dict[str, Any]]
    expected: Any
    tools: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskSet(abc.ABC):
    """Abstract base class for a set of validation tasks.

    Subclasses implement :meth:`load` to return a list of :class:`Task`
    objects. For SWE-bench Verified, this would download the public task
    corpus from GitHub and convert each task into the :class:`Task` shape;
    for GAIA, the same but from HuggingFace.

    The :class:`SyntheticTaskSet` implementation provides a built-in set
    of tasks that run without any external dependencies — useful for CI.
    """

    @abc.abstractmethod
    def load(self, *, limit: Optional[int] = None) -> List[Task]:
        """Load and return the task list, optionally truncated to ``limit``."""
        ...


# ---------------------------------------------------------------------- #
# Synthetic task set (built-in, no external dependencies)
# ---------------------------------------------------------------------- #
def _echo_tool(text: str) -> str:
    return f"echo:{text}"


def _reverse_tool(text: str) -> str:
    return text[::-1]


def _lookup_tool(key: str) -> str:
    return f"value-for-{key}"


class SyntheticTaskSet(TaskSet):
    """A built-in synthetic task set for CI.

    These tasks exercise the same record/replay code paths as real
    SWE-bench / GAIA tasks but use a stub LLM and in-memory tools, so
    they run without any API key or dataset download. The point is to
    verify the *harness* works end-to-end; the *fidelity numbers* from
    this set are not meaningful as product claims (a stub LLM is trivially
    reproducible), but the harness's ability to detect a divergence IS
    meaningful.
    """

    def __init__(self, num_tasks: int = 5) -> None:
        self.num_tasks = num_tasks

    def load(self, *, limit: Optional[int] = None) -> List[Task]:
        n = min(limit or self.num_tasks, self.num_tasks)
        tasks: list[Task] = []
        for i in range(n):
            tasks.append(
                Task(
                    id=f"synthetic:{i:03d}",
                    description=f"Synthetic validation task {i}",
                    messages=[
                        {"role": "user", "content": f"Task {i}: please echo and reverse the word 'hello{i}'."}
                    ],
                    expected={
                        "final_text": f"echo:hello{i} → reversed: {('hello' + str(i))[::-1]}",
                    },
                    tools={"echo": _echo_tool, "reverse": _reverse_tool},
                    metadata={"synthetic": True},
                )
            )
        # A second batch — GAIA-style tasks with a lookup tool.
        for i in range(n):
            tasks.append(
                Task(
                    id=f"synthetic-lookup:{i:03d}",
                    description=f"Synthetic lookup task {i}",
                    messages=[
                        {"role": "user", "content": f"Look up the value for key 'k{i}'."}
                    ],
                    expected={"final_text": f"value-for-k{i}"},
                    tools={"lookup": _lookup_tool},
                    metadata={"synthetic": True, "style": "gaia"},
                )
            )
        return tasks


def load_synthetic_tasks(num_tasks: int = 5, *, limit: Optional[int] = None) -> List[Task]:
    """Convenience function: load synthetic tasks."""
    return SyntheticTaskSet(num_tasks=num_tasks).load(limit=limit)


# ---------------------------------------------------------------------- #
# Real task set loaders (stubs — require external deps to implement)
# ---------------------------------------------------------------------- #
class SwebenchVerifiedTaskSet(TaskSet):
    """Loader for the real SWE-bench Verified task set.

    SWE-bench Verified is a public benchmark of 500 real GitHub issue
    resolution tasks, frozen at specific Docker repository states. To
    use this loader you need:

      1. The SWE-bench Verified dataset. The canonical source is the
         HuggingFace dataset ``princeton-nlp/SWE-bench_Verified``; load
         it with::

             from datasets import load_dataset
             ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

      2. A real LLM client (OpenAI / Anthropic) to record the initial
         cassettes. Set ``OPENAI_API_KEY`` (or ``ANTHROPIC_API_KEY``).

      3. Docker (optional, recommended) for sandboxed tool execution —
         see §8 of the product proposal.

    This loader is a *stub*: the :meth:`load` method raises
    :class:`NotImplementedError` with a pointer to the setup steps.
    Implement it by converting each SWE-bench task dict into a
    :class:`Task` object.
    """

    DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
    DATASET_SIZE = 500

    def __init__(self, *, dataset_dir: Optional[Path] = None) -> None:
        self.dataset_dir = dataset_dir

    def load(self, *, limit: Optional[int] = None) -> List[Task]:
        raise NotImplementedError(
            "SwebenchVerifiedTaskSet.load requires the `datasets` package "
            "and a downloaded copy of the SWE-bench Verified corpus. "
            "Install with `pip install datasets`, then implement this "
            "method to convert each task dict from "
            f"load_dataset({self.DATASET_NAME!r}, split='test') into a "
            "agentreplay.validation.tasks.Task object. See the class "
            "docstring for details."
        )


class GaiaTaskSet(TaskSet):
    """Loader for the real GAIA task set.

    GAIA is a public benchmark of 466 real-world multi-step assistant
    tasks. To use this loader you need:

      1. The GAIA dataset. The canonical source is the HuggingFace
         dataset ``gaia-benchmark/GAIA``; load it with::

             from datasets import load_dataset
             ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

      2. A real LLM client with web/search tool access (GAIA tasks
         involve live web lookups — this is the harder, more realistic
         test of the recording layer per §6 of the product proposal).

    This loader is a *stub* for the same reason as
    :class:`SwebenchVerifiedTaskSet`.
    """

    DATASET_NAME = "gaia-benchmark/GAIA"
    DATASET_SIZE = 466

    def __init__(self, *, dataset_dir: Optional[Path] = None) -> None:
        self.dataset_dir = dataset_dir

    def load(self, *, limit: Optional[int] = None) -> List[Task]:
        raise NotImplementedError(
            "GaiaTaskSet.load requires the `datasets` package and a "
            "downloaded copy of the GAIA corpus. Install with "
            "`pip install datasets`, then implement this method to "
            "convert each task dict from "
            f"load_dataset({self.DATASET_NAME!r}, '2023_all', split='validation') "
            "into a agentreplay.validation.tasks.Task object. See the "
            "class docstring for details."
        )


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
TASK_SETS: Dict[str, Callable[..., TaskSet]] = {
    "synthetic": SyntheticTaskSet,
    "swebench-verified": SwebenchVerifiedTaskSet,
    "gaia-subset": GaiaTaskSet,
}


def get_task_set(name: str, **kwargs: Any) -> TaskSet:
    """Look up a task set by name.

    Recognised names: ``"synthetic"``, ``"swebench-verified"``,
    ``"gaia-subset"``.
    """
    if name not in TASK_SETS:
        raise ValueError(
            f"unknown task set {name!r}; known: {list(TASK_SETS)}"
        )
    return TASK_SETS[name](**kwargs)
