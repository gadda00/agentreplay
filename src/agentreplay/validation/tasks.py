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
# Real task set loaders
# ---------------------------------------------------------------------- #
class SwebenchVerifiedTaskSet(TaskSet):
    """Loader for the real SWE-bench Verified task set.

    SWE-bench Verified is a public benchmark of 500 real GitHub issue
    resolution tasks, frozen at specific Docker repository states. To
    use this loader you need:

      1. The ``datasets`` package: ``pip install datasets``
      2. A real LLM client (OpenAI / Anthropic) to record the initial
         cassettes. Set ``OPENAI_API_KEY`` (or ``ANTHROPIC_API_KEY``).
      3. Docker (optional, recommended) for sandboxed tool execution —
         see §8 of the product proposal.

    The dataset is downloaded from HuggingFace on first use and cached
    locally by the ``datasets`` library.
    """

    DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
    DATASET_SIZE = 500

    def __init__(self, *, dataset_dir: Optional[Path] = None) -> None:
        self.dataset_dir = dataset_dir

    def load(self, *, limit: Optional[int] = None) -> List[Task]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise NotImplementedError(
                "SwebenchVerifiedTaskSet.load requires the `datasets` package. "
                "Install with `pip install datasets` and retry."
            ) from exc

        ds = load_dataset(self.DATASET_NAME, split="test")
        tasks: list[Task] = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            # SWE-bench Verified rows have: repo, instance_id, base_commit,
            # patch, test_patch, problem_statement, hints_text,
            # FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit
            instance_id = row.get("instance_id", f"swe-bench:{i}")
            problem = row.get("problem_statement", "")
            repo = row.get("repo", "unknown")
            tasks.append(
                Task(
                    id=f"swe-bench:{instance_id}",
                    description=f"SWE-bench Verified task {instance_id} ({repo})",
                    messages=[
                        {"role": "user", "content": problem}
                    ],
                    expected={
                        "patch": row.get("patch", ""),
                        "fail_to_pass": row.get("FAIL_TO_PASS", "[]"),
                        "pass_to_pass": row.get("PASS_TO_PASS", "[]"),
                    },
                    tools={},  # SWE-bench agents bring their own tools
                    metadata={
                        "repo": repo,
                        "base_commit": row.get("base_commit", ""),
                        "test_patch": row.get("test_patch", ""),
                        "environment_setup_commit": row.get("environment_setup_commit", ""),
                        "hints_text": row.get("hints_text", ""),
                    },
                )
            )
        return tasks


class GaiaTaskSet(TaskSet):
    """Loader for the real GAIA task set.

    GAIA is a public benchmark of 466 real-world multi-step assistant
    tasks. To use this loader you need:

      1. The ``datasets`` package: ``pip install datasets``
      2. A real LLM client with web/search tool access (GAIA tasks
         involve live web lookups — this is the harder, more realistic
         test of the recording layer per §6 of the product proposal).
      3. Note: the GAIA dataset requires accepting the dataset license
         on HuggingFace. Visit
         https://huggingface.co/datasets/gaia-benchmark/GAIA and accept
         the terms before first use.

    The dataset is downloaded from HuggingFace on first use and cached
    locally by the ``datasets`` library.
    """

    DATASET_NAME = "gaia-benchmark/GAIA"
    DATASET_SIZE = 466

    def __init__(self, *, dataset_dir: Optional[Path] = None) -> None:
        self.dataset_dir = dataset_dir

    def load(self, *, limit: Optional[int] = None) -> List[Task]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise NotImplementedError(
                "GaiaTaskSet.load requires the `datasets` package. "
                "Install with `pip install datasets` and retry."
            ) from exc

        ds = load_dataset(self.DATASET_NAME, "2023_all", split="validation")
        tasks: list[Task] = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            # GAIA rows have: task_id, Question, Level, Final Answer,
            # file_name, file_path, Annotator Metadata
            task_id = row.get("task_id", f"gaia:{i}")
            question = row.get("Question", "")
            level = row.get("Level", 1)
            final_answer = row.get("Final Answer", "")
            tasks.append(
                Task(
                    id=f"gaia:{task_id}",
                    description=f"GAIA Level {level} task: {question[:100]}",
                    messages=[
                        {"role": "user", "content": question}
                    ],
                    expected={
                        "final_answer": final_answer,
                        "level": level,
                    },
                    tools={},  # GAIA agents bring their own web/search tools
                    metadata={
                        "task_id": task_id,
                        "level": level,
                        "file_name": row.get("file_name", ""),
                        "file_path": row.get("file_path", ""),
                        "annotator_metadata": row.get("Annotator Metadata", ""),
                    },
                )
            )
        return tasks


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
