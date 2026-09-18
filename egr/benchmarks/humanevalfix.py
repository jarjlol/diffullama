"""HumanEvalFix (from OctoPack): a repair benchmark giving buggy code + failing tests. See
docs/01-architecture.md module map, README Sec 6.

STATUS: no network/`datasets` access in this environment -- reuses `MutantsBenchmark`'s
fixture mutants as a stand-in "buggy code + failing tests" corpus (same six problems, same
mutation operators; only the task-id prefix and the dropped `ground_truth_line` differ, since
HumanEvalFix does not score localization).

TODO(compute/network): once available, replace `tasks()`'s body with a loader over the real
`bigcode/humanevalpack` "HumanEvalFix" Python split. The `Task` shape this yields already
matches what `loop.py` expects, so nothing downstream should need to change.
"""
from __future__ import annotations

from collections.abc import Iterator

from egr.benchmarks.mutants import MutantsBenchmark
from egr.types import Task


class HumanEvalFixBenchmark:
    name = "humanevalfix"

    def tasks(self, limit: int | None = None) -> Iterator[Task]:
        for task in MutantsBenchmark().tasks(limit=limit):
            yield Task(
                task_id=task.task_id.replace("mutants/", "humanevalfix/"),
                prompt=task.prompt, seed_program=task.seed_program, tests=task.tests,
                entry_point=task.entry_point, ground_truth_line=None,
                oracle_program=task.oracle_program,
            )
