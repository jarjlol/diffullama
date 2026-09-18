"""HumanEval+ (EvalPlus). See docs/01-architecture.md module map, README Sec 6.

================================================================================================
STATUS: NOT CONNECTED. This needs either the `evalplus` package or network access to fetch its
dataset, neither of which is available in this environment. The repair track (Track A) is
fully exercised without this benchmark, by `mutants.py` and `humanevalfix.py`, which need
neither. The generation track (Track B -- the model provides depth-0's program from a bare
prompt) is ALSO not yet wired into `loop.py`, which always verifies `task.seed_program`; that
is separate future work, not just a missing dataset.

TODO(compute/network): install `evalplus`, replace `tasks()`'s body with a real loader over
HumanEval+'s 164 problems, rendering each problem's `base_input`/`plus_input` cases into
asserts via `egr.benchmarks.base.render_asserts` (H-10: EvalPlus stores inputs, not literal
assert strings). Decide how Track B feeds an initial program into `loop.py` before wiring this
benchmark into any `--backend diffullama` generation run.
================================================================================================
"""
from __future__ import annotations

from collections.abc import Iterator

from egr.types import Task


class HumanEvalPlusBenchmark:
    name = "humaneval_plus"

    def tasks(self, limit: int | None = None) -> Iterator[Task]:
        raise RuntimeError(
            "HumanEvalPlusBenchmark is not connected in this environment (no `evalplus` "
            "package / network access). Use --benchmark mutants or --benchmark humanevalfix "
            "for the repair track -- both need neither. See this module's docstring and "
            "deploy.md's 'known simplifications'."
        )
        yield  # pragma: no cover -- keeps this a generator without ever executing
