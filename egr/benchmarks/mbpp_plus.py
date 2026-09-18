"""MBPP+ (EvalPlus). See docs/01-architecture.md module map, README Sec 6.

STATUS: NOT CONNECTED -- same reason and same fix as `humaneval_plus.py`: no `evalplus`
package / network access in this environment. See that module's docstring; the same TODO
applies here for MBPP+'s 378 problems.
"""
from __future__ import annotations

from collections.abc import Iterator

from egr.types import Task


class MbppPlusBenchmark:
    name = "mbpp_plus"

    def tasks(self, limit: int | None = None) -> Iterator[Task]:
        raise RuntimeError(
            "MbppPlusBenchmark is not connected in this environment (no `evalplus` package / "
            "network access). Use --benchmark mutants or --benchmark humanevalfix for the "
            "repair track -- both need neither. See egr/benchmarks/humaneval_plus.py's "
            "docstring and deploy.md's 'known simplifications'."
        )
        yield  # pragma: no cover -- keeps this a generator without ever executing
