"""Task record + Benchmark protocol. See docs/01-architecture.md module map."""
from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Protocol

from egr.types import Task

log = logging.getLogger(__name__)


class Benchmark(Protocol):
    name: str

    def tasks(self, limit: int | None = None) -> Iterator[Task]: ...


def render_asserts(entry_point: str, inputs: Sequence[tuple], expected: Sequence) -> str:
    """Render (args, expected-output) pairs into runnable `test_N` functions.

    H-10: EvalPlus stores test INPUTS evaluated differentially against `canonical_solution`,
    not literal assert strings -- an adapter that assumes otherwise silently produces an empty
    test suite, and an empty suite passes.
    """
    if not inputs:
        raise ValueError("render_asserts got zero cases -- would produce an empty test suite")
    lines: list[str] = []
    for i, (args, expected_out) in enumerate(zip(inputs, expected, strict=True)):
        args_repr = ", ".join(repr(a) for a in args)
        lines.append(f"def test_{i}():")
        lines.append(f"    from solution import {entry_point}")
        lines.append(f"    assert {entry_point}({args_repr}) == {expected_out!r}")
        lines.append("")
    return "\n".join(lines)
