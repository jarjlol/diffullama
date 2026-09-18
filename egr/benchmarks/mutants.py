"""Mutation injection with known ground truth. See docs/01-architecture.md module map and
docs/02-experiment-plan.md Sec 3.1.

================================================================================================
STATUS: `_FIXTURES` below is a small, SELF-CONTAINED placeholder -- six hand-written problems,
not the real 164 HumanEval+ canonical solutions. This environment has neither network access
nor the `evalplus`/`datasets` packages installed (parallel situation to DiffuLLaMA itself --
see backend.py's module docstring).

TODO(compute/network): once available, replace `_FIXTURES` with a loader over the real
HumanEval+ canonical solutions (`entry_point`, `solution` source, and rendered test cases per
docs/04-build-phases.md Phase 2's 542-problem acceptance criterion). Nothing downstream --
evidence.py, localize.py, policy.py, loop.py, cli.py -- needs to change: they only depend on
the `Task` objects `tasks()` yields.
================================================================================================
"""
from __future__ import annotations

import ast
import copy
import logging
from collections.abc import Iterator

from egr.benchmarks.base import render_asserts
from egr.types import Task

log = logging.getLogger(__name__)

_FIXTURES = [
    {"id": "add", "entry_point": "add",
     "solution": "def add(a, b):\n    return a + b\n",
     "cases": [((1, 2), 3), ((0, 0), 0), ((-1, 1), 0), ((5, 7), 12)]},
    {"id": "is_even", "entry_point": "is_even",
     "solution": "def is_even(n):\n    return n % 2 == 0\n",
     "cases": [((2,), True), ((3,), False), ((0,), True), ((-4,), True)]},
    {"id": "max_of_list", "entry_point": "max_of_list",
     "solution": (
         "def max_of_list(nums):\n"
         "    best = nums[0]\n"
         "    for n in nums[1:]:\n"
         "        if n > best:\n"
         "            best = n\n"
         "    return best\n"
     ),
     "cases": [(([1, 5, 2],), 5), (([3, 3, 3],), 3), (([-1, -5, -2],), -1)]},
    {"id": "factorial", "entry_point": "factorial",
     "solution": (
         "def factorial(n):\n"
         "    result = 1\n"
         "    for i in range(n, 0, -1):\n"
         "        result *= i\n"
         "    return result\n"
     ),
     "cases": [((0,), 1), ((1,), 1), ((5,), 120)]},
    {"id": "count_positive", "entry_point": "count_positive",
     "solution": (
         "def count_positive(nums):\n"
         "    count = 0\n"
         "    for n in nums:\n"
         "        if n > 0:\n"
         "            count += 1\n"
         "    return count\n"
     ),
     "cases": [(([1, -1, 2, -2, 3],), 3), (([-1, -2],), 0), (([],), 0)]},
    {"id": "reverse_string", "entry_point": "reverse_string",
     "solution": "def reverse_string(s):\n    return s[::-1]\n",
     "cases": [(("abc",), "cba"), (("",), ""), (("a",), "a")]},
]


class _ComparisonFlip(ast.NodeTransformer):
    _FLIP = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}

    def __init__(self) -> None:
        self.mutated_lineno: int | None = None

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.mutated_lineno is None:
            for i, op in enumerate(node.ops):
                if type(op) in self._FLIP:
                    node.ops[i] = self._FLIP[type(op)]()
                    self.mutated_lineno = node.lineno
                    break
        return node


class _OffByOne(ast.NodeTransformer):
    def __init__(self) -> None:
        self.mutated_lineno: int | None = None

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if (self.mutated_lineno is None and isinstance(node.value, int)
                and not isinstance(node.value, bool)):
            node.value += 1
            self.mutated_lineno = node.lineno
        return node


class _OperatorSwap(ast.NodeTransformer):
    _SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv}

    def __init__(self) -> None:
        self.mutated_lineno: int | None = None

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.mutated_lineno is None and type(node.op) in self._SWAP:
            node.op = self._SWAP[type(node.op)]()
            self.mutated_lineno = node.lineno
        return node


_AST_OPERATORS = {
    "comparison_flip": _ComparisonFlip,
    "off_by_one": _OffByOne,
    "operator_swap": _OperatorSwap,
}


def _apply(operator_cls: type[ast.NodeTransformer], source: str) -> tuple[str, int] | None:
    tree = ast.parse(source)
    transformer = operator_cls()
    mutated_tree = transformer.visit(copy.deepcopy(tree))
    if transformer.mutated_lineno is None:
        return None
    ast.fix_missing_locations(mutated_tree)
    return ast.unparse(mutated_tree) + "\n", transformer.mutated_lineno


def _syntax_corruption(source: str) -> tuple[str, int] | None:
    """Deliberately produces a SyntaxError by dropping a block's trailing colon -- exercises
    the `syntax` evidence path, which a base diffusion LM hits often (README Sec 6).
    """
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if stripped.endswith(":"):
            lines[i] = stripped[:-1] + "\n"
            lineno = i
            lineno += 1
            return "".join(lines), lineno
    return None


def _mutant_actually_fails(entry_point: str, source: str, cases: list[tuple]) -> bool:
    """Sanity-checks a mutant at FIXTURE-CONSTRUCTION time, not evaluation time -- this execs
    OUR OWN generated source, never model output, so H-8's sandbox does not apply here. A
    mutation that happens not to change behaviour on these cases would be a useless
    localization example, so it is filtered out rather than yielded.
    """
    namespace: dict = {}
    try:
        exec(compile(source, "<mutant>", "exec"), namespace)
        fn = namespace[entry_point]
        for args, expected in cases:
            if fn(*args) != expected:
                return True
        return False
    except Exception:
        return True


class MutantsBenchmark:
    """Injects single-statement mutations into small fixture solutions. Ground truth (the
    mutated line) is known by construction -- the only way to score localization directly, at
    zero GPU-hours (docs/04-build-phases.md Phase 2).
    """

    name = "mutants"

    def tasks(self, limit: int | None = None) -> Iterator[Task]:
        count = 0
        for fixture in _FIXTURES:
            solution = fixture["solution"]
            entry_point = fixture["entry_point"]
            cases = fixture["cases"]
            inputs = [c[0] for c in cases]
            expected = [c[1] for c in cases]
            tests = render_asserts(entry_point, inputs, expected)

            candidates: list[tuple[str, tuple[str, int] | None]] = [
                (op_name, _apply(cls, solution)) for op_name, cls in _AST_OPERATORS.items()
            ]
            candidates.append(("syntax_corruption", _syntax_corruption(solution)))

            for op_name, result in candidates:
                if result is None:
                    continue
                mutated_source, lineno = result
                if mutated_source == solution:
                    continue
                if op_name != "syntax_corruption" and not _mutant_actually_fails(entry_point, mutated_source, cases):
                    log.debug("%s::%s: mutant did not change behaviour, skipping", fixture["id"], op_name)
                    continue

                yield Task(
                    task_id=f"mutants/{fixture['id']}::{op_name}",
                    prompt=f"# {fixture['id']}: repair the failing implementation of {entry_point}",
                    seed_program=mutated_source, tests=tests, entry_point=entry_point,
                    ground_truth_line=lineno, oracle_program=solution,
                )
                count += 1
                if limit is not None and count >= limit:
                    return
