"""Regression test for the cross-parse node-identity bug found while smoke-testing this
module: `neighbourhood()` must derive the witness and its ancestors from a single `ast.parse`
call, or `parent_leaf`/`function` silently degenerate to `leaf`.
"""
from __future__ import annotations

import unittest

from egr.localize import AstLocalizer, escalate, locate
from egr.types import Evidence

NESTED = (
    "def f(nums):\n"
    "    total = 0\n"
    "    for n in nums:\n"
    "        if n > 0:\n"
    "            total += n\n"
    "    return total\n"
)


def _ev(line: int) -> Evidence:
    return Evidence(kind="assertion", lines=(line,), scores={}, summary="", signature="x")


class TestScopeLadder(unittest.TestCase):
    def test_scopes_differ_when_nesting_allows_it(self):
        ev = _ev(5)  # "total += n", nested inside `if` inside `for`
        leaf = locate(NESTED, ev, "leaf")
        parent_leaf = locate(NESTED, ev, "parent_leaf")
        function = locate(NESTED, ev, "function")

        self.assertEqual(NESTED[leaf.start:leaf.end], "            total += n\n")
        self.assertIn("if n > 0:", NESTED[parent_leaf.start:parent_leaf.end])
        self.assertNotIn("for n in nums:", NESTED[parent_leaf.start:parent_leaf.end])
        self.assertIn("def f(nums):", NESTED[function.start:function.end])
        self.assertIn("return total", NESTED[function.start:function.end])

        # each scope must be a superset of the narrower one, in line terms
        self.assertLessEqual(parent_leaf.start, leaf.start)
        self.assertGreaterEqual(parent_leaf.end, leaf.end)
        self.assertLessEqual(function.start, parent_leaf.start)
        self.assertGreaterEqual(function.end, parent_leaf.end)

    def test_escalate_ladder_and_termination(self):
        self.assertEqual(escalate("leaf"), "parent_leaf")
        self.assertEqual(escalate("parent_leaf"), "function")
        self.assertIsNone(escalate("function"))

    def test_locate_returns_none_without_evidence_lines(self):
        ev = Evidence(kind="timeout", lines=(), scores={}, summary="", signature="x")
        self.assertIsNone(locate(NESTED, ev, "leaf"))

    def test_ast_localizer_standalone(self):
        witness = AstLocalizer().witness(NESTED, _ev(5))
        self.assertEqual(witness.lineno, 5)  # the `total += n` statement itself


if __name__ == "__main__":
    unittest.main()
