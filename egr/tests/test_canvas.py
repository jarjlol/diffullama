"""H-2 (indentation bleed) and H-11 (hand-built positional arrays)."""
from __future__ import annotations

import unittest

from egr.canvas import Canvas, char_span_of_lines, snap_to_lines
from egr.tests.helpers import CharBackend
from egr.types import Span

PROGRAM = (
    "def total(nums):\n"
    "    acc = 0\n"
    "    for n in nums:\n"
    "        acc += n\n"
    "    return acc\n"
)


class TestSpansAreLineAligned(unittest.TestCase):
    def _assert_line_aligned(self, text: str, span: Span):
        self.assertTrue(span.start == 0 or text[span.start - 1] == "\n", f"start={span.start}")
        self.assertTrue(span.end == len(text) or text[span.end - 1] == "\n", f"end={span.end}")

    def test_mid_line_span_snaps_outward(self):
        # char span covering only "acc" in the middle of "        acc += n\n" -- not aligned.
        start = PROGRAM.index("acc +=")
        raw = Span(start, start + 3, "char")
        snapped = snap_to_lines(PROGRAM, raw)
        self._assert_line_aligned(PROGRAM, snapped)
        self.assertEqual(PROGRAM[snapped.start:snapped.end], "        acc += n\n")

    def test_span_crossing_two_lines_snaps_both_ends(self):
        start = PROGRAM.index("for n")
        end = PROGRAM.index("return") + 3  # ends mid "return"
        snapped = snap_to_lines(PROGRAM, Span(start, end, "char"))
        self._assert_line_aligned(PROGRAM, snapped)

    def test_ast_statement_spans_snap_cleanly(self):
        import ast

        tree = ast.parse(PROGRAM)
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt):
                continue
            char_span = char_span_of_lines(PROGRAM, node.lineno, node.end_lineno)
            snapped = snap_to_lines(PROGRAM, char_span)
            self._assert_line_aligned(PROGRAM, snapped)


class TestSpanTableLength(unittest.TestCase):
    def test_span_table_length(self):
        backend = CharBackend()
        samples = [PROGRAM, "x = 1\n", "", "▁ café — em dash\n", "a" * 200]
        for text in samples:
            canvas = Canvas.build(text, backend)
            self.assertEqual(
                len(canvas.char_spans), len(canvas.ids) - 1,
                f"mismatch for sample {text!r}",
            )


if __name__ == "__main__":
    unittest.main()
