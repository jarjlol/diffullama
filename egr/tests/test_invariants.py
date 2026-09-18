"""H-1 (shift off-by-one) and H-9 (encoding). Written before the selector, per
docs/03-hazards.md -- these must exist and pass before any other module is trusted.
"""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from egr.canvas import Canvas
from egr.tests.helpers import CharBackend
from egr.types import Span

PKG_ROOT = Path(__file__).resolve().parent.parent


def _source_files():
    """Every .py file in the package proper -- excluding tests/ (which is allowed to construct
    off-by-one bugs on purpose, see above) and docs/ (documentation tooling, not the package).
    """
    excluded = {"tests", "docs"}
    for path in PKG_ROOT.rglob("*.py"):
        if excluded & set(path.relative_to(PKG_ROOT).parts):
            continue
        yield path


class TestShiftInvariant(unittest.TestCase):
    def setUp(self):
        self.backend = CharBackend()
        self.program = "abcdefgh"
        self.canvas = Canvas.build(self.program, self.backend)

    def test_shift_invariant(self):
        """Round-trip a known char span through canvas_positions and back; the decoded
        region must be byte-identical to the intended source text.
        """
        char_span = Span(2, 5, "char")  # "cde"
        canvas_span = self.canvas.canvas_positions(char_span)

        # The model's shift means returned[k] == canvas[k + 1] (docs/01-architecture.md
        # Sec 5.1), so token index 2 ("c") must land at canvas position 3, not 2.
        self.assertEqual(canvas_span, Span(3, 6, "canvas"))
        self.assertEqual(self.canvas.decode_span(canvas_span), "cde")

    def test_shift_invariant_fails_without_the_plus_one(self):
        """Demonstrates, by construction, that the naive (unshifted) mapping this test would
        also have accepted is in fact WRONG -- proving the assertion above is discriminating,
        not vacuous. A future edit that deletes the `+ 1` in `canvas_positions` reproduces
        exactly this failure and `test_shift_invariant` above catches it.
        """
        char_span = Span(2, 5, "char")  # "cde"
        naive = Span(char_span.start, char_span.end, "canvas")  # the bug: no shift applied
        decoded_naive = self.canvas.decode_span(naive)
        self.assertNotEqual(decoded_naive, "cde", "naive (unshifted) mapping should be wrong")
        self.assertEqual(decoded_naive, "bcd")  # shifted left by exactly one -- the actual bug

    def test_src_mask_freezes_everything_else(self):
        canvas_span = self.canvas.canvas_positions(Span(2, 5, "char"))
        mask = self.canvas.src_mask([canvas_span])
        for pos in range(len(mask)):
            expected_frozen = not (canvas_span.start <= pos < canvas_span.end)
            self.assertEqual(mask[pos] == 1, expected_frozen, f"position {pos}")
        self.assertEqual(mask[0], 1, "BOS (position 0) must always be frozen")


class TestOnlyOneShiftSite(unittest.TestCase):
    def test_only_one_shift_site(self):
        """P3: no module except canvas.py performs arithmetic on token indices."""
        pattern = re.compile(r"\+\s*1\b")
        offenders = []
        for path in _source_files():
            if path.name == "canvas.py":
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(PKG_ROOT)}:{lineno}: {stripped}")
        self.assertEqual(offenders, [], "index arithmetic (`+ 1`) found outside canvas.py")


class TestUtf8Roundtrip(unittest.TestCase):
    def test_utf8_roundtrip(self):
        program = "x = '▁'  # SentencePiece marker\ny = 'em—dash'\nz = 'café'\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prog.py"
            path.write_text(program, encoding="utf-8")
            read_back = path.read_text(encoding="utf-8")
        self.assertEqual(read_back, program)

        backend = CharBackend()
        canvas = Canvas.build(read_back, backend)
        full_span = Span(1, len(canvas.ids), "canvas")
        self.assertEqual(canvas.decode_span(full_span), program)


if __name__ == "__main__":
    unittest.main()
