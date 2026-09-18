"""THE index authority: text <-> tokens <-> canvas. See docs/01-architecture.md Sec 5.

P3 (docs/01-architecture.md Sec 1): no module except this one performs arithmetic on token
indices. Everything else passes around `Span` objects. `egr/tests/test_invariants.py::
test_only_one_shift_site` greps the package to enforce it.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from egr.types import Span

if TYPE_CHECKING:
    from egr.backend import DiffusionBackend
    from egr.types import Evidence

log = logging.getLogger(__name__)


def snap_to_lines(text: str, char_span: Span) -> Span:
    """Expand a char span outward to whole line boundaries, indentation included.

    Chosen deliberately over token-partial masking (docs/01-architecture.md Sec 5.2): Python
    indentation is semantic, and masking part of a line's leading whitespace asks the model to
    reproduce an exact indent it can only half see. Closes H-2 (indentation bleed).
    """
    if char_span.kind != "char":
        raise ValueError(f"snap_to_lines wants a char span, got {char_span.kind!r}")
    start = text.rfind("\n", 0, char_span.start) + 1  # 0 if no newline before start
    end = text.find("\n", char_span.end)
    end = len(text) if end == -1 else end + 1  # include the line's own trailing newline
    return Span(start, end, "char")


def char_span_of_lines(text: str, start_line: int, end_line: int) -> Span:
    """1-indexed, inclusive line range (as `ast` reports them) -> a char Span covering exactly
    those whole lines. This is char/line bookkeeping, not token-index arithmetic, so it is
    exempt from the one-site rule above.
    """
    offsets, pos = [], 0
    for line in text.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(line)
    offsets.append(pos)
    lo = max(start_line - 1, 0)
    hi = min(end_line, len(offsets) - 1)
    return Span(offsets[lo], offsets[hi], "char")


def render_annotation(evidence: "Evidence", style: str) -> str:
    """Channel 2: render the failure as a frozen comment block placed above the program.

    An ablation, not an assumption (docs/01-architecture.md Sec 4/8) -- whether a base,
    non-instruction-tuned model exploits this is an open empirical question, so it ships
    behind `style` and is measured, never assumed to help.
    """
    if style == "none" or evidence is None:
        return ""
    lines = [f"# FAILED: {evidence.summary}"]
    if style == "comment":
        top = sorted(evidence.scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        for line_no, score in top:
            lines.append(f"# line {line_no}: score={score:.2f}")
    elif style != "compact":
        raise ValueError(f"unknown annotate style {style!r}")
    return "\n".join(lines) + "\n"


@dataclass
class Canvas:
    program: str             # program text INCLUDING any rendered annotation prefix
    ids: list[int]            # [BOS] + program tokens
    char_spans: list[Span]    # one per program token; char_spans[i] <-> canvas position i + 1
    bos_id: int

    @classmethod
    def build(
        cls,
        program: str,
        backend: "DiffusionBackend",
        *,
        annotate: str = "none",
        evidence: "Evidence | None" = None,
    ) -> "Canvas":
        prefix = render_annotation(evidence, annotate) if evidence is not None else ""
        text = prefix + program
        ids, char_spans = backend.tokenize(text)
        ids, char_spans = list(ids), list(char_spans)
        if len(char_spans) != len(ids):
            # H-11: hand-built positional arrays have burned this project three times.
            # Lengths are asserted here, at construction, rather than trusted.
            raise ValueError(
                f"tokenizer returned {len(ids)} ids but {len(char_spans)} char spans"
            )
        canvas_ids = [backend.bos_id] + ids
        log.debug("built canvas: %d program tokens, %d canvas positions", len(ids), len(canvas_ids))
        return cls(program=text, ids=canvas_ids, char_spans=char_spans, bos_id=backend.bos_id)

    def canvas_positions(self, char_span: Span) -> Span:
        """char span -> canvas positions.

        `+ 1` here is the model's inherited autoregressive shift (model.py:120, model.py:156):
        `returned[k] == canvas[k + 1]`, proven exactly in docs/01-architecture.md Sec 5.1 and by
        `egr/tests/test_shift_offset_reference.py`. This is the ONLY place it appears.
        """
        if char_span.kind != "char":
            raise ValueError(f"canvas_positions wants a char span, got {char_span.kind!r}")
        overlapping = [
            i for i, s in enumerate(self.char_spans)
            if s.start < char_span.end and s.end > char_span.start
        ]
        if not overlapping:
            raise ValueError(f"no token overlaps char span {char_span}")
        lo, hi = min(overlapping), max(overlapping)
        start = lo + 1
        end = hi + 2  # hi's own canvas position is hi + 1; end is exclusive, so one more
        return Span(start, end, "canvas")

    def src_mask(self, holes: Sequence[Span]) -> list[int]:
        """1 = frozen, 0 = the model may write here. Position 0 (BOS) is always frozen."""
        mask = [1] * len(self.ids)
        for hole in holes:
            if hole.kind != "canvas":
                raise ValueError(f"src_mask wants canvas spans, got {hole.kind!r}")
            for pos in range(max(hole.start, 1), min(hole.end, len(mask))):
                mask[pos] = 0
        return mask

    def decode_span(self, canvas_span: Span) -> str:
        """Inverse of `canvas_positions`, used only by tests to check round-trips."""
        if canvas_span.kind != "canvas":
            raise ValueError(f"decode_span wants a canvas span, got {canvas_span.kind!r}")
        lo, hi = canvas_span.start - 1, canvas_span.end - 1  # canvas position -> token index
        spans = self.char_spans[lo:hi]
        if not spans:
            return ""
        return self.program[spans[0].start:spans[-1].end]
