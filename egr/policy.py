"""The six remasking policies, one registry. See docs/01-architecture.md Sec 8.

Signature note: the architecture doc's illustrative `RemaskPolicy.plan()` pseudocode shows
`(program, ev, canvas, *, budget, scope, rng)`. Implementing `static`/`resample` (which need to
know the entry-point function even though they ignore evidence) and `confidence` (which needs
the backend's `.confidence()`) surfaced two missing-but-necessary parameters. Both are added as
universal keyword-only args every policy receives -- most ignore them -- rather than special-
cased per policy, so `loop.py` still calls every policy identically (P1).
"""
from __future__ import annotations

import ast
import logging
import random
from typing import TYPE_CHECKING, Protocol

from egr.canvas import Canvas, char_span_of_lines, snap_to_lines
from egr.localize import locate, neighbourhood
from egr.types import Evidence, RemaskPlan, Span

if TYPE_CHECKING:
    from egr.backend import DiffusionBackend

log = logging.getLogger(__name__)


class RemaskPolicy(Protocol):
    name: str

    def plan(self, program: str, ev: Evidence, canvas: Canvas, *, budget: int, scope: str,
             slack: int, entry_point: str, backend: "DiffusionBackend",
             rng: random.Random) -> RemaskPlan | None: ...


def _budget_trim(canvas: Canvas, canvas_span: Span, budget: int, slack: int) -> Span:
    """Widens the natural neighbourhood by up to `slack` extra positions past its end -- H-3's
    "hole = span tokens + slack" -- so a fix that needs slightly more tokens than the original
    buggy span occupied (e.g. putting back a dropped `:`) still fits.

    The widened end is snapped OUTWARD to the next whole line via `snap_to_lines`, rather than
    stopped at a raw token count: a first version widened by a flat token count and landed
    partway into the following line, which breaks the same whole-line invariant H-2 exists to
    protect (and, concretely, corrupted MockBackend's oracle splice, which relies on holes
    corresponding to whole lines). Growing in whole-line increments means the actual widening
    can overshoot `slack` tokens; that is preferred over a mid-line cut.

    Also checks a `budget + slack` ceiling, but does NOT enforce it by cutting: cutting an
    oversized span at an arbitrary token offset has the identical mid-line problem. So for
    this POC's scale (small fixture-sized functions, docs/04-build-phases.md Phase 0-2),
    exceeding the ceiling is only logged: "same budget for every policy"
    (docs/01-architecture.md Sec 8) is a target, not a hard cap this function enforces. A
    rigorous line-safe trim is flagged as future work in deploy.md's "known simplifications".

    HARD LIMIT, discovered empirically rather than designed for (deploy.md "known
    simplifications" records it): if the natural neighbourhood already reaches
    `len(canvas.ids)` -- e.g. `function` scope on a short function -- there is no frozen
    canvas left to widen INTO, so `slack` cannot add any room at all, regardless of its
    value. A fix that needs more tokens than the whole current program occupies is then
    genuinely unrepairable at this depth. The real mitigation is building the canvas with
    trailing padding positions reserved for growth, which `Canvas.build()` does not do; adding
    that is future work, not a bug fix, since it changes what a canvas IS.
    """
    widened = canvas_span
    if slack > 0 and canvas_span.end < len(canvas.ids):
        start_token = canvas_span.start - 1
        last_token = min(canvas_span.end + slack, len(canvas.ids)) - 1
        last_token = min(last_token, len(canvas.char_spans) - 1)
        char_lo = canvas.char_spans[start_token].start
        char_hi = canvas.char_spans[last_token].end
        snapped_char_span = snap_to_lines(canvas.program, Span(char_lo, char_hi, "char"))
        widened = canvas.canvas_positions(snapped_char_span)

    limit = max(budget + slack, 1)
    size = widened.end - widened.start
    if size > limit:
        log.debug("neighbourhood span (%d tokens) exceeds budget+slack=%d; not trimming, "
                  "since trimming would cut mid-line", size, limit)
    return widened


def _entry_point_function(program: str, entry_point: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    return None


def _deepest_statement(fn: ast.FunctionDef) -> ast.stmt | None:
    """The most deeply nested statement in `fn`'s body, ties broken by line number. Depth is
    the live recursion-stack length at the point a statement is visited, not an incremented
    counter -- so this needs no index arithmetic at all.
    """
    best: list = []
    stack: list[ast.stmt] = []

    def walk(node: ast.AST) -> None:
        is_stmt = isinstance(node, ast.stmt) and node is not fn
        if is_stmt:
            stack.append(node)
            depth = len(stack)
            if not best or depth > best[0] or (depth == best[0] and node.lineno < best[1].lineno):
                best[:] = [depth, node]
        for child in ast.iter_child_nodes(node):
            walk(child)
        if is_stmt:
            stack.pop()

    walk(fn)
    return best[1] if best else None


class ExecutionGroundedPolicy:
    """`ours`: evidence -> witness -> neighbourhood. The paper's core claim; differs from
    `static` by exactly one thing -- whether the witness came from execution evidence.
    """

    name = "ours"

    def plan(self, program, ev, canvas, *, budget, scope, slack, entry_point, backend, rng) -> RemaskPlan | None:
        char_span = locate(program, ev, scope)
        if char_span is None:
            return None
        canvas_span = _budget_trim(canvas, canvas.canvas_positions(char_span), budget, slack)
        witness_line = ev.lines[0] if ev.lines else "?"
        return RemaskPlan(
            canvas_spans=(canvas_span,), n_masked=canvas_span.end - canvas_span.start,
            scope=scope, rationale=f"execution evidence ({ev.kind}) -> line {witness_line}",
        )


class StaticStructuralPolicy:
    """`static`: AST neighbourhood of a STATICALLY chosen node, no execution. A POC-level
    stand-in for CDC's code-property-graph witness selection: the entry point's most deeply
    nested statement, deterministic and evidence-free. Differs from `ours` by exactly one
    thing -- this witness ignores the failing test entirely.
    """

    name = "static"

    def plan(self, program, ev, canvas, *, budget, scope, slack, entry_point, backend, rng) -> RemaskPlan | None:
        fn = _entry_point_function(program, entry_point)
        if fn is None:
            return None
        witness = _deepest_statement(fn)
        if witness is None:
            return None
        char_span = neighbourhood(program, witness.lineno, scope)
        if char_span is None:
            return None
        canvas_span = _budget_trim(canvas, canvas.canvas_positions(char_span), budget, slack)
        return RemaskPlan(
            canvas_spans=(canvas_span,), n_masked=canvas_span.end - canvas_span.start,
            scope=scope, rationale=f"static witness: deepest-nested stmt at line {witness.lineno}",
        )


class ConfidencePolicy:
    """`confidence`: lowest max-softmax positions, snapped to whole lines.

    Must be built on `backend.confidence()` (additive max-softmax), NEVER on a real model's
    `x0_scores` (H-5) -- that measures sampling luck, not certainty, and would make this
    baseline artificially weak, biasing the comparison in our favour.
    """

    name = "confidence"

    def plan(self, program, ev, canvas, *, budget, scope, slack, entry_point, backend, rng) -> RemaskPlan | None:
        scores = backend.confidence(canvas.ids)
        if len(scores) != len(canvas.ids):
            raise ValueError("backend.confidence() must return one score per canvas position")
        candidates = range(1, len(scores))  # position 0 is BOS, never a candidate
        if not candidates:
            return None
        worst = min(candidates, key=lambda i: scores[i])
        # `char_spans` is indexed by token index; canvas position -> token index is the
        # documented inverse of canvas.py's shift (canvas position = token index + 1), read
        # here rather than recomputed -- see canvas.py's own `decode_span` for the same relation.
        token_idx = worst - 1
        if not (0 <= token_idx < len(canvas.char_spans)):
            return None
        line_span = snap_to_lines(canvas.program, canvas.char_spans[token_idx])
        canvas_span = _budget_trim(canvas, canvas.canvas_positions(line_span), budget, slack)
        return RemaskPlan(
            canvas_spans=(canvas_span,), n_masked=canvas_span.end - canvas_span.start,
            scope=scope, rationale=f"lowest confidence at canvas position {worst} ({scores[worst]:.3f})",
        )


class RandomSpanPolicy:
    """`random`: random lines, matched token budget. Isolates informed localization from an
    uninformed hole of the same size -- baseline B2.
    """

    name = "random"

    def plan(self, program, ev, canvas, *, budget, scope, slack, entry_point, backend, rng) -> RemaskPlan | None:
        n_tokens = len(canvas.ids) - 1  # excludes BOS
        if n_tokens < 1:
            return None
        width = max(min(budget + slack, n_tokens), 1)
        last_start = n_tokens - width
        start_token = rng.randint(0, last_start) if last_start > 0 else 0
        lo = canvas.char_spans[start_token].start
        hi = canvas.char_spans[start_token + width - 1].end
        char_span = snap_to_lines(canvas.program, Span(lo, hi, "char"))
        canvas_span = _budget_trim(canvas, canvas.canvas_positions(char_span), budget, slack)
        return RemaskPlan(
            canvas_spans=(canvas_span,), n_masked=canvas_span.end - canvas_span.start,
            scope=scope, rationale=f"random span, target width={width} tokens",
        )


class FullResamplePolicy:
    """`resample`: mask the entire function body. The harshest baseline -- "just try again" --
    and a fair one, since targeted remasking saves no compute (model.py forwards the full
    sequence regardless of mask size, docs/01-architecture.md README Sec 7).
    """

    name = "resample"

    def plan(self, program, ev, canvas, *, budget, scope, slack, entry_point, backend, rng) -> RemaskPlan | None:
        fn = _entry_point_function(program, entry_point)
        if fn is None:
            return None
        end = getattr(fn, "end_lineno", None) or fn.lineno
        char_span = char_span_of_lines(program, fn.lineno, end)
        canvas_span = canvas.canvas_positions(char_span)
        return RemaskPlan(
            canvas_spans=(canvas_span,), n_masked=canvas_span.end - canvas_span.start,
            scope="function", rationale="full function-body resample",
        )


class NoRepairPolicy:
    """`none`: never repairs. B0's single-shot baseline -- the loop stops after depth 0
    because `plan()` always signals "no maskable region."
    """

    name = "none"

    def plan(self, program, ev, canvas, *, budget, scope, slack, entry_point, backend, rng) -> RemaskPlan | None:
        return None


POLICIES: dict[str, type] = {
    "ours": ExecutionGroundedPolicy,
    "static": StaticStructuralPolicy,
    "confidence": ConfidencePolicy,
    "random": RandomSpanPolicy,
    "resample": FullResamplePolicy,
    "none": NoRepairPolicy,
}
