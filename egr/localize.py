"""Evidence -> witness AST node -> neighbourhood -> char Span. See docs/01-architecture.md
Sec 7.3. Line-range spans from `ast` are already whole-line aligned by construction, so H-2's
snap-to-whole-lines policy falls out of this module for free -- nothing here masks a partial
line.

Correctness note: `ast.parse()` on the same text twice returns two graphs of DISTINCT node
objects, so an ancestor lookup keyed by node identity silently returns nothing if the witness
node and the parent-map came from different parses. `neighbourhood()` therefore does its own
single parse internally and never accepts an `ast.stmt` handed in from elsewhere -- caught by
`egr/tests/test_localize.py::test_scopes_differ_when_nesting_allows_it`.
"""
from __future__ import annotations

import ast
import logging
from typing import Protocol

from egr.canvas import char_span_of_lines
from egr.types import Evidence, Span

log = logging.getLogger(__name__)

SCOPES = ("leaf", "parent_leaf", "function")


class Localizer(Protocol):
    def witness(self, program: str, ev: Evidence) -> ast.stmt | None: ...


def _smallest_stmt_containing(tree: ast.AST, target_line: int) -> ast.stmt | None:
    """The smallest `ast.stmt` whose line range contains `target_line`."""
    best: tuple[int, ast.stmt] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        if node.lineno <= target_line <= end:
            width = end - node.lineno
            if best is None or width < best[0]:
                best = (width, node)
    return best[1] if best else None


class AstLocalizer:
    """The witness is the smallest `ast.stmt` whose line range contains the top-ranked line
    in `ev.lines`. Standalone convenience wrapper -- `locate()` below does not call this, to
    avoid the cross-parse identity trap described at the top of this file.
    """

    name = "ast"

    def witness(self, program: str, ev: Evidence) -> ast.stmt | None:
        if not ev.lines:
            return None
        try:
            tree = ast.parse(program)
        except SyntaxError:
            return None
        return _smallest_stmt_containing(tree, ev.lines[0])


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_stmt(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.stmt | None:
    """The nearest strictly-ancestor statement -- what CDC's Fig. 8(b) calls "Parent+Leaf".

    If `node` is a direct child of a function body (not nested in an `if`/`for`/`while`), its
    nearest statement ancestor is the enclosing `FunctionDef` itself, so `parent_leaf`
    degenerates to `function` scope for top-level statements. Documented, not special-cased --
    a POC-scoped simplification.
    """
    cur = parents.get(node)
    while cur is not None and not isinstance(cur, ast.stmt):
        cur = parents.get(cur)
    return cur


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.stmt | None:
    cur = node
    while cur is not None:
        if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef):
            return cur
        cur = parents.get(cur)
    return None


def _fallback_neighbourhood(program: str, target_line: int, scope: str) -> Span | None:
    """Used when `program` does not parse at all -- which, for a `syntax` evidence kind, is
    the common case (verify.py's own gate already establishes `ast.parse` fails). There is no
    AST to walk, so this is a text-only heuristic: grow by whole lines outward (`parent_leaf`)
    or out to the nearest enclosing `def`/`async def` line found by prefix scan (`function`).

    Coarser than the AST-based path, but keeps a base diffusion LM's most common failure mode
    (malformed output) inside the repair loop instead of aborting on every syntax error.
    """
    lines = program.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return None
    target_line = max(1, min(target_line, n))

    if scope == "leaf":
        lo = hi = target_line
    elif scope == "parent_leaf":
        lo, hi = target_line, target_line
        if lo > 1:
            lo -= 1
        if hi < n:
            hi += 1
    else:  # "function"
        lo = target_line
        while lo > 1 and not lines[lo - 2].lstrip().startswith(("def ", "async def ")):
            lo -= 1
        hi = target_line
        while hi < n and not lines[hi].lstrip().startswith(("def ", "async def ")):
            hi += 1

    return char_span_of_lines(program, lo, hi)


def neighbourhood(program: str, target_line: int, scope: str) -> Span | None:
    """target_line + scope -> a whole-line char Span covering the neighbourhood.

    Single parse, used for both the witness lookup and the ancestor walk -- see the module
    docstring for why that matters.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}, expected one of {SCOPES}")

    try:
        tree = ast.parse(program)
    except SyntaxError:
        return _fallback_neighbourhood(program, target_line, scope)

    witness = _smallest_stmt_containing(tree, target_line)
    if witness is None:
        log.debug("no statement contains line %d", target_line)
        return None

    if scope == "leaf":
        target = witness
    elif scope == "parent_leaf":
        parents = _parent_map(tree)
        target = _enclosing_stmt(witness, parents) or witness
    else:  # "function"
        parents = _parent_map(tree)
        target = _enclosing_function(witness, parents) or witness

    start = target.lineno
    end = getattr(target, "end_lineno", None) or start
    return char_span_of_lines(program, start, end)


def locate(program: str, ev: Evidence, scope: str) -> Span | None:
    """evidence -> neighbourhood, combined. The single entry point `policy.py` uses.

    Returns `None` when no maskable region could be derived -- e.g. no evidence lines, or the
    top-ranked line no longer exists in `program`. `RemaskPolicy.plan()` treats that as a
    first-class outcome, not an error.
    """
    if not ev.lines:
        return None
    return neighbourhood(program, ev.lines[0], scope)


_NEXT_SCOPE = dict(zip(SCOPES, (*SCOPES[1:], None), strict=True))


def escalate(scope: str) -> str | None:
    """leaf -> parent_leaf -> function -> None (ladder exhausted). Used by loop.py's
    no-progress detector (H-4).
    """
    if scope not in _NEXT_SCOPE:
        raise ValueError(f"unknown scope {scope!r}, expected one of {SCOPES}")
    return _NEXT_SCOPE[scope]
