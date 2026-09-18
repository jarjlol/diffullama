"""Verdict + trace -> a typed Evidence value. See docs/01-architecture.md Sec 7.1.

| Verdict kind    | Primary signal                                                     |
|-----------------|----------------------------------------------------------------------|
| SYNTAX_ERROR    | `SyntaxError.lineno` / `offset` -- direct, most reliable signal      |
| ERROR           | innermost frame of the first failing test, inside the candidate file |
| FAIL            | Ochiai over the test spectra, execution count as tie-break           |
| TIMEOUT         | the most-executed line -- a runaway loop's body                      |
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping

from egr.types import Evidence, Observation, Task, Verdict

log = logging.getLogger(__name__)


def derive(obs: Observation, program: str, task: Task) -> Evidence:
    if obs.verdict is Verdict.SYNTAX_ERROR:
        return _from_syntax(obs)
    if obs.verdict is Verdict.TIMEOUT:
        return _from_timeout(obs)
    if obs.verdict is Verdict.ERROR:
        return _from_exception(obs)
    if obs.verdict is Verdict.FAIL:
        return _from_assertion(obs)
    raise ValueError(f"no evidence to derive for verdict {obs.verdict}")


def ochiai(spectra: Mapping[int, tuple[int, int]], total_failed: int) -> dict[int, float]:
    """score(line) = failed(line) / sqrt(total_failed * (failed(line) + passed(line))).

    Canonical spectrum-based fault localization: no training, degrades gracefully with a
    single failing test, gives a ranking rather than a heuristic guess.
    """
    if total_failed == 0:
        return {}
    scores = {}
    for line, (n_pass, n_fail) in spectra.items():
        denom = math.sqrt(total_failed * (n_fail + n_pass))
        scores[line] = (n_fail / denom) if denom else 0.0
    return scores


def _rank(scores: Mapping[int, float], exec_counts: Mapping[int, int]) -> list[int]:
    """Ochiai score first; tests spectra ties broken by execution count, then -- because an
    unspecified tie-break is a reproducibility bug -- by line number, deterministically.
    """
    return sorted(scores, key=lambda ln: (-scores[ln], -exec_counts.get(ln, 0), ln))


def _signature(kind: str, lines: tuple[int, ...], summary: str) -> str:
    """Stable hash driving no-progress detection (H-4): identical evidence twice in a row
    means the last repair changed nothing that mattered.
    """
    payload = f"{kind}|{lines}|{summary}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _from_syntax(obs: Observation) -> Evidence:
    lineno, _offset, msg = obs.syntax_error
    summary = f"SyntaxError at line {lineno}: {msg}"
    return Evidence(
        kind="syntax", lines=(lineno,), scores={lineno: 1.0},
        summary=summary, signature=_signature("syntax", (lineno,), msg),
    )


def _from_exception(obs: Observation) -> Evidence:
    failing = [o for o in obs.outcomes if not o.passed]
    first = failing[0] if failing else None
    frames = first.frames if first else ()
    lines = tuple(dict.fromkeys(ln for _fname, ln in reversed(frames)))  # innermost first
    if not lines:
        lines = tuple(sorted(obs.exec_counts, key=lambda ln: -obs.exec_counts[ln]))
    summary = (
        f"{first.exc_type}: {first.message}" if first and first.exc_type
        else "exception, no frame captured inside the candidate file"
    )
    scores = {ln: 1.0 / rank for rank, ln in enumerate(lines, start=1)}
    return Evidence(
        kind="exception", lines=lines, scores=scores,
        summary=summary, signature=_signature("exception", lines, summary),
    )


def _from_assertion(obs: Observation) -> Evidence:
    total_failed = sum(1 for o in obs.outcomes if not o.passed)
    scores = ochiai(obs.spectra, total_failed)
    ranked = tuple(_rank(scores, obs.exec_counts))
    failing = [o for o in obs.outcomes if not o.passed]
    first = failing[0] if failing else None
    summary = f"{first.name} failed: {first.message}" if first else "assertion failed"
    return Evidence(
        kind="assertion", lines=ranked, scores=scores,
        summary=summary, signature=_signature("assertion", ranked[:3], summary),
    )


def _from_timeout(obs: Observation) -> Evidence:
    lines = tuple(sorted(obs.exec_counts, key=lambda ln: -obs.exec_counts[ln]))
    detail = f" (line {lines[0]} ran {obs.exec_counts[lines[0]]}x)" if lines else ""
    summary = "execution exceeded the wall-clock timeout" + detail
    scores = {ln: float(obs.exec_counts.get(ln, 0)) for ln in lines}
    return Evidence(
        kind="timeout", lines=lines, scores=scores,
        summary=summary, signature=_signature("timeout", lines[:1], summary),
    )
