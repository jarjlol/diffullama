"""solution.py -- the ONLY file Arbor may edit.

Implement select(item) -> int: the index of the candidate to submit as the answer.

`item` is one question answered by one diffusion LM:

    {"uid": "gsm8k-test-0042", "model": "diffullama", "question": "...",
     "candidates": [{"text": "...", "answer": "18" | None,
                     "features": {"len_chars": 412,
                                  "ar.gpt2.nll_sum": 41.2, "ar.gpt2.n_tokens": 63,
                                  "ar.<scorer>.nll_sum": ..., "dlm.selfscore": ...}}, ...]}

No gold answer is ever passed in. Use only the question, the candidate texts,
their extracted answers and the cached features. The loop must stay CPU-only:
features needing a GPU are precomputed by ../pipeline/score_candidates.py.

IDEA 4.1 (verification-weighted voting): majority voting over extracted answers,
but each candidate contributes weight 1.0 only if every explicit calculation in
its trimmed text re-evaluates correctly (and it contains at least one checkable
calculation); otherwise it contributes 0.5. A wrong majority often rests on
samples with an arithmetic slip that a deterministic checker sees. Select the
answer with the highest total weight; ties -> the answer whose earliest
candidate index is smallest; all answers None -> 0.
"""
from __future__ import annotations

import re

_NUM = r"-?[\d,]+(?:\.\d+)?"

# The candidate's own answer: everything before it starts a new 'Question:'
# (same trim as task.trim, reimplemented here since solution.py is standalone).
_MARKERS = re.compile(r"\n\s*(?:Question|Q)\s*:", re.I)

# Explicit calculations. One alternation, annotated form first so a
# <<a op b = c>> is matched whole before the plain form could match inside it.
#   annotated: << num op num = num >>   (op in + - * /)
#   plain:     num op num = num         (op in + - * / x ×)
_CALC = re.compile(
    r"<<\s*(" + _NUM + r")\s*([+\-*/])\s*(" + _NUM + r")\s*=\s*(" + _NUM + r")\s*>>"
    r"|(" + _NUM + r")\s*([+\-*/x×])\s*(" + _NUM + r")\s*=\s*(" + _NUM + r")"
)


def _trim(text: str) -> str:
    return _MARKERS.split(text or "", maxsplit=1)[0].strip()


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _verified(text: str) -> bool:
    """True iff the candidate has >=1 checkable calculation and all of them pass."""
    checked = 0
    for m in _CALC.finditer(_trim(text)):
        if m.group(1) is not None:  # annotated form
            a, op, b, stated = m.group(1), m.group(2), m.group(3), m.group(4)
        else:                       # plain form
            a, op, b, stated = m.group(5), m.group(6), m.group(7), m.group(8)
        av, bv, sv = _num(a), _num(b), _num(stated)
        if av is None or bv is None or sv is None:
            continue  # unparsable pair: skip
        if op == "/":
            if bv == 0:
                continue  # division by zero: skip
            expected = av / bv
        elif op in ("*", "x", "×"):
            expected = av * bv
        elif op == "+":
            expected = av + bv
        else:
            expected = av - bv
        checked += 1
        if abs(expected - sv) >= 1e-6:
            return False  # arithmetic slip
    return checked >= 1


def select(item: dict) -> int:
    scores: dict[str, list] = {}  # answer -> [total_weight, first_candidate_index]
    for i, c in enumerate(item["candidates"]):
        a = c.get("answer")
        if a is None:
            continue
        w = 1.0 if _verified(c["text"]) else 0.5
        rec = scores.get(a)
        if rec is None:
            scores[a] = [w, i]
        else:
            rec[0] += w
    if not scores:
        return 0
    return min(scores.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[1][1]
