"""task.py -- PROTECTED ground truth and grading. Do not edit during a research run.

GSM8K answer extraction and grading, plus data loading. Kept apart from eval.py so
the data pipeline (../pipeline/) grades with exactly the same code.

Grading is numeric: the final answer is extracted from a candidate and compared
with the gold number after normalisation (commas, '$', trailing '.', and '18.0'
vs '18' all normalise to the same value).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
_MARKERS = re.compile(r"\n\s*(?:Question|Q)\s*:", re.I)


def normalize(s: str | None) -> str | None:
    """'1,200' -> '1200'; '$18.00' -> '18'; '-3.50' -> '-3.5'; non-numbers -> None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return str(int(v)) if v == int(v) else repr(round(v, 6)).rstrip("0").rstrip(".")


def trim(text: str) -> str:
    """The candidate's own answer: everything before it starts a new 'Question:'.

    Shared by grading and by the AR scorer, so a candidate is scored on exactly the
    text it is graded on.
    """
    return _MARKERS.split(text or "", maxsplit=1)[0].strip()


def extract_answer(text: str) -> str | None:
    """Final numeric answer of ONE candidate completion.

    A few-shot prompted model often keeps going and writes the next exemplar
    ('Question: ...'); everything from there on is discarded. Then, in order:
    the GSM8K '#### N' marker, an explicit 'answer is N', else the last number.
    """
    if not text:
        return None
    body = trim(text)
    for pat in (rf"####\s*({_NUM})", rf"answer is\s*:?\s*({_NUM})"):
        m = re.search(pat, body, re.I)
        if m:
            return normalize(m.group(1))
    nums = re.findall(_NUM, body)
    return normalize(nums[-1]) if nums else None


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_items(split: str) -> list[dict]:
    """One record per (model, question): question + candidates + features. No gold."""
    return _jsonl(DATA / split / "items.jsonl")


def load_gold(split: str) -> dict[str, str]:
    return {r["uid"]: r["answer"] for r in _jsonl(DATA / split / "gold.jsonl")}
