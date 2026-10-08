"""Stdlib-only helpers for the ResearchAgent implementation.

Copied rather than imported from limitations/scripts/common.py: each assignment
directory in this repository is deliberately self-contained (docs/research-workstreams.md),
so a later change to one pipeline cannot silently alter another's results.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDEATION = ROOT / "ideation"
# Overridable so the self-test can run the whole pipeline in a scratch directory.
DATA = Path(os.environ.get("IDEATION_DATA_DIR", IDEATION / "data"))
OUT = Path(os.environ.get("IDEATION_OUT_DIR", IDEATION / "output"))

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "we", "were", "which",
    "with", "our", "can", "these", "those", "than", "then", "such", "also", "but", "not",
    "they", "their", "them", "there", "here", "into", "more", "most", "other", "some", "only",
    "both", "each", "when", "while",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


def clean_pdf_text(text: str) -> str:
    """Rejoin pdftotext line-break hyphenation ('com- pared' -> 'compared') and collapse whitespace."""
    text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


class TfidfCosine:
    """L2-normalised TF-IDF vectors compared by cosine similarity."""

    def __init__(self, corpus_tokens: list[list[str]]):
        self.n_docs = len(corpus_tokens)
        df: Counter[str] = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log((self.n_docs + 1) / (n + 1)) + 1.0 for t, n in df.items()}
        self.vectors = [self.vector(d) for d in corpus_tokens]

    def vector(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec = {t: (c / len(tokens)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def scores(self, query_tokens: list[str]) -> list[float]:
        q = self.vector(query_tokens)
        return [sum(v * vec.get(t, 0.0) for t, v in q.items()) for vec in self.vectors]


def stable_id(*parts: str) -> str:
    """Deterministic short id, so re-runs address the same LLM request file."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


def read_text(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path, content: str) -> None:
    """Atomic write: the file either appears complete or not at all.

    Cached LLM responses are trusted on sight, so a file truncated by a shutdown or
    kill mid-write would otherwise be read back as a valid answer. Write to a
    sibling temp file, fsync, then rename over the target (atomic on POSIX).
    A leftover *.partial is never read as a response and is cleaned by daily_run.sh.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.partial")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
