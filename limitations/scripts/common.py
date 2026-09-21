"""Shared stdlib-only text utilities for the multi-agent limitation pipeline.

Implements the retrieval primitives that arXiv:2601.11578 specifies as BM25 (sparse)
+ FAISS (dense).  FAISS and sentence-transformers are unavailable in this build
environment, so the dense half is substituted with a hand-written TF-IDF cosine
ranker.  The substitution is deliberate and is documented in REPORT.md 3.2 --
it follows the precedent already set by litreview/scripts/common.py.

Pure standard library.  No numpy, no sklearn, no faiss.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- tokenisation

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "we", "were", "which", "with", "our", "can", "these", "those", "than", "then",
    "such", "also", "but", "not", "they", "their", "them", "there", "here", "into",
    "more", "most", "other", "some", "only", "both", "each", "when", "while",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _WORD.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


# ------------------------------------------------------------------ BM25 (sparse)

class BM25:
    """Okapi BM25.  Standard parameters k1=1.5, b=0.75."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df: Counter[str] = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {
            term: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def score(self, query_tokens: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], self.doc_len[idx]
        total = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += self.idf.get(term, 0.0) * f * (self.k1 + 1.0) / denom
        return total

    def scores(self, query_tokens: list[str]) -> list[float]:
        return [self.score(query_tokens, i) for i in range(self.n_docs)]


# ------------------------------------------------- TF-IDF cosine (dense substitute)

class TfidfCosine:
    """L2-normalised TF-IDF vectors compared by cosine similarity.

    Stands in for the paper's FAISS dense retriever.  It is a lexical-semantic
    approximation, not an embedding model: it cannot match paraphrases that share
    no vocabulary.  REPORT.md 3.2 records this as the single largest fidelity gap
    in the retrieval stage.
    """

    def __init__(self, corpus_tokens: list[list[str]]):
        self.n_docs = len(corpus_tokens)
        df: Counter[str] = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log((self.n_docs + 1) / (n + 1)) + 1.0 for t, n in df.items()}
        self.vectors = [self._vector(d) for d in corpus_tokens]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec = {t: (c / len(tokens)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def scores(self, query_tokens: list[str]) -> list[float]:
        q = self._vector(query_tokens)
        if not q:
            return [0.0] * self.n_docs
        out = []
        for vec in self.vectors:
            if len(q) > len(vec):
                small, large = vec, q
            else:
                small, large = q, vec
            out.append(sum(v * large.get(t, 0.0) for t, v in small.items()))
        return out


# ------------------------------------------------------------------ score fusion

def minmax(values: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]; a constant vector maps to all zeros."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def hybrid_fuse(sparse: list[float], dense: list[float], weight: float = 0.5) -> list[float]:
    """Equal-weight fusion of normalised sparse and dense scores (paper 3.2)."""
    s, d = minmax(sparse), minmax(dense)
    return [weight * a + (1.0 - weight) * b for a, b in zip(s, d)]


# ------------------------------------------------------------------- section split

_SECTION = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*)\s+)?([A-Z][A-Za-z0-9 \-&/,:]{2,70})\s*$"
)


def split_sections(text: str, min_chars: int = 200) -> list[dict]:
    """Split a paper's plain text into (heading, body) chunks.

    The paper stores each section of each cited paper as an individual chunk
    (paper 3.2).  Headings are detected heuristically from the pdftotext output.
    """
    lines = text.splitlines()
    sections, current = [], {"heading": "FRONT MATTER", "lines": []}
    for line in lines:
        m = _SECTION.match(line)
        if m and len(line.strip()) < 72 and not line.strip().endswith("."):
            if current["lines"]:
                sections.append(current)
            current = {"heading": line.strip(), "lines": []}
        else:
            current["lines"].append(line)
    if current["lines"]:
        sections.append(current)

    out = []
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        if len(body) >= min_chars:
            out.append({"heading": s["heading"], "text": body})
    return out


# -------------------------------------------------------------------------- io

def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
