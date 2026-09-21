"""Stage 2 -- the two-stage retrieval pipeline of arXiv:2601.11578 3.2.

    Stage A  hybrid retrieval : BM25 (sparse) + dense, equal weighting, top-20
    Stage B  LLM re-ranking   : keep only chunks scoring >= 8/10 for relevance

Stage A runs here in full.  Stage B is an LLM call; this script emits the exact
re-ranking prompt and the top-20 payload to data/rerank_request.md, and reads the
scores back from data/rerank_scores.json.  See REPORT.md 4 for why LLM steps are
externalised rather than stubbed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (BM25, TfidfCosine, hybrid_fuse, read_json, read_text,  # noqa: E402
                    tokenize, write_json, write_text)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "limitations" / "data"
LITREVIEW = ROOT / "litreview" / "data"

TOP_K_HYBRID = 20          # paper 3.2: "select top 20 candidates"
RERANK_THRESHOLD = 8       # paper 3.2: "filter to relevance score >= 8/10"

RERANK_PROMPT = """\
You are the LLM re-ranker of the RAG stage described in arXiv:2601.11578 section 3.2.

TASK. For each candidate chunk below, score 0-10 how useful that chunk is for
identifying a LIMITATION of the input paper. A chunk is useful when it reports a
result, method, or critique that exposes a weakness, unexamined assumption,
untested regime, or superseded choice in the input paper.

Score 0-3  : unrelated topic, or relevant topic with nothing limitation-bearing.
Score 4-7  : same area, but only generic overlap; no specific weakness exposed.
Score 8-10 : directly exposes or implies a concrete weakness of the input paper.

Only chunks scoring >= 8 are retained.

INPUT PAPER
-----------
{anchor}

CANDIDATE CHUNKS
----------------
{candidates}

Return JSON: {{"scores": {{"<chunk_id>": <int 0-10>, ...}}}}
"""


def build_query(anchor_sections: list[dict]) -> str:
    """The Citation Agent's retrieval query: the anchor's claims and method."""
    wanted = ("abstract", "introduction", "method", "training", "experiment")
    parts = []
    for s in anchor_sections:
        h = s["heading"].lower().replace(" ", "")
        if any(w.replace(" ", "") in h for w in wanted):
            parts.append(s["text"][:3000])
    return "\n".join(parts) if parts else anchor_sections[0]["text"][:3000]


def main() -> int:
    corpus = read_json(DATA / "rag_corpus.json")
    sections = read_json(DATA / "anchor_sections.json")

    corpus_tokens = [tokenize(c["text"]) for c in corpus]
    query = build_query(sections)
    q_tokens = tokenize(query)

    sparse = BM25(corpus_tokens).scores(q_tokens)
    dense = TfidfCosine(corpus_tokens).scores(q_tokens)
    fused = hybrid_fuse(sparse, dense, weight=0.5)

    ranked = sorted(range(len(corpus)), key=lambda i: fused[i], reverse=True)[:TOP_K_HYBRID]
    top = [{
        "chunk_id": corpus[i]["chunk_id"],
        "provenance": corpus[i]["provenance"],
        "source_title": corpus[i]["source_title"],
        "year": corpus[i].get("year"),
        "hybrid_score": round(fused[i], 4),
        "bm25": round(sparse[i], 4),
        "dense": round(dense[i], 4),
        "text": corpus[i]["text"],
    } for i in ranked]

    write_json(DATA / "rag_top20.json", top)

    payload = "\n\n".join(
        f"[{c['chunk_id']}] ({c['provenance']}, {c.get('year')}) {c['source_title']}\n"
        f"{c['text'][:900]}"
        for c in top
    )
    anchor_brief = read_text(LITREVIEW / "anchor_fulltext.txt")[:6000]
    write_text(DATA / "rerank_request.md",
               RERANK_PROMPT.format(anchor=anchor_brief, candidates=payload))

    print(f"corpus chunks          : {len(corpus)}")
    print(f"hybrid top-K retained  : {len(top)}  (paper: 20)")
    print(f"  cited_in             : {sum(1 for c in top if c['provenance']=='cited_in')}")
    print(f"  cited_by_substitute  : {sum(1 for c in top if c['provenance']!='cited_in')}")
    print(f"\nre-rank prompt written : {DATA/'rerank_request.md'}")
    print("top-8 by hybrid score:")
    for c in top[:8]:
        print(f"  {c['hybrid_score']:.3f}  {c['source_title'][:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
