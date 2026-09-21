"""Stage 2b -- apply the LLM re-ranker's scores and the paper's >= 8/10 filter."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_json, write_json  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "limitations" / "data"
THRESHOLD = 8


def main() -> int:
    top = read_json(DATA / "rag_top20.json")
    scored = read_json(DATA / "rerank_scores.json")
    scores = scored["scores"]

    missing = [c["chunk_id"] for c in top if c["chunk_id"] not in scores]
    if missing:
        raise SystemExit(f"re-ranker did not score {len(missing)} chunks: {missing}")
    assert len(scores) == len(top), f"score/candidate count mismatch: {len(scores)} vs {len(top)}"

    retained = []
    for c in top:
        s = scores[c["chunk_id"]]
        if s >= THRESHOLD:
            retained.append({**c, "rerank_score": s,
                             "justification": scored["justifications"].get(c["chunk_id"], "")})
    retained.sort(key=lambda c: -c["rerank_score"])
    write_json(DATA / "rag_retained.json", retained)

    print(f"scored          : {len(scores)}")
    print(f"retained (>= {THRESHOLD}) : {len(retained)}")
    print(f"dropped         : {len(top) - len(retained)}")
    print()
    for c in retained:
        print(f"  {c['rerank_score']:>2}/10  [{c['provenance']:<20}] {c['source_title'][:62]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
