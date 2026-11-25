"""Convert OpenRouter MBPP results into feature files."""
from __future__ import annotations

import argparse

from common import WORK, read_jsonl, write_jsonl
from decision_rerank import LETTERS

HERE = WORK / "mbpp"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    got = {r["uid"]: r["probs"] for r in read_jsonl(a.results) if "probs" in r}
    rows = read_jsonl(HERE / "candidates" / a.model / "test.jsonl")
    out = []
    for r in rows:
        probs = got.get(r["uid"])
        if probs is None:
            continue
        n = len(r["candidates"])
        assert set(probs) == set(LETTERS[:n]), f"{r['uid']}: {sorted(probs)} vs {n}"
        out.append({"uid": r["uid"],
                    "features": [{"decision.listwise.p": float(probs[L])} for L in LETTERS[:n]]})
    p = HERE / "features" / a.model / "test.decision-listwise.jsonl"
    write_jsonl(p, out)
    print(f"{a.model}: {len(out)}/{len(rows)} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
