"""Convert OpenRouter HumanEval results into feature files."""
from __future__ import annotations

import argparse

from common import WORK, read_jsonl, write_jsonl
from decision_rerank import LETTERS

HERE = WORK / "humaneval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--mode", choices=["listwise", "pointwise"], default="listwise")
    a = ap.parse_args()

    if a.mode == "pointwise":
        got = {}
        for r in read_jsonl(a.results):
            if "noul" in r:
                got.setdefault(r["uid"], {})[r["idx"]] = r["noul"]
        rows = read_jsonl(HERE / "candidates" / a.model / f"{a.split}.jsonl")
        out = []
        for r in rows:
            ps = got.get(r["uid"])
            if ps is None:
                continue
            n = len(r["candidates"])
            assert set(ps) == set(range(n)), f"{r['uid']}: {sorted(ps)} vs n={n}"
            out.append({"uid": r["uid"],
                        "features": [{"decision.pointwise.p_yes": float(ps[j])} for j in range(n)]})
        p = HERE / "features" / a.model / f"{a.split}.decision-pointwise.jsonl"
        write_jsonl(p, out)
        print(f"{a.model}: {len(out)}/{len(rows)} pointwise -> {p}")
        return 0

    got = {r["uid"]: r["probs"] for r in read_jsonl(a.results) if "probs" in r}
    rows = read_jsonl(HERE / "candidates" / a.model / f"{a.split}.jsonl")
    out = []
    for r in rows:
        probs = got.get(r["uid"])
        if probs is None:
            continue
        n = len(r["candidates"])
        assert set(probs) == set(LETTERS[:n]), f"{r['uid']}: keys {sorted(probs)} vs n={n}"
        out.append({"uid": r["uid"],
                    "features": [{"decision.listwise.p": float(probs[L])} for L in LETTERS[:n]]})
    p = HERE / "features" / a.model / f"{a.split}.decision-listwise.jsonl"
    write_jsonl(p, out)
    print(f"{a.model}: {len(out)}/{len(rows)} items -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
