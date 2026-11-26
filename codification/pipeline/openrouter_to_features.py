"""Convert OpenRouter decision results into pipeline feature files.

    python3 openrouter_to_features.py --results work/openrouter/test.results.jsonl

Writes work/features/<model>/test.decision-listwise.jsonl, aligned 1:1 with the
candidates. Asserts that every candidate has exactly one probability and that
the 10 codes are the expected A..J.
"""
from __future__ import annotations

import argparse
import collections

from common import WORK, read_jsonl, write_jsonl
from decision_rerank import LETTERS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--mode", choices=["listwise", "pointwise"], default="listwise")
    a = ap.parse_args()

    rows = read_jsonl(a.results)
    errors = [r for r in rows if "error" in r]
    if a.mode == "pointwise":
        rows = [r for r in rows if "noul" in r]
        by_model = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in rows:
            by_model[r["model"]][r["uid"]][r["idx"]] = r["noul"]
        for model, got in sorted(by_model.items()):
            cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / model / f"{a.split}.jsonl")}
            out = []
            for uid in sorted(cands):
                ps = got.get(uid)
                if ps is None:
                    continue
                n = len(cands[uid]["candidates"])
                assert set(ps) == set(range(n)), f"{uid}: {sorted(ps)} vs n={n}"
                out.append({"uid": uid,
                            "features": [{"decision.pointwise.p_yes": float(ps[j])} for j in range(n)]})
            p = WORK / "features" / model / f"{a.split}.decision-pointwise.jsonl"
            write_jsonl(p, out)
            print(f"{model}: {len(out)}/{len(cands)} pointwise -> {p} ({len(errors)} errors)")
        return 0

    rows = [r for r in rows if "probs" in r]
    by_model = collections.defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["uid"]] = r["probs"]

    for model, got in sorted(by_model.items()):
        cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / model / f"{a.split}.jsonl")}
        missing = sorted(set(cands) - set(got))
        extra = sorted(set(got) - set(cands))
        print(f"{model}: {len(got)} results, {len(cands)} candidates, "
              f"{len(missing)} missing, {len(extra)} extra, {len(errors)} errors")
        out = []
        for uid in sorted(cands):
            probs = got.get(uid)
            if probs is None:
                continue
            n = len(cands[uid]["candidates"])
            assert set(probs) == set(LETTERS[:n]), f"{uid}: prob keys {sorted(probs)} vs n={n}"
            out.append({"uid": uid,
                        "features": [{"decision.listwise.p": float(probs[L])} for L in LETTERS[:n]]})
        p = WORK / "features" / model / f"{a.split}.decision-listwise.jsonl"
        write_jsonl(p, out)
        print(f"  wrote {len(out)} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
