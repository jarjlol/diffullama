"""Iteration 2, prompt A1 -- the step-budget arm: reranking gain vs gain from more denoising steps.

    python3 step_arm.py                      # table to stdout
    python3 step_arm.py --json out.json      # also write the numbers

Reads work/candidates/<model>{,@k16,@k32}/test.jsonl and the matching feature files directly
(no assemble step, so the Arbor task pack is never touched). Uses only the test questions
present at every step budget, so all budgets are compared on the same questions.

Idea 1.1's dependent variable (ideation/output/research_problems_and_ideas.md, idea 1.1 §4):
    dQ_rerank(K)        = Q_selector(K) - Q_first(K)
    dQ_steps(K1 -> K2)  = Q_first(K2) - Q_first(K1)
    A_rerank            = dQ_rerank(16) - dQ_steps(16 -> 32)
reported here for majority vote and for the AR selectors. Compute is reported next to it:
mean generation seconds per question (all N = 10 candidates) at each K, and the per-sample
step count. Note the spec's comparison holds K fixed but a selector needs all N samples while
"first" needs one; see the per-sample columns before calling reranking "more efficient".
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from common import TASK, WORK, read_jsonl, task

sys.path.insert(0, str(TASK))
import baselines  # noqa: E402  (protected selector code, imported read-only)

BUDGETS = {16: "@k16", 32: "@k32", 64: ""}


def load(model: str, tag: str) -> dict:
    d = WORK / "candidates" / f"{model}{tag}"
    p = d / "test.jsonl"
    if not p.exists():
        return {}
    rows = {r["uid"]: r for r in read_jsonl(p)}
    fdir = WORK / "features" / f"{model}{tag}"
    for f in sorted(fdir.glob("test.*.jsonl")) if fdir.exists() else []:
        for r in read_jsonl(f):
            if r["uid"] in rows:
                for c, feat in zip(rows[r["uid"]]["candidates"], r["features"]):
                    c.setdefault("features", {}).update(feat)
    for r in rows.values():
        for c in r["candidates"]:
            c.setdefault("features", {})
    return rows


def selectors(rows: dict) -> dict:
    scorers = sorted({k[len("ar."):-len(".nll_sum")] for r in rows.values() for c in r["candidates"]
                      for k in c["features"] if k.startswith("ar.") and k.endswith(".nll_sum")})
    sel = {"first": lambda c: 0, "majority": baselines._majority}
    for s in scorers:
        sel[f"ar.{s}.sum"] = lambda c, s=s: baselines._argbest(c, f"ar.{s}.nll_sum")
        sel[f"ar.{s}.mean"] = lambda c, s=s: baselines._mean_nll(c, s)
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    ap.add_argument("--json")
    a = ap.parse_args()
    gold = {q["uid"]: q["answer"] for q in read_jsonl(WORK / "data" / "test.jsonl")}
    out = {}
    for m in a.models:
        runs = {K: load(m, tag) for K, tag in BUDGETS.items()}
        runs = {K: r for K, r in runs.items() if r}
        if len(runs) < 2:
            print(f"{m}: need at least two step budgets, have {sorted(runs)}"); continue
        uids = sorted(set.intersection(*(set(r) for r in runs.values())))
        res = {}
        for K, rows in sorted(runs.items()):
            sel = selectors({u: rows[u] for u in uids})
            acc = {}
            for name, f in sel.items():
                hit = 0
                for u in uids:
                    cands = rows[u]["candidates"]
                    i = f(cands)
                    hit += task.is_correct(cands[i].get("answer"), gold[u]) if i is not None else 0
                acc[name] = hit / len(uids)
            acc["oracle"] = sum(any(task.is_correct(c.get("answer"), gold[u]) for c in rows[u]["candidates"])
                                for u in uids) / len(uids)
            secs = [rows[u]["gen_seconds"] for u in uids if "gen_seconds" in rows[u]]
            res[K] = {"acc": acc, "gen_s_per_q": statistics.mean(secs) if secs else None}
        Ks = sorted(res)
        names = sorted(set.intersection(*(set(res[K]["acc"]) for K in Ks)) - {"first", "oracle"})
        d_steps = {f"{k1}->{k2}": res[k2]["acc"]["first"] - res[k1]["acc"]["first"] for k1, k2 in zip(Ks, Ks[1:])}
        d_rerank = {K: {n: res[K]["acc"][n] - res[K]["acc"]["first"] for n in names} for K in Ks}
        a_rerank = ({n: d_rerank[16][n] - d_steps["16->32"] for n in names}
                    if 16 in res and 32 in res else {})
        out[m] = {"n": len(uids), "budgets": res, "dQ_steps": d_steps, "dQ_rerank": d_rerank, "A_rerank": a_rerank}

        print(f"\n== {m}  (test, n = {len(uids)} questions common to K = {Ks})")
        print(f"  {'K':>4} {'first':>7} {'major':>7} {'oracle':>7}  best AR (name)            s/q (N=10)")
        for K in Ks:
            r = res[K]["acc"]
            ar = {n: v for n, v in r.items() if n.startswith("ar.")}
            best = max(ar, key=ar.get) if ar else None
            s = res[K]["gen_s_per_q"]
            best_s = f"{ar[best]:.3f} ({best})" if best else "--"
            print(f"  {K:>4} {r['first']:7.3f} {r['majority']:7.3f} {r['oracle']:7.3f}  "
                  f"{best_s:<26} {f'{s:.1f}' if s else '--'}")
        print("  dQ_steps (first sample): " + ", ".join(f"{k} {v:+.3f}" for k, v in d_steps.items()))
        print("  dQ_rerank, majority:     " + ", ".join(f"K={K} {d_rerank[K]['majority']:+.3f}" for K in Ks))
        if a_rerank:
            top = sorted(a_rerank.items(), key=lambda kv: -kv[1])
            print("  A_rerank = dQ_rerank(16) - dQ_steps(16->32): " + ", ".join(f"{n} {v:+.3f}" for n, v in top[:4]))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
