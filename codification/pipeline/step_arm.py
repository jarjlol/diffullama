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

from common import HERE, TASK, WORK, read_jsonl, task

sys.path.insert(0, str(TASK))
sys.path.insert(0, str(HERE.parent))
import baselines  # noqa: E402  (protected selector code, imported read-only)
from paired_stats import binom_two_sided, boot_ci  # noqa: E402  (same bootstrap CI and exact McNemar as iteration 1)

BUDGETS = {16: "@k16", 32: "@k32", 64: ""}


def load(model: str, tag: str, split: str = "test") -> dict:
    d = WORK / "candidates" / f"{model}{tag}"
    p = d / f"{split}.jsonl"
    if not p.exists():
        return {}
    rows = {r["uid"]: r for r in read_jsonl(p)}
    fdir = WORK / "features" / f"{model}{tag}"
    for f in sorted(fdir.glob(f"{split}.*.jsonl")) if fdir.exists() else []:
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


def correct(rows: dict, uids: list, f, gold: dict) -> list:
    """Per question: is the answer chosen by selector f right? (None choice counts as wrong.)"""
    out = []
    for u in uids:
        cands = rows[u]["candidates"]; i = f(cands)
        out.append(bool(i is not None and task.is_correct(cands[i].get("answer"), gold[u])))
    return out


def paired(a: list, b: list) -> dict:
    """a - b on the same questions: mean difference, 95% bootstrap CI (10 000, seed 0), exact McNemar p."""
    d = [int(x) - int(y) for x, y in zip(a, b)]
    w, l = sum(v > 0 for v in d), sum(v < 0 for v in d)
    lo, hi = boot_ci(d)
    return {"diff": sum(d) / len(d), "ci95": [lo, hi], "wins": w, "losses": l, "mcnemar_p": binom_two_sided(w, w + l)}


def fmt_pair(label: str, r: dict) -> str:
    return (f"  {label:44s} {r['diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  "
            f"wins/losses {r['wins']}/{r['losses']}  McNemar p={r['mcnemar_p']:.3f}")


def k256_table(m: str, gold: dict, out: dict) -> None:
    """K = 256 majority runs (the first 100 test questions) next to K = 64 and the greedy fidelity runs,
    all on the SAME 100 questions. Written by the queue jobs generate_candidates.py --steps 256 --limit 100."""
    rows256 = load(m, "@k256")
    if not rows256:
        print(f"\n== {m} K = 256: no candidates yet"); return
    rows64 = load(m, "")
    uids = sorted(set(rows256) & set(rows64))
    fid = {}
    for name in ("dream-sampler", "dream-published", "llada-published"):
        p = WORK / "fidelity" / f"{name}.jsonl"
        if p.exists() and name.startswith(m.split("-")[0]):
            fid[name] = {r["uid"]: r["correct"] for r in read_jsonl(p)}
    uids = [u for u in uids if all(u in v for v in fid.values())]
    sel256, sel64 = selectors({u: rows256[u] for u in uids}), selectors({u: rows64[u] for u in uids})
    names = sorted((set(sel256) & set(sel64)))
    cor = {(K, n): correct(rows, uids, (sel256 if K == 256 else sel64)[n], gold)
           for K, rows in ((64, rows64), (256, rows256)) for n in names}
    cor[(256, "oracle")] = [any(task.is_correct(c.get("answer"), gold[u]) for c in rows256[u]["candidates"]) for u in uids]
    cor[(64, "oracle")] = [any(task.is_correct(c.get("answer"), gold[u]) for c in rows64[u]["candidates"]) for u in uids]
    res = {"n": len(uids), "acc": {f"K{K}:{n}": sum(v) / len(uids) for (K, n), v in cor.items()},
           "greedy": {k: sum(v[u] for u in uids) / len(uids) for k, v in fid.items()}, "paired": {}}
    print(f"\n== {m}  K = 256 vs K = 64 on the {len(uids)} questions with a K = 256 run (test; N = 10, T 0.7)")
    print(f"  {'':8s} " + " ".join(f"{n[:14]:>14s}" for n in names + ['oracle']))
    for K in (64, 256):
        print(f"  K={K:<5d} " + " ".join(f"{sum(cor[(K, n)]) / len(uids):14.3f}" for n in names + ["oracle"]))
    for k, v in res["greedy"].items():
        print(f"  greedy single sample, {k}: {v:.3f}  (config {k}, greedy, 256 steps)")
    for n in ("first", "majority") + tuple(x for x in names if x.startswith("ar.") and x.endswith(".sum")):
        r = paired(cor[(256, n)], cor[(64, n)]); res["paired"][f"K256-K64:{n}"] = r
        print(fmt_pair(f"{n}: K=256 minus K=64", r))
    r = paired(cor[(256, "majority")], cor[(256, "first")]); res["paired"]["K256:majority-first"] = r
    print(fmt_pair("K=256: majority minus first sample", r))
    for k, v in fid.items():
        r = paired(cor[(256, "majority")], [bool(v[u]) for u in uids]); res["paired"][f"K256 majority - greedy {k}"] = r
        print(fmt_pair(f"K=256 majority minus greedy ({k})", r))
    out[f"{m}@k256"] = res


def sel_by_K(rows: dict, uids: list) -> dict:
    return selectors({u: rows[u] for u in uids})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    ap.add_argument("--split", choices=["dev", "test"], default="test",
                    help="dev = the step arm on the first 200 dev questions (extra credit); the K = 256 table is test only")
    ap.add_argument("--json")
    a = ap.parse_args()
    gold = {q["uid"]: q["answer"] for q in read_jsonl(WORK / "data" / f"{a.split}.jsonl")}
    out = {}
    for m in a.models:
        runs = {K: load(m, tag, a.split) for K, tag in BUDGETS.items()}
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
        pr = {}
        for k1, k2 in zip(Ks, Ks[1:]):
            for n in ("first", "majority"):
                pr[f"{n}: K={k2} minus K={k1}"] = paired(correct(runs[k2], uids, sel_by_K(runs[k2], uids)[n], gold),
                                                        correct(runs[k1], uids, sel_by_K(runs[k1], uids)[n], gold))
        for K in Ks:
            pr[f"K={K}: majority minus first"] = paired(correct(runs[K], uids, sel_by_K(runs[K], uids)["majority"], gold),
                                                        correct(runs[K], uids, sel_by_K(runs[K], uids)["first"], gold))
        out[m]["paired"] = pr

        print(f"\n== {m}  ({a.split}, n = {len(uids)} questions common to K = {Ks})")
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
        print("  paired differences on the common questions (95% bootstrap CI, exact McNemar):")
        for label, r in out[m]["paired"].items():
            print(fmt_pair(label, r))
    for m in (a.models if a.split == "test" else []):
        k256_table(m, gold, out)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
