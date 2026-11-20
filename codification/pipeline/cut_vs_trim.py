"""Iteration 2, Results 2: does AR reranking still lose to majority vote when the scorer sees the CUT text?

    python3 cut_vs_trim.py --model dream-base --split test
    python3 cut_vs_trim.py --model llada-base --split test --json out.json

Reads work/candidates/<model>/<split>.jsonl (the iteration-1 64-step candidates) and every feature file next to it,
including the `--cut` re-scoring (`ar.<scorer>-cut.*`, written by `score_candidates.py --cut`). For every scorer that
has both a trimmed and a cut version it prints the accuracy of the summed-NLL and per-token selectors on both texts,
next to first / majority / oracle, and paired differences (mean difference, 95% bootstrap CI, exact McNemar, the same
functions as paired_stats.py): cut minus trim, and each selector minus majority. Nothing is interpreted.
Gold comes from work/data/<split>.jsonl; the Arbor task pack is not touched, no `assemble` step.
"""
from __future__ import annotations

import argparse
import json
import sys

from common import HERE, TASK, WORK, read_jsonl, task

sys.path.insert(0, str(TASK))
sys.path.insert(0, str(HERE.parent))
import baselines  # noqa: E402  (protected selector code, imported read-only)
from paired_stats import binom_two_sided, boot_ci  # noqa: E402


def load(model: str, split: str) -> dict:
    rows = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / model / f"{split}.jsonl")}
    fdir = WORK / "features" / model
    for f in sorted(fdir.glob(f"{split}.*.jsonl")) if fdir.exists() else []:
        for r in read_jsonl(f):
            if r["uid"] in rows:
                for c, feat in zip(rows[r["uid"]]["candidates"], r["features"]):
                    c.setdefault("features", {}).update(feat)
    for r in rows.values():
        for c in r["candidates"]:
            c.setdefault("features", {})
    return rows


def scorer_names(rows: dict) -> list[str]:
    return sorted({k[len("ar."):-len(".nll_sum")] for r in rows.values() for c in r["candidates"]
                   for k in c["features"] if k.startswith("ar.") and k.endswith(".nll_sum")})


def paired(a: list, b: list) -> dict:
    d = [int(x) - int(y) for x, y in zip(a, b)]
    w, l = sum(v > 0 for v in d), sum(v < 0 for v in d)
    lo, hi = boot_ci(d)
    return {"diff": sum(d) / len(d), "ci95": [lo, hi], "wins": w, "losses": l, "mcnemar_p": binom_two_sided(w, w + l)}


def fmt(label: str, r: dict) -> str:
    return (f"  {label:46s} {r['diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  "
            f"wins/losses {r['wins']}/{r['losses']}  McNemar p={r['mcnemar_p']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--json")
    a = ap.parse_args()
    gold = {q["uid"]: q["answer"] for q in read_jsonl(WORK / "data" / f"{a.split}.jsonl")}
    rows = load(a.model, a.split)
    names = scorer_names(rows)
    pairs = [s for s in names if not s.endswith("-cut") and f"{s}-cut" in names]
    # only questions where every compared selector produced a pick, so all columns share the same n
    uids = sorted(rows)
    sel = {"first": lambda c: 0, "majority": baselines._majority}
    for s in names:
        sel[f"{s}:sum"] = lambda c, s=s: baselines._argbest(c, f"ar.{s}.nll_sum")
        sel[f"{s}:mean"] = lambda c, s=s: baselines._mean_nll(c, s)
    cor = {}
    for n, f in sel.items():
        v = []
        for u in uids:
            cands = rows[u]["candidates"]; i = f(cands)
            v.append(bool(i is not None and task.is_correct(cands[i].get("answer"), gold[u])))
        cor[n] = v
    oracle = [any(task.is_correct(c.get("answer"), gold[u]) for c in rows[u]["candidates"]) for u in uids]
    N = len(uids)
    print(f"{a.model} {a.split}: {N} questions, 64-step candidates (iteration 1), N = 10")
    print(f"  first {sum(cor['first']) / N:.3f}  majority {sum(cor['majority']) / N:.3f}  oracle {sum(oracle) / N:.3f}")
    print(f"  scorers with a cut version: {pairs or 'none yet'};  all scorers: {names}")
    for s in names:
        have = sum(all(c["features"].get(f"ar.{s}.nll_sum") is not None for c in rows[u]["candidates"]) for u in uids)
        print(f"  coverage {s}: {have}/{N} questions" + ("   <-- INCOMPLETE, numbers for this scorer are not comparable" if have < N else ""))
    out = {"n": N, "first": sum(cor["first"]) / N, "majority": sum(cor["majority"]) / N, "oracle": sum(oracle) / N, "acc": {}, "paired": {}}
    print(f"  {'selector':32s} {'accuracy':>9s}")
    for n in sorted(k for k in cor if k not in ("first", "majority")):
        out["acc"][n] = sum(cor[n]) / N
        print(f"  {n:32s} {out['acc'][n]:9.3f}")
    print("paired differences (95% bootstrap CI, exact McNemar):")
    for s in pairs:
        for kind in ("sum", "mean"):
            r = paired(cor[f"{s}-cut:{kind}"], cor[f"{s}:{kind}"]); out["paired"][f"{s} {kind}: cut - trim"] = r
            print(fmt(f"{s} {kind}: cut minus trim", r))
    for n in sorted(k for k in cor if k not in ("first", "majority")):
        r = paired(cor[n], cor["majority"]); out["paired"][f"{n} - majority"] = r
        print(fmt(f"{n} minus majority", r))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
