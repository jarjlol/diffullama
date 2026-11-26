"""MBPP selector table (exploratory; MBPP was queued after HumanEval)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import WORK, read_jsonl
from decision_eval import bootstrap_ci, mcnemar_exact

HERE = WORK / "mbpp"


def load_features(model: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    d = HERE / "features" / model
    if not d.exists():
        return out
    for f in sorted(d.glob("test.*.jsonl")):
        for r in read_jsonl(f):
            merged = out.setdefault(r["uid"], [{} for _ in r["features"]])
            for j, fs in enumerate(r["features"]):
                merged[j].update(fs)
    return out


def best(feats, key, norm=False):
    vals = []
    for i, f in enumerate(feats):
        v, n = f.get(key), f.get(key.replace("nll_sum", "n_tokens"))
        if v is None or (norm and not n):
            continue
        vals.append((v / n if norm else v, i))
    return min(vals)[1] if vals else None


def top(feats, key):
    vals = [(f.get(key), i) for i, f in enumerate(feats)]
    vals = [(v, i) for v, i in vals if v is not None]
    return max(vals, key=lambda x: (x[0], -x[1]))[1] if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ex = {r["uid"]: r for r in read_jsonl(HERE / "exec" / a.model / "test.jsonl")}
    feats = load_features(a.model)
    rows = []
    for uid in sorted(ex):
        passed = ex[uid]["passed"]
        fs = feats.get(uid, [{} for _ in passed])
        rec = {"uid": uid, "n": len(passed), "oracle": any(passed),
               "random(exp.)": sum(passed) / len(passed), "correct": {}}
        rec["correct"]["first"] = bool(passed[0])
        keys = {k for f in fs for k in f}
        for k in sorted(keys):
            if k.startswith("ar.") and k.endswith(".nll_sum"):
                s = k[len("ar."):-len(".nll_sum")]
                for name, norm in ((f"ar.{s}.sum", False), (f"ar.{s}.mean", True)):
                    j = best(fs, k, norm)
                    rec["correct"][name] = bool(passed[j]) if j is not None else False
            elif k == "decision.listwise.p":
                j = top(fs, k)
                rec["correct"]["decision.listwise"] = bool(passed[j]) if j is not None else False
            elif k == "decision.pointwise.p_yes":
                j = top(fs, k)
                rec["correct"]["decision.pointwise"] = bool(passed[j]) if j is not None else False
        rows.append(rec)

    names = sorted({n for r in rows for n in r["correct"]})
    n = len(rows)
    table = {k: sum(r["correct"][k] for r in rows if k in r["correct"])
             / max(1, sum(1 for r in rows if k in r["correct"])) for k in names}
    table["random(exp.)"] = sum(r["random(exp.)"] for r in rows) / n
    table["oracle"] = sum(r["oracle"] for r in rows) / n
    first = table["first"]
    head = lambda k: (table[k] - first) / (table["oracle"] - first) if table["oracle"] > first else 0.0

    print(f"\n=== MBPP · {n} items · {a.model} ===")
    print(f"{'selector':28s} {'acc':>8s} {'headroom':>9s}")
    for k in names:
        print(f"{k:28s} {table[k]:8.3f} {head(k):9.1%}")
    print(f"{'random(exp.)':28s} {table['random(exp.)']:8.3f}")
    print(f"{'oracle':28s} {table['oracle']:8.3f}")

    def pair(x, y):
        rs = [r for r in rows if x in r["correct"] and y in r["correct"]]
        aa = [r["correct"][x] for r in rs]; bb = [r["correct"][y] for r in rs]
        return aa, bb, sum(aa)/len(aa) - sum(bb)/len(bb)

    tests = {}
    print("\npaired exact McNemar (two-sided)")
    for k in names:
        if k == "first" or table[k] <= first:
            continue
        for base in ("first", "ar.qwen7b.mean"):
            if base not in table or k == base:
                continue
            aa, bb, delta = pair(k, base)
            p, n01, n10 = mcnemar_exact(aa, bb)
            lo, hi = bootstrap_ci(aa, bb)
            tests[f"{k} vs {base}"] = {"delta": delta, "p": p, "ci95": [lo, hi]}
            print(f"  {k:24s} vs {base:16s} delta={delta:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.2e}")
    if a.json:
        Path(a.json).write_text(json.dumps({"n": n, "model": a.model, "table": table, "tests": tests},
                                           indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
