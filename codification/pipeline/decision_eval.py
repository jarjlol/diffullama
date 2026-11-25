"""Offline comparison of every selector on the cached GSM8K candidates.

    python3 decision_eval.py --split test
    python3 decision_eval.py --split test --models dream-base llada-base --json result.json

Reads candidates from work/candidates/<model>/<split>.jsonl and merges every
work/features/<model>/<split>.*.jsonl into each candidate. Selectors are built
from whatever features are present: ar.<scorer>.sum/.mean and
nimble.listwise/.pointwise. Gold comes from ../rerank_task/data/<split>/gold.jsonl
through the protected task.py; no selector ever sees it.

Prints per-model and macro-average accuracy, headroom over `first`, paired exact
McNemar against `majority` and the best AR selector, and a paired bootstrap CI
for the two pre-registered comparisons.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

from common import TASK, WORK, read_jsonl, task

sys.path.insert(0, str(TASK))


def load_gold(split: str) -> dict[str, str]:
    return {r["uid"]: r["answer"] for r in read_jsonl(TASK / "data" / split / "gold.jsonl")}


def load_features(model: str, split: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    d = WORK / "features" / model
    if not d.exists():
        return out
    for f in sorted(d.glob(f"{split}.*.jsonl")):
        if ".nimble-" in f.name and "shuffle" in f.name:
            continue
        for r in read_jsonl(f):
            merged = out.setdefault(r["uid"], [{} for _ in r["features"]])
            for j, fs in enumerate(r["features"]):
                merged[j].update(fs)
    return out


def majority(cands: list[dict]) -> int:
    answers = [c.get("answer") for c in cands]
    counts = collections.Counter(a for a in answers if a is not None)
    if not counts:
        return 0
    best = max(counts.values())
    return next(i for i, a in enumerate(answers) if a is not None and counts[a] == best)


def argmin(cands: list[dict], key: str, normalise: bool = False) -> int | None:
    vals = []
    for i, c in enumerate(cands):
        v = c["features"].get(key)
        n = c["features"].get(key.replace("nll_sum", "n_tokens"))
        if v is None or not math.isfinite(v) or (normalise and (not n or not math.isfinite(n))):
            continue
        vals.append((v / n if normalise else v, i))
    return min(vals)[1] if vals else None


def argmax(cands: list[dict], key: str) -> int | None:
    vals = [(c["features"].get(key), i) for i, c in enumerate(cands)]
    vals = [(v, i) for v, i in vals if v is not None and math.isfinite(v)]
    return max(vals, key=lambda x: (x[0], -x[1]))[1] if vals else None


def selectors(cands: list[dict], feat_keys: set[str]) -> dict:
    ar = sorted({k[len("ar."):-len(".nll_sum")] for k in feat_keys
                 if k.startswith("ar.") and k.endswith(".nll_sum")})
    sel = {
        "first": lambda c: 0,
        "majority": majority,
    }
    for s in ar:
        sel[f"ar.{s}.sum"] = lambda c, s=s: argmin(c, f"ar.{s}.nll_sum")
        sel[f"ar.{s}.mean"] = lambda c, s=s: argmin(c, f"ar.{s}.nll_sum", normalise=True)
    for k in sorted(feat_keys):
        if k.startswith("nimble.") or k.startswith("decision."):
            name = "decision.listwise" if k == "decision.listwise.p" else (
                   "decision.pointwise" if k == "decision.pointwise.p_yes" else k[:-2])
            sel[name] = lambda c, k=k: argmax(c, k)
        elif k == "qwen.choice.p":
            sel["qwen.choice"] = lambda c, k=k: argmax(c, k)
        elif k == "qwen.yes.p":
            sel["qwen.yes"] = lambda c, k=k: argmax(c, k)
    return sel


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[float, int, int]:
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), n01, n10


def bootstrap_ci(a: list[bool], b: list[bool], seed: int = 0, n: int = 10_000) -> tuple[float, float]:
    rng = random.Random(seed)
    d = [int(x) - int(y) for x, y in zip(a, b)]
    N = len(d)
    means = []
    for _ in range(n):
        means.append(sum(d[rng.randrange(N)] for _ in range(N)) / N)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--json", default=None, help="write the full table + tests here")
    ap.add_argument("--dump", default=None, help="write per-item predictions here")
    a = ap.parse_args()

    models = a.models or sorted(p.name for p in (WORK / "candidates").iterdir()
                                if (p / f"{a.split}.jsonl").exists())
    gold = load_gold(a.split)
    rows, per_item = [], {}
    for m in models:
        cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / m / f"{a.split}.jsonl")}
        feats = load_features(m, a.split)
        keys = {k for fs in feats.values() for f in fs for k in f}
        sel = selectors(next(iter(cands.values()))["candidates"], keys)
        for uid in sorted(cands):
            r = cands[uid]
            fs_all = feats.get(uid)
            if fs_all is not None and len(fs_all) != len(r["candidates"]):
                raise SystemExit(f"{m}/{uid}: {len(fs_all)} feature rows vs "
                                 f"{len(r['candidates'])} candidates")
            cs = []
            for j, c in enumerate(r["candidates"]):
                fs = fs_all[j] if fs_all is not None else {}
                cs.append({**c, "features": fs})
            g = gold[uid]
            ok = [task.is_correct(c.get("answer"), g) for c in cs]
            rec = {"uid": uid, "model": m, "n": len(cs), "oracle": any(ok),
                   "random(exp.)": sum(ok) / len(ok), "correct": {}}
            rec["present"] = {}
            for name, f in sel.items():
                i = f(cs)
                rec["present"][name] = i is not None
                if i is not None:
                    rec["correct"][name] = bool(ok[i])
                rec.setdefault("choice", {})[name] = i
            per_item[(m, uid)] = rec
            rows.append(rec)

    names = sorted({n for r in rows for n in r["correct"]})
    table = {}
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        table[m] = {n: (sum(r["correct"][n] for r in rs if n in r["correct"])
                        / sum(1 for r in rs if r["present"].get(n, False)))
                    if any(r["present"].get(n, False) for r in rs) else None for n in names}
        table[m]["coverage"] = {n: sum(1 for r in rs if r["present"].get(n, False)) / len(rs)
                                for n in names}
        table[m]["random(exp.)"] = sum(r["random(exp.)"] for r in rs) / len(rs)
        table[m]["oracle"] = sum(r["oracle"] for r in rs) / len(rs)
        table[m]["n"] = len(rs)
    macro = {n: sum(table[m][n] for m in models if table[m][n] is not None)
             / sum(1 for m in models if table[m][n] is not None) for n in names}
    macro["random(exp.)"] = sum(table[m]["random(exp.)"] for m in models) / len(models)
    macro["oracle"] = sum(table[m]["oracle"] for m in models) / len(models)

    first = macro["first"]
    head = lambda n: (macro[n] - first) / (macro["oracle"] - first) if macro["oracle"] > first else 0.0

    print(f"\n=== {a.split} · {len(rows)} items · {', '.join(models)} ===")
    print(f"{'selector':28s} " + " ".join(f"{m:>11s}" for m in models) + f" {'macro':>8s} {'headroom':>9s}")
    for n in names:
        cells = " ".join(f"{table[m][n]:11.3f}" if table[m][n] is not None else f"{'-':>11s}"
                         for m in models)
        if n.startswith("decision."):
            cov = min(table[m]["coverage"][n] for m in models if table[m][n] is not None)
            cells += f"  cov={cov:.2f}"
        print(f"{n:28s} {cells} {macro[n]:8.3f} {head(n):9.1%}")
    print(f"{'random(exp.)':28s} " + " ".join(f"{table[m]['random(exp.)']:11.3f}" for m in models)
          + f" {macro['random(exp.)']:8.3f}")
    print(f"{'oracle':28s} " + " ".join(f"{table[m]['oracle']:11.3f}" for m in models)
          + f" {macro['oracle']:8.3f}")

    # Stacked per-item correctness: with equal n per model this equals the macro average.
    def pair(n: str, base: str) -> tuple[list[bool], list[bool], float]:
        rs = [r for r in rows if n in r["correct"] and base in r["correct"]]
        aa = [r["correct"][n] for r in rs]
        bb = [r["correct"][base] for r in rs]
        return aa, bb, sum(aa) / len(aa) - sum(bb) / len(bb)

    # Pre-registered secondary comparator; do not pick the AR variant on test.
    best_ar = "ar.qwen7b-cut.sum" if "ar.qwen7b-cut.sum" in names else None
    tests = {}
    print("\npaired exact McNemar (two-sided), stacked over models")
    for n in names:
        if n in ("first", "majority") or macro[n] <= macro["first"]:
            continue
        for base in ["majority"] + ([best_ar] if best_ar and best_ar != n else []):
            aa, bb, delta = pair(n, base)
            p, n01, n10 = mcnemar_exact(aa, bb)
            lo, hi = bootstrap_ci(aa, bb)
            tests[f"{n} vs {base}"] = {"delta": delta, "p": p,
                                      "n01": n01, "n10": n10, "ci95": [lo, hi]}
            print(f"  {n:24s} vs {base:22s} delta={delta:+.3f} "
                  f"CI[{lo:+.3f},{hi:+.3f}] p={p:.2e} ({n01}+ / {n10}-)")
            for m in models:
                rsm = [r for r in rows if r["model"] == m and n in r["correct"] and base in r["correct"]]
                if not rsm:
                    continue
                am = [r["correct"][n] for r in rsm]; bm = [r["correct"][base] for r in rsm]
                pm, _, _ = mcnemar_exact(am, bm)
                tests[f"{n} vs {base} [{m}]"] = {"delta": sum(am)/len(am) - sum(bm)/len(bm),
                                                 "p": pm, "n": len(am)}
                print(f"    [{m:10s}] delta={tests[f'{n} vs {base} [{m}]']['delta']:+.3f} "
                      f"p={pm:.2e} n={len(am)}")

    # Holm over the pre-registered family for decision.listwise (H1, H2).
    family = [f"decision.listwise vs {b}" for b in (["majority"] + ([best_ar] if best_ar else []))
              if f"decision.listwise vs {b}" in tests]
    if family:
        ordered = sorted(family, key=lambda k: tests[k]["p"])
        m = len(ordered); prev = 0.0
        for i, k in enumerate(ordered):
            adj = min(1.0, max(prev, tests[k]["p"] * (m - i)))
            tests[k]["p_holm"] = adj; prev = adj
        print("  Holm (H1,H2): " + ", ".join(f"{k.split(' vs ')[1]} p_adj={tests[k]['p_holm']:.2e}"
                                            for k in ordered))

    if a.json:
        Path(a.json).write_text(json.dumps({"split": a.split, "models": models, "table": table,
                                            "macro": macro, "tests": tests}, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for (m, uid), r in sorted(per_item.items()):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
