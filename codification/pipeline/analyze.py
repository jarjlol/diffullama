"""Step 7 -- does representation preservation predict reranking headroom? (idea 1.1's claim)

    python3 analyze.py --split test

Joins work/preservation.json (mean per-layer CKA, DLM vs its AR base) with the
baseline table for the same split (baselines.run), and reports, per AR scorer:

    model  CKA  first  oracle  best-AR-acc  headroom

plus Spearman's rho between CKA and headroom across models, with an exact
permutation p-value. With ~4 models that have an AR base, rho can only take a
handful of values and p >= 1/24: report the numbers, never call it significant.
LLaDA (no AR base) is listed separately as the control.
"""
from __future__ import annotations

import argparse
import itertools
import json

from common import TASK, WORK

import sys
sys.path.insert(0, str(TASK))
import baselines  # noqa: E402


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def perm_p(x, y):
    """Two-sided exact permutation p-value for Spearman's rho."""
    obs = abs(spearman(x, y)); perms = list(itertools.permutations(y))
    return sum(abs(spearman(x, list(p))) >= obs - 1e-12 for p in perms) / len(perms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    pres = json.loads((WORK / "preservation.json").read_text(encoding="utf-8"))
    table = baselines.run(a.split)
    scorers = sorted({k.split(".")[1] for r in table.values() for k in r if k.startswith("ar.")})
    report = {"split": a.split, "per_scorer": {}}
    # Both variants are reported; picking the better one per model on the same split would be selection on test.
    for s in scorers:
        for key in (f"ar.{s}.sum", f"ar.{s}.mean"):
            rows = [{"model": m, "cka": (pres.get(m) or {}).get("mean"), "first": r["first"],
                     "majority": r["majority"], "oracle": r["oracle"], "ar_acc": r[key],
                     "headroom": r["_headroom"].get(key)} for m, r in table.items() if key in r]
            with_base = [x for x in rows if x["cka"] is not None and x["headroom"] is not None]
            out = {"rows": rows, "n_models": len(with_base)}
            if len(with_base) >= 3:
                xs, ys = [x["cka"] for x in with_base], [x["headroom"] for x in with_base]
                out["spearman"], out["perm_p"] = spearman(xs, ys), perm_p(xs, ys)
                # Robustness: the same correlation under each CKA variant (debiased, without position 0).
                out["by_cka_variant"] = {}
                for v in sorted({v for x in with_base for v in (pres[x["model"]].get("variants") or {})}):
                    pts = [((pres[x["model"]]["variants"].get(v) or {}).get("mean"), x["headroom"]) for x in with_base]
                    pts = [p for p in pts if p[0] is not None]
                    if len(pts) >= 3:
                        out["by_cka_variant"][v] = {"spearman": spearman(*zip(*pts)), "perm_p": perm_p(*map(list, zip(*pts))), "n": len(pts)}
            report["per_scorer"][key] = out
    if a.json:
        print(json.dumps(report, indent=2)); return 0
    for s, out in report["per_scorer"].items():
        print(f"\n== selector {s}  (split {a.split})")
        print(f"  {'model':<12} {'CKA':>6} {'first':>6} {'major':>6} {'AR':>6} {'oracle':>6} {'headroom':>9}")
        for x in out["rows"]:
            cka = f"{x['cka']:.3f}" if x["cka"] is not None else "  -- "
            h = f"{x['headroom']:+.1%}" if x["headroom"] is not None else "--"
            print(f"  {x['model']:<12} {cka:>6} {x['first']:6.3f} {x['majority']:6.3f} {x['ar_acc']:6.3f} {x['oracle']:6.3f} {h:>9}")
        if "spearman" in out:
            print(f"  Spearman(CKA, headroom) = {out['spearman']:+.3f}, exact permutation p = {out['perm_p']:.3f}, "
                  f"n = {out['n_models']} models  -- descriptive only at this n")
            for v, r in out.get("by_cka_variant", {}).items():
                print(f"    with {v:<22} rho = {r['spearman']:+.3f}, p = {r['perm_p']:.3f}, n = {r['n']}")
        else:
            print(f"  only {out['n_models']} models with both CKA and headroom: no correlation reported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
