"""baselines.py -- PROTECTED. Reference selectors the codified idea is compared against.

    python3 baselines.py --split test          # table to stdout
    python3 baselines.py --split test --json   # machine-readable, for the report

Selectors (all choose one of the same N cached candidates per question):
  first           the first sample            -- what a single run of the model gives
  random(exp.)    expected accuracy of a uniformly random candidate (exact, no sampling)
  majority        self-consistency: most frequent extracted answer, ties -> earliest
                  (the anchor paper's own SC baseline, Table 2, used 3 samples)
  dlm.selfscore   highest diffusion-LM self-score (its own confidence), if cached
  ar.<s>.sum      lowest summed NLL under AR scorer <s>       (idea 1.1 as specified)
  ar.<s>.mean     lowest per-token NLL under AR scorer <s>    (length-normalised)
  oracle          any candidate correct (hit@N) -- the ceiling for ANY selector

`headroom` = (selector - first) / (oracle - first): the fraction of the reachable gap
a selector closes. None of these baselines read gold except `oracle`.
"""
from __future__ import annotations

import argparse
import collections
import json

import task


def _majority(cands):
    answers = [c.get("answer") for c in cands]
    counts = collections.Counter(a for a in answers if a is not None)
    if not counts:
        return 0
    best = max(counts.values())
    return next(i for i, a in enumerate(answers) if a is not None and counts[a] == best)


def _argbest(cands, key, lower=True):
    vals = [(c["features"].get(key), i) for i, c in enumerate(cands)]
    vals = [(v, i) for v, i in vals if v is not None]
    if not vals:
        return None
    return (min(vals) if lower else max(vals, key=lambda x: (x[0], -x[1])))[1]


def _mean_nll(cands, s):
    out = []
    for i, c in enumerate(cands):
        f = c["features"]
        tot, n = f.get(f"ar.{s}.nll_sum"), f.get(f"ar.{s}.n_tokens")
        if tot is not None and n:
            out.append((tot / n, i))
    return min(out)[1] if out else None


def run(split: str) -> dict:
    items, gold = task.load_items(split), task.load_gold(split)
    scorers = sorted({k.split(".")[1] for it in items for c in it["candidates"] for k in c["features"] if k.startswith("ar.") and k.endswith(".nll_sum")})
    has_self = any("dlm.selfscore" in c["features"] for it in items for c in it["candidates"])
    sel = {"first": lambda c: 0, "majority": _majority}
    if has_self:
        sel["dlm.selfscore"] = lambda c: _argbest(c, "dlm.selfscore", lower=False)
    for s in scorers:
        sel[f"ar.{s}.sum"] = lambda c, s=s: _argbest(c, f"ar.{s}.nll_sum")
        sel[f"ar.{s}.mean"] = lambda c, s=s: _mean_nll(c, s)

    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    n = collections.Counter()
    for it in items:
        m, g, cands = it["model"], gold[it["uid"]], it["candidates"]
        ok = [task.is_correct(c.get("answer"), g) for c in cands]
        n[m] += 1
        acc[m]["oracle"] += any(ok)
        acc[m]["random(exp.)"] += sum(ok) / len(ok)
        for name, f in sel.items():
            i = f(cands)
            acc[m][name] += ok[i] if i is not None else 0.0
    table = {m: {k: v / n[m] for k, v in acc[m].items()} | {"n": n[m]} for m in sorted(n)}
    for m, r in table.items():
        gap = r["oracle"] - r["first"]
        r["_headroom"] = {k: ((r[k] - r["first"]) / gap if gap > 0 else None)
                          for k in r if k not in ("n", "oracle", "first") and not k.startswith("_")}
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    t = run(a.split)
    if a.json:
        print(json.dumps(t, indent=2)); return
    for m, r in t.items():
        print(f"\n== {m}  (n={r['n']}, split={a.split})")
        for k in sorted(k for k in r if k not in ("n",) and not k.startswith("_")):
            h = r["_headroom"].get(k)
            print(f"  {k:<16} {r[k]:.4f}" + (f"   headroom {h:+.1%}" if h is not None else ""))


if __name__ == "__main__":
    main()
