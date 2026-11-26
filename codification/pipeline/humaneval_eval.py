"""Score every HumanEval selector against execution results.

    python3 humaneval_eval.py --models dream-base llada-base
    python3 humaneval_eval.py --json he_result.json --dump he_preds.jsonl

Ground truth is `passed[j]` from humaneval_exec.py: the official test run in a
subprocess. Selectors never see it. `example-filter` uses only the docstring
examples in the prompt, so it is a legitimate baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import WORK, read_jsonl
from decision_eval import bootstrap_ci, mcnemar_exact

HERE = WORK / "humaneval"


def load_exec(model: str, split: str) -> dict[str, dict]:
    p = HERE / "exec" / model / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p}; run humaneval_exec.py --score {model} first")
    return {r["uid"]: r for r in read_jsonl(p)}


def load_features(model: str, split: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    d = HERE / "features" / model
    if not d.exists():
        return out
    for f in sorted(d.glob(f"{split}.*.jsonl")):
        for r in read_jsonl(f):
            merged = out.setdefault(r["uid"], [{} for _ in r["features"]])
            for j, fs in enumerate(r["features"]):
                merged[j].update(fs)
    return out


def best(feats: list[dict], key: str, normalise: bool = False) -> int | None:
    vals = []
    for i, f in enumerate(feats):
        v = f.get(key)
        n = f.get(key.replace("nll_sum", "n_tokens"))
        if v is None or (normalise and not n):
            continue
        vals.append((v / n if normalise else v, i))
    return min(vals)[1] if vals else None


def top(feats: list[dict], key: str) -> int | None:
    vals = [(f.get(key), i) for i, f in enumerate(feats)]
    vals = [(v, i) for v, i in vals if v is not None]
    return max(vals, key=lambda x: (x[0], -x[1]))[1] if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()

    rows = []
    for m in a.models:
        ex = load_exec(m, a.split)
        feats = load_features(m, a.split)
        mbr_path = HERE / "mbr" / m / f"{a.split}.jsonl"
        mbr = {r["uid"]: r["choice"] for r in read_jsonl(mbr_path)} if mbr_path.exists() else {}
        for uid in sorted(ex):
            passed = ex[uid]["passed"]
            examples = ex[uid]["example_passed"]
            rec = {"uid": uid, "model": m, "n": len(passed), "oracle": any(passed),
                   "random(exp.)": sum(passed) / len(passed), "correct": {}, "choice": {}}
            correct = rec["correct"]
            choice = rec["choice"]
            choice["first"] = 0
            correct["first"] = bool(passed[0])
            idx = next((j for j, ok in enumerate(examples) if ok), 0)
            choice["example-filter"] = idx
            correct["example-filter"] = bool(passed[idx])
            if uid in mbr:
                choice["mbr-exec"] = mbr[uid]
                correct["mbr-exec"] = bool(passed[mbr[uid]])
            fs = feats.get(uid)
            if fs is not None and len(fs) != len(passed):
                raise SystemExit(f"{m}/{uid}: {len(fs)} feature rows vs {len(passed)} candidates")
            fs = fs if fs is not None else [{} for _ in passed]
            keys = {k for f in fs for k in f}
            for k in sorted(keys):
                if k.startswith("ar.") and k.endswith(".nll_sum"):
                    s = k[len("ar."):-len(".nll_sum")]
                    for name, norm in ((f"ar.{s}.sum", False), (f"ar.{s}.mean", True)):
                        j = best(fs, k, norm)
                        correct[name] = bool(passed[j]) if j is not None else False
                        choice[name] = j
                elif k == "decision.listwise.p":
                    j = top(fs, k)
                    correct["decision.listwise"] = bool(passed[j]) if j is not None else False
                    choice["decision.listwise"] = j
                elif k == "decision.pointwise.p_yes":
                    j = top(fs, k)
                    correct["decision.pointwise"] = bool(passed[j]) if j is not None else False
                    choice["decision.pointwise"] = j
            rows.append(rec)

    names = sorted({n for r in rows for n in r["correct"]})
    table = {}
    for m in a.models:
        rs = [r for r in rows if r["model"] == m]
        table[m] = {n: (sum(r["correct"][n] for r in rs if n in r["correct"])
                        / sum(1 for r in rs if n in r["correct"]))
                    if any(n in r["correct"] for r in rs) else None for n in names}
        table[m]["random(exp.)"] = sum(r["random(exp.)"] for r in rs) / len(rs)
        table[m]["oracle"] = sum(r["oracle"] for r in rs) / len(rs)
        table[m]["n"] = len(rs)
        table[m]["coverage"] = {n: sum(1 for r in rs if n in r["correct"]) / len(rs) for n in names}
    macro = {n: sum(table[m][n] for m in a.models if table[m][n] is not None)
             / sum(1 for m in a.models if table[m][n] is not None) for n in names}
    macro["random(exp.)"] = sum(table[m]["random(exp.)"] for m in a.models) / len(a.models)
    macro["oracle"] = sum(table[m]["oracle"] for m in a.models) / len(a.models)

    first = macro["first"]
    head = lambda n: (macro[n] - first) / (macro["oracle"] - first) if macro["oracle"] > first else 0.0
    print(f"\n=== HumanEval {a.split} · {len(rows)} items · {', '.join(a.models)} ===")
    print(f"{'selector':28s} " + " ".join(f"{m:>11s}" for m in a.models) + f" {'macro':>8s} {'headroom':>9s}")
    for n in names:
        cells = " ".join(f"{table[m][n]:11.3f}" if table[m][n] is not None else f"{'-':>11s}"
                         for m in a.models)
        print(f"{n:28s} {cells} {macro[n]:8.3f} {head(n):9.1%}")
    print(f"{'random(exp.)':28s} " + " ".join(f"{table[m]['random(exp.)']:11.3f}" for m in a.models)
          + f" {macro['random(exp.)']:8.3f}")
    print(f"{'oracle':28s} " + " ".join(f"{table[m]['oracle']:11.3f}" for m in a.models)
          + f" {macro['oracle']:8.3f}")

    def pair(n: str, base: str):
        rs = [r for r in rows if n in r["correct"] and base in r["correct"]]
        aa = [r["correct"][n] for r in rs]
        bb = [r["correct"][base] for r in rs]
        return aa, bb, sum(aa) / len(aa) - sum(bb) / len(bb)

    print("\npaired exact McNemar (two-sided), stacked over models")
    tests = {}
    for n in names:
        if n in ("first", "example-filter") or macro[n] <= macro["first"]:
            continue
        for base in ["example-filter", "first"]:
            if base not in macro or n == base:
                continue
            aa, bb, delta = pair(n, base)
            p, n01, n10 = mcnemar_exact(aa, bb)
            lo, hi = bootstrap_ci(aa, bb)
            tests[f"{n} vs {base}"] = {"delta": delta, "p": p, "n01": n01, "n10": n10, "ci95": [lo, hi]}
            print(f"  {n:24s} vs {base:16s} delta={delta:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
                  f"p={p:.2e} ({n01}+ / {n10}-)")

    if a.json:
        Path(a.json).write_text(json.dumps({"split": a.split, "models": a.models, "table": table,
                                            "macro": macro, "tests": tests}, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
