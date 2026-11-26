"""Evaluate selectors on the GSM-Symbolic candidate pool.

    python3 gsm_symbolic_eval.py --json ../../results/gsm_symbolic_eval.json

Reads work/gsm_symbolic/questions.jsonl (question + gold), candidates from
work/gsm_symbolic/candidates/<model>/test.jsonl and any features under
work/gsm_symbolic/features/<model>/test.*.jsonl. Gold is graded with the same
protected task.is_correct used by generation and the GSM8K pipeline, so a
selector never sees the answer key.

Selectors present in the output depend on the features computed: first,
majority, ar.<scorer>.sum/.mean, decision.listwise, decision.pointwise, oracle.

Statistics:
  * paired exact McNemar on questions, reported per model (the valid test);
  * a paired cluster bootstrap that resamples questions and carries both models
    together for the macro delta (the stacked McNemar is anti-conservative and
    is not used);
  * headroom over `first` toward the oracle.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

from common import WORK, read_jsonl, task

HERE = WORK / "gsm_symbolic"
GSM8K_K64_DELTA = 0.052          # controls.json: GSM8K listwise - majority, K=64
GSM8K_K64_CI = (0.030, 0.074)


def load_questions() -> dict[str, dict]:
    return {r["uid"]: r for r in read_jsonl(HERE / "questions.jsonl")}


def load_features(model: str, tag: str = "") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    d = HERE / "features" / model
    if not d.exists():
        return out
    for f in sorted(d.glob(f"test{tag}.*.jsonl")):
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
    sel = {"first": lambda c: 0, "majority": majority}
    for s in ar:
        sel[f"ar.{s}.sum"] = lambda c, s=s: argmin(c, f"ar.{s}.nll_sum")
        sel[f"ar.{s}.mean"] = lambda c, s=s: argmin(c, f"ar.{s}.nll_sum", normalise=True)
    for k in sorted(feat_keys):
        if k.startswith("decision."):
            name = "decision.listwise" if k == "decision.listwise.p" else (
                   "decision.pointwise" if k == "decision.pointwise.p_yes" else k[:-2])
            sel[name] = lambda c, k=k: argmax(c, k)
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


def cluster_bootstrap(per_model: dict[str, dict[str, dict]], sel: str, base: str,
                      seed: int = 0, n: int = 10_000) -> dict:
    """Resample questions; carry every model with the same resample.

    The macro delta is the mean over models of that model's accuracy difference
    on the resampled questions, matching the macro columns in the table.
    """
    available = per_model
    per_model = {}
    for m, d in available.items():
        keep = {uid: r for uid, r in d.items()
                if sel in r["correct"] and base in r["correct"]}
        if keep:
            per_model[m] = keep
    uids = sorted(set.intersection(*(set(d) for d in per_model.values())))
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        draw = [uids[rng.randrange(len(uids))] for _ in range(len(uids))]
        md = []
        for d in per_model.values():
            a = sum(d[u]["correct"][sel] for u in draw) / len(draw)
            b = sum(d[u]["correct"][base] for u in draw) / len(draw)
            md.append(a - b)
        deltas.append(sum(md) / len(md))
    deltas.sort()
    point = sum(sum(d[u]["correct"][sel] - d[u]["correct"][base] for d in per_model.values())
                for u in uids) / (len(uids) * len(per_model))
    return {"delta": point, "ci95": [deltas[int(0.025 * n)], deltas[int(0.975 * n) - 1]],
            "n_questions": len(uids), "reps": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--reps", type=int, default=10_000)
    ap.add_argument("--tag", default="", help="candidate-file tag, e.g. .k256 for the K=256 subset")
    ap.add_argument("--common", action="store_true",
                    help="score only questions present for every model (equal denominators)")
    a = ap.parse_args()

    questions = load_questions()
    models = a.models or sorted(p.name for p in (HERE / "candidates").iterdir()
                                if (p / f"test{a.tag}.jsonl").exists())
    all_cands = {}
    for m in models:
        all_cands[m] = {r["uid"]: r for r in read_jsonl(HERE / "candidates" / m / f"test{a.tag}.jsonl")
                        if r["uid"] in questions}
    common = set.intersection(*(set(d) for d in all_cands.values())) if all_cands else set()
    rows, per_item = [], {}
    for m in models:
        cands = all_cands[m]
        if a.common:
            cands = {u: r for u, r in cands.items() if u in common}
        if not cands:
            continue
        feats = load_features(m, a.tag)
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
                cs.append({**c, "features": fs_all[j] if fs_all is not None else {}})
            # questions.jsonl.answer is the worked solution ending in '#### N';
            # grade its final number, exactly as extraction reads a candidate.
            g = task.extract_answer(questions[uid]["answer"]) or questions[uid]["answer"]
            ok = [task.is_correct(c.get("answer"), g) for c in cs]
            rec = {"uid": uid, "model": m, "n": len(cs), "oracle": any(ok),
                   "random(exp.)": sum(ok) / len(ok), "correct": {}, "present": {}, "choice": {}}
            for name, f in sel.items():
                i = f(cs)
                rec["present"][name] = i is not None
                if i is not None:
                    rec["correct"][name] = bool(ok[i])
                rec["choice"][name] = i
            per_item[(m, uid)] = rec
            rows.append(rec)

    names = sorted({n for r in rows for n in r["correct"]})
    table = {}
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            continue
        table[m] = {}
        for n in names:
            den = sum(1 for r in rs if r["present"].get(n, False))
            table[m][n] = (sum(r["correct"][n] for r in rs if n in r["correct"]) / den) if den else None
        table[m]["coverage"] = {n: sum(1 for r in rs if r["present"].get(n, False)) / len(rs)
                                for n in names}
        table[m]["random(exp.)"] = sum(r["random(exp.)"] for r in rs) / len(rs)
        table[m]["oracle"] = sum(r["oracle"] for r in rs) / len(rs)
        table[m]["n"] = len(rs)
    models = [m for m in models if m in table]
    macro = {n: sum(table[m][n] for m in models if table[m][n] is not None)
             / sum(1 for m in models if table[m][n] is not None) for n in names}
    macro["random(exp.)"] = sum(table[m]["random(exp.)"] for m in models) / len(models)
    macro["oracle"] = sum(table[m]["oracle"] for m in models) / len(models)
    first = macro["first"]
    head = lambda n: (macro[n] - first) / (macro["oracle"] - first) if macro["oracle"] > first else 0.0

    order = ["first", "majority"] + [n for n in names if n.startswith("ar.")] + \
            [n for n in names if n.startswith("decision.")]
    print(f"\n=== gsm-symbolic · {len(rows)} items · {', '.join(models)} ===")
    print(f"{'selector':28s} " + " ".join(f"{m:>11s}" for m in models) + f" {'macro':>8s} {'headroom':>9s}")
    for n in order:
        cells = " ".join(f"{table[m][n]:11.3f}" if table[m][n] is not None else f"{'-':>11s}"
                         for m in models)
        if n.startswith("decision.") or n.startswith("ar."):
            cov = min(table[m]["coverage"][n] for m in models if table[m][n] is not None)
            cells += f"  cov={cov:.2f}"
        print(f"{n:28s} {cells} {macro[n]:8.3f} {head(n):9.1%}")
    print(f"{'random(exp.)':28s} " + " ".join(f"{table[m]['random(exp.)']:11.3f}" for m in models)
          + f" {macro['random(exp.)']:8.3f}")
    print(f"{'oracle':28s} " + " ".join(f"{table[m]['oracle']:11.3f}" for m in models)
          + f" {macro['oracle']:8.3f}")

    per_model_items: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for (m, uid), r in per_item.items():
        per_model_items[m][uid] = r

    tests = {}
    print("\npaired exact McNemar vs majority (two-sided)")
    for n in ("decision.listwise", "decision.pointwise"):
        if n not in names:
            continue
        for m in models:
            rs = [r for (mm, _), r in per_item.items() if mm == m and n in r["correct"]]
            if not rs:
                continue
            aa = [r["correct"][n] for r in rs]
            bb = [r["correct"]["majority"] for r in rs]
            p, n01, n10 = mcnemar_exact(aa, bb)
            delta = sum(aa) / len(aa) - sum(bb) / len(bb)
            tests[f"{n} vs majority [{m}]"] = {"delta": delta, "p": p, "n01": n01, "n10": n10, "n": len(rs)}
            print(f"  {n:20s} [{m:10s}] delta={delta:+.3f} p={p:.2e} ({n01}+ / {n10}-) n={len(rs)}")

    print("\npaired cluster bootstrap over questions (macro, 10k resamples)")
    for n in ("decision.listwise", "decision.pointwise"):
        if n not in names or "majority" not in names:
            continue
        boot = cluster_bootstrap(per_model_items, n, "majority", n=a.reps)
        boot["per_model"] = {}
        for m in models:
            if table[m].get(n) is None:
                continue
            one = cluster_bootstrap({m: per_model_items[m]}, n, "majority", n=a.reps)
            boot["per_model"][m] = one
        tests[f"{n} vs majority"] = boot
        print(f"  {n:20s} delta={boot['delta']:+.3f} CI[{boot['ci95'][0]:+.3f},{boot['ci95'][1]:+.3f}]")

    lm = None
    if "decision.listwise" in names and "majority" in names:
        lm = macro["decision.listwise"] - macro["majority"]
        tests["listwise_minus_majority"] = {"delta": lm,
                                            "gsm8k_k64_delta": GSM8K_K64_DELTA,
                                            "gsm8k_k64_ci95": list(GSM8K_K64_CI),
                                            "shrinks_vs_gsm8k": lm < GSM8K_K64_DELTA}
        print(f"\nlistwise - majority = {lm:+.3f}; GSM8K K=64 = {GSM8K_K64_DELTA:+.3f} "
              f"CI[{GSM8K_K64_CI[0]:+.3f},{GSM8K_K64_CI[1]:+.3f}]  "
              f"-> {'SHRINKS' if lm < GSM8K_K64_DELTA else 'DOES NOT SHRINK'}")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "dataset": "gsm_symbolic", "models": models, "n_questions": len(questions),
            "tag": a.tag, "common_questions_only": a.common,
            "table": table, "macro": macro, "tests": tests,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for (m, uid), r in sorted(per_item.items()):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
