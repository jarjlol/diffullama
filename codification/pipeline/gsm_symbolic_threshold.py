"""Weak-vote threshold rule (#81) evaluated on GSM-Symbolic (#79).

    python3 gsm_symbolic_threshold.py \
        --dev-results work/openrouter/dev.gsm8k.plain.results.jsonl \
        --test-results work/gsm_symbolic/prompts/test.listwise.results.jsonl \
        --json ../../results/gsm_symbolic_threshold.json

Rule: if the most common extracted answer has at least T of the 10 votes, keep
the majority vote; otherwise call the decision model listwise. T is chosen on
the 200 GSM8K dev questions only (higher dev macro accuracy; ties go to the
lower T, i.e. fewer listwise calls). The chosen T and all of T=4,5,6 are then
reported once on GSM-Symbolic. No tuning on the GSM-Symbolic questions.
"""
from __future__ import annotations

import argparse
import collections
import json

from common import WORK, read_jsonl, task
from decision_rerank import LETTERS

HERE = WORK / "gsm_symbolic"
TS = (4, 5, 6)


def majority_info(cands: list[dict]) -> tuple[int, int]:
    """(first index of the modal answer, modal count)."""
    answers = [c.get("answer") for c in cands]
    counts = collections.Counter(a for a in answers if a is not None)
    if not counts:
        return 0, 0
    best = max(counts.values())
    idx = next(i for i, a in enumerate(answers) if a is not None and counts[a] == best)
    return idx, best


def load_results(paths: list[str]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for p in paths:
        for r in read_jsonl(p):
            if "probs" in r:
                out[r["model"]][r["uid"]] = r["probs"]
    return out


def evaluate(model: str, cands: dict[str, dict], gold: dict[str, str],
             probs: dict[str, dict]) -> dict:
    per_t = {T: {"n": 0, "correct": 0, "listwise_calls": 0} for T in TS}
    base_maj = base_lw = n = 0
    for uid, p in probs.items():
        row = cands[uid]
        cs = row["candidates"]
        gold_uid = gold[uid]
        maj_idx, maj_n = majority_info(cs)
        lw_idx = LETTERS.index(max(p, key=p.get))
        maj_ok = task.is_correct(cs[maj_idx].get("answer"), gold_uid)
        lw_ok = task.is_correct(cs[lw_idx].get("answer"), gold_uid)
        base_maj += int(maj_ok)
        base_lw += int(lw_ok)
        n += 1
        for T in TS:
            d = per_t[T]
            d["n"] += 1
            if maj_n >= T:
                d["correct"] += int(maj_ok)
            else:
                d["correct"] += int(lw_ok)
                d["listwise_calls"] += 1
    return {"n": n,
            "always_majority": base_maj / n if n else None,
            "always_listwise": base_lw / n if n else None,
            "per_T": {str(T): {"accuracy": d["correct"] / d["n"] if d["n"] else None,
                               "listwise_calls": d["listwise_calls"],
                               "listwise_fraction": d["listwise_calls"] / d["n"] if d["n"] else None}
                      for T, d in per_t.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-results", nargs="*", default=[str(WORK / "openrouter" / "dev.gsm8k.plain.results.jsonl")])
    ap.add_argument("--test-results", default=str(HERE / "prompts/test.listwise.results.jsonl"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    dev_q = {r["uid"]: r for r in read_jsonl(WORK / "data" / "dev.jsonl")}
    dev_gold = {uid: (task.extract_answer(q["answer"]) or q["answer"]) for uid, q in dev_q.items()}
    dev_c = {m: {r["uid"]: r for r in read_jsonl(WORK / "candidates" / m / "dev.jsonl")}
             for m in ("dream-base", "llada-base")}
    dev = {}
    for model, probs in load_results(a.dev_results).items():
        if model not in dev_c:
            continue
        probs = {u: p for u, p in probs.items() if u in dev_c[model]}
        if probs:
            dev[model] = evaluate(model, dev_c[model], dev_gold, probs)

    macro = {}
    for T in TS:
        vals = [dev[m]["per_T"][str(T)]["accuracy"] for m in dev
                if dev[m]["per_T"][str(T)]["accuracy"] is not None]
        macro[T] = sum(vals) / len(vals) if vals else None
    valid = [(macro[T], -T, T) for T in TS if macro[T] is not None]
    chosen = max(valid)[2] if valid else None

    test_q = {r["uid"]: r for r in read_jsonl(HERE / "questions.jsonl")}
    test_gold = {uid: (task.extract_answer(q["answer"]) or q["answer"]) for uid, q in test_q.items()}
    test_c = {m: {r["uid"]: r for r in read_jsonl(HERE / "candidates" / m / "test.jsonl")}
              for m in ("dream-base", "llada-base")}
    test = {}
    for model, probs in load_results([a.test_results]).items():
        if model not in test_c:
            continue
        probs = {u: p for u, p in probs.items() if u in test_c[model]}
        if probs:
            test[model] = evaluate(model, test_c[model], test_gold, probs)

    print(f"\ndev (tuned here; n={sum(dev[m]['n'] for m in dev)} question-model pairs)")
    for model, r in dev.items():
        print(f"  {model:10s} majority={r['always_majority']:.3f} listwise={r['always_listwise']:.3f}  "
              + "  ".join(f"T{T}={r['per_T'][str(T)]['accuracy']:.3f}"
                          f"({r['per_T'][str(T)]['listwise_fraction']:.0%} lw)" for T in TS))
    print(f"  macro: " + "  ".join(f"T{T}={macro[T]:.3f}" for T in TS) + f"  -> chosen T={chosen}")

    print(f"\ngsm-symbolic test (reported once)")
    for model, r in test.items():
        print(f"  {model:10s} majority={r['always_majority']:.3f} listwise={r['always_listwise']:.3f}  "
              + "  ".join(f"T{T}={r['per_T'][str(T)]['accuracy']:.3f}"
                          f"({r['per_T'][str(T)]['listwise_fraction']:.0%} lw)" for T in TS))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"dev": dev, "dev_macro": {str(T): v for T, v in macro.items()},
                       "chosen_T": chosen, "test": test}, f, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
