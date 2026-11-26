"""Calibration and serving cost for the GSM-Symbolic decision runs (#83).

    python3 gsm_symbolic_calibration.py \
        --listwise work/gsm_symbolic/prompts/test.listwise.results.jsonl \
        --pointwise work/gsm_symbolic/prompts/test.pointwise.results.jsonl \
        --json ../../results/gsm_symbolic_calibration.json

Listwise calibration uses the max probability of the chosen option against
whether that option is correct (selective calibration). Pointwise calibration
uses every (candidate, p_yes) pair against that candidate's correctness; those
pairs are clustered within questions, so the pointwise numbers are descriptive,
not an independent-sample estimate.

Cost comes from the client's `usage` field. The endpoint is the `:free` tier, so
the recorded cost is 0; a paid-price estimate is also reported at the flash
input price ($0.01 / 1M input tokens, output negligible at 1 token/request),
labelled as an estimate.
"""
from __future__ import annotations

import argparse
import json
import statistics

from common import WORK, read_jsonl, task
from decision_rerank import LETTERS

HERE = WORK / "gsm_symbolic"


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> tuple[float, list[dict]]:
    rows = []
    total = len(pairs)
    acc = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not bucket:
            rows.append({"bin": b, "lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        a = sum(y for _, y in bucket) / len(bucket)
        rows.append({"bin": b, "lo": lo, "hi": hi, "n": len(bucket), "conf": conf, "acc": a})
        acc += len(bucket) / total * abs(a - conf)
    return acc, rows


def brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def cost_block(rows: list[dict], n_correct: int) -> dict:
    usage = [r.get("usage") or {} for r in rows]
    tin = sum(int(u.get("input_tokens") or 0) for u in usage)
    tout = sum(int(u.get("output_tokens") or 0) for u in usage)
    recorded = sum(float(u.get("cost") or 0.0) for u in usage)
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    paid = tin * 0.01 / 1e6
    return {"requests": len(rows), "input_tokens": tin, "output_tokens": tout,
            "recorded_cost_usd": recorded,
            "paid_estimate_usd_at_0.01_per_1M_input": paid,
            "cost_per_correct_recorded": recorded / n_correct if n_correct else None,
            "cost_per_correct_paid_estimate": paid / n_correct if n_correct else None,
            "latency_s_mean": statistics.mean(lat) if lat else None,
            "latency_s_median": statistics.median(lat) if lat else None,
            "latency_s_p95": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listwise", default=str(HERE / "prompts/test.listwise.results.jsonl"))
    ap.add_argument("--pointwise", default=str(HERE / "prompts/test.pointwise.results.jsonl"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    questions = {r["uid"]: r for r in read_jsonl(HERE / "questions.jsonl")}
    gold = {uid: (task.extract_answer(q["answer"]) or q["answer"]) for uid, q in questions.items()}
    cands = {m: {r["uid"]: r for r in read_jsonl(HERE / "candidates" / m / "test.jsonl")}
             for m in ("dream-base", "llada-base")}
    out = {"listwise": {}, "pointwise": {}}

    lw = [json.loads(l) for l in open(a.listwise, encoding="utf-8") if l.strip()]
    lw = [r for r in lw if "probs" in r]
    for model in ("dream-base", "llada-base"):
        rs = [r for r in lw if r.get("model") == model]
        pairs, n_ok = [], 0
        for r in rs:
            row = cands[model][r["uid"]]
            p = r["probs"]
            chosen = max(p, key=p.get)
            idx = LETTERS.index(chosen)
            ok = task.is_correct(row["candidates"][idx].get("answer"), gold[r["uid"]])
            pairs.append((float(p[chosen]), int(ok)))
            n_ok += int(ok)
        if not pairs:
            continue
        acc, bins = ece(pairs)
        out["listwise"][model] = {"n": len(pairs), "correct": n_ok,
                                  "accuracy": n_ok / len(pairs),
                                  "ece10": acc, "brier": brier(pairs),
                                  "reliability": bins,
                                  **cost_block(rs, n_ok)}

    pw = [json.loads(l) for l in open(a.pointwise, encoding="utf-8") if l.strip()]
    pw = [r for r in pw if "noul" in r]
    for model in ("dream-base", "llada-base"):
        rs = [r for r in pw if r.get("model") == model]
        pairs, n_ok = [], 0
        for r in rs:
            row = cands[model][r["uid"]]
            ok = task.is_correct(row["candidates"][r["idx"]].get("answer"), gold[r["uid"]])
            pairs.append((float(r["noul"]), int(ok)))
            n_ok += int(ok)
        if not pairs:
            continue
        acc, bins = ece(pairs)
        out["pointwise"][model] = {"n": len(pairs), "correct": n_ok,
                                   "accuracy_per_candidate": n_ok / len(pairs),
                                   "ece10": acc, "brier": brier(pairs),
                                   "reliability": bins,
                                   **cost_block(rs, n_ok)}

    print(f"\n{'run':10s} {'model':12s} {'n':>5s} {'acc':>6s} {'ECE10':>7s} {'Brier':>7s} "
          f"{'lat_s':>7s} {'tokens':>9s} {'est$':>8s}")
    for kind in ("listwise", "pointwise"):
        for model, r in out[kind].items():
            acc = r.get("accuracy", r.get("accuracy_per_candidate"))
            print(f"{kind:10s} {model:12s} {r['n']:5d} {acc:6.3f} {r['ece10']:7.3f} "
                  f"{r['brier']:7.3f} {r['latency_s_mean']:7.2f} {r['input_tokens']:9d} "
                  f"{r['paid_estimate_usd_at_0.01_per_1M_input']:8.4f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
