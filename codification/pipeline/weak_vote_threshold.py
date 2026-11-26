"""Weak-vote threshold rule for GSM8K (issue #81, part b).

Use the majority answer when its top count is at least T; otherwise fall back to
the decision model's listwise choice.  T is fixed on dev (200 items per model)
and then reported once on test (500 per model); the test numbers are
descriptive, not tuned.

    python3 weak_vote_threshold.py --split dev --models dream-base \
        --listwise-openrouter work/openrouter/dev.gsm8k.plain.results.jsonl \
        --listwise-prompts work/openrouter/dev.gsm8k.plain.prompts.jsonl
    python3 weak_vote_threshold.py --split test --models dream-base llada-base
    python3 weak_vote_threshold.py --table

The listwise choice never sees gold. Gold is used only to score a choice through
the protected task.is_correct, exactly as decision_eval.py does.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from common import TASK, WORK, read_jsonl, task

RESULTS = Path(__file__).resolve().parent.parent / "results"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gold_for(split: str) -> dict[str, str]:
    return {r["uid"]: r["answer"] for r in read_jsonl(TASK / "data" / split / "gold.jsonl")}


def majority_index(cands: list[dict]) -> tuple[int, int]:
    """First index with the modal non-None answer, and the modal count."""
    answers = [c.get("answer") for c in cands]
    counts = collections.Counter(a for a in answers if a is not None)
    if not counts:
        return 0, 0
    best = max(counts.values())
    idx = next(i for i, a in enumerate(answers) if a is not None and counts[a] == best)
    return idx, best


def listwise_from_features(model: str, split: str) -> dict[str, int]:
    d = WORK / "features" / model
    paths = sorted(d.glob(f"{split}.decision-listwise.jsonl"))
    if not paths:
        raise SystemExit(f"no listwise features in {d}")
    out: dict[str, int] = {}
    for p in paths:
        for r in read_jsonl(p):
            vals = [f.get("decision.listwise.p") for f in r["features"]]
            if all(v is not None for v in vals):
                out[r["uid"]] = max(range(len(vals)), key=lambda i: (vals[i], -i))
    return out


def listwise_from_openrouter(results: Path, prompts: Path | None) -> dict[tuple[str, str], int]:
    """Map (model, uid) -> original candidate index from API probabilities.

    If a prompt file is given, its `perm` maps shown positions back to the
    original candidate order (the plain files use the identity permutation).
    """
    perm: dict[tuple[str, str], list[int]] = {}
    if prompts is not None:
        for r in read_jsonl(prompts):
            perm[(r["model"], r["uid"])] = r.get("perm") or list(range(len(r["state"]["candidates"])))
    out: dict[tuple[str, str], int] = {}
    for r in read_jsonl(results):
        probs = r.get("probs")
        if not probs:
            continue
        key = (r["model"], r["uid"])
        shown = max(probs, key=lambda k: (probs[k], k))
        pos = LETTERS.index(shown)
        p = perm.get(key)
        out[key] = p[pos] if p is not None and pos < len(p) else pos
    return out


def collect(split: str, models: list[str], openrouter_results: Path | None,
            openrouter_prompts: Path | None) -> dict:
    gold = gold_for(split)
    if openrouter_results is not None:
        lw = listwise_from_openrouter(openrouter_results, openrouter_prompts)
    else:
        lw = {}
    out = {"split": split, "models": {}, "n_gold": len(gold)}
    for m in models:
        cpath = WORK / "candidates" / m / f"{split}.jsonl"
        if not cpath.exists():
            raise SystemExit(f"missing {cpath}")
        feat_lw = {} if openrouter_results is not None else listwise_from_features(m, split)
        recs = {}
        for r in read_jsonl(cpath):
            uid = r["uid"]
            cands = r["candidates"]
            mi, top = majority_index(cands)
            li = lw.get((m, uid)) if openrouter_results is not None else feat_lw.get(uid)
            if li is None:
                continue
            g = gold[uid]
            recs[uid] = {
                "top_count": top,
                "n_answers": sum(1 for c in cands if c.get("answer") is not None),
                "majority_correct": bool(task.is_correct(cands[mi].get("answer"), g)),
                "listwise_correct": bool(task.is_correct(cands[li].get("answer"), g)),
                "same_choice": mi == li,
            }
        out["models"][m] = recs
        print(f"{split}/{m}: {len(recs)} items; "
              f"majority {sum(x['majority_correct'] for x in recs.values())/len(recs):.3f}; "
              f"listwise {sum(x['listwise_correct'] for x in recs.values())/len(recs):.3f}", flush=True)
    return out


def summarize(split_data: dict, thresholds: list[int]) -> dict:
    models = split_data["models"]
    table = {}
    for m, recs in models.items():
        n = len(recs)
        row = {
            "n": n,
            "majority": sum(r["majority_correct"] for r in recs.values()) / n,
            "listwise": sum(r["listwise_correct"] for r in recs.values()) / n,
            "top_count_hist": dict(sorted(collections.Counter(r["top_count"] for r in recs.values()).items())),
            "mean_n_answers": sum(r["n_answers"] for r in recs.values()) / n,
        }
        for T in thresholds:
            use_maj = {u: r for u, r in recs.items() if r["top_count"] >= T}
            acc = sum((r["majority_correct"] if r["top_count"] >= T else r["listwise_correct"])
                      for r in recs.values()) / n
            row[f"T{T}"] = {"accuracy": acc, "judge_fraction": 1 - len(use_maj) / n,
                            "n_majority": len(use_maj)}
        table[m] = row
    macro = {}
    for T in thresholds:
        macro[f"T{T}"] = {
            "accuracy": sum(table[m][f"T{T}"]["accuracy"] for m in models) / len(models),
            "judge_fraction": sum(table[m][f"T{T}"]["judge_fraction"] for m in models) / len(models),
        }
    macro["majority"] = sum(table[m]["majority"] for m in models) / len(models)
    macro["listwise"] = sum(table[m]["listwise"] for m in models) / len(models)
    return {"models": table, "macro": macro}


def pick_threshold(dev: dict, thresholds: list[int]) -> tuple[int, dict]:
    """Max dev macro accuracy; ties go to the higher threshold (fewer judge calls)."""
    ranked = sorted(thresholds, key=lambda T: (round(dev["macro"][f"T{T}"]["accuracy"], 12),
                                               T), reverse=True)
    best = ranked[0]
    return best, {str(T): dev["macro"][f"T{T}"] for T in thresholds}


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"delta": 0.0, "n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"delta": sum(a) / len(a) - sum(b) / len(b), "n01": n01, "n10": n10, "p": p}


def paired_tests(split_data: dict, T: int) -> dict:
    out = {}
    for m, recs in split_data["models"].items():
        rule = [r["majority_correct"] if r["top_count"] >= T else r["listwise_correct"]
                for r in recs.values()]
        out[m] = {
            "vs_listwise": mcnemar_exact(rule, [r["listwise_correct"] for r in recs.values()]),
            "vs_majority": mcnemar_exact(rule, [r["majority_correct"] for r in recs.values()]),
        }
    stacked = [r for recs in split_data["models"].values() for r in recs.values()]
    rule = [r["majority_correct"] if r["top_count"] >= T else r["listwise_correct"]
            for r in stacked]
    out["stacked"] = {
        "vs_listwise": mcnemar_exact(rule, [r["listwise_correct"] for r in stacked]),
        "vs_majority": mcnemar_exact(rule, [r["majority_correct"] for r in stacked]),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    ap.add_argument("--listwise-openrouter", default=None,
                    help="API results JSONL (dev); omit to read work/features/<model>/<split>.decision-listwise.jsonl")
    ap.add_argument("--listwise-prompts", default=None, help="matching prompts JSONL with perm")
    ap.add_argument("--thresholds", nargs="*", type=int, default=[4, 5, 6])
    ap.add_argument("--dump", default=None, help="write the collected per-item JSON here")
    ap.add_argument("--table", action="store_true", help="combine dev/test dumps and fit T on dev")
    ap.add_argument("--out", default=str(RESULTS / "weak_vote_threshold.json"))
    a = ap.parse_args()

    if a.table:
        dev = json.loads((RESULTS / "gsm8k_votes_dev.json").read_text(encoding="utf-8"))
        test = json.loads((RESULTS / "gsm8k_votes_test.json").read_text(encoding="utf-8"))
        ds = summarize(dev, a.thresholds)
        ts = summarize(test, a.thresholds)
        best, dev_by_T = pick_threshold(ds, a.thresholds)
        out = {"thresholds": a.thresholds, "selected_on_dev": best,
               "dev": ds, "test": ts,
               "dev_selection": dev_by_T,
               "test_tests_T%d" % best: paired_tests(test, best),
               "note": "T is selected on dev by macro accuracy (ties -> higher T). "
                       "The test columns for every T are descriptive, not tuned. "
                       "McNemar is two-sided exact on stacked items."}
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"\nselected T={best} on dev")
        print(f"{'split':5s} {'model':10s} {'maj':>6s} {'list':>6s} " +
              " ".join(f"T{T:>2d}".rjust(15) for T in a.thresholds))
        for split_name, s in (("dev", ds), ("test", ts)):
            for m, row in s["models"].items():
                cells = " ".join(f"{row[f'T{T}']['accuracy']:.3f}/"
                                 f"{row[f'T{T}']['judge_fraction']:.2f}".rjust(15)
                                 for T in a.thresholds)
                print(f"{split_name:5s} {m:10s} {row['majority']:.3f} {row['listwise']:.3f} {cells}")
            mac = s["macro"]
            cells = " ".join(f"{mac[f'T{T}']['accuracy']:.3f}/"
                             f"{mac[f'T{T}']['judge_fraction']:.2f}".rjust(15)
                             for T in a.thresholds)
            print(f"{split_name:5s} {'macro':10s} {mac['majority']:.3f} {mac['listwise']:.3f} {cells}")
        print(f"wrote {a.out}")
        return 0

    if a.split is None:
        ap.error("pass --split or --table")
    data = collect(a.split, a.models,
                   Path(a.listwise_openrouter) if a.listwise_openrouter else None,
                   Path(a.listwise_prompts) if a.listwise_prompts else None)
    dest = Path(a.dump) if a.dump else RESULTS / f"gsm8k_votes_{a.split}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
