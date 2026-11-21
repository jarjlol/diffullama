"""N4 steps 1, 2 and the extraction audit -- CPU only, Python stdlib, reads cached items.

    python3 codification/n4/n4_items.py --task-dir codification/rerank_task \
        --models dream-base llada-base --out /tmp/n4      # writes n4.json, audit.txt; prints a summary

Rules from ITERATION_2_PROMPTS.md N4: explore on dev only; this script never reads the
final split (it only opens the split directories given in --splits, default dev and test,
where test = the 200-question gate split). The failure-class rules below are fixed BEFORE
looking at any items. Output for A3 is counts and labels only, never a gold answer;
audit.txt carries candidate text and extracted answers but no gold.

1. Answer diversity, per model and split: mean distinct extracted answers per question,
   mean Shannon entropy (bits) of the answer distribution (None counted as its own
   answer), and the share of candidates with no extractable answer.
2. Majority-vote failures: dev questions where majority (ties -> earliest, as
   baselines._majority) is wrong and the oracle is right. Each gets ONE primary class,
   tried in this order:
     arithmetic_slip   at least half of the candidates voting for the wrong plurality
                       answer contain a checkable calculation "a op b = c" (or GSM8K
                       "<<a op b=c>>") whose c is wrong
     close_second      the correct answer has >= (plurality votes - 1) votes
     shared_wrong      the wrong plurality has >= N/2 votes (many samples agree on the
                       same wrong result; a proxy for "all candidates share a wrong step")
     other             none of the above
   and an independent flag:
     extraction_suspect  some candidate whose extracted answer is wrong has the gold
                         number as the last number of its text cut after the "#### N"
                         line (the extractor may have taken a number the model did not mean)
3. Extraction audit: 20 candidates from the step-2 dev questions (seeded sample), with the
   tail of the cut text and the extracted answer, for a person or agent to read.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

_CALC = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*([+\-*/x×])\s*(-?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)")


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def calc_errors(text: str) -> tuple[int, int]:
    """(checked, wrong) calculations in text; '<<a*b=c>>' annotations included."""
    checked = wrong = 0
    for a, op, b, c in _CALC.findall(text.replace("<<", " ").replace(">>", " ")):
        try:
            x, y, z = _f(a), _f(b), _f(c)
            v = {"+": x + y, "-": x - y, "*": x * y, "x": x * y, "×": x * y, "/": x / y if y else None}[op]
        except (ValueError, KeyError):
            continue
        if v is None:
            continue
        checked += 1
        wrong += abs(v - z) > 1e-6 * max(1.0, abs(v))
    return checked, wrong


def cut_after_answer(text: str, trim) -> str:
    """Text up to the end of the '#### N' line, as score_candidates.py --cut (I-22)."""
    t = trim(text)
    m = re.search(r"####[^\n]*", t)
    return t[:m.end()] if m else t


def majority(answers):
    counts = collections.Counter(a for a in answers if a is not None)
    if not counts:
        return 0, counts
    best = max(counts.values())
    return next(i for i, a in enumerate(answers) if a is not None and counts[a] == best), counts


def entropy(answers) -> float:
    c = collections.Counter(answers); n = len(answers)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True, help="the rerank_task directory (holds task.py and data/)")
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert "final" not in " ".join(a.splits), "N4 must not read the final split"
    sys.path.insert(0, str(Path(a.task_dir).resolve()))
    import task  # noqa: E402  (protected grading code, used read-only)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"diversity": {}, "failures": {}, "rules": __doc__.split("2. Majority")[1].split("3. Extraction")[0].strip()}
    audit_pool = []
    for split in a.splits:
        items, gold = task.load_items(split), task.load_gold(split)
        per = collections.defaultdict(lambda: {"n": 0, "distinct": 0, "entropy": 0.0, "none": 0, "cands": 0})
        for it in items:
            m = it["model"]
            if a.models and m not in a.models:
                continue
            answers = [c.get("answer") for c in it["candidates"]]
            assert answers, f"{it['uid']}: no candidates"
            d = per[m]; d["n"] += 1; d["cands"] += len(answers)
            d["distinct"] += len(set(answers)); d["entropy"] += entropy(answers); d["none"] += answers.count(None)
            if split != "dev":
                continue
            g = gold[it["uid"]]
            ok = [task.is_correct(x, g) for x in answers]
            i_maj, counts = majority(answers)
            if ok[i_maj] or not any(ok):
                continue
            wrong = answers[i_maj]; top = counts[wrong]
            right_votes = sum(ok)
            wrong_idx = [i for i, x in enumerate(answers) if x == wrong]
            texts = [cut_after_answer(c["text"], task.trim) for c in it["candidates"]]
            slips = sum(1 for i in wrong_idx if calc_errors(texts[i])[1] > 0)
            if slips * 2 >= len(wrong_idx):
                cls = "arithmetic_slip"
            elif right_votes >= top - 1:
                cls = "close_second"
            elif top * 2 >= len(answers):
                cls = "shared_wrong"
            else:
                cls = "other"
            suspect = False
            for i, x in enumerate(answers):
                if ok[i]:
                    continue
                nums = re.findall(task._NUM, texts[i])
                if nums and task.is_correct(task.normalize(nums[-1]), g):
                    suspect = True
            f = report["failures"].setdefault(m, {"n_questions": 0, "majority_wrong_oracle_right": 0,
                                                  "classes": collections.Counter(), "extraction_suspect": 0})
            f["majority_wrong_oracle_right"] += 1
            f["classes"][cls] += 1
            f["extraction_suspect"] += suspect
            for i in range(len(answers)):
                audit_pool.append((m, it["uid"], i, answers[i], texts[i]))
        for m, d in per.items():
            report["diversity"].setdefault(m, {})[split] = {
                "questions": d["n"], "mean_distinct_answers": round(d["distinct"] / d["n"], 3),
                "mean_entropy_bits": round(d["entropy"] / d["n"], 3), "none_share": round(d["none"] / d["cands"], 4)}
            if split == "dev" and m in report["failures"]:
                report["failures"][m]["n_questions"] = d["n"]
    for f in report["failures"].values():
        f["classes"] = dict(f["classes"])
    (out / "n4.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rng = random.Random(a.seed)
    sample = rng.sample(audit_pool, min(a.audit_n, len(audit_pool)))
    with open(out / "audit.txt", "w", encoding="utf-8") as fh:
        fh.write("Extraction audit: does the extracted answer equal the number the candidate meant? (no gold here)\n\n")
        for m, uid, i, ans, text in sample:
            fh.write(f"=== {m} {uid} candidate {i} | extracted: {ans}\n...{text[-400:]}\n\n")

    print(json.dumps({"diversity": report["diversity"], "failures": report["failures"]}, indent=2))
    print("\nFailure classes for A3 (dev counts, labels only):")
    for cls in ("arithmetic_slip", "close_second", "shared_wrong", "other"):
        print(f"  {cls:<16}" + "  ".join(f"{m} {f['classes'].get(cls, 0)}/{f['majority_wrong_oracle_right']}"
                                         for m, f in sorted(report["failures"].items())))
    print(f"\nwrote {out / 'n4.json'} and {out / 'audit.txt'} ({len(sample)} candidates)")


if __name__ == "__main__":
    main()
