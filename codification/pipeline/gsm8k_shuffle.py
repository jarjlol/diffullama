"""GSM8K candidate-order shuffle control (dev, 200 items, seed 1).

Builds two prompt sets over the same questions -- original order and a seeded
permutation -- scores both through OpenRouter, and reports accuracy. The
permutation is stored in each prompt row and used to map the chosen letter back
to the original candidate.
"""
from __future__ import annotations

import argparse
import json
import random

from common import TASK, WORK, read_jsonl
from decision_rerank import LETTERS, candidate_text

OUT = WORK / "openrouter"


def prompt_rows(rows, shuffled: bool, seed: int):
    for r in rows:
        texts = [candidate_text(c["text"]) for c in r["candidates"]]
        perm = list(range(len(texts)))
        if shuffled:
            random.Random(f"{seed}|{r['uid']}").shuffle(perm)
        shown = [texts[i] for i in perm]
        yield {
            "uid": r["uid"], "model": "dream-base", "perm": perm,
            "state": {"problem": r["question"],
                      "candidates": {LETTERS[i]: t for i, t in enumerate(shown)}},
            "questions": {"best": {
                "type": "choice",
                "instructions": "Which candidate solution reaches the correct final answer to the math problem?",
                "criteria": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(shown))},
            }},
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--plain", default=None)
    ap.add_argument("--shuf", default=None)
    a = ap.parse_args()

    rows = read_jsonl(WORK / "candidates" / "dream-base" / "dev.jsonl")[: a.n]
    if a.make:
        OUT.mkdir(parents=True, exist_ok=True)
        for variant, shuffled in (("plain", False), ("shuf", True)):
            p = OUT / f"dev.gsm8k.{variant}.prompts.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for req in prompt_rows(rows, shuffled, a.seed):
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
            print(f"wrote {len(rows)} -> {p}")
        return 0

    import sys
    sys.path.insert(0, str(TASK))
    import task

    gold = {r["uid"]: r["answer"] for r in read_jsonl(TASK / "data" / "dev" / "gold.jsonl")}
    cand = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / "dream-base" / "dev.jsonl")}

    def accuracy(results_path, prompts_path):
        res = {r["uid"]: r["probs"] for r in read_jsonl(results_path)}
        perms = {r["uid"]: r["perm"] for r in read_jsonl(prompts_path)}
        ok = n = 0
        for r in rows:
            probs = res.get(r["uid"])
            if not probs:
                continue
            j = max(range(len(probs)), key=lambda k: probs[LETTERS[k]])
            idx = perms[r["uid"]][j]
            ok += task.is_correct(cand[r["uid"]]["candidates"][idx].get("answer"), gold[r["uid"]])
            n += 1
        return ok / n, n

    if a.eval:
        if not (a.plain and a.shuf):
            raise SystemExit("--eval needs --plain and --shuf")
        plain = accuracy(a.plain, OUT / "dev.gsm8k.plain.prompts.jsonl")
        shuf = accuracy(a.shuf, OUT / "dev.gsm8k.shuf.prompts.jsonl")
        print(f"plain: {plain[0]:.3f} (n={plain[1]})   shuffled: {shuf[0]:.3f} (n={shuf[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
