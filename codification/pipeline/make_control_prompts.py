"""Closed-book memorisation probe for GSM8K (40 items).

Each request offers the gold answer and a wrong numeric answer from another
question, order randomised. The decision model sees only the question and the
two numbers, so above-chance accuracy is evidence of memorisation.
"""
from __future__ import annotations

import argparse
import json
import random

from common import TASK, WORK, read_jsonl

OUT = WORK / "openrouter"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    gold = {r["uid"]: r["answer"] for r in read_jsonl(TASK / "data" / "test" / "gold.jsonl")}
    rows = read_jsonl(WORK / "candidates" / "dream-base" / "test.jsonl")[: a.n]
    answers = [v for v in gold.values() if v and v.lstrip("-").replace(".", "").isdigit()]
    rng = random.Random(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "test.closedbook.prompts.jsonl"
    meta = OUT / "test.closedbook.meta.jsonl"
    with open(p, "w", encoding="utf-8") as f, open(meta, "w", encoding="utf-8") as m:
        for r in rows:
            g = gold[r["uid"]]
            wrong = rng.choice([x for x in answers if x != g])
            opts = [g, wrong]
            rng.shuffle(opts)
            gold_letter = "A" if opts[0] == g else "B"
            req = {"uid": r["uid"],
                   "state": {"problem": r["question"]},
                   "questions": {"answer": {
                       "type": "choice",
                       "instructions": "Which number is the correct final answer to the problem?",
                       "criteria": {"A": opts[0], "B": opts[1]},
                   }}}
            f.write(json.dumps(req) + "\n")
            m.write(json.dumps({"uid": r["uid"], "gold_letter": gold_letter,
                                "options": opts}) + "\n")
    print(f"wrote {len(rows)} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
