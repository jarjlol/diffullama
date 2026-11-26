"""HumanEval candidate-order shuffle control (50 items, seed 1).

Generates prompts with the 10 candidates permuted, scores them through
OpenRouter, and compares the passing-candidate choice rate to the unshuffled run.
"""
from __future__ import annotations

import argparse
import json
import random

from common import WORK, read_jsonl
from decision_rerank import LETTERS
from humaneval_exec import candidate_code, problems

HERE = WORK / "humaneval"
OUT = WORK / "openrouter"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--eval", dest="eval_path", default=None)
    a = ap.parse_args()

    ps = problems()
    rows = read_jsonl(HERE / "candidates" / a.model / "test.jsonl")[: a.limit]
    if a.make:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / "test.he.shuffle.prompts.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                prob = ps[r["uid"]]
                codes = [candidate_code(c["text"], prob["prompt"], prob["entry_point"])
                         for c in r["candidates"]]
                perm = list(range(len(codes)))
                random.Random(f"{a.seed}|{r['uid']}").shuffle(perm)
                shown = [codes[i] for i in perm]
                req = {"uid": r["uid"], "model": a.model, "perm": perm,
                       "state": {"problem": prob["prompt"],
                                 "candidates": {LETTERS[i]: t for i, t in enumerate(shown)}},
                       "questions": {"best": {
                           "type": "choice",
                           "instructions": "Which candidate correctly implements the function described by the docstring and will pass the hidden tests?",
                           "criteria": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(shown))},
                       }}}
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} shuffled requests -> {p}")
        return 0

    res = {r["uid"]: r["probs"] for r in read_jsonl(a.eval_path)}
    ex = {r["uid"]: r["passed"] for r in read_jsonl(HERE / "exec" / a.model / "test.jsonl")}
    ok = n = 0
    for r in rows:
        probs = res.get(r["uid"])
        if not probs:
            continue
        perm = list(range(len(r["candidates"])))
        random.Random(f"{a.seed}|{r['uid']}").shuffle(perm)
        j = max(range(len(perm)), key=lambda k: probs[LETTERS[k]])
        ok += ex[r["uid"]][perm[j]]
        n += 1
    print(f"shuffled pass@1: {ok}/{n} = {ok/n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
