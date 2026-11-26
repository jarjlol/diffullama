"""Judge-alone and gold-injection controls for GSM8K.

judge:  the decider sees only the question and the distinct candidate answers
        (no reasoning text). If it matches listwise, the reasoning text adds
        nothing.
gold:   only items where no candidate has the gold answer; the gold is added as
        an extra option. Picking it is direct evidence of answer knowledge.

    python3 judge_controls.py --make --mode judge
    python3 judge_controls.py --eval --mode judge --results ... --prompts ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from common import TASK, WORK, read_jsonl
from decision_rerank import LETTERS

OUT = WORK / "openrouter"


def options_for(r, gold, mode):
    answers = []
    for c in r["candidates"]:
        a = c.get("answer")
        if a is not None and a not in answers:
            answers.append(a)
    if mode == "judge":
        if len(answers) < 2:
            return None, answers
    else:
        if task_correct(answers, gold):
            return None, answers
        answers = answers + [gold]
    perm = list(range(len(answers)))
    random.Random(f"judge|{r['uid']}").shuffle(perm)
    return [answers[i] for i in perm], answers


def task_correct(answers, gold) -> bool:
    import sys
    sys.path.insert(0, str(TASK))
    import task
    return any(task.is_correct(a, gold) for a in answers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["judge", "gold"], required=True)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--prompts", default=None)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(TASK))
    import task
    gold = {r["uid"]: r["answer"] for r in read_jsonl(TASK / "data" / "test" / "gold.jsonl")}

    if a.make:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"gsm8k.{a.mode}.prompts.jsonl"
        n = 0
        with open(p, "w", encoding="utf-8") as f:
            for model in ("dream-base", "llada-base"):
                for r in read_jsonl(WORK / "candidates" / model / "test.jsonl"):
                    shown, original = options_for(r, gold[r["uid"]], a.mode)
                    if shown is None:
                        continue
                    req = {"uid": r["uid"], "model": model,
                           "state": {"problem": r["question"],
                                     "answers": {LETTERS[i]: v for i, v in enumerate(shown)}},
                           "questions": {"best": {
                               "type": "choice",
                               "instructions": "Which option is the correct final answer to the problem?",
                               "criteria": {LETTERS[i]: f"Option {LETTERS[i]}" for i in range(len(shown))},
                           }}}
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n += 1
        print(f"wrote {n} -> {p}")
        return 0

    if not (a.results and a.prompts):
        raise SystemExit("--eval needs --results and --prompts")
    res = {r["uid"]: r for r in read_jsonl(a.results)}
    ok = n = 0
    gold_picked = gold_n = 0
    for req in read_jsonl(a.prompts):
        rec = res.get(req["uid"])
        if not rec or "probs" not in rec:
            continue
        probs = rec["probs"]
        allowed = req["state"]["answers"]
        valid = {k: v for k, v in probs.items() if k in allowed}
        if not valid:
            continue
        letter = max(valid, key=lambda k: valid[k])
        # Recover the option order from the prompt (state is a dict keyed by letter).
        choice = allowed[letter]
        g = gold[req["uid"]]
        if a.mode == "judge":
            ok += task.is_correct(choice, g)
        else:
            gold_picked += task.is_correct(choice, g)
            gold_n += 1
        n += 1
    if a.mode == "judge":
        print(f"judge-alone: {ok}/{n} = {ok/n:.3f}")
    else:
        print(f"gold injection: picked gold {gold_picked}/{gold_n} = {gold_picked/gold_n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
