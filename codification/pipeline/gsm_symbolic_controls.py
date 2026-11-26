"""Controls for the GSM-Symbolic arm (#79), same protocol as the GSM8K arm.

Build (on the GPU box):

    python3 gsm_symbolic_controls.py build --control judge
    python3 gsm_symbolic_controls.py build --control gold
    python3 gsm_symbolic_controls.py build --control closedbook

Evaluate (after the result file is fetched back):

    python3 gsm_symbolic_controls.py eval --control judge --results ...
    python3 gsm_symbolic_controls.py eval --control gold --results ...
    python3 gsm_symbolic_controls.py eval --control closedbook --results ...

  judge       question + the distinct candidate final answers, no reasoning.
              Skipped where there is only one distinct answer (nothing to judge).
  gold        only questions where no candidate carries the gold answer; the
              gold is appended as the last option. Tests whether the model can
              recognise the right number when none of the candidates has it.
  closedbook  question + two numbers (gold vs a wrong number from another
              question), order randomised with seed 0. Recognition, not recall.

Every prompt never contains a candidate's reasoning in `judge`/`closedbook`;
`gold` also hides it. Gold is used only to construct the prompt, never as a
feature for the main selectors.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from common import WORK, read_jsonl, task
from decision_rerank import LETTERS

HERE = WORK / "gsm_symbolic"
OUT = HERE / "controls"
MODELS = ("dream-base", "llada-base")

GOLD: dict[str, str] = {}
CAND: dict[str, dict[str, dict]] = {}
QUESTIONS: dict[str, dict] = {}


def distinct(answers: list[str]) -> list[str]:
    seen, out = set(), []
    for a in answers:
        if a is not None and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / den
    return max(0.0, mid - half), min(1.0, mid + half)


def choice_request(uid: str, model: str | None, problem: str, options: list[str],
                   instructions: str = "Which option is the correct final answer to the problem?") -> dict:
    req = {
        "uid": uid,
        "state": {"problem": problem, "answers": {LETTERS[i]: a for i, a in enumerate(options)}},
        "questions": {"best": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {LETTERS[i]: f"Option {LETTERS[i]}" for i in range(len(options))},
        }},
    }
    if model is not None:
        req["model"] = model
    return req


def build(control: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{control}.prompts.jsonl"
    n = 0
    if control in ("judge", "gold"):
        with open(path, "w", encoding="utf-8") as f:
            for model in MODELS:
                for uid in sorted(CAND[model]):
                    answers = distinct([c.get("answer") for c in CAND[model][uid]["candidates"]])
                    if control == "judge":
                        if len(answers) < 2:
                            continue
                        options = answers
                    else:
                        if any(task.is_correct(a, GOLD[uid]) for a in answers):
                            continue
                        options = answers + [GOLD[uid]]
                    req = choice_request(uid, model, QUESTIONS[uid]["question"], options)
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n += 1
    elif control == "closedbook":
        rng = random.Random(0)
        golds = [v for v in GOLD.values() if v]
        key = []
        with open(path, "w", encoding="utf-8") as f, open(OUT / "closedbook.key.jsonl", "w",
                                                          encoding="utf-8") as kf:
            for uid in sorted(QUESTIONS):
                gold = GOLD[uid]
                wrong = rng.choice(golds)
                while task.is_correct(wrong, gold):
                    wrong = rng.choice(golds)
                gold_letter = "A" if rng.random() < 0.5 else "B"
                wrong_letter = "B" if gold_letter == "A" else "A"
                req = {
                    "uid": uid,
                    "state": {"problem": QUESTIONS[uid]["question"]},
                    "questions": {"answer": {
                        "type": "choice",
                        "instructions": "Which number is the correct final answer to the problem?",
                        "criteria": {gold_letter: gold, wrong_letter: wrong},
                    }},
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
                kf.write(json.dumps({"uid": uid, "gold_letter": gold_letter,
                                     "gold": gold, "wrong": wrong}) + "\n")
                n += 1
    print(f"wrote {n} {control} requests -> {path}")


def eval_control(control: str, results: str, json_out: str | None) -> None:
    rows = [json.loads(l) for l in open(results, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "probs" in r]
    if control == "closedbook":
        key = {r["uid"]: r for r in read_jsonl(OUT / "closedbook.key.jsonl")}
        n_ok = n = 0
        by_pos = collections.Counter()
        for r in rows:
            k = key[r["uid"]]
            chosen = max(r["probs"], key=r["probs"].get)
            ok = chosen == k["gold_letter"]
            n_ok += int(ok)
            n += 1
            by_pos[(k["gold_letter"], ok)] += 1
        out = {"control": control, "n": n, "correct": n_ok,
               "accuracy": n_ok / n if n else None, "chance": 0.5,
               "ci95_wilson": wilson(n_ok, n),
               "by_gold_position": {f"{p}|{'ok' if o else 'miss'}": c
                                    for (p, o), c in sorted(by_pos.items())}}
        print(f"\nclosed-book two-option: {n_ok}/{n} = {n_ok / n:.3f} (chance 0.50)")
        print("  by gold position: " + ", ".join(f"{p}:{by_pos[(p, True)]}/{by_pos[(p, True)] + by_pos[(p, False)]}"
                                                for p in ("A", "B")))
    else:
        out = {"control": control, "per_model": {}}
        print(f"\n{control}-alone")
        for model in MODELS:
            rs = [r for r in rows if r.get("model") == model]
            ok = n = 0
            chance = 0.0
            prompts = 0
            for r in rs:
                uid = r["uid"]
                answers = distinct([c.get("answer") for c in CAND[model][uid]["candidates"]])
                options = answers if control == "judge" else answers + [GOLD[uid]]
                chosen = LETTERS.index(max(r["probs"], key=r["probs"].get))
                ok += int(task.is_correct(options[chosen], GOLD[uid]))
                chance += 1 / len(options)
                n += 1
                prompts += len(options)
            out["per_model"][model] = {"n": n, "correct": ok,
                                       "accuracy": ok / n if n else None,
                                       "chance": chance / n if n else None,
                                       "ci95_wilson": wilson(ok, n),
                                       "mean_options": prompts / n if n else None}
            print(f"  {model:10s} {ok}/{n} = {ok / n:.3f}  chance={chance / n:.3f}  "
                  f"mean options={prompts / n:.1f}")
        vals = [v["accuracy"] for v in out["per_model"].values() if v["accuracy"] is not None]
        out["macro_accuracy"] = sum(vals) / len(vals) if vals else None
        print(f"  macro = {out['macro_accuracy']:.3f}")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {json_out}")


def main() -> int:
    global GOLD, CAND, QUESTIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "eval"])
    ap.add_argument("--control", choices=["judge", "gold", "closedbook"], required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    QUESTIONS = {r["uid"]: r for r in read_jsonl(HERE / "questions.jsonl")}
    GOLD = {uid: (task.extract_answer(r["answer"]) or r["answer"])
            for uid, r in QUESTIONS.items()}
    CAND = {m: {r["uid"]: r for r in read_jsonl(HERE / "candidates" / m / "test.jsonl")}
            for m in MODELS}
    if a.action == "build":
        build(a.control)
    else:
        eval_control(a.control, a.results, a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
