"""Build OpenRouter decision requests for the GSM-Symbolic candidate pool.

    python3 gsm_symbolic_prompts.py --mode listwise
    python3 gsm_symbolic_prompts.py --mode pointwise

State, schema and wording are copied verbatim from make_openrouter_prompts.py
(the GSM8K run), and candidates go through the same cut-text function
(decision_rerank.candidate_text: task.trim, then cut at the '#### N' line).
Only the candidate path changes -- work/gsm_symbolic/candidates/<model>/test.jsonl
instead of work/candidates/<model>/test.jsonl.

Writes work/gsm_symbolic/prompts/test.<mode>.prompts.jsonl: one line per
(model, question) for listwise, one per (model, question, candidate) for
pointwise. The file is POSTed to the host with network access. Gold answers in
questions.jsonl are never read here.
"""
from __future__ import annotations

import argparse
import json

from common import WORK, read_jsonl
from decision_rerank import LETTERS, candidate_text

HERE = WORK / "gsm_symbolic"
OUT = HERE / "prompts"


def listwise_request(row: dict, model: str) -> dict:
    texts = [candidate_text(c["text"]) for c in row["candidates"]]
    return {
        "uid": row["uid"], "model": model, "n": len(texts),
        "state": {"problem": row["question"],
                  "candidates": {LETTERS[i]: t for i, t in enumerate(texts)}},
        "questions": {"best": {
            "type": "choice",
            "instructions": "Which candidate solution reaches the correct final answer to the math problem?",
            "criteria": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(texts))},
        }},
    }


def pointwise_requests(row: dict, model: str) -> list[dict]:
    texts = [candidate_text(c["text"]) for c in row["candidates"]]
    return [{
        "uid": row["uid"], "model": model, "idx": j, "n": len(texts),
        "state": {"problem": row["question"], "candidate": t},
        "questions": {"correct": {
            "type": "noul",
            "instructions": "Does this candidate's final answer agree with the correct answer to the math problem?",
        }},
    } for j, t in enumerate(texts)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    ap.add_argument("--mode", choices=["listwise", "pointwise"], default="listwise")
    ap.add_argument("--limit", type=int, default=0, help="first N completed questions per model")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.split}.{a.mode}.prompts.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for model in a.models:
            rows = read_jsonl(HERE / "candidates" / model / f"{a.split}.jsonl")
            rows = rows[: a.limit] if a.limit else rows
            for r in rows:
                reqs = ([listwise_request(r, model)] if a.mode == "listwise"
                        else pointwise_requests(r, model))
                for req in reqs:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n += 1
            print(f"{model}: {len(rows)} questions, "
                  f"{len(rows) if a.mode == 'listwise' else sum(len(r['candidates']) for r in rows)} requests")
    print(f"wrote {n} requests -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
