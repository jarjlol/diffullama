"""Decision-model scores for HumanEval candidates.

    CUDA_VISIBLE_DEVICES=1 ~/venvs/vllm/bin/python humaneval_decision.py --model dream-base --mode listwise

Listwise asks which candidate will pass the hidden tests; pointwise asks whether
one candidate implements the docstring. Neither sees the tests or any execution
result. Writes work/humaneval/features/<model>/<split>.decision-<mode>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time

from common import WORK, read_jsonl
from decision_rerank import LETTERS, Pplx
from humaneval_exec import candidate_code

HERE = WORK / "humaneval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--mode", choices=["listwise", "pointwise"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = read_jsonl(HERE / "candidates" / a.model / f"{a.split}.jsonl")
    rows = rows[: a.limit] if a.limit else rows
    out = HERE / "features" / a.model / f"{a.split}.decision-{a.mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{a.model}/{a.split}/{a.mode}: {len(done)} done, {len(todo)} to go -> {out}")

    d = Pplx()
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        codes = [candidate_code(c["text"], r["question"], "") for c in r["candidates"]]
        if a.mode == "listwise":
            criteria = {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(codes))}
            resp = d.decider.decide({
                "state": {"problem": r["question"],
                          "candidates": {LETTERS[i]: c for i, c in enumerate(codes)}},
                "questions": {"best": {
                    "type": "choice",
                    "instructions": "Which candidate correctly implements the function described by the docstring and will pass the hidden tests?",
                    "criteria": criteria,
                }},
            })
            ans = resp["answers"]["best"]["probabilities"]
            feats = [{"decision.listwise.p": float(ans[LETTERS[i]])} for i in range(len(codes))]
        else:
            feats = []
            for c in codes:
                resp = d.decider.decide({
                    "state": {"problem": r["question"], "candidate": c},
                    "questions": {"correct": {
                        "type": "noul",
                        "instructions": "Does this candidate correctly implement the function described by the docstring and will it pass the hidden tests?",
                    }},
                })
                feats.append({"decision.pointwise.p_yes": float(resp["answers"]["correct"]["noul"])})
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"uid": r["uid"], "features": feats}) + "\n")
        if k % 5 == 0:
            print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
