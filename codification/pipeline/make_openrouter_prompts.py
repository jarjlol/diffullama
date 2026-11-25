"""Build the exact OpenRouter decision requests for the GSM8K test split.

    python3 make_openrouter_prompts.py --split test

Writes work/openrouter/prompts.jsonl: one line per (model, question) with the
same state and schema the local `Pplx` backend would send. The file is moved to
the host that has network access; scoring never sees gold.
"""
from __future__ import annotations

import argparse
import json

from common import WORK, read_jsonl
from decision_rerank import LETTERS, candidate_text

OUT = WORK / "openrouter"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    ap.add_argument("--mode", choices=["listwise", "pointwise"], default="listwise")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.split}.{a.mode}.prompts.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for model in a.models:
            for r in read_jsonl(WORK / "candidates" / model / f"{a.split}.jsonl"):
                texts = [candidate_text(c["text"]) for c in r["candidates"]]
                if a.mode == "listwise":
                    req = {
                        "uid": r["uid"], "model": model, "n": len(texts),
                        "state": {"problem": r["question"],
                                  "candidates": {LETTERS[i]: t for i, t in enumerate(texts)}},
                        "questions": {"best": {
                            "type": "choice",
                            "instructions": "Which candidate solution reaches the correct final answer to the math problem?",
                            "criteria": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(texts))},
                        }},
                    }
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n += 1
                else:
                    for j, t in enumerate(texts):
                        req = {
                            "uid": r["uid"], "model": model, "idx": j, "n": len(texts),
                            "state": {"problem": r["question"], "candidate": t},
                            "questions": {"correct": {
                                "type": "noul",
                                "instructions": "Does this candidate's final answer agree with the correct answer to the math problem?",
                            }},
                        }
                        f.write(json.dumps(req, ensure_ascii=False) + "\n")
                        n += 1
    print(f"wrote {n} requests -> {path}  ({path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
