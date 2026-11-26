"""OpenRouter listwise requests for MBPP candidates."""
from __future__ import annotations

import argparse
import json

from common import WORK, read_jsonl
from decision_rerank import LETTERS
from humaneval_exec import extract_code, parsable_prefix

HERE = WORK / "mbpp"
OUT = WORK / "openrouter"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "test.mbpp.prompts.jsonl"
    rows = read_jsonl(HERE / "candidates" / a.model / "test.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            codes = [parsable_prefix(extract_code(c["text"])) for c in r["candidates"]]
            req = {"uid": r["uid"], "model": a.model, "n": len(codes),
                   "state": {"problem": r["question"],
                             "candidates": {LETTERS[i]: t for i, t in enumerate(codes)}},
                   "questions": {"best": {
                       "type": "choice",
                       "instructions": "Which candidate correctly implements the task and will pass the tests?",
                       "criteria": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(codes))},
                   }}}
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
