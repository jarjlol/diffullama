"""OpenRouter decision requests for HumanEval candidates (listwise)."""
from __future__ import annotations

import argparse
import json

from common import WORK, read_jsonl
from decision_rerank import LETTERS
from humaneval_exec import candidate_code, problems

HERE = WORK / "humaneval"
OUT = WORK / "openrouter"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.split}.he.prompts.jsonl"
    ps = problems()
    rows = read_jsonl(HERE / "candidates" / a.model / f"{a.split}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            p = ps[r["uid"]]
            codes = [candidate_code(c["text"], p["prompt"], p["entry_point"]) for c in r["candidates"]]
            req = {
                "uid": r["uid"], "model": a.model, "n": len(codes),
                "state": {"problem": p["prompt"],
                          "candidates": {LETTERS[i]: t for i, t in enumerate(codes)}},
                "questions": {"best": {
                    "type": "choice",
                    "instructions": "Which candidate correctly implements the function described by the docstring and will pass the hidden tests?",
                    "criteria": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(codes))},
                }},
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} requests -> {path} ({path.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
