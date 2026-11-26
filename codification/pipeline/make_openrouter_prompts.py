"""Build the exact OpenRouter decision requests for the GSM8K test split.

    python3 make_openrouter_prompts.py --split test

Writes work/openrouter/prompts.jsonl: one line per (model, question) with the
same state and schema the local `Pplx` backend would send. The file is moved to
the host that has network access; scoring never sees gold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import WORK, read_jsonl
from decision_rerank import LETTERS, candidate_text

OUT = WORK / "openrouter"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    ap.add_argument("--mode", choices=["listwise", "pointwise"], default="listwise")
    ap.add_argument("--out", default=None,
                    help="override output path (e.g. the K=256 step-sensitivity run)")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N questions per model, sorted by uid (parameter sweeps)")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="keep only the first N candidates (N sweep); order is unchanged")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(a.out) if a.out else OUT / f"{a.split}.{a.mode}.prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for model in a.models:
            rows = read_jsonl(WORK / "candidates" / model / f"{a.split}.jsonl")
            if a.limit:                      # sweep subset: first N uids, deterministic
                rows = sorted(rows, key=lambda r: r["uid"])
            if a.limit:
                rows = rows[: a.limit]
            for r in rows:
                cands = r["candidates"][: a.max_candidates] if a.max_candidates else r["candidates"]
                texts = [candidate_text(c["text"]) for c in cands]
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
