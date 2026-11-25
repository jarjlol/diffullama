"""AR-likelihood scores for HumanEval candidates (Qwen2.5-7B, the GSM8K verifier).

    CUDA_VISIBLE_DEVICES=1 ~/venvs/scorer/bin/python humaneval_ar.py --model dream-base

NLL is summed over the generated code only, conditioned on the problem prompt,
capped at 512 code tokens. The prompt itself is never scored. Writes
work/humaneval/features/<model>/<split>.<scorer>.jsonl, aligned with the
candidate order in work/humaneval/candidates/<model>/<split>.jsonl.
"""
from __future__ import annotations

import argparse
import time

from common import WORK, cfg, read_jsonl
from humaneval_exec import candidate_code
from score_candidates import HFScorer

HERE = WORK / "humaneval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--scorer", default="qwen7b")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    scorer = HFScorer(cfg()["scorers"][a.scorer])
    rows = read_jsonl(HERE / "candidates" / a.model / f"{a.split}.jsonl")
    rows = rows[: a.limit] if a.limit else rows
    out = HERE / "features" / a.model / f"{a.split}.{a.scorer}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{a.model}/{a.split}/{a.scorer}: {len(done)} done, {len(todo)} to go -> {out}")

    t0 = time.time()
    for k, r in enumerate(todo, 1):
        feats = []
        for c in r["candidates"]:
            code = candidate_code(c["text"], r["question"], "")
            ids = scorer.tok(code, add_special_tokens=False, truncation=True, max_length=512).input_ids
            code = scorer.tok.decode(ids)
            nll, n = scorer.nll(r["question"], [code])[0]
            feats.append({f"ar.{a.scorer}.nll_sum": nll, f"ar.{a.scorer}.n_tokens": n})
        with open(out, "a", encoding="utf-8") as f:
            f.write(__import__("json").dumps({"uid": r["uid"], "features": feats}) + "\n")
        if k % 10 == 0:
            print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
