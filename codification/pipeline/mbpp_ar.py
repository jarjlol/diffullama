"""AR-likelihood scores for MBPP candidates (Qwen2.5-7B)."""
from __future__ import annotations

import argparse
import json
import time

from common import WORK, cfg, read_jsonl
from humaneval_exec import extract_code, parsable_prefix
from score_candidates import HFScorer

HERE = WORK / "mbpp"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--scorer", default="qwen7b")
    a = ap.parse_args()
    scorer = HFScorer(cfg()["scorers"][a.scorer])
    rows = read_jsonl(HERE / "candidates" / a.model / "test.jsonl")
    out = HERE / "features" / a.model / f"test.{a.scorer}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{a.model}: {len(done)} done, {len(todo)} to go")
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        feats = []
        for c in r["candidates"]:
            code = parsable_prefix(extract_code(c["text"]))
            ids = scorer.tok(code, add_special_tokens=False, truncation=True, max_length=512).input_ids
            code = scorer.tok.decode(ids)
            nll, n = scorer.nll(r["question"], [code])[0]
            feats.append({f"ar.{a.scorer}.nll_sum": nll, f"ar.{a.scorer}.n_tokens": n})
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"uid": r["uid"], "features": feats}) + "\n")
        if k % 10 == 0:
            print(f"  {k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
