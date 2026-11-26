"""AR-scorer features for the GSM-Symbolic candidates.

    python3 gsm_symbolic_ar.py --model dream-base --scorer qwen7b --cut
    python3 gsm_symbolic_ar.py --model llada-base --scorer qwen7b --cut

Same scorer, context and feature names as score_candidates.py, but rooted at
work/gsm_symbolic. Writes work/gsm_symbolic/features/<model>/test.<scorer>[-cut].jsonl,
resumable by uid. Use the same venv/process rules as score_candidates.py: run in
a process where attention_patch has NOT been applied (HFScorer aborts otherwise).
"""
from __future__ import annotations

import argparse
import time

from common import WORK, append_jsonl, cfg, read_jsonl, task
from score_candidates import HFScorer, cut_after_answer

HERE = WORK / "gsm_symbolic"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scorer", required=True, help="a key of config.json 'scorers'")
    ap.add_argument("--cut", action="store_true",
                    help="score text only up to the '#### N' line; names ar.<scorer>-cut.*")
    a = ap.parse_args()
    name = f"{a.scorer}-cut" if a.cut else a.scorer
    src = HERE / "candidates" / a.model / "test.jsonl"
    out = HERE / "features" / a.model / f"test.{name}.jsonl"
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in read_jsonl(src) if r["uid"] not in done]
    print(f"{a.model} x {name}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return 0
    sc = HFScorer(cfg()["scorers"][a.scorer])
    prep = cut_after_answer if a.cut else task.trim
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        texts = [prep(cd["text"]) for cd in r["candidates"]]
        res = sc.nll(f"Question: {r['question']}\nAnswer:", texts)
        append_jsonl(out, {"uid": r["uid"], "features": [
            {f"ar.{name}.nll_sum": s, f"ar.{name}.n_tokens": n} for s, n in res]})
        if k % 10 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el / k:.2f}s/question  eta {el / k * (len(todo) - k) / 60:.0f} min",
                  flush=True)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
