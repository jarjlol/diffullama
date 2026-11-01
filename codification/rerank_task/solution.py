"""solution.py -- the ONLY file Arbor may edit.

Implement select(item) -> int: the index of the candidate to submit as the answer.

`item` is one question answered by one diffusion LM:

    {"uid": "gsm8k-test-0042", "model": "diffullama", "question": "...",
     "candidates": [{"text": "...", "answer": "18" | None,
                     "features": {"len_chars": 412,
                                  "ar.gpt2.nll_sum": 41.2, "ar.gpt2.n_tokens": 63,
                                  "ar.<scorer>.nll_sum": ..., "dlm.selfscore": ...}}, ...]}

No gold answer is ever passed in. Use only the question, the candidate texts,
their extracted answers and the cached features. The loop must stay CPU-only:
features needing a GPU are precomputed by ../pipeline/score_candidates.py.

SEED (idea 1.1 as specified): pick the candidate with the lowest summed NLL
under the GPT-2 scorer. Known weaknesses, left for the search to fix: summed
NLL favours short candidates, and it ignores agreement between candidates.
"""
from __future__ import annotations


def select(item: dict) -> int:
    cands = item["candidates"]
    scored = [(c["features"].get("ar.gpt2.nll_sum"), i) for i, c in enumerate(cands)]
    scored = [(s, i) for s, i in scored if s is not None]
    if not scored:
        return 0
    return min(scored)[1]
