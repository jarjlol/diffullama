# N4 — CPU analysis for iteration 2 (neel's agent, 21 Nov 2026)

Spec: `ITERATION_2_PROMPTS.md` § N4. Branch `iter2/neel` (from `iter2/aaditya-worker` @ 1d71241).
Status: step 3 **done** (below). Steps 1, 2 and the extraction audit are **scripted and tested on synthetic
items, not yet run on real items**: the cached `items.jsonl` (git-ignored) live only on Blackwell, and the Drive
copy is not reachable from neel's agent's container. Tuned-majority bar: already frozen by aryan/codex
(`results/iter2/aryan/support-N4/`: on gate 200 it equals plain majority, 0 wins / 0 losses on both models).

## Step 3 — leave-one-out on CKA vs headroom (done)

```
$ python3 codification/n4/loo_cka.py codification/results/iter1/numbers/analyze_test500.txt
models with an AR base: diffugpt-m, diffugpt-s, diffullama, dream-base
selector                  all  -diffugpt-m  -diffugpt-s  -diffullama  -dream-base
ar.gpt2.sum             +1.00        +1.00        +1.00        +1.00        +1.00
ar.gpt2.mean            +0.40        +0.50        +1.00        +0.50        -0.50
ar.llama2-7b.sum        +0.80        +1.00        +1.00        +0.50        +0.50
ar.llama2-7b.mean       +0.80        +1.00        +1.00        +0.50        +0.50
ar.qwen0.5b.sum         +0.80        +1.00        +1.00        +0.50        +0.50
ar.qwen0.5b.mean        +1.00        +1.00        +1.00        +1.00        +1.00
ar.qwen7b.sum           +1.00        +1.00        +1.00        +1.00        +1.00
ar.qwen7b.mean          +0.80        +1.00        +1.00        +0.50        +0.50
ar.tinyllama.sum        +0.80        +1.00        +1.00        +0.50        +0.50
ar.tinyllama.mean       +0.80        +1.00        +1.00        +0.50        +0.50
```

Check: the "all" column reproduces `analyze.py`'s reported rho for every selector (+1.0 / +0.8 / +0.4, as in
`ITERATION_1.md` §Results). Reading: dropping any one model keeps rho >= +0.5 for 9 of 10 selectors; the one
sign flip is gpt2-mean without Dream (+0.40 -> -0.50). Every leave-one-out rho is over 3 points, so it can only
take a few values (±1, ±0.5) and carries no evidential weight. **Caveat (I-22):** these headrooms come from
iteration-1 AR scores, which scored text after the `#### N` line; rerun this on the cut re-score
(`analyze.py` output on the cut features) before the paper uses it.

## Steps 1, 2 and the extraction audit — to run on Blackwell (CPU only, seconds)

```
cd ~/codify/worker && python3 codification/n4/n4_items.py --task-dir ~/arbor-rerank-iter2 \
    --models dream-base llada-base --out ~/codify/iter2/neel/n4
```

(`--task-dir` = any `rerank_task` copy whose `data/dev` and `data/test` hold the cached items and gold; the
script refuses a split named `final`.) Rules, fixed before looking at real items: see the docstring of
`n4/n4_items.py` (primary class order arithmetic_slip → close_second → shared_wrong → other; independent
extraction_suspect flag; text cut after the `#### N` line as `score_candidates.py --cut`).

Synthetic test (`rerank_task` copy, 4 real dev uids, one planted case per class plus one majority-right
question and one planted extraction mismatch): classes 1/1/1/0 of 3 majority-wrong-oracle-right questions,
extraction_suspect 1, the majority-right question skipped — as planted.

## Failure classes for A3

_Pending the real run above (dev counts per model, labels only)._
