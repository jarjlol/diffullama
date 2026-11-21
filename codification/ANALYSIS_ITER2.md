# N4 — CPU analysis for iteration 2 (neel's agent, 21 Nov 2026)

Spec: `ITERATION_2_PROMPTS.md` § N4. Branch `iter2/neel` (from `iter2/aaditya-worker` @ 1d71241).
Status: steps 1, 2 and 3 **done** (below; step 1–2 run on Blackwell by aaditya's agent, 21 Nov 09:23 UTC, #4 comment from Aadityahv). The extraction audit is **done** (below; the 20 candidates were pasted by arjun's agent on #4, 21 Nov 11:36 UTC, and read by hand). Earlier status: steps 1–2 were scripted and tested on synthetic
items before the real run: the cached `items.jsonl` (git-ignored) live only on Blackwell, and the Drive
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

## Steps 1–2 — real run (aaditya's agent, Blackwell CPU, 1 s, code @ 024ab82, `--task-dir ~/arbor-rerank-iter2`)

Answer diversity (iteration-1 candidates, N = 10; test = the 200-question gate split):

| Model | Split | Questions | Mean distinct answers | Mean entropy (bits) | No-answer share |
|---|---|---|---|---|---|
| dream-base | dev | 200 | 5.755 | 2.062 | 0.001 |
| dream-base | test | 200 | 6.15 | 2.194 | 0.002 |
| llada-base | dev | 200 | 4.64 | 1.629 | 0.0075 |
| llada-base | test | 200 | 5.105 | 1.818 | 0.001 |

Reading: Dream's candidates are more diverse than LLaDA's (about 1 more distinct answer per question, +0.4 bits). This is a confound for any preservation-vs-headroom comparison between the two, since rerankable headroom grows with diversity. The extraction-suspect flag fired on 1 Dream question and 0 LLaDA questions, so answer extraction is unlikely to explain majority-vote failures (to be confirmed by reading the 20-candidate audit).

## Failure classes for A3

Dev split, questions where majority vote is wrong and some candidate is right (rules fixed before the run; see `n4/n4_items.py`). Counts and labels only, no gold:

- **arithmetic_slip** — dream-base 10/29, llada-base 11/26. Most candidates voting for the wrong plurality contain a calculation that does not re-check. Example: the plurality answer comes from a step like "a × b = c" with c wrong.
- **close_second** — dream-base 13/29, llada-base 7/26. The correct answer had at most one vote fewer than the plurality. Example: 3 votes vs 2.
- **shared_wrong** — dream-base 3/29, llada-base 3/26. At least half of the candidates agree on the same wrong answer with no detectable arithmetic slip. Example: a misread of the question repeated across samples.
- **other** — dream-base 3/29, llada-base 5/26.

Implications for A3: about 34% of Dream's (10/29) and 42% of LLaDA's (11/26) recoverable failures are arithmetic slips, which a per-candidate re-check can see. This is the signal iteration-1's merged rule 4.1 used, and it tied majority. That suggests a stronger arithmetic check (all steps, and whether the answer follows from the last step: S³'s s2/s3, `SOTA.md` S-2) rather than a new signal. close_second is the largest Dream class. Any vote weighting that moves one vote flips these, so they are the cheapest wins and also the easiest overfits; test on gate, not dev.

## Extraction audit (20 candidates from the dev majority-wrong-oracle-right questions, seed 0; read by hand)

Source: `~/codify/iter2/neel/n4/audit.txt` on Blackwell, pasted verbatim by arjun's agent on #4 (comment of 2026-11-21 11:36 UTC). Each candidate is shown as its tail (cut after `#### N`) with the extracted answer; no gold.

| Verdict | Count | Candidates |
|---|---|---|
| Extracted the number the candidate wrote after `####` (correct extraction) | 15 | llada 3569/4, 5567/0, 6389/7, 4508/4, 0558/0, 6373/8, 2728/6, 6551/6, 7360/5, 0558/7; dream 1765/1, 7277/5, 4814/3, 2993/7, 5034/0 |
| Candidate's `####` value contradicts its own reasoning (model slip, extractor faithful) | 1 | llada 6845/3: reasoning ends "$4.60", marker says `#### 460` |
| Degenerate repetition with no `####`; the last-number fallback returns a noise number | 4 | dream 6088/3 → 1, 4814/2 → 54, 7360/8 → 12, 7084/6 → 7550 |

**No extraction mismatch** in the sense of the N4 spec: whenever a candidate wrote a final answer, `task.extract_answer` took it. Two findings follow:
- **Degenerate Dream samples cast noise votes:** 4 of the 9 Dream candidates in the sample loop on repeated text and get a fallback "answer". Filtering such candidates before voting (S³'s non-degeneracy component, `SOTA.md` S-2; Leap idea 3 in `memory/RESEARCH_REPORT.md`) is a cheap, CPU-only test for A3.
- **Junk after `####` is common** (11 of the 15 clean ones run on into unrelated text: a new "Question:", instructions, page fragments). This is the I-22 pattern, and grading is unaffected because the number is read right after `####`.
