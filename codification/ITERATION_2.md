# Codification — iteration 2 (DRAFT, started 2026-11-21; numbers drop in when the runs finish)

Structure follows `ITERATION_1.md`. Everything marked **TODO** or **not run** is empty on purpose: no number appears here
without the command that produced it. Paper results freeze: **Sun 22 Nov 2026, 18:00 IST**. Later extra-credit results go
into the report (#15), not the paper.

Status of this file at drafting time (Fri 20 Nov 2026, about 23:15 IST on the Blackwell clock): **no iteration-2 GPU run has
finished.** Both lanes were killed at 22:56 (see "Operations note"). The only iteration-2 results so far are the fidelity
checks in I-21.

## What iteration 2 changed, and why

The brief (see `ITERATION_1.md`, "What the brief asks for") ends with: compare with the state of the art or the baseline ideas.
Iteration 1 found that the idea as specified loses to majority vote and that the best Arbor selector ties it
(`ITERATION_1.md` §Results). Iteration 2 asks what that result depends on.

Plan and priorities: `ITERATION_2_PROMPTS.md`, "Priority tiers". Scope decision: issue #7
(<https://github.com/jarjlol/diffullama/issues/7>). A human reply on #7 chose the full iteration 2 as the target, run in
priority tiers with the paper-critical Tier 1 first (comment 7045012389, 2026-11-20); confirmation from the other members was
still pending when this was written [unverified].

| Tier | Item | Prompt | Why | Status |
|---|---|---|---|---|
| 1 | Fidelity check: greedy, published settings, 100 test questions, Dream and LLaDA | A2 step 3 | Our first-sample accuracy is 36 points (Dream, 41.0 vs 77.2) and 22 points (LLaDA, 48.7 vs 70.3) below published; rule out a pipeline bug | **done for dream-sampler and llada-published; dream-published stopped at 55 of 100** (I-21) |
| 1 | Step-budget arm: K = 16 and 32 on 500 test questions, Dream and LLaDA | A1 | Part (b) of problem P1 (rerank vs more steps) was unanswered | **not finished**: LLaDA K = 16 had 65 of 500 questions at 22:56; Dream K = 16 not started |
| 1 | Re-score the iteration-1 64-step candidates on the text cut after `#### N` | `score_candidates.py --cut` | Junk after the answer line confounds the AR scores (I-22) | **not run** (queued) |
| 1 | Share cached items with Neel; failure classes of majority vote (N4); SOTA table (N3) | A2 step 1, N4, N3 | The paper must place us against published methods | not in this file's scope |
| 2 | DLM self-score (the model's own likelihood) as a selector | A2 step 2, `self_score.py` | Obvious competitor to an external AR scorer; gap `dlm.selfscore` in iteration 1 | **not run** (queued) |
| 2 | Per-question preservation test on Dream | `per_question_preservation.py` | Tests the "preservation predicts rerank success" claim at n = 500 instead of n = 4 | **not run** (queued) |
| 2 | 256-step majority vote on the 100 fidelity questions | `generate_candidates.py --steps 256 --limit 100` | Separates the sampler effect from the selector effect | **not run** (queued) |
| 2/3 | Arbor rerun with majority vote as the seed, more models, second dataset | A3, A4 | Only if Tier 1 and 2 are done and clean | **not started** |

Code added for iteration 2 (all on `iter2/aaditya` and `iter2/aaditya-worker`):

| Commit | What |
|---|---|
| `dd8b2ec` | `--steps` override in `generate_candidates.py` (output goes to `candidates/<model>@k<steps>/`); `fidelity_check.py` |
| `0cb3b4a` | `step_arm.py`: dQ_steps, dQ_rerank, A_rerank |
| `104dddf` | `score_candidates.py --cut`: score the text only up to the end of the `#### N` line |
| `1bb05df` .. `a73da33` | `self_score.py` (Monte-Carlo masked likelihood), `per_question_preservation.py` |
| `1aa13a6`, `97527de` | `step_arm.py` paired tests and K = 256 table; `make_iter2_reports.py` |

## Issue log (continued from I-20)

| # | Issue | Evidence | Fix |
|---|---|---|---|
| I-21 | **Our first-sample accuracy is far below the published numbers, and the gap is explained by the sampler, not by a pipeline bug.** | `python3 fidelity_check.py --report` on Blackwell, 2026-11-20: `dream-sampler n=100 greedy 0.810 \| iter-1 first sample on same 100 Qs 0.410`; `llada-published n=100 greedy 0.730 \| iter-1 first sample on same 100 Qs 0.510`; `dream-published n=100 greedy 0.800 \| iter-1 first sample on same 100 Qs 0.410` (this run stopped at 55 on 20 Nov, see the operations note, and finished after the restart; `results/iter2/fidelity.md`). Iteration-1 majority vote on the same first 100 questions: Dream 0.70, LLaDA 0.75 (oracle 0.86 / 0.93). Published: Dream 77.2 (8-shot), LLaDA 70.3 (4-shot) (`ITERATION_1.md` "Published numbers"). | Greedy decoding at each model's published sampler (256 steps, temperature 0) on our own prompt and our own answer extraction reaches the published level (Dream 81.0 vs 77.2; LLaDA 73.0 vs 70.3), so prompt format and extraction are not what lowered iteration 1. What iteration 1 changed is temperature 0.7 and 64 steps (I-12, I-16), which are needed for 10 distinct candidates. The two are not separated yet: the 256-step majority runs are queued. n = 100 per config, so about 4 points standard error. [unverified: which of temperature and step count explains the gap] |
| I-22 | **Most candidates carry junk after the `#### N` answer line, and the AR scorers scored it.** `task.trim` only cuts at the next `Question:` marker, so repeated words and unrelated sentences after the answer stay in the scored text. | First 100 test questions, 1000 candidates per model, trimmed text (`task.trim`): candidates with more than 20 characters after the end of the `#### ...` line: **Dream 660, LLaDA 497**; after the number itself: **Dream 749, LLaDA 583** (the senior session counted 753 and 581; the difference is definition noise). Median characters after the `####` line, among candidates that have one: Dream 375, LLaDA 60. Mean answer tokens (Dream tokenizer): Dream trimmed 246 vs cut 119, LLaDA trimmed 160 vs cut 91. Example: `... #### 9000 metersWednesday, December 1, 2004 \xa0 Facts of the ... the the the the` (Dream, `gsm8k-test-0003`, candidate 0). | `score_candidates.py --cut` (commit `104dddf`, `cut_after_answer()`): text up to the end of the `#### N` line; candidates without `####` stay as `task.trim` gives them. `self_score.py` imports the same function. Grading is unaffected (`extract_answer` reads the number after `####`). **Outcome** (`results/iter2/cpu/cvt_dream_test.txt`, `cvt_llada_test.txt`, 500 test questions, paired 95% bootstrap CI and exact McNemar): AR reranking still trails majority vote on the cut text. Dream: `qwen7b` summed NLL 0.488 (cut) vs 0.358 (trimmed), +0.130 [+0.090, +0.170], 90/25, p < 0.001; cut `qwen7b` minus majority -0.236 [-0.278, -0.194], 14/132. LLaDA: 0.522 (cut) vs 0.524 (trimmed), -0.002 [-0.026, +0.022], 19/20, p = 1.0; cut minus majority -0.210 [-0.252, -0.172], 11/116. The earlier prediction (LLaDA's tails are shorter, median 60 vs 375 characters, so the cut changes Dream's AR scores more than LLaDA's) held for the summed-NLL selector; the per-token selector got worse on Dream (0.338 to 0.276) [not examined further] |

## Operations note: GPU interruption

On Fri 20 Nov 2026 at 22:56 IST another user of the shared `antpc` workstation ran `sudo pkill -u antpc -f lane.sh` and
`sudo pkill -u antpc -f /home/antpc/venvs/` (read from `journalctl` by the senior session; not checked by the worker
[unverified]). Both iteration-2 lanes and every Python process in our venvs were killed. State at that point on
Blackwell (`wc -l` of the git-ignored `work/` files): `candidates/llada-base@k16/test.jsonl` 65 of 500 rows; no
`dream-base@k16` or `@k32`; `fidelity/dream-published.jsonl` 55 of 100. The queued jobs were stopped and
`~/codify/iter2/STOP_gpu0` and `STOP_gpu1` were set, so no GPU job runs until the interruption is cleared. Generation resumes by question
id, so the 65 LLaDA rows are kept.

## Results

Fill from the commands shown; do not interpret in the tables. All on GSM8K test, N = 10, temperature 0.7, 64 steps unless
the budget says otherwise.

### 1. Fidelity (done for two configs)

Command: `python3 fidelity_check.py --report` (in `codification/pipeline/`, Blackwell).

| Config | n | Greedy (published settings) | Iter-1 first sample, same Qs | Iter-1 majority, same Qs | Published |
|---|---|---|---|---|---|
| Dream, our 4-shot prompt, 256 steps (`dream-sampler`) | 100 | 0.810 | 0.410 | 0.700 | 77.2 (8-shot) |
| Dream, published 8-shot CoT prompt (`dream-published`) | 100 | 0.800 | 0.410 | 0.700 | 77.2 (8-shot) |
| LLaDA, our 4-shot prompt, 256 steps (`llada-published`) | 100 | 0.730 | 0.510 | 0.750 | 70.3 (4-shot) |

### 2. Cut-text re-scoring of iteration 1 (does AR reranking still lose?): not run

Commands: `python3 score_candidates.py --model <m> --split test --scorer <s> --cut`, then
`python3 ../rerank_task/baselines.py --split test`. Scorer names carry the suffix `-cut`.

| Model | first | majority | oracle | best AR, trimmed text (name) | best AR, cut text (name) | Notes |
|---|---|---|---|---|---|---|
| dream-base | TODO | TODO | TODO | TODO | TODO | |
| llada-base | TODO | TODO | TODO | TODO | TODO | |

### 3. Step-budget arm (K = 16, 32, 64): not finished

Command: `python3 step_arm.py --json work/step_arm.json` (and `python3 make_iter2_reports.py`).

| Model | K | first | majority | oracle | best AR (name) | gen s/question | n |
|---|---|---|---|---|---|---|---|
| dream-base | 16 | TODO | TODO | TODO | TODO | TODO | |
| dream-base | 32 | TODO | TODO | TODO | TODO | TODO | |
| dream-base | 64 | TODO | TODO | TODO | TODO | TODO | |
| llada-base | 16 | TODO | TODO | TODO | TODO | TODO | |
| llada-base | 32 | TODO | TODO | TODO | TODO | TODO | |
| llada-base | 64 | TODO | TODO | TODO | TODO | TODO | |

dQ_steps(16 to 32), dQ_rerank(K) and A_rerank = dQ_rerank(16) - dQ_steps(16 to 32), with paired 95% bootstrap CIs and McNemar p:
TODO (printed by `step_arm.py`).

### 4. DLM self-score: not run

Commands: `python3 self_score.py --model <m> --split <dev|test>` (feature `dlm.selfscore`, cut text), then `baselines.py`,
`paired_stats.py` against majority vote on dev and gate only (not the final 300).

| Model | split | first | majority | dlm.selfscore | best AR | n | paired vs majority |
|---|---|---|---|---|---|---|---|
| dream-base | dev | TODO | TODO | TODO | TODO | | TODO |
| dream-base | test | TODO | TODO | TODO | TODO | | TODO |
| llada-base | dev | TODO | TODO | TODO | TODO | | TODO |
| llada-base | test | TODO | TODO | TODO | TODO | | TODO |

### 5. 256-step runs on the 100 fidelity questions: not run

Commands: `generate_candidates.py --model <m> --split test --limit 100 --steps 256`, then `step_arm.py` (K = 256 table).

| Model | K | first | majority | oracle | greedy (I-21) | n |
|---|---|---|---|---|---|---|
| dream-base | 64 | TODO | TODO | TODO | | 100 |
| dream-base | 256 | TODO | TODO | TODO | 0.810 | 100 |
| llada-base | 64 | TODO | TODO | TODO | | 100 |
| llada-base | 256 | TODO | TODO | TODO | 0.730 | 100 |

### 6. Per-question preservation (exploratory): not run

Commands: `per_question_preservation.py extract --model dream`, `... extract --model qwen7b`, `... analyze --split test`.
Predictor: layer-mean centred cosine of the pooled question states of Dream vs Qwen2.5-7B. Outcomes and methods are in the
script docstring and the header of its output. AR pick: `qwen7b-cut` once it exists.

| Outcome | n | events | AUC | permutation p | OR per SD [95% CI] |
|---|---|---|---|---|---|
| first wrong and AR pick right (O1) | TODO | | | | |
| majority right (O2) | TODO | | | | |
| AR pick right among first-wrong (O3) | TODO | | | | |

## What this says

TODO. Written only after the tables above are filled, and only from them.

## Not verified / limits

- Everything marked TODO or "not run" above has no number yet.
- Fidelity numbers are 100 questions each; standard error about 4 points.
- The cause of the gap in I-21 (temperature versus step count) is not separated yet.
- The self-score uses 8 Monte-Carlo draws per candidate (LLaDA's own evaluation default is 128) and a separate random seed per candidate, so the 10 candidates of a question are ranked on independent noisy estimates; it is a deliberately cheap, high-variance selector. How much of the 30.4% (Dream, 500 questions) or 45.0% (LLaDA, gate) is estimator noise is [unverified]; scoring a subset with the same masks for all candidates (common random numbers) was not tried [review of PR #41, neel's agent].
- The per-question preservation analysis used hidden states extracted in fp32 on the CPU, not bf16 on a GPU; whether the cosines would differ is [unverified].
- The senior session's description of the interruption is quoted from its reading of `journalctl`.
- Generation seconds are measured on shared GPUs and are not clean compute figures.
- Decision #7: a human reply exists; the rest of the team's confirmation was pending at drafting time.
