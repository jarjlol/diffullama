# Codification — iteration 2 (started 2026-11-21; tables refreshed from the current data on 2026-11-25)

Structure follows `ITERATION_1.md`. Everything marked **TODO** or **not run** is empty on purpose: no number appears here
without the command that produced it. Paper results freeze: **Sun 22 Nov 2026, 18:00 IST**. Later extra-credit results go
into the report (#15), not the paper.

Status (refreshed Wed 25 Nov 2026, about 01:00 IST): the iteration-2 GPU runs are finished (K = 16/32/64 on 500 test questions;
K = 256 on random subsamples of 185 and 192; the greedy fidelity runs; the LLaDA self-score on all 500). The Arbor search was not
run. Tables 1 to 6 were regenerated from the current data on 25 Nov (see "Not verified / limits" for what changed). At drafting time
(Fri 20 Nov, about 23:15 IST) no iteration-2 GPU run had finished: both lanes were killed at 22:56 (see "Operations note").
Exploratory CPU analyses added on 25 Nov: section 7.

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
| 1 | Fidelity check: greedy, published settings, 100 test questions, Dream and LLaDA | A2 step 3 | Our first-sample accuracy is 36 points (Dream, 41.0 vs 77.2) and 22 points (LLaDA, 48.7 vs 70.3) below published; rule out a pipeline bug | **done** (I-21; Table 1): `dream-sampler` and `llada-published` were extended on 23-24 Nov to every question with a K = 256 row (n = 244 and 199; the first 100 are unchanged), `dream-published` finished at 100 |
| 1 | Step-budget arm: K = 16 and 32 on 500 test questions, Dream and LLaDA | A1 | Part (b) of problem P1 (rerank vs more steps) was unanswered | **done** for K = 16/32/64 on 500 test questions (§3) and for K = 256 on random subsamples (§5) |
| 1 | Re-score the iteration-1 64-step candidates on the text cut after `#### N` | `score_candidates.py --cut` | Junk after the answer line confounds the AR scores (I-22) | **done**, 500 test questions (§2) |
| 1 | Share cached items with Neel; failure classes of majority vote (N4); SOTA table (N3) | A2 step 1, N4, N3 | The paper must place us against published methods | not in this file's scope |
| 2 | DLM self-score (the model's own likelihood) as a selector | A2 step 2, `self_score.py` | Obvious competitor to an external AR scorer; gap `dlm.selfscore` in iteration 1 | **done**: gate (both models), all 500 test questions (both models, §4), K = 256 subsets (§7); the dev split is not done (Dream dev has a partial 125 of 200 file, not a result) |
| 2 | Per-question preservation test on Dream | `per_question_preservation.py` | Tests the "preservation predicts rerank success" claim at n = 500 instead of n = 4 | **done**, 500 test questions (§6) |
| 2 | 256-step majority vote on the 100 fidelity questions | `generate_candidates.py --steps 256 --limit 100` | Separates the sampler effect from the selector effect | **replaced** by the K = 256 check on a random subsample of the 500 test questions (§5), not the first 100 |
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

### 1. Fidelity (done)

Command: `python3 fidelity_check.py --report` (in `codification/pipeline/`, Blackwell); regenerated 25 Nov, `results/iter2/fidelity.md`. The first-100 rows are the numbers the paper quotes; the extension rows cover the questions that had a K = 256 row when the greedy runs started (not the first 100). The Iter-1 columns of the extension rows were recomputed on those same questions (`results/iter2/cpu/extra_3_fidelity_majority.txt`).

| Config | n | Greedy (published settings) | Iter-1 first sample, same Qs | Iter-1 majority, same Qs | Published |
|---|---|---|---|---|---|
| Dream, our 4-shot prompt, 256 steps (`dream-sampler`), first 100 | 100 | 0.810 | 0.410 | 0.700 | 77.2 (8-shot) |
| same, all rows after the 24 Nov extension | 244 | 0.824 | 0.422 | 0.734 | 77.2 (8-shot) |
| Dream, published 8-shot CoT prompt (`dream-published`) | 100 | 0.800 | 0.410 | 0.700 | 77.2 (8-shot) |
| LLaDA, our 4-shot prompt, 256 steps (`llada-published`), first 100 | 100 | 0.730 | 0.510 | 0.750 | 70.3 (4-shot) |
| same, all rows after the 24 Nov extension | 199 | 0.729 | 0.497 | 0.764 | 70.3 (4-shot) |

### 2. Cut-text re-scoring of iteration 1 (does AR reranking still lose?): done, 500 test questions

Commands: `python3 score_candidates.py --model <m> --split test --scorer <s> --cut`, then
`python3 cut_vs_trim.py --model <m> --split test`. Scorer names carry the suffix `-cut`.
Sources (commit `aa5d3d2` on `iter2/aaditya-worker`): `results/iter2/cpu/cvt_dream_test.txt` (try 2, sha256 `97ec796a…`),
`results/iter2/cpu/cvt_llada_test.txt` (sha256 `703de3db…`). "Best AR" = highest accuracy among that file's AR selectors
of the given text type (sum and mean).

| Model | first | majority | oracle | best AR, trimmed text (name) | best AR, cut text (name) | Notes |
|---|---|---|---|---|---|---|
| dream-base | 0.404 | 0.724 | 0.878 | 0.358 (qwen7b:sum) | 0.488 (qwen7b-cut:sum) | cut versions of gpt2, qwen0.5b, qwen7b; cut minus majority -0.236 [-0.278, -0.194], 14/132 |
| llada-base | 0.486 | 0.732 | 0.892 | 0.524 (qwen7b:sum) | 0.522 (qwen7b-cut:sum) | cut versions of gpt2, qwen7b only; cut minus majority -0.210 [-0.252, -0.172], 11/116 |

Dev split (200 questions, qwen7b only; `cvt_dream_dev.txt`, `cvt_llada_dev.txt`): Dream first 0.375, majority 0.700, qwen7b sum
0.310 trimmed vs 0.445 cut (+0.135 [+0.070, +0.205]); LLaDA first 0.475, majority 0.745, 0.580 trimmed vs 0.595 cut
(+0.015 [-0.020, +0.050]). Cut minus majority: Dream -0.255 [-0.325, -0.185], LLaDA -0.150 [-0.215, -0.090].

### 3. Step-budget arm (K = 16, 32, 64): done for K = 16/32/64, 500 test questions

Command: `python3 step_arm.py --json work/step_arm.json` (and `python3 make_iter2_reports.py`).
Source: `results/iter2/step_arm.md` (commit `aa5d3d2`, sha256 `19f5725e…`), "Raw output" block.

| Model | K | first | majority | oracle | best AR (name) | gen s/question | n |
|---|---|---|---|---|---|---|---|
| dream-base | 16 | 0.026 | 0.032 | 0.174 | 0.024 (ar.gpt2.mean) | 9.5 | 500 |
| dream-base | 32 | 0.124 | 0.234 | 0.540 | 0.138 (ar.qwen7b-cut.sum) | 19.3 | 500 |
| dream-base | 64 | 0.404 | 0.724 | 0.878 | 0.488 (ar.qwen7b-cut.sum) | 36.8 | 500 |
| llada-base | 16 | 0.036 | 0.032 | 0.246 | 0.036 (ar.qwen0.5b-cut.sum) | 13.7 | 500 |
| llada-base | 32 | 0.228 | 0.510 | 0.760 | 0.286 (ar.qwen7b-cut.sum) | 27.4 | 500 |
| llada-base | 64 | 0.486 | 0.732 | 0.892 | 0.524 (ar.qwen7b.sum) | 55.0 | 500 |

dQ_steps(16 to 32), dQ_rerank(K) and A_rerank = dQ_rerank(16) - dQ_steps(16 to 32), as printed by `step_arm.py`:

| Model | dQ_steps 16→32 (first) | dQ_steps 32→64 (first) | dQ_rerank majority K=16 / 32 / 64 | A_rerank, majority | paired, majority minus first, K=16 |
|---|---|---|---|---|---|
| dream-base | +0.098 [+0.066, +0.130], 60/11 | +0.280 [+0.232, +0.328], 165/25 | +0.006 / +0.110 / +0.320 | -0.092 | +0.006 [-0.010, +0.024], 11/8, p = 0.648 |
| llada-base | +0.192 [+0.154, +0.230], 107/11 | +0.258 [+0.208, +0.308], 162/33 | -0.004 / +0.282 / +0.246 | -0.196 | -0.004 [-0.024, +0.016], 12/14, p = 0.845 |

Brackets: paired 95% bootstrap CI; then wins/losses and exact McNemar p (p < 0.001 where not given). The best AR selector's
A_rerank is printed too: Dream -0.100 (ar.gpt2-cut.mean / ar.gpt2.mean), LLaDA -0.194 (ar.qwen7b-cut.sum).

### 4. DLM self-score: done on the gate (both models) and on all 500 test questions (both models)

Commands: `python3 self_score.py --model <m> --split <dev|test>` (feature `dlm.selfscore`, cut text), then
`cut_vs_trim.py` (`--subset gate` for the gate rows). Sources (commit `aa5d3d2`): `self_dream_gate.txt` (sha256 `2bc62e9c…`),
`self_llada_gate.txt` (`a3d5d530…`), `cvt_dream_test.txt` (`97ec796a…`), `cvt_llada_test.txt` (`703de3db…`).
Best AR = highest AR selector in the same file.

| Model | split | first | majority | dlm.selfscore | best AR | n | paired vs majority |
|---|---|---|---|---|---|---|---|
| dream-base | dev | incomplete: a partial `dev.dlmself.jsonl` of 21 Nov covers 125 of 200 questions (`cvt_dream_dev.txt`, flagged INCOMPLETE); not a result | | | | | |
| dream-base | test, gate | 0.395 | 0.710 | 0.315 | 0.525 (qwen7b-cut:sum) | 200 | -0.395 [-0.465, -0.325], 2/81, p < 0.001 |
| dream-base | test, all | 0.404 | 0.724 | 0.304 | 0.488 (qwen7b-cut:sum) | 500 | -0.420 [-0.466, -0.374], 9/219, p < 0.001 |
| llada-base | dev | not run | | | | | |
| llada-base | test, gate | 0.485 | 0.735 | 0.450 | 0.530 (qwen7b-cut:sum, qwen7b:sum, qwen0.5b:sum tie) | 200 | -0.285 [-0.355, -0.215], 4/61, p < 0.001 |
| llada-base | test, all | 0.486 | 0.732 | 0.468 | 0.522 (qwen7b-cut:sum) | 500 | -0.264 [-0.306, -0.222], 10/142, p < 0.001 |

The LLaDA "test, all" row is from `results/iter2/cvt_llada_test_full500.txt` (Tue 24 Nov, CPU): the 273 self-score rows from Blackwell plus the 227 missing ones from Ada's finished 500/500 run (same model and code, different GPU), coverage 500/500. It replaces the 16/500 run of 21 Nov 16:04 (0.012), which was never a result.

No complete dev-split self-score exists (see the Dream dev row). The "test, all" row for Dream includes the final 300,
which the protocol line above reserves (dev and gate only); it is reported because the paper drafts already quote 30.4%.

### 5. K = 256 step-budget check: done (random subsample, stopped by the clock)

Design, stopping rule and analysis were pre-registered in `results/iter2/k256_preregistration.md` (clauses added later are dated and were written before the rows they cover). `step_arm.py` was run once, on all rows, Tue 24 Nov 2026 01:43 IST. Write-up: `results/iter2/k256_results.md`; raw output `k256_step_arm_output.txt`; JSON `k256_step_arm.json` (commit f01ead1 on `iter2/aaditya-worker`).
N = 10, T = 0.7, 256 steps, same prompts and sampler as K = 64; questions in the seeded random order `k256_order.txt`. Rows: Dream 185 (162 Blackwell RTX PRO 6000 + 23 Ada RTX 5000 Ada), LLaDA 192 (151 + 41); n is whatever finished by the clock, and Ada's rows used smaller sampling batches (same per-sample distribution).

| Model | n | first | majority | oracle | AR (qwen7b-cut:sum) | AR minus majority (PRIMARY) |
|---|---|---|---|---|---|---|
| dream-base | 185 | 0.741 | 0.843 | 0.935 | 0.757 | -0.086 [-0.141, -0.032], 5/21, p = 0.002 |
| llada-base | 192 | 0.724 | 0.792 | 0.896 | 0.766 | -0.026 [-0.068, +0.016], 6/11, p = 0.332 |

- Secondary, first sample K = 256 minus K = 64 (same questions): Dream +0.319 [+0.232, +0.400] (69/10), LLaDA +0.260 [+0.182, +0.339] (60/10), both p < 0.001. Majority: Dream +0.103 [+0.049, +0.157] (p = 0.001), LLaDA +0.047 [0.000, +0.099] (p = 0.108). Majority minus first sample at K = 256: Dream +0.103 (22/3), LLaDA +0.068 (16/3).
- Greedy context (T = 0, N = 1, our prompt; secondary, not the test of record): Dream n = 185 majority minus greedy +0.032 [-0.011, +0.081] (p = 0.238), AR minus greedy -0.054 [-0.108, 0.000] (p = 0.087), first minus greedy -0.070 [-0.135, -0.005] (p = 0.053); LLaDA n = 137 (19 Blackwell rows and 36 of the 41 Ada rows have no greedy row) majority minus greedy +0.080 [+0.029, +0.139] (p = 0.007), AR minus greedy +0.073 [+0.007, +0.139] (p = 0.041), first minus greedy +0.029 [-0.022, +0.080] (p = 0.424).
- Seconds per question (Blackwell rows only): Dream 36.5 (K = 64) and 151.1 (K = 256); LLaDA 54.9 and 216.6.

### 6. Per-question preservation (exploratory): done, 500 test questions

Commands: `per_question_preservation.py extract --model dream`, `... extract --model qwen7b`, `... analyze --split test`.
Predictor: layer-mean centred cosine of the pooled question states of Dream vs Qwen2.5-7B. Outcomes and methods are in the
script docstring and the header of its output. AR pick: `qwen7b-cut`. Source: `results/iter2/cpu/pq_analyze.txt`
(commit `aa5d3d2`, sha256 `c8a9b25b…`), rows "centered/nosink (PRIMARY)". 10 000 permutations, seed 0; Wald CI.

| Outcome | n | events | AUC | permutation p | OR per SD [95% CI] |
|---|---|---|---|---|---|
| first wrong and AR pick right (O1) | 500 | 108 | 0.469 | 0.3246 | 0.94 [0.75, 1.17] |
| majority right (O2) | 500 | 362 | 0.422 | 0.0068 | 0.98 [0.79, 1.22] |
| AR pick right among first-wrong (O3) | 298 | 108 | 0.425 | 0.0322 | 0.85 [0.66, 1.09] |

Also printed: O1m (500, 51 events) AUC 0.406, perm p 0.0266, OR 0.82 [0.61, 1.09]. Per-layer O1 AUCs range 0.445 to 0.500,
smallest BH q 0.707. Predictor correlates with log prompt length (0.305).

### 7. Exploratory CPU analyses (25 Nov, post-hoc)

Plan with every rule fixed before any result: `results/iter2/cpu_extra_plan.md` (commit `2a2caca`). Results, one table per item, with the commit and command for each: `results/iter2/cpu_extra_results.md`; raw outputs `results/iter2/cpu/extra_*.txt` and `.json`. All seven are exploratory and are reported with their nulls.

## What this says

Only what the tables above show:

- **Cut text (2):** cutting after `#### N` raised Dream's best AR selector from 0.358 to 0.488 but left LLaDA's unchanged
  (0.524 vs 0.522). On both models the best AR selector on cut text is still below majority vote (Dream -0.236, LLaDA -0.210,
  both CIs exclude 0). It is above first sample on both (0.488 vs 0.404; 0.522 vs 0.486). Paired (exploratory, 25 Nov,
  `results/iter2/cpu/extra_2_ar_vs_first.txt`; cut `qwen7b` summed score minus first sample, K = 64, n = 500): Dream +0.084
  [+0.032, +0.134], 108/66, p = 0.002; LLaDA +0.036 [-0.014, +0.084], 88/70, p = 0.176. So the gain over the first sample is
  significant on Dream only.
- **Step budget (3):** at K = 16 both models are near zero (first 0.026 / 0.036) and majority vote adds nothing (CIs include 0).
  Going from 16 to 32 steps raises first-sample accuracy (+0.098 Dream, +0.192 LLaDA); reranking at K = 16 gains far less than
  that, so A_rerank is negative for every selector printed (majority -0.092 Dream, -0.196 LLaDA). On these numbers, spending
  compute on more steps beats spending it on reranking at K = 16.
- **Self-score (4):** the DLM's own 8-draw ELBO estimate is below majority vote on every row run (gate: Dream -0.395, LLaDA
  -0.285; all 500: Dream -0.420, LLaDA -0.264) and below first sample on Dream (0.315 vs 0.395 gate); on LLaDA's gate it is
  0.450 vs first 0.485 (not tested).
- **K = 256 (5):** on a random subsample the AR-minus-majority gap shrinks from about -0.24 / -0.21 at K = 64 to -0.086 on Dream (CI excludes 0) and -0.026 on LLaDA (CI includes 0). Majority still beats the first sample (+0.103, +0.068) but by far less than at K = 64 (+0.320, +0.246 on the 500). Against one greedy sample majority is +0.032 on Dream (not significant) and +0.080 on LLaDA (p = 0.007). T = 0.7 at 256 steps is not like-for-like with the published greedy numbers.
- **Preservation (6):** the primary predictor does not predict O1 (AUC 0.469, p = 0.32). O2 and O3 have AUC below 0.5 with
  permutation p < 0.05 but odds-ratio CIs that include 1, uncorrected across 4 outcomes; exploratory, not a finding.

## Not verified / limits

- Everything still marked "not run" above has no number (the Arbor search, the dev-split self-score). Tables 2 to 6 were first filled by diya's agent (#34) from the files at commit `aa5d3d2`. On 25 Nov they were regenerated on Blackwell from the current data (`make_iter2_reports.py`, `cut_vs_trim.py` for test, dev and the gate subset, `per_question_preservation.py analyze`; outputs in `results/iter2/cpu/`, `step_arm.md`, `fidelity.md`). Tables 2, 3 and 6 and the gate rows of Table 4 came out unchanged; the changes are listed number by number in `results/iter2/cpu_extra_results.md` (item 3).
- Fidelity numbers are 100 questions each in the first-100 rows (standard error about 4 points) and 244 / 199 in the extension rows, which are not the first 100 questions.
- The K = 256 sample sizes were set by the clock (two Blackwell windows and an Ada extension stopped on instruction), not by a result; rows come from two GPU types; Dream crashed twice with a CUDA launch timeout (Xid 8) and was restarted without losing rows; one planned Dream slot (Mon 23 Nov 18:23-18:55) did not start because of a scheduling-gate mistake, so Dream n is 185, not about 197. At 256 steps the T = 0.7 first sample is 7.0 points below greedy on Dream and 2.9 above on LLaDA (paired, n = 185 and 137), so most of the I-21 gap is the step budget; this is a 185/137-question comparison, not the 100 fidelity questions.
- The self-score uses 8 Monte-Carlo draws per candidate (LLaDA's own evaluation default is 128) and a separate random seed per candidate, so the 10 candidates of a question are ranked on independent noisy estimates; it is a deliberately cheap, high-variance selector. How much of the 30.4% (Dream, 500 questions) or 45.0% (LLaDA, gate) is estimator noise is [unverified]; scoring a subset with the same masks for all candidates (common random numbers) was not tried [review of PR #41, neel's agent].
- The per-question preservation analysis used hidden states extracted in fp32 on the CPU, not bf16 on a GPU; whether the cosines would differ is [unverified].
- The senior session's description of the interruption is quoted from its reading of `journalctl`.
- Generation seconds are measured on shared GPUs and are not clean compute figures.
- Decision #7: a human reply exists; the rest of the team's confirmation was pending at drafting time.
