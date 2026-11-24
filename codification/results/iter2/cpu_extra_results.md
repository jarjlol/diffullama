# Seven exploratory CPU analyses: results (Wed 25 Nov 2026, run 00:32-00:35 IST on Blackwell, CPU only)

Plan and every rule: `cpu_extra_plan.md` (commit `2a2caca`, 00:29 IST, before any real run; the plan's time stamp was corrected in `4617485`).
All seven are EXPLORATORY (post-hoc) and none is corrected for multiple comparisons. Each test was run once, on all rows; items 4 and 7 were run on dev first.
Script: `codification/pipeline/cpu_extra.py` (as committed in `a48973d`, run from a clean copy at `~/codify/extra_run` with
`CODIFY_WORK_DIR=~/codify/diffullama/codification/pipeline/work`, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=8`, `nice -n 10`). Plot: `cpu_extra_plot.py` (run locally, matplotlib).
Paired differences are "first named minus second named", 95% bootstrap CI (10 000 draws, seed 0), exact McNemar p, as in `step_arm.py`.
Raw outputs: `cpu/extra_<n>_*.txt` and `.json`. A smoke test on a scratch copy built from the dev split relabelled as test preceded the real runs (no real test file was read).

## Item 1. Equal compute (steps per question)

Command: `python3 cpu_extra.py --item 1` (6 s). Raw: `cpu/extra_1_equal_compute.txt/.json`; plot `cpu/extra_1_equal_compute.png`.
Both sides of each comparison use 256 denoising steps per question. Equal steps is not equal wall-clock time (see the note below).

| Model | n | K=256 first | K=64 majority of cands 0-3 | PRIMARY: K=256 first minus K=64 maj-of-4 | K=32 majority of cands 0-7 | K=256 first minus K=32 maj-of-8 |
|---|---|---|---|---|---|---|
| Dream | 185 | 0.741 | 0.578 | **+0.162 [+0.081, +0.238], 44/14, p < 0.001** | 0.151 | +0.589 [+0.514, +0.665], 113/4, p < 0.001 |
| LLaDA | 192 | 0.724 | 0.620 | **+0.104 [+0.042, +0.167], 29/9, p = 0.002** | 0.453 | +0.271 [+0.198, +0.344], 60/8, p < 0.001 |

Curve (majority at n candidates, x = n x K steps per question; full n = 1 to 10 in the txt/json, both the first-n and the random-subset version):

| Model | Config (n questions) | steps/q at n = 10 | acc, first-n, n = 4 | acc, 200 random subsets, n = 4 | acc, n = 10 (all candidates) | s/q for N = 10 (Blackwell rows) |
|---|---|---|---|---|---|---|
| Dream | K16 (500) | 160 | 0.028 | 0.030 | 0.032 | 9.5 |
| Dream | K32 (500) | 320 | 0.160 | 0.151 | 0.234 | 19.3 |
| Dream | K64 (500) | 640 | 0.582 | 0.559 | 0.724 | 36.8 |
| Dream | K64 (185 subset) | 640 | 0.578 | 0.568 | 0.741 | 36.5 |
| Dream | K256 (185 subset) | 2560 | 0.811 | 0.815 | 0.843 | 151.1 (162 rows) |
| LLaDA | K16 (500) | 160 | 0.036 | 0.038 | 0.032 | 13.7 |
| LLaDA | K32 (500) | 320 | 0.348 | 0.354 | 0.510 | 27.4 |
| LLaDA | K64 (500) | 640 | 0.644 | 0.622 | 0.732 | 55.0 |
| LLaDA | K64 (192 subset) | 640 | 0.620 | 0.609 | 0.745 | 54.9 |
| LLaDA | K256 (192 subset) | 2560 | 0.766 | 0.757 | 0.792 | 216.6 (151 rows) |

Reading: at 256 steps per question a single 256-step sample is ahead of the best way to spend the same steps on voting (four 64-step samples), by 16.2 points on
Dream and 10.4 on LLaDA. At 640 steps per question (ten 64-step samples) majority at K = 64 reaches 0.741 / 0.745 on the subset, against 0.741 / 0.724 for ONE
256-step sample at 256 steps. Within K = 256, voting adds a further 10.3 / 6.8 points (n = 10).
Wall-clock: the proxy "0.1 x n x s/q" gives 15.1 s (Dream) and 21.7 s (LLaDA) for one K = 256 sample against 14.6 s and 22.0 s for four K = 64 samples. That scaling is
an assumption (batched generation, not verified to be linear in n), and the K = 256 s/q come from Blackwell rows on a shared card; do not quote the proxy as a measurement.
Caveats: all at T = 0.7 (a K = 256 greedy sample would be a different comparison); the K = 256 questions are a random subsample, so K = 64 is also given on the same subset.

## Item 2. AR (cut `qwen7b` summed score) versus first sample, paired

Command: `python3 cpu_extra.py --item 2`. Raw: `cpu/extra_2_ar_vs_first.txt/.json`. This closes the "[unverified as a paired result]" line in `ITERATION_2.md`.

| Model | K | n | AR (qwen7b-cut sum) | first | AR minus first | wins/losses | McNemar p |
|---|---|---|---|---|---|---|---|
| Dream | 16 | 500 | 0.018 | 0.026 | -0.008 [-0.024, +0.008] | 6/10 | 0.454 |
| Dream | 32 | 500 | 0.138 | 0.124 | +0.014 [-0.022, +0.050] | 47/40 | 0.520 |
| Dream | 64 | 500 | 0.488 | 0.404 | **+0.084 [+0.032, +0.134]** | 108/66 | 0.002 |
| Dream | 256 | 185 | 0.757 | 0.741 | +0.016 [-0.043, +0.076] | 18/15 | 0.728 |
| LLaDA | 16 | 500 | 0.034 | 0.036 | -0.002 [-0.024, +0.020] | 15/16 | 1.000 |
| LLaDA | 32 | 500 | 0.286 | 0.228 | +0.058 [+0.012, +0.106] | 86/57 | 0.019 |
| LLaDA | 64 | 500 | 0.522 | 0.486 | +0.036 [-0.014, +0.084] | 88/70 | 0.176 |
| LLaDA | 256 | 192 | 0.766 | 0.724 | +0.042 [-0.016, +0.099] | 19/11 | 0.200 |

At K = 64 (the paper's setting) the AR selector is significantly better than the first sample on Dream only. At K = 256 neither difference is distinguishable from zero.

## Item 3. Refresh the reports

Commands (Blackwell, `~/codify/extra_run/codification/pipeline`): `python3 make_iter2_reports.py --out ../../../reports`; `python3 cut_vs_trim.py --model <dream|llada>-base --split <test|dev>`;
`python3 cut_vs_trim.py --model <m>-base --split test --subset gate`; `~/venvs/scorer/bin/python per_question_preservation.py analyze --split test`; `python3 fidelity_majority.py`.
Regenerated files: `results/iter2/step_arm.md`, `results/iter2/fidelity.md` (replaced), and `results/iter2/cpu/refresh_25nov/*` (copies of the re-run outputs; the original job outputs in `cpu/` are untouched).
Two small code changes to the generators (no number changes): `step_arm.py` also stores `greedy_n` in the JSON, and `make_iter2_reports.py` uses it for the greedy rows (they were labelled with n = all
questions although computed on the overlap) and its stale "K = 256 covers only the first 100 test questions" bullet is corrected.
`ITERATION_2.md` updated: status paragraph, the tier status table (6 rows), Table 1 (two extension rows), the Dream dev row of Table 4, the "What this says" bullets (cut-text AR vs first now tested; all-500 self-score numbers), the "Not verified / limits" bullets (the `aa5d3d2` bullet and the fidelity bullet), and a pointer to section 7.

Numbers that changed, old to new:

| File | Quantity | Old | New |
|---|---|---|---|
| `step_arm.md` | K = 256 block (Dream, LLaDA) | "no candidates yet" | Dream n = 185: first 0.741, majority 0.843, oracle 0.935, AR (qwen7b-cut sum) 0.757; LLaDA n = 192: 0.724 / 0.792 / 0.896 / 0.766; K = 64 on the same questions 0.422 / 0.741 (Dream), 0.464 / 0.745 (LLaDA); paired lines as in `k256_results.md` |
| `step_arm.md` | short table | no K = 256 rows | + 15 rows (K = 256 and K = 64 on the subsets, greedy: dream-sampler 0.811 n = 185, dream-published 0.780 n = 41, llada-published 0.693 n = 137) |
| `step_arm.md` | K = 16/32/64 raw block and table | | unchanged (checked by diff) |
| `step_arm.md` | "What is [unverified]" bullets | "K = 256 covers only the first 100 test questions" | "random subsample ... n = 185 / 192 ..." |
| `fidelity.md` | `dream-sampler` | n = 100, greedy 0.810, first 0.410, median 16.9 s/q | n = 244, greedy 0.824, first 0.422, median 16.5 s/q (majority on the same questions 0.734; the first 100 are unchanged at 0.810 / 0.410 / 0.700) |
| `fidelity.md` | `llada-published` | n = 100, greedy 0.730, first 0.510, median 17.5 s/q | n = 199, greedy 0.729, first 0.497, median 17.9 s/q (majority 0.764; the first 100 are unchanged at 0.730 / 0.510 / 0.750) |
| `fidelity.md` | `dream-published` | n = 100, greedy 0.800, first 0.410 | unchanged |
| `cvt_llada_test` | `dlm.selfscore` coverage | 16/500 "INCOMPLETE" | 500/500 |
| `cvt_llada_test` | `dlm.selfscore` accuracy / minus majority | 0.012 / -0.720 [-0.758, -0.680], 0/360 | 0.468 / -0.264 [-0.306, -0.222], 10/142 (already in `ITERATION_2.md` since 24 Nov) |
| `cvt_dream_dev` | `dlm.selfscore` | no line | coverage 125/200 INCOMPLETE; 0.165; minus majority -0.535 [-0.605, -0.460], 2/109. Not a result: a partial file from 21 Nov, 75 questions counted as wrong. |
| `cvt_dream_test`, `cvt_llada_dev`, `self_dream_gate`, `self_llada_gate`, `pq_analyze` | all numbers | | unchanged (diff shows only blank lines) |

Tables 2, 3 and 6 and the gate rows of Table 4 in `ITERATION_2.md` are therefore unchanged. One thing for the paper owners, not changed here: the paper's fidelity sentence uses the first-100
numbers (81.0 and 73.0, n = 100); the extension rows (82.4 on 244, 72.9 on 199) are in `ITERATION_2.md` Table 1 only.

## Item 4. Majority with the cut-text AR tie-break

Rule as in the plan (iteration-1 node 2.1 on `qwen7b-cut` per-token mean NLL). Commands: `python3 cpu_extra.py --item 4 --split dev` first, then `--split test` once.
Raw: `cpu/extra_4_tiebreak_dev.txt`, `cpu/extra_4_tiebreak_test.txt`.

| Model | Split | n | majority | majority + AR tie-break | questions with a top-vote tie | choice differs from majority | difference | wins/losses | p |
|---|---|---|---|---|---|---|---|---|---|
| Dream | dev | 200 | 0.700 | 0.665 | 39 | 30 | -0.035 [-0.065, -0.010] | 1/8 | 0.039 |
| LLaDA | dev | 200 | 0.745 | 0.745 | 23 | 15 | +0.000 [-0.025, +0.025] | 3/3 | 1.000 |
| Dream | **test** | 500 | 0.724 | 0.718 | 67 | 47 | -0.006 [-0.022, +0.008] | 6/9 | 0.607 |
| LLaDA | **test** | 500 | 0.732 | 0.730 | 57 | 31 | -0.002 [-0.018, +0.014] | 7/8 | 1.000 |

No question had a missing score. The tie-break does not help on test (null for both models); on Dream dev it was nominally worse, on test it is indistinguishable from plain majority.
About 11-13% of questions have a top-vote tie, so there is little room for any tie-break rule.

## Item 5. DLM self-score at K = 256

Command: `python3 cpu_extra.py --item 5`. Raw: `cpu/extra_5_selfscore_k256.txt/.json`. Highest score wins; no question lacked a score.

| Model | n | majority | first | selector | accuracy | minus majority | w/l | p | minus first | w/l | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dream | 185 | 0.843 | 0.741 | `dlm.selfscore` (cut) | 0.714 | -0.130 [-0.184, -0.081] | 1/25 | < 0.001 | -0.027 [-0.092, +0.038] | 16/21 | 0.511 |
| Dream | 185 | | | `dlm.selftrim` | 0.708 | -0.135 [-0.189, -0.081] | 3/28 | < 0.001 | -0.032 [-0.097, +0.032] | 16/22 | 0.418 |
| LLaDA | 192 | 0.792 | 0.724 | `dlm.selfscore` (cut) | 0.724 | -0.068 [-0.115, -0.021] | 5/18 | 0.011 | +0.000 [-0.057, +0.057] | 15/15 | 1.000 |
| LLaDA | 192 | | | `dlm.selftrim` | 0.729 | -0.062 [-0.109, -0.016] | 5/17 | 0.017 | +0.005 [-0.052, +0.062] | 16/15 | 1.000 |

The DLM's own score loses to majority vote at K = 256 on both models and is not better than the first sample.

## Item 6. Vote share as confidence (descriptive)

Command: `python3 cpu_extra.py --item 6`. Raw: `cpu/extra_6_vote_share.txt/.json` (bins 0 to 10 and the coverage curve for t = 1 to 10).

| Model | K | n | majority acc | AUROC (top-vote count -> majority correct) | correct / incorrect | accuracy at top-vote >= 5 (coverage) | at top-vote >= 8 (coverage) | at top-vote 10 (n) |
|---|---|---|---|---|---|---|---|---|
| Dream | 64 | 500 | 0.724 | 0.857 | 362 / 138 | 0.949 (0.468) | 1.000 (0.136) | 1.000 (8) |
| Dream | 256 | 185 | 0.843 | 0.842 | 156 / 29 | 0.903 (0.892) | 0.939 (0.714) | 0.982 (56) |
| LLaDA | 64 | 500 | 0.732 | 0.884 | 366 / 134 | 0.938 (0.576) | 0.978 (0.272) | 0.971 (35) |
| LLaDA | 256 | 192 | 0.792 | 0.792 | 152 / 40 | 0.834 (0.911) | 0.901 (0.682) | 0.941 (101) |

Majority accuracy rises steadily with vote share (Dream K = 64: 0.04 at top-vote 1, 0.32 at 2, 0.67 at 3, 0.92 at 5). At K = 256 the votes are concentrated (Dream: 56 of 185 questions are unanimous), so
the same threshold keeps most questions and the curve is flatter. AUROC is lower at K = 256 than at K = 64 for LLaDA (0.792 vs 0.884) and about equal for Dream. Descriptive only, no test.

## Item 7. Drop degenerate candidates before voting

Rule: drop a candidate if its extracted answer is None or `"####"` is not in `task.trim(text)`; plain majority if all ten are dropped. `codification/n4/` has no degeneracy definition (checked: it
measures answer diversity and failure classes), so the plan's rule was used. Commands: `python3 cpu_extra.py --item 7 --split dev`, then `--split test` once. Raw: `cpu/extra_7_degenerate_filter_{dev,test}.txt`.

| Model | Split | n | majority | filtered majority | candidates dropped | questions with all ten dropped | choice differs | difference | w/l | p |
|---|---|---|---|---|---|---|---|---|---|---|
| Dream | dev | 200 | 0.700 | 0.720 | 348 / 2000 | 0 | 16 | +0.020 [+0.005, +0.040] | 4/0 | 0.125 |
| LLaDA | dev | 200 | 0.745 | 0.740 | 110 / 2000 | 0 | 4 | -0.005 [-0.015, +0.000] | 0/1 | 1.000 |
| Dream | **test** | 500 | 0.724 | 0.734 | 974 / 5000 | 1 | 37 | +0.010 [-0.004, +0.024] | 9/4 | 0.267 |
| LLaDA | **test** | 500 | 0.732 | 0.742 | 320 / 5000 | 0 | 13 | +0.010 [+0.002, +0.020] | 5/0 | 0.062 |

The filter gives +1.0 point on both models on test, in the direction the dev run suggested for Dream but not for LLaDA; neither is significant by McNemar (the LLaDA bootstrap CI excludes 0 only
because it has 5 wins and 0 losses; the exact p is 0.062). Treat as a small, unconfirmed gain.

## Draft sentence for the paper (item 1, for Main to decide; not applied)

"At equal compute (256 denoising steps per question), one 256-step sample beats majority vote over four 64-step samples by 16.2 points on Dream (95% CI [8.1, 23.8], n = 185) and by 10.4 on LLaDA ([4.2, 16.7], n = 192); exploratory, T = 0.7."

## What did not run

Nothing was skipped. Not done on purpose: no K = 256 filter or tie-break run (the plan fixed K = 64 for items 4 and 7); no per-n wall-clock measurement (proxy only); the dev self-score for Dream and LLaDA (not part of this plan).
