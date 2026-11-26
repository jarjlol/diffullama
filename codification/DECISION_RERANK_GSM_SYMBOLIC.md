# Decision reranking on GSM-Symbolic — interim results

Run date: 2026-11-26. Branch `iter2/decision-rerank`. Pre-registration:
`DECISION_RERANK_GSM_SYMBOLIC_PREREG.md` (committed before the first full
scored run).

**Status: interim, not final.** Generation was still running at the hard stop
before the 08:35 machine reboot. Dream has **149/250** questions; LLaDA has
**165/250**. Both are question-order prefixes: all of P1 plus the first part of
P2-0. P2-1 and P2-2 are not represented, so the subsets are template-skewed and
must not be read as random samples of the 250. No 250-question claim is made.

## Post-reboot update (10:27)

Generation stopped at 08:47; the candidate files contain **Dream 175/250** and
**LLaDA 217/250**. An incremental listwise pass scored every one of them
(listwise coverage 1.00 for both). Pointwise was not rescored; its cells remain
the earlier 149/165 snapshot and their coverage is shown.

| selector | Dream (n=175) | LLaDA (n=217) | macro |
|---|---:|---:|---:|
| first | 0.114 | 0.120 | 0.117 |
| majority | 0.320 | 0.240 | 0.280 |
| **decision.listwise** | **0.440** | **0.364** | **0.402** |
| decision.pointwise (cov 0.85 / 0.76) | 0.510 | 0.485 | 0.497 |
| oracle | 0.623 | 0.461 | 0.542 |

H1 on the common 175 questions: listwise − majority = **+0.137**, 95% CI
[+0.097, +0.180]; Dream exact McNemar p=5.1e-4, LLaDA p=4.6e-7. The macro
accuracy dropped from the 160/187 snapshot (0.446 → 0.402) because more P2-0 and
the first P2-1 questions entered; the margin remains above the GSM8K K=64 +5.2.


## Final pre-reboot update (08:22)

An incremental listwise pass scored every question completed before the reboot:
**Dream 160/250, LLaDA 187/250, listwise coverage 1.00 for both.** Pointwise was
not rescored; its cells below remain the 149/165 snapshot and their coverage is
shown.

| selector | Dream (n=160) | LLaDA (n=187) | macro |
|---|---:|---:|---:|
| first | 0.125 | 0.139 | 0.132 |
| majority | 0.350 | 0.278 | 0.314 |
| **decision.listwise** | **0.469** | **0.422** | **0.446** |
| decision.pointwise (cov 0.93 / 0.88) | 0.510 | 0.485 | 0.497 |
| oracle | 0.656 | 0.529 | 0.593 |

H1 on the common 160 questions: listwise − majority = **+0.144**, 95% CI
[+0.100, +0.188]; Dream exact McNemar p=0.0013, LLaDA p=4.6e-7. The interim
section below is kept for history; where they differ, this table is the later
snapshot.


## Main table

| selector | Dream (n=149) | LLaDA (n=165) | macro |
|---|---:|---:|---:|
| first | 0.134 | 0.158 | 0.146 |
| majority | 0.369 | 0.315 | 0.342 |
| **decision.listwise** | **0.500** | **0.479** | **0.489** |
| decision.pointwise | 0.532 | 0.523 | 0.528 |
| oracle | 0.685 | 0.600 | 0.642 |

Listwise coverage 0.99 / 1.00; pointwise 0.95 / 0.92 (a few candidates arrived
after the prompt build). AR qwen7b-cut is not yet computed; no AR row is claimed.

**H1, common 149 questions:** listwise − majority = **+0.149**, 95% CI
[+0.105, +0.196], paired cluster bootstrap over questions. Per-model exact
McNemar: Dream +0.128, p=0.0013 (26 losses / 7 wins); LLaDA +0.168, p=4.6e-7
(27/2). The GSM8K K=64 comparison is **+5.2 points** [+3.0, +7.4].

The margin does not shrink on this interim set, but the mechanism matters. Mean
top-vote count is **2.2** (Dream) / **2.5** (LLaDA) versus 4.6 / 5.4 on GSM8K;
only **5.6% / 9.7%** of questions have a ≥5-vote majority versus 46.8% / 57.6%.
Majority vote collapses on the fresh templates; absolute listwise accuracy is
lower than on GSM8K (0.49 vs 0.78), so the larger margin is not evidence that
selection improved.

## Per-template breakdown (listwise)

| template | Dream n | Dream listwise | LLaDA n | LLaDA listwise | Dream oracle | LLaDA oracle |
|---|---:|---:|---:|---:|---:|---:|
| P1-0 | 50 | 0.580 | 50 | 0.560 | 0.740 | 0.740 |
| P1-1 | 50 | 0.640 | 50 | 0.880 | 0.820 | 0.980 |
| P1-2 | 25 | 0.440 | 25 | 0.200 | 0.640 | 0.280 |
| P2-0 | 24 | 0.083 | 41 | 0.049 | 0.333 | 0.146 |

P2-0 (average of two percentages, decimal gold) is a genuine failure mode: the
judge picks a correct candidate on only 2/24 and 2/41 items and is often
confidently wrong. A strict-cut sensitivity that ends the candidate text
immediately after `#### N` moves it to 1/5 per model, so trailing junk after the
answer line is not the explanation. The final 250 adds the rest of P2-0 and all
of P2-1/P2-2; the aggregate may move down.

## Controls (completed set)

| control | Dream | LLaDA | chance |
|---|---:|---:|---:|
| closed-book two-option (250 questions) | 0.900 (225/250) | — | 0.50 |
| judge-alone, distinct answers | 0.282 (n=149) | 0.289 (n=166) | 0.121 / 0.130 |
| gold injection, no candidate has gold | 0.191 (n=47) | 0.313 (n=67) | 0.100 |

Closed-book two-option is **225/250 = 0.900** [0.857, 0.931], with no position
bias (A 113/124, B 112/126). The decision model therefore retains real answer
signal on fresh questions. The claim is **selection, not reasoning
verification**; GSM-Symbolic reduces memorisation of GSM8K items but does not
prove that no contamination or template leakage remains.

## Calibration

Listwise max-probability is overconfident: ECE₁₀ 0.318 / 0.296, Brier 0.361 /
0.344. Pointwise is much better calibrated: ECE₁₀ 0.060 / 0.070, Brier 0.092 /
0.100. Cost is recorded from `usage`; the `:free` tier recorded $0, and the paid
estimate at $0.01/1M input tokens is about $0.003 per listwise run per model.

## Threshold rule

T=5 was chosen on 200 GSM8K dev questions (macro 0.790). On GSM-Symbolic it calls
listwise on 95% / 91% of items and scores 0.500 / 0.479, i.e. no gain over
always-listwise; T=4 is worse (0.493 / 0.467). The weak-vote rule does not
transfer when majority vote is this weak.

## Files

`codification/results/gsm_symbolic_eval_interim.json`,
`gsm_symbolic_eval_interim_common.json`, `gsm_symbolic_controls.json`,
`gsm_symbolic_calibration.json`, `gsm_symbolic_threshold.json`,
`gsm_symbolic_sweeps.json`, `gsm_symbolic_gsm8k_sweeps.json`.
