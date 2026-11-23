# Step-budget arm (prompt A1) and K = 256 check, generated 2026-11-21

Commands (run in `codification/pipeline/`, inputs in the git-ignored `work/`):

```bash
python3 step_arm.py --json work/step_arm.json
python3 make_iter2_reports.py        # wrote this file
```

## Raw output

```text

== dream-base  (test, n = 500 questions common to K = [16, 32, 64])
     K   first   major  oracle  best AR (name)            s/q (N=10)
    16   0.026   0.032   0.174  0.024 (ar.gpt2.mean)       9.5
    32   0.124   0.234   0.540  0.138 (ar.qwen7b-cut.sum)  19.3
    64   0.404   0.724   0.878  0.488 (ar.qwen7b-cut.sum)  36.8
  dQ_steps (first sample): 16->32 +0.098, 32->64 +0.280
  dQ_rerank, majority:     K=16 +0.006, K=32 +0.110, K=64 +0.320
  A_rerank = dQ_rerank(16) - dQ_steps(16->32): majority -0.092, ar.gpt2-cut.mean -0.100, ar.gpt2.mean -0.100, ar.gpt2-cut.sum -0.102
  paired differences on the common questions (95% bootstrap CI, exact McNemar):
  first: K=32 minus K=16                       +0.098 [+0.066,+0.130]  wins/losses 60/11  McNemar p=0.000
  majority: K=32 minus K=16                    +0.202 [+0.166,+0.240]  wins/losses 106/5  McNemar p=0.000
  first: K=64 minus K=32                       +0.280 [+0.232,+0.328]  wins/losses 165/25  McNemar p=0.000
  majority: K=64 minus K=32                    +0.490 [+0.442,+0.536]  wins/losses 254/9  McNemar p=0.000
  K=16: majority minus first                   +0.006 [-0.010,+0.024]  wins/losses 11/8  McNemar p=0.648
  K=32: majority minus first                   +0.110 [+0.078,+0.142]  wins/losses 65/10  McNemar p=0.000
  K=64: majority minus first                   +0.320 [+0.278,+0.362]  wins/losses 164/4  McNemar p=0.000

== llada-base  (test, n = 500 questions common to K = [16, 32, 64])
     K   first   major  oracle  best AR (name)            s/q (N=10)
    16   0.036   0.032   0.246  0.036 (ar.qwen0.5b-cut.sum) 13.7
    32   0.228   0.510   0.760  0.286 (ar.qwen7b-cut.sum)  27.4
    64   0.486   0.732   0.892  0.524 (ar.qwen7b.sum)      55.0
  dQ_steps (first sample): 16->32 +0.192, 32->64 +0.258
  dQ_rerank, majority:     K=16 -0.004, K=32 +0.282, K=64 +0.246
  A_rerank = dQ_rerank(16) - dQ_steps(16->32): ar.qwen7b-cut.sum -0.194, majority -0.196, ar.gpt2-cut.sum -0.198, ar.qwen0.5b.sum -0.200
  paired differences on the common questions (95% bootstrap CI, exact McNemar):
  first: K=32 minus K=16                       +0.192 [+0.154,+0.230]  wins/losses 107/11  McNemar p=0.000
  majority: K=32 minus K=16                    +0.478 [+0.434,+0.522]  wins/losses 241/2  McNemar p=0.000
  first: K=64 minus K=32                       +0.258 [+0.208,+0.308]  wins/losses 162/33  McNemar p=0.000
  majority: K=64 minus K=32                    +0.222 [+0.178,+0.266]  wins/losses 130/19  McNemar p=0.000
  K=16: majority minus first                   -0.004 [-0.024,+0.016]  wins/losses 12/14  McNemar p=0.845
  K=32: majority minus first                   +0.282 [+0.238,+0.326]  wins/losses 153/12  McNemar p=0.000
  K=64: majority minus first                   +0.246 [+0.206,+0.286]  wins/losses 130/7  McNemar p=0.000

== dream-base K = 256: no candidates yet

== llada-base K = 256: no candidates yet
```

## Short table

| model | budget | selector | value | n questions |
|---|---|---|---|---|
| dream-base | K16 | first | 0.026 | 500 |
| dream-base | K16 | majority | 0.032 | 500 |
| dream-base | K16 | oracle | 0.174 | 500 |
| dream-base | K16 | gen s/question (10 samples) | 9.5 | 500 |
| dream-base | K32 | first | 0.124 | 500 |
| dream-base | K32 | majority | 0.234 | 500 |
| dream-base | K32 | oracle | 0.540 | 500 |
| dream-base | K32 | gen s/question (10 samples) | 19.3 | 500 |
| dream-base | K64 | first | 0.404 | 500 |
| dream-base | K64 | majority | 0.724 | 500 |
| dream-base | K64 | oracle | 0.878 | 500 |
| dream-base | K64 | gen s/question (10 samples) | 36.8 | 500 |
| llada-base | K16 | first | 0.036 | 500 |
| llada-base | K16 | majority | 0.032 | 500 |
| llada-base | K16 | oracle | 0.246 | 500 |
| llada-base | K16 | gen s/question (10 samples) | 13.7 | 500 |
| llada-base | K32 | first | 0.228 | 500 |
| llada-base | K32 | majority | 0.510 | 500 |
| llada-base | K32 | oracle | 0.760 | 500 |
| llada-base | K32 | gen s/question (10 samples) | 27.4 | 500 |
| llada-base | K64 | first | 0.486 | 500 |
| llada-base | K64 | majority | 0.732 | 500 |
| llada-base | K64 | oracle | 0.892 | 500 |
| llada-base | K64 | gen s/question (10 samples) | 55.0 | 500 |

## What is [unverified] or not covered

- Nothing here is interpreted; every number is copied from the commands above.
- All accuracies are on GSM8K test questions, N = 10 candidates, temperature 0.7 (K = 16/32/64/256); the greedy rows
  are one sample per question at the published settings of `fidelity_check.py`.
- K = 256 covers only the first 100 test questions (so its CIs are wide); K = 16/32/64 are compared on their common questions.
- Generation seconds were measured on shared GPUs (the `antpc` account is shared, lanes were interrupted on 2026-11-20), so
  they are not clean compute measurements [unverified as a cost comparison].
- Paired differences use paired_stats.py's bootstrap (10 000 resamples, seed 0) and exact McNemar; no multiple-comparison
  correction across the rows.
- The AR rows score the iteration-1 trimmed text unless a `-cut` scorer appears in the output; candidates carry junk after the
  '#### N' line (see ITERATION_2 notes), which affects those rows.
