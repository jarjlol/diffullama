# Step-budget arm (prompt A1) and K = 256 check, generated 2026-11-25

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

== dream-base  K = 256 vs K = 64 on the 185 questions with a K = 256 run (test; N = 10, T 0.7, random order)
  K=256 rows generated on: ['antpc|NVIDIA RTX 5000 Ada Generation', 'antpc|NVIDIA RTX PRO 6000 Blackwell Workstation Edition']
  s/question (N = 10), BLACKWELL rows only (Ada gen_seconds is another GPU with 5-sample batches, not comparable): K=64 36.5 (n=185), K=256 151.1 (n=162)
           ar.gpt2-cut.me ar.gpt2-cut.su   ar.gpt2.mean    ar.gpt2.sum ar.qwen7b-cut. ar.qwen7b-cut. ar.qwen7b.mean  ar.qwen7b.sum          first       majority         oracle
  K=64             0.286          0.319          0.405          0.281          0.265          0.541          0.373          0.357          0.422          0.741          0.886
  K=256            0.714          0.654          0.708          0.654          0.735          0.757          0.724          0.692          0.741          0.843          0.935
  representativeness, K = 64 on all 500 questions: first 0.404, majority 0.724   (same selectors on the 185-question subset: first 0.422, majority 0.741)
  PRIMARY K=256: ar.qwen7b-cut.sum minus majority -0.086 [-0.141,-0.032]  wins/losses 5/21  McNemar p=0.002
  SECONDARY first: K=256 minus K=64            +0.319 [+0.232,+0.400]  wins/losses 69/10  McNemar p=0.000
  majority: K=256 minus K=64                   +0.103 [+0.049,+0.157]  wins/losses 24/5  McNemar p=0.001
  ar.gpt2-cut.sum: K=256 minus K=64            +0.335 [+0.249,+0.422]  wins/losses 74/12  McNemar p=0.000
  ar.gpt2.sum: K=256 minus K=64                +0.373 [+0.292,+0.454]  wins/losses 77/8  McNemar p=0.000
  ar.qwen7b-cut.sum: K=256 minus K=64          +0.216 [+0.141,+0.292]  wins/losses 51/11  McNemar p=0.000
  ar.qwen7b.sum: K=256 minus K=64              +0.335 [+0.254,+0.416]  wins/losses 71/9  McNemar p=0.000
  K=256: majority minus first sample           +0.103 [+0.054,+0.157]  wins/losses 22/3  McNemar p=0.000
  K=256 majority minus greedy (dream-sampler), n=185 +0.032 [-0.011,+0.081]  wins/losses 12/6  McNemar p=0.238
  K=256 ar.qwen7b-cut.sum minus greedy (dream-sampler), n=185 -0.054 [-0.108,+0.000]  wins/losses 9/19  McNemar p=0.087
  K=256 first minus greedy (dream-sampler), n=185 -0.070 [-0.135,-0.005]  wins/losses 13/26  McNemar p=0.053
  K=256 majority minus greedy (dream-published), n=41 -0.024 [-0.122,+0.073]  wins/losses 2/3  McNemar p=1.000
  K=256 ar.qwen7b-cut.sum minus greedy (dream-published), n=41 -0.146 [-0.268,-0.049]  wins/losses 0/6  McNemar p=0.031
  K=256 first minus greedy (dream-published), n=41 -0.122 [-0.244,+0.000]  wins/losses 1/6  McNemar p=0.125

== llada-base  K = 256 vs K = 64 on the 192 questions with a K = 256 run (test; N = 10, T 0.7, random order)
  K=256 rows generated on: ['antpc|NVIDIA RTX 5000 Ada Generation', 'antpc|NVIDIA RTX PRO 6000 Blackwell Workstation Edition']
  s/question (N = 10), BLACKWELL rows only (Ada gen_seconds is another GPU with 5-sample batches, not comparable): K=64 54.9 (n=192), K=256 216.6 (n=151)
           ar.gpt2-cut.me ar.gpt2-cut.su   ar.gpt2.mean    ar.gpt2.sum ar.qwen7b-cut. ar.qwen7b-cut. ar.qwen7b.mean  ar.qwen7b.sum          first       majority         oracle
  K=64             0.411          0.380          0.396          0.406          0.490          0.490          0.432          0.516          0.464          0.745          0.880
  K=256            0.698          0.698          0.708          0.677          0.750          0.766          0.755          0.740          0.724          0.792          0.896
  representativeness, K = 64 on all 500 questions: first 0.486, majority 0.732   (same selectors on the 192-question subset: first 0.464, majority 0.745)
  PRIMARY K=256: ar.qwen7b-cut.sum minus majority -0.026 [-0.068,+0.016]  wins/losses 6/11  McNemar p=0.332
  SECONDARY first: K=256 minus K=64            +0.260 [+0.182,+0.339]  wins/losses 60/10  McNemar p=0.000
  majority: K=256 minus K=64                   +0.047 [+0.000,+0.099]  wins/losses 17/8  McNemar p=0.108
  ar.gpt2-cut.sum: K=256 minus K=64            +0.318 [+0.245,+0.391]  wins/losses 67/6  McNemar p=0.000
  ar.gpt2.sum: K=256 minus K=64                +0.271 [+0.193,+0.349]  wins/losses 62/10  McNemar p=0.000
  ar.qwen7b-cut.sum: K=256 minus K=64          +0.276 [+0.208,+0.344]  wins/losses 56/3  McNemar p=0.000
  ar.qwen7b.sum: K=256 minus K=64              +0.224 [+0.151,+0.297]  wins/losses 51/8  McNemar p=0.000
  K=256: majority minus first sample           +0.068 [+0.026,+0.115]  wins/losses 16/3  McNemar p=0.004
  K=256 majority minus greedy (llada-published), n=137 +0.080 [+0.029,+0.139]  wins/losses 13/2  McNemar p=0.007
  K=256 ar.qwen7b-cut.sum minus greedy (llada-published), n=137 +0.073 [+0.007,+0.139]  wins/losses 15/5  McNemar p=0.041
  K=256 first minus greedy (llada-published), n=137 +0.029 [-0.022,+0.080]  wins/losses 9/5  McNemar p=0.424
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
| dream-base@k256 | K64 | first | 0.422 | 185 |
| dream-base@k256 | K64 | majority | 0.741 | 185 |
| dream-base@k256 | K256 | first | 0.741 | 185 |
| dream-base@k256 | K256 | majority | 0.843 | 185 |
| dream-base@k256 | K256 | oracle | 0.935 | 185 |
| dream-base@k256 | K64 | oracle | 0.886 | 185 |
| dream-base@k256 | greedy | dream-sampler | 0.811 | 185 |
| dream-base@k256 | greedy | dream-published | 0.780 | 41 |
| llada-base@k256 | K64 | first | 0.464 | 192 |
| llada-base@k256 | K64 | majority | 0.745 | 192 |
| llada-base@k256 | K256 | first | 0.724 | 192 |
| llada-base@k256 | K256 | majority | 0.792 | 192 |
| llada-base@k256 | K256 | oracle | 0.896 | 192 |
| llada-base@k256 | K64 | oracle | 0.880 | 192 |
| llada-base@k256 | greedy | llada-published | 0.693 | 137 |

## What is [unverified] or not covered

- Nothing here is interpreted; every number is copied from the commands above.
- All accuracies are on GSM8K test questions, N = 10 candidates, temperature 0.7 (K = 16/32/64/256); the greedy rows
  are one T = 0 sample per question, configs as in `fidelity_check.py` (`dream-sampler` and `llada-published` use our 4-shot prompt).
- K = 256 covers a seeded random subsample of the 500 test questions (n = 185 Dream, 192 LLaDA; stopped by the clock, see
  `k256_results.md`), so its CIs are wider; K = 16/32/64 are compared on their common 500 questions. The greedy comparisons use
  the questions that had a greedy row (n differs by config).
- Generation seconds were measured on shared GPUs (the `antpc` account is shared, lanes were interrupted on 2026-11-20), so
  they are not clean compute measurements [unverified as a cost comparison].
- Paired differences use paired_stats.py's bootstrap (10 000 resamples, seed 0) and exact McNemar; no multiple-comparison
  correction across the rows.
- The AR rows score the iteration-1 trimmed text unless a `-cut` scorer appears in the output; candidates carry junk after the
  '#### N' line (see ITERATION_2 notes), which affects those rows.
