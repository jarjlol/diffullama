# K = 256 run: pre-declared design and analysis

Written Sun 22 Nov 2026 (commit 23b02b5 carries the time) and committed **before any K = 256 candidate exists**. Changing anything below after the first
K = 256 row is written must be stated as a deviation next to the result.

## Question
Iteration 1 measured everything at 64 denoising steps, where first-sample accuracy is far below the published 256-step greedy numbers
(Dream 0.404 vs 0.81, LLaDA 0.486 vs 0.73). Does the iteration-1 finding (AR reranking loses to majority vote) still hold at the published
**step budget**? This run changes only the step budget (K = 256). It keeps the iteration-1 sampler (N = 10, T = 0.7, same prompts, 256 new
tokens), because N distinct candidates need T > 0. The published figures are greedy (T = 0) at 256 steps, so the run answers "at the published
step budget", not "at the published sampler" (see `fidelity.md`).

## Design
- Models: Dream-v0-Base-7B and LLaDA-8B-Base, test split, the same 500 questions as iteration 1.
- **Question order:** `k256_order.txt`, a seeded (`random.Random(20261122)`) permutation of the 500 test uids. Blackwell (both models) walks it
  from the front, Ada (Dream only) walks it from the back (`generate_candidates.py --order ... [--reverse]`). Any finished set is a random
  subsample of the 500. (`prepare_data.py` keeps the original GSM8K order, so "the first n" would not be random.)
- **Stopping rule: the clock only.** Generation stops at the end of the GPU window (Blackwell: 25 min before the window end; Ada: when the
  GPU is needed or a cap of 60 rows). Never on a significance result and never after looking at any K = 256 accuracy.
- **Hardware:** every row records `host` and `gpu`. If both machines reach a uid, the **Blackwell row is kept** (decided now); Ada rows
  enter via `test.ext.jsonl` and are merged by uid. The paper states that rows come from two GPU types.
- Scoring of the K = 256 rows: qwen7b and gpt2 on cut and trimmed text, DLM self-score on cut and trimmed text (same code as K = 64).

## Analysis (`step_arm.py`, K = 256 section; same paired exact McNemar and 10,000-draw bootstrap CI as iteration 1)
- **Primary, per model:** accuracy of `ar.qwen7b-cut.sum` minus majority vote at K = 256, on all uids with a finished K = 256 run.
- **Secondary, per model:** first-sample accuracy, K = 256 minus K = 64, paired on the same uids. Also reported (not tests of record):
  majority and the other selectors K = 256 minus K = 64, majority minus first sample at K = 256, DLM self-score vs majority, oracle.
- **Representativeness:** K = 64 accuracy (first, majority) on the finished subset next to the same numbers on all 500 questions.
- Greedy fidelity runs exist for the first 100 test questions only; K = 256 majority is compared with them on the overlap, as context.
- Reported n is whatever finished, per model; Dream and LLaDA n will differ. Nothing is dropped for being wrong or slow.

## Power
At K = 64 the AR-minus-majority gap is about 21 to 24 points (Dream -0.236, LLaDA -0.210 on 500), so n of 50 to 60 is enough to see whether
a gap of that size survives; a smaller gap would need a larger n and will be reported with its CI, not as "no effect".

## What would change the reading
- AR minus majority CI includes 0 or is positive at K = 256: the iteration-1 loss is a step-budget artifact; say so.
- AR minus majority still clearly negative: the finding holds at the published step budget (at T = 0.7).
