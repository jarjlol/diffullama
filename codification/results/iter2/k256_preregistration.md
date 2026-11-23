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

## Clarifications (Sun 22 Nov 2026, 21:45 IST; made before any K = 256 row exists, checked on Ada and Blackwell at 21:42 IST)
- **Window (updated 22:20 IST, still before any K = 256 row).** krispy's pause has no fixed end, so "krispy's end time" no longer applies. krispy allowed our runs from 08:00 IST
  on Mon 23 Nov (relayed by Aaditya); nothing on Blackwell starts earlier. The window ceiling is 8 hours after the first real start of a window job and is never extended
  after seeing results. Generation (clock-only stopping) ends 7 h after that start (1 h before the ceiling; Aaditya chose 7 h over 5 h 30 min at about 23:50 IST, still before any K = 256 row);
  the scoring lines follow at once and have the last hour (about 40 min needed). If krispy needs the GPUs earlier we stop and hand them back; the rows finished by then are reported with their n.
- **Ada sampling.** Ada's Dream rows use `CODIFY_SAMPLE_CHUNK=5` (the 10 samples of a question in two batches of 5, because one batch of 10 ran out of memory on the
  shared 32 GB card). Same independent per-sample distribution, different random stream. Every row records `sample_chunk` next to `host` and `gpu`.
- **Compute column.** s/question is computed from Blackwell rows only; Ada's `gen_seconds` (other GPU, 5-sample batches, shared card) is not comparable.
- **Ada extension (Mon 23 Nov 2026, about 16:05 IST; written after the Blackwell window's generation ended, before any K = 256 result was computed or looked at).**
  Blackwell generation ended at its clock stop with 161 Dream rows (+ 23 Dream rows from Ada, merged by uid) and 116 LLaDA rows. `step_arm.py` has not been run on any K = 256 row, and no
  per-question or aggregate K = 256 accuracy has been looked at. Ada is free, so we add LLaDA K = 256 rows on Ada (`ada_run3.sh`: from the back of the same random order,
  `CODIFY_SAMPLE_CHUNK=1` (2 ran out of memory in the float64 Gumbel-noise step on the shared 32 GB card; same per-sample distribution), recorded in `sample_chunk`) until a fixed clock time, **Tue 24 Nov 2026 06:00 IST**, then stop whatever the rows show. The two Ada Dream runs were stopped by a STOP file at 15:55 IST
  (24 rows on Ada, 23 of them merged; the last row is not used) and are not resumed. The primary and secondary analyses are run ONCE, on all rows (Blackwell + Ada, Blackwell kept on a uid clash) after the
  extra rows are scored, and report n per model and per host. No interim analysis is run before the cutoff. If Ada's job is yielded or killed by another user, we report what was finished by the cutoff.
- **Second Blackwell window (Mon 23 Nov 2026, written about 17:10 IST, before any row of it exists and before any K = 256 result was computed or looked at).** The first window ended at 16:21 IST with both GPUs idle.
  krispy allowed Blackwell until 20:00 IST today (relayed by Aaditya), so the "8 h window is never extended" rule above is replaced for this one extra slot by an explicit second window of the same kind: the queues
  continue the same random order from the front (GPU 0 Dream, GPU 1 LLaDA; Ada works from the back; a Blackwell row wins a uid clash), stop by the clock 30 min before the ceiling, and the ceiling is 19:55 IST at the latest
  (five minutes before krispy's 20:00); scoring runs only on rows not yet scored. The stop is by clock only, never by results; if krispy needs the GPUs earlier we hand them back and report the realised n.
  The analysis stays one single run on all rows (Blackwell + Ada) after Ada's cutoff (Tue 24 Nov 06:00 IST), exactly as declared above; nothing is analysed before that.
