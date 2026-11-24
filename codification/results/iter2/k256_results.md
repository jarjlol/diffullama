# K = 256 step-budget check: result (single analysis run, Tue 24 Nov 2026 01:43 IST)

Design, stopping rule and analysis were fixed in `k256_preregistration.md` before any K = 256 row existed; clauses added later (windows, Ada extension, greedy run,
crash handling, cutoff change) were each written before the corresponding rows and before any K = 256 accuracy was looked at. `step_arm.py` was run once, on all rows.
Raw output: `k256_step_arm_output.txt`; machine-readable: `k256_step_arm.json`.

## What was run
- Dream-v0-Base-7B and LLaDA-8B-Base, test split, 256 denoising steps, N = 10, T = 0.7, same prompts and sampler as the 64-step runs, questions taken in the seeded random order `k256_order.txt`.
- Stopping was by the clock (two Blackwell windows and an Ada extension that Aaditya stopped on Mon 23 Nov 22:13 IST), never by a result. n is whatever finished.
- Rows: Dream 185 (162 Blackwell RTX PRO 6000 + 23 Ada RTX 5000 Ada), LLaDA 192 (151 Blackwell + 41 Ada, 1-sample batches on Ada). On a uid clash the Blackwell row was kept (no clash occurred).
- Sampling on Ada used `CODIFY_SAMPLE_CHUNK` batches (Dream 5, LLaDA 1): same per-sample distribution, different random stream; every row records `host`, `gpu` and `sample_chunk`.
- s/question (Blackwell rows only, N = 10): Dream 36.5 s at K = 64 and 151.1 s at K = 256; LLaDA 54.9 s and 216.6 s.
- Representativeness: K = 64 majority on the finished subset 0.741 (Dream) and 0.745 (LLaDA) against 0.724 and 0.732 on all 500; first sample 0.422 and 0.464 against 0.404 and 0.486.

## Primary: ar.qwen7b-cut.sum minus majority vote at K = 256 (paired, exact McNemar, 95% bootstrap CI)
| Model | n | AR (qwen7b cut, sum) | majority | difference | wins/losses | McNemar p |
|---|---|---|---|---|---|---|
| Dream | 185 | 0.757 | 0.843 | -0.086 [-0.141, -0.032] | 5 / 21 | 0.002 |
| LLaDA | 192 | 0.766 | 0.792 | -0.026 [-0.068, +0.016] | 6 / 11 | 0.332 |

Reading, by the rules in the preregistration:
- **Dream:** the CI is entirely below 0, so the iteration-1 finding (AR reranking loses to majority vote) still holds at 256 steps (at T = 0.7). The gap shrank from about -0.24 at K = 64 (500 questions) to -0.09.
- **LLaDA:** the CI includes 0, so at 256 steps the LLaDA loss is not distinguishable from zero at this n; by the preregistered rule the iteration-1 loss for LLaDA looks like (at least partly) a step-budget effect. The direction is still negative (6 wins, 11 losses).
- The two models therefore differ; say so, and report n and CI rather than "no effect" for LLaDA.

## Secondary (first sample, K = 256 minus K = 64, same uids)
- Dream +0.319 [+0.232, +0.400] (69 wins / 10 losses, p < 0.001). LLaDA +0.260 [+0.182, +0.339] (60 / 10, p < 0.001).
- Majority K = 256 minus K = 64: Dream +0.103 [+0.049, +0.157] (p = 0.001); LLaDA +0.047 [+0.000, +0.099] (p = 0.108).

## Greedy (T = 0, N = 1, 256 steps, our 4-shot prompt) as context; secondary, not the test of record
Greedy rows exist for the uids that had K = 256 rows when the greedy runs started, so n differs; the paired comparisons use the uids with a greedy row.
- Dream (`dream-sampler`, n = 185): majority minus greedy +0.032 [-0.011, +0.081] (p = 0.238); AR minus greedy -0.054 [-0.108, 0.000] (p = 0.087); first sample minus greedy -0.070 [-0.135, -0.005] (p = 0.053).
- LLaDA (`llada-published` config = our prompt, T = 0; n = 137 of 192: 132 of the 151 Blackwell rows (the other 19 were generated after the last greedy pass started at 18:55) plus 5 of the 41 Ada rows that happen to be in the first-100 fidelity run): majority minus greedy +0.080 [+0.029, +0.139] (p = 0.007); AR minus greedy +0.073 [+0.007, +0.139] (p = 0.041); first minus greedy +0.029 [-0.022, +0.080] (p = 0.424).
- The `dream-published` lines in the raw output use the lm-eval prompt and only 41 overlapping questions; context only.

## Caveats to state in the paper
- T = 0.7 at K = 256 is not like-for-like with the published greedy numbers (papers: Dream 77.2, LLaDA 70.3; our own greedy reproduction at 256 steps on 100 questions: 0.81 and 0.73, see `fidelity.md`).
- n was set by two clock-limited GPU windows plus an Ada extension stopped on instruction, not by any result; rows come from two GPU types; Dream crashed twice with a CUDA launch timeout on Blackwell GPU 0 (Xid 8, auto-restarted, no rows lost).
- One planned Dream K = 256 slot (18:23-18:55 on Mon 23 Nov) did not start because of a scheduling-gate mistake; no data were affected, Dream n is 185 instead of about 197.
