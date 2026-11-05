# Experiment 2.1

**Hypothesis**: Mechanism: majority vote with qwen7b per-token NLL tie-break — count non-None answers; among answers TIED for the top count, pick the answer whose best candidate has the lowest per-token NLL (min of ar.qwen7b.nll_sum/ar.qwen7b.n_tokens over that answer's candidates); remaining ties/missing features -> earliest candidate index; all-None -> 0.
Hypothesis: ties are where majority loses most (23 dev dream tie items, 8 of 23 earliest-choice correct), and qwen7b per-token NLL may separate tied answers even though it is anti-predictive across all candidates.
Observable: dream-base and llada-base dev >= 0.7000/0.7450; coordinator diagnostic predicts ~neutral to slightly negative (qwen7b fixed only 1/14 dream, 4/4 llada majority-wrong ties) — report per-model deltas vs plain majority.
Conflicts: none — complements node 1; same decision-rule axis (vote), different tie-handling.
**Branch**: `coordinator/n2-1-mechanism-majority-vote-with-qwe-deb33ae3`
**Attempt**: 1
**Status**: done
**Eval status**: scored (stop_reason=finished)
**Turns**: 14

---



Working tree clean; solution.py was auto-committed by the agent infra (02d8557) and my commit added the results summary (1a17551).

## Report

**Idea**: Majority vote over extracted answers, with ties among top-count answers broken by lowest qwen7b per-token NLL (`nll_sum/n_tokens`, best candidate per answer).

**Changes**: `solution.py` — rewrote `select()`: `Counter` over non-None answers → unique top answer → earliest candidate of it; ties → per tied answer compute min per-token qwen7b NLL over its usable candidates → unique NLL winner returns its best candidate; remaining ties / all-missing NLL → earliest candidate index among top-count answers; all-None → 0.

**Validation**: static check PASS; 7 synthetic edge cases covering all 4 spec branches PASS; cross-check of `select()` vs spec logic on all 1000 dev items: 0 mismatches; two eval runs identical (deterministic).

**Baseline vs Result** (B_dev, n=200/model):

| | diffugpt-m | diffugpt-s | diffullama | dream-base | llada-base | **macro** |
|---|---|---|---|---|---|---|
| trunk (idea 1.1, ar.gpt2.sum) | 0.000 | 0.005 | 0.010 | 0.235 | 0.490 | **0.148** |
| plain majority (earliest tie-break) | 0.010 | 0.015 | 0.000 | 0.700 | 0.745 | **0.294** |
| **this node** | 0.010 | 0.005 | 0.015 | 0.675 | 0.745 | **0.290** |

**Score**: **0.2900**

**Required diagnostics (recomputed, dev)**:

| model | (b) tie items | (c) NLL resolved tie (answer changed) | fixed / broke / changed-still-wrong | (d) net vs majority |
|---|---|---|---|---|
| diffugpt-m | 119 | 97 | 0 / 0 / 97 | +0 |
| diffugpt-s | 120 | 102 | 1 / 3 / 98 | −2 |
| diffullama | 148 | 131 | 3 / 0 / 128 | +3 |
| dream-base | 39 (31 wrong-earliest) | 27 | 1 / 6 / 20 | **−5** |
| llada-base | 23 (18 wrong-earliest) | 18 | 4 / 4 / 10 | **0** |

**Analysis / Insight**: Fix counts match the coordinator diagnostic exactly (1 dream, 4 llada), but the "never broke a tie in the wrong direction" claim is **false on dev**: the NLL tie-break flipped 6 correct dream answers and 4 correct llada answers to wrong ones. The qwen7b per-token NLL's anti-predictiveness survives restriction to tie candidates — on dream it resolved 27 ties but only 1 resolution was correct while 6 were harmful (net −5). Net: slightly negative vs plain majority (0.2900 vs 0.2940), hypothesis partially met (llada at 0.7450 floor exactly, dream 0.6750 < 0.7000), but the mechanism is still ~2× the trunk. The tie-break adds value only where the base model is strong enough to vote correctly most of the time *and* its wrong ties happen to be NLL-separable; on dream that condition fails.
