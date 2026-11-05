# Experiment 3.1

**Hypothesis**: Mechanism: likelihood-weighted voting (falsification) — score each distinct extracted answer by sum of exp(-per-token NLL) over its candidates; per-token NLL = ar.<scorer>.nll_sum/ar.<scorer>.n_tokens; scorer: dream-base -> qwen7b, diffullama -> llama2-7b (its own AR base), all other models -> qwen7b; pick highest-scoring answer; ties -> earliest index; all-None -> 0.
Hypothesis: per-token NLL carries no usable signal in the decisive cases (correct-minority lower NLL in 2/14 dev-dream, 8/18 dev-llada — at/below chance), so this lands below plain majority; node exists to record the audited negative (or a surprising positive).
Observable: B_dev macro below node 1.1's ~0.294 (dream ~0.65, llada ~0.71-0.73 predicted) -> prune with lesson; >= 0.295 -> retain.
Conflicts: none — tests the brief's proposed mechanism head-on.
**Branch**: `coordinator/n3-1-mechanism-likelihood-weighted-vo-4bdb7eb4`
**Attempt**: 1
**Status**: done
**Eval status**: scored (stop_reason=finished)
**Turns**: 12

---



Done. Implementation verified, evaluated, committed (solution.py at `8ee9eb6`, results summary at `e45dc6d`).

# Report

**Idea**: 3.1 — falsification of likelihood-weighted voting: score each distinct extracted answer by Σ exp(−per-token NLL) over its candidates, pick the highest-scoring answer.

**Changes**: `solution.py` — rewrote `select()` to implement exactly the specified mechanism: per-item scorer (dream-base→qwen7b, diffullama→llama2-7b, others→qwen7b), weight = `math.exp(-(ar.<s>.nll_sum / ar.<s>.n_tokens))` summed per distinct non-None answer, candidates missing either feature skipped, ties → earliest candidate index, all-None/no-scores → 0. No state, stdlib-only (`math`), passes static check.

**Baseline vs Result** (B_dev, n=200/model):

| model      | this (3.1) | plain majority | delta   |
|------------|-----------|----------------|---------|
| diffugpt-s | 0.0100    | 0.0150         | −0.0050 |
| diffugpt-m | 0.0050    | 0.0100         | −0.0050 |
| diffullama | 0.0150    | 0.0000         | +0.0150 |
| dream-base | 0.6500    | 0.7000         | −0.0500 |
| llada-base | 0.7300    | 0.7450         | −0.0150 |
| **macro**  | **0.2820**| 0.2940         | −0.0120 |

**Score**: **0.2820** (B_dev macro; baseline/trunk 0.148, idea 1.1's GPT-2 summed-NLL anchor ~0.294).

**Coordinator prediction — CONFIRMED**: dream-base hit exactly ~0.65 (predicted, below majority 0.7000); llada-base hit 0.7300 (top of predicted 0.71–0.73 band, below majority 0.7450). Macro 0.2820 < 0.294 → **prune with lesson**.

**Insight**: Per-token NLL under the AR scorer is anti-predictive in exactly the decisive cases — when a correct-minority answer splits the plurality (correct-minority has lower NLL in only 2/14 dev-dream, 8/18 dev-llada splits, at/below chance). On dream/llada the NLL weights systematically downweight the plurality's high-likelihood candidates, so the weighted vote flips to the (usually wrong) high-probability minority answer. The only gain was on diffullama (+0.015), where even plain majority sits at 0.000 — noise-level improvement on a model with no usable signal. Lesson: NLL magnitude here measures fluency/length-style artifacts, not correctness; counting agreement (plain majority) beats likelihood-weighting.
