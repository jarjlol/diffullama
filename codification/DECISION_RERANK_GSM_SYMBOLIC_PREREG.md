# Decision reranking on GSM-Symbolic — pre-registration

Written 2026-11-26 before the first full scored run (the only prior API calls were
a 3-question plumbing smoke test). Issue: #79; sweeps: #83.

## Question

Does the listwise margin over majority vote survive when the GSM8K questions are
replaced by GSM-Symbolic's symbolic-template variants? On GSM8K (K=64, N=10) the
margin was +5.2 points, 95% CI [+3.0, +7.4]. If that gain is mostly memorised
answer knowledge, the margin should shrink here.

## Data

- `work/gsm_symbolic/questions.jsonl`, sha256
  `7fbf4418fdca7635bb5ebf240ce16bf58f153a45a0581bd61d5a19019759194e`,
  250 questions: 125 P1 + 125 P2, seed 0.
- Six template groups (`p1-0`, `p1-1`, `p1-2`, `p2-0`, `p2-1`, `p2-2`) with
  25–50 variants each. Each variant is a distinct problem: names, numbers and
  surface form change, the solution structure does not.
- Gold is the worked solution's `#### N` line, read with
  `rerank_task/task.py:extract_answer` and graded with `task.is_correct`.
- **Deviation from #79:** 250 questions, not 500. Both templates are represented
  (125 each) and every question is scored; the exact n is stated everywhere.
  This is the half-cost fallback the issue allows.

## Models and candidate protocol

Unchanged from the GSM8K arm: Dream-7B and LLaDA-8B, N=10, K=64 denoising
steps, temperature 0.7, 4-shot prompt, 256 new tokens, seed 0. Candidates are
cached; no regeneration for the main run. One process per GPU, started before
this note.

## Selectors

| selector | definition |
|---|---|
| `first` | candidate A |
| `majority` | most common extracted final answer (first occurrence on ties) |
| `ar.qwen7b-cut.sum` / `.mean` | Qwen2.5-7B NLL of the cut candidate text, summed / per token; computed after generation on a free GPU |
| `decision.listwise` | one typed choice over all N candidates |
| `decision.pointwise` | yes/no on each candidate, pick max `p_yes`; exploratory |
| `oracle` | any candidate correct (pass@N) |

Decision model: `perplexity/pplx-decider-flash-v1.3:free` through OpenRouter
`/api/alpha/decisions`. Prompt state/schema and wording are copied verbatim from
`make_openrouter_prompts.py`; candidate text is `decision_rerank.candidate_text`
(`task.trim`, then cut at the `#### N` line). No gold enters a prompt.

## Hypotheses and tests

- **H1 (primary):** `decision.listwise` > `majority` on the macro average.
- **H2 (primary):** `decision.listwise` > `ar.qwen7b-cut.sum`.
- **Secondary, exploratory:** `decision.pointwise` vs the same baselines; the
  N sweep, order sweep, K=256 subset and controls are descriptive.

Statistics fixed in advance:

- Paired **cluster bootstrap that resamples questions**, carrying both models
  together for the macro delta; 10,000 resamples, seed 0, percentile 95% CI.
  The stacked McNemar is not used (it double-counts questions).
- **Per-model exact McNemar** on the same questions (two-sided).
- Headroom = (accuracy − first) / (oracle − first).
- Coverage is reported; a selector is not compared on a different item set.
- No threshold is tuned on GSM-Symbolic test. The weak-vote threshold
  (keep majority when the top answer has ≥ T of 10 votes, else listwise) is
  fixed on the 200 GSM8K dev questions; T = 4, 5, 6 are reported on test once.

## Planned additional sweeps (#83)

1. **N sweep:** first 100 questions per model, N ∈ {2,4,6,8,10}, listwise over
   the first N candidates; accuracy and option count vs majority at the same N.
2. **Order sweep:** first 100 questions per model, three seeded permutations of
   the 10 candidates; accuracy per order and fraction of items whose chosen
   **candidate** is identical across all three.
3. **K=256 subset:** after the main run, first 50 questions per model at K=256
   (Dream ≈151 s/q, LLaDA ≈217 s/q); listwise only, compared with K=64.
   If it does not finish, report the completed n and do not extrapolate.
4. **Calibration and cost:** ECE (10 bins) and Brier on listwise max-probability;
   `usage`, latency and cost per correct answer from the client records.
5. **Controls:** judge-alone (distinct answers, no reasoning), gold injection
   (gold added where no candidate carries it), closed-book two-option (gold vs a
   wrong number), same protocol as #76.

## Interpretation fixed in advance

The claim is **selection, not reasoning verification**. GSM-Symbolic is designed
to reduce memorisation of GSM8K items, but it is not proof that no contamination
or template leakage remains. The decision model may still know the underlying
arithmetic. A margin that shrinks toward zero supports the memorisation concern;
a margin that survives supports the selection claim, not a verification claim.

Other stated confounds: 64 denoising steps is below Dream/LLaDA's published
budget; Dream runs at temperature 0.7 for candidate diversity; one decision-model
snapshot and one prompt wording; LLaDA has no AR base, so it is the from-scratch
control and gets no preservation claim.
