# Which idea to codify

**Selected: Idea 1.1 (P1-M1)** — layer-wise *representation preservation* (CKA between each diffusion
model and its own autoregressive base) as a predictor of how much accuracy a lightweight AR scorer
recovers by **reranking** the diffusion model's N candidate answers.

Source: `research_problems_and_ideas.md`, the ResearchAgent output of the ideation assignment
(top-5 per problem; not yet committed by the teammate who ran it). Decided 2026-11-01.

## Why P1, not P2

The brief ends with: get final results on an appropriate metric and dataset, and *check whether they
beat the state of the art, or where the idea stands against baseline ideas*. That requires a method
that produces task performance to compare.

- **P2's five ideas** (noise-schedule studies) are characterisation methods. They map a
  quality–efficiency frontier and propose diagnostic metrics (token-commitment curves, likelihood
  drift, response surfaces), but produce no system that can be scored against a baseline. Three of
  the five (2.1–2.3), and the P2 statement itself, also depend on DiffuLLaMA's unreleased adaptation
  corpus.
- **P1's ideas** share a concrete, codifiable core — rerank a diffusion LM's N sampled answers with
  a small AR model — which yields **task accuracy** on GSM8K / HumanEval / WinoGrande that can be
  compared with baselines. This core is also the inference-only answer to limitation L6 (DiffuLLaMA's
  own Table 2: hit@3 far above self-consistency), so it connects back to the anchor paper.

## Why 1.1 among P1's ideas

| Idea | Mechanism | Verdict |
|---|---|---|
| **1.1 P1-M1** | CKA per layer (DLM vs its AR base) + AR-scorer reranking | **chosen** — standard, well-defined measure; every DLM shares its base's architecture and tokenizer, so layer-to-layer CKA is exact; no training; **no factual errors found**; highest-ranked (tied 4.9) |
| 1.2 P1-M6 | attention-symmetry ablation (causal vs full attention) | **runner-up.** Sharpest anchor-specific question, but misstates Dream's backbone twice and needs per-model attention-mask surgery |
| 1.3 P1-M4 | spectral alignment via gradients + trained linear probe | most complex; gradient computation on 7B models; more failure points |
| 1.4 P1-M8 | conditional divergence at "semantic pivot" tokens | pivot definition is heuristic and under-specified |
| 1.5 P1-M2 | Procrustes alignment | **excluded** — wrong lineages for Dream, LLaDA and DiffuCoder, and "8-bit NF4" |

## Known issues to fix during codification

The brief expects these to be found and fixed by refining the idea with the model. Recorded now, from
reading the specification:

1. **Missing baselines.** The spec compares reranking only against the *first* candidate and an
   *oracle*. It needs **self-consistency (majority vote)** — the anchor's own Table 2 baseline (27.7 on
   SATMath, 33.1 on MAWPS) — plus the DLM's own-likelihood reranker, and at least one **published
   dLLM test-time-scaling method** for the SOTA comparison.
2. **Length bias.** Selecting the minimum *summed* NLL favours short answers; per-token or
   length-normalised scoring must be specified and compared.
3. **Statistical power of the headline claim.** The predictive claim (preservation → reranking
   gain) is a correlation across only ~5 diffusion models. The spec admits this ("directionality
   rather than significance"). The per-model reranking results must carry the paper on their own.
4. **LLaDA is missing** as the from-scratch control with no AR base, and it is the natural
   null case for "preservation".
5. **Answer extraction** for GSM8K (final-number parsing) and the HumanEval execution harness are not
   specified.
6. **Novelty is unchecked.** AR-scorer reranking of diffusion candidates already exists in nearby
   work (e.g. *Diffuse Thinking*, *Planner and Executor*, both in the litreview pool) and
   DiffuCoder measured pass@k on adapted models. The defensible contribution is the **link** between
   AR-representation retention and reranking gain, not the reranking itself. Check the anchor
   authors' later work first.

## Codification route

Arbor (`REFERENCE_SYSTEMS.md`) — the only reference system with a complete, open implementation —
driven by the local vLLM setup from `ideation/local/`. Arbor needs a runnable evaluation script and a
dev / held-out test split; the reranking task provides both (e.g. GSM8K train-derived dev, test as
held-out).

## How the harness resolves each issue (2026-11-01)

See `README.md` and `rerank_task/` for the code.

1. **Baselines.** `rerank_task/baselines.py` reports first sample, random (exact expectation),
   majority vote (self-consistency), AR summed NLL, AR mean NLL and the oracle (hit@N).
   - Still open: the DLM self-score reranker and the published-method comparison
     (README §D).
2. **Length bias.** Summed and per-token NLL are both reported. The seed `solution.py` keeps
   the summed version on purpose, as the specified starting point Arbor must improve on.
3. **Power.** `pipeline/analyze.py` reports Spearman with an exact permutation p-value and
   labels it descriptive. The per-model accuracy table is the main result.
4. **LLaDA** is included as `llada-base`, with no AR base, as the control.
5. **Extraction** is in `rerank_task/task.py`, shared by the pipeline and eval:
   - the candidate is cut at the next `Question:`;
   - the answer is taken from `#### N`, then "answer is N", then the last number;
   - grading is numeric.

   HumanEval is dropped (GSM8K only).
6. **Novelty.** Still to do (README §A).

Ambiguities settled while building:
- The AR scorer conditions on `Question: q\nAnswer:` only, without the few-shot exemplars.
- CKA uses mean-pooled block outputs on 2000 × 128-token WikiText-103 validation segments,
  identical token ids for both models, with the DLM bidirectional and the base causal.
- Each model runs in its own process, because `attention_patch` is process-wide.

## Correction (2026-11-02)

The table above says idea 1.1 has "no factual errors found". That is wrong. Its matched-scorer
definition calls Dream-7B, LLaDA-8B and DiffuCoder-7B "LLaMA-based" (idea text, line 67), which
is the same lineage error recorded for the other ideas in `ideation/REPORT.md` §7. The
codification already works around it: a Qwen2.5-0.5B scorer gives Dream a same-family scorer, and
LLaDA has no base and serves as the control. Logged as I-13 in `ITERATION_1.md`. The choice of 1.1
stands, because the error is in a secondary control, not in the method.
