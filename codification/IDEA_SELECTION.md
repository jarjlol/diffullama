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
