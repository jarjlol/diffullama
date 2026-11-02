# 05 — Novelty and the state of the art to compare against

Method: arXiv API searches on 2026-11-02 (relevance-sorted, abstract fields), plus abstracts of the
hits. **Everything below is abstract-level.** Before any number goes into the report, read it from
the paper's full text and record the shot count and checkpoint (`docs/verification-standards.md`).

## Searches run

| Query (abstract field) | Hits | Relevant |
|---|---|---|
| "diffusion language" AND rerank | 4 | none on answer reranking (DiffuRank is document retrieval) |
| "diffusion language" AND "self-consistency" | 6 | 2602.22871, 2609.39560 |
| "diffusion" AND "language model" AND "CKA" | **0** | — |
| "diffusion language" AND "test-time scaling" | 7 | 2604.06260, 2602.01842, 2602.04344, 2510.27469 |
| "diffusion language" AND verifier | 10 | 2609.35472 (via a related query), 2602.01849 |
| "diffusion language" AND "autoregressive" AND "representation similarity" | **0** | — |

## Closest prior work

| Paper | What it does (abstract) | Overlap with idea 1.1 |
|---|---|---|
| **2609.35472** *Why Deterministic PRM Guidance Underperforms in Discrete Diffusion Reasoning* (Sep 2026) | Dream-v0-Instruct-7B, 8 candidates per GSM8K problem. **Independent sampling plus a trained ORM reranker reaches 75.13%**, against 65.18% for PRM-guided pruning | **High** on "rerank DLM candidates with an external scorer". Theirs is trained; ours is a training-free small AR LM. The closest comparator |
| **2609.39560** *Self-Repulsive Sampling for Diffusion Language Models* (Sep 2026) | LLaDA-8B-Instruct, **10 paths**, 128 steps: deterministic self-repulsion reaches **80.38% plurality accuracy on GSM8K** against 70.17% for greedy | High on "N = 10 + majority vote on a DLM". A SOTA point for the majority-vote side |
| 2604.06260 *S³: Stratified Scaling Search* | verifier-guided search *during* denoising; LLaDA-8B-Instruct on GSM8K and others; argues best-of-K is limited | medium; a different axis (in-loop search, not post-hoc selection) |
| 2602.01842 *Prism* | hierarchical search plus self-verification; LLaDA-8B-Instruct, Dream-7B-Instruct | medium |
| 2602.22871 *Reward-Guided Stitching* | DLM samples many traces; a PRM stitches steps; self-consistency framing | medium |
| 2510.27469 *Diffuse Thinking* | a DLM proposes thoughts, an LLM evaluates them | medium; already in our litreview pool, cited in `IDEA_SELECTION.md` |
| DiffuCoder (2506.20639, in `project-docs/12` §5.1) | measures the "AR-ness" of adapted models' decoding order | medium on the preservation half |

## Novelty verdict (provisional)

- **"AR reranking improves DLM answers"** is not novel. 2609.35472 does it with a trained reranker,
  and best-of-N with verifiers is standard.
- **"Representation preservation (CKA to the AR base) predicts how much reranking recovers"** found no
  prior work in these searches: zero hits for CKA in DLM abstracts. That is the defensible
  contribution, and it is also the part with the weakest statistics (`04` §1). The per-question
  version of the test (`04` §1, fix 2) strengthens exactly the novel part.
- Limits: abstract-level search on arXiv only, a single day. Re-run weekly; this area moved twice in
  four weeks earlier this term (CDC, DiffPDE).

## What "beat SOTA" can honestly mean here

The published test-time numbers above use **Instruct** checkpoints, while our runs use **Base**
checkpoints. They are not directly comparable. Options:

1. **Cheapest and honest:** report against our own baselines on the same candidates. Majority vote
   is the anchor's own SC baseline (Table 2) and the bar, and oracle is the ceiling. Then cite the
   SOTA numbers as context with the checkpoint difference stated.
2. **Comparable:** add `Dream-org/Dream-v0-Instruct-7B` and `GSAI-ML/LLaDA-8B-Instruct` to the
   campaign at the same N = 10. Then our majority vote and AR-rerank sit in the same table as
   2609.39560 (LLaDA-Instruct, 10 paths) and 2609.35472 (Dream-Instruct, ORM reranker). Dream-Instruct
   also adds an AR-base model to the CKA analysis. Cost: two 7B runs, about 11 h each at 64 steps on the subset.
