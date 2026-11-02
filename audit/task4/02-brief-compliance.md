# 02 — Task 4 brief, clause by clause

The brief (as pasted by Aaditya, 2026-11-02): codify the best of the top-5 ideas **using LLMs**;
identify specification gaps, inconsistencies and infeasibilities, before or after initial trials;
**refine the idea by giving proper feedback to the model**; get final results on an appropriate
metric and dataset; **check whether they beat the state of the art, or where the idea stands
against baseline ideas**. Four reference systems: DeepScientist, Arbor, AutoSOTA, EvoScientist.

| Clause | Status | Evidence | Gap |
|---|---|---|---|
| Choose the best of the top-5 | ✅ done, argued | `IDEA_SELECTION.md`; P2 rejected because it yields no score to compare | Team confirmation (D-2026-11-01-b) still open. The selection's claim that 1.1 has "no factual errors" is wrong (see below) |
| Codify it with LLMs | 🟡 partial | Pipeline written with Claude (commit trailers); Arbor configured for a local vLLM model | Arbor has not run. LLM involvement in the existing code is visible in git, not documented as method |
| Use the reference papers | ✅ | `REFERENCE_SYSTEMS.md`: only Arbor's code implements its paper (name-based scan of every `.py`) | AutoSOTA/DeepScientist ideas could still be borrowed in prose (e.g. tiered promotion of findings) |
| Identify spec gaps / infeasibility, **before** trials | ✅ strong | I-1…I-6 (`IDEA_SELECTION.md`), I-7…I-10, I-12 (`ITERATION_1.md`), plus four new items in this audit | — |
| … and **after initial trials** | ❌ not yet | Requires GPU output | The campaign's smoke runs and gates produce exactly this (I-11) |
| **Refine by giving feedback to the model** | ❌ no evidence | Arbor `interaction_mode` is `auto` in `research_config.yaml`; the README asks for `review` | Nothing records what we told any model. See below |
| Final results, appropriate metric + dataset | ❌ pending | GSM8K accuracy, macro over models; harness tested on mocks | GPU |
| Beat SOTA or place against baselines | 🟡 partial | Baselines coded (first, random, majority, AR sum/mean, oracle); published base-model numbers verified (`ITERATION_1.md`) | No published *test-time method* compared. Candidates now identified in `05` |

## Error in the idea that the selection missed

`IDEA_SELECTION.md` and `ideation/REPORT.md` §7 say idea 1.1 has "none of the lineage errors". It
has one. The idea text defines the matched scorer as "TinyLlama-1.1B … for LLaMA-based DLMs
(DiffuLLaMA, Dream-7B, LLaDA-8B, DiffuCoder-7B)" (`ideation/output/research_problems_and_ideas.md`
line 67). Dream-7B is Qwen2.5-based, LLaDA-8B is trained from scratch, and DiffuCoder-7B is
Qwen2.5-Coder-based (all verified in REPORT §7 and the Dream/HF model cards). The codification
quietly works around it by adding a Qwen2.5-0.5B scorer. It should be logged as an issue fixed by
feedback (I-13), because that is exactly the kind of evidence the brief asks for.

Other spec defects not yet logged:
- "8-bit quantization and activation checkpointing" for 7B models is unnecessary on a 96 GB card, and
  activation checkpointing is a training technique. Ignored.
- "N = 10 candidates by varying the random seed at the initial noise stage". For a masked DLM the
  initial state is all-mask and deterministic; diversity comes from sampling at each step. The
  implementation samples, which is correct, but the spec is wrong.
- The problem statement asks whether reranking is "more efficient than simply increasing the
  denoising-step budget". The step-budget arm is out of scope in the implementation, so **half of
  the research question is not being answered** (`04` §3).

## How to produce the missing "feedback to the model" evidence

The brief grades the refinement loop, not only the final code. Two cheap sources:

1. **This codification was itself LLM-driven.** Each issue I-7…I-13 was found and fixed in a Claude
   Code session where a human pointed at a problem and the model proposed and verified a fix. Record
   each one with a line saying what the human said, what the model changed, and how the change was
   verified. `ITERATION_1.md`'s issue log is most of this already. Add a "feedback given" column.
2. **Arbor in `review` mode.** Set `ui.interaction_mode: review`, seed the three directions listed in
   `ITERATION_1.md` §Arbor, and paste every accept/reject with its reason into the table there. Keep
   `.arbor/sessions/<run>/` as the artefact.
